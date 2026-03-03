// Command gemma3 generates text using a Gemma 3 model loaded from a GGUF file.
//
// Usage:
//
//	go run ./examples/gemma3/ --gguf /path/to/gemma-3-4b-it-qat.gguf
//	go run ./examples/gemma3/ --repo google/gemma-3-4b-it-gguf
//
// The tokenizer is loaded from the HuggingFace repository (requires network on first run).
// The GGUF file provides weights and config via metadata.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"time"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"

	models "github.com/ajroetker/huggingface-gomlx"
	"github.com/ajroetker/huggingface-gomlx/architectures/gemma3"
)

var (
	flagGGUF          = flag.String("gguf", "", "Path to local GGUF model file.")
	flagRepo          = flag.String("repo", "", "HuggingFace repo containing a GGUF model (e.g. google/gemma-3-4b-it-gguf).")
	flagTokenizerRepo = flag.String("tokenizer-repo", "google/gemma-3-4b-it", "HuggingFace repo for tokenizer.")
	flagPrompt        = flag.String("prompt", "Write a short poem about the sea.", "User message for chat prompt.")
	flagMaxTokens     = flag.Int("max-tokens", 100, "Maximum number of tokens to generate.")
	flagMaxSeqLen     = flag.Int("max-seq-len", 256, "Maximum total sequence length (prompt + generated).")
	flagTemperature   = flag.Float64("temperature", 0.8, "Sampling temperature (0 = greedy).")
	flagTopK          = flag.Int("top-k", 64, "Top-k sampling (0 = disabled).")
)

func main() {
	flag.Parse()
	if *flagGGUF == "" && *flagRepo == "" {
		log.Fatal("either --gguf or --repo is required")
	}

	hfToken := os.Getenv("HF_TOKEN")

	// Load GGUF model (weights + config from metadata).
	var model *models.Model
	var err error
	if *flagRepo != "" {
		fmt.Printf("Downloading GGUF model from %s...\n", *flagRepo)
		repo := hub.New(*flagRepo).WithAuth(hfToken)
		model, err = models.NewFromGGUFRepo(repo)
	} else {
		fmt.Println("Loading GGUF model...")
		model, err = models.NewFromGGUF(*flagGGUF)
	}
	if err != nil {
		log.Fatalf("Failed to load GGUF model: %v", err)
	}
	fmt.Print(model.Summary())

	// Load tokenizer from HuggingFace repo.
	fmt.Println("Loading tokenizer...")
	repo := hub.New(*flagTokenizerRepo).WithAuth(hfToken)
	tok, err := tokenizers.New(repo)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	bosID := 2 // Gemma default BOS.
	if id, err := tok.SpecialTokenID(tokenizers.TokBeginningOfSentence); err == nil {
		bosID = id
	}
	eosID := 1 // Gemma default EOS.
	if id, err := tok.SpecialTokenID(tokenizers.TokEndOfSentence); err == nil {
		eosID = id
	}

	// Create backend.
	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n", backend.Name())

	// Load weights into context.
	fmt.Println("Loading weights into context...")
	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// Get the Gemma 3 builder.
	builder, ok := model.Builder.(*gemma3.Builder)
	if !ok {
		log.Fatal("Model builder is not a Gemma 3 builder")
	}
	cfg := builder.Gemma3Config()

	// Tokenize prompt.
	chatPrompt := formatChatPrompt(*flagPrompt)
	promptTokens := tokenizePrompt(tok, chatPrompt, bosID)
	fmt.Printf("Prompt: %q\n", *flagPrompt)
	fmt.Printf("Prompt tokens: %d\n", len(promptTokens))

	if len(promptTokens) >= *flagMaxSeqLen {
		log.Fatalf("Prompt (%d tokens) exceeds max sequence length (%d)", len(promptTokens), *flagMaxSeqLen)
	}

	// Build prefill and decode execution graphs.
	fmt.Println("Building execution graphs...")

	prefillExec := context.MustNewExec(backend, ctx.Reuse(),
		func(ctx *context.Context, inputIDs, seqLenNode *Node) []*Node {
			return builder.ForwardPrefill(ctx, inputIDs, seqLenNode)
		},
	)
	prefillExec.SetMaxCache(10)

	decodeExec := context.MustNewExec(backend, ctx.Reuse(),
		func(ctx *context.Context, tokenID, posID, keys, values, insertPos *Node) []*Node {
			return builder.ForwardDecode(ctx, tokenID, posID, keys, values, insertPos)
		},
	)
	decodeExec.SetMaxCache(10)

	// --- Prefill ---
	fmt.Println("\nGenerating...")
	startTime := time.Now()

	promptIDs := make([]int64, len(promptTokens))
	for i, t := range promptTokens {
		promptIDs[i] = int64(t)
	}

	prefillResults := prefillExec.MustExec(
		[][]int64{promptIDs}, // inputIDs [1, seqLen]
		int32(len(promptTokens)), // seqLen
	)
	logitsTensor := prefillResults[0]
	kvKeys := prefillResults[1]
	kvValues := prefillResults[2]

	prefillDuration := time.Since(startTime)

	// Sample first token.
	logits := tensors.MustCopyFlatData[float32](logitsTensor)
	nextToken := sampleToken(logits, *flagTemperature, *flagTopK)

	// Print first token.
	tokenText := tok.Decode([]int{int(nextToken)})
	fmt.Print(tokenText)

	if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
		fmt.Printf("\n\n--- Prefill: %d tokens in %.2fs ---\n", len(promptTokens), prefillDuration.Seconds())
		return
	}

	// --- Decode loop ---
	realKVLen := len(promptTokens)
	numGenerated := 1
	decodeStart := time.Now()

	// Pad KV cache to full max sequence length upfront to avoid repeated
	// graph recompilation from changing buffer shapes mid-generation.
	bufferSize := *flagMaxSeqLen
	kvKeys = growKVBuffer(backend, kvKeys, bufferSize, cfg.KVHeads(), cfg.HeadDim)
	kvValues = growKVBuffer(backend, kvValues, bufferSize, cfg.KVHeads(), cfg.HeadDim)

	for i := 0; i < *flagMaxTokens-1; i++ {
		if realKVLen+1 >= *flagMaxSeqLen {
			break
		}

		results := decodeExec.MustExec(
			[][]int64{{int64(nextToken)}},      // tokenID [1, 1]
			[][]int64{{int64(realKVLen)}},       // positionID [1, 1]
			kvKeys,                              // allKeys
			kvValues,                            // allValues
			int32(realKVLen),                    // kvInsertPos
		)
		logitsTensor = results[0]
		kvKeys = results[1]
		kvValues = results[2]
		realKVLen++
		numGenerated++

		logits = tensors.MustCopyFlatData[float32](logitsTensor)
		nextToken = sampleToken(logits, *flagTemperature, *flagTopK)

		tokenText = tok.Decode([]int{int(nextToken)})
		if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
			break
		}
		fmt.Print(tokenText)
	}

	decodeDuration := time.Since(decodeStart)
	totalDuration := time.Since(startTime)

	fmt.Println("\n\n---")
	fmt.Printf("Prefill: %d tokens in %.2fs (%.1f tokens/s)\n",
		len(promptTokens), prefillDuration.Seconds(),
		float64(len(promptTokens))/prefillDuration.Seconds())
	if numGenerated > 1 {
		fmt.Printf("Decode:  %d tokens in %.2fs (%.1f tokens/s)\n",
			numGenerated-1, decodeDuration.Seconds(),
			float64(numGenerated-1)/decodeDuration.Seconds())
	}
	fmt.Printf("Total:   %d tokens in %.2fs\n", numGenerated, totalDuration.Seconds())
}

// formatChatPrompt wraps a user message in the Gemma 3 chat template.
func formatChatPrompt(userMessage string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", userMessage)
}

// tokenizePrompt encodes the prompt and prepends BOS.
func tokenizePrompt(tok tokenizers.Tokenizer, prompt string, bosID int) []int32 {
	encoded := tok.Encode(prompt)
	tokens := make([]int32, 0, len(encoded)+1)
	tokens = append(tokens, int32(bosID))
	for _, t := range encoded {
		tokens = append(tokens, int32(t))
	}
	return tokens
}

// sampleToken samples a token from logits using temperature and top-k.
func sampleToken(logits []float32, temperature float64, topK int) int32 {
	if temperature <= 0 {
		// Greedy: argmax.
		maxIdx := 0
		maxVal := logits[0]
		for i, v := range logits {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		return int32(maxIdx)
	}

	// Apply temperature.
	scaled := make([]float32, len(logits))
	for i, v := range logits {
		scaled[i] = v / float32(temperature)
	}

	// Top-k filtering.
	if topK > 0 && topK < len(scaled) {
		// Find the top-k threshold.
		sorted := make([]float32, len(scaled))
		copy(sorted, scaled)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
		threshold := sorted[topK-1]

		for i := range scaled {
			if scaled[i] < threshold {
				scaled[i] = float32(math.Inf(-1))
			}
		}
	}

	// Softmax.
	maxVal := float32(math.Inf(-1))
	for _, v := range scaled {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	probs := make([]float64, len(scaled))
	for i, v := range scaled {
		p := math.Exp(float64(v - maxVal))
		probs[i] = p
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Multinomial sample.
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}

// growKVBuffer pads the KV cache buffer to a new size along the sequence dimension.
// kvTensor shape: [numLayers, batch, kvHeads, oldSeqLen, headDim]
// Returns a tensor with shape: [numLayers, batch, kvHeads, newSeqLen, headDim]
func growKVBuffer(backend backends.Backend, kvTensor *tensors.Tensor, newSeqLen, kvHeads, headDim int) *tensors.Tensor {
	oldShape := kvTensor.Shape()
	oldSeqLen := oldShape.Dimensions[3]
	if newSeqLen <= oldSeqLen {
		return kvTensor
	}

	numLayers := oldShape.Dimensions[0]
	batchSize := oldShape.Dimensions[1]
	padAmount := newSeqLen - oldSeqLen

	padExec := context.MustNewExec(backend, context.New(),
		func(_ *context.Context, kv *Node) *Node {
			g := kv.Graph()
			// Create a zero tensor for the padding region.
			zeroPad := Const(g, make([]float32, numLayers*batchSize*kvHeads*padAmount*headDim))
			zeroPad = Reshape(zeroPad, numLayers, batchSize, kvHeads, padAmount, headDim)
			zeroPad = ConvertDType(zeroPad, kv.DType())
			return Concatenate([]*Node{kv, zeroPad}, 3)
		},
	)
	return padExec.MustExec(kvTensor)[0]
}
