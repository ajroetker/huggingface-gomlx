//go:build integration

package models_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/require"

	"github.com/gomlx/huggingface-gomlx"
	"github.com/gomlx/huggingface-gomlx/architectures/bert"

	// Import architectures to register them.
	_ "github.com/gomlx/huggingface-gomlx/architectures/bert"
)

// getBackend returns the XLA backend for testing.
func getBackend() backends.Backend {
	// Auto-install XLA PJRT if not available.
	if err := xla.AutoInstall(); err != nil {
		panic(fmt.Sprintf("failed to auto-install XLA: %v", err))
	}
	// Use default config which will pick up the versioned plugin.
	backends.DefaultConfig = ""
	return backends.MustNew()
}

// TestBERTGraphBuild verifies that BERT can build a computation graph
// and check output shapes with mock weights.
func TestBERTGraphBuild(t *testing.T) {
	// Parse a minimal BERT config.
	configJSON := `{
		"model_type": "bert",
		"vocab_size": 100,
		"hidden_size": 32,
		"num_hidden_layers": 2,
		"num_attention_heads": 2,
		"intermediate_size": 64,
		"hidden_act": "gelu",
		"layer_norm_eps": 1e-12,
		"max_position_embeddings": 64
	}`

	cfg, err := models.ParseConfigContent([]byte(configJSON))
	require.NoError(t, err)

	// Get the BERT builder and configure it.
	builder, err := models.NewBuilder(cfg.ModelType)
	require.NoError(t, err)

	err = builder.ParseConfig(cfg)
	require.NoError(t, err)

	// Type assert to BERT builder to access Forward method.
	bertBuilder, ok := builder.(*bert.Builder)
	require.True(t, ok, "expected BERT builder")

	// Create context with mock weights.
	ctx := context.New()
	initMockBERTWeights(ctx, cfg)

	// Get backend (simplego is a pure Go backend, no XLA needed).
	backend := getBackend()

	// Build graph.
	batchSize := 2
	seqLen := 8

	g := graph.NewGraph(backend, "bert_test")

	inputIDs := graph.Parameter(g, "input_ids", shapes.Make(dtypes.Int32, batchSize, seqLen))
	attentionMask := graph.Parameter(g, "attention_mask", shapes.Make(dtypes.Float32, batchSize, seqLen))

	// Build BERT forward pass.
	// Use Reuse() to allow existing variables to be reused (instead of erroring on duplicates).
	reuseCtx := ctx.Reuse()
	hidden, pooled := bertBuilder.Forward(reuseCtx, inputIDs, attentionMask, nil, nil)

	// Check output shapes.
	expectedHiddenShape := shapes.Make(dtypes.Float32, batchSize, seqLen, cfg.HiddenSize)
	require.True(t, hidden.Shape().Equal(expectedHiddenShape),
		"hidden shape: got %s, want %s", hidden.Shape(), expectedHiddenShape)

	if pooled != nil {
		expectedPooledShape := shapes.Make(dtypes.Float32, batchSize, cfg.HiddenSize)
		require.True(t, pooled.Shape().Equal(expectedPooledShape),
			"pooled shape: got %s, want %s", pooled.Shape(), expectedPooledShape)
	}

	t.Logf("Graph built successfully!")
	t.Logf("  Hidden shape: %s", hidden.Shape())
	if pooled != nil {
		t.Logf("  Pooled shape: %s", pooled.Shape())
	}
}

// TestBERTExecution actually runs the BERT model with mock weights.
func TestBERTExecution(t *testing.T) {
	// Parse a minimal BERT config.
	configJSON := `{
		"model_type": "bert",
		"vocab_size": 100,
		"hidden_size": 32,
		"num_hidden_layers": 2,
		"num_attention_heads": 2,
		"intermediate_size": 64,
		"hidden_act": "gelu",
		"layer_norm_eps": 1e-12,
		"max_position_embeddings": 64
	}`

	cfg, err := models.ParseConfigContent([]byte(configJSON))
	require.NoError(t, err)

	// Get the BERT builder and configure it.
	builder, err := models.NewBuilder(cfg.ModelType)
	require.NoError(t, err)

	err = builder.ParseConfig(cfg)
	require.NoError(t, err)

	// Type assert to BERT builder.
	bertBuilder, ok := builder.(*bert.Builder)
	require.True(t, ok, "expected BERT builder")

	// Create context with mock weights.
	ctx := context.New()
	initMockBERTWeights(ctx, cfg)

	// Get backend.
	backend := getBackend()

	// Create executable - the function signature must match context.NewExec expectations.
	batchSize := 2
	seqLen := 8

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, inputIDs, attentionMask *graph.Node) *graph.Node {
		// Use Reuse() to allow existing variables to be reused.
		reuseCtx := ctx.Reuse()
		hidden, _ := bertBuilder.Forward(reuseCtx, inputIDs, attentionMask, nil, nil)
		return hidden
	})
	require.NoError(t, err)

	// Create input tensors.
	inputIDsData := make([]int32, batchSize*seqLen)
	for i := range inputIDsData {
		inputIDsData[i] = int32(i % 100)
	}
	inputIDs := tensors.FromFlatDataAndDimensions(inputIDsData, batchSize, seqLen)

	attentionMaskData := make([]float32, batchSize*seqLen)
	for i := range attentionMaskData {
		attentionMaskData[i] = 1.0
	}
	attentionMask := tensors.FromFlatDataAndDimensions(attentionMaskData, batchSize, seqLen)

	// Run inference.
	results := exec.MustExec(inputIDs, attentionMask)
	require.Len(t, results, 1)

	output := results[0]
	t.Logf("Execution successful!")
	t.Logf("  Output shape: %s", output.Shape())
	t.Logf("  Output dtype: %s", output.DType())

	// Verify output shape.
	require.Equal(t, 3, output.Shape().Rank())
	require.Equal(t, batchSize, output.Shape().Dimensions[0])
	require.Equal(t, seqLen, output.Shape().Dimensions[1])
	require.Equal(t, cfg.HiddenSize, output.Shape().Dimensions[2])

	// Print some output values for verification.
	// Output is 3D [batch, seq, hidden], so Value() returns [][][]float32
	data3d := output.Value().([][][]float32)
	t.Logf("  First token output (first 10 values): %v", data3d[0][0][:min(10, len(data3d[0][0]))])
}

// initMockBERTWeights creates random weights in the context for testing.
func initMockBERTWeights(ctx *context.Context, cfg *models.BaseConfig) {
	embCtx := ctx.In("embeddings")

	// Word embeddings - use small random values.
	embCtx.VariableWithValue("embeddings", makeTensor2D(cfg.VocabSize, cfg.HiddenSize))

	// Position embeddings.
	embCtx.VariableWithValue("position_embeddings", makeTensor2D(cfg.MaxPositionEmbeddings, cfg.HiddenSize))

	// Token type embeddings (2 types: sentence A and B).
	embCtx.VariableWithValue("token_type_embeddings", makeTensor2D(2, cfg.HiddenSize))

	// Layer norm - gain initialized to 1, offset to 0.
	lnCtx := embCtx.In("layer_norm")
	lnCtx.VariableWithValue("gain", makeOnes(cfg.HiddenSize))
	lnCtx.VariableWithValue("offset", makeZeros(cfg.HiddenSize))

	// Encoder layers.
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerCtx := ctx.In("encoder").In("layer").In(fmt.Sprintf("%d", i))

		// Attention Q, K, V.
		attnCtx := layerCtx.In("attention")
		for _, name := range []string{"query", "key", "value"} {
			projCtx := attnCtx.In(name)
			projCtx.VariableWithValue("weights", makeTensor2D(cfg.HiddenSize, cfg.HiddenSize))
			projCtx.VariableWithValue("biases", makeZeros(cfg.HiddenSize))
		}

		// Attention output.
		outCtx := attnCtx.In("output").In("dense")
		outCtx.VariableWithValue("weights", makeTensor2D(cfg.HiddenSize, cfg.HiddenSize))
		outCtx.VariableWithValue("biases", makeZeros(cfg.HiddenSize))

		// Attention layer norm.
		attnLnCtx := attnCtx.In("output").In("layer_norm")
		attnLnCtx.VariableWithValue("gain", makeOnes(cfg.HiddenSize))
		attnLnCtx.VariableWithValue("offset", makeZeros(cfg.HiddenSize))

		// Feed-forward.
		ffCtx := layerCtx.In("ff")
		intCtx := ffCtx.In("intermediate")
		intCtx.VariableWithValue("weights", makeTensor2D(cfg.IntermediateSize, cfg.HiddenSize))
		intCtx.VariableWithValue("biases", makeZeros(cfg.IntermediateSize))

		ffOutCtx := ffCtx.In("output")
		ffOutCtx.VariableWithValue("weights", makeTensor2D(cfg.HiddenSize, cfg.IntermediateSize))
		ffOutCtx.VariableWithValue("biases", makeZeros(cfg.HiddenSize))

		// FF layer norm.
		ffLnCtx := ffCtx.In("layer_norm")
		ffLnCtx.VariableWithValue("gain", makeOnes(cfg.HiddenSize))
		ffLnCtx.VariableWithValue("offset", makeZeros(cfg.HiddenSize))
	}

	// Pooler.
	poolerCtx := ctx.In("pooler")
	poolerCtx.VariableWithValue("weights", makeTensor2D(cfg.HiddenSize, cfg.HiddenSize))
	poolerCtx.VariableWithValue("biases", makeZeros(cfg.HiddenSize))
}

// makeTensor2D creates a 2D tensor with small deterministic values.
func makeTensor2D(rows, cols int) *tensors.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}
	return tensors.FromFlatDataAndDimensions(data, rows, cols)
}

func makeOnes(size int) []float32 {
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}
	return data
}

func makeZeros(size int) []float32 {
	return make([]float32, size)
}
