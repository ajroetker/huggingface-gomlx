// Package gemma3 provides a Gemma 3 architecture implementation for GoMLX.
//
// Gemma 3 uses:
//   - RoPE (Rotary Position Embedding) for positions
//   - RMSNorm with 4 norms per layer (pre/post attention, pre/post FFN)
//   - QK-norm (RMSNorm on Q and K per-head after projection, before RoPE)
//   - GeLU activation in gated MLP (not SiLU like Llama)
//   - Grouped Query Attention (GQA)
//   - Hybrid local/global attention (sliding window for local layers)
//   - Embedding scaling by sqrt(hidden_size)
//   - Explicit head_dim (not hidden_size/num_heads)
//
// Reference: https://arxiv.org/abs/2503.19786
package gemma3

import (
	"fmt"
	"math"
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	models "github.com/ajroetker/huggingface-gomlx"
	"github.com/ajroetker/huggingface-gomlx/architectures/common"
)

func init() {
	models.RegisterArchitecture("gemma3", func() models.ArchitectureBuilder { return &Builder{} })
}

// Config holds Gemma 3-specific configuration.
type Config struct {
	*models.BaseConfig

	// Attention.
	HeadDim      int `json:"head_dim"`       // Explicit head dimension (e.g., 256)
	NumKVHeads   int `json:"num_kv_heads"`    // Key-value heads for GQA
	SlidingWindow int `json:"sliding_window"` // Window size for local attention layers

	// RoPE.
	RopeTheta float64 `json:"rope_theta"`

	// Normalization.
	RMSNormEps float64 `json:"rms_norm_eps"`
}

// KVHeads returns the number of key-value heads.
func (c *Config) KVHeads() int {
	if c.NumKVHeads > 0 {
		return c.NumKVHeads
	}
	return c.NumAttentionHeads
}

// HeadsPerKVGroup returns how many query heads share each KV head.
func (c *Config) HeadsPerKVGroup() int {
	return c.NumAttentionHeads / c.KVHeads()
}

// IsLocalAttentionLayer returns true if the given layer uses sliding window (local) attention.
// Gemma 3 uses a repeating pattern: 5 local layers followed by 1 global layer.
func (c *Config) IsLocalAttentionLayer(layerIdx int) bool {
	// Pattern: layers 0-4 local, layer 5 global, layers 6-10 local, layer 11 global, ...
	return layerIdx%6 != 5
}

// Builder implements the Gemma 3 architecture.
type Builder struct {
	config *Config
	isGGUF bool
}

// Name returns the architecture name.
func (b *Builder) Name() string {
	return "Gemma3"
}

// ParseConfig extracts Gemma 3-specific configuration from BaseConfig.Raw.
func (b *Builder) ParseConfig(base *models.BaseConfig) error {
	b.config = &Config{BaseConfig: base}

	if v, ok := base.GetInt("head_dim"); ok {
		b.config.HeadDim = v
	} else {
		// Gemma 3 4B default.
		b.config.HeadDim = 256
	}

	if v, ok := base.GetInt("num_key_value_heads"); ok {
		b.config.NumKVHeads = v
	}

	if v, ok := base.GetInt("sliding_window"); ok {
		b.config.SlidingWindow = v
	} else {
		b.config.SlidingWindow = 1024
	}

	if v, ok := base.GetFloat("rope_theta"); ok {
		b.config.RopeTheta = v
	} else {
		b.config.RopeTheta = 1e6
	}

	if v, ok := base.GetFloat("rms_norm_eps"); ok {
		b.config.RMSNormEps = v
	} else {
		b.config.RMSNormEps = 1e-6
	}

	return nil
}

// Config returns the base configuration.
func (b *Builder) Config() *models.BaseConfig {
	return b.config.BaseConfig
}

// LoadWeights loads weights into the GoMLX context.
// Selects the appropriate weight mapping based on the weight source type.
func (b *Builder) LoadWeights(ctx *context.Context, weights models.WeightSource) error {
	var mapping map[string]string
	if _, ok := weights.(*models.GGUFSource); ok {
		mapping = b.ggufWeightMapping()
		b.isGGUF = true
	} else {
		mapping = b.hfWeightMapping()
	}
	return models.LoadWeightsFromMapping(weights, mapping, ctx)
}

// WeightMapping returns the GGUF weight mapping (primary target).
func (b *Builder) WeightMapping() map[string]string {
	return b.ggufWeightMapping()
}

// ggufWeightMapping returns the mapping from GGUF tensor names to context scope paths.
func (b *Builder) ggufWeightMapping() map[string]string {
	mapping := make(map[string]string)
	cfg := b.config

	// Embeddings.
	mapping["token_embd.weight"] = "embeddings/embeddings"

	// Layers.
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		blk := fmt.Sprintf("blk.%d", i)
		scope := fmt.Sprintf("layers/%d", i)

		// Pre-attention norm.
		mapping[blk+".attn_norm.weight"] = scope + "/input_norm/weight"

		// Attention projections.
		mapping[blk+".attn_q.weight"] = scope + "/attention/query/weights"
		mapping[blk+".attn_k.weight"] = scope + "/attention/key/weights"
		mapping[blk+".attn_v.weight"] = scope + "/attention/value/weights"
		mapping[blk+".attn_output.weight"] = scope + "/attention/output/weights"

		// QK-norm.
		mapping[blk+".attn_q_norm.weight"] = scope + "/attention/q_norm/weight"
		mapping[blk+".attn_k_norm.weight"] = scope + "/attention/k_norm/weight"

		// Post-attention norm.
		mapping[blk+".attn_post_norm.weight"] = scope + "/post_attn_norm/weight"

		// Pre-FFN norm.
		mapping[blk+".ffn_norm.weight"] = scope + "/ffn_norm/weight"

		// MLP (gated).
		mapping[blk+".ffn_gate.weight"] = scope + "/mlp/gate/weights"
		mapping[blk+".ffn_up.weight"] = scope + "/mlp/up/weights"
		mapping[blk+".ffn_down.weight"] = scope + "/mlp/down/weights"

		// Post-FFN norm.
		mapping[blk+".ffn_post_norm.weight"] = scope + "/post_ffn_norm/weight"
	}

	// Final norm.
	mapping["output_norm.weight"] = "norm/weight"

	// LM head (may be absent if tied to embeddings).
	mapping["output.weight"] = "lm_head/weights"

	return mapping
}

// hfWeightMapping returns the mapping from HuggingFace tensor names to context scope paths.
func (b *Builder) hfWeightMapping() map[string]string {
	mapping := make(map[string]string)
	cfg := b.config
	prefix := "model"

	// Embeddings.
	mapping[prefix+".embed_tokens.weight"] = "embeddings/embeddings"

	// Layers.
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		lp := fmt.Sprintf("%s.layers.%d", prefix, i)
		scope := fmt.Sprintf("layers/%d", i)

		// Pre-attention norm.
		mapping[lp+".input_layernorm.weight"] = scope + "/input_norm/weight"

		// Attention projections.
		mapping[lp+".self_attn.q_proj.weight"] = scope + "/attention/query/weights"
		mapping[lp+".self_attn.k_proj.weight"] = scope + "/attention/key/weights"
		mapping[lp+".self_attn.v_proj.weight"] = scope + "/attention/value/weights"
		mapping[lp+".self_attn.o_proj.weight"] = scope + "/attention/output/weights"

		// QK-norm.
		mapping[lp+".self_attn.q_norm.weight"] = scope + "/attention/q_norm/weight"
		mapping[lp+".self_attn.k_norm.weight"] = scope + "/attention/k_norm/weight"

		// Post-attention norm.
		mapping[lp+".post_attention_layernorm.weight"] = scope + "/post_attn_norm/weight"

		// Pre-FFN norm.
		mapping[lp+".pre_feedforward_layernorm.weight"] = scope + "/ffn_norm/weight"

		// MLP (gated).
		mapping[lp+".mlp.gate_proj.weight"] = scope + "/mlp/gate/weights"
		mapping[lp+".mlp.up_proj.weight"] = scope + "/mlp/up/weights"
		mapping[lp+".mlp.down_proj.weight"] = scope + "/mlp/down/weights"

		// Post-FFN norm.
		mapping[lp+".post_feedforward_layernorm.weight"] = scope + "/post_ffn_norm/weight"
	}

	// Final norm.
	mapping[prefix+".norm.weight"] = "norm/weight"

	// LM head.
	mapping["lm_head.weight"] = "lm_head/weights"

	return mapping
}

// BuildEmbeddings builds the embedding layer with Gemma 3 scaling.
func (b *Builder) BuildEmbeddings(ctx *context.Context, inputIDs *Node) *Node {
	embCtx := ctx.In("embeddings")
	cfg := b.config

	embeddings := common.Embedding(embCtx, inputIDs, cfg.VocabSize, cfg.HiddenSize)

	// Gemma 3 scales embeddings by sqrt(hidden_size).
	scale := ConstAs(embeddings, math.Sqrt(float64(cfg.HiddenSize)))
	embeddings = Mul(embeddings, scale)

	return embeddings
}

// BuildAttention builds the self-attention layer with QK-norm, RoPE, and optional sliding window.
func (b *Builder) BuildAttention(ctx *context.Context, hidden, positionIDs *Node, layerIdx int) *Node {
	cfg := b.config
	attnCtx := ctx.In("attention")

	batchSize := hidden.Shape().Dimensions[0]
	seqLen := hidden.Shape().Dimensions[1]
	headDim := cfg.HeadDim
	kvHeads := cfg.KVHeads()
	headsPerGroup := cfg.HeadsPerKVGroup()

	// Q, K, V projections (no bias).
	query := common.DenseWeightOnly(attnCtx.In("query"), hidden)
	key := common.DenseWeightOnly(attnCtx.In("key"), hidden)
	value := common.DenseWeightOnly(attnCtx.In("value"), hidden)

	// Reshape for multi-head attention.
	// Query: [batch, seq, num_heads * head_dim] -> [batch, heads, seq, head_dim]
	query = Reshape(query, batchSize, seqLen, cfg.NumAttentionHeads, headDim)
	query = Transpose(query, 1, 2)

	// Key/Value: [batch, seq, kv_heads * head_dim] -> [batch, kv_heads, seq, head_dim]
	key = Reshape(key, batchSize, seqLen, kvHeads, headDim)
	key = Transpose(key, 1, 2)

	value = Reshape(value, batchSize, seqLen, kvHeads, headDim)
	value = Transpose(value, 1, 2)

	// QK-norm: apply RMSNorm to Q and K per-head (on the head_dim axis).
	// Shape is [batch, heads, seq, head_dim], norm over last axis.
	qNormCtx := attnCtx.In("q_norm")
	kNormCtx := attnCtx.In("k_norm")
	query = common.RMSNorm(qNormCtx, query, cfg.RMSNormEps)
	key = common.RMSNorm(kNormCtx, key, cfg.RMSNormEps)

	// Apply RoPE to query and key.
	query, key = common.RoPE(query, key, positionIDs, cfg.RopeTheta, seqLen, headDim)

	// Expand KV heads for grouped query attention.
	if headsPerGroup > 1 {
		key = repeatKV(key, headsPerGroup)
		value = repeatKV(value, headsPerGroup)
	}

	// Attention scores: Q @ K^T / sqrt(head_dim)
	scores := Einsum("bhqd,bhkd->bhqk", query, key)
	scale := ConstAs(scores, 1.0/math.Sqrt(float64(headDim)))
	scores = Mul(scores, scale)

	// Apply causal mask (global or sliding window depending on layer).
	g := hidden.Graph()
	if cfg.IsLocalAttentionLayer(layerIdx) && cfg.SlidingWindow > 0 {
		mask := common.CreateSlidingWindowCausalMask(g, seqLen, cfg.SlidingWindow, scores.DType())
		scores = Add(scores, mask)
	} else {
		mask := common.CreateCausalMask(g, seqLen, scores.DType())
		scores = Add(scores, mask)
	}

	// Softmax and attention output.
	attnWeights := Softmax(scores, -1)
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

	// Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
	attnOutput = Transpose(attnOutput, 1, 2)
	attnOutput = Reshape(attnOutput, batchSize, seqLen, cfg.NumAttentionHeads*headDim)

	// Output projection.
	attnOutput = common.DenseWeightOnly(attnCtx.In("output"), attnOutput)

	return attnOutput
}

// repeatKV repeats key/value heads for grouped query attention.
func repeatKV(x *Node, repeats int) *Node {
	if repeats == 1 {
		return x
	}
	batchSize := x.Shape().Dimensions[0]
	kvHeads := x.Shape().Dimensions[1]
	seqLen := x.Shape().Dimensions[2]
	headDim := x.Shape().Dimensions[3]

	x = InsertAxes(x, 2)
	x = BroadcastToDims(x, batchSize, kvHeads, repeats, seqLen, headDim)
	return Reshape(x, batchSize, kvHeads*repeats, seqLen, headDim)
}

// BuildMLP builds the GeLU-gated MLP.
func (b *Builder) BuildMLP(ctx *context.Context, hidden *Node) *Node {
	mlpCtx := ctx.In("mlp")

	// Gated GeLU: gate_proj(x) * GeLU(up_proj(x)), then down_proj.
	gate := common.DenseWeightOnly(mlpCtx.In("gate"), hidden)
	up := common.DenseWeightOnly(mlpCtx.In("up"), hidden)

	activated := Mul(activations.GeluApproximate(gate), up)

	return common.DenseWeightOnly(mlpCtx.In("down"), activated)
}

// BuildDecoderLayer builds a single decoder layer with 4 norms.
func (b *Builder) BuildDecoderLayer(ctx *context.Context, hidden, positionIDs *Node, layerIdx int) *Node {
	cfg := b.config

	// Pre-attention RMSNorm.
	normalized := common.RMSNorm(ctx.In("input_norm"), hidden, cfg.RMSNormEps)

	// Self-attention with residual.
	attnOutput := b.BuildAttention(ctx, normalized, positionIDs, layerIdx)

	// Post-attention RMSNorm (applied to attention output before residual add).
	attnOutput = common.RMSNorm(ctx.In("post_attn_norm"), attnOutput, cfg.RMSNormEps)
	hidden = Add(hidden, attnOutput)

	// Pre-FFN RMSNorm.
	normalized = common.RMSNorm(ctx.In("ffn_norm"), hidden, cfg.RMSNormEps)

	// MLP with residual.
	mlpOutput := b.BuildMLP(ctx, normalized)

	// Post-FFN RMSNorm (applied to MLP output before residual add).
	mlpOutput = common.RMSNorm(ctx.In("post_ffn_norm"), mlpOutput, cfg.RMSNormEps)
	hidden = Add(hidden, mlpOutput)

	return hidden
}

// BuildDecoder builds the full decoder stack.
func (b *Builder) BuildDecoder(ctx *context.Context, hidden, positionIDs *Node) *Node {
	for i := 0; i < b.config.NumHiddenLayers; i++ {
		hidden = b.BuildDecoderLayer(ctx.In("layers").In(itoa(i)), hidden, positionIDs, i)
	}

	// Final normalization.
	hidden = common.RMSNorm(ctx.In("norm"), hidden, b.config.RMSNormEps)

	return hidden
}

// Forward runs the forward pass.
func (b *Builder) Forward(ctx *context.Context, inputIDs, positionIDs *Node) *Node {
	g := inputIDs.Graph()
	cfg := b.config

	// Embeddings with scaling.
	hidden := b.BuildEmbeddings(ctx, inputIDs)

	// Create position IDs if not provided.
	if positionIDs == nil {
		batchSize := inputIDs.Shape().Dimensions[0]
		seqLen := inputIDs.Shape().Dimensions[1]
		positionIDs = common.GetPositionIDs(g, batchSize, seqLen)
	}

	// Decoder.
	hidden = b.BuildDecoder(ctx, hidden, positionIDs)

	// LM head. Check if lm_head weights exist; if not, use tied embeddings.
	lmHeadCtx := ctx.In("lm_head")
	lmHeadVar := lmHeadCtx.GetVariableByScopeAndName(lmHeadCtx.Scope(), "weights")
	if lmHeadVar != nil {
		hidden = common.DenseWeightOnly(lmHeadCtx, hidden)
	} else {
		// Tied embeddings: reuse token_embd.weight.
		embCtx := ctx.In("embeddings")
		embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
		if embVar != nil {
			embWeights := embVar.ValueGraph(g)
			// Embedding is [vocab, hidden]. Use Einsum for matmul: hidden @ emb^T.
			batchSize := hidden.Shape().Dimensions[0]
			seqLen := hidden.Shape().Dimensions[1]
			hiddenFlat := Reshape(hidden, batchSize*seqLen, cfg.HiddenSize)
			logits := Einsum("bh,vh->bv", hiddenFlat, embWeights)
			hidden = Reshape(logits, batchSize, seqLen, cfg.VocabSize)
		}
	}

	return hidden
}

// CreateExecGraphFn returns a function suitable for context.NewExec.
func (b *Builder) CreateExecGraphFn() func(*context.Context, *Node) *Node {
	return func(ctx *context.Context, inputIDs *Node) *Node {
		return b.Forward(ctx, inputIDs, nil)
	}
}

// GetVariableShape returns the expected shape for a variable.
func (b *Builder) GetVariableShape(name string) shapes.Shape {
	cfg := b.config

	switch {
	case strings.Contains(name, "embed_tokens") || strings.Contains(name, "token_embd"):
		return shapes.Make(dtypes.Float32, cfg.VocabSize, cfg.HiddenSize)
	case strings.Contains(name, "attn_q") || strings.Contains(name, "q_proj"):
		return shapes.Make(dtypes.Float32, cfg.NumAttentionHeads*cfg.HeadDim, cfg.HiddenSize)
	case strings.Contains(name, "attn_k") || strings.Contains(name, "k_proj"):
		return shapes.Make(dtypes.Float32, cfg.KVHeads()*cfg.HeadDim, cfg.HiddenSize)
	case strings.Contains(name, "attn_v") || strings.Contains(name, "v_proj"):
		return shapes.Make(dtypes.Float32, cfg.KVHeads()*cfg.HeadDim, cfg.HiddenSize)
	case strings.Contains(name, "ffn_gate") || strings.Contains(name, "gate_proj"):
		return shapes.Make(dtypes.Float32, cfg.IntermediateSize, cfg.HiddenSize)
	case strings.Contains(name, "ffn_up") || strings.Contains(name, "up_proj"):
		return shapes.Make(dtypes.Float32, cfg.IntermediateSize, cfg.HiddenSize)
	case strings.Contains(name, "ffn_down") || strings.Contains(name, "down_proj"):
		return shapes.Make(dtypes.Float32, cfg.HiddenSize, cfg.IntermediateSize)
	default:
		return shapes.Shape{}
	}
}

func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}
