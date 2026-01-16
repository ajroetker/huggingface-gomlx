package common

import (
	"math"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// Embedding retrieves embeddings from the context.
// Uses the GoMLX layers.Embedding which expects "embeddings" variable in scope.
func Embedding(ctx *context.Context, inputIDs *Node, vocabSize, hiddenSize int) *Node {
	embeddings := layers.Embedding(ctx, inputIDs, dtypes.Float32, vocabSize, hiddenSize)

	// Ensure 3D output: [batch, seq, hidden].
	// layers.Embedding may return 2D when seq_len=1.
	if embeddings.Shape().Rank() == 2 {
		embeddings = InsertAxes(embeddings, 1)
	}

	return embeddings
}

// AbsolutePositionEmbedding adds absolute position embeddings.
// Expects "position_embeddings" variable in scope with shape [max_positions, hidden_size].
func AbsolutePositionEmbedding(ctx *context.Context, x *Node, seqLen int) *Node {
	g := x.Graph()

	posEmbVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "position_embeddings")
	if posEmbVar == nil {
		return x // No position embeddings, just return input.
	}
	posEmb := posEmbVar.ValueGraph(g)

	// Slice to sequence length: [max_pos, hidden] -> [seq_len, hidden]
	posEmb = Slice(posEmb, AxisRange(0, seqLen), AxisRange())

	// Add position embeddings (broadcasts over batch).
	return Add(x, posEmb)
}

// BuildRelativePositionEmbeddings creates a [seq_len, seq_len, hidden] tensor of relative position embeddings.
// For each (query_pos=i, key_pos=j) pair, looks up the embedding for relative position (i - j).
// This follows DeBERTa convention where relative_pos = query_pos - key_pos.
func BuildRelativePositionEmbeddings(g *Graph, relEmbeddings *Node, seqLen int) *Node {
	const maxRelPos = 256

	// Build the indices matrix [seq_len, seq_len, 1] for Gather.
	indices := make([]int32, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			relPos := i - j // query_pos - key_pos (DeBERTa convention)
			// Clamp to valid range and add offset.
			if relPos < -maxRelPos {
				relPos = -maxRelPos
			} else if relPos >= maxRelPos {
				relPos = maxRelPos - 1
			}
			idx := relPos + maxRelPos // Shift to [0, 511]
			indices[i*seqLen+j] = int32(idx)
		}
	}

	// Create indices tensor with shape [seq_len, seq_len, 1].
	indicesNode := Const(g, indices)
	indicesNode = Reshape(indicesNode, seqLen, seqLen, 1)

	// Gather position embeddings: [512, hidden] indexed by [seq, seq, 1] -> [seq, seq, hidden]
	return Gather(relEmbeddings, indicesNode)
}

// RoPE applies Rotary Position Embedding to query and key tensors.
// query/key shape: [batch, heads, seq, head_dim]
// positionIDs shape: [batch, seq] or nil (will use sequential positions)
func RoPE(query, key *Node, positionIDs *Node, theta float64, seqLen, headDim int) (*Node, *Node) {
	g := query.Graph()

	// Compute rotary frequencies.
	// freq_i = theta^(-2i/d) for i in [0, d/2)
	invFreq := make([]float32, headDim/2)
	for i := 0; i < headDim/2; i++ {
		invFreq[i] = float32(1.0 / math.Pow(theta, float64(2*i)/float64(headDim)))
	}
	invFreqNode := Const(g, invFreq)
	invFreqNode = Reshape(invFreqNode, 1, 1, headDim/2) // [1, 1, head_dim/2]

	// Create position indices.
	var positions *Node
	if positionIDs != nil {
		positions = ConvertDType(positionIDs, dtypes.Float32)
		positions = Reshape(positions, positions.Shape().Dimensions[0], seqLen, 1) // [batch, seq, 1]
	} else {
		posArray := make([]float32, seqLen)
		for i := 0; i < seqLen; i++ {
			posArray[i] = float32(i)
		}
		positions = Const(g, posArray)
		positions = Reshape(positions, 1, seqLen, 1) // [1, seq, 1]
	}

	// freqs = positions * inv_freq: [batch, seq, head_dim/2]
	freqs := Mul(positions, invFreqNode)

	// Compute sin and cos.
	sinFreqs := Sin(freqs)
	cosFreqs := Cos(freqs)

	// Expand for heads: [batch, 1, seq, head_dim/2]
	sinFreqs = InsertAxes(sinFreqs, 1)
	cosFreqs = InsertAxes(cosFreqs, 1)

	// Apply rotary embedding.
	query = applyRotaryEmb(query, sinFreqs, cosFreqs, headDim)
	key = applyRotaryEmb(key, sinFreqs, cosFreqs, headDim)

	return query, key
}

// applyRotaryEmb applies rotary embedding to a tensor.
// x shape: [batch, heads, seq, head_dim]
// sin/cos shape: [batch, 1, seq, head_dim/2]
func applyRotaryEmb(x, sin, cos *Node, headDim int) *Node {
	// Split into two halves.
	x1 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, headDim/2))
	x2 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(headDim/2, headDim))

	// Rotate: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
	rotatedX1 := Sub(Mul(x1, cos), Mul(x2, sin))
	rotatedX2 := Add(Mul(x2, cos), Mul(x1, sin))

	// Concatenate back.
	return Concatenate([]*Node{rotatedX1, rotatedX2}, -1)
}

// CreateCausalMask creates a causal attention mask.
// Returns a mask of shape [1, 1, seq_len, seq_len] where mask[i][j] = 0 if i >= j else -inf.
func CreateCausalMask(g *Graph, seqLen int, dtype dtypes.DType) *Node {
	mask := make([]float32, seqLen*seqLen)
	negInf := float32(-1e9)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask[i*seqLen+j] = negInf
			}
		}
	}

	maskNode := Const(g, mask)
	maskNode = Reshape(maskNode, 1, 1, seqLen, seqLen)
	return ConvertDType(maskNode, dtype)
}

// ExpandAttentionMask expands attention mask for multi-head attention.
// Input mask: [batch, seq_len] (1 for valid, 0 for masked)
// Output: [batch, 1, 1, seq_len] with 0 for valid, large negative for masked.
func ExpandAttentionMask(mask *Node, dtype dtypes.DType) *Node {
	// Expand dimensions: [batch, seq] -> [batch, 1, 1, seq]
	mask = InsertAxes(mask, 1, 1)

	// Convert to float and apply masking.
	mask = ConvertDType(mask, dtype)

	// Convert 0->large_negative, 1->0
	negInf := ConstAs(mask, float64(-1e9))
	one := ConstAs(mask, 1.0)
	return Mul(Sub(one, mask), negInf)
}

// GetPositionIDs creates sequential position IDs.
func GetPositionIDs(g *Graph, batchSize, seqLen int) *Node {
	positions := make([]int32, seqLen)
	for i := 0; i < seqLen; i++ {
		positions[i] = int32(i)
	}
	posNode := Const(g, positions)
	posNode = Reshape(posNode, 1, seqLen)

	// Broadcast to batch size.
	return BroadcastToDims(posNode, batchSize, seqLen)
}

// CreateSinusoidalPositionEmbedding creates sinusoidal position embeddings.
// Returns tensor of shape [maxLen, hiddenSize].
func CreateSinusoidalPositionEmbedding(g *Graph, maxLen, hiddenSize int, dtype dtypes.DType) *Node {
	posEmb := make([]float32, maxLen*hiddenSize)

	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < hiddenSize; i++ {
			if i%2 == 0 {
				// sin(pos / 10000^(2i/d))
				posEmb[pos*hiddenSize+i] = float32(math.Sin(float64(pos) / math.Pow(10000.0, float64(i)/float64(hiddenSize))))
			} else {
				// cos(pos / 10000^(2(i-1)/d))
				posEmb[pos*hiddenSize+i] = float32(math.Cos(float64(pos) / math.Pow(10000.0, float64(i-1)/float64(hiddenSize))))
			}
		}
	}

	embeddings := Const(g, posEmb)
	embeddings = Reshape(embeddings, maxLen, hiddenSize)
	return ConvertDType(embeddings, dtype)
}

// Variable helper to get variable value from context or create it with a shape.
func GetOrCreateVariable(ctx *context.Context, g *Graph, name string, shape shapes.Shape) *Node {
	v := ctx.GetVariableByScopeAndName(ctx.Scope(), name)
	if v != nil {
		return v.ValueGraph(g)
	}
	return ctx.VariableWithShape(name, shape).ValueGraph(g)
}
