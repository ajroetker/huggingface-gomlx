package common

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// CreateSlidingWindowCausalMask creates a causal attention mask with a sliding window.
// Positions can only attend to at most windowSize previous positions (and themselves).
// Returns a mask of shape [1, 1, seq_len, seq_len] where:
//   - mask[i][j] = 0 if j <= i and i - j < windowSize (within window and causal)
//   - mask[i][j] = -1e9 otherwise
func CreateSlidingWindowCausalMask(g *graph.Graph, seqLen, windowSize int, dtype dtypes.DType) *graph.Node {
	mask := make([]float32, seqLen*seqLen)
	negInf := float32(-1e9)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i || i-j >= windowSize {
				mask[i*seqLen+j] = negInf
			}
		}
	}

	maskNode := graph.Const(g, mask)
	maskNode = graph.Reshape(maskNode, 1, 1, seqLen, seqLen)
	return graph.ConvertDType(maskNode, dtype)
}
