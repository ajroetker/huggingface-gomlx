package common

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// DenseWithBias applies a dense layer using pre-loaded weights.
// Expects variables "weights" and "biases" in the context scope.
// Handles 2D [batch, features] and 3D [batch, seq, features] inputs.
func DenseWithBias(ctx *context.Context, x *Node) *Node {
	g := x.Graph()

	weightsVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights")
	if weightsVar == nil {
		panic(fmt.Sprintf("DenseWithBias: missing variable 'weights' in scope %q", ctx.Scope()))
	}
	weights := weightsVar.ValueGraph(g)

	biasesVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "biases")
	if biasesVar == nil {
		panic(fmt.Sprintf("DenseWithBias: missing variable 'biases' in scope %q", ctx.Scope()))
	}
	biases := biasesVar.ValueGraph(g)

	return ApplyDenseWithBias(x, weights, biases)
}

// ApplyDenseWithBias applies a dense layer with explicit weight and bias tensors.
// weights shape: [out_features, in_features] (PyTorch convention)
func ApplyDenseWithBias(x, weights, biases *Node) *Node {
	var output *Node

	if x.Shape().Rank() == 3 {
		// 3D input: [batch, seq, in] @ [out, in].T -> [batch, seq, out]
		output = Einsum("bsi,oi->bso", x, weights)
		// Reshape biases for broadcasting: [out] -> [1, 1, out]
		biases = Reshape(biases, 1, 1, biases.Shape().Dimensions[0])
	} else {
		// 2D input: [batch, in] @ [out, in].T -> [batch, out]
		output = Einsum("bi,oi->bo", x, weights)
		// Reshape biases for broadcasting: [out] -> [1, out]
		biases = Reshape(biases, 1, biases.Shape().Dimensions[0])
	}

	return Add(output, biases)
}

// DenseWeightOnly applies only the weight matrix (no bias).
// Expects variable "weights" in the context scope.
func DenseWeightOnly(ctx *context.Context, x *Node) *Node {
	g := x.Graph()

	weightsVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weights")
	if weightsVar == nil {
		panic(fmt.Sprintf("DenseWeightOnly: missing variable 'weights' in scope %q", ctx.Scope()))
	}
	weights := weightsVar.ValueGraph(g)

	return ApplyDenseWeightOnly(x, weights)
}

// ApplyDenseWeightOnly applies a weight-only dense layer.
func ApplyDenseWeightOnly(x, weights *Node) *Node {
	rank := x.Shape().Rank()
	switch rank {
	case 2:
		return Einsum("bi,oi->bo", x, weights)
	case 3:
		return Einsum("bsi,oi->bso", x, weights)
	case 4:
		// For position embeddings [seq_q, seq_k, hidden]
		return Einsum("qki,oi->qko", x, weights)
	default:
		panic(fmt.Sprintf("DenseWeightOnly: unsupported input rank %d", rank))
	}
}

// MLP applies a 2-layer MLP with ReLU activation.
// Expects variables in scopes "0" (first layer) and "3" (second layer),
// matching PyTorch nn.Sequential indexing with dropout at indices 1 and 2.
func MLP(ctx *context.Context, x *Node) *Node {
	// First layer with ReLU.
	x = DenseWithBias(ctx.In("0"), x)
	x = activations.Relu(x)

	// Second layer.
	x = DenseWithBias(ctx.In("3"), x)

	return x
}

// MLPWithGELU applies a 2-layer MLP with GELU activation.
func MLPWithGELU(ctx *context.Context, x *Node) *Node {
	// First layer with GELU.
	x = DenseWithBias(ctx.In("0"), x)
	x = GELU(x)

	// Second layer.
	x = DenseWithBias(ctx.In("3"), x)

	return x
}

// GELU approximation using the tanh formula.
func GELU(x *Node) *Node {
	// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	sqrt2OverPi := ConstAs(x, 0.7978845608028654) // sqrt(2/pi)
	coeff := ConstAs(x, 0.044715)
	half := ConstAs(x, 0.5)
	one := ConstAs(x, 1.0)

	x3 := Mul(x, Mul(x, x))
	inner := Mul(sqrt2OverPi, Add(x, Mul(coeff, x3)))
	return Mul(half, Mul(x, Add(one, Tanh(inner))))
}

// SiLU (Swish) activation function used by Llama.
func SiLU(x *Node) *Node {
	return Mul(x, Sigmoid(x))
}
