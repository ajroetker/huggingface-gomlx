// Package common provides shared components for transformer architectures.
package common

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// LayerNorm applies layer normalization using pre-loaded weights.
// Expects variables "gain" and "offset" in the context scope.
func LayerNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	g := x.Graph()

	gainVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "gain")
	if gainVar == nil {
		panic(fmt.Sprintf("LayerNorm: missing variable 'gain' in scope %q", ctx.Scope()))
	}
	gain := gainVar.ValueGraph(g)

	offsetVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "offset")
	if offsetVar == nil {
		panic(fmt.Sprintf("LayerNorm: missing variable 'offset' in scope %q", ctx.Scope()))
	}
	offset := offsetVar.ValueGraph(g)

	return ApplyLayerNormWithParams(x, gain, offset, epsilon)
}

// ApplyLayerNormWithParams applies layer normalization with explicit parameters.
func ApplyLayerNormWithParams(x, gain, offset *Node, epsilon float64) *Node {
	// Normalize over the last axis.
	mean := ReduceAndKeep(x, ReduceMean, -1)
	normalized := Sub(x, mean)
	variance := ReduceAndKeep(Square(normalized), ReduceMean, -1)
	eps := ConstAs(x, epsilon)
	normalized = Div(normalized, Sqrt(Add(variance, eps)))

	// Reshape gain and offset to broadcast with x.
	xRank := x.Shape().Rank()
	broadcastShape := make([]int, xRank)
	for i := range broadcastShape {
		broadcastShape[i] = 1
	}
	broadcastShape[xRank-1] = gain.Shape().Dimensions[0]

	gain = Reshape(gain, broadcastShape...)
	offset = Reshape(offset, broadcastShape...)

	// Apply gain and offset.
	normalized = Mul(normalized, gain)
	normalized = Add(normalized, offset)

	return normalized
}

// RMSNorm applies root mean square layer normalization (used by Llama).
// Expects variable "weight" in the context scope.
func RMSNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	g := x.Graph()

	weightVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weight")
	if weightVar == nil {
		panic(fmt.Sprintf("RMSNorm: missing variable 'weight' in scope %q", ctx.Scope()))
	}
	weight := weightVar.ValueGraph(g)

	return ApplyRMSNormWithParams(x, weight, epsilon)
}

// ApplyRMSNormWithParams applies RMS normalization with explicit weight parameter.
func ApplyRMSNormWithParams(x, weight *Node, epsilon float64) *Node {
	// RMS = sqrt(mean(x^2))
	variance := ReduceAndKeep(Square(x), ReduceMean, -1)
	eps := ConstAs(x, epsilon)
	rms := Sqrt(Add(variance, eps))
	normalized := Div(x, rms)

	// Reshape weight to broadcast with x.
	xRank := x.Shape().Rank()
	broadcastShape := make([]int, xRank)
	for i := range broadcastShape {
		broadcastShape[i] = 1
	}
	broadcastShape[xRank-1] = weight.Shape().Dimensions[0]

	weight = Reshape(weight, broadcastShape...)

	return Mul(normalized, weight)
}
