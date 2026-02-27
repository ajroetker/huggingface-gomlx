package models

import (
	"fmt"
	"strings"

	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// WeightSource abstracts over different weight storage formats (safetensors, GGUF).
type WeightSource interface {
	// GetTensor loads a single tensor by name.
	GetTensor(name string) (*tensors.Tensor, error)

	// ListTensorNames returns all available tensor names.
	ListTensorNames() []string
}

// SafetensorsSource adapts *safetensors.Model to the WeightSource interface.
type SafetensorsSource struct {
	Model *safetensors.Model
}

// GetTensor loads a tensor from the safetensors model.
func (s *SafetensorsSource) GetTensor(name string) (*tensors.Tensor, error) {
	tn, err := s.Model.GetTensor(name)
	if err != nil {
		return nil, err
	}
	return tn.Tensor, nil
}

// ListTensorNames returns all tensor names in the safetensors model.
func (s *SafetensorsSource) ListTensorNames() []string {
	return s.Model.ListTensorNames()
}

// GGUFSource adapts *gguf.Model to the WeightSource interface.
type GGUFSource struct {
	Model *gguf.Model
}

// GetTensor loads a tensor from the GGUF model, dequantizing if needed.
func (g *GGUFSource) GetTensor(name string) (*tensors.Tensor, error) {
	tn, err := g.Model.GetTensor(name)
	if err != nil {
		return nil, err
	}
	return tn.Tensor, nil
}

// ListTensorNames returns all tensor names in the GGUF model.
func (g *GGUFSource) ListTensorNames() []string {
	return g.Model.ListTensorNames()
}

// LoadWeightsFromMapping loads weights from a WeightSource into a GoMLX context
// using the given mapping from tensor names to context scope paths.
// Missing tensors (not found errors) are silently skipped.
func LoadWeightsFromMapping(weights WeightSource, mapping map[string]string, ctx *context.Context) error {
	for tensorKey, scopePath := range mapping {
		tensor, err := weights.GetTensor(tensorKey)
		if err != nil {
			// Skip missing weights.
			if strings.Contains(err.Error(), "not found") {
				continue
			}
			return fmt.Errorf("failed to load tensor %q: %w", tensorKey, err)
		}

		// Navigate to the right scope and create variable.
		scopeParts := strings.Split(scopePath, "/")
		varCtx := ctx
		for _, part := range scopeParts[:len(scopeParts)-1] {
			varCtx = varCtx.In(part)
		}
		varName := scopeParts[len(scopeParts)-1]
		varCtx.VariableWithValue(varName, tensor)
	}

	return nil
}
