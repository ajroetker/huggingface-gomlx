package models

import (
	"fmt"
	"sync"

	"github.com/gomlx/gomlx/pkg/ml/context"
)

// ModelInputs contains the standard inputs for transformer models.
type ModelInputs struct {
	InputIDs      interface{} // Token IDs: [batch, seq_len]
	AttentionMask interface{} // Attention mask: [batch, seq_len]
	TokenTypeIDs  interface{} // Optional segment IDs: [batch, seq_len]
	PositionIDs   interface{} // Optional position IDs: [batch, seq_len]
}

// ModelOutputs contains the standard outputs from transformer models.
type ModelOutputs struct {
	LastHiddenState interface{} // Final hidden states: [batch, seq_len, hidden_size]
	PoolerOutput    interface{} // Pooled output: [batch, hidden_size] (optional)
	AllHiddenStates []interface{} // All layer hidden states (optional)
	Attentions      []interface{} // All attention weights (optional)
}

// ArchitectureBuilder defines the interface for building model architectures.
type ArchitectureBuilder interface {
	// Name returns the architecture name for logging/debugging.
	Name() string

	// ParseConfig extracts architecture-specific config from BaseConfig.Raw.
	// This is called after the base config is parsed.
	ParseConfig(base *BaseConfig) error

	// LoadWeights loads weights into the GoMLX context from any weight source.
	// The context should use hierarchical scopes matching WeightMapping.
	LoadWeights(ctx *context.Context, weights WeightSource) error

	// WeightMapping returns the mapping from safetensors keys to context scope paths.
	// Used for debugging and documentation.
	WeightMapping() map[string]string

	// Config returns the base configuration.
	Config() *BaseConfig
}

// BuilderConstructor is a function that creates a new ArchitectureBuilder.
type BuilderConstructor func() ArchitectureBuilder

// registry holds all registered architecture builders.
var (
	registry   = make(map[string]BuilderConstructor)
	registryMu sync.RWMutex
)

// RegisterArchitecture registers an architecture builder for a model type.
// Multiple model types can map to the same builder (e.g., "bert" and "roberta").
func RegisterArchitecture(modelType string, constructor BuilderConstructor) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry[modelType] = constructor
}

// GetArchitecture returns the builder constructor for a model type.
func GetArchitecture(modelType string) (BuilderConstructor, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	constructor, ok := registry[modelType]
	return constructor, ok
}

// ListArchitectures returns all registered model types.
func ListArchitectures() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()

	types := make([]string, 0, len(registry))
	for t := range registry {
		types = append(types, t)
	}
	return types
}

// NewBuilder creates a new architecture builder for the given model type.
func NewBuilder(modelType string) (ArchitectureBuilder, error) {
	constructor, ok := GetArchitecture(modelType)
	if !ok {
		return nil, fmt.Errorf("unsupported model type %q; supported types: %v", modelType, ListArchitectures())
	}
	return constructor(), nil
}
