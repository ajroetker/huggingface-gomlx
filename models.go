package models

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/huggingface-gomlx/safetensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// Model represents a loaded Hugging Face model with its weights and architecture.
type Model struct {
	// Config is the parsed model configuration.
	Config *BaseConfig

	// Builder is the architecture-specific builder.
	Builder ArchitectureBuilder

	// Weights contains the loaded safetensors weights.
	Weights *safetensors.File

	// weightsPath is the local path to the weights file.
	weightsPath string
}

// New creates a Model from a Hugging Face repository.
// It downloads config.json and model.safetensors, parses the config,
// and sets up the architecture builder.
func New(repo *hub.Repo) (*Model, error) {
	// Download repository info first.
	if err := repo.DownloadInfo(false); err != nil {
		return nil, errors.Wrap(err, "failed to download repo info")
	}

	// Download and parse config.json.
	configPath, err := repo.DownloadFile("config.json")
	if err != nil {
		return nil, errors.Wrap(err, "failed to download config.json")
	}

	config, err := ParseConfigFile(configPath)
	if err != nil {
		return nil, err
	}

	// Look up architecture builder.
	builder, err := NewBuilder(config.ModelType)
	if err != nil {
		return nil, err
	}

	// Parse architecture-specific config.
	if err := builder.ParseConfig(config); err != nil {
		return nil, errors.Wrapf(err, "failed to parse %s config", config.ModelType)
	}

	// Download safetensors weights.
	weightsPath, err := downloadWeights(repo)
	if err != nil {
		return nil, err
	}

	// Load weights.
	weights, err := safetensors.Open(weightsPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load weights from %s", weightsPath)
	}

	return &Model{
		Config:      config,
		Builder:     builder,
		Weights:     weights,
		weightsPath: weightsPath,
	}, nil
}

// NewFromLocal creates a Model from a local directory containing config.json and model.safetensors.
func NewFromLocal(dir string) (*Model, error) {
	// Parse config.json.
	configPath := filepath.Join(dir, "config.json")
	config, err := ParseConfigFile(configPath)
	if err != nil {
		return nil, err
	}

	// Look up architecture builder.
	builder, err := NewBuilder(config.ModelType)
	if err != nil {
		return nil, err
	}

	// Parse architecture-specific config.
	if err := builder.ParseConfig(config); err != nil {
		return nil, errors.Wrapf(err, "failed to parse %s config", config.ModelType)
	}

	// Find and load weights.
	weightsPath := filepath.Join(dir, "model.safetensors")
	weights, err := safetensors.Open(weightsPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load weights from %s", weightsPath)
	}

	return &Model{
		Config:      config,
		Builder:     builder,
		Weights:     weights,
		weightsPath: weightsPath,
	}, nil
}

// LoadWeightsIntoContext loads all model weights into the given GoMLX context.
// This should be called once before building the computation graph.
func (m *Model) LoadWeightsIntoContext(ctx *context.Context) error {
	return m.Builder.LoadWeights(ctx, m.Weights)
}

// WeightMapping returns the mapping from safetensors keys to context scope paths.
func (m *Model) WeightMapping() map[string]string {
	return m.Builder.WeightMapping()
}

// WeightsPath returns the local path to the weights file.
func (m *Model) WeightsPath() string {
	return m.weightsPath
}

// downloadWeights downloads the model weights, handling both single-file and sharded cases.
func downloadWeights(repo *hub.Repo) (string, error) {
	// Try single file first.
	weightsPath, err := repo.DownloadFile("model.safetensors")
	if err == nil {
		return weightsPath, nil
	}

	// Check if it's a "not found" error or something else.
	if !strings.Contains(err.Error(), "404") && !strings.Contains(err.Error(), "not found") {
		return "", errors.Wrap(err, "failed to download model.safetensors")
	}

	// Try sharded weights.
	indexPath, err := repo.DownloadFile("model.safetensors.index.json")
	if err != nil {
		return "", errors.Wrap(err, "failed to download model weights (tried model.safetensors and model.safetensors.index.json)")
	}

	// For now, we only support single-file weights.
	// TODO: Implement sharded weight loading.
	return "", errors.Errorf("sharded weights not yet supported (found %s)", indexPath)
}

// Summary returns a summary of the model configuration and weights.
func (m *Model) Summary() string {
	var sb strings.Builder
	sb.WriteString("Model Summary:\n")
	sb.WriteString("  Architecture: " + m.Builder.Name() + "\n")
	sb.WriteString("  Model Type: " + m.Config.ModelType + "\n")
	sb.WriteString("  Hidden Size: " + itoa(m.Config.HiddenSize) + "\n")
	sb.WriteString("  Num Layers: " + itoa(m.Config.NumHiddenLayers) + "\n")
	sb.WriteString("  Num Heads: " + itoa(m.Config.NumAttentionHeads) + "\n")
	sb.WriteString("  Vocab Size: " + itoa(m.Config.VocabSize) + "\n")
	sb.WriteString("  Weights: " + m.Weights.String() + "\n")
	return sb.String()
}

func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}
