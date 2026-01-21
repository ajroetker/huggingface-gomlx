// Command hf2go is a CLI tool for inspecting Hugging Face models and generating GoMLX code.
//
// Usage:
//
//	hf2go --model <model_id>              # Inspect a HuggingFace model
//	hf2go --model <model_id> --weights    # Show weight mapping
//	hf2go --local <dir>                   # Inspect local model directory
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/huggingface-gomlx"

	// Import architectures to register them.
	_ "github.com/gomlx/huggingface-gomlx/architectures/bert"
	_ "github.com/gomlx/huggingface-gomlx/architectures/deberta"
	_ "github.com/gomlx/huggingface-gomlx/architectures/llama"
)

func main() {
	modelID := flag.String("model", "", "HuggingFace model ID (e.g., bert-base-uncased)")
	localDir := flag.String("local", "", "Local model directory")
	showWeights := flag.Bool("weights", false, "Show weight mapping")
	listArchs := flag.Bool("list", false, "List supported architectures")
	flag.Parse()

	if *listArchs {
		fmt.Println("Supported architectures:")
		archs := models.ListArchitectures()
		sort.Strings(archs)
		for _, arch := range archs {
			fmt.Printf("  - %s\n", arch)
		}
		return
	}

	if *modelID == "" && *localDir == "" {
		fmt.Fprintln(os.Stderr, "Error: --model or --local is required")
		flag.Usage()
		os.Exit(1)
	}

	var model *models.Model
	var err error

	if *localDir != "" {
		fmt.Printf("Loading model from local directory: %s\n", *localDir)
		model, err = models.NewFromLocal(*localDir)
	} else {
		fmt.Printf("Loading model from HuggingFace: %s\n", *modelID)
		repo := hub.New(*modelID)
		model, err = models.New(repo)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}

	// Print model summary.
	fmt.Println()
	fmt.Println(model.Summary())

	// Print config details.
	cfg := model.Config
	fmt.Println("Configuration:")
	fmt.Printf("  model_type: %s\n", cfg.ModelType)
	fmt.Printf("  vocab_size: %d\n", cfg.VocabSize)
	fmt.Printf("  hidden_size: %d\n", cfg.HiddenSize)
	fmt.Printf("  num_hidden_layers: %d\n", cfg.NumHiddenLayers)
	fmt.Printf("  num_attention_heads: %d\n", cfg.NumAttentionHeads)
	fmt.Printf("  intermediate_size: %d\n", cfg.IntermediateSize)
	fmt.Printf("  hidden_act: %s\n", cfg.HiddenAct)
	fmt.Printf("  layer_norm_eps: %e\n", cfg.LayerNormEps)

	// Print architecture-specific fields if present.
	if posAttType, ok := cfg.GetStringSlice("pos_att_type"); ok {
		fmt.Printf("  pos_att_type: %v\n", posAttType)
	}
	if normRelEbd, ok := cfg.GetStringSlice("norm_rel_ebd"); ok {
		fmt.Printf("  norm_rel_ebd: %v\n", normRelEbd)
	}
	if shareAttKey, ok := cfg.GetBool("share_att_key"); ok {
		fmt.Printf("  share_att_key: %v\n", shareAttKey)
	}
	if ropeTheta, ok := cfg.GetFloat("rope_theta"); ok {
		fmt.Printf("  rope_theta: %f\n", ropeTheta)
	}
	if rmsNormEps, ok := cfg.GetFloat("rms_norm_eps"); ok {
		fmt.Printf("  rms_norm_eps: %e\n", rmsNormEps)
	}

	// Print weight mapping if requested.
	if *showWeights {
		fmt.Println()
		fmt.Println("Weight Mapping (safetensors -> GoMLX context):")
		mapping := model.WeightMapping()

		// Sort by safetensors key.
		keys := make([]string, 0, len(mapping))
		for k := range mapping {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, k := range keys {
			fmt.Printf("  %s\n    -> %s\n", k, mapping[k])
		}

		// Print actual weights in safetensors file.
		fmt.Println()
		fmt.Println("Weights in safetensors file:")
		tensorNames := model.Weights.ListTensorNames()
		sort.Strings(tensorNames)
		for _, name := range tensorNames {
			meta, err := model.Weights.GetTensorMetadata(name)
			if err != nil {
				fmt.Printf("  %s: (error: %v)\n", name, err)
				continue
			}
			fmt.Printf("  %s: %s %v\n", name, meta.Dtype, meta.Shape)
		}
	}
}
