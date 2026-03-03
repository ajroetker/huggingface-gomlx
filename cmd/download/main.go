// Command download downloads a specific file from a HuggingFace repo.
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gomlx/go-huggingface/hub"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <repo-id> <filename>\n", os.Args[0])
		os.Exit(1)
	}
	repoID := os.Args[1]
	filename := os.Args[2]

	repo := hub.New(repoID)
	if err := repo.DownloadInfo(false); err != nil {
		log.Fatal(err)
	}
	path, err := repo.DownloadFile(filename)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)
}
