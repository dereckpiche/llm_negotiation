#!/usr/bin/env bash
# download_hf_models.sh
# Usage: bash download_hf_models.sh model1 model2 ...

# Exit immediately if a command exits with a non-zero status
set -e

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Install it with:"
    echo "    pip install huggingface_hub"
    exit 1
fi

# Make sure at least one model is passed
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 model1 model2 ..."
    exit 1
fi

# Loop through all provided model names
for model in "$@"; do
    echo "Downloading model: $model"
    hf download "$model" --local-dir ~/.cache/huggingface/hub
    echo "✅ Finished: $model"
done
