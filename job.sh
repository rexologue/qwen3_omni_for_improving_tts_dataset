#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-restore_dataset.example.yaml}"
IMAGE_NAME="${IMAGE_NAME:-my-qwen-omni-tool}"
GPU_ID="${GPU_ID:-0}"

echo "Building Docker image..."
docker build . -t "$IMAGE_NAME"

echo "Starting processing on GPU $GPU_ID with config $CONFIG_PATH..."
python3 launcher.py --config "$CONFIG_PATH" --image "$IMAGE_NAME" --gpu "$GPU_ID"
