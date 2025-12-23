#!/bin/bash

# Configuration
IMAGE_NAME="my-qwen-omni-tool"
CONTAINER_NAME="qwen-omni-runner"
GPU_ID="4"

# Paths
BASE_DIR="$(pwd)"
HOST_DATASET_DIR="/home/user5/audio/.data/11labs_dump_from_20251207_to_20251212"
HOST_OUTPUT_CSV="$BASE_DIR/result.csv"
HOST_CONFIG_PATH="$BASE_DIR/tag_dataset.docker.yaml"
HOST_EXAMPLE_JSON="$BASE_DIR/examples1.json"
HOST_EXAMPLES_DIR="/home/user5/audio/audio_stuff/few-shot"
HOST_MODEL="/home/user5/.models/qwen3-omni-awq"

# Internal Paths
CONTAINER_DATA_IN="/data/input"
CONTAINER_OUT_CSV="/data/output.csv"
CONTAINER_CONFIG="/app/config.yaml"
CONTAINER_MODEL="/model"
CONTAINER_EXAMPLE_JSON="/app/examples.json"

# Pre-flight
touch "$HOST_OUTPUT_CSV"

# REBUILD IS MANDATORY FOR THE FIX
echo "Building Docker image (Baking models)..."
docker build . -t $IMAGE_NAME

echo "Starting processing on GPU $GPU_ID..."

docker run --rm \
    --gpus "device=$GPU_ID" \
    --shm-size 16g \
    --name $CONTAINER_NAME \
    --user $(id -u):$(id -g) \
    -e USER=omni \
    -e HOME=/home/omni \
    -v "$HOST_MODEL":$CONTAINER_MODEL \
    -v "$HOST_EXAMPLES_DIR":$HOST_EXAMPLES_DIR \
    -v "$HOST_EXAMPLE_JSON":$CONTAINER_EXAMPLE_JSON \
    -v "$HOST_DATASET_DIR":$CONTAINER_DATA_IN \
    -v "$HOST_OUTPUT_CSV":$CONTAINER_OUT_CSV \
    -v "$HOST_CONFIG_PATH":$CONTAINER_CONFIG \
    $IMAGE_NAME \
    python3 tag_dataset.py --config $CONTAINER_CONFIG