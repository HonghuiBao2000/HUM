#!/bin/bash

# Parameters
OUTPUT_PATH=${1:-./ckp/hum_v1.pth}
DATASET=${2:-m_IOATBC}
LR=${3:-5e-5}
CONFIG=${4:-configs/train_HUM.yaml}
NUM_GPUS=${5:-4}
PORT=${6:-29500}

echo "========================================================================"
echo "Training HUM v1"
echo "========================================================================"
echo "Output: $OUTPUT_PATH"
echo "Dataset: $DATASET"
echo "Learning Rate: $LR"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Port: $PORT"
echo "========================================================================"
echo ""

# Run training
torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT run_hum.py \
    --output "$OUTPUT_PATH" \
    --dataset "$DATASET" \
    --lr "$LR" \
    --config "$CONFIG"

echo ""
echo "========================================================================"
echo "Training Completed!"
echo "========================================================================"
echo "Model saved to: $OUTPUT_PATH"
echo "========================================================================"


