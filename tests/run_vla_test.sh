#!/bin/bash
# Quick test script for VLA single rollout

# Default values
MODEL_PATH=${1:-"Haozhan72/Openvla-oft-SFT-libero10-traj1"}
TASK_SUITE=${2:-"LIBERO_10"}
TASK_ID=${3:-0}
DEVICE=${4:-"cuda:0"}

echo "================================"
echo "VLA Single Rollout Test"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Task: $TASK_SUITE #$TASK_ID"
echo "Device: $DEVICE"
echo "================================"
echo ""

# Set environment variable
export ROBOT_PLATFORM=libero
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=../LIBERO

# Run test
cd /home/lianghao/workspace/cosmos-rl
python tests/test_vla_single_rollout.py \
    --model_path "$MODEL_PATH" \
    --task_suite "$TASK_SUITE" \
    --task_id "$TASK_ID" \
    --device "$DEVICE" \
    --vla_type "openvla-oft"

echo ""
echo "Video saved to: ./rollouts/test/"

