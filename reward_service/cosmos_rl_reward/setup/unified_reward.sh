#!/bin/bash

# Usage: ./unified_reward.sh [WORKSPACE_DIR] [VENV_DIR]

# Parse arguments
# The virtual environment of UnifiedReward can be the same as the base Cosmos-RL-Reward virtual environment.
# It only requires a openai lib to send request to the remote endpoint.

WORKSPACE_DIR="${1:-/workspace}"
VENV_DIR="${2:-$HOME/unified_reward}"

echo "[unified_reward setup] Installing Python dependencies..."
python -m pip install openai
echo "[unified_reward setup] Done."