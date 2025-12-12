#!/bin/bash

# Usage: ./cosmos_reason1.sh [WORKSPACE_DIR] [VENV_DIR]

# Parse arguments
# The virtual environment of CosmosReason1 reward can be the same as the base Cosmos-RL-Reward virtual environment. So no operations are needed here and no new virtual environment is created.
# The model CosmosReason1 reward uses is from huggingface without need for local download or installation. So no operations are needed here either.

WORKSPACE_DIR="${1:-/workspace}"
VENV_DIR="${2:-$HOME/cosmos_reason1}"

exit 0