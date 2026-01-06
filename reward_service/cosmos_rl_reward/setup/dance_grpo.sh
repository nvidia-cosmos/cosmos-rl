#!/bin/bash

# Usage: ./dance_grpo.sh [WORKSPACE_DIR] [VENV_DIR]
# WORKSPACE_DIR: Directory where DanceGRPO and VideoReward will be cloned (default: /workspace)
# VENV_DIR: Directory for the virtual environment (default: ~/dance_grpo)
# 
# Example:
#   ./dance_grpo.sh /workspace ~/my_venv
#   ./dance_grpo.sh /custom/workspace  # Uses default venv at ~/dance_grpo
#   ./dance_grpo.sh                    # Uses all defaults

# Function to print warning messages
warning() {
    echo "WARNING: $1" >&2
}

# Function to print info messages
info() {
    echo "INFO: $1"
}

# Parse arguments
WORKSPACE_DIR="${1:-/workspace}"
VENV_DIR="${2:-$HOME/dance_grpo}"

info "Starting DanceGRPO setup..."
info "Workspace directory: $WORKSPACE_DIR"
info "Virtual environment: $VENV_DIR"

# Create workspace directory if it doesn't exist
if [ ! -d "$WORKSPACE_DIR" ]; then
    info "Creating workspace directory: $WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR" || warning "Failed to create workspace directory"
fi

# Step 1: Create virtual environment
info "Creating virtual environment at $VENV_DIR..."
pip install -U virtualenv
python -m virtualenv "$VENV_DIR" || warning "Failed to create virtual environment"

# Step 2: Clone DanceGRPO repository if it doesn't exist
DANCE_GRPO_DIR="$WORKSPACE_DIR/DanceGRPO"
if [ ! -d "$DANCE_GRPO_DIR" ]; then
    info "Cloning DanceGRPO repository..."
    git clone https://github.com/XueZeyue/DanceGRPO.git "$DANCE_GRPO_DIR" || warning "Failed to clone DanceGRPO"
else
    info "DanceGRPO already exists at $DANCE_GRPO_DIR"
fi

# Step 3: Install PyTorch
info "Installing PyTorch 2.8.0 and torchvision..."
"$VENV_DIR/bin/pip" install torch==2.8.0 torchvision || warning "Failed to install PyTorch"

# Step 4: Install basic dependencies
info "Installing redis, msgpack, datasets, trl..."
"$VENV_DIR/bin/pip" install redis msgpack datasets trl==0.12 || warning "Failed to install basic dependencies"

# Step 5: Install transformers and related packages
info "Installing transformers and related packages..."
"$VENV_DIR/bin/pip" install transformers==4.46.1 tokenizers==0.20.1 huggingface_hub==0.26.1 accelerate==1.0.1 || warning "Failed to install transformers packages"

# Step 6: Install flash-attn
info "Installing packaging and ninja..."
"$VENV_DIR/bin/pip" install packaging ninja || warning "Failed to install packaging and ninja"

info "Installing flash-attn (this may take a while)..."
"$VENV_DIR/bin/pip" install flash-attn==2.8.1 --no-build-isolation --no-cache-dir || warning "Failed to install flash-attn"

# Step 7: Install DanceGRPO
info "Installing DanceGRPO from source..."
(cd "$DANCE_GRPO_DIR" && "$VENV_DIR/bin/pip" install -r requirements-lint.txt && "$VENV_DIR/bin/pip" install .) || warning "Failed to install DanceGRPO"

# Step 8: Install additional dependencies
info "Installing additional dependencies..."
"$VENV_DIR/bin/pip" install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 huggingface_hub==0.24.0 protobuf==3.20.0 bitsandbytes==0.48.2 || warning "Failed to install additional dependencies"
# Get the path of current file
CURRENT_FILE_PATH="$(realpath "$0")"
COSMOS_RL_REWARD_DIR="$(dirname "$(dirname "$(dirname "$CURRENT_FILE_PATH")")")"
"$VENV_DIR/bin/pip" install "$COSMOS_RL_REWARD_DIR" || warning "Failed to install cosmos_rl_reward"

# Step 9: Initialize git-lfs
info "Initializing git-lfs..."
apt install git-lfs -y
git lfs install || warning "Failed to initialize git-lfs"

# Step 10: Clone VideoReward repository if it doesn't exist
VIDEO_REWARD_DIR="$WORKSPACE_DIR/VideoReward"
if [ ! -d "$VIDEO_REWARD_DIR" ]; then
    info "Cloning VideoReward repository..."
    git clone --depth 1 https://huggingface.co/KwaiVGI/VideoReward "$VIDEO_REWARD_DIR" || warning "Failed to clone VideoReward"
else
    info "VideoReward already exists at $VIDEO_REWARD_DIR"
fi

# Step 11: Clone Qwen2-VL-2B-Instruct repository if it doesn't exist
QWEN_DIR="$WORKSPACE_DIR/Qwen2-VL-2B-Instruct"
if [ ! -d "$QWEN_DIR" ]; then
    info "Cloning Qwen2-VL-2B-Instruct repository..."
    git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct "$QWEN_DIR" || warning "Failed to clone Qwen2-VL-2B-Instruct"
else
    info "Qwen2-VL-2B-Instruct already exists at $QWEN_DIR"
fi

info "DanceGRPO setup completed successfully!"
info "To activate the environment, run: source $VENV_DIR/bin/activate"

exit 0

