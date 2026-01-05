#!/bin/bash
#
# Usage: ./image_reward.sh [WORKSPACE_DIR] [VENV_DIR]
# WORKSPACE_DIR: Directory to place/download resources if needed (default: /workspace)
# VENV_DIR:      Directory for the virtual environment (default: ~/image_reward)
#

WORKSPACE_DIR="${1:-/workspace}"
VENV_DIR="${2:-$HOME/image_reward}"

echo "INFO: WORKSPACE_DIR=${WORKSPACE_DIR}"
echo "INFO: VENV_DIR=${VENV_DIR}"

if [ ! -d "$VENV_DIR" ]; then
    echo "INFO: Creating virtual environment at $VENV_DIR..."
    python -m pip install -U virtualenv >/dev/null 2>&1 || true
    python -m virtualenv "$VENV_DIR"
fi

PIP_BIN="$VENV_DIR/bin/pip"
if [ ! -x "$PIP_BIN" ]; then
    PIP_BIN="$VENV_DIR/bin/pip3"
fi

if [ ! -x "$PIP_BIN" ]; then
    "$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
    PIP_BIN="$VENV_DIR/bin/pip"
fi
if [ ! -x "$PIP_BIN" ]; then
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV_DIR/bin/python" /tmp/get-pip.py
    PIP_BIN="$VENV_DIR/bin/pip"
fi

# Base build tools
"$PIP_BIN" install -U pip setuptools wheel

# Ensure torch/torchvision present (avoid downgrading if already satisfied)
"$VENV_DIR/bin/python" -c "import torch, torchvision" >/dev/null 2>&1 || \
"$PIP_BIN" install torch torchvision

echo "INFO: Installing ImageReward dependencies into $VENV_DIR ..."
"$PIP_BIN" install redis msgpack requests diffusers==0.33.1 transformers==4.40.0 tokenizers==0.19.1 peft==0.10.0
"$PIP_BIN" install image-reward
"$PIP_BIN" install git+https://github.com/openai/CLIP.git


echo "INFO: ImageReward setup completed."


