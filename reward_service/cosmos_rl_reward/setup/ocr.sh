#!/bin/bash
#
# 
# Usage:
#   ./ocr.sh [WORKSPACE_DIR] [VENV_DIR]
#   - WORKSPACE_DIR: Ignored (kept for interface consistency with other scripts)
#   - VENV_DIR:      Directory for the virtual environment (default: ~/ocr)
# Backward-compatible:
#   ./ocr.sh [VENV_DIR]    # when only one arg is provided, it is treated as VENV_DIR
# 

WORKSPACE_DIR="${1:-/workspace}"
VENV_DIR="${2:-${1:-$HOME/ocr}}"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "INFO: Creating virtual environment at $VENV_DIR..."
    python -m pip install -U virtualenv >/dev/null 2>&1 || true
    python -m virtualenv "$VENV_DIR"
fi

# Pick the pip inside venv
PIP_BIN="$VENV_DIR/bin/pip"
if [ ! -x "$PIP_BIN" ]; then
    PIP_BIN="$VENV_DIR/bin/pip3"
fi
# Ensure pip exists inside venv (handle broken/incomplete venv cases)
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


# Install OCR dependencies into the venv
"$PIP_BIN" install torch>=2.6.0 torchvision>=0.21.0
"$PIP_BIN" install redis msgpack requests 
"$PIP_BIN" install paddlepaddle-gpu==2.6.2
"$PIP_BIN" install paddleocr==2.9.1
"$PIP_BIN" install python-Levenshtein
"$PIP_BIN" install opencv-python-headless==4.10.0.84
