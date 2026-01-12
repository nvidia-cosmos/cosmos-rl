#!/bin/bash
#
# Usage:
#   ./pickscore.sh [WORKSPACE_DIR] [VENV_DIR]
#   - WORKSPACE_DIR: Download/cache root if needed (default: /workspace)
#   - VENV_DIR:      Virtualenv directory (default: ~/pickscore)
#
# This installs dependencies for PickScore reward:
# - torch / torchvision
# - transformers / tokenizers
# - pillow
# - redis / msgpack / requests (service-common)
#

WORKSPACE_DIR="${1:-/workspace}"
VENV_DIR="${2:-${1:-$HOME/pickscore}}"

echo "INFO: WORKSPACE_DIR=${WORKSPACE_DIR}"
echo "INFO: VENV_DIR=${VENV_DIR}"

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

# Core deps
"$PIP_BIN" install torch>=2.6.0 torchvision>=0.21.0
"$PIP_BIN" install transformers>=4.40.0 tokenizers>=0.19.0
"$PIP_BIN" install pillow

# Common service deps
"$PIP_BIN" install redis msgpack requests

echo "INFO: PickScore setup completed. Venv python: ${VENV_DIR}/bin/python"

