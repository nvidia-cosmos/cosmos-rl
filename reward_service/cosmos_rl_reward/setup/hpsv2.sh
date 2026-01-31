#!/usr/bin/env bash
set -euo pipefail

# Usage: bash hpsv2.sh <DOWNLOAD_PATH> <VENV_PATH>
# DOWNLOAD_PATH: base directory to store ckpts and sources (service derives paths from this)
# VENV_PATH: path to virtualenv directory whose bin/python will be used (or interpreter)

DOWNLOAD_PATH="${1:-/workspace}"
VENV_PATH="${2:-/root/venv}"

echo "[hpsv2 setup] Using download path: ${DOWNLOAD_PATH}"
echo "[hpsv2 setup] Using venv: ${VENV_PATH}"

# Install wget if not available
if ! command -v wget &> /dev/null; then
    echo "[hpsv2 setup] Installing wget..."
    apt-get update && apt-get install -y wget
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "[hpsv2 setup] Creating virtual environment at $VENV_PATH..."
    python -m pip install -U virtualenv >/dev/null 2>&1 || true
    python -m virtualenv "$VENV_PATH"
fi

PIP_BIN="$VENV_PATH/bin/pip"
if [ ! -x "$PIP_BIN" ]; then
    PIP_BIN="$VENV_PATH/bin/pip3"
fi

if [ ! -x "$PIP_BIN" ]; then
    "$VENV_PATH/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
    PIP_BIN="$VENV_PATH/bin/pip"
fi
if [ ! -x "$PIP_BIN" ]; then
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV_PATH/bin/python" /tmp/get-pip.py
    PIP_BIN="$VENV_PATH/bin/pip"
fi

PYTHON_BIN="${VENV_PATH}/bin/python"

# Resolve output ckpt path under download_path/hpsv2/ckpts
OUT_DIR="${DOWNLOAD_PATH%/}/hpsv2/ckpts"
OUT_FILE="${OUT_DIR}/HPS_v2.1_compressed.pt"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

# Download if file doesn't exist or is empty
if [ ! -s "${OUT_FILE}" ]; then
  rm -f "${OUT_FILE}"
  echo "[hpsv2 setup] Downloading HPSv2 checkpoint..."
  wget "https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt" \
    -O "${OUT_FILE}"
  
  if [ ! -s "${OUT_FILE}" ]; then
    echo "[hpsv2 setup] ERROR: Download failed!"
    exit 1
  fi
else
  echo "[hpsv2 setup] HPSv2 checkpoint already exists."
fi

echo "[hpsv2 setup] Installing Python dependencies into venv..."
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel

"${PYTHON_BIN}" -c "import torch, torchvision" >/dev/null 2>&1 || \
"${PYTHON_BIN}" -m pip install torch torchvision
"${PYTHON_BIN}" -m pip install redis msgpack

"${PYTHON_BIN}" -m pip install "hpsv2x==1.2.0"

echo "[hpsv2 setup] Done."
echo "[hpsv2 setup] Checkpoint path: ${OUT_FILE}"
echo "[hpsv2 setup] Service derives ckpt from download_path → ${OUT_FILE}"


