#!/usr/bin/env bash
set -euo pipefail

# Usage: bash hpsv3.sh <DOWNLOAD_PATH> <VENV_PATH>
# DOWNLOAD_PATH: base directory to store ckpts and sources (service can derive paths from this)
# VENV_PATH: path to virtualenv directory whose bin/python will be used (or interpreter)
#
# This script:
# - creates the venv if missing
# - installs hpsv3 (PyPI) and minimal runtime deps
# - downloads MizzenAI/HPSv3 weight file to <DOWNLOAD_PATH>/hpsv3/HPSv3.safetensors (optional; inferencer can also download)

DOWNLOAD_PATH="${1:-/workspace}"
VENV_PATH="${2:-/root/hpsv3}"

echo "[hpsv3 setup] Using download path: ${DOWNLOAD_PATH}"
echo "[hpsv3 setup] Using venv: ${VENV_PATH}"

if [ ! -d "$VENV_PATH" ]; then
  echo "[hpsv3 setup] Creating virtual environment at $VENV_PATH..."
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

echo "[hpsv3 setup] Installing Python dependencies into venv..."
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel

"${PYTHON_BIN}" -c "import torch, torchvision" >/dev/null 2>&1 || \
  "${PYTHON_BIN}" -m pip install torch torchvision
"${PYTHON_BIN}" -m pip install redis msgpack requests pillow

"${PYTHON_BIN}" -m pip install "hpsv3>=1.0.0" safetensors huggingface_hub qwen-vl-utils matplotlib tensorboard

OUT_DIR="${DOWNLOAD_PATH%/}/hpsv3"
OUT_FILE="${OUT_DIR}/HPSv3.safetensors"
mkdir -p "${OUT_DIR}"

echo "[hpsv3 setup] Downloading checkpoint to ${OUT_FILE} (best effort)..."
"${PYTHON_BIN}" - <<PY
from huggingface_hub import hf_hub_download

out_dir = r"${OUT_DIR}"
path = hf_hub_download(
    "MizzenAI/HPSv3",
    "HPSv3.safetensors",
    repo_type="model",
    local_dir=out_dir,
    local_dir_use_symlinks=False,
)
print("[hpsv3 setup] downloaded:", path)
PY

echo "[hpsv3 setup] Done."
echo "[hpsv3 setup] Checkpoint path: ${OUT_FILE}"

