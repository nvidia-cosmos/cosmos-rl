#!/usr/bin/env bash
set -euo pipefail

# Usage: bash gen_eval.sh <DOWNLOAD_PATH> <VENV_PATH>
# DOWNLOAD_PATH: base directory to store reward_ckpts and sources
# VENV_PATH: path to virtualenv directory whose bin/python will be used

DOWNLOAD_PATH="${1:-/workspace}"
VENV_PATH="${2:-/root/venv}"

echo "[gen_eval setup] Using download path: ${DOWNLOAD_PATH}"
echo "[gen_eval setup] Using venv: ${VENV_PATH}"

if [ ! -d "$VENV_PATH" ]; then
    echo "[gen_eval setup] Creating virtual environment at $VENV_PATH..."
    python -m pip install -U virtualenv >/dev/null 2>&1 || true
    python -m virtualenv --system-site-packages "$VENV_PATH"
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

mkdir -p "${DOWNLOAD_PATH}/reward_ckpts"
cd "${DOWNLOAD_PATH}/reward_ckpts"

CKPT_NAME="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
if [ ! -f "${CKPT_NAME}" ]; then
  echo "[gen_eval setup] Downloading detector ckpt..."
  wget -q --show-progress \
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth" \
    -O "${CKPT_NAME}"
else
  echo "[gen_eval setup] Detector ckpt already exists: ${CKPT_NAME}"
fi

# Download only the object names file from DiffusionNFT assets
OBJ_FILE="object_names.txt"
if [ ! -f "${OBJ_FILE}" ]; then
  echo "[gen_eval setup] Downloading ${OBJ_FILE}..."
  wget -q --show-progress \
    "https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/flow_grpo/assets/object_names.txt" \
    -O "${OBJ_FILE}" || true
  if [ -f "${OBJ_FILE}" ]; then
    echo "[gen_eval setup] ${OBJ_FILE} downloaded."
  else
    echo "[gen_eval setup] WARNING: Failed to download ${OBJ_FILE}. The code will fallback to detector metadata."
  fi
fi

echo "[gen_eval setup] Installing Python dependencies into venv..."
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel

"${PYTHON_BIN}" -c "import torch, torchvision" >/dev/null 2>&1 || \
"${PYTHON_BIN}" -m pip install torch==2.6.0 torchvision==0.21.0
"${PYTHON_BIN}" -m pip install redis msgpack

"${PYTHON_BIN}" -m pip install -U openmim
"${PYTHON_BIN}" -m pip install -U pip setuptools # openmim will downgrade setuptools to make mim install failed weith Python3.12
"${PYTHON_BIN}" -m mim install mmengine open-clip-torch clip-benchmark

cd "${DOWNLOAD_PATH}"

# Install MMCV (1.x) with CUDA ops
"${PYTHON_BIN}" -m pip uninstall -y mmcv mmcv-full mmcv-lite 2>/dev/null || true
if [ ! -d "${DOWNLOAD_PATH}/mmcv" ]; then
  git clone https://github.com/open-mmlab/mmcv.git
fi
cd mmcv
git fetch --all
git checkout v1.7.2
rm -rf build dist *.egg-info
"${PYTHON_BIN}" -m pip install ninja
MAX_JOBS=$(nproc) MMCV_WITH_OPS=1 FORCE_CUDA=1 "${PYTHON_BIN}" setup.py build_ext --inplace
MAX_JOBS=$(nproc) MMCV_WITH_OPS=1 FORCE_CUDA=1 "${PYTHON_BIN}" setup.py develop
cd "${DOWNLOAD_PATH}"

# Install MMDetection (2.x)
if [ ! -d "${DOWNLOAD_PATH}/mmdetection" ]; then
  git clone https://github.com/open-mmlab/mmdetection.git
fi
cd mmdetection
git fetch --all
git checkout 2.x
"${PYTHON_BIN}" -m pip install . --no-build-isolation
cd "${DOWNLOAD_PATH}"


"${PYTHON_BIN}" -m pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
"${PYTHON_BIN}" -m pip install opencv-python-headless

echo "[gen_eval setup] Done."
echo "[gen_eval setup] Files placed under: ${DOWNLOAD_PATH}"
echo "[gen_eval setup] - ckpt:        ${DOWNLOAD_PATH}/reward_ckpts/${CKPT_NAME}"
echo "[gen_eval setup] - class names: ${DOWNLOAD_PATH}/reward_ckpts/object_names.txt"
echo "[gen_eval setup] - config:      ${DOWNLOAD_PATH}/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
echo "[gen_eval setup] Service will use download_path from rewards.toml."


