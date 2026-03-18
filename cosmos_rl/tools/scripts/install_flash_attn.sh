#!/bin/bash
# Install flash-attn by preferring a pre-built wheel from Dao-AILab GitHub releases,
# then falling back to building from source if the wheel is not available.
#
# Usage:
#   bash install_flash_attn.sh [version]   # default version: 2.8.3
#   bash install_flash_attn.sh 2.7.4.post1
#   GITHUB_PREFIX="https://ghproxy.com/" bash install_flash_attn.sh 2.8.3   # optional mirror
#
set -euo pipefail

FLASH_VER="${1:-2.8.3}"

GITHUB_PREFIX="${GITHUB_PREFIX:-}"
BASE_URL="${GITHUB_PREFIX}https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_VER}"

# Prefer uv if available
PIP_CMD="pip"
PYTHON_CMD="python"
if command -v uv &> /dev/null; then
    PIP_CMD="uv pip"
    PYTHON_CMD="uv run python"
fi

# Detect Python tags (e.g. cp311)
py_major=$($PYTHON_CMD - <<'EOF'
import sys
print(sys.version_info.major)
EOF
)
py_minor=$($PYTHON_CMD - <<'EOF'
import sys
print(sys.version_info.minor)
EOF
)
py_tag="cp${py_major}${py_minor}"
abi_tag="${py_tag}"

# Detect torch version (major.minor), e.g. 2.6.0 -> 2.6
torch_mm=$($PYTHON_CMD - <<'EOF'
import torch
v = torch.__version__.split("+")[0]
parts = v.split(".")
print(f"{parts[0]}.{parts[1]}")
EOF
)

# Detect CUDA major, e.g. 12 from 12.4
cuda_major=$($PYTHON_CMD - <<'EOF'
import torch
from packaging.version import Version
v = Version(torch.version.cuda)
print(v.base_version.split(".")[0])
EOF
)

cu_tag="cu${cuda_major}"
torch_tag="torch${torch_mm}"
platform_tag="linux_x86_64"
cxx_abi="cxx11abiFALSE"

wheel_name="flash_attn-${FLASH_VER}+${cu_tag}${torch_tag}${cxx_abi}-${py_tag}-${abi_tag}-${platform_tag}.whl"
wheel_url="${BASE_URL}/${wheel_name}"

echo "[install_flash_attn] Version: ${FLASH_VER} | Python: ${py_tag} | Torch: ${torch_tag} | CUDA: ${cu_tag}"
echo "[install_flash_attn] Trying pre-built wheel: ${wheel_url}"

$PIP_CMD uninstall flash-attn -y 2>/dev/null || true

if $PIP_CMD install "${wheel_url}"; then
    echo "[install_flash_attn] Installed pre-built wheel successfully."
    exit 0
fi

echo "[install_flash_attn] Wheel not available or install failed. Building from source..."
$PIP_CMD install "flash-attn==${FLASH_VER}" --no-build-isolation
