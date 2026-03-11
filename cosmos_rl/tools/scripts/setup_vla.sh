#!/bin/bash
# Setup script for VLA environments (Libero / RoboTwin) that depend on
# EGL rendering, Vulkan, and optional RoboTwin Python deps.
#
# Groups: setup_egl, setup_vulkan_and_robotwin_apt, detect_nvcc, detect_driver, setup_libero, setup_robotwin, setup_hosts_fqdn
# Sets ROBOTWIN_PATH to the RoboTwin repo (clone at same level as cosmos-rl, branch RLinf_support).
#
# Usage:
#   bash setup_vla.sh            # interactive (prompts before nvidia-gl install)
#   AUTO_CONFIRM=1 bash setup_vla.sh   # non-interactive / CI
#
set -euo pipefail

GITHUB_PREFIX="${GITHUB_PREFIX:-}"
ROBOTWIN_REPO_URL="${ROBOTWIN_REPO_URL:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_BRANCH="${ROBOTWIN_BRANCH:-RLinf_support}"

# Cosmos-rl repo root (parent of tools/scripts containing this script)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]:-$0}")"
COSMOS_RL_ROOT="$(cd "$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")")" && pwd)"

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
banner(){ echo -e "\n${BLUE}── $* ──${NC}"; }

# ── setup_hosts_fqdn: add 127.0.0.1 <fqdn> to /etc/hosts if missing ─────────────
setup_hosts_fqdn() {
    banner "Checking /etc/hosts for FQDN"
    local fqdn
    fqdn=$(host "$(hostname -i)" 2>/dev/null | awk '{print $5}' | sed 's/\.$//')
    if [ -z "$fqdn" ]; then
        warn "Could not resolve FQDN (host \$(hostname -i) failed); skip hosts entry"
        return 0
    fi
    if grep -qF "127.0.0.1 $fqdn" /etc/hosts 2>/dev/null; then
        ok "127.0.0.1 $fqdn already in /etc/hosts"
        return 0
    fi
    echo "127.0.0.1 $fqdn" | tee -a /etc/hosts
    ok "Added 127.0.0.1 $fqdn to /etc/hosts"
}

# ── detect_driver: NVIDIA driver version (for libnvidia-gl) ─────────────────────
detect_driver() {
    banner "Detecting NVIDIA driver"
    DRIVER_VERSION=""
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
        [ -n "$DRIVER_VERSION" ] && ok "Driver version from nvidia-smi: ${DRIVER_VERSION}"
    fi
    if [ -z "$DRIVER_VERSION" ] && [ -f /proc/driver/nvidia/version ]; then
        DRIVER_VERSION=$(head -1 /proc/driver/nvidia/version | sed -n 's/.*NVRM version:.*\s\([0-9]\+\.[0-9.]\+\)\s.*/\1/p')
        [ -n "$DRIVER_VERSION" ] && ok "Driver version from /proc: ${DRIVER_VERSION}"
    fi
    if [ -z "$DRIVER_VERSION" ]; then
        local cuda_lib
        cuda_lib=$(ldconfig -p 2>/dev/null | grep 'libcuda.so ' | head -1 | awk '{print $NF}')
        if [ -n "$cuda_lib" ]; then
            DRIVER_VERSION=$(basename "$cuda_lib" | sed 's/libcuda\.so\.//')
            [ -n "$DRIVER_VERSION" ] && ok "Driver version from libcuda.so: ${DRIVER_VERSION}"
        fi
    fi
    if [ -z "$DRIVER_VERSION" ]; then
        err "Could not determine NVIDIA driver version (tried nvidia-smi, /proc, libcuda.so)"
        return 1
    fi
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
    ok "Detected NVIDIA driver ${DRIVER_VERSION} (major: ${DRIVER_MAJOR})"
}

# ── setup_egl: EGL packages + libnvidia-gl + EGL ICD ───────────────────────────
setup_egl() {
    banner "Installing EGL packages"
    apt-get update -qq
    apt-get install -y -qq \
        cmake \
        libglvnd0 libgl1 libegl1 \
        libgles2 \
        libglib2.0-dev \
        libglew2.2 \
        libnvidia-egl-wayland1 \
        libnvidia-egl-wayland-dev \
        > /dev/null 2>&1 || true
    ok "System GL/EGL libraries installed"

    detect_driver || return 1

    local pkg="libnvidia-gl-${DRIVER_MAJOR}"
    if ! apt-cache show "$pkg" &> /dev/null; then
        err "Package ${pkg} not found in apt sources"
        warn "You may need to add the NVIDIA apt repository or run apt update"
        return 1
    fi

    local install_version
    local exact
    exact=$(apt-cache madison "$pkg" \
        | grep -E "^\s*${pkg}\s*\|\s*${DRIVER_VERSION}" \
        | head -1 | awk -F'|' '{print $2}' | tr -d '[:space:]')
    if [ -n "$exact" ]; then
        install_version="${pkg}=${exact}"
        ok "Exact version match: ${exact}"
    else
        local latest
        latest=$(apt-cache madison "$pkg" | head -1 | awk -F'|' '{print $2}' | tr -d '[:space:]')
        [ -z "$latest" ] && { err "Could not determine installable version for ${pkg}"; return 1; }
        install_version="${pkg}=${latest}"
        warn "No exact match for ${DRIVER_VERSION}; using latest ${DRIVER_MAJOR}-series: ${latest}"
    fi

    if dpkg -l 2>/dev/null | grep -q "^ii.*${pkg}"; then
        local current
        current=$(dpkg -l | grep "^ii.*${pkg}" | awk '{print $3}')
        if [ "${pkg}=${current}" = "$install_version" ]; then
            ok "${install_version} already installed – nothing to do"
        else
            warn "Installed: ${current} → target: ${install_version}"
        fi
    fi

    info "Will install: ${install_version}"
    if [ "${AUTO_CONFIRM:-0}" != "1" ] && [ -t 0 ]; then
        read -rp "Continue? [y/N] " ans
        if [[ ! "$ans" =~ ^[Yy]$ ]]; then
            warn "Skipped libnvidia-gl installation"
            return 0
        fi
    fi

    if apt-get install -y --allow-downgrades "$install_version" > /dev/null; then
        ok "${install_version} installed successfully"
    else
        err "apt-get install failed for ${install_version}"
        return 1
    fi

    banner "Configuring EGL ICD"
    mkdir -p /usr/share/glvnd/egl_vendor.d/
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EJSON'
{
   "file_format_version" : "1.0.0",
   "ICD" : {
      "library_path" : "libEGL_nvidia.so.0"
   }
}
EJSON
    cat > /usr/share/glvnd/egl_vendor.d/50_mesa.json << 'EJSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_mesa.so.0"
    }
}
EJSON
    ldconfig
    ok "EGL vendor configs written (NVIDIA primary, Mesa fallback)"
}

# ── setup_vulkan_and_sapien: Vulkan + Sapien build/FFmpeg deps ───────────
setup_vulkan_and_sapien() {
    banner "Installing Vulkan and RoboTwin system dependencies"
    apt-get update -qq

    # Vulkan runtime and tools
    apt-get install -y -qq \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools
    ok "Vulkan packages installed"

    # Vulkan ICD/layer config so the loader finds the NVIDIA driver (e.g. in Docker/minimal)
    mkdir -p /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d
    if [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
        cat > /etc/vulkan/icd.d/nvidia_icd.json << 'EJSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libGLX_nvidia.so.0",
        "api_version" : "1.3.194"
    }
}
EJSON
        ok "Vulkan NVIDIA ICD written"
    fi
    if [ ! -f /etc/vulkan/implicit_layer.d/nvidia_layers.json ]; then
        cat > /etc/vulkan/implicit_layer.d/nvidia_layers.json << 'EJSON'
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
EJSON
        ok "Vulkan NVIDIA Optimus layer written (laptops)"
    fi

    # RoboTwin: build-essential, python3-dev, FFmpeg libs
    apt-get install -y -qq \
        cmake \
        build-essential \
        python3-dev \
        unzip \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev
    ok "RoboTwin apt dependencies installed"
}

# ── detect_nvcc: locate nvcc and set TORCH_CUDA_ARCH_LIST ─────────────────────
detect_nvcc() {
    banner "Detecting nvcc and CUDA arch list"
    NVCC_EXE=""
    if [ -x "$(command -v nvcc)" ]; then
        NVCC_EXE=$(command -v nvcc)
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        NVCC_EXE="/usr/local/cuda/bin/nvcc"
    else
        err "nvcc not found. RoboTwin Python env may fail to build extensions."
        return 1
    fi
    ok "nvcc: ${NVCC_EXE}"
    local cuda_major cuda_minor
    cuda_major=$("$NVCC_EXE" --version | grep 'Cuda compilation tools' | awk '{print $5}' | tr -d ',' | awk -F '.' '{print $1}')
    cuda_minor=$("$NVCC_EXE" --version | grep 'Cuda compilation tools' | awk '{print $5}' | tr -d ',' | awk -F '.' '{print $2}')
    if [ "$cuda_major" -gt 12 ] || { [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 8 ]; }; then
        export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0;10.0"
        ok "TORCH_CUDA_ARCH_LIST=7.0;8.0;9.0;10.0 (CUDA 12.8+ / Blackwell)"
    else
        export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
        ok "TORCH_CUDA_ARCH_LIST=7.0;8.0;9.0"
    fi
}

# ── setup_libero: Libero first-run config + robosuite private macros ──────────────
setup_libero() {
    banner "Configuring Libero"
    mkdir -p ~/.libero
    touch ~/.libero/config.yaml

    # Resolve robosuite path from the same Python we use (active venv)
    ROBOSUITE_PATH=$($PYTHON_CMD -c "import robosuite; print(robosuite.__path__[0])" 2>/dev/null) || true
    if [ -z "$ROBOSUITE_PATH" ] || [ ! -d "$ROBOSUITE_PATH" ]; then
        warn "robosuite not found; skip Libero/robosuite config or install VLA deps first"
        return 0
    fi

    # Setup robosuite private macro file (avoids "[robosuite WARNING] No private macro file found!")
    # Only run when macros_private.py is missing to avoid interactive "overwrite? (y/n)" prompt
    if [ -f "$ROBOSUITE_PATH/macros_private.py" ]; then
        ok "Robosuite macros_private.py already present"
    elif [ -f "$ROBOSUITE_PATH/scripts/setup_macros.py" ]; then
        $PYTHON_CMD "$ROBOSUITE_PATH/scripts/setup_macros.py"
        ok "Robosuite macros_private.py configured"
    elif [ -f "$ROBOSUITE_PATH/macros.py" ]; then
        cp "$ROBOSUITE_PATH/macros.py" "$ROBOSUITE_PATH/macros_private.py"
        ok "Robosuite macros_private.py created (copied from macros.py)"
    fi

    # Libero default path (may print [Warning] about changing default path – expected)
    if $PYTHON_CMD -c "from libero.libero import set_libero_default_path" 2>/dev/null; then
        $PYTHON_CMD -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
        ok "Libero default path configured"
    else
        warn "libero not found; run after installing VLA deps if needed"
    fi
}

# ── setup_robotwin: RoboTwin Python deps + sapien/mplib patches ─────────────────
setup_robotwin() {
    banner "Setting up RoboTwin Python environment"
    detect_nvcc || return 1

    $PIP_CMD install mplib==0.2.1 gymnasium==0.29.1 av open3d zarr openai sapien==3.0.0b1
    $PIP_CMD install git+${GITHUB_PREFIX}https://github.com/facebookresearch/pytorch3d.git@v0.7.9 --no-build-isolation
    $PIP_CMD install warp-lang==1.11.1
    # $PIP_CMD install git+${GITHUB_PREFIX}https://github.com/NVlabs/curobo.git --no-build-isolation

    SAPIEN_LOCATION=$($PIP_CMD show sapien | grep 'Location' | awk '{print $2}')/sapien
    URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
    ok "Patched sapien wrapper/urdf_loader.py (encoding + .srdf)"

    MPLIB_LOCATION=$($PIP_CMD show mplib | grep 'Location' | awk '{print $2}')/mplib
    PLANNER=$MPLIB_LOCATION/planner.py
    sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
    ok "Patched mplib planner.py (collide check)"
    ok "RoboTwin Python environment ready"

    # RoboTwin repo: same directory level as cosmos-rl, branch RLinf_support
    ROBOTWIN_PATH="$(dirname "$COSMOS_RL_ROOT")/RoboTwin"
    if [ -d "$ROBOTWIN_PATH/.git" ]; then
        info "RoboTwin repo at ${ROBOTWIN_PATH}; fetching and checking out ${ROBOTWIN_BRANCH}"
        git -C "$ROBOTWIN_PATH" fetch origin
        git -C "$ROBOTWIN_PATH" checkout "$ROBOTWIN_BRANCH" 2>/dev/null || git -C "$ROBOTWIN_PATH" checkout -b "$ROBOTWIN_BRANCH" "origin/${ROBOTWIN_BRANCH}"
    else
        info "Cloning RoboTwin into ${ROBOTWIN_PATH} (branch ${ROBOTWIN_BRANCH})"
        git clone -b "$ROBOTWIN_BRANCH" "${GITHUB_PREFIX}${ROBOTWIN_REPO_URL}" "$ROBOTWIN_PATH"
    fi
    export ROBOTWIN_PATH
    echo "export ROBOTWIN_PATH=\"$ROBOTWIN_PATH\"" > "$COSMOS_RL_ROOT/.robotwin_env"
    ok "ROBOTWIN_PATH=${ROBOTWIN_PATH}"
    if [ -d "$ROBOTWIN_PATH/assets" ]; then
        ok "RoboTwin assets folder already present; skip download"
    elif command -v uv &> /dev/null && [ -d "$COSMOS_RL_ROOT/.venv" ]; then
        uv run --directory "$COSMOS_RL_ROOT" bash -c "cd \"$ROBOTWIN_PATH\" && bash script/_download_assets.sh"
    elif [ -f "$COSMOS_RL_ROOT/.venv/bin/activate" ]; then
        source "$COSMOS_RL_ROOT/.venv/bin/activate" && cd "$ROBOTWIN_PATH" && bash script/_download_assets.sh
    else
        warn "No venv at $COSMOS_RL_ROOT/.venv and uv not in PATH; skip asset download. Run manually: uv run --directory $COSMOS_RL_ROOT bash -c \"cd \\\"\$ROBOTWIN_PATH\\\" && bash script/_download_assets.sh\""
    fi
}

# ── Main: pip/python commands then run all groups ──────────────────────────────
PIP_CMD="pip"
PYTHON_CMD="python"
if command -v uv &> /dev/null; then
    PIP_CMD="uv pip"
    PYTHON_CMD="uv run python"
fi

#setup_hosts_fqdn
setup_egl
setup_vulkan_and_sapien

banner "Python/pip"
ok "Using PIP_CMD=${PIP_CMD} PYTHON_CMD=${PYTHON_CMD}"

setup_libero
setup_robotwin

echo ""
ok "VLA environment setup complete"
echo ""
info "To set ROBOTWIN_PATH in your shell after this script exits, run:"
echo "  source $COSMOS_RL_ROOT/.robotwin_env"
echo ""
