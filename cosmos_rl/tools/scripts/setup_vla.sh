#!/bin/bash
# Setup script for VLA environments (Libero / RoboTwin) that depend on
# EGL rendering.  Combines the former setup_vla.sh and setup_nvgl.sh into
# a single, structured script.
#
# Usage:
#   bash setup_vla.sh            # interactive (prompts before nvidia-gl install)
#   AUTO_CONFIRM=1 bash setup_vla.sh   # non-interactive / CI
#
set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
banner(){ echo -e "\n${BLUE}── $* ──${NC}"; }

# ── 1. EGL packages + libnvidia-gl (auto-detect driver version) ───────────────
install_egl_packages() {
    banner "Installing EGL packages"

    # 1a. System EGL / GL libraries
    apt-get update -qq
    apt-get install -y -qq \
        cmake \
        libglvnd0 libgl1 libegl1 \
        libgles2 \
        libglib2.0-dev \
        libglew2.2 \
        libnvidia-egl-wayland1 \
        libnvidia-egl-wayland-dev \
        > /dev/null 2>&1 || true   # some packages may not exist on all distros
    ok "System GL/EGL libraries installed"

    # 1b. Detect NVIDIA driver version
    #     Try multiple methods: nvidia-smi → /proc/driver → libcuda.so filename
    local driver_version=""

    if command -v nvidia-smi &> /dev/null; then
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
        [ -n "$driver_version" ] && ok "Driver version from nvidia-smi: ${driver_version}"
    fi

    if [ -z "$driver_version" ] && [ -f /proc/driver/nvidia/version ]; then
        driver_version=$(head -1 /proc/driver/nvidia/version | sed -n 's/.*NVRM version:.*\s\([0-9]\+\.[0-9.]\+\)\s.*/\1/p')
        [ -n "$driver_version" ] && ok "Driver version from /proc: ${driver_version}"
    fi

    if [ -z "$driver_version" ]; then
        # Last resort: parse libcuda.so symlink (e.g. libcuda.so.570.195.03)
        local cuda_lib
        cuda_lib=$(ldconfig -p 2>/dev/null | grep 'libcuda.so ' | head -1 | awk '{print $NF}')
        if [ -n "$cuda_lib" ]; then
            driver_version=$(basename "$cuda_lib" | sed 's/libcuda\.so\.//')
            [ -n "$driver_version" ] && ok "Driver version from libcuda.so: ${driver_version}"
        fi
    fi

    if [ -z "$driver_version" ]; then
        err "Could not determine NVIDIA driver version (tried nvidia-smi, /proc, libcuda.so)"
        return 1
    fi

    local major_version
    major_version=$(echo "$driver_version" | cut -d'.' -f1)
    ok "Detected NVIDIA driver ${driver_version} (major: ${major_version})"

    # 1c. Resolve libnvidia-gl package + version
    local pkg="libnvidia-gl-${major_version}"

    if ! apt-cache show "$pkg" &> /dev/null 2>&1; then
        err "Package ${pkg} not found in apt sources"
        warn "You may need to add the NVIDIA apt repository or run apt update"
        return 1
    fi

    local install_version
    local exact
    exact=$(apt-cache madison "$pkg" \
        | grep -E "^\s*${pkg}\s*\|\s*${driver_version}" \
        | head -1 | awk -F'|' '{print $2}' | tr -d '[:space:]')

    if [ -n "$exact" ]; then
        install_version="${pkg}=${exact}"
        ok "Exact version match: ${exact}"
    else
        local latest
        latest=$(apt-cache madison "$pkg" | head -1 | awk -F'|' '{print $2}' | tr -d '[:space:]')
        if [ -z "$latest" ]; then
            err "Could not determine installable version for ${pkg}"
            return 1
        fi
        install_version="${pkg}=${latest}"
        warn "No exact match for ${driver_version}; using latest ${major_version}-series: ${latest}"
    fi

    # 1d. Check if already installed at the right version
    if dpkg -l 2>/dev/null | grep -q "^ii.*${pkg}"; then
        local current
        current=$(dpkg -l | grep "^ii.*${pkg}" | awk '{print $3}')
        if [ "${pkg}=${current}" = "$install_version" ]; then
            ok "${install_version} already installed – nothing to do"
        else
            warn "Installed: ${current} → target: ${install_version}"
        fi
    fi

    # 1e. Install libnvidia-gl (with optional confirmation)
    info "Will install: ${install_version}"
    if [ "${AUTO_CONFIRM:-0}" != "1" ]; then
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

    # 1f. EGL ICD vendor configuration
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

install_egl_packages

# ── 2. Libero first-run setup ─────────────────────────────────────────────────
banner "Configuring Libero"

mkdir -p ~/.libero
touch ~/.libero/config.yaml
ROBOSUITE_PATH=$(uv pip show robosuite | grep 'Location' | awk '{print $2}')/robosuite
uv run python $ROBOSUITE_PATH/scripts/setup_macros.py
uv run python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
ok "Libero default path configured"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
ok "VLA environment setup complete"

