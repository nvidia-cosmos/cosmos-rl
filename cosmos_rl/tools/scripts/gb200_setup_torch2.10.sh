#!/bin/bash

# Cosmos RL Setup Script
# Converts the Dockerfile installation steps into a standalone setup script
#
# Usage:
#   ./setup.sh [--build-mode=efa|no-efa] [--extras=all|wfm,vla] [--python-version=3.12]
#
# Examples:
#   ./setup.sh --build-mode=no-efa
#   ./setup.sh --build-mode=efa --extras=all
#   ./setup.sh --build-mode=no-efa --extras=wfm,vla

set -e
# Build upon docker://nvcr.io/nvidia/pytorch:25.06-py3

# Default values
COSMOS_RL_BUILD_MODE="no-efa"
COSMOS_RL_EXTRAS=""
TARGET_ARCH="arm64"
CUDA_VERSION="12.8.1"
GDRCOPY_VERSION="v2.4.4"
EFA_INSTALLER_VERSION="1.42.0"
AWS_OFI_NCCL_VERSION="v1.16.0"
NCCL_VERSION="2.26.2-1+cuda12.8"
FLASH_ATTN_VERSION="2.8.3"
PYTHON_VERSION="3.12"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --build-mode=*)
            COSMOS_RL_BUILD_MODE="${arg#*=}"
            shift
            ;;
        --extras=*)
            COSMOS_RL_EXTRAS="${arg#*=}"
            shift
            ;;
        --arch=*)
            TARGET_ARCH="${arg#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [--build-mode=efa|no-efa] [--extras=all|wfm,vla] [--python-version=3.12] [--arch=arm64]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Cosmos RL Setup Script"
echo "=========================================="
echo "Build mode: $COSMOS_RL_BUILD_MODE"
echo "Extras: ${COSMOS_RL_EXTRAS:-none}"
echo "Python version: $PYTHON_VERSION"
echo "Target arch: $TARGET_ARCH"
echo "=========================================="

HOST_ARCH="$(dpkg --print-architecture)"
if [ "$TARGET_ARCH" != "arm64" ]; then
    echo "Error: this setup script only supports --arch=arm64."
    exit 1
fi
if [ "$HOST_ARCH" != "arm64" ]; then
    echo "Error: host architecture is $HOST_ARCH, but this setup requires arm64."
    exit 1
fi

export TZ=Etc/UTC
export DEBIAN_FRONTEND=noninteractive

# Update and upgrade system
echo "Updating system packages..."
apt-get update -y && apt-get upgrade -y

# Install basic dependencies
echo "Installing basic dependencies..."
apt-get install -y --allow-unauthenticated \
    curl git gpg lsb-release tzdata wget dnsutils \
    software-properties-common

##################################################
# Install Redis
echo "Installing Redis (self-build, MALLOC=libc)..."

# Optional: remove distro/package Redis if present
apt-get update -qq
apt-get remove -qq -y redis-server redis-tools || true

# Build deps
apt-get install -qq -y --no-install-recommends \
  ca-certificates curl git build-essential tcl pkg-config

# Build + install
REDIS_VERSION="8.6.1"
rm -rf /tmp/redis-src
git clone --branch ${REDIS_VERSION} --depth 1 https://github.com/redis/redis.git /tmp/redis-src
cd /tmp/redis-src

make -j"$(nproc)" MALLOC=libc
make install
cd -
rm -rf /tmp/redis-src
ln -sf /usr/local/bin/redis-server /usr/bin/redis-server
ln -sf /usr/local/bin/redis-cli /usr/bin/redis-cli


echo "Upgrading pip, setuptools..."
unset PIP_CONSTRAINT
pip install -U pip setuptools packaging psutil

###################################################
## Install vllm with no torch dependencies

echo "Installing vLLM 0.17.0..."
# pip uninstall -y numpy
# pip install "numpy<2"
pip install vllm==0.17.0


###################################################
## Install flash_attn and related packages
echo "Installing flash_attn and related packages..."

## No need of these since torch related already installed in this container
pip install -U torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129
pip install -U flashinfer-python==0.6.1
MAX_JOBS=64 FLASH_ATTN_CUDA_ARCHS=100 TORCH_CUDA_ARCH_LIST="10.0+PTX" pip install -U flash_attn==2.8.3 --no-build-isolation
pip install -U transformer_engine[pytorch] --no-build-isolation


###################################################
## Install Apex
echo "Installing Apex..."
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex@bf903a2

###################################################
## Install nvshmem, grouped_gemm, and DeepEP for MoE
echo "Installing nvshmem, grouped_gemm, and DeepEP..."
pip install nvidia-nvshmem-cu12==3.4.5
TORCH_CUDA_ARCH_LIST="10.0+PTX" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4 --no-build-isolation

apt-get update && apt-get install -y libibverbs-dev

if [ ! -d /tmp/deepep ]; then
    git clone https://github.com/deepseek-ai/DeepEP.git /tmp/deepep
    cd /tmp/deepep
    TORCH_CUDA_ARCH_LIST="10.0+PTX" python setup.py build
    TORCH_CUDA_ARCH_LIST="10.0+PTX" python setup.py install
    cd -
    rm -rf /tmp/deepep
fi

###################################################
## Install EFA (if requested)
if [ "$COSMOS_RL_BUILD_MODE" = "efa" ]; then
    echo "Installing AWS EFA..."

    # Remove HPCX and MPI to avoid conflicts
    rm -rf /opt/hpcx /usr/local/mpi
    rm -f /etc/ld.so.conf.d/hpcx.conf
    ldconfig

    apt-get remove -y --purge --allow-change-held-packages \
        ibverbs-utils libibverbs-dev libibverbs1 libmlx5-1 2>/dev/null || true

    # Install EFA installer
    cd $HOME
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
    cd aws-efa-installer
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify
    cd ..
    rm -rf aws-efa-installer aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz

    # Install AWS-OFI-NCCL plugin
    apt-get install -y libhwloc-dev
    cd $HOME
    curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz
    tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz
    cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}
    ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws
    make -j $(nproc)
    make install
    cd ..
    rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

    export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/lib:$LD_LIBRARY_PATH
    export PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH
fi

###################################################
## Install Cosmos RL
echo "Installing Cosmos RL..."
apt-get install -y cmake

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


TARGET=$SCRIPT_DIR   # can be a source dir, .whl, or .tar.gz

rm -rf /tmp/pip_local_wheel && mkdir -p /tmp/pip_local_wheel

# 1) Build a wheel from the local target (no deps)
pip wheel --no-deps --no-build-isolation -w /tmp/pip_local_wheel "$TARGET"

# 2) Extract deps from the wheel METADATA, excluding torch* and nvidia*
python3 - <<'PY'
import glob, zipfile, re, os
from packaging.requirements import Requirement
from packaging.markers import default_environment

# Choose extras you actually want:
SELECTED_EXTRAS = {}   # set() means "no extras"

EXCLUDE = re.compile(r"(?i)^(tensordict|torchao)(\s|$|==|>=|<=|~=|!=|\[)")

wheels = sorted(glob.glob("/tmp/pip_local_wheel/*.whl"), key=os.path.getmtime)
wheel = wheels[-1]

with zipfile.ZipFile(wheel) as z:
    meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")][0]
    lines = z.read(meta).decode("utf-8", "replace").splitlines()

reqs = []
for line in lines:
    if not line.startswith("Requires-Dist: "):
        continue
    spec = line[len("Requires-Dist: "):].strip()

    # quick name-based excludes (torch*, nvidia*)
    if EXCLUDE.match(spec):
        continue

    r = Requirement(spec)

    # drop self-references like "cosmos_rl[...]" if present
    if r.name.lower() in {"cosmos-rl", "cosmos_rl"}:
        continue

    if r.marker is None:
        reqs.append(str(r).split(";")[0].strip())
        continue

    marker_str = str(r.marker)
    env = default_environment()

    if "extra" in marker_str:
        # include if marker matches ANY selected extra
        ok = any(r.marker.evaluate({**env, "extra": e}) for e in SELECTED_EXTRAS)
    else:
        # normal markers (python_version, platform, etc.)
        ok = r.marker.evaluate({**env, "extra": ""})

    if ok:
        reqs.append(str(r).split(";")[0].strip())  # drop marker for pip install

open("/tmp/local-requires-filtered.txt", "w").write("\n".join(reqs) + "\n")
print("Wheel:", wheel)
print("Extras selected:", sorted(SELECTED_EXTRAS))
print("Wrote /tmp/local-requires-filtered.txt with", len(reqs), "requirements")
PY

# 3) Install deps (minus excluded)
pip install -r /tmp/local-requires-filtered.txt
pip install tensordict
pip install -U torchao==0.16.10
pip install nvidia-nccl-cu12>=2.26.2

if [ -n "$COSMOS_RL_EXTRAS" ]; then
    pip install "${SCRIPT_DIR}[${COSMOS_RL_EXTRAS}]" --no-deps
else
    pip install "${SCRIPT_DIR}" --no-deps
fi



# Run VLA setup if vla is in extras
if [[ ",$COSMOS_RL_EXTRAS," == *,vla,* ]]; then
    echo "Running VLA setup..."
    bash "${SCRIPT_DIR}/tools/scripts/setup_vla.sh"
fi

pip uninstall -y xformers || true
pip install mamba_ssm==2.3.0 --no-build-isolation
pip install causal_conv1d==1.6.0 --no-build-isolation
apt-get install -y ffmpeg
pip install webdataset
pip install decord
pip install wandb
pip install --no-cache-dir  -U "torchcodec==0.10.*" --index-url https://download.pytorch.org/whl/cu129


###################################################
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Environment variables to set:"
echo "  export LD_LIBRARY_PATH=/opt/gdrcopy/lib:\$LD_LIBRARY_PATH"
echo "  export LIBRARY_PATH=/opt/gdrcopy/lib:\$LIBRARY_PATH"
echo "  export PATH=/opt/gdrcopy/bin:\$PATH"
if [ "$COSMOS_RL_BUILD_MODE" = "efa" ]; then
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/lib:\$LD_LIBRARY_PATH"
    echo "  export PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:\$PATH"
fi
