# Usage:
# To build without AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=no-efa .
# To build with AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=efa .
# To build with specific dependency groups:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_EXTRAS=all .
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_EXTRAS=wfm,vla .
# To select the PyTorch dependency profile:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_TORCH_VARIANT=2.8 .
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_TORCH_VARIANT=2.10 .

ARG COSMOS_RL_BUILD_MODE=efa
ARG COSMOS_RL_EXTRAS=""
ARG COSMOS_RL_TORCH_VARIANT=2.8

ARG CUDA_VERSION=12.8.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

ARG GDRCOPY_VERSION=v2.4.4
ARG EFA_INSTALLER_VERSION=1.42.0
ARG AWS_OFI_NCCL_VERSION=v1.16.0
# NCCL version, should be found at https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG FLASH_ATTN_VERSION=2.8.3
ARG PYTHON_VERSION=3.12
ARG COSMOS_RL_TORCH_VARIANT

ENV TZ=Etc/UTC

RUN apt-get update -y && apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    curl git gpg lsb-release tzdata wget
RUN apt-get purge -y cuda-compat-*
RUN apt-get update && apt-get install -y dnsutils

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV PATH=/opt/gdrcopy/bin:$PATH

###################################################
## Install NCCL with specific version
RUN apt-get remove -y --purge --allow-change-held-packages \
    libnccl2 \
    libnccl-dev
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update -y \
    && apt-get install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION}

###################################################
## Install cuDNN
RUN apt-get update -y && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

###################################################
## Install redis
# Download and add Redis GPG key, Redis APT repository
RUN curl -fsSL https://packages.redis.io/gpg  | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb  $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

# Update package list
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y redis-server

###################################################
RUN apt-get install -qq -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
## Install python
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --allow-change-held-packages \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
## Create a virtual environment

RUN python${PYTHON_VERSION} -m venv /opt/venv/cosmos_rl
ENV PATH="/opt/venv/cosmos_rl/bin:$PATH"

RUN pip install -U pip setuptools wheel packaging psutil

# even though we don't depend on torchaudio, vllm does. in order to
# make sure the cuda version matches, we install it here.
RUN set -eux; \
        case "${COSMOS_RL_TORCH_VARIANT}" in \
            2.8) \
                TORCH_VERSION=2.8.0; \
                TORCHVISION_VERSION=0.23.0; \
                TORCHAUDIO_VERSION=2.8.0; \
                TORCHAO_VERSION=0.13.0; \
                VLLM_VERSION=0.11.0; \
                FLASHINFER_VERSION=0.6.1; \
                ;; \
            2.10) \
                TORCH_VERSION=2.10.0; \
                TORCHVISION_VERSION=0.25.0; \
                TORCHAUDIO_VERSION=2.10.0; \
                TORCHAO_VERSION=0.16.0; \
                VLLM_VERSION=0.17.0; \
                FLASHINFER_VERSION=0.6.4; \
                FLASH_ATTN_WHEEL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"; \
                ;; \
            *) \
                echo "Unsupported COSMOS_RL_TORCH_VARIANT: ${COSMOS_RL_TORCH_VARIANT}. Expected 2.8 or 2.10."; \
                exit 1; \
                ;; \
        esac; \
        pip install torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" torchaudio=="${TORCHAUDIO_VERSION}" --index-url https://download.pytorch.org/whl/cu128; \
        pip install \
            torchao=="${TORCHAO_VERSION}" \
            ${FLASH_ATTN_WHEEL:-flash_attn=="${FLASH_ATTN_VERSION}"} \
            vllm=="${VLLM_VERSION}" \
            flashinfer-python=="${FLASHINFER_VERSION}" \
            transformer_engine[pytorch] --no-build-isolation

# install apex
RUN APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex@bf903a2

###################################################

# Install nvshmem grouped_gemm and DeepEP for MoE
RUN pip install nvidia-nvshmem-cu12==3.4.5
RUN TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4 --no-build-isolation
RUN apt-get update && apt-get install -y  libibverbs-dev
RUN git clone https://github.com/deepseek-ai/DeepEP.git /tmp/deepep \
    && cd /tmp/deepep \
    && if [ "${COSMOS_RL_TORCH_VARIANT}" = "2.8" ]; then \
        python setup.py build && python setup.py install; \
    elif [ "${COSMOS_RL_TORCH_VARIANT}" = "2.10" ]; then \
        pip install . --no-build-isolation; \
    else \
        echo "Unsupported COSMOS_RL_TORCH_VARIANT for DeepEP: ${COSMOS_RL_TORCH_VARIANT}. Expected 2.8 or 2.10."; \
        exit 1; \
    fi

# Phase for building any lib that we want to builf from source
FROM no-efa-base AS source-build

# install git
RUN apt-get update -y && apt-get install -y git

WORKDIR /workspace

RUN git clone --branch v${FLASH_ATTN_VERSION} --single-branch https://github.com/Dao-AILab/flash-attention.git

WORKDIR /workspace/flash-attention/hopper

RUN python setup.py bdist_wheel


FROM no-efa-base AS efa-base

# Remove HPCX and MPI to avoid conflicts with AWS-EFA
RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

RUN apt-get remove -y --purge --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1

###################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install AWS-OFI-NCCL plugin
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
#Switch from sh to bash to allow parameter expansion
SHELL ["/bin/bash", "-c"]
RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH


###################################################
## Image target: cosmos_rl
FROM ${COSMOS_RL_BUILD_MODE}-base AS pre-package

WORKDIR /workspace

# install fa3
COPY --from=source-build /workspace/flash-attention/hopper/dist/*.whl /workspace
RUN pip install /workspace/*.whl
RUN rm /workspace/*.whl

###################################################
## Image target: cosmos_rl
FROM pre-package AS package

ARG COSMOS_RL_EXTRAS

COPY . /workspace/cosmos_rl
RUN apt install -y cmake && \
    pip install /workspace/cosmos_rl${COSMOS_RL_EXTRAS:+[$COSMOS_RL_EXTRAS]} && \
    if [[ ",$COSMOS_RL_EXTRAS," == *,vla,* ]]; then \
        bash /workspace/cosmos_rl/tools/scripts/setup_vla.sh; \
    fi && \
    rm -rf /workspace/cosmos_rl
RUN pip uninstall -y xformers
