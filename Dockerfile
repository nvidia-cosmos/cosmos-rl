# x86_64 Usage:
# To build without AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=no-efa .
# To build with AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=efa .

# arm64 Usage:
# To build without AWS-EFA:
#   docker buildx build --platform linux/arm64 --build-arg COSMOS_RL_BUILD_MODE=no-efa -t cosmos_rl:arm64 .
# To build with AWS-EFA:
#   docker buildx build --platform linux/arm64 --build-arg COSMOS_RL_BUILD_MODE=efa -t cosmos_rl:arm64 .

ARG COSMOS_RL_BUILD_MODE=no-efa
ARG CUDA_VERSION=12.8.1
ARG TARGETARCH
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS no-efa-base

ARG TARGETARCH
RUN echo "Building for TARGETARCH: $TARGETARCH"

ARG GDRCOPY_VERSION=v2.4.4
ARG EFA_INSTALLER_VERSION=1.42.0
ARG AWS_OFI_NCCL_VERSION=v1.16.0
# NCCL version, should be found at https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG PYTHON_VERSION=3.12

ENV TZ=Etc/UTC

RUN apt-get update -y && apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    curl git gpg lsb-release tzdata wget unzip nginx default-jre dnsutils zlib1g-dev libzstd-dev && \
    apt-get purge -y cuda-compat-* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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
## Install redis
# Build Redis from source without jemalloc to avoid 64KB page size issues on ARM64
RUN if [ "$TARGETARCH" != "amd64" ]; then \
        apt-get update -qq && \
        DEBIAN_FRONTEND=noninteractive apt-get install -qq -y build-essential tcl pkg-config wget && \
        cd /tmp && \
        wget https://download.redis.io/redis-stable.tar.gz && \
        tar -xzf redis-stable.tar.gz && \
        cd redis-stable && \
        make distclean || true && \
        JEMALLOC_CONFIGURE_OPTS="--with-lg-page=16" make -j"$(nproc)" MALLOC=jemalloc && \
        make install PREFIX=/usr/local && \
        cd / && \
        rm -rf /tmp/redis-stable* ; \
    else \
        curl -fsSL https://packages.redis.io/gpg  | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
        chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb  $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list && \
        apt-get update -qq && \
        DEBIAN_FRONTEND=noninteractive apt-get install -qq -y redis-server ; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

###################################################
## Install python
RUN apt-get update -qq && \
    apt-get install -qq -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --allow-change-held-packages \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## Create a virtual environment

RUN python${PYTHON_VERSION} -m venv /opt/venv/cosmos_rl
ENV PATH="/opt/venv/cosmos_rl/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv/cosmos_rl"

# Create virtual environment activation for both interactive and non-interactive bash sessions
RUN echo 'source /opt/venv/cosmos_rl/bin/activate' >> /root/.bashrc
RUN echo 'source /opt/venv/cosmos_rl/bin/activate' > /etc/bash.bashrc
ENV BASH_ENV=/etc/bash.bashrc

WORKDIR /workspace/cosmos_rl

RUN pip install --no-cache-dir -U pip setuptools wheel packaging psutil

# even though we don't depend on torchaudio, vllm does. in order to
# make sure the cuda version matches, we install it here.
# Currently, only the nightly index supports aarch64 wheels for torch.

RUN if [ "$TARGETARCH" != "amd64" ]; then \
    pip install --no-cache-dir \
        torch==2.10.0.dev20251029+cu128 \ 
        torchvision==0.25.0.dev20251030 \ 
        torchaudio==2.10.0.dev20251030 \
        --index-url https://download.pytorch.org/whl/nightly/cu128 ; \
    else \
        pip install --no-cache-dir \
            torch==2.8.0 \ 
            torchvision==0.23.0 \ 
            torchaudio==2.8.0 \
            --index-url https://download.pytorch.org/whl/cu128 ; \
    fi

RUN pip install ninja && MAX_JOBS=8 pip install --no-cache-dir flash-attn --no-build-isolation

RUN pip install --no-cache-dir \
    torchao==0.13.0 \
    flashinfer-python \
    transformer_engine[pytorch] --no-build-isolation

RUN if [ "$TARGETARCH" != "amd64" ]; then \
        apt-get update && \
        apt-get upgrade -y && \
        apt-get install -y git ; \
    fi

RUN if [ "$TARGETARCH" != "amd64" ]; then \
        # Install vllm from source using existing torch on ARM64 due to lack of prebuilt ARM GPU wheels on PyPI for torch. 
        # Pin vllm to v0.11.0 for headdim issue: https://github.com/vllm-project/vllm/issues/27562.
        git clone https://github.com/vllm-project/vllm.git /tmp/vllm && \
        cd /tmp/vllm && \
        git checkout v0.11.0 && \ 
        python use_existing_torch.py && \
        pip install -r requirements/build.txt && \
        pip install --no-build-isolation . && \
        cd / && \
        rm -rf /tmp/vllm ; \
    else \
        pip install --no-cache-dir vllm==0.11.0 --no-build-isolation ; \
    fi

# Copy source code last - this layer will rebuild when code changes
# but pip installs above will be cached
COPY . .

RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt

###################################################
FROM no-efa-base AS efa-base

# Remove HPCX and MPI to avoid conflicts with AWS-EFA
RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

RUN for pkg in ibverbs-utils libibverbs-dev libibverbs1 libmlx5-1; do \
        if dpkg -l | grep -q "^ii  $pkg "; then \
            apt-get remove -y --purge --allow-change-held-packages $pkg; \
        fi; \
    done && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

###################################################
## Install EFA installer
RUN cd $HOME && \
    apt-get update -y && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    cd $HOME && \
    rm -rf $HOME/aws-efa-installer* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

###################################################
## Install AWS-OFI-NCCL plugin
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#Switch from sh to bash to allow parameter expansion
SHELL ["/bin/bash", "-c"]
RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz && \
    tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz && \
    cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} && \
    ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}* && \
    ldconfig

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH


###################################################
## Image target: cosmos_rl (runtime optimized)
FROM ${COSMOS_RL_BUILD_MODE}-base AS package

# Clean up unnecessary packages to reduce image size
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install additional dependencies that depend on the source code
# Use SETUPTOOLS_SCM_PRETEND_VERSION since .git is excluded from Docker context
RUN if [ "$TARGETARCH" != "amd64" ]; then \
        pip install --no-cache-dir -U git+https://github.com/nvidia-cosmos/cosmos-reason1.git#subdirectory=cosmos_reason1_utils && \
        PIP_CONSTRAINT=/workspace/cosmos_rl/constraints.txt SETUPTOOLS_SCM_PRETEND_VERSION=0.3.1 pip install --no-cache-dir -e . && \
        pip cache purge ; \
    else \
        pip install --no-cache-dir -U git+https://github.com/nvidia-cosmos/cosmos-reason1.git#subdirectory=cosmos_reason1_utils && \
        SETUPTOOLS_SCM_PRETEND_VERSION=0.3.1 pip install --no-cache-dir -e . && \
        pip cache purge ; \
    fi

# Installing TAO-Core
RUN . /opt/venv/cosmos_rl/bin/activate && \
    cd tao-core && \
    bash release/python/build_wheel.sh && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    cp nvidia_tao_core/microservices/nginx.conf /etc/nginx/ && \
    cd .. && \
    rm -rf tao-core

RUN pip uninstall -y ray


ENV NVIDIA_PRODUCT_NAME="TAO Toolkit"
ENV TAO_TOOLKIT_VERSION="6.25.7"
ENV NVIDIA_TAO_TOOLKIT_VERSION="${TAO_TOOLKIT_VERSION}-PyTorch"

# Defining the telemetry URL.
ENV TAO_TELEMETRY_SERVER="https://api.tao.ngc.nvidia.com"

EXPOSE 8000

# Microservices entrypoint
ENV FLASK_APP=nvidia_tao_core.microservices.app

ENV RUN_CLI=0

CMD if [ "$RUN_CLI" = "1" ]; then \
        /bin/bash; \
    else \
        /bin/bash $(get-microservice-script); \
    fi
