FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update -qq && apt-get install -qq -y --allow-change-held-packages \
    build-essential tzdata git openssh-server curl netcat elfutils \
    python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
    lsb-release gpg

# Download and add Redis GPG key
RUN curl -fsSL https://packages.redis.io/gpg  | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg

# Add Redis APT repository
RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb  $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

# Update package list
RUN apt-get update -qq

# Install specific Redis version
RUN apt-get install -qq -y redis-server

#################################################
## Install EFA installer
ARG EFA_INSTALLER_VERSION=1.42.0
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install AWS-OFI-NCCL plugin
ARG AWS_OFI_NCCL_VERSION=v1.16.0
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

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
ARG GDRCOPY_VERSION=v2.4.4
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

RUN pip install -U pip setuptools wheel packaging
# even though we don't depend on torchaudio, vllm does. in order to
# make sure the cuda version matches, we install it here.
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /workspace/cosmos_rl/requirements.txt
RUN pip install \
    torchao==0.11.0 \
    vllm==0.9.1 \
    flash-attn==2.8.0.post2 \
    https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl \
    -r /workspace/cosmos_rl/requirements.txt

FROM base AS package

COPY README.md pyproject.toml /workspace/cosmos_rl/
COPY tools /workspace/cosmos_rl/tools
COPY configs /workspace/cosmos_rl/configs
COPY cosmos_rl /workspace/cosmos_rl/cosmos_rl

RUN pip install -e /workspace/cosmos_rl
