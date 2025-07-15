Installation
============

Requirements
------------

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.2
- **PyTorch**: Version >= 2.6.0
- **vLLM**: Version >= 0.8.5

Setup
------------

You can either build docker image from source or install dependencies inside your python env:

🐳 Option 1: Build docker image from source
::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

    docker build -t cosmos-rl-dev:dev .
    docker run -it --gpus all --shm-size=24G -w /workspace/cosmos-rl cosmos-rl-dev:dev

.. note::
    EFA driver is included in the docker image specifically for aws instances with EFA net interface (Sagemaker AI Pod).

    If you are not using EFA, you can build the docker image with `docker build --build-arg COSMOS_RL_BUILD_MODE=no-efa -t cosmos-rl-dev:dev --target package .`.

🔨 Option 2: Run in your own environment
:::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

    apt-get update && apt-get install redis-server
    pip install -r requirements.txt
    pip install -e .
