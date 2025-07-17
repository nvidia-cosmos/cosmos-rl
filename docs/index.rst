.. cosmos-rl documentation master file, created by
   sphinx-quickstart on Mon Jun  9 17:33:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cosmos-rl’s documentation!
==============================================

cosmos-rl is fully compatible with PyTorch and is designed for the future of distributed training.

Main Features
-------------
- **6D Parallelism**: Sequence, Tensor, Context, Pipeline, FSDP, DDP.

- **Elastic & Fault Tolerance**: A set of techniques to improve the robustness of distributed training.

- **Async RL**
   - **Flexible**
      - **Rollout** and **Policy** are decoupled into independent processes/GPUs.
      - No colocation of **Rollout** and **Policy** is required.
      - Number of **Rollout/Policy** instances can be scaled independently.
   - **Fast**
      - *IB/NVLink* are used for high-speed weight synchronization.
      - **Policy** training and **Rollout** weight synchronization are **PARALLELIZED**.
   - **Robust**
      - Support `AIPO <https://arxiv.org/pdf/2505.24034>`_ for stable off-policy training.
      - Async/Sync strategy can be selected upon to user's choice.
- **Multi-training Algorithms**
      - Supports state-of-the-art LLM RL algorithms (e.g., GRPO, DAPO, etc.).
      - Well-architected design ensures high extensibility, requiring only minimal configuration to implement custom training algorithms.
- **Diversified Model Support**
      - Natively supports LLaMA/Qwen/Qwen-VL/Qwen3-MoE series models.
      - Compatible with all Huggingface LLMs.
      - Easily extensible to other model architectures by customizing interface.

.. note::
   6D Parallelism is fully supported by Policy Model.
   For Rollout Model, only Tensor Parallelism and Pipeline Parallelism are supported.

.. toctree::
   :caption: Quick Start

   quickstart/installation
   quickstart/single_node_example
   quickstart/configuration
   quickstart/dataflow
   quickstart/customization
   quickstart/hf_models_support

.. toctree::
   :caption: Multi nodes training

   multinodes/overview
   multinodes/dgxc_lepton
   multinodes/slurm


.. toctree::
   :caption: Elastic & Fault Tolerance

   elastic/overview

.. toctree::
   :caption: Async RL

   async/overview

.. toctree::
   :caption: Parallelism

   parallelism/overview

.. toctree::
   :caption: Quantization

   quantization/fp8
