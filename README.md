> [!IMPORTANT]
> ## 🚀 [Cosmos 3 Has Arrived](https://github.com/nvidia/cosmos)
>
> Cosmos 3 is NVIDIA's next-generation foundation model platform for Physical AI. Compared with Cosmos-RL, Cosmos 3 unifies reasoning, world prediction, simulation, transfer, and action generation within a single model family and ecosystem.
>
> Rather than relying on separate models for reasoning, prediction, transfer, and policy learning, a single Cosmos 3 model can understand the world, reason about physical interactions, predict future outcomes, transform observations across domains, and generate actions for embodied agents. This unified architecture enables stronger performance across a broad range of Physical AI applications, including robotics, autonomous vehicles, and smart spaces.
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on Cosmos 3.
>
> 👉 Visit the new Cosmos home: https://github.com/nvidia/cosmos
>
> There you will find the latest Cosmos 3 models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Cosmos-RL. We encourage all users to migrate to Cosmos 3 for the latest state-of-the-art Physical AI capabilities.

<p align="center">
    <img src="https://raw.githubusercontent.com/nvidia-cosmos/cosmos-rl/main/assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>


## Getting Started

Cosmos-RL is a flexible and scalable Reinforcement Learning framework specialized for Physical AI applications.

[Documentation](https://nvidia-cosmos.github.io/cosmos-rl).

## System Architecture
Cosmos-RL provides toolchain to enable large scale RL training workload with following features:
1. **Parallelism**
    - Tensor Parallelism
    - Sequence Parallelism
    - Context Parallelism
    - FSDP Parallelism
    - Pipeline Parallelism
2. **Fully asynchronous (replicas specialization)**
    - Policy (Consumer): Replicas of training instances
    - Rollout (Producer): Replicas of generation engines
    - Low-precision training (FP8) and rollout (FP8 & FP4) support
3. **Single-Controller Architecture**
    - Efficient messaging system (e.g., `weight-sync`, `rollout`, `evaluate`) to coordinate policy and rollout replicas
    - Dynamic NCCL Process Groups for on-the-fly GPU [un]registration to enable fault-tolerant and elastic large-scale RL training

![Policy-Rollout-Controller Decoupled Architecture](https://raw.githubusercontent.com/nvidia-cosmos/cosmos-rl/main/assets/rl_infra.svg)

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
