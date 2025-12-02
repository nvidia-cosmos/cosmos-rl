<p align="center">
    <img src="https://raw.githubusercontent.com/nvidia-cosmos/cosmos-rl/main/assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>


## Getting Started

Cosmos-RL is a flexible and scalable Reinforcement Learning framework specialized for Physical AI applications.

[Documentation](https://nvidia-cosmos.github.io/cosmos-rl)

## Multi-GPU Support

Cosmos-RL supports multi-GPU execution with a single parameter `num_gpus`:

### Evaluation

**Simple: Just set `num_gpus`**
```toml
# Use 8 GPUs - automatically launches 8 instances with data parallelism
num_gpus = 8

[model]
model_name = "nvidia/Cosmos-Reason1-7B"
# tp_size = 1 (default), total_shard = 8 (auto-calculated)
```

**Advanced: Tensor Parallelism**
```toml
num_gpus = 4

[model]
model_name = "nvidia/Cosmos-Reason1-7B"
tp_size = 4  # Split model across 4 GPUs
# total_shard = 1 (auto-calculated: 4 ÷ 4 = 1)
```

**Formula**: `num_gpus = total_shard × tp_size` (auto-calculated)

### Inference
- Use `--num_gpus N` for multi-GPU inference

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
