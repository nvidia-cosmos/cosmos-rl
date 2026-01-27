# Nemotron-VL Training Guide

This guide provides comprehensive instructions for training the Nemotron Vision-Language (VL) models using the Cosmos-RL framework.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Training](#running-training)
- [Key Features](#key-features)
- [Monitoring and Logging](#monitoring-and-logging)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

## Overview

This folder is dedicated for Nemotron-3-Nano based vision-language model training with following support:

- **Multi-modal support**: Vision + Language understanding
- **MoE Architecture**: Mixture-of-Experts with auxiliary-free load balancing
- **Advanced Parallelism**: Expert Parallelism (EP), FSDP, and Context Parallelism (CP is coming soon)
- **Flexible Dataset Support**: Custom datasets for vision-language and text-only training
- **Gradient Checkpointing**: Memory-efficient training for large models

## Prerequisites

### 1. Manully Install

```bash
# Install cosmos-rl framework
git clone -b dev/nemotron https://github.com/nvidia-cosmos/cosmos-rl.git
cd cosmos-rl && pip install -e .

# Optional dependencies
pip install wandb

# Install additional dependencies required by nemotron model
# 1. `deep_ep`: required if Expert Parallelism is enabled
# 2.  Mamba dependencies:
#   - `causal_conv1d`
#   - `mamba_ssm`
```

### 2. Prebuilt docker image (recommended)

Image: `nvcr.io/nvidian/cosmos-rl:nemotron`
```bash
# ASSUMED IN DOCKER

# Remove the built-in version of cosmos-rl
pip uninstall -y cosmos-rl
# Clone the latest branch for nemotron-3-nano support
git clone -b dev/nemotron https://github.com/nvidia-cosmos/cosmos-rl.git
cd cosmos-rl && pip install -e . --no-deps

# `deep_ep`, `mamba_ssm`, `causal_conv1d` are already built-in.
```

### Model Checkpoints

- Base LLM: `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
   Storage `/$CDG_NFS_PATH/nemotron-vlm/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Vision-Language Model: `NVIDIA-Nemotron-3-Nano-VL-30B-A3B-BF16`
   Storage: `/$CDG_NFS_PATH/nemotron-vlm/NVIDIA-Nemotron-3-Nano-VL-30B-A3B-BF16`

## Quick Start

### Basic Training Command

```bash
cd cosmos-rl
cosmos-rl --config config.toml ./nemotron_vl/launcher.py
```

This command:
1. Loads configuration from `config.toml`
2. Executes the launcher script at [nemotron_vl/launcher.py](../nemotron_vl/launcher.py)
3. Initializes distributed training environment
4. Starts the training loop

## Running Training

```bash
# Basic training with default config
cosmos-rl --config ./nemotron_vl/nemotron_vl.toml ./nemotron_vl/launcher.py
```


## Configuration

The training configuration uses TOML format. Key sections are described below.

### Redis Configuration

```toml
redis = "12800"
```

Specifies the Redis port for distributed coordination.

### Training Parameters (`[train]`)

```toml
[train]
resume = true                          # Resume from checkpoint if available
epoch = 1                              # Number of training epochs
output_dir = "./outputs/Nemotron-3-Nano-VL-Alignment"
epsilon = 1e-6

# Optimizer settings
optm_name = "AdamW"
optm_lr = 1e-5                         # Learning rate
optm_impl = "fused"                    # Use fused optimizer implementation
optm_weight_decay = 0.01
optm_betas = [0.9, 0.999]
optm_warmup_steps = 20
optm_decay_type = "cosine"
optm_grad_norm_clip = 1.0

# Precision and memory
param_dtype = "bfloat16"               # Parameter data type
fsdp_reduce_dtype = "float32"          # Reduction data type for FSDP
master_dtype = "float32"               # Master weights data type
fsdp_offload = false                   # CPU offloading (enable for memory saving)
fsdp_reshard_after_forward = "default" # Options: "default", "always", "never"

# Batch sizes
train_batch_per_replica = 32           # Training batch size per replica
validation_batch_per_replica = 2       # Validation batch size

# Checkpointing and validation
enable_validation = false
```

### Custom Settings (`[custom]`)

```toml
[custom]
enable_moe_load_balancing_training = true   # Enable MoE load balancing
n_step_per_workload_report = 10             # Report MoE stats every N steps
include_video = false                       # Include video data in training
train_layers = ["Qwen3VLVisionPatchMerger"] # Optional: name pattern to specify the trainable modules, useful for VL model. Here `Qwen3VLVisionPatchMerger` stands for the projector of vision features from ViT 
```

### Policy Configuration (`[policy]`)

```toml
[policy]
model_name_or_path = "/path/to/NVIDIA-Nemotron-3-Nano-VL-30B-A3B-BF16"
model_max_length = 8192                # Maximum sequence length
model_gradient_checkpointing = true    # Enable gradient checkpointing
```

### Parallelism Configuration (`[policy.parallelism]`)

```toml
[policy.parallelism]
n_init_replicas = 1
tp_size = 8              # Tensor/Expert parallelism size
cp_size = 1              # Context parallelism size (not supported)
dp_shard_size = 1        # Data parallelism (FSDP) shard size
pp_size = 1              # Pipeline parallelism size (not supported)
dp_replicate_size = 1    # Data parallelism replicate size
cp_rotate_method = "allgather"
```

**Parallelism Strategy:**
- Expert Parallelism (EP) is automatically applied to MoE layers within the TP mesh
- FSDP is applied for data parallelism when `dp_shard_size > 1`

### Dataset Configuration (`[train.train_policy]`)

```toml
[train.train_policy]
type = "sft"                          # Training type: supervised fine-tuning
dataset.name = "HuggingFaceH4/llava-instruct-mix-vsft"
dataset.subset = ""
dataset.split = "train"
dataset.test_size = 100
enable_dataset_cache = false
dataloader_num_workers = 4
dataloader_prefetch_factor = 4
conversation_column_name = ""         # Column name for conversation data
mini_batch = 2                        # Gradient accumulation steps
```

**Supported Dataset Formats:**
1. **Vision-Language Datasets**: JSONL files with multi-modal messages
2. **Text-Only Datasets**: HuggingFace datasets with conversation format

### Logging Configuration (`[logging]`)

```toml
[logging]
logger = ['console', 'wandb']         # Logging backends
project_name = "nemotron-3-nano-vlm"  # W&B project name
experiment_name = "alignment"         # W&B run name
```

### Checkpoint Configuration (`[train.ckpt]`)

```toml
[train.ckpt]
enable_checkpoint = true
save_freq = 100                       # Save checkpoint every N steps
save_mode = "async"                   # Async or sync checkpoint saving
```

## Key Features

### 1. MoE Load Balancing

The framework implements auxiliary-free load balancing for Mixture-of-Experts layers:

- **Bias Update Mechanism**: Automatically adjusts expert routing bias based on token distribution
- **Per-Layer Tracking**: Monitors expert utilization per layer
- **Automatic Reporting**: Logs balance metrics to W&B every N steps

Configuration:
```toml
[custom]
enable_moe_load_balancing_training = true
n_step_per_workload_report = 10
```

**Reported Metrics:**
- `moe/entropy_mean`: Average routing entropy across layers
- `moe/max_fraction_max`: Maximum expert load fraction
- `moe/tokens_per_layer_mean`: Average tokens per layer
- `moe/layer_expert_table`: Detailed per-expert token distribution

### 2. Custom Dataset Support

The launcher supports two dataset classes:

#### CustomDataset (Vision-Language)

For multi-modal training with images/videos:

```python
# Dataset structure in JSONL format
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "Describe this image"}
      ]
    },
    {
      "role": "assistant",
      "content": "This is a..."
    }
  ]
}
```

Features:
- Automatic pixel constraint: `max_pixels = model_max_length * 0.9 * (16 * 2)^2`
- Video support: Set `include_video = true` in config
- FPS adjustment for video frames


### Checkpoints

Checkpoints are saved to `output_dir` and include:
- Model weights (safetensors format)
- Optimizer state
- Training step and epoch
- Configuration

Checkpoint format:
```
outputs/Nemotron-3-Nano-VL-Alignment/
├── checkpoint/
│   ├── step_100/   (model_state.pt, optimizer_state.pt, config.json, ...)
│   ├── step_200/
│   └── ...
└── safetensors/
    ├── step_100/   (*.safetensors, maybe config.json, index.json, ...)
    ├── step_200/
    └── ...
```

