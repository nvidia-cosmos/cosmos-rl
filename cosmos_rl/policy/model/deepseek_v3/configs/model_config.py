# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import attrs


@attrs.define
class TrainingConfig:
    """
    Training configuration parameters including parallelism, precision, and optimization settings.

    Attributes:
        compile (bool): Whether to compile the model using torch.compile
        data_parallel_shard_degree (int): Degree of data parallelism for weight sharding (FSDP/HSDP)
        data_parallel_replicate_degree (int): Degree of data parallelism for weight replication (DDP/HSDP)
        tensor_parallel_degree (int): Tensor parallelism degree. 1 means disabled.
        disable_loss_parallel (bool): Disable loss parallel when sequence parallel is enabled
        mixed_precision_param (str): Param precision for mixed training (bfloat16/float32)
        mixed_precision_reduce (str): Reduction precision for mixed training (float32)
        enable_cpu_offload (bool): Enable CPU offloading of parameters/gradients in FSDP
    """

    compile: bool = False

    data_parallel_replicate_degree: int = 1

    # Set to -1 to compute FSDP based on world size and model parameters.
    # Only data_parallel_shard_degree can be set to -1, other parallelism
    # degrees must be set to positive integers.
    data_parallel_shard_degree: int = -1

    tensor_parallel_degree: int = 1
    context_parallel_degree: int = 1
    expert_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1

    # Pipeline parallelism options. Supported schedules are:
    # 1F1B, Interleaved1F1B, GPipe, LoopedBFS, InterleavedZeroBubble,
    # ZBVZeroBubble
    pipeline_parallel_schedule: str = "GPipe"
    pipeline_parallel_layers_per_stage: int | None = None
    pipeline_parallel_microbatch_size: int = 1

    disable_loss_parallel: bool = False
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"
    enable_cpu_offload: bool = False
    warmup_steps: int = 1000
    steps: int = 400_000
    use_linear_decay: bool = True
    use_cosine_decay: bool = False
    fsdp_reshard_after_forward: str = "default"


@attrs.define
class ExperimentalConfig:
    """
    Experimental features and advanced parallelism configurations.

    Attributes:
        enable_async_tensor_parallel (bool): Enable async tensor parallel (requires compile)
        enable_compiled_autograd (bool): Enable compiled autograd for backward pass optimization
    """

    enable_async_tensor_parallel: bool = False
    enable_compiled_autograd: bool = False


@attrs.define
class OptimizerConfig:
    """
    Optimizer config.

    Attributes:
        name (str): Optimizer name
        lr (float): Learning rate
        fused (bool): Whether the fused implementation (CUDA only) is used.
        early_step_in_backward (bool): Whether to apply optimizer in the backward. Caution, optimizer_in_backward
            is not compatible with gradients clipping, users should not call
            register_post_accumulate_grad_hook after the optimizer is built
    """

    name: str = "AdamW"
    lr: float = 3e-4
    init_lr: float = 1e-5
    end_lr: float = 2.5e-5
    fused: bool = False
    early_step_in_backward: bool = False
    lr_multiplier_llm: float = 1.0


@attrs.define
class ActivationCheckpointConfig:
    """
    Activation checkpointing (gradient checkpointing) configuration.

    Attributes:
        mode (str): Checkpointing mode - 'none', 'full', or 'selective'
        selective_ac_option (str): Selective checkpointing strategy ('op' or layer frequency)
    """

    mode: str = "selective"  # "none", "full", "selective"
    models: str = "llm"  # "llm"
    selective_ac_option: str = "op"


@attrs.define
class Float8Config:
    """
    Float8 mixed precision training configurations.

    Attributes:
        enable_float8_linear (bool): Use float8 linear layers from torchao
    """

    enable_float8_linear: bool = False


@attrs.define
class CheckpointConfig:
    """
    fsdp2 checkpoint config

    Attributes:
        enable_checkpoint (bool): Whether to enable checkpointing
        folder (str): The folder to store the checkpoints
        interval_type (str): Checkpointing interval unit of measurement ['step', 'seconds']
        interval (int): Checkpointing interval, in steps or seconds depending on --checkpoint.interval_type
        model_weights_only (bool): When model_weights_only=True, only model weights will be saved at the end of training.
                With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
                When model_weights_only=False, the full checkpoint will be saved.
                A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
                The default value is false.
        export_dtype (str): Converts to the specified precision when training completes and model_weights_only=true.
                Currently supports float32, float16, and bfloat16.
                The default value is float32.
        async_mode (str): Which async checkpoint mode to use. Currently there are 3 different modes.
                1. "disabled": synchronized checkpointing will be used.
                2. "async": torch.distributed.checkpoint.async_save will be used.
                3. "async_with_pinned_mem": this option utilizes a dedicated pinned memory
                   space and creates a separate process for faster GPU->CPU transfer
                   performance and eliminating GIL contention. The cost is increased CPU
                   memory usage. If insufficient CPU memory is available, performance may
                   degrade due to memory paging. For most users, "async" should suffice as
                   the performance overhead is typically small (on the order of tens of
                   seconds) compared to checkpointing frequency. This mode can be employed
                   to pursue near-zero checkpointing times (e.g., < 1 second) given
                   appropriate hardware support such as ample CPU memory and fast PCIe.

                "disabled" is the default mode.
        create_seed_checkpoint (bool): Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
                Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
                Could be implemented as a separate script, but this way shares more code.
    """

    enable_checkpoint: bool = False
    folder: str = "checkpoint"
    interval_type: str = "steps"
    interval: int = 500
    model_weights_only: bool = False
    export_dtype: str = "float32"
    async_mode: str = "disabled"
    create_seed_checkpoint: bool = False


@attrs.define
class CommConfig:
    """
    Communication config.

    Attributes:
        init_timeout_seconds (int): Timeout for communication operations, during initialization and first train step.
        train_timeout_seconds (int): Timeout for communication operations after the first train step -- usually a tighter bound than during initialization.
        trace_buf_size (int): Flight recorder ring buffer size, >0 means recording by default, 0 means disabled
    """

    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100
    trace_buf_size: int = 20000


@attrs.define
class FSDP2ModelConfig:
    """
    A class to hold model configuration arguments.
    Args:
        tokenizer_type (str): This is used for tokenizer initialization
        dim (int): Dimension of the model.
        n_layers (int): Number of layers in the model.
        head_dim (int): Dimension of the head.
        hidden_dim (int): Dimension of the hidden layer.
        n_heads (int): Number of heads in the model.
        n_kv_heads (Optional[int]): Number of key-value heads in the model.
        rope_theta (float): Theta value for RoPE.
        norm_type (str): Type of normalization.
        norm_eps (float): Epsilon value for normalization.
        vocab_size (int): Size of the vocabulary.
        max_seq_len (int): Maximum sequence length.
        vision_encoder (str): Path to the vision encoder.
        vision_encoder_in_channels (int): Number of channels in the input image for the vision encoder.
        mm_projector (str): Multi-modal projector type.
        depth_init (bool): Flag to indicate if the depth is initialized.
        use_fsdp2 (bool): Flag to indicate if the model is using fsdp2. Default is True.
        use_rope_from_torchtitan (bool): Flag to indicate if using the rope implementation from torchtitan/llama or from HF. True if the checkpoint is converted from original Llama weight instead of HF weight. Default is False.
            see: https://github.com/pytorch/torchtitan/issues/335

        training (TrainingConfig): Training configuration.
        experimental (ExperimentalConfig): Experimental configuration.
        activation_checkpoint (ActivationCheckpointConfig): Activation checkpointing configuration.
        float8 (Float8Config): Float8 mixed precision training configurations.
        checkpoint (CheckpointConfig): fsdp2 checkpoint config.
        optimizer (OptimizerConfig): Optimizer config.
        comm (CommConfig): Communication config.

        seed (int): Random seed.
        deterministic (bool): Whether to use deterministic training.

        qwen_max_video_token_length (int): Maximum video token length.
        qwen_target_fps (float): Target fps for video sampling.
    """

    tokenizer_type: str
    # Shared config for all models
    # Config for kv-cache
    max_batch_size: int = 1
    max_seq_len: int = 128000  # config of the base model, used for kv cache size

    training_seq_len: int = 4096  # sequence length used for training data

    # For backward compatibility
    use_fsdp2: bool = True
    use_rope_from_torchtitan: bool = False

    ckpt_dir: str = None
    ckpt_path: str = None
    s3_credential_path: str = "credentials/pbss_dir.secret"
    cache_dir: str = None
    precision: str = "bfloat16"

    fsdp_enabled: bool = False
    z_loss_coeff: float = 0.0  # We dont use z-loss

    # For pretraining
    freeze_llm: bool = False

    # Torchtitan use Llama original rope implementation, but it only works for original llama weight; HF weight permute the rope, and adapt different rope implementation
    # Reference: https://github.com/pytorch/torchtitan/issues/335

    # training for fsdp2
    training: TrainingConfig = TrainingConfig()
    experimental: ExperimentalConfig = ExperimentalConfig()
    activation_checkpoint: ActivationCheckpointConfig = ActivationCheckpointConfig()
    float8: Float8Config = Float8Config()
    checkpoint: CheckpointConfig = CheckpointConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    comm: CommConfig = CommConfig()
    seed: int = 0
    deterministic: bool = False
    # Config for kv-cache
    use_cache: bool = False

    # Config for loss
    loss_per_token: bool = True

    # Config for aux loss
    aux_loss_coeff: float = 0.0
    prepend_padding: bool = False  # for video

    def __getitem__(self, item):
        return getattr(self, item)


@attrs.define
class DeepseekConfig(FSDP2ModelConfig):
    dtype: str = "bfloat16"

    # Config for Deepseek R1.
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 4
    n_dense_layers: int = 3
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    train_gate: bool = False
    gate_bias_update_factor: float = 0.0
    score_func: str = "sigmoid"
    route_scale: float = 2.5
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    enable_deepep: bool = False
    fake_balanced_gate: bool = False
