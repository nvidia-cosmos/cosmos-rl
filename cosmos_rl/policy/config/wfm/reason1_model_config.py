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

from typing import Optional, Literal
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """
    Training configuration parameters including parallelism, precision, and optimization settings.
    """

    compile: bool = Field(
        default=False, description="Whether to compile the model using torch.compile"
    )
    data_parallel_shard_degree: int = Field(
        default=-1,
        description="Degree of data parallelism for weight sharding (FSDP/HSDP)",
    )
    data_parallel_replicate_degree: int = Field(
        default=1,
        description="Degree of data parallelism for weight replication (DDP/HSDP)",
    )
    tensor_parallel_degree: int = Field(
        default=1, description="Tensor parallelism degree. 1 means disabled."
    )
    context_parallel_degree: int = Field(
        default=1, description="Context parallelism degree. 1 means disabled."
    )

    disable_loss_parallel: bool = Field(
        default=False,
        description="Disable loss parallel when sequence parallel is enabled",
    )
    mixed_precision_param: str = Field(
        default="bfloat16", description="Param precision for mixed training"
    )
    mixed_precision_reduce: str = Field(
        default="float32", description="Reduction precision for mixed training"
    )
    enable_cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offloading of parameters/gradients in FSDP",
    )
    warmup_steps: int = 1000
    steps: int = 400000
    use_linear_decay: bool = True
    use_cosine_decay: bool = False
    fsdp_reshard_after_forward: str = "default"


class ExperimentalConfig(BaseModel):
    """
    Experimental features and advanced parallelism configurations.
    """

    pipeline_parallel_degree: int = Field(
        default=1, description="Pipeline parallelism degree. 1 means disabled."
    )
    enable_async_tensor_parallel: bool = Field(
        default=False, description="Enable async tensor parallel (requires compile)"
    )
    enable_compiled_autograd: bool = Field(
        default=False,
        description="Enable compiled autograd for backward pass optimization",
    )


class OptimizerConfig(BaseModel):
    """
    Optimizer config.
    """

    name: str = Field(default="AdamW", description="Optimizer name")
    lr: float = Field(default=3e-4, description="Learning rate")
    init_lr: float = Field(default=1e-5, description="Initial learning rate")
    end_lr: float = Field(default=2.5e-5, description="End learning rate")
    fused: bool = Field(
        default=False,
        description="Whether the fused implementation (CUDA only) is used",
    )
    early_step_in_backward: bool = Field(
        default=False,
        description="Whether to apply optimizer in the backward. Not compatible with gradient clipping",
    )
    lr_multiplier_vision_encoder: float = 0.1
    lr_multiplier_mm_projector: float = 1.0
    lr_multiplier_llm: float = 1.0


class ActivationCheckpointConfig(BaseModel):
    """
    Activation checkpointing (gradient checkpointing) configuration.
    """

    mode: Literal["none", "full", "selective"] = Field(
        default="selective", description="Checkpointing mode"
    )
    models: Literal["vlm", "llm", "vision"] = Field(
        default="vlm", description="Which models to apply checkpointing to"
    )
    selective_ac_option: str = Field(
        default="op", description="Selective checkpointing strategy"
    )


class Float8Config(BaseModel):
    """
    Float8 mixed precision training configurations.
    """

    enable_float8_linear: bool = Field(
        default=False, description="Use float8 linear layers from torchao"
    )


class CheckpointConfig(BaseModel):
    """
    FSDP2 checkpoint config
    """

    enable_checkpoint: bool = Field(
        default=False, description="Whether to enable checkpointing"
    )
    folder: str = Field(
        default="checkpoint", description="The folder to store the checkpoints"
    )
    interval_type: str = Field(
        default="steps", description="Checkpointing interval unit of measurement"
    )
    interval: int = Field(default=500, description="Checkpointing interval")
    model_weights_only: bool = Field(
        default=False,
        description="When True, only model weights will be saved at the end of training",
    )
    export_dtype: str = Field(
        default="float32",
        description="Converts to the specified precision when training completes and model_weights_only=true",
    )
    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = Field(
        default="disabled", description="Which async checkpoint mode to use"
    )
    create_seed_checkpoint: bool = Field(
        default=False,
        description="Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint",
    )


class CommConfig(BaseModel):
    """
    Communication config.
    """

    init_timeout_seconds: int = Field(
        default=300,
        description="Timeout for communication operations, during initialization and first train step",
    )
    train_timeout_seconds: int = Field(
        default=100,
        description="Timeout for communication operations after the first train step",
    )
    trace_buf_size: int = Field(
        default=20000,
        description="Flight recorder ring buffer size, >0 means recording by default, 0 means disabled",
    )


class VisionEncoderConfig(BaseModel):
    """
    Vision encoder config:

    By default, this is config for Pixtral's vision encoder
    """

    dim: int = 1024
    num_channels: int = 3
    image_size: int = 1024
    patch_size: int = 16
    rope_theta: float = 10000
    hidden_dim: int = 4096
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    image_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    use_rope_from_torchtitan: bool = False

    # Only for llama
    multiple_of: Optional[int] = None
    ffn_dim_multiplier: Optional[int] = None
    depth_init: bool = True
    hidden_act: Optional[str] = None
    qkv_bias: Optional[bool] = None
    proj_bias: Optional[bool] = None
    use_cache: bool = False  # This is because VIT also use the Attention class, for shared interface, but it should always be False


class FSDP2ModelConfig(BaseModel):
    """
    A class to hold model configuration arguments.
    """

    tokenizer_type: str = Field(description="This is used for tokenizer initialization")

    # Shared config for all models
    max_batch_size: int = Field(default=1, description="Config for kv-cache")
    max_seq_len: int = Field(
        default=128000, description="config of the base model, used for kv cache size"
    )
    training_seq_len: int = Field(
        default=4096, description="sequence length used for training data"
    )

    # For backward compatibility
    use_fsdp2: bool = Field(
        default=True, description="Flag to indicate if the model is using fsdp2"
    )
    use_rope_from_torchtitan: bool = Field(
        default=False,
        description="Flag to indicate if using the rope implementation from torchtitan/llama or from HF",
    )

    vision_encoder: str = Field(
        default="openai/clip-vit-base-patch32", description="Path to the vision encoder"
    )
    vision_encoder_in_channels: int = Field(
        default=3,
        description="Number of channels in the input image for the vision encoder",
    )
    vision_encoder_config: VisionEncoderConfig = Field(
        default_factory=VisionEncoderConfig
    )
    mm_projector: Optional[str] = Field(
        default=None, description="Multi-modal projector type"
    )

    ckpt_dir: Optional[str] = None
    ckpt_path: Optional[str] = None
    s3_credential_path: str = "credentials/pbss_dir.secret"
    cache_dir: Optional[str] = None
    precision: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    fsdp_enabled: bool = False
    z_loss_coeff: float = Field(default=0.0, description="We dont use z-loss")

    # For pretraining
    freeze_vision_encoder: bool = False
    freeze_mm_projector: bool = False
    freeze_llm: bool = False

    # Nested configurations
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration"
    )
    experimental: ExperimentalConfig = Field(
        default_factory=ExperimentalConfig, description="Experimental configuration"
    )
    activation_checkpoint: ActivationCheckpointConfig = Field(
        default_factory=ActivationCheckpointConfig,
        description="Activation checkpointing configuration",
    )
    float8: Float8Config = Field(
        default_factory=Float8Config,
        description="Float8 mixed precision training configurations",
    )
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="fsdp2 checkpoint config"
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer config"
    )
    comm: CommConfig = Field(
        default_factory=CommConfig, description="Communication config"
    )

    seed: int = Field(default=0, description="Random seed")
    deterministic: bool = Field(
        default=False, description="Whether to use deterministic training"
    )

    # Image data processing and prompt formatting
    num_tiles: int = 1
    add_tile_tag: bool = False
    add_image_start_end_tag: bool = False
    add_answer_tag: bool = True
    tile_tag_type: str = "space_separated"

    # Config for kv-cache
    use_cache: bool = False

    # Parallelism configurations
    cp_size: Optional[int] = None
    ep_size: Optional[int] = None

    # Config for loss
    loss_per_token: bool = True

    # Config for aux loss
    aux_loss_coeff: float = 0.0
    prepend_padding: bool = Field(default=False, description="for video")
