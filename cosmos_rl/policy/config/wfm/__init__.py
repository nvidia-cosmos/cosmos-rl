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

import os
import math
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Literal, Optional, Union

from cosmos_rl.policy.config.wfm.qwen_config import QwenModelConfig


class EmbeddingConcatStrategy(str, Enum):
    FULL_CONCAT = "full_concat"  # Concatenate embeddings all layers
    MEAN_POOLING = "mean_pooling"  # Average pool embeddings all layers
    POOL_EVERY_N_LAYERS_AND_CONCAT = (
        "pool_every_n_layers_and_concat"  # Pool every n layers and concatenatenate
    )

    def __str__(self) -> str:
        return self.value


class TokenizerConfig(BaseModel):
    chunk_duration: int = 81
    load_mean_std: bool = False
    compile_encode: bool = False
    temporal_window: int = 16


class ConditionerConfig(BaseModel):
    name: str
    type: Literal["remap_key", "text_attr", "boolean_flag"] = "remap_key"
    output_key: str = None
    input_key: Union[str, List[str]] = None
    dropout_rate: float = 0.0


class SacConfig(BaseModel):
    mode: str = "block_wise"
    every_n_blocks: int = 1


class NetConfig(BaseModel):
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    concat_padding_mask: bool = True
    pos_emb_cls: str = "rope3d"
    pos_emb_learnable: bool = True
    pos_emb_interpolation: str = "crop"
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 256
    atten_backend: str = "minimal_a2a"
    extra_per_block_abs_pos_emb: bool = False
    rope_h_extrapolation_ratio: float = 3.0
    rope_w_extrapolation_ratio: float = 3.0
    rope_t_extrapolation_ratio: float = 1.0
    sac_config: SacConfig = Field(default_factory=SacConfig)
    rope_enable_fps_modulation: bool = False
    use_crossattn_projection: bool = False
    crossattn_proj_in_channels: int = 100352
    crossattn_emb_channels: int = 1024


class EMAConfig(BaseModel):
    enabled: bool = True
    rate: float = 0.1
    iteration_shift: int = 0


class SDEConfig(BaseModel):
    p_mean: float = math.log(4.0)
    p_std: float = 1.2
    sigma_max: int = 200
    sigma_min: float = 0.01


class AestheticRewardConfig(BaseModel):
    enabled: bool = False
    model_path: str = None


class ImageRewardConfig(BaseModel):
    enabled: bool = False
    model_path: str = "ImageReward-v1.0"


class FakeRewardConfig(BaseModel):
    enabled: bool = False

    # Model-specific parameters
    reward_fn: str = "fake_reward"
    scale: float = 1.0


class RemoteRewardConfig(BaseModel):
    enabled: bool = True
    score_key: str = "overall_reward"
    scale: float = 1.0
    reward_fn: str = "dance_grpo"
    reward_clip_min: float = -5.0
    reward_clip_max: float = 5.0
    token: str = ""  # Will be provided via command line
    enqueue_url: str = ""  # Will be provided via command line
    fetch_url: str = ""  # Will be provided via command line


class RewardConfig(BaseModel):
    ALL_REWARD_MODELS: List[str] = Field(
        default_factory=lambda: [
            "aesthetic",
            "image_reward",
            "remote_reward",
            "fake_reward",
        ]
    )
    adv_clip_max: float = 5.0
    adv_clip_min: float = -5.0
    aesthetic: AestheticRewardConfig = Field(default_factory=AestheticRewardConfig)
    image_reward: ImageRewardConfig = Field(default_factory=ImageRewardConfig)
    remote_reward: RemoteRewardConfig = Field(default_factory=RemoteRewardConfig)
    fake_reward: FakeRewardConfig = Field(default_factory=FakeRewardConfig)


class RLConfig(BaseModel):
    enabled: bool = True

    # Reference model parameters
    update_ref_every_iter: int = 16
    num_rollout: int = Field(default=8, description="Number of rollout group size.")
    sample_steps: int = 10
    train_on: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    on_policy: bool = True

    # Sampler parameters for RL
    solver_option: str = "2ab"
    s_churn: float = 1.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.0
    s_noise: float = 1.0
    guidance: float = 0.0
    min_num_conditional_frames: int = Field(
        default=1,
        description="Minimum number of conditional frames. 0 for t2v, 1 for i2v, 2 for v2v.",
    )
    max_num_conditional_frames: int = Field(
        default=1,
        description="Maximum number of conditional frames. 0 for t2v, 1 for i2v, 2 for v2v.",
    )

    clip_ratio: float = 0.0001
    kl_beta: float = Field(
        default=0.01, description="Coefficient for reverse KL divergence."
    )
    data_beta: float = Field(default=0.0, description="Coefficient for wfm loss.")
    use_rl_sigma_and_noise: bool = Field(
        default=True, description="Whether to use rollout noise."
    )
    data_on_first_only: bool = Field(
        default=False, description="Whether to compute wfm loss once."
    )

    use_same_seed: bool = True
    exp_reward: bool = Field(
        default=False, description="Whether to use exponential reward."
    )

    # Reward config
    reward_config: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="before")
    def preprocess(cls, data: dict) -> dict:
        if "s_t_max" in data and isinstance(data["s_t_max"], str):
            data["s_t_max"] = float(data["s_t_max"])
        if "s_t_min" in data and isinstance(data["s_t_min"], str):
            data["s_t_min"] = float(data["s_t_min"])
        return data


class TextEncoderConfig(BaseModel):
    compute_online: bool = False
    embedding_concat_strategy: str = str(EmbeddingConcatStrategy.MEAN_POOLING)
    n_layers_per_group: int = 5
    ckpt_path: str = "nvidia/Cosmos-Reason1-7B"
    encoder_model_config: QwenModelConfig = Field(default_factory=QwenModelConfig)
    tokenizer_type: str = "Qwen/Qwen2-VL-2B-Instruct"


class ModelConfig(BaseModel):
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    conditioner: List[ConditionerConfig] = Field(default_factory=list)
    text_encoder_class: str = "T5"
    text_encoder_config: Optional[TextEncoderConfig] = None
    net: NetConfig = Field(default_factory=NetConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    sde: SDEConfig = Field(default_factory=SDEConfig)
    fsdp_shard_size: int = 8
    dp_replicate_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "video"
    input_caption_key: str = "ai_caption"
    input_image_key: str = "images"
    loss_reduce: str = "mean"
    loss_scale: float = 10.0
    use_torch_compile: bool = False
    adjust_video_noise: bool = True
    state_ch: int = 16
    state_t: int = 24
    resolution: str = "480"
    scaling: str = "rectified_flow"
    rectified_flow_t_scaling_factor: float = 1.0
    rectified_flow_loss_weight_uniform: bool = True
    resize_online: bool = True
    rl: RLConfig = Field(default_factory=RLConfig)
    min_num_conditional_frames: int = 1
    max_num_conditional_frames: int = 2
    sigma_conditional: float = 0.0001
    conditioning_strategy: str = "frame_replace"
    denoise_replace_gt_frames: bool = True
    high_sigma_strategy: str = "LOGUNIFORM200_100000"
    high_sigma_ratio: float = 0.05
    low_sigma_ratio: float = 0.05
    use_rf_ckpts: bool = Field(
        default=False, description="Whether to use rectified flow ckpts."
    )


class OptimizerConfig(BaseModel):
    optim_type: str = "fusedadam"
    lr: float = 1e-5
    weight_decay: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    master_weights: bool = True
    capturable: bool = True


class SchedulerConfig(BaseModel):
    scheduler_type: str = "lambda_linear"
    verbosity_interval: int = 0
    warm_up_steps: List[int] = Field(default_factory=lambda: [1000])
    cycle_lengths: List[int] = Field(default_factory=lambda: [10000000000000])
    f_start: List[float] = Field(default_factory=lambda: [1.0e-6])
    f_max: List[float] = Field(default_factory=lambda: [1.0])
    f_min: List[float] = Field(default_factory=lambda: [1.0])


class ImageDatasetConfig(BaseModel):
    dataset_name: Optional[str] = None
    is_train: Optional[bool] = None
    resolution: Optional[str] = None
    augmentor_name: Optional[str] = None
    object_store: Optional[str] = None
    detshuffle: Optional[bool] = None
    caption_type: Optional[str] = None
    embedding_type: Optional[str] = None
    dataset_resolution_type: Optional[str] = None
    len_t5: Optional[int] = None
    t5_dim: Optional[int] = None


class VideoDatasetConfig(ImageDatasetConfig):
    num_video_frames: Optional[int] = None
    chunk_size: Optional[int] = None
    min_fps_thres: Optional[int] = None
    max_fps_thres: Optional[int] = None
    long_caption_ratio: Optional[int] = None
    medium_caption_ratio: Optional[int] = None
    short_caption_ratio: Optional[int] = None
    user_caption_ratio: Optional[int] = None
    use_native_fps: Optional[bool] = None
    video_decoder_name: Optional[str] = None


class FunctionConfig(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class BaseDataloaderConfig(BaseModel):
    use_cache: bool = False
    cache_size: Optional[int] = None
    concat_size: Optional[int] = None
    cache_augment_fn: Optional[FunctionConfig] = None
    cache_replay_name: Optional[str] = None
    webdataset: bool = False
    batch_size: int
    shuffle: bool = False
    drop_last: bool = True
    num_workers: int
    pin_memory: bool = True
    prefetch_factor: int = 4


class ImageDataLoaderConfig(BaseDataloaderConfig):
    dataset: ImageDatasetConfig


class VideoDataLoaderConfig(BaseDataloaderConfig):
    dataset: VideoDatasetConfig


class ImageDataLoaderEntry(BaseModel):
    dataloader: ImageDataLoaderConfig = Field(default_factory=ImageDataLoaderConfig)
    ratio: float


class VideoDataLoaderEntry(BaseModel):
    dataloader: VideoDataLoaderConfig = Field(default_factory=VideoDataLoaderConfig)
    ratio: float


class JointDataLoaderConfig(BaseModel):
    type: Literal["web"] = "web"
    image_dataloader: Optional[ImageDataLoaderEntry] = None
    video_dataloader: Optional[VideoDataLoaderEntry] = None


class LocalImageDataLoaderConfig(BaseDataloaderConfig):
    type: Literal["image"] = "image"
    ratio: float
    dataset_dir: str
    image_size: tuple[int, int]
    offline_text_embedding: bool = False
    text_encoder_type: str = "t5_xxl"


class LocalVideoDataLoaderConfig(BaseDataloaderConfig):
    type: Literal["video"] = "video"
    ratio: float
    dataset_dir: str
    num_frames: int
    video_size: tuple[int, int]
    offline_text_embedding: bool = False
    text_encoder_type: str = "t5_xxl"


class LocalDataLoaderConfig(BaseModel):
    type: Literal["local"] = "local"
    image_dataloader: Optional[LocalImageDataLoaderConfig] = None
    video_dataloader: Optional[LocalVideoDataLoaderConfig] = None


class JobConfig(BaseModel):
    project: str = "cosmos_wfm_v2"
    group: str = "official_runs_vid2vid_debug"
    name: str
    wandb_mode: str = "online"
    timestamp: Optional[str] = None

    @property
    def path(self) -> str:
        return (
            f"{self.project}/{self.group}/{self.name}/{self.timestamp}"
            if self.timestamp
            else f"{self.project}/{self.group}/{self.name}"
        )

    @property
    def path_local(self) -> str:
        local_root = os.environ.get("COSMOS_OUTPUT_DIR", "./outputs")
        return f"{local_root}/{self.path}"


class DDPConfig(BaseModel):
    find_unused_parameters: bool = False
    static_graph: bool = True
    broadcast_buffers: bool = True


class CuDNNConfig(BaseModel):
    deterministic: bool = False
    benchmark: bool = True


class StragglerDetectionConfig(BaseModel):
    enabled: bool = True
    report_freq: int = 100
    profile_freq: int = 1
    max_diff: float = 1.5
    raise_error: bool = True
    analyze_forward: bool = True
    analyze_backward: bool = True
    analyze_optimizer: bool = True
    analyze_dataloading: bool = True


class ProfilingConfig(BaseModel):
    enable_profiling: bool = False
    enable_memory_snapshot: bool = False
    save_s3: bool = False
    profile_freq: int = 1
    target_ranks: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    record_shape: bool = False
    profile_memory: bool = False
    with_stack: bool = True
    with_modules: bool = True


class TrainerConfig(BaseModel):
    callbacks: List[FunctionConfig] = Field(default_factory=list)
    use_s3: bool = False
    distributed_parallelism: Literal["ddp", "fsdp"] = "fsdp"
    ddp: DDPConfig = Field(default_factory=DDPConfig)
    cudnn: CuDNNConfig = Field(default_factory=CuDNNConfig)
    seed: int = 0
    grad_scaler_args: Dict[str, bool] = Field(default_factory=dict)
    max_iter: int = 25
    max_val_iter: Optional[int] = None
    logging_iter: int = 2
    run_validation: bool = False
    validation_iter: int = 100
    timeout_period: int = 999999999
    memory_format: str = "preserve_format"
    grad_accum_iter: int = 1
    straggler_detection: StragglerDetectionConfig = Field(
        default_factory=StragglerDetectionConfig
    )
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)


class ModelParallelConfig(BaseModel):
    # TP + CP + PP + DP
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_comm_backend: Optional[str] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    data_parallel_size: int = (
        -1
    )  # -1 means this field will be dynamically determined by the number
    hierarchical_context_parallel_sizes: Optional[Any] = None
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    moe_extended_tp: bool = False
    perform_initialization: bool = True
    use_cpu_initialization: bool = False
    fp16: bool = False
    bf16: bool = False
    params_dtype: str = "torch.float32"
    timers: Optional[Any] = None
    finalize_model_grads_func: Optional[Any] = None
    grad_scale_func: Optional[Any] = None
    no_sync_func: Optional[Any] = None
    grad_sync_func: Optional[Any] = None
    param_sync_func: Optional[Any] = None
    deterministic_mode: bool = False
    enable_autocast: bool = False
    autocast_dtype: str = "torch.float32"
    num_microbatches_with_partial_activation_checkpoints: Optional[Any] = None
    gradient_accumulation_fusion: bool = False
    async_tensor_model_parallel_allreduce: bool = False
    use_te_rng_tracker: bool = False
    tp_comm_overlap: bool = False
    tp_comm_bulk_wgrad: bool = True
    tp_comm_bulk_dgrad: bool = True
    tp_comm_overlap_ag: bool = True
    tp_comm_overlap_rs: bool = True
    tp_comm_overlap_rs_dgrad: bool = False
    tp_comm_split_ag: bool = True
    tp_comm_atomic_ag: bool = False
    tp_comm_split_rs: bool = True
    tp_comm_atomic_rs: bool = False
    cross_entropy_loss_fusion: bool = False
    cross_entropy_fusion_impl: str = "native"
    tp_comm_overlap_disable_qkv: bool = False
    tp_comm_overlap_disable_fc1: bool = False
    tp_comm_bootstrap_backend: str = "nccl"
    pipeline_dtype: Optional[Any] = None
    variable_seq_lengths: bool = False
    overlap_p2p_comm: bool = False
    batch_p2p_comm: bool = True
    batch_p2p_sync: bool = True
    use_ring_exchange_p2p: bool = False
    deallocate_pipeline_outputs: bool = False
    defer_embedding_wgrad_compute: bool = False
    wgrad_deferral_limit: int = 0
    pipeline_model_parallel_split_rank: Optional[Any] = None
    overlap_p2p_comm_warmup_flush: bool = False
    microbatch_group_size_per_vp_stage: int = 1
    cpu_offloading: bool = False
    cpu_offloading_num_layers: int = 0
    _cpu_offloading_context: Optional[Any] = None
    cpu_offloading_activations: bool = True
    cpu_offloading_weights: bool = True
    barrier_with_L1_time: bool = True


class ObjectStoreConfig(BaseModel):
    enabled: bool = False
    credentials: str = ""
    bucket: str = ""


class JITConfig(BaseModel):
    enabled: bool = False
    input_shape: Optional[Any] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    strict: bool = True


class CheckpointConfig(BaseModel):
    type: Optional[str] = None
    dcp_async_mode_enabled: bool = False
    save_to_object_store: ObjectStoreConfig = Field(default_factory=ObjectStoreConfig)
    save_iter: int = 1000
    load_from_object_store: ObjectStoreConfig = Field(default_factory=ObjectStoreConfig)
    load_path: str = ""
    model_id: Optional[str] = Field(
        default=None, description="Huggingface model ID to load from"
    )
    load_training_state: bool = False
    only_load_scheduler_state: bool = False
    strict_resume: bool = False
    jit: JITConfig = Field(default_factory=JITConfig)
    verbose: bool = True
    keys_not_to_resume: List[str] = Field(default_factory=list)
    broadcast_via_filesystem: bool = False
    load_ema_to_reg: bool = False
    dcp_allow_mismatched_size: bool = False


class CosmosVisionGenConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    dataloader_train: Union[JointDataLoaderConfig, LocalDataLoaderConfig] = Field(
        discriminator="type", default_factory=JointDataLoaderConfig
    )
    dataloader_val: Union[JointDataLoaderConfig, LocalDataLoaderConfig] = Field(
        discriminator="type", default_factory=JointDataLoaderConfig
    )
    job: JobConfig = Field(default_factory=JobConfig)
    model_parallel: ModelParallelConfig = Field(default_factory=ModelParallelConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "CosmosVisionGenConfig":
        if "job" in config_data:
            # Set unique timestamp for output directory
            if (
                "timestamp" not in config_data["job"]
                or config_data["job"]["timestamp"] is None
            ):
                config_data["job"]["timestamp"] = datetime.now().strftime(
                    "%Y%m%d%H%M%S"
                )
        config = cls.model_validate(config_data)
        return config
