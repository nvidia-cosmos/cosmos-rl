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

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import core_schema
from datetime import datetime
from typing import Any, Dict, Union, Optional, List, Literal
import os
import json
import hashlib
from cosmos_rl.utils.modelscope import update_config_if_modelscope
from cosmos_rl.utils.logging import logger


def config_hash(config: BaseModel) -> str:
    """
    Compute the hash of a config object
    """
    if isinstance(config, BaseModel):
        return hashlib.md5(json.dumps(config.model_dump()).encode()).hexdigest()
    else:
        return "unhashable"


class CustomJsonSchemaGenerator(GenerateJsonSchema):
    def generate(
        self, schema: core_schema.CoreSchema, mode="serialization"
    ) -> dict[str, Any]:
        json_schema = super().generate(schema, mode)

        if "properties" in json_schema:
            properties = json_schema["properties"]
            filtered_properties = {
                k: v
                for k, v in properties.items()
                if not (isinstance(v, dict) and v.get("hide_in_doc") is True)
            }
            json_schema["properties"] = filtered_properties

        # Remove 'hide_in_doc' from all the sub-models
        if "$defs" in json_schema:
            defs = json_schema["$defs"]
            for model_def in defs:
                filtered_sub_properties = {
                    k: v
                    for k, v in defs[model_def].get("properties", {}).items()
                    if not (isinstance(v, dict) and v.get("hide_in_doc") is True)
                }
                json_schema["$defs"][model_def]["properties"] = filtered_sub_properties

        return json_schema


class DatasetConfig(BaseModel):
    name: str = Field(
        default="",
        description="Huggingface dataset name or local path to parquet file",
    )

    subset: Optional[str] = Field(
        default="",
        description="Dataset subset if exists",
    )

    revision: Optional[str] = Field(
        default="",
        description={
            "help": "Dataset git revision if exist, can be a branch name, a tag, or a commit hash."
        },
    )

    split: Union[str, List[str]] = Field(
        default="",
        description="A list of dataset splits to train",
    )

    test_size: Optional[Union[float, int]] = Field(
        default=None,
        description="Size of the test set. If float, it is the ratio (between 0.0 and 1.0) of the dataset; if int, it is the absolute size of the test set.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if isinstance(self.split, str):
            self.split = [self.split]
        return self


class SFTDataConfig(BaseModel):
    type: Literal["sft"]

    trainer_type: str = Field(
        default="sft",
        description="Type of the trainer for SFT.",
    )

    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Dataset configuration for SFT training. It includes dataset name, subset, revision, train split, and test split.",
    )

    mini_batch: int = Field(
        default=2,
        description="mini-batch size for training.",
    )

    dataloader_shuffle: bool = Field(
        default=True,
        description="Shuffle the dataloader. If False, the dataloader will be used in the order it is loaded.",
    )
    enable_dataset_cache: bool = Field(
        default=False,
        description="Enable dataset cache process results, maybe accelerate the dataset loading",
    )
    dataloader_num_workers: int = Field(
        default=0, description="Number of subprocess to use for data loading"
    )
    dataloader_prefetch_factor: Optional[int] = Field(
        default=None,
        description="Number of batches loaded in advance by each worker.",
    )
    conversation_column_name: str = Field(
        default="conversations",  # "conversation",
        description="Column name for formated conversation json",
    )
    system_prompt: str = Field(
        default="",
        description="System prompt for the model, which will be prepended to the prompt",
    )

    balance_dp_token: bool = Field(
        default=True,
        description="Whether to balance the number of tokens in each data parallel replica when calculating the loss.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if self.dataloader_num_workers <= 0:
            self.dataloader_prefetch_factor = None
            self.dataloader_num_workers = 0
        return self


class CheckpointConfig(BaseModel):
    enable_checkpoint: bool = Field(
        default=False,
        description="Enable checkpointing for training. If set to False, no checkpoint will be saved.",
    )

    save_freq: int = Field(
        default=20, description="Checkpoint save frequency for training steps"
    )
    save_freq_in_epoch: int = Field(
        default=0,
        description="Checkpoint save frequency for training epochs. Default to 0 (disabled).",
    )
    save_mode: str = Field(
        default="async",
        description="Checkpoint save mode for training steps",
        choices=["async", "sync"],
    )
    max_keep: int = Field(
        default=5,
        description="Maximum number of checkpoints to keep. If set to -1, all checkpoints will be kept.",
    )
    export_safetensors: bool = Field(
        default=True,
        description="Whether to export a safetensors weight for huggingface usage, include related config files.",
    )
    upload_hf: bool = Field(
        default=False,
        description="Whether to upload the safetensors weight to huggingface.",
    )
    hf_repo_name: str = Field(
        default="Comos-Reason1",
        description="The huggingface repo name to upload the safetensors weight.",
    )
    upload_s3: Union[bool, str] = Field(
        default=False,
        description="Whether to upload the checkpoint and safetensors to S3. Default to False, set `final` will upload the final checkpoint, `all` will upload all checkpoints.",
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="The S3 bucket name to upload the checkpoint and safetensors weight.",
    )
    s3_prefix: str = Field(
        default="outputs",
        description="The S3 prefix to upload the checkpoint and safetensors weight.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if self.upload_s3:
            if self.upload_s3 not in ["final", "all"]:
                raise ValueError(
                    "upload_s3 must be one of ['final', 'all'] or False, got {}".format(
                        self.upload_s3
                    )
                )
            if self.s3_bucket is None:
                raise ValueError(
                    "s3_bucket must be specified when upload_s3 is True, got None"
                )
        if self.save_mode not in ["async", "sync"]:
            raise ValueError(
                f"Invalid save_mode: {self.save_mode}. Must be one of ['async', 'sync']"
            )
        if self.save_freq_in_epoch <= 0 and self.save_freq <= 0:
            raise ValueError(
                f"save_freq must be greater than 0 when save_freq_in_epoch disabled, got {self.save_freq}"
            )
        return self


class OverlongRewardConfig(BaseModel):
    enable_overlong_penalty: bool = Field(
        default=False,
        description="Enable overlong penalty for the model. If set to True, the output will be penalized for responses that are too long.",
    )
    buffer_length: int = Field(
        default=4096,
        description="Length of the buffer for overlong penalty. If the response length exceeds this value, the output will be penalized.",
    )
    penalty_factor: float = Field(
        default=1.0,
        description="Penalty factor for overlong penalty. The penalty increases linearly with the length of the response exceeding the buffer length from 0 to the penalty_factor.",
    )


class GrpoConfig(BaseModel):
    type: Literal["grpo"]

    trainer_type: str = Field(
        default="grpo",
        description="Type of the trainer for GRPO.",
    )

    variant: str = Field(
        default="grpo",
        description="Variant of the GRPO, currently support `grpo`, `gspo`, `dapo`",
        choices=["grpo", "gspo", "dapo"],
    )

    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Dataset configuration for GRPO training. It includes dataset name, subset, revision, train split, test split and test size.",
    )

    dataloader_shuffle: bool = Field(
        default=True,
        description="Shuffle the dataloader. If False, the dataloader will be used in the order it is loaded.",
    )
    enable_dataset_cache: bool = Field(
        default=False,
        description="Enable dataset cache process results, maybe accelerate the dataset loading",
    )
    dataloader_num_workers: int = Field(
        default=0, description="Number of subprocess to use for data loading"
    )
    dataloader_prefetch_factor: Optional[int] = Field(
        default=None,
        description="Number of batches loaded in advance by each worker.",
    )
    dataloader_batch_size: Optional[int] = Field(
        default=1,
        description="Batch size for each iteration of the dataloader for when fetch prompts from controller. This is only the setting of the dataloader iterator on the controller side.",
    )
    prompt_column_name: str = Field(
        default="",
        description="Column name for prompt",
    )
    response_column_name: str = Field(
        default="",
        description="Column name for response/reference answer",
    )
    reward_function: Union[str, List[str], Dict[str, float]] = Field(
        default_factory=lambda: ["single_choice"],
        description="Reward functions for the model. Currently support `single_choice`, `boxed_math`, and `format`. You can add weight to each reward function by passing a dict, e.g., {'single_choice': 0.9, 'format': 0.1}",
    )
    filter_reward_metric: Union[str, List[str]] = Field(
        default_factory=list,
        description="Reward function to filter in dynamic sampling for DAPO. If specified, only samples with different this rewards will be used for training. If None, no filtering will be applied.",
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for sampling. The higher the temperature, the more random the completions.",
    )

    epsilon_low: float = Field(
        default=0.2,
        description="Epsilon value for clipping.",
    )

    epsilon_high: float = Field(
        default=0.2,
        description="Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
        "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`.",
    )

    positive_nll_coef: Optional[float] = Field(
        default=None,
        description=(
            "[Optional] Coefficient for Positive Example LM Loss. Set a positive value to enable; None disables.\n"
            "Ref: VAPO Sec. 4.3 (Positive Example LM Loss): https://arxiv.org/pdf/2504.05118"
        ),
    )

    lower_bound_ratio: float = Field(
        default=3.0,
        description="Lower-bound ratio for dual-clip.",
    )

    loss_type: str = Field(
        default="token-mean",
        description="The type of loss to use for GRPO training.",
        choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"],
    )

    unbiased_loss_max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to use for unbiased loss introduced in Dr.GRPO. If set to None, will not use unbiased loss."
        "Only available when `loss_type` is `seq-mean-token-mean`",
    )

    unbiased_advantage: bool = Field(
        default=False,
        description="Whether to divide the advantage by the standard deviation of rewards.",
    )

    overlong_reward: OverlongRewardConfig = Field(
        default_factory=OverlongRewardConfig,
        description="Configuration for overlong reward penalty. If enabled, the output will be penalized for responses that are too long.",
    )

    kl_beta: float = Field(
        default=0.0,
        description="KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
        "training speed, but may be numerically unstable for long training runs.",
    )

    unbiased_kl_estimate: bool = Field(
        default=False,
        description=(
            "[Optional] Unbiased K3 with IS: D_KL ≈ E_{π_old}[ w · ( r − log r − 1 ) ], w=π_θ/π_old, r=π_ref/π_θ.\n"
            "Note: This option is ignored when `kl_beta` is 0.0.\n"
            "Ref: DeepSeek-V3.2 Sec.3.1 (Unbiased KL Estimate): https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf"
        ),
    )

    off_policy_masking_delta: Optional[float] = Field(
        default=None,
        description=(
            "Off-Policy Sequence Masking threshold δ (None disables). "
            "Per-sequence mask:"
            " M_i = 0 if Â_i < 0 and (1/|o_i|)∑_t log[π_old(o_{i,t}|·)/π_θ(o_{i,t}|·)] > δ; "
            "else M_i = 1."
            "Ref: DeepSeek-V3.2 Sec.3.1 Off-Policy Sequence Masking : https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf."
        ),
    )

    aipo_rho: Optional[float] = Field(
        default=None,
        description="Rho value for AIPO (Asynchronous Importance weighted Policy Optimization). The clipping constant of the importance sampling ratio, suggest [2,10]. "
        "reference: https://arxiv.org/pdf/2505.24034",
    )

    mu_iterations: int = Field(
        default=1,
        description="Number of iterations per batch (denoted as μ in the algorithm).",
    )

    mini_batch: int = Field(
        default=2,
        description="mini-batch size for GRPO training. Mini-batch is used to split the batch per optimization into smaller batches to fit into GPU memory.",
    )

    batch_size_per_optimize: Optional[int] = Field(
        default=None,
        description="batch size for each optimization in GRPO training. The batch in each training step is split into smaller batches which each performs one step optimization. If not set, it will be the same as the whole batch size per GPU for each training step.",
    )

    max_token_len_per_mini_batch: Optional[int] = Field(
        default=None,
        description="Maximum token length per mini batch. If set, dynamic mini-batch sizing will be applied based on this limit.",
    )

    entropy_coeff: float = Field(
        default=0.0,
        description="Coefficient for entropy regularization.",
    )

    allowed_outdated_steps: int = Field(
        default=4,
        description="Allowed outdated-async steps for rollout engine. "
        "If the number of left uncompleted rollout samples is larger than the `(allowed_outdated_steps + 1) * n_policy_replicas * train_batch_per_replica`, "
        "then rollout engine traffic will be throttled. ",
    )

    on_policy: bool = Field(
        default=False,
        description="Enable fully synchronized (on-policy) rollout. If set to True, the rollout engine will wait until the expected weight version is updated before next generation starts.",
    )

    outdated_rollout_fetch_batch_size: int = Field(
        default=0,
        description="Number of outdated rollouts to fetch. If set to 0, the rollout engine will stop generating rollouts if the weight is outdated.",
    )

    min_filter_prefix_tokens: Optional[int] = Field(
        default=None,
        description="Minimum number of tokens to filter the prefix tokens for the rollouts inside the same group. "
        "If the number of tokens is larger than the `min_filter_prefix_tokens`, the rollouts with the same prefix but different rewards will be filtered out in loss calculation.",
    )

    max_retry_for_on_policy: int = Field(
        default=10,
        description="Maximum number of retries for on-policy rollout to have enough samples. If non-positive, will retry with no upper limit until enough samples are generated.",
    )

    reference_reset_interval: Optional[int] = Field(
        default=None,
        description="Interval to reset the reference model to the current model. If set to None or 0, the reference model will not be reset during training.",
    )

    reset_optimizer_with_reference: bool = Field(
        default=True,
        description="Whether to reset the optimizer state when the reference model is reset.",
    )

    balance_dp_token: bool = Field(
        default=False,
        description="Whether to balance the number of tokens in each data parallel replica when calculating the loss.",
    )

    # Refer to the decoupled objective concept from the AREAL paper: https://arxiv.org/abs/2505.24298.
    use_decoupled_loss: bool = Field(
        default=False,
        description="Whether to use decoupled loss. A decoupled loss separates the optimization of the behavior policy and the target policy, which can help to reduce the variance of the gradient estimate.",
    )

    # Related to the above decoupled loss to cap the behavior importance weights.
    behav_imp_weight_cap: Optional[float] = Field(
        default=None,
        description="Clipping cap for behavior importance weights. Useful when decoupled loss is used to avoid large variance.",
    )

    rollout_as_token_ids: bool = Field(
        default=False,
        description="Whether to use token ids for rollouts instead of text. This can save tokenization time during rollout generation.",
    )

    collect_rollout_logprobs: bool = Field(
        default=False,
        description="Whether to collect logprobs for rollouts instead of text. This can save logprob calculation time during rollout generation.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        assert self.variant in [
            "grpo",
            "dapo",
            "gspo",
        ], "variant must be one of ['grpo', 'dapo', 'gspo']"
        if self.dataloader_num_workers <= 0:
            self.dataloader_prefetch_factor = None
            self.dataloader_num_workers = 0
        if isinstance(self.reward_function, str):
            self.reward_function = {self.reward_function: 1.0}
        elif isinstance(self.reward_function, list):
            self.reward_function = {k: 1.0 for k in self.reward_function}
        assert (
            len(self.reward_function) > 0
        ), "reward_function must be a dict of reward functions"
        if isinstance(self.filter_reward_metric, str):
            self.filter_reward_metric = [self.filter_reward_metric]
        if self.dataloader_batch_size is not None and self.dataloader_batch_size <= 0:
            logger.warning(
                "dataloader_batch_size is not positive so disable it as None."
            )
            self.dataloader_batch_size = None
        if self.use_decoupled_loss:
            self.rollout_as_token_ids = True
            self.collect_rollout_logprobs = True
            logger.warning(
                "Decoupled loss is enabled, so rollout_as_token_ids is set to True."
            )

        return self


class SubProfilerConfig(BaseModel):
    do_profile: bool = Field(
        default=False, description="Whether to profile, only used in runtime."
    )
    active_steps: int = Field(default=1, description="Number of active steps")
    warmup_steps: int = Field(default=1, description="Number of warmup steps")
    wait_steps: int = Field(default=1, description="Number of wait steps")
    rank_filter: List[int] = Field(default_factory=list, description="Rank filter")
    record_shape: bool = Field(default=False, description="Whether to record shape")
    profile_memory: bool = Field(default=False, description="Whether to profile memory")
    with_stack: bool = Field(default=False, description="Whether to profile stack")
    with_modules: bool = Field(default=False, description="Whether to profile modules")


class ProfilerConfig(BaseModel):
    enable_profiler: bool = Field(
        default=False,
        description="Enable profiler for training",
    )
    enable_nsys: bool = Field(
        default=False,
        description="Enable nsys for training",
    )
    sub_profiler_config: SubProfilerConfig = Field(
        default_factory=SubProfilerConfig, description="Sub profiler config"
    )


class FP8Config(BaseModel):
    enable_fp8: bool = Field(default=False, description="Whether to enable fp8.")
    fp8_recipe: str = Field(
        default="dynamic_scaling",
        description="Recipe for weight scale calculation.",
        choices=["dynamic_scaling", "delayed_scaling"],
    )
    quant_recipe: str = Field(
        default="rowwise",
        description="Quantization strategy for weight.",
        choices=["rowwise", "tensorwise"],
    )


class FP4Config(BaseModel):
    enable_fp4: bool = Field(default=False, description="Whether to enable fp4.")
    fp4_recipe: str = Field(
        default="dynamic_scaling",
        description="Recipe for weight scale calculation.",
        choices=["dynamic_scaling", "delayed_scaling"],
    )
    quant_recipe: str = Field(
        default="rowwise",
        description="Quantization strategy for weight.",
        choices=["rowwise", "tensorwise"],
    )


class TrainingConfig(BaseModel):
    train_policy: Union[SFTDataConfig, GrpoConfig] = Field(
        discriminator="type", default=GrpoConfig(type="grpo")
    )

    # --------- Optimizer ---------

    optm_name: str = Field(
        default="AdamW",
        description="Optimizer name",
        choices=["AdamW", "Adam"],
    )
    optm_lr: Union[float, List[float]] = Field(
        default=1e-6,
        description="Learning rate for optimizer, can be a float or a list of floats for multiple optimizers",
    )
    optm_impl: Union[str, List[str]] = Field(
        default="fused",
        description="Implementation type for optimizer. More info: https://pytorch.org/docs/stable/optim.html, can be a list of strings for multiple optimizers",
        choices=["fused", "foreach", "for-loop"],
    )
    optm_weight_decay: float = Field(
        default=0.01, description="Weight decay for optimizer"
    )
    optm_betas: tuple[float, float] = Field(
        default=(0.9, 0.999), description="Betas for optimizer"
    )
    optm_warmup_steps: Union[int, float] = Field(
        default=20,
        description="Warmup steps for optimizer, can be an integer or a float, if it is a float and range in [0.0, 1.0], it will be multiplied by the total steps",
    )
    optm_decay_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of total steps for decay, range in [0.0, 1.0], 0 means no decay.",
    )
    optm_decay_type: Optional[str] = Field(
        default=None,
        description="Type of decay for optimizer",
        choices=["sqrt", "cosine", "linear", "none"],
    )
    optm_min_lr_factor: float = Field(
        default=0.0, description="Minimum lr factor for optimizer, range in [0.0, 1.0]"
    )
    optm_grad_norm_clip: float = Field(
        default=1.0, description="Gradient norm clip for optimizer"
    )

    # --------- FSDP ---------

    master_dtype: str = Field(
        default="float32",
        description="The master weight data type for optimizers, is orthognal to `param_dtype`. Should be high precision for convergence consideration",
        choices=["bfloat16", "float16", "float32"],
    )
    param_dtype: str = Field(
        default="bfloat16",
        description="The data type for forward/backward. Outside forward/backward, params are in `master_dtype`",
        choices=["bfloat16", "float16", "float32"],
    )
    transfer_dtype: str = Field(
        default=None,
        description="The data type for transfer parameters between Policy and Rollout.",
        choices=["bfloat16", "float16", "float32"],
    )
    logprob_dtype: str = Field(
        default="float32",
        description="The data type for logprobs calculation.",
        choices=["bfloat16", "float16", "float32"],
    )

    fsdp_reduce_dtype: str = Field(
        default="float32",
        description="The data type for reduction in FSDP",
        choices=["float32"],
    )
    fsdp_offload: bool = Field(
        default=False,
        description="Whether to offload the model to CPU if using FSDP",
    )

    fsdp_reshard_after_forward: str = Field(
        default="default",
        description="Reshard the param after forward pass in FSDP",
        choices=["always", "never", "default"],
    )

    train_batch_per_replica: int = Field(
        default=8,
        description="The batch size for training per iteration in one replica, this is the local batch size for each gradient accumulation step",
    )

    # --------- Engineering ---------

    fp8: FP8Config = Field(default_factory=FP8Config)
    fp4: FP4Config = Field(default_factory=FP4Config)
    ckpt: CheckpointConfig = Field(default_factory=CheckpointConfig)
    resume: Union[bool, str] = Field(
        default=False,
        description="Resume training from a checkpoint. If True, will resume from the latest checkpoint of the `output_dir`. If a string, will resume from the specified checkpoint path.",
    )
    epoch: int = Field(default=1, description="Number of epochs for training")
    output_dir: str = Field(default="./outputs", description="Output directory")
    timestamp: str = Field(
        default="",
        description="Timestamp for the output directory and wandb ID, if not set, will be generated automatically",
    )
    epsilon: float = Field(default=1e-6, description="Epsilon for optimizer")
    async_tp_enabled: bool = Field(
        default=False, description="Whether to use async tensor parallelism"
    )
    compile: bool = Field(default=True, description="Whether to use torch.compile")
    sync_weight_interval: int = Field(
        default=1,
        description="The interval of train step for synchronizing weights between replicas.",
    )
    deterministic: bool = Field(
        default=False,
        description="Whether to use deterministic training. If set to True, will use deterministic training, which is expected to be slower.",
    )
    activation_offload: bool = Field(
        default=False,
        description="Whether to use activation offload",
    )
    fa_version: Optional[int] = Field(
        default=None,
        description="FlashAttention version to use. If None, will use the default version.",
        choices=[2, 3],
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for training. If deterministic is set to True, will by default be set to 42.",
    )

    local_dataset: Optional[bool] = Field(
        default=True,
        description="Whether to use local dataset to query sample. If set to True, will use the local dataset.",
    )

    force_use_hf: Optional[bool] = Field(
        default=False,
        description="Whether to force using Huggingface dataset even if local dataset is available.",
    )

    non_text: bool = Field(
        default=False,
        description="Whether train in non-text mode. If set to True, the inputs and outputs are not pure text, but may contain other modalities like images, videos, tensors, etc.",
    )

    # --------- smoke-test helpers ---------

    max_num_steps: Optional[int] = Field(
        default=None,
        description="Optional upper bound on total training steps. If set, training stops when either this step count or the epoch-based limit is reached (whichever comes first). Handy for quick smoke tests.",
    )

    sequence_packing: bool = Field(
        default=False,
        description="Whether to enable sequence packing for training. If set to True, the input sequences will be packed into a single tensor for training.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if self.async_tp_enabled and not self.compile:
            raise ValueError(
                "Async tensor parallelism requires torch.compile to be enabled"
            )
        if self.max_num_steps is not None and self.max_num_steps <= 0:
            raise ValueError("max_num_steps must be positive if specified")

        if isinstance(self.train_policy, GrpoConfig):
            if self.train_policy.on_policy:
                assert (
                    self.sync_weight_interval == 1
                ), "sync_weight_interval must be 1 when on_policy is enabled"

        if self.deterministic and self.seed is None:
            self.seed = 42

        return self


class ParallelismConfig(BaseModel):
    n_init_replicas: int = Field(
        default=1,
        description="Number of initial replicas to be created",
    )
    tp_size: int = Field(default=2, description="Tensor parallelism size")
    cp_size: int = Field(default=1, description="Context parallelism size")
    ep_size: int = Field(default=1, description="Expert parallelism size")
    dp_shard_size: int = Field(
        default=1, description="Data Parallelism size in sharded mode"
    )
    pp_size: int = Field(default=1, description="Pipeline parallelism size")
    pp_dynamic_shape: bool = Field(
        default=False, description="Pipeline parallelism dynamic shape"
    )
    pp_micro_batch_size: int = Field(
        default=1,
        description="Pipeline parallelism micro batch size, `n_micro_batch = batch_size / pp_micro_batch_size`, which must be divisible by `pp` stages",
    )
    dp_replicate_size: int = Field(
        default=1,
        description="Data Parallelism size in replica mode. Only configurable in SFT type job, must be 1 in GRPO type job for dynamic scaling support purpose.",
        choices=[1],
    )

    @property
    def world_size(self):
        world_size = os.environ.get("WORLD_SIZE", 1)
        return int(world_size)

    @property
    def local_world_size(self):
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE", 1)
        return int(local_world_size)


class RolloutParallelismConfig(ParallelismConfig):
    pass


class LoraConfig(BaseModel):
    r: int = Field(default=8, description="LoRA rank")
    lora_alpha: float = Field(default=8.0, description="LoRA alpha")
    lora_dropout: float = Field(default=0.0, description="LoRA dropout")
    target_modules: Union[List[str], str] = Field(
        default=None,
        description="LoRA target modules, can be a list of strings or 'all-linear'",
    )
    use_rslora: bool = Field(
        default=False,
        description="When set to True, uses [Rank-Stabilized LoRA](https://huggingface.co/papers/2312.03732)"
        " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
        " was proven to work better. Otherwise, it will use the original default"
        " value of `lora_alpha/r`.",
    )
    modules_to_save: Optional[List[str]] = Field(
        default=None,
        description="List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. ",
    )
    alpha_pattern: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-module overrides for lora_alpha. Keys are regex patterns; evaluated in insertion order, first match wins.",
    )
    r_pattern: Optional[Dict[str, int]] = Field(
        default=None,
        description="Per-module overrides for LoRA rank r. Keys are regex patterns; evaluated in insertion order, first match wins.",
    )
    init_lora_weights: Union[
        bool,
        Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]"],
    ] = Field(
        default=True,
        description="How to initialize the weights of the adapter layers."
        "Passing True (default) results in the default initialization from the reference implementation from Microsoft, with the LoRA B weight being set to 0. "
        "This means that without further training, the LoRA adapter will be a no-op."
        "Setting the initialization to False leads to random initialization of LoRA A and B, meaning that LoRA is not a no-op before training; this setting is intended for debugging purposes."
        "Passing ‘gaussian’ results in Gaussian initialization scaled by the LoRA rank for linear and layers. Pass 'loftq' to use LoftQ initialization. Passing 'eva' results in a data-driven initialization of Explained Variance Adaptation."
        "EVA initializes LoRA based on the SVD of layer input activations and achieves SOTA performance due to its ability to adapt to the finetuning data. Pass 'olora' to use OLoRA initialization. Passing 'pissa' results in the initialization of https://huggingface.co/papers/2404.02948",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if isinstance(self.target_modules, str):
            assert (
                self.target_modules == "all-linear"
            ), "target_modules must be a list of strings or 'all-linear'"
        return self


class DiffusersConfig(BaseModel):
    is_video: bool = Field(
        default=False, description="True if this model is video generate model"
    )
    max_prompt_length: int = Field(
        default=300, description="Maximum sequence length to use for the prompt"
    )
    weighting_scheme: str = Field(
        default="logit_normal", description="Method used to sample timestep"
    )
    train_flow_shift: float = Field(
        default=3.0, description="flow shift used for training"
    )
    offload: bool = Field(
        default=True,
        description="Whether to dynamic offload model parts from cuda to cpu",
    )
    logit_mean: float = Field(
        default=0.0,
        description="random sampling timestep logits mean for noise addition",
    )
    logit_std: float = Field(
        default=1.0,
        description="random sampling timestep logits std for noise addition",
    )
    inference_size: List[int] = Field(
        default=[1024, 1024],
        description="Image/video size for generation, [height, width]",
    )
    inference_frames: int = Field(
        default=41, description="Total frame of video size for generation"
    )
    train_frames: int = Field(
        default=41, description="Total frame of video size for training"
    )
    inference_step: int = Field(
        default=50, description="Total denoise step used for validation generation"
    )
    guidance_scale: float = Field(
        default=4.5, description="CFG guidance scale for validation generation"
    )


class PolicyConfig(BaseModel):
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)

    diffusers_config: Optional[DiffusersConfig] = Field(default_factory=DiffusersConfig)

    is_diffusers: bool = Field(
        default=False, description="Whether this model is diffusers or not"
    )

    model_name_or_path: str = Field(
        # default="Qwen/Qwen2.5-3B-Instruct",  #'Qwen/Qwen2.5-VL-7B-Instruct'
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="The model name or path, compatible with huggingface model name or local path",
    )

    model_revision: Optional[str] = Field(
        default=None,
        description="The revision of the model to use",
    )

    model_max_length: int = Field(
        default=4096,
        description="The maximum length for training, longer than this will be ignored for training stability",
    )

    model_gradient_checkpointing: bool = Field(
        default=True, description="Whether to use gradient checkpointing"
    )

    lora: LoraConfig | None = Field(default=None, description="LoRA configuration")
    trainable_map: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Mapping of name -> bool. Keys can either be: "
        "- exact parameter names (from model.named_parameters()) "
        "- exact module paths (from model.named_modules()) ",
    )
    freeze_pattern: Optional[List[str]] = Field(
        default=None,
        description="Pattern-based configuration to freeze parts of the model. "
        "A list of regex patterns that match against parameter names; "
        "matched parameters will be frozen (requires_grad=False). "
        "Example: freeze_pattern = ['^visual\\..*'] freezes all visual components; "
        "freeze_pattern = ['^model\\.layers\\.[0-9]+\\.'] freezes layers 0-9.",
    )
    enable_liger_kernel: bool = Field(
        default=False, description="Whether to use liger kernel."
    )

    @model_validator(mode="after")
    def check_params_value(self):
        assert (
            self.model_name_or_path is not None and self.model_name_or_path != ""
        ), "model_name_or_path is required"
        assert self.parallelism.tp_size > 0, "tp_size must be greater than 0"
        assert self.parallelism.ep_size > 0, "ep_size must be greater than 0"
        assert self.parallelism.cp_size > 0, "cp_size must be greater than 0"
        assert self.parallelism.pp_size > 0, "pp_size must be greater than 0"
        assert (
            self.parallelism.dp_shard_size >= -1 and self.parallelism.dp_shard_size != 0
        ), "dp_shard_size must be greater than 0 or -1 to be auto-inferred"
        return self


class SamplingConfig(BaseModel):
    temperature: float = Field(default=1.0, description="Temperature for sampling.")
    top_p: float = Field(default=1.0, description="Top-p for sampling.")
    top_k: int = Field(default=-1, description="Top-k for sampling.")
    repetition_penalty: float = Field(
        default=1.0, description="Repetition penalty for sampling."
    )
    use_flashinfer: bool = Field(
        default=False, description="Use flashinfer for sampling."
    )


class MultiTurnRolloutConfig(BaseModel):
    enable: bool = Field(
        default=False, description="Whether to enable multi-turn rollout."
    )
    enable_tools: bool = Field(
        default=False, description="Whether to enable tools in multi-turn rollout."
    )
    enable_thinking: bool = Field(
        default=False, description="Whether to enable thinking in multi-turn rollout."
    )
    custom_chat_template_path: Optional[str] = Field(
        default=None, description="The path to the custom chat template in chat."
    )
    max_assistant_turns: int = Field(
        default=5, description="Max assistant turn count for multi-turn rollout."
    )
    add_generation_prompt: bool = Field(
        default=True,
        description="Whether to add generation prompt in multi-turn rollout.",
    )
    continue_final_message: bool = Field(
        default=False,
        description="Whether to continue the final message in multi-turn rollout.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if self.enable_tools:
            if self.add_generation_prompt:
                assert not self.continue_final_message, "continue_final_message must be False when add_generation_prompt is True"
        return self


class ValidationConfig(BaseModel):
    enable: bool = Field(
        default=False,
        description="Enable validation during training.",
    )
    val_before_train: bool = Field(
        default=False,
        description="Enable validation before training starts (at step 0, after weight initialization).",
    )
    freq: int = Field(
        default=20,
        description="Validation frequency during training, in terms of training steps",
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size for validation, will use the same batch size as training if not set.",
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Dataset configuration for validation. It includes dataset name, subset, revision and test split.",
    )

    temperature: float = Field(
        default=0.0, description="Temperature for sampling during validation."
    )
    top_p: Optional[float] = Field(
        default=None, description="Top-p for sampling during validation."
    )
    top_k: Optional[int] = Field(
        default=1, description="Top-k for sampling during validation."
    )
    repetition_penalty: float = Field(
        default=1.0, description="Repetition penalty for sampling during validation."
    )
    n_generation: int = Field(
        default=1,
        description="n parameter same like what in OpenAI chat API for validation.",
    )
    max_response_length: Optional[int] = Field(
        default=None,
        description="Max output length of rollout generation during validation.",
    )
    reward_function: Union[str, List[str], Dict[str, float]] = Field(
        default=[],
        description="Reward functions for the model. Currently support `single_choice`, `boxed_math`, and `format`. You can add weight to each reward function by passing a dict, e.g., {'single_choice': 0.9, 'format': 0.1}",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if isinstance(self.reward_function, str):
            self.reward_function = {self.reward_function: 1.0}
        elif isinstance(self.reward_function, list):
            self.reward_function = {k: 1.0 for k in self.reward_function}
        assert isinstance(
            self.reward_function, dict
        ), "reward_function must be a dict of reward functions"
        return self


class RolloutConfig(BaseModel):
    parallelism: RolloutParallelismConfig = Field(
        default_factory=RolloutParallelismConfig
    )
    enforce_eager: bool = Field(
        default=True, description="Whether to enable eager execution for vLLM."
    )
    include_stop_str_in_output: bool = Field(
        default=False, description="Whether to include stop string in output."
    )
    gpu_memory_utilization: float = Field(
        default=0.8,
        description="GPU memory utilization factor for rollout backend.",
    )
    enable_chunked_prefill: bool = Field(
        default=False, description="Whether to enable chunked prefill for vLLM."
    )
    max_response_length: int = Field(
        default=2048, description="Max output length of rollout generation."
    )
    n_generation: int = Field(
        default=16, description="n parameter same like what in OpenAI chat API."
    )
    n_generation_to_batch: bool = Field(
        default=False,
        description="Whether to treat n_generation as batch dimension in rollout generation.",
    )

    batch_size: int = Field(default=1, description="Batch size for rollout.")

    quantization: str = Field(
        default="none",
        description="Quantization in vllm rollout generation.",
        choices=["none", "fp8"],
    )

    seed: Optional[int] = Field(default=None, description="random seed for rollout.")

    sampling_config: SamplingConfig = Field(default_factory=SamplingConfig)

    vllm_use_flashinfer: bool = Field(
        default=False, description="Use flashinfer for vllm rollout."
    )

    backend: str = Field(
        default="vllm",
        description="Backend for rollout. Currently support `vllm` and `trtllm`, and other custom backends.",
        choices=["vllm", "trtllm"],
    )

    multi_turn_config: MultiTurnRolloutConfig = Field(
        default_factory=MultiTurnRolloutConfig,
        description="Configuration for multi-turn rollout.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if isinstance(self.parallelism, dict):
            self.parallelism = RolloutParallelismConfig(**self.parallelism)

        backends_to_check = ["vllm", "trtllm"]
        if self.backend in backends_to_check:
            _fields_no_need_to_check = ["n_init_replicas", "tp_size", "pp_size"]
            for field_name, field_info in RolloutParallelismConfig.model_fields.items():
                if field_name not in _fields_no_need_to_check:
                    default_value = field_info.default
                    actual_value = getattr(self.parallelism, field_name)
                    if actual_value != default_value:
                        raise ValueError(
                            f"Only {_fields_no_need_to_check} fields can be set for rollout parallelism."
                        )
        return self


class LoggingConfig(BaseModel):
    logger: List[str] = Field(
        default_factory=list,
        description="List of loggers to use, e.g., ['console', 'wandb']",
    )
    project_name: str = Field(
        default="cosmos_rl",
        description="Wandb project name for logging. If set, the training will be logged to this project.",
    )
    group_name: Optional[str] = Field(
        default=None,
        description="Wandb group name for logging. If set, the training will be logged to this group.",
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="A short display name for this run. If not set, will use the `output_dir` as the experiment name.",
    )
    report_mfu: bool = Field(
        default=False,
        description="Whether to report the MFU (Model FLOPs Utilization) to wandb.",
        json_schema_extra={"hide_in_doc": True},
    )

    @model_validator(mode="after")
    def check_params_value(self):
        if self.logger:
            self.logger = [logger.lower() for logger in self.logger]
        return self


class VLAConfig(BaseModel):
    vla_type: str = Field(
        default="openvla-oft",
        description="VLA type, could be 'openvla-oft' or 'openvla'",
        choices=["openvla-oft", "openvla"],
    )

    use_proprio: bool = Field(
        default=False, description="Whether to use proprioceptive information."
    )

    proprio_dim: int = Field(
        default=7, description="Dimension of proprioceptive information."
    )

    num_images_in_input: int = Field(
        default=1, description="Number of images in input."
    )

    training_chunk_size: int = Field(
        default=16, description="Number of chunks to train in one iteration."
    )

    save_video: bool = Field(
        default=False, description="Whether to save video of validation rollout."
    )

    @model_validator(mode="after")
    def check_params_value(self):
        return self


class DistillationConfig(BaseModel):
    enable: bool = Field(default=False, description="Whether to enable distillation.")

    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)

    model_name_or_path: str = Field(
        # default="Qwen/Qwen2.5-3B-Instruct",  #'Qwen/Qwen2.5-VL-7B-Instruct'
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="The teacher model name or path, compatible with huggingface model name or local path",
    )

    model_revision: Optional[str] = Field(
        default=None,
        description="The revision of the teacher model to use",
    )

    compile: bool = Field(
        default=True, description="Whether to use torch.compile for teacher model."
    )
    # --------- FSDP ---------

    master_dtype: str = Field(
        default="float32",
        description="The master weight data type for teacher model, is orthognal to `param_dtype`. Should be high precision for convergence consideration",
        choices=["bfloat16", "float16", "float32"],
    )

    param_dtype: str = Field(
        default="bfloat16",
        description="The data type for forward/backward of teacher model. Outside forward/backward, params are in `master_dtype`",
        choices=["bfloat16", "float16", "float32"],
    )

    logprob_dtype: str = Field(
        default="float32",
        description="The data type for logprobs calculation of teacher model.",
        choices=["bfloat16", "float16", "float32"],
    )

    fsdp_reduce_dtype: str = Field(
        default="float32",
        description="The data type for reduction in FSDP for teacher model.",
        choices=["float32"],
    )

    fsdp_offload: bool = Field(
        default=False,
        description="Whether to offload the teacher model to CPU if using FSDP",
    )

    fsdp_reshard_after_forward: str = Field(
        default="never",
        description="Reshard the param after forward pass in FSDP for teacher model. Default to 'never' to avoid unnecessary overhead.",
        choices=["always", "never", "default"],
    )

    batch_size_per_replica: int = Field(
        default=1, description="Batch size for teacher model per replica."
    )

    max_token_len_per_mini_batch: Optional[int] = Field(
        default=None,
        description="Maximum token length per mini batch. If set, dynamic mini-batch sizing will be applied based on this limit for teacher model.",
    )

    sequence_packing: bool = Field(
        default=False,
        description="Whether to enable sequence packing for teacher model. If set to True, the input sequences will be packed into a single tensor for training stability.",
    )

    mini_batch: int = Field(
        default=2,
        description="mini batch size for teacher model in each replica.",
    )

    seed: Optional[int] = Field(
        default=None, description="Random seed for teacher model."
    )

    kl_penalty_coef: float = Field(
        default=1.0, description="The coefficient for KL penalty."
    )

    kl_discount_factor: float = Field(
        default=0.0, description="The discount factor for KL penalty."
    )

    include_prompt: bool = Field(
        default=False,
        description="Whether to include prompt in the teacher model KL calculation.",
    )

    @model_validator(mode="after")
    def check_params_value(self):
        assert (
            self.model_name_or_path is not None and self.model_name_or_path != ""
        ), "model_name_or_path is required"
        assert self.parallelism.tp_size > 0, "tp_size must be greater than 0"
        assert self.parallelism.ep_size > 0, "ep_size must be greater than 0"
        assert self.parallelism.cp_size > 0, "cp_size must be greater than 0"
        assert self.parallelism.pp_size > 0, "pp_size must be greater than 0"
        assert (
            self.parallelism.dp_shard_size >= -1 and self.parallelism.dp_shard_size != 0
        ), "dp_shard_size must be greater than 0 or -1 to be auto-inferred"
        return self


class Config(BaseModel):
    custom: Dict[str, Any] = Field(
        default_factory=dict, description="Custom script configuration."
    )
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    vla: VLAConfig = Field(default_factory=VLAConfig)
    redis: str = Field(
        default="",
        description="Redis server address port, format: port",
        json_schema_extra={"hide_in_doc": True},
    )
    eth_ips: str = Field(
        default="",
        description="List of eth ip addresses, format: ip1;ip2;ip3",
        json_schema_extra={"hide_in_doc": True},
    )
    mode: str = Field(
        default="disaggregated",
        description="Running mode, could be 'disaggregated' or 'colocated'",
        choices=["disaggregated", "colocated"],
    )

    @classmethod
    def from_dict(cls, config_data: dict[str, Any]) -> "Config":
        if "train" in config_data:
            # Set unique timestamp for output directory
            if (
                "timestamp" not in config_data["train"]
                or config_data["train"]["timestamp"] == ""
            ):
                config_data["train"]["timestamp"] = datetime.now().strftime(
                    "%Y%m%d%H%M%S"
                )
                config_data["train"]["output_dir"] = os.path.join(
                    config_data["train"]["output_dir"],
                    config_data["train"]["timestamp"],
                )
        config = cls.model_validate(config_data)
        config = update_config_if_modelscope(config)
        return config

    @model_validator(mode="before")
    def preprocess(cls, data: dict) -> dict:
        # Handle for train_policy type
        if len(data) == 0:
            # empty data, all fields set to default values.
            return data

        if "train_policy" in data["train"]:
            train_policy_data = data["train"]["train_policy"]

            # Determine the type based on characteristic fields
            if any(
                key in train_policy_data
                for key in ["temperature", "epsilon_low", "epsilon_high", "kl_beta"]
            ):
                data["train"]["train_policy"]["type"] = "grpo"
            else:
                data["train"]["train_policy"]["type"] = "sft"
        return data

    @model_validator(mode="after")
    def check_params_value(self):
        if self.policy.parallelism.pp_size > 1:
            assert (
                self.policy.parallelism.pp_micro_batch_size > 0
            ), "pp_micro_batch_size must be greater than 0"
            assert (
                self.train.train_batch_per_replica
                % self.policy.parallelism.pp_micro_batch_size
                == 0
            ), "train_batch must be divisible by pp_micro_batch_size"

            # Here we assume that PP uses `Single-stage per rank` which is true for:
            #   - GPipe
            #   - 1F1B
            # But not correct for those `InterleavedXXX` style schedule
            assert (
                (
                    self.train.train_batch_per_replica
                    // self.policy.parallelism.pp_micro_batch_size
                )
                % self.policy.parallelism.pp_size
                == 0
            ), "train_batch / pp_micro_batch_size must be divisible by pp_size"

        # Validate constraints for GRPO with LoRA
        if (
            isinstance(self.train.train_policy, GrpoConfig)
            and self.policy.lora is not None
        ):
            # compile must be disabled due to known incompatibilities
            if self.train.compile:
                raise ValueError(
                    "Invalid config: GRPO with LoRA requires train.compile=False."
                )
            # TP must be 1 to avoid unsupported/distributed behaviors
            if self.policy.parallelism.tp_size != 1:
                raise ValueError(
                    "Invalid config: GRPO with LoRA requires policy.parallelism.tp_size == 1."
                )

        # Handle for evaludation configuration.
        if isinstance(self.validation.dataset.split, str):
            self.validation.dataset.split = [self.validation.dataset.split]

        if self.train.transfer_dtype is None:
            # Default use master_dtype as transfer_dtype
            self.train.transfer_dtype = self.train.master_dtype

        if self.distillation.enable:
            self.train.train_policy.rollout_as_token_ids = True
            logger.info(
                "Distillation is enabled, so rollout_as_token_ids is set to True."
            )
        return self


COSMOS_CONFIG_SCHEMA = Config.model_json_schema(
    schema_generator=CustomJsonSchemaGenerator
)
