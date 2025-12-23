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
import collections
import math
import numpy as np
from contextlib import contextmanager
from einops import rearrange
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.distributed._tensor.api import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed import get_process_group_ranks
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config.wfm import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from cosmos_rl.utils.wfm.checkpointer import (
    non_strict_load_model,
)
from cosmos_rl.utils.wfm.utils import (
    timer,
    count_params,
    DataType,
    DenoisePrediction,
    VIDEO_RES_SIZE_INFO,
    IS_PREPROCESSED_KEY,
)
from cosmos_rl.utils.wfm.context_parallel import (
    broadcast,
    broadcast_split_tensor,
)
from cosmos_rl.utils.wfm.distributed import (
    hsdp_device_mesh,
    broadcast_dtensor_model_states,
    get_global_parallel_dims,
)
from cosmos_rl.utils.wfm.ema import (
    FastEmaModelUpdater,
    DTensorFastEmaModelUpdater,
)
from cosmos_rl.utils.wfm.ref import (
    FastRefModelUpdater,
    DTensorFastRefModelUpdater,
)
from cosmos_rl.utils.wfm.optimizer import get_base_optimizer
from cosmos_rl.utils.wfm.lr_scheduler import get_base_scheduler
from cosmos_rl.utils.wfm.torch_future import clip_grad_norm_
from cosmos_rl.policy.model.wfm.rewards import get_reward_model
from cosmos_rl.policy.model.wfm.sampler import (
    Sampler,
    EDMSDE,
    EDMScaling,
    RectifiedFlowScaling,
)
from cosmos_rl.policy.model.wfm.conditioner.condition import T2VCondition
from cosmos_rl.policy.model.wfm.conditioner import Vid2VidConditioner
from cosmos_rl.policy.model.wfm.tokenizer.wan2pt1 import Wan2pt1VAEInterface
from cosmos_rl.policy.model.wfm.networks.model_weights_stats import (
    WeightTrainingStat,
)
from cosmos_rl.policy.model.wfm.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT

# text encoder
from cosmos_rl.policy.model.wfm.networks.vlm_qwen.processor import build_tokenizer
from cosmos_rl.policy.model.wfm.networks.vlm_qwen.qwen_omni import QwenVLBaseModel


class InferenceInfos:
    """
    Container for GRPO inference information.
    """

    def __init__(self):
        self.rl_cached_trajectory: Optional[List] = None
        self.timesteps: int = 0
        self.data_batch: Optional[Dict] = None
        self.rewards: Dict = {}
        self.is_rewards_synced: bool = False
        self.gaussian_noise: Optional[torch.Tensor] = None
        self.final_sample: Optional[torch.Tensor] = None
        self.x0_fn: Optional[Callable] = None


class WorldFoundationalModel(nn.Module):
    """
    World foundational model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        logger.warning(f"WorldFoundationalModel: precision {self.precision}")

        # 1. set data keys and data information
        self.sigma_data = config.sigma_data
        self.setup_data_key()

        # 2. setup up wfm processing and scaling~(pre-condition), sampler
        self.sde = EDMSDE(**config.sde.model_dump())
        self.sampler = Sampler()
        self.scaling = (
            EDMScaling(self.sigma_data)
            if config.scaling == "edm"
            else RectifiedFlowScaling(
                self.sigma_data,
                config.rectified_flow_t_scaling_factor,
                config.rectified_flow_loss_weight_uniform,
            )
        )

        # 3. tokenizer
        with timer("WorldFoundationalModel: set_up_tokenizer"):
            self.tokenizer = Wan2pt1VAEInterface(
                **config.tokenizer.model_dump(),
                s3_credential_path=os.environ.get(
                    "S3_TRAINING_CREDENTIAL_PATH", "credentials/s3_training.secret"
                ),
            )
            assert (
                self.tokenizer.latent_ch == self.config.state_ch
            ), f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        logger.critical(
            f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}"
        )
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 5. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=config.fsdp_shard_size,
            )
        else:
            self.fsdp_device_mesh = None

        # 6. wfm neural networks part
        self.set_up_model()

        # # 7. text encoder
        self.text_encoder = None
        if (
            self.config.text_encoder_config is not None
            and self.config.text_encoder_config.compute_online
        ):
            self.set_up_text_encoder()

        # 8. training states
        self.parallel_dims = get_global_parallel_dims()
        self.data_parallel_size = self.parallel_dims.get_size_in_dim("dp")

        # GRPO trajectory caching variables
        if config.rl.enabled:
            self.inference_infos = InferenceInfos()
            logger.info(
                f"[WFM RL] Initialized with config: num_rollout={config.rl.num_rollout} (model instances per rollout group), sample_steps={config.rl.sample_steps}, update_ref_every_iter={config.rl.update_ref_every_iter}"
            )

            # Initialize rollout process groups
            self._setup_rollout_process_groups()

            # Setup GRPO training components after rollout groups
            self._setup_rl_training_components()

            self._setup_reward_model()

    def setup_data_key(self) -> None:
        self.input_data_key = (
            self.config.input_data_key
        )  # by default it is video key for Video wfm model
        self.input_image_key = self.config.input_image_key
        self.input_caption_key = self.config.input_caption_key

    def build_net(self):
        config = self.config
        init_device = "meta"
        with timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = MinimalV1LVGDiT(**config.net.model_dump())

            self._param_count = count_params(net, verbose=False)

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh)
                net = fully_shard(
                    net, mesh=self.fsdp_device_mesh, reshard_after_forward=True
                )

            with timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                # IMPORTANT: model init should not depends on current tensor shape, or it can handle Dtensor shape.
                net.init_weights()

            if self.fsdp_device_mesh:
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(
                        param, DTensor
                    ), f"param should be DTensor, {name} got {type(param)}"
        return net

    @timer("WorldFoundationalModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with timer("Creating PyTorch model and ema if enabled"):
            self.conditioner = Vid2VidConditioner(config.conditioner)
            assert (
                sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad)
                == 0
            ), "conditioner should not have learnable parameters"
            self.net = self.build_net()
            self._param_count = count_params(self.net, verbose=False)

            if config.ema.enabled:
                self.net_ema = self.build_net()
                self.net_ema.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots(
                    [1, 7, 16 - s**-2, 12 - s**-2]
                ).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

            # Reference model setup for GRPO
            if config.rl.enabled:
                self.net_ref = self.build_net()
                self.net_ref.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ref_worker = DTensorFastRefModelUpdater()
                else:
                    self.net_ref_worker = FastRefModelUpdater()

                self.net_ref_worker.copy_to(src_model=self.net, tgt_model=self.net_ref)
            else:
                self.net_ref = None

        torch.cuda.empty_cache()

    @timer("WorldFoundationalModel: set_up_text_encoder")
    def set_up_text_encoder(self):
        assert self.config.text_encoder_class.startswith(
            "reason1"
        ), "Only reason1 online computation is supported"

        qwen_vl_processor = build_tokenizer(
            self.config.text_encoder_config.tokenizer_type
        )
        self.text_encoder = QwenVLBaseModel(
            self.config.text_encoder_config.encoder_model_config,
            qwen_vl_processor,
        )
        self.text_encoder.load_hf_weights(
            self.config.text_encoder_config.ckpt_path, device="cuda"
        )
        self.text_encoder.eval()
        torch.cuda.empty_cache()

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        """Apply FSDP to the net and net_ema."""
        # Back-to-back fully_shard calls allow for wrapping submodules and the top-level module.
        self.net.fully_shard(mesh=dp_mesh)
        self.net = fully_shard(self.net, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.net, dp_mesh)
        if hasattr(self, "net_ema") and self.net_ema:
            self.net_ema.fully_shard(mesh=dp_mesh)
            self.net_ema = fully_shard(
                self.net_ema, mesh=dp_mesh, reshard_after_forward=True
            )
            broadcast_dtensor_model_states(self.net_ema, dp_mesh)
            self.net_ema_worker = DTensorFastEmaModelUpdater()
            # No need to copy weights to EMA when applying FSDP, it is already copied before applying FSDP.
        if hasattr(self, "net_ref") and self.net_ref:
            self.net_ref.fully_shard(mesh=dp_mesh)
            self.net_ref = fully_shard(
                self.net_ref, mesh=dp_mesh, reshard_after_forward=True
            )
            broadcast_dtensor_model_states(self.net_ref, dp_mesh)
            self.net_ref_worker = DTensorFastRefModelUpdater()
            # No need to copy weights to Ref when applying FSDP, it is already copied before applying FSDP.

    def init_optimizer_scheduler(
        self, optimizer_config: OptimizerConfig, scheduler_config: SchedulerConfig
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer = get_base_optimizer(**optimizer_config.model_dump(), model=self.net)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config.model_dump())
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        # TODO: (qsh 2025-01-13) clean ema up!

        if self.config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(
        self, memory_format: torch.memory_format = torch.preserve_format
    ) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if self.net_ref is not None:
            self.net_ref.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)

        if (
            hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile
        ):  # compatible with old config
            if torch.__version__ < "2.3":
                logger.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container."
                )
            # Increasing cache size. It's required because of the model size and dynamic input shapes resulting in
            # multiple different triton kernels. For 28 TransformerBlocks, the cache limit of 256 should be enough for
            # up to 9 different input shapes, as 28*9 < 256. If you have more Blocks or input shapes, and you observe
            # graph breaks at each Block (detectable with torch._dynamo.explain) or warnings about
            # exceeding cache limit, you may want to increase this size.
            # Starting with 24.05 Pytorch container, the default value is 256 anyway.
            # You can read more about it in the comments in Pytorch source code under path torch/_dynamo/cache_size.py.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape. It incurs higher compilation costs
            # at initial iterations, but can result in more specialized and efficient kernels.
            # dynamic=True currently throws errors in pytorch 2.3.
            self.net = torch.compile(
                self.net, dynamic=False, disable=not self.config.use_torch_compile
            )

    # ------------------------ training ------------------------

    def _setup_rollout_process_groups(self):
        """
        Initialize ALL process groups for GRPO rollouts once during model setup.
        Groups model instances (not individual ranks) into rollout groups using cosmos-rl's actual process groups.
        Each rollout group contains num_rollout model instances.
        """
        if self.parallel_dims is None:
            logger.warning(
                "[WFM RL] Distributed not initialized, skipping rollout process group setup"
            )
            self.rollout_groups = []
            return

        torch.distributed.barrier()
        # Get parallelism information
        my_dp_rank, dp_world_size = (
            self.parallel_dims.dp_coord
        )  # Number of model instances
        assert (
            dp_world_size % self.config.rl.num_rollout == 0
        ), f"dp_world_size {dp_world_size} must be divisible by num_rollout {self.config.rl.num_rollout}"

        # Create a mapping from data parallel rank to all global ranks in that model instance
        # We'll collect this information across all ranks
        dp_rank_to_global_ranks = {}

        # Get my own parallelism information
        my_global_rank = torch.distributed.get_rank()

        # Gather all rank mappings across all processes
        # Each rank contributes its (dp_rank, global_rank) pair
        all_rank_mappings = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            all_rank_mappings, (my_dp_rank, my_global_rank)
        )

        # Build the mapping from dp_rank to list of global ranks
        for dp_rank, global_rank in all_rank_mappings:
            if dp_rank not in dp_rank_to_global_ranks:
                dp_rank_to_global_ranks[dp_rank] = []
            dp_rank_to_global_ranks[dp_rank].append(global_rank)

        # Sort the global ranks for each data parallel rank for consistency
        for dp_rank in dp_rank_to_global_ranks:
            dp_rank_to_global_ranks[dp_rank].sort()

        # Group model instances into rollout groups
        rollout_group_size = self.config.rl.num_rollout
        num_rollout_groups = (
            dp_world_size + rollout_group_size - 1
        ) // rollout_group_size

        logger.info(
            f"[WFM RL] Creating {num_rollout_groups} rollout groups with {rollout_group_size} model instances each ..."
        )

        self.rollout_groups = []
        for rollout_group_id in range(num_rollout_groups):
            # Determine which model instances (dp_ranks) belong to this rollout group
            s_id = rollout_group_id * rollout_group_size

            # Collect all global ranks from all model instances in this rollout group
            group_ranks = sum(
                [
                    dp_rank_to_global_ranks[i]
                    for i in range(s_id, s_id + rollout_group_size)
                ],
                [],
            )
            group_ranks.sort()

            self.rollout_groups.append(
                {
                    "group": torch.distributed.new_group(ranks=group_ranks),
                    "ranks": group_ranks,
                    "src_rank": min(
                        group_ranks
                    ),  # The source rank is the minimum rank from the first model instance in the group
                }
            )

        # sync all ranks
        torch.distributed.barrier()
        logger.info(
            f"[WFM RL] Setup complete: created {len(self.rollout_groups)} rollout groups, each with {rollout_group_size} model instances"
        )
        for i, group_info in enumerate(self.rollout_groups):
            logger.info(
                f"[WFM RL] Rollout group {i}: ranks={group_info['ranks']}, src_rank={group_info['src_rank']}"
            )

        rank = torch.distributed.get_rank()

        # Find which rollout group this rank belongs to
        self.my_rollout_group = [w for w in self.rollout_groups if rank in w["ranks"]][
            0
        ]
        logger.info(
            f"[WFM RL] Rank {rank} belongs to rollout group: {self.my_rollout_group}",
        )

    def _setup_rl_training_components(self):
        """
        Setup GRPO training components including solver and x0_fn preparation.
        """

        from cosmos_rl.policy.model.wfm.sampler import (
            get_multi_step_fn,
            is_multi_step_fn_supported,
            get_runge_kutta_fn,
            is_runge_kutta_fn_supported,
            SolverConfig,
        )

        is_multistep = is_multi_step_fn_supported(self.config.rl.solver_option)
        is_rk = is_runge_kutta_fn_supported(self.config.rl.solver_option)
        assert (
            is_multistep or is_rk
        ), f"Only support multistep or Runge-Kutta method, got {self.config.rl.solver_option}"

        solver_cfg = SolverConfig(
            s_churn=self.config.rl.s_churn,
            s_t_max=self.config.rl.s_t_max,
            s_t_min=self.config.rl.s_t_min,
            s_noise=self.config.rl.s_noise,
            is_multi=is_multistep,
            rk=self.config.rl.solver_option,
            multistep=self.config.rl.solver_option,
        )
        self.rl_solver_cfg = solver_cfg

        if self.rl_solver_cfg.is_multi:
            self.rl_update_step_fn = get_multi_step_fn(self.rl_solver_cfg.multistep)
        else:
            self.rl_update_step_fn = get_runge_kutta_fn(self.rl_solver_cfg.rk)

        logger.info(
            f"[WFM RL] Setup training components with {self.rl_solver_cfg.rk} solver (multistep={self.rl_solver_cfg.is_multi})"
        )

    def _sync_data_across_rollouts(
        self, data_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Synchronize data batch across rollout instances using pre-initialized process groups.
        Each rollout group gets the same data batch but with different random seeds.

        Args:
            data_batch: Original data batch for this model instance

        Returns:
            Synchronized data batch (same across rollout group)
        """
        if self.inference_infos.rl_cached_trajectory is not None:
            return self.inference_infos.data_batch

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed not initialized, this should not happen when enabling GRPO."
            )

        if self.config.rl.use_same_seed:
            data_batch["seed"] = torch.randint(0, 1000000, (1,)).item()
            logger.info(
                f"[WFM RL]: syncing data across rollout groups. Manually set random seed before syncing. Seed: {data_batch['seed']}"
            )
        else:
            data_batch["seed"] = None
            logger.info(
                "[WFM RL]: syncing data across rollout groups. No random seed is set."
            )

        data_batch["num_conditional_frames"] = torch.randint(
            self.config.rl.min_num_conditional_frames,
            self.config.rl.max_num_conditional_frames + 1,
            size=(1,),
        ).item()
        logger.info(
            f"[WFM RL] Set num_conditional_frames to {data_batch['num_conditional_frames']}"
        )

        # Support random train_on here by replacing element -1 with random indices from the remaining indices in all but train_on indices
        effective_train_on = [i for i in self.config.rl.train_on if i != -1]
        # sample without replacement, no indices from effective_train_on
        potential_train_on = torch.tensor(
            [
                w
                for w in range(self.config.rl.sample_steps)
                if w not in self.config.rl.train_on
            ]
        )
        n_samples = len(self.config.rl.train_on) - len(effective_train_on)
        indices = torch.randperm(len(potential_train_on))[:n_samples]
        random_indices = potential_train_on[indices].tolist()

        data_batch["train_on"] = effective_train_on + random_indices
        logger.info(f"[WFM RL] Set train_on steps to: {data_batch['train_on']}")

        # Broadcast data within my rollout group
        self.inference_infos.data_batch = {
            key: broadcast(value, self.my_rollout_group["group"])
            for key, value in data_batch.items()
        }

        torch.cuda.empty_cache()
        torch.distributed.barrier()
        return self.inference_infos.data_batch

    def _setup_reward_model(self):
        """
        Setup reward model.
        """
        self.reward_models = {}
        for k, v in self.config.rl.reward_config.model_dump().items():
            logger.info(f"[WFM RL] Reward model: {k}: {v}")
        for k in self.config.rl.reward_config.ALL_REWARD_MODELS:
            v = getattr(self.config.rl.reward_config, k)
            if v.enabled:
                self.reward_models[k] = get_reward_model(v)

        self.inference_infos.rewards = {}
        self.inference_infos.is_rewards_synced = False

    def _sync_reward_within_context_parallel_group(
        self, reward: torch.Tensor, valid: bool
    ):
        """
        Synchronize reward within the context parallel group
        """
        cp_group = self.get_context_parallel_group()
        cp_ranks = get_process_group_ranks(cp_group) if cp_group else [0]
        if len(cp_ranks) == 1:
            return reward, valid

        gather_rewards = [torch.zeros_like(reward) for _ in cp_ranks]
        torch.distributed.all_gather(gather_rewards, reward, group=cp_group)
        gather_valid = [None for _ in cp_ranks]
        torch.distributed.all_gather_object(gather_valid, valid, group=cp_group)

        valid_reward = [w for w, v in zip(gather_rewards, gather_valid) if v]

        if len(valid_reward) == 0:  # means that all rewards are invalid
            return reward, False
        else:
            return valid_reward[0], True

    def _sync_rewards_across_rollouts(self):
        """
        Synchronize rewards across rollout groups to compute the mean reward
        """
        if self.inference_infos.is_rewards_synced:
            return self.inference_infos.rewards

        logger.info("[WFM RL] Start gathering rewards ...")
        sync_rewards = {}
        for k, v in self.inference_infos.rewards.items():
            valid = v["valid"]
            if valid and v["bg_id"] is not None:
                try:
                    reward = (
                        self.reward_models[k]
                        .fetch_reward(v["bg_id"])
                        .to(v["reward"].device)
                    )
                except Exception as e:
                    logger.warning(
                        f"[WFM RL] Reward model {k} failed: {e}. Ignore current rollout.",
                    )
                    reward, valid = torch.zeros_like(v["reward"]), False
            else:
                reward = v["reward"]
            # Valid here means that this cp group produces a valid reward
            # If not valid, the corresponding reward is set to 0
            reward, valid = self._sync_reward_within_context_parallel_group(
                reward, valid
            )

            sync_rewards[k] = {"reward": reward, "valid": valid}
            if self.config.rl.num_rollout > 1:
                gather_rewards = torch.stack(
                    [torch.zeros_like(reward) for _ in self.my_rollout_group["ranks"]],
                    dim=0,
                )
                torch.distributed.all_gather_into_tensor(
                    gather_rewards, reward, group=self.my_rollout_group["group"]
                )
                gather_valid = [False for _ in self.my_rollout_group["ranks"]]
                torch.distributed.all_gather_object(
                    gather_valid, valid, group=self.my_rollout_group["group"]
                )
                gather_valid = torch.tensor(gather_valid, device=reward.device)
                mean_reward = (
                    torch.sum(gather_rewards, dim=0) / torch.sum(gather_valid)
                    if torch.sum(gather_valid) > 0
                    else torch.zeros_like(reward)
                )
                reward = reward if valid else mean_reward
                gather_rewards = torch.where(
                    gather_valid[:, None], gather_rewards, mean_reward
                )

                sync_rewards[k]["reward"] = reward
                sync_rewards[k]["mean_reward"] = gather_rewards.mean(dim=0)
                sync_rewards[k]["std_reward"] = gather_rewards.std(dim=0)
                sync_rewards[k]["advantage"] = (
                    reward - sync_rewards[k]["mean_reward"]
                ) / (sync_rewards[k]["std_reward"] + 1e-4)
                valid_rate = torch.mean(gather_valid.float())
                sync_rewards[k]["valid_rate"] = gather_valid.float()
                logger.info(
                    f"[WFM RL] Finished gathering {k}. Valid rate: {valid_rate}"
                )
            else:
                sync_rewards[k]["mean_reward"] = reward
                sync_rewards[k]["std_reward"] = torch.zeros_like(reward)
                sync_rewards[k]["advantage"] = reward
                sync_rewards[k]["valid_rate"] = torch.ones_like(reward)

        self.inference_infos.rewards = sync_rewards
        self.inference_infos.is_rewards_synced = True

        return self.inference_infos.rewards

    def get_mu_from_model(self, _x0_fn, _noise_x, _sigma_cur, _sigma_next, **kwargs):
        ones_B = torch.ones(
            _noise_x.size(0), device=_noise_x.device, dtype=_noise_x.dtype
        )
        if self.rl_solver_cfg.is_multi:
            # For multistep solver
            x0_pred_current = _x0_fn(_noise_x, _sigma_cur * ones_B)

            _mu_current, _ = self.rl_update_step_fn(
                _noise_x,
                _sigma_cur * ones_B,
                _sigma_next * ones_B,
                x0_pred_current,
                **kwargs,
            )
        else:
            # For Runge-Kutta methods
            _mu_current, _ = self.rl_update_step_fn(
                _noise_x, _sigma_cur * ones_B, _sigma_next * ones_B, _x0_fn
            )
        return _mu_current

    @staticmethod
    def mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Mean normalize a tensor by subtracting the mean and dividing by the standard deviation.

        Args:
        tensor (torch.tensor): The tensor to normalize

        Returns:
        torch.tensor: The normalized tensor
        """
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
            tensor.std(dim=-1, keepdim=True) + 1e-8
        )

    @staticmethod
    def get_context_parallel_group():
        parallelism = get_global_parallel_dims()
        if parallelism is not None and "cp" in parallelism.mesh.mesh_dim_names:
            return parallelism.mesh["cp"].get_group()
        return None

    def broadcast_split_for_model_parallelsim(
        self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
    ):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(
                x0_B_C_T_H_W, seq_dim=2, process_group=cp_group
            )
            epsilon_B_C_T_H_W = broadcast_split_tensor(
                epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group
            )
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(
                        sigma_B_T, seq_dim=1, process_group=cp_group
                    )
            if condition is not None:
                condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_data_key
        if isinstance(self.net, WeightTrainingStat):
            if is_image:
                self.net.accum_image_sample_counter += (
                    data_batch[input_key].shape[0] * self.data_parallel_size
                )
            else:
                self.net.accum_video_sample_counter += (
                    data_batch[input_key].shape[0] * self.data_parallel_size
                )

    def draw_training_sigma_and_epsilon(
        self, x0_size: int, condition: Any
    ) -> torch.Tensor:
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device="cuda")
        sigma_B = self.sde.sample_t(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(
            sigma_B, "b -> b 1"
        )  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO
        # TODO: (qsh 2025-01-13) use math.sqrt(T) directly?
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        return sigma_B_1, epsilon

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor):
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    # ------------------------ Sampling ------------------------

    def generate_samples(
        self, batch_size: int, condition: T2VCondition
    ) -> torch.Tensor:
        """
        Generate samples with given condition. It is WITHOUT classifier-free-guidance.

        Args:
            batch_size (int):
            condition (T2VCondition): condition information generated from self.conditioner
        """
        _H, _W = self.get_video_latent_height_width()
        state_shape = [
            self.config.state_ch,
            self.config.state_t,
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]
        x_sigma_max = (
            torch.randn(batch_size, *state_shape, **self.tensor_kwargs)
            * self.sde.sigma_max
        )

        def x0_fn(x, t):
            return self.denoise(x, t, condition).x0  # ODE function

        return self.sampler(x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max)

    def generate_cfg_samples(
        self,
        batch_size: int,
        condition: T2VCondition,
        uncondition: T2VCondition,
        guidance=1.5,
    ) -> torch.Tensor:
        """
        Generate samples with with classifier-free-guidance.

        Args:
            batch_size (int):
            condition (T2VCondition): condition information generated from self.conditioner
            uncondition (T2VCondition): uncondition information, possibily generated from self.conditioner
        """
        _H, _W = self.get_video_latent_height_width()
        state_shape = [
            self.config.state_ch,
            self.config.state_t,
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x_sigma_max = (
            torch.randn(batch_size, *state_shape, **self.tensor_kwargs)
            * self.sde.sigma_max
        )

        def x0_fn(x, t):
            cond_x0 = self.denoise(x, t, condition).x0
            uncond_x0 = self.denoise(x, t, uncondition).x0
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return self.sampler(x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max)

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        _, x0, _ = self.get_data_and_condition(
            data_batch
        )  # we need always process the data batch first.
        is_image_batch = self.is_image_batch(data_batch)

        if is_negative_prompt:
            condition, uncondition = (
                self.conditioner.get_condition_with_negative_prompt(data_batch)
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(
                data_batch
            )

        condition = condition.edit_data_type(
            DataType.IMAGE if is_image_batch else DataType.VIDEO
        )
        uncondition = uncondition.edit_data_type(
            DataType.IMAGE if is_image_batch else DataType.VIDEO
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(
            x0, condition, None, None
        )
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(
            x0, uncondition, None, None
        )

        # For inference, check if parallel_state is initialized
        if self.parallel_dims is not None:
            # TODO: (qsh 2025-01-21) to_tp???
            pass
        else:
            assert not self.net.is_context_parallel_enabled, "parallel_dims is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert (
                    "guided_mask" in data_batch
                ), "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn

    # @torch.no_grad()
    # def validation_step(
    #     self, data: dict[str, torch.Tensor], iteration: int
    # ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    #     """
    #     Current code does nothing.
    #     """
    #     raw_data, x0, _ = self.get_data_and_condition(data)
    #     guidance = data["guidance"]
    #     data = to(data, **self.tensor_kwargs)
    #     sample = self.rollout_runner.rollout_generation(
    #         data,
    #         guidance=guidance,
    #         # make sure no mismatch and also works for cp
    #         state_shape=x0.shape[1:],
    #         n_sample=x0.shape[0],
    #     )
    #     sample = self.decode(sample)
    #     gt = raw_data
    #     caption = data["ai_caption"]
    #     return {"gt": gt, "result": sample, "caption": caption}, torch.tensor([0]).to(
    #         **self.tensor_kwargs
    #     )

    @torch.no_grad()
    def forward(self, xt, t, condition: T2VCondition):
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (T2VCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        return self.denoise(xt, t, condition)

    def on_after_backward(self, iteration: int = 0) -> None:
        """Hook after loss.backward() is called.

        This method is called immediately after the backward pass, allowing for custom operations
        or modifications to be performed on the gradients before the optimizer step.

        Args:
            iteration (int): Current iteration number.
        """
        pass

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, T2VCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[
            self.input_image_key if is_image_batch else self.input_data_key
        ]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(
            DataType.IMAGE if is_image_batch else DataType.VIDEO
        )
        return raw_state, latent_state, condition

    def _normalize_video_databatch_inplace(
        self, data_batch: dict[str, Tensor], input_key: str = None
    ) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if (
                IS_PREPROCESSED_KEY in data_batch
                and data_batch[IS_PREPROCESSED_KEY] is True
            ):
                assert torch.is_floating_point(
                    data_batch[input_key]
                ), "Video data is not in float format."
                assert torch.all(
                    (data_batch[input_key] >= -1.0001)
                    & (data_batch[input_key] <= 1.0001)
                ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
            else:
                assert (
                    data_batch[input_key].dtype == torch.uint8
                ), "Video data is not in uint8 format."
                data_batch[input_key] = (
                    data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                )
                data_batch[IS_PREPROCESSED_KEY] = True

            if self.config.resize_online:
                from torchvision.transforms.v2 import UniformTemporalSubsample

                expected_length = self.tokenizer.get_pixel_num_frames(
                    self.config.state_t
                )
                original_length = data_batch[input_key].shape[2]
                if original_length != expected_length:
                    video = rearrange(data_batch[input_key], "b c t h w -> b t c h w")
                    video = UniformTemporalSubsample(expected_length)(video)
                    data_batch[input_key] = rearrange(
                        video, "b t c h w -> b c t h w"
                    ).contiguous()

    def _augment_image_dim_inplace(
        self, data_batch: dict[str, Tensor], input_key: str = None
    ) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if (
                IS_PREPROCESSED_KEY in data_batch
                and data_batch[IS_PREPROCESSED_KEY] is True
            ):
                assert (
                    data_batch[input_key].shape[2] == 1
                ), f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                return
            else:
                data_batch[input_key] = rearrange(
                    data_batch[input_key], "b c h w -> b c 1 h w"
                ).contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        if self.net_ref is not None:
            ref_state_dict = self.net_ref.state_dict(prefix="net_ref.")
            net_state_dict.update(ref_state_dict)
        return net_state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        copy_mode: bool = False,
    ):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
            copy_mode (bool, optional): If True, the ema model and reference model (if applicable) will be copied from the main model
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        _ref_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v
            elif k.startswith("net_ref."):
                _ref_state_dict[k.replace("net_ref.", "")] = v

        state_dict = _reg_state_dict

        if copy_mode:
            logger.info(
                "Use the same state_dict for ema and ref model as that from the main model"
            )
            _ema_state_dict = _reg_state_dict
            _ref_state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(
                _reg_state_dict, strict=strict, assign=assign
            )

            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )

            ref_results = None
            if self.net_ref is not None:
                ref_results: _IncompatibleKeys = self.net_ref.load_state_dict(
                    _ref_state_dict, strict=strict, assign=assign
                )

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.ema.enabled else [])
                + (ref_results.missing_keys if ref_results is not None else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.ema.enabled else [])
                + (ref_results.unexpected_keys if ref_results is not None else []),
            )
        else:
            logger.critical("load model in non-strict mode")
            logger.critical(non_strict_load_model(self.net, _reg_state_dict))
            if self.config.ema.enabled:
                logger.critical("load ema model in non-strict mode")
                incompatible_keys = non_strict_load_model(self.net_ema, _ema_state_dict)
                logger.critical(incompatible_keys)
                if (
                    len(incompatible_keys.incorrect_shapes) == 0
                    and len(incompatible_keys.missing_keys) == 0
                ):
                    logger.critical(
                        "Successfully loaded ema model from the checkpoint."
                    )
                else:
                    logger.critical(
                        "Incorrect shapes or missing keys for the ema model, this implies that we are starting from non-ema checkpoint. Will copy reg model to ema model."
                    )
                    self.net_ema_worker.copy_to(
                        src_model=self.net, tgt_model=self.net_ema
                    )
                    logger.critical("Copied reg model to ema model")

            if self.net_ref is not None:
                logger.critical("load ref model in non-strict mode")
                incompatible_keys = non_strict_load_model(self.net_ref, _ref_state_dict)
                logger.critical(incompatible_keys)
                if (
                    len(incompatible_keys.incorrect_shapes) == 0
                    and len(incompatible_keys.missing_keys) == 0
                ):
                    logger.critical(
                        "Successfully loaded reference model from the checkpoint."
                    )
                else:
                    logger.critical(
                        "Incorrect shapes or missing keys for the ref model, this implies that we are starting from non-GRPO checkpoint. Will copy ema model (if enabled) to current model, and current model to reference model"
                    )
                    if self.config.ema.enabled:
                        self.net_ema_worker.copy_to(
                            src_model=self.net_ema, tgt_model=self.net
                        )
                    logger.critical("Copied ema model to current model")
                    self.net_ref_worker.copy_to(
                        src_model=self.net, tgt_model=self.net_ref
                    )
                    logger.critical("Copied current model to reference model")

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert (
            is_image != is_video
        ), "Only one of the input_image_key or input_data_key should be present in the data_batch."
        return is_image

    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: T2VCondition
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (T2VCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")
        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = (
            self.scaling(sigma=sigma_B_1_T_1_1)
        )

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=(xt_B_C_T_H_W * c_in_B_1_T_1_1).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        x0_pred_B_C_T_H_W = (
            c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        )

        # get noise prediction based on sde
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: T2VCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = self.sde.marginal_prob(x0_B_C_T_H_W, sigma_B_T)
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(
            std_B_T, "b t -> b 1 t 1 1"
        )
        # make prediction
        model_pred = self.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(
            weights_per_sigma_B_T, "b t -> b 1 t 1 1"
        )
        # TODO: (yenchenl 2025-01-23) Remove `kendall_loss` later
        kendall_loss = edm_loss_B_C_T_H_W
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
        }
        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.sigma_data)

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return (
            height // self.tokenizer.spatial_compression_factor,
            width // self.tokenizer.spatial_compression_factor,
        )

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
