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


import torch
import os
import time
import random
import json
import numpy as np
from typing import Optional

from functools import partial
import torch.distributed as dist
from cosmos_rl.utils.logging import logger

import cosmos_rl.utils.util as util
import cosmos_rl.utils.distributed as dist_util
import math

from cosmos_rl.policy.trainer.optm import LRSchedulersContainer


from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.optm import build_optimizers
from cosmos_rl.policy.trainer.base import (
    Trainer,
    TrainerRegistry,
    wrap_to_cuda_tensor,
    extract_from_cuda_tensor,
)
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.policy.model import ModelRegistry


@TrainerRegistry.register(trainer_type="pi_sft")
class PISFTTrainer(Trainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: Optional[torch.cuda.Stream] = None,
        data_packer: BaseDataPacker = None,
        val_data_packer: BaseDataPacker = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            parallel_dims=parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            **kwargs,
        )

        # set seed
        if config.train.seed:
            torch.manual_seed(config.train.seed)
            torch.cuda.manual_seed(config.train.seed)
            torch.cuda.manual_seed_all(config.train.seed)
            random.seed(config.train.seed)
            np.random.seed(config.train.seed)
        if config.train.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(mode=True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True

        # init model
        self.model = ModelRegistry.build_model(self.config)
        self.model.set_gradient_checkpointing_enabled(
                config.policy.model_gradient_checkpointing
            )

        # Materialize meta-initialized parameters on the training device (LLMTrainer-style).
        self.model._apply(
            lambda t: torch.empty_like(t, device=self.device)
            if getattr(t, "is_meta", False)
            else t.to(self.device),
            recurse=True,
        )

        self.build_optimizers()
        self.lr_schedulers = None
        self.ckpt_manager = CheckpointMananger(
            self.config, self.parallel_dims, self.global_rank
        )

    def sync_all_states(
        self,
        *,
        is_send: bool,
        send_hook: callable,
        recv_hook: callable,
    ) -> int:
        """Sync model/optimizer/scheduler/rng across dp_replicate, mirroring LLMTrainer."""
        len_params = 0

        # 1) model params + buffers
        state_dict = self.model.state_dict()
        state_dict.update(dict(self.model.named_buffers()))
        for dest_name in sorted(state_dict.keys()):
            obj = state_dict[dest_name]
            assert isinstance(obj, torch.Tensor)
            local_view = wrap_to_cuda_tensor(
                self.device, dest_name, obj, in_place=obj.is_cuda
            )
            if is_send:
                send_hook(local_view)
            else:
                recv_hook(local_view)
                if isinstance(obj, torch.distributed.tensor.DTensor):
                    to_write = obj.to_local()
                else:
                    to_write = obj
                if not to_write.is_cuda:
                    to_write.copy_(local_view)
            len_params += 1

        # 2) optimizer state
        optimizer_state = self.optimizers.state_dict()
        for dest_name in sorted(optimizer_state.keys()):
            obj = optimizer_state[dest_name]
            local_view = wrap_to_cuda_tensor(self.device, dest_name, obj)
            if local_view.data_ptr() is None:
                continue
            if is_send:
                send_hook(local_view)
            else:
                recv_hook(local_view)
                optimizer_state[dest_name] = extract_from_cuda_tensor(
                    self.device, dest_name, obj, local_view
                )
        if not is_send:
            self.optimizers.load_state_dict(optimizer_state)
        len_params += len(optimizer_state)

        # 3) scheduler state
        if self.lr_schedulers is not None:
            lr_state = self.lr_schedulers.state_dict()
            for dest_name in sorted(lr_state.keys()):
                obj = lr_state[dest_name]
                local_view = wrap_to_cuda_tensor(self.device, dest_name, obj)
                if is_send:
                    send_hook(local_view)
                else:
                    recv_hook(local_view)
                    lr_state[dest_name] = extract_from_cuda_tensor(
                        self.device, dest_name, obj, local_view
                    )
            if not is_send:
                self.lr_schedulers.load_state_dict(lr_state)
            len_params += len(lr_state)

        # 4) rng_state
        rng_state = self.ckpt_manager.get_rng_state()
        for dest_name in sorted(rng_state.keys()):
            obj = rng_state[dest_name]
            local_view = wrap_to_cuda_tensor(self.device, dest_name, obj)
            if is_send:
                send_hook(local_view)
            else:
                recv_hook(local_view)
                rng_state[dest_name] = extract_from_cuda_tensor(
                    self.device, dest_name, obj, local_view
                )
        len_params += len(rng_state)
        if not is_send:
            self.ckpt_manager.set_rng_state(rng_state)
        return len_params


    def step_training(self, global_batch, total_steps, train_step, save_freq):
        # timing started
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # prepare lr and optimizer state
        self.optimizers.zero_grad()
        if self.lr_schedulers is None:
            assert (
                train_step == 0
            ), "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            self.lr_schedulers = self.build_lr_schedulers(total_steps)
        
        # split global_batch into mini_batches
        global_batch_size = len(global_batch)
        mini_batch_begin_idxs = list(
            range(
                0,
                global_batch_size,
                self.config.train.train_policy.mini_batch,
            )
        )

        # get loss and backward gradient
        acc_loss = torch.zeros(1, device=self.device)
        for i in mini_batch_begin_idxs:
            raw_batch = global_batch[i : i + self.config.train.train_policy.mini_batch]
            batch = self.data_packer.sft_collate_fn(raw_batch)
            observation, actions = batch["observation"], batch["actions"]
            observation = self._move_observation_to_device(observation)
            actions = actions.to(dtype=torch.float32, device=self.device)
            loss = self.model(observation, actions)
            loss.mean().backward()
            acc_loss += loss.mean().detach()
        # Average loss over all mini-batches for logging
        acc_loss = acc_loss / len(mini_batch_begin_idxs)

        # calculate grad norm
        grad_norm = dist_util.gradient_norm_clipping(
            self.model.parameters(),
            self.config.train.optm_grad_norm_clip,
            foreach=True,
            pp_mesh=self.parallel_dims.mesh["pp"]
            if self.parallel_dims.pp_enabled
            else None,
            return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
        )

        # optimizer step
        self.optimizers.step()
        self.lr_schedulers.step()

        # # official implementation applies a more aggressive gradient clipping here.
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         param.grad.detach_()
        #         param.grad = None

        # timing ended
        end_event.record()

        # Report data
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(acc_loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(acc_loss, self.parallel_dims.mesh["dp_cp"]),
            )
        else:
            global_avg_loss = global_max_loss = acc_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if util.is_master_rank(self.parallel_dims, self.global_rank):
                # Calculate last iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds

                report_data = {
                    "train/iteration_time": iter_time,
                    "train/loss_avg": global_avg_loss,
                    "train/loss_max": global_max_loss,
                    "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                    "train/grad_norm": grad_norm if grad_norm is not None else -1,
                }
        
        return report_data


    def _move_observation_to_device(self, observation):
        """Move all tensors in observation to self.device (matching OpenPI's jax.tree.map pattern)."""
        observation.images = {k: v.to(self.device) for k, v in observation.images.items()}
        observation.image_masks = {k: v.to(self.device) for k, v in observation.image_masks.items()}
        observation.state = observation.state.to(self.device)
        observation.tokenized_prompt = observation.tokenized_prompt.to(self.device)
        observation.tokenized_prompt_mask = observation.tokenized_prompt_mask.to(self.device)
        return observation

    def build_optimizers(self):
        self.optimizers = build_optimizers([self.model], self.config)

    def checkpointing(
        self,
        total_steps: int,
        train_step: int,
        save_freq: int,
        is_last_step: bool = False,
        pp_last_stage: bool = False,
        val_score: Optional[float] = None,
    ):
        if (
            is_last_step or (train_step % save_freq == 0 and train_step > 0)
        ) and self.parallel_dims.dp_replicate_coord[0] == 0:
            if self.config.train.ckpt.enable_checkpoint:
                logger.info(f"Saving cosmos checkpoint at step {train_step}...")
                model_state_dict = self.model.get_trained_model_state_dict()
                self.ckpt_manager.save_checkpoint(
                    model=model_state_dict,
                    optimizer=self.optimizers,
                    scheduler=self.lr_schedulers,
                    step=train_step,
                    total_steps=total_steps,
                )
                if util.is_master_rank(self.parallel_dims, self.global_rank):
                    norm_stats = getattr(self.data_packer, "norm_stats", None)
                    if norm_stats is not None:
                        step_dir = os.path.join(self.config.train.output_dir, "checkpoints", f"step_{train_step}", "policy")
                        os.makedirs(step_dir, exist_ok=True)
                        norm_stats_path = os.path.join(step_dir, "norm_stats.json")
                        if not os.path.exists(norm_stats_path):
                            with open(norm_stats_path, "w") as f:
                                json.dump(norm_stats, f, indent=2, ensure_ascii=False)

    def load_model(self):
        ckpt_total_steps = 0
        train_step = 0
        if (not self.parallel_dims.dp_replicate_enabled
        ) or self.parallel_dims.dp_replicate_coord[0] == 0:
            resumed = False
            if self.config.train.resume:
                try:
                    ckpt_extra_vars = self.model_resume_from_checkpoint()
                    ckpt_total_steps = ckpt_extra_vars.get("total_steps", 0)
                    train_step = ckpt_extra_vars.get("step", 0)
                    resumed = True
                except Exception as e:
                    logger.error(
                        f"Cannot resume due to error: {e}. Trying to load from HuggingFace..."
                    )
            if not resumed:
                self.model_load_from_hf()
        
        if self.parallel_dims.dp_replicate_enabled:
            if self.config.train.resume:
                ckpt_total_steps = dist_util.broadcast_object_cpu(
                    ckpt_total_steps,
                    group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                    group_src=0,
                )
                if (
                    self.parallel_dims.dp_replicate_coord[0] != 0
                    and ckpt_total_steps is not None
                ):
                    # Initialize lr_schedulers on non-zero dp_replicate ranks when resuming training
                    # only when ckpt_total_steps > 0, means a checkpoint is loaded
                    self.lr_schedulers = self.build_lr_schedulers(ckpt_total_steps)
                if ckpt_total_steps is not None:
                    assert (
                        self.lr_schedulers is not None
                    ), "lr_schedulers should not be None after broadcasting when resuming training with data parallel replication."

            group = self.parallel_dims.mesh["dp_replicate"].get_group()
            send_recv_hook = partial(dist.broadcast, src=0, group=group)
            len_params = self.sync_all_states(
                is_send=self.parallel_dims.dp_replicate_coord[0] == 0,
                send_hook=send_recv_hook,
                recv_hook=send_recv_hook,
            )
            logger.info(
                f"Synchronized {len_params} parameters across data parallel replicas."
            )

        self.model.train()
        return ckpt_total_steps, train_step


    def model_load_from_hf(self):
        start_time = time.time()
        self.model.load_hf_weights(
            self.config.policy.model_name_or_path,
            self.parallel_dims,
            self.device,
            revision=self.config.policy.model_revision,
        )
        end_time = time.time()
        logger.info(
            f"Time taken to load model from HF: {end_time - start_time:.2f} seconds"
        )

    def model_resume_from_checkpoint(self):
        ckpt_extra_vars, self.lr_schedulers = self.ckpt_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=self.build_lr_schedulers,
            model_name_or_path=self.config.policy.model_name_or_path,
            revision=self.config.policy.model_revision,
        )
        return ckpt_extra_vars


    def step_validation(self, *args, **kwargs):
        pass

    def build_lr_schedulers(self, training_steps):
        """OpenPI-aligned warmup + cosine LR scheduler."""
        warmup_steps = int(self.config.train.optm_warmup_steps)
        decay_steps = training_steps

        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                init_factor = 1.0 / (warmup_steps + 1)
                return init_factor + (1.0 - init_factor) * step / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LRSchedulersContainer(self.optimizers, lr_factor)

    def export_safetensors(self, *args, **kwargs):
        pass