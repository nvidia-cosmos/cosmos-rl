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
from typing import Optional

import cosmos_rl.utils.util as util
import cosmos_rl.utils.distributed as dist_util

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from cosmos_rl.policy.trainer.base import (
    TrainerRegistry,
)
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer


@TrainerRegistry.register(trainer_type="pi_sft")
class PISFTTrainer(SFTTrainer):
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

        self._log_interval = max(1, getattr(self.config.logging, "log_interval", 100))
        self._loss_interval_sum = 0.0
        self._loss_interval_count = 0

    def step_training(self, global_batch, total_steps, train_step, *args, **kwargs):
        # timing started
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # prepare lr and optimizer state
        if self.lr_schedulers is None:
            assert (
                train_step == 0
            ), "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            self.lr_schedulers = build_lr_schedulers(self.optimizers, self.config, total_steps)

        self.lr_schedulers.step()
        self.optimizers.zero_grad()

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
            (loss.mean() / len(mini_batch_begin_idxs)).backward()
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
            loss_value = (
                global_avg_loss.item()
                if isinstance(global_avg_loss, torch.Tensor)
                else float(global_avg_loss)
            )
            self._loss_interval_sum += loss_value
            self._loss_interval_count += 1
            should_log_first = train_step == 0
            should_log_interval = (train_step + 1) % self._log_interval == 0
            if not (should_log_first or should_log_interval):
                return report_data
            if util.is_master_rank(self.parallel_dims, self.global_rank):
                # Calculate last iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds

                if should_log_first:
                    log_loss_avg = loss_value
                else:
                    log_loss_avg = (
                        self._loss_interval_sum / self._loss_interval_count
                        if self._loss_interval_count > 0
                        else loss_value
                    )

                report_data = {
                    "train/iteration_time": iter_time,
                    "train/loss_avg": log_loss_avg,
                    "train/loss_max": global_max_loss,
                    "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                    "train/grad_norm": grad_norm if grad_norm is not None else -1,
                }
            if should_log_interval:
                self._loss_interval_sum = 0.0
                self._loss_interval_count = 0

        return report_data

    def _move_observation_to_device(self, observation):
        """Move all tensors in observation to self.device (matching OpenPI's jax.tree.map pattern)."""
        observation.images = {
            k: v.to(self.device) for k, v in observation.images.items()
        }
        observation.image_masks = {
            k: v.to(self.device) for k, v in observation.image_masks.items()
        }
        observation.state = observation.state.to(self.device)
        observation.tokenized_prompt = observation.tokenized_prompt.to(self.device)
        observation.tokenized_prompt_mask = observation.tokenized_prompt_mask.to(self.device)
        return observation


    def step_validation(self, *args, **kwargs):
        pass

    def build_lr_schedulers(self, training_steps: int, **_kwargs):
        pass
