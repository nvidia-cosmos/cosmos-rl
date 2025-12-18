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

from typing import Dict, Any, List
import torch
import os

from cosmos_rl.utils.util import is_master_rank, str2torch_dtype
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.logging import logger


@TrainerRegistry.register(trainer_type="grpo_vla")
class OpenVLAGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Custom GRPO Trainer.
        Args:
            *args: Positional arguments for the base GRPOTrainer.
            **kwargs: Keyword arguments for the base GRPOTrainer.
        """
        super().__init__(*args, **kwargs)

    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform a single training step using the provided rollouts.
        This method can be overridden for custom training step logic.
        Here we simply call the superclass method for example.
        Args:
            rollouts: A list of Rollout objects containing the training data or unique identifiers for the training data.
            current_step: The current training step.
            total_steps: The total number of training steps.
            remain_samples_num: The number of remaining rollout generated samples to process in the whole training.
            inter_policy_nccl: The NCCL communicator for inter-policy communication.
            is_master_replica: Whether this replica is the master replica.
            do_save_checkpoint: Whether to save a checkpoint after this step.
        Returns:
            A dictionary of training metrics used for logging and reporting.
        """
        logger.info(
            f"[VLA Train] Starting VLA training for step {current_step}/{total_steps}"
        )
        self.model._set_fsdp_reshard_after_forward(self.config.train.fsdp_reshard_after_forward)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        self.optimizers.zero_grad()
        total_loss = 0.0
        max_loss = -float("inf")
        TRAINING_CHUNK_SIZE = self.config.vla.training_chunk_size

        policy_inputs = [
            self.data_packer.get_policy_input(r, self.device) for r in rollouts
        ]
        max_chunks = max(p.input_ids.shape[0] for p in policy_inputs)
        for policy_input in policy_inputs:
            episode_data = self.data_packer.policy_collate_fn(policy_input, max_chunks)
            task_id = policy_input.task_id
            trial_id = policy_input.trial_id
            weight_version = policy_input.weight_version
            # finish_step = policy_input.finish_step
            # complete = policy_input.complete
            advantage = policy_input.advantage
            episode_valid_responses = episode_data["logprob_masks"].sum()
            num_training_chunks = (
                max_chunks + TRAINING_CHUNK_SIZE - 1
            ) // TRAINING_CHUNK_SIZE

            logger.info(
                f"[VLA Train] Task {task_id}_{trial_id} finished @ {policy_input.finish_step}, weight version: {weight_version}"
            )
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                for chunk_idx in range(num_training_chunks):
                    start_idx = chunk_idx * TRAINING_CHUNK_SIZE
                    end_idx = min((chunk_idx + 1) * TRAINING_CHUNK_SIZE, max_chunks)
                    chunk_data = {
                        k: v[start_idx:end_idx].to(self.device)
                        for k, v in episode_data.items()
                    }
                    log_probs = self.model.forward_with_trajectory_structure(
                        chunk_data["input_ids"],
                        chunk_data["pixel_values"],
                        chunk_data["attention_mask"],
                        labels=chunk_data["responses"],
                        temperature=self.config.train.train_policy.temperature,
                    ).logprobs

                    chunk_response_mask = chunk_data["logprob_masks"]
                    chunk_valid_responses = chunk_response_mask.sum()

                    epsilon_low = self.config.train.train_policy.epsilon_low
                    epsilon_high = self.config.train.train_policy.epsilon_high
                    negative_approx_kl = log_probs - chunk_data["old_log_probs"]
                    ratio = torch.exp(negative_approx_kl)

                    pg_losses = -advantage * ratio
                    pg_losses2 = -advantage * torch.clamp(
                        ratio, 1.0 - epsilon_low, 1.0 + epsilon_high
                    )
                    pg_loss = (
                        torch.max(pg_losses, pg_losses2) * chunk_response_mask
                    ).sum()
                    pg_clipfrac = (
                        torch.gt(pg_losses2, pg_losses).float() * chunk_response_mask
                    ).sum()
                    ppo_kl = (-negative_approx_kl * chunk_response_mask).sum()

                    policy_loss = pg_loss / episode_valid_responses
                    pg_clipfrac = pg_clipfrac / episode_valid_responses
                    ppo_kl = ppo_kl / episode_valid_responses

                    loss = policy_loss / len(policy_inputs)
                    loss.backward()

                    total_loss += loss.item()
                    max_loss = max(max_loss, loss.item())
                    # Logging
                    logger.info(
                        f"[VLA Train] Task {task_id}_{trial_id} Chunk {chunk_idx+1}/{num_training_chunks}: "
                        f"loss={loss.item()}, ratio [{ratio.min().item()},{ratio.max().item()}], "
                        f"clipfrac={pg_clipfrac.item()}, ppo_kl={ppo_kl.item()}, "
                        f"mask_sum={chunk_valid_responses.item():.0f}"
                        + (" [PADDED]" if chunk_valid_responses == 0 else "")
                    )
        self.lr_schedulers.step()
        current_lr = self.lr_schedulers.get_last_lr()[0]
        grad_norm = self.all_reduce_states(inter_policy_nccl)

        end_event.record()
        logger.info(
            f"[VLA Train] Step {current_step} training time: {start_event.elapsed_time(end_event)/1000.0:.2f}s"
        )
        logger.info(
            f"[VLA Train] {len(policy_inputs)} episodes, current lr: {current_lr:.6f}, grad norm: {grad_norm:.6f}"
        )

        report_data = {}
        if self.config.logging.logger and is_master_rank(
            self.parallel_dims, self.global_rank
        ):
            report_data["train/iteration_time"] = float(
                start_event.elapsed_time(end_event)
            )
            report_data["train/learning_rate"] = float(current_lr)
            report_data["train/grad_norm"] = float(grad_norm)
            report_data["train/loss_avg"] = float(total_loss / len(policy_inputs))
            report_data["train/loss_max"] = float(max_loss)
            report_data["train_step"] = int(current_step)

        if is_master_replica and do_save_checkpoint:
            if self.config.train.ckpt.export_safetensors:
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=os.path.join(
                        "safetensors",
                        f"step_{current_step}",
                    ),
                    trainable_only=False,
                    is_final=current_step == total_steps,
                    dtype=str2torch_dtype(self.config.train.param_dtype),
                )
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=current_step,
                total_steps=total_steps,
                remain_samples_num=remain_samples_num,
                is_final=current_step == total_steps,
            )
            self.ckpt_manager.save_check(step=current_step)
        return report_data
