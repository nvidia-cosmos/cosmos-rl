# Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import torch

from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import is_master_rank, str2torch_dtype


@TrainerRegistry.register(trainer_type="grpo_pi05")
class PI05GRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
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
        logger.info(
            f"[PI05 Train] Starting PI05 training for step {current_step}/{total_steps}"
        )
        wall_t0 = time.time()
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
        max_chunks = max(p.chains.shape[0] for p in policy_inputs)
        for policy_input in policy_inputs:
            episode_data = self.data_packer.policy_collate_fn(policy_input, max_chunks)
            task_id = policy_input.task_id
            trial_id = policy_input.trial_id
            weight_version = policy_input.weight_version
            advantage = policy_input.advantage
            num_training_chunks = (
                max_chunks + TRAINING_CHUNK_SIZE - 1
            ) // TRAINING_CHUNK_SIZE

            logger.info(
                f"[PI05 Train] Task {task_id}_{trial_id} finished @ {policy_input.finish_step}, weight version: {weight_version}"
            )
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                for chunk_idx in range(num_training_chunks):
                    start_idx = chunk_idx * TRAINING_CHUNK_SIZE
                    end_idx = min((chunk_idx + 1) * TRAINING_CHUNK_SIZE, max_chunks)
                    chunk_data = {
                        k: v[start_idx:end_idx].to(self.device)
                        for k, v in episode_data.items()
                    }

                    # Replay chain to compute current policy logprobs (dense over action dims).
                    num_imgs = int(chunk_data["images"].shape[1])
                    images_list = [chunk_data["images"][:, i] for i in range(num_imgs)]
                    img_masks_list = [
                        chunk_data["image_masks"][:, i].to(torch.bool)
                        for i in range(num_imgs)
                    ]

                    log_probs_full, entropy_full = self.model.get_log_prob_value(
                        images=images_list,
                        img_masks=img_masks_list,
                        lang_tokens=chunk_data["input_ids"],
                        lang_masks=chunk_data["attention_mask"],
                        state=chunk_data["states"],
                        chains=chunk_data["chains"],
                        denoise_inds=chunk_data["denoise_inds"],
                        compute_values=False,
                    )

                    # Slice to env dims and aggregate like RLinf.
                    log_probs_full = log_probs_full[
                        :, :, : self.model.action_chunk, : self.model.action_env_dim
                    ]
                    entropy_full = entropy_full[
                        :, :, : self.model.action_chunk, : self.model.action_env_dim
                    ]
                    log_probs = log_probs_full.mean(dim=1)  # [B, Ta, Da]
                    # entropy = entropy_full.mean(dim=[1, 2, 3], keepdim=False)[
                    #     :, None
                    # ]  # [B, 1]

                    # Flatten to "token" dimension for PPO-style math
                    B = log_probs.shape[0]
                    log_probs = log_probs.reshape(B, -1)  # [B, L]
                    old_lp = chunk_data["old_log_probs"].reshape(B, -1)  # [B, L]

                    # Action-level mask (preferred): [B, L] where L = action_chunk * action_env_dim.
                    # Backward-compat: if mask is [B], expand to [B, L].
                    response_mask = chunk_data["logprob_masks"].to(torch.float32)
                    if response_mask.dim() == 1:
                        response_mask = response_mask[:, None].expand_as(log_probs)
                    elif response_mask.shape != log_probs.shape:
                        raise ValueError(
                            f"PI05 logprob_masks shape mismatch: mask={tuple(response_mask.shape)} "
                            f"log_probs={tuple(log_probs.shape)}"
                        )
                    loss_mask = response_mask.bool()
                    loss_mask_count = loss_mask.count_nonzero() or 1

                    # RLinf-style config parameters
                    clip_ratio_low = self.config.train.train_policy.epsilon_low
                    clip_ratio_high = self.config.train.train_policy.epsilon_high
                    clip_ratio_c = getattr(
                        self.config.train.train_policy, "clip_ratio_c", None
                    )

                    # RLinf-style ratio computation with mask for numerical stability
                    ratio = torch.where(
                        loss_mask,
                        torch.exp(torch.clamp(log_probs - old_lp, min=-20.0, max=20.0)),
                        torch.zeros_like(log_probs),
                    )
                    approx_kl = torch.where(
                        loss_mask,
                        (log_probs - old_lp).detach(),
                        torch.zeros_like(log_probs),
                    )

                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
                    )
                    policy_loss1 = -advantage * ratio
                    policy_loss2 = -advantage * clipped_ratio

                    # Standard PPO clip
                    policy_loss = torch.max(policy_loss1, policy_loss2)

                    # RLinf dual-clip: clip_ratio_c (e.g., 3.0)
                    if clip_ratio_c is not None and clip_ratio_c > 1.0:
                        policy_loss3 = torch.sign(advantage) * clip_ratio_c * advantage
                        policy_loss = torch.min(policy_loss, policy_loss3)

                    # Aggregate loss (RLinf masked_mean style)
                    pg_loss = (policy_loss * response_mask).sum() / loss_mask_count

                    # Metrics
                    clip_mask = policy_loss1.detach() < policy_loss2.detach()
                    pg_clipfrac = (
                        clip_mask.logical_and_(loss_mask).count_nonzero()
                        / loss_mask_count
                    )
                    ppo_kl = -approx_kl.sum() / loss_mask_count

                    loss = pg_loss / len(policy_inputs)
                    loss.backward()

                    total_loss += loss.item()
                    max_loss = max(max_loss, loss.item())
                    logger.info(
                        f"[PI05 Train] Task {task_id}_{trial_id} Chunk {chunk_idx+1}/{num_training_chunks}: "
                        f"loss={loss.item():.6f}, ratio [{ratio.min().item():.3f},{ratio.max().item():.3f}], "
                        f"clipfrac={pg_clipfrac.item():.4f}, ppo_kl={ppo_kl.item():.6f}, "
                        f"mask_sum={loss_mask_count}"
                        + (" [PADDED]" if loss_mask_count == 0 else "")
                    )
        self.lr_schedulers.step()
        current_lr = self.lr_schedulers.get_last_lr()[0]
        grad_norm = self.all_reduce_states(inter_policy_nccl)

        end_event.record()
        # NOTE: `CUDA error: device not ready` (or similar) here usually means an earlier CUDA
        # kernel/NCCL error was raised asynchronously and only surfaced at this sync point.
        # We explicitly synchronize so failures point here consistently, and we can fall back
        # to wall-clock timing if event timing fails.
        iter_time_s: float
        try:
            torch.cuda.synchronize()
            iter_time_s = start_event.elapsed_time(end_event) / 1000.0
        except Exception as e:
            # Keep a useful log message; re-raise because the CUDA context is likely unhealthy.
            logger.error(
                "[PI05 Train] CUDA failure surfaced during end-of-step synchronization/timing. "
                "This usually indicates an earlier asynchronous CUDA kernel/NCCL error. "
                "Re-run with `CUDA_LAUNCH_BLOCKING=1` (and optionally `COSMOS_CUDA_SYNC_DEBUG=1`) "
                "to pinpoint the first failing op."
            )
            logger.exception(e)
            raise
        logger.info(
            f"[PI05 Train] Step {current_step} training time: {iter_time_s:.2f}s (wall={time.time()-wall_t0:.2f}s)"
        )
        logger.info(
            f"[PI05 Train] {len(policy_inputs)} episodes, current lr: {current_lr:.6f}, grad norm: {grad_norm:.6f}"
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
