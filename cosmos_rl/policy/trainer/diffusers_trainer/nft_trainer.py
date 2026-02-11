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
# Refer to https://github.com/NVlabs/DiffusionNFT/blob/main/scripts/train_nft_sd3.py

import numpy as np
import os
import random
from functools import partial
from PIL import Image
from typing import Optional, Dict, Any, List

import torch

import cosmos_rl.utils.distributed as dist_util
import cosmos_rl.utils.report.metrics_collection as metrics_collection
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.diffusers_trainer.diffusers_trainer import (
    DiffusersTrainer,
)
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.ema import EMAModuleWrapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.util import copy_weights_with_decay, is_master_rank
from cosmos_rl.utils.report.wandb_logger import is_wandb_available
from cosmos_rl.utils.logging import logger

try:
    import imageio
except ImportError:
    logger.warning(
        "imageio is not installed, video logging will not work. Install it with `pip install imageio` if needed."
    )


def get_weight_copy_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


@TrainerRegistry.register(trainer_type="diffusion_nft")
class NFTTrainer(DiffusersTrainer):
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

        self.height, self.width = self.config.policy.diffusers.inference_size
        self.num_frames = self.config.policy.diffusers.train_frames
        self.weight_copy_decay_type = (
            self.config.policy.diffusers.weight_copy_decay_type
        )
        self.model.transformer.set_adapter("default")
        # Create old model for RL training
        self.trainable_params = self.model.trainable_params
        self.model.transformer.set_adapter("old")
        self.old_trainable_params = self.model.trainable_params
        self.model.transformer.set_adapter("default")
        copy_weights_with_decay(
            src_params=self.trainable_params,
            tgt_params=self.old_trainable_params,
        )

        # Create ema if needed
        if self.config.train.ema_enable:
            self.ema = EMAModuleWrapper(
                parameters=self.trainable_params,
                decay=self.config.train.ema_decay,
                update_step_interval=self.config.train.ema_update_step_interval,
                device=self.device,
            )

        self.grpo_config = self.config.train.train_policy

        # For iteration control
        self.mini_batch = self.grpo_config.mini_batch
        self.batch_size_per_optimize = self.grpo_config.batch_size_per_optimize

        # For GRPO
        self.mu_iterations = self.grpo_config.mu_iterations
        self.num_train_timesteps = int(
            self.config.policy.diffusers.sample.num_steps
            * self.config.policy.diffusers.timesteps_fraction
        )
        self.optimizers.zero_grad()

    def save_checkpoint(
        self,
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
    ):
        logger.info(f"[Policy] Saving cosmos checkpoint at step {current_step}...")
        if self.config.train.ema_enable and self.ema is not None:
            self.ema.copy_ema_to(self.trainable_params, store_temp=True)
        model_state_dict = self.model.get_trained_model_state_dict()
        self.ckpt_manager.save_checkpoint(
            model=model_state_dict,
            optimizer=self.optimizers,
            scheduler=self.lr_schedulers,
            step=current_step,
            total_steps=total_steps,
            **{
                "remain_samples_num": remain_samples_num,
                "is_final": current_step == total_steps,
            },
        )
        self.ckpt_manager.save_check(step=current_step)
        if self.config.train.ema_enable and self.ema is not None:
            self.ema.copy_temp_to(self.trainable_params)

    def model_resume_from_checkpoint(self):
        ckpt_extra_vars, self.lr_schedulers = self.ckpt_manager.load_checkpoint(
            model=self.model.trained_model[0],
            optimizer=self.optimizers,
            scheduler=partial(build_lr_schedulers, self.optimizers, self.config),
            model_name_or_path=self.config.policy.model_name_or_path,
            revision=self.config.policy.model_revision,
            strict=not self.is_lora,  # For LoRA training, ckpt only save lora adapter's parameters, should load with restrict=False
        )
        return ckpt_extra_vars

    def weight_resume(self):
        model_loaded = False
        if self.config.train.resume:
            try:
                # Need to reload again from checkpoint to make sure the model is in the correct state
                self.model_resume_from_checkpoint()
                model_loaded = True
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    logger.info(
                        f"[Policy] Fail to resume from {self.config.train.resume} because the checkpoint file does not exist, trying to load from HuggingFace..."
                    )
                else:
                    logger.error(
                        f"[Policy] Cannot resume from {self.config.train.resume} {e}. Trying to load from HuggingFace..."
                    )
                if not model_loaded:
                    self.model_load_from_hf()
                    model_loaded = True
        elif not model_loaded:
            logger.info("[Policy] Resume not set. Trying to load from HuggingFace...")
            self.model_load_from_hf()
            model_loaded = True

        assert model_loaded, "Model weight must be populated before training starts."
        self.model.transformer.train()

        return False

    def update_lr_schedulers(self, total_steps: Optional[int] = None):
        pass

    def all_reduce_states(self, inter_policy_nccl: HighAvailabilitylNccl) -> float:
        """
        # Add nccl allreduce operations for all parameters and necessary states.
        """
        with torch.cuda.stream(self.train_stream):
            dist_util.gradient_reduce_across_dp_replicas_(
                self.trainable_params, inter_policy_nccl
            )
            """
            Compute the global grad norm on all parameters and then apply
            gradient clipping using the global grad norm.
            """
            # Must pass empty list even if model_part is None,
            # GradNorm across pp stages will fail if some rank does not join the barrier
            grad_norm = dist_util.gradient_norm_clipping(
                self.trainable_params,
                self.config.train.optm_grad_norm_clip,
                foreach=True,
                pp_mesh=self.parallel_dims.mesh["pp"]
                if self.parallel_dims.pp_enabled
                else None,
                return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
            )
            self.optimizers.step()
            self.optimizers.zero_grad()
        return grad_norm

    def report_mm_wandb(
        self,
        mm_datas: torch.Tensor,
        prompts: List[str],
        rewards: torch.Tensor,
        current_step: int,
    ):
        """Log multimodal data to Weights & Biases (Wandb) for visualization."""

        mm_datas_cpu = mm_datas.detach().cpu()
        num_to_log = min(15, len(mm_datas_cpu))
        rewards_cpu = rewards.detach().cpu()

        if self.model.is_video:
            modality = "video"
            sample_indices = random.sample(range(len(mm_datas_cpu)), num_to_log)
            ext = "mp4"
        else:
            modality = "image"
            sample_indices = list(range(num_to_log))  # log first N
            ext = "jpg"

        paths: List[str] = []
        mm_data_dir = os.path.join(self.config.train.output_dir, "mm_data")
        if not os.path.exists(mm_data_dir):
            os.makedirs(mm_data_dir, exist_ok=True)
        for out_idx, sample_idx in enumerate(sample_indices):
            out_path = os.path.join(mm_data_dir, f"{current_step}_{out_idx}.{ext}")
            if modality == "video":
                video = mm_datas_cpu[sample_idx].numpy()  # [T, C, H, W]
                frames = (video.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                imageio.mimsave(
                    out_path,
                    list(frames),
                    fps=8,
                    codec="libx264",
                    format="FFMPEG",
                )
            else:
                img = mm_datas_cpu[sample_idx].numpy().transpose(1, 2, 0)
                pil = Image.fromarray((img * 255).astype(np.uint8))
                pil.resize((self.width, self.height)).save(out_path)
            paths.append(out_path)

        if modality == "video":
            data = [
                {
                    "path": p,
                    "prompt": prompts[i],
                    "reward": float(rewards_cpu[i]),
                }
                for p, i in zip(paths, sample_indices)
            ]
        else:
            data = [
                {
                    "path": p,
                    "prompt": prompts[i],
                    "reward": float(rewards_cpu[i]),
                }
                for p, i in zip(paths, sample_indices)
            ]
        return data, modality

    def set_neg_prompt_embed(self):
        neg_text_embedding_dict = self.model.text_embedding(
            [""],
            device=self.device,
            max_sequence_length=128,
        )
        self.neg_prompt_embed = neg_text_embedding_dict["encoder_hidden_states"]
        self.neg_pooled_prompt_embed = neg_text_embedding_dict["pooled_projections"]

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
        logger.debug(f"Starting training step {current_step}/{total_steps}")
        assert (
            self.config.train.train_policy.kl_beta > 0.0
        ), "KL beta must be greater than 0 for diffusion NFT training."
        if current_step == 1:
            self.set_neg_prompt_embed()
        # Pack the list of rollouts into a batch for training
        packed_train_batch = self.data_packer.get_policy_input(
            sample=rollouts, device=self.device
        )

        self.model.transformer.train()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        batch_size, num_timesteps = packed_train_batch["timesteps"].shape
        per_optimize_batch_size = (
            min(self.batch_size_per_optimize, batch_size)
            if self.batch_size_per_optimize is not None
            and self.batch_size_per_optimize > 0
            else batch_size
        )

        def _slice_sample_batch(sample_batch: Dict[str, Any], index):
            """Slice per-sample tensors in a packed batch dict.

            Only tensors whose 0th dim matches the current batch size are sliced.
            """

            return {
                k: (
                    v[index]
                    if isinstance(v, torch.Tensor) and v.shape[0] == batch_size
                    else v
                )
                for k, v in sample_batch.items()
            }

        grad_norm_sum = torch.tensor(0.0, device=self.device)
        grad_norm_count = 0

        info_accumulated: Dict[str, List[torch.Tensor]] = {}

        # The cosmos-predict2.5 diffusers pipeline would set no_grad globally, so we need to re-enable it here
        with torch.enable_grad():
            for i_mu in range(self.mu_iterations):
                logger.info(
                    f"Mu iteration {i_mu + 1}/{self.mu_iterations} for training step {current_step}/{total_steps}"
                    if self.mu_iterations > 1
                    else f"Training step: {current_step}/{total_steps}"
                )

                # shuffle within the batch for better training stability
                perm = torch.randperm(batch_size, device=self.device)

                train_sample_batch = {
                    k: v[perm]
                    if isinstance(v, torch.Tensor) and v.shape[0] == batch_size
                    else v
                    for k, v in packed_train_batch.items()
                }
                # Generate random permutations for the timesteps of each sample in the batch
                perms_time = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=self.device)
                        for _ in range(batch_size)
                    ]
                )
                train_sample_batch["timesteps"] = train_sample_batch["timesteps"][
                    torch.arange(batch_size, device=self.device)[:, None], perms_time
                ]

                for i_opt in range(0, batch_size, per_optimize_batch_size):
                    opt_end = min(i_opt + per_optimize_batch_size, batch_size)
                    optimize_batch = _slice_sample_batch(
                        train_sample_batch, slice(i_opt, opt_end)
                    )
                    optimize_batch_size = opt_end - i_opt
                    if optimize_batch_size <= 0:
                        continue

                    mini_batch_size = (
                        min(self.mini_batch, optimize_batch_size)
                        if self.mini_batch is not None and self.mini_batch > 0
                        else optimize_batch_size
                    )

                    step_interval_env = os.environ.get(
                        "COSMOS_NFT_STEP_INTERVAL",
                        os.environ.get("COSMOS_GRPO_STEP_INTERVAL", None),
                    )
                    step_interval = (
                        int(step_interval_env) if step_interval_env is not None else 0
                    )
                    local_mini_step = 0
                    all_reduced = False

                    for i_mini in range(0, optimize_batch_size, mini_batch_size):
                        logger.debug(
                            f"Mini batch {i_mini // mini_batch_size + 1}/{(optimize_batch_size + mini_batch_size - 1) // mini_batch_size} for optimize batch {i_opt // per_optimize_batch_size + 1}/{(batch_size + per_optimize_batch_size - 1) // per_optimize_batch_size}"
                        )
                        mini_end = min(i_mini + mini_batch_size, optimize_batch_size)
                        mini_batch = {
                            k: (
                                v[slice(i_mini, mini_end)]
                                if isinstance(v, torch.Tensor)
                                and v.shape[0] == optimize_batch_size
                                else v
                            )
                            for k, v in optimize_batch.items()
                        }
                        cur_mini_size = mini_end - i_mini
                        loss_scaling_factor = cur_mini_size / optimize_batch_size

                        if self.config.policy.diffusers.sample.guidance_scale > 1.0:
                            embeds = torch.cat(
                                [
                                    self.neg_prompt_embed.repeat(cur_mini_size, 1, 1),
                                    mini_batch["prompt_embeds"],
                                ]
                            )
                            if self.neg_pooled_prompt_embed is not None:
                                pooled_embeds = torch.cat(
                                    [
                                        self.neg_pooled_prompt_embed.repeat(
                                            cur_mini_size, 1
                                        ),
                                        mini_batch["pooled_prompt_embeds"],
                                    ]
                                )
                            else:
                                pooled_embeds = None
                        else:
                            embeds = mini_batch["prompt_embeds"]
                            pooled_embeds = mini_batch["pooled_prompt_embeds"]

                        for j_idx, j_timestep_orig_idx in enumerate(
                            range(self.num_train_timesteps)
                        ):
                            assert j_idx == j_timestep_orig_idx

                            x0 = mini_batch["latents_clean"]
                            t = mini_batch["timesteps"][:, j_idx] / 1000.0
                            t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))
                            noise = torch.randn_like(x0.float())
                            xt = (1 - t_expanded) * x0 + t_expanded * noise

                            self.model.transformer.set_adapter("old")
                            transformer_inputs = (
                                self.model.nft_prepare_transformer_input(
                                    latents=xt,
                                    prompt_embeds=embeds,
                                    pooled_prompt_embeds=pooled_embeds
                                    if pooled_embeds is not None
                                    else None,
                                    timestep=mini_batch["timesteps"][:, j_idx],
                                    num_frames=self.num_frames,
                                    height=self.height,
                                    width=self.width,
                                )
                            )
                            with torch.no_grad():
                                old_prediction = self.model.transformer(
                                    **transformer_inputs
                                )[0].detach()
                            self.model.transformer.set_adapter("default")

                            forward_prediction = self.model.transformer(
                                **transformer_inputs
                            )[0]
                            # Use the base model (with adapter disabled) to a reference prediciton
                            with torch.no_grad():
                                if hasattr(self.model.transformer, "disable_adapters"):
                                    self.model.transformer.disable_adapters()
                                    ref_forward_prediction = self.model.transformer(
                                        **transformer_inputs
                                    )[0]
                                    self.model.transformer.enable_adapters()
                                elif hasattr(self.model.transformer, "disable_adapter"):
                                    with self.model.transformer.disable_adapter():
                                        ref_forward_prediction = self.model.transformer(
                                            **transformer_inputs
                                        )[0]
                                else:
                                    raise NotImplementedError(
                                        "The transformer model does not support adapter disabling."
                                    )
                                self.model.transformer.set_adapter("default")

                            advantages_clip = torch.clamp(
                                mini_batch["advantages"],
                                self.config.train.train_policy.advantage_low,
                                self.config.train.train_policy.advantage_high,
                            )
                            normalized_advantages_clip = (
                                advantages_clip
                                / self.config.train.train_policy.advantage_high
                            ) / 2.0 + 0.5
                            r = torch.clamp(normalized_advantages_clip, 0, 1)

                            positive_prediction = (
                                self.config.train.train_policy.kl_beta
                                * forward_prediction
                                + (1 - self.config.train.train_policy.kl_beta)
                                * old_prediction.detach()
                            )
                            implicit_negative_prediction = (
                                (1.0 + self.config.train.train_policy.kl_beta)
                                * old_prediction.detach()
                                - self.config.train.train_policy.kl_beta
                                * forward_prediction
                            )

                            x0_prediction = xt - t_expanded * positive_prediction
                            with torch.no_grad():
                                weight_factor = (
                                    torch.abs(x0_prediction.double() - x0.double())
                                    .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                    .clip(min=0.00001)
                                )
                            positive_loss = (
                                (x0_prediction - x0) ** 2 / weight_factor
                            ).mean(dim=tuple(range(1, x0.ndim)))

                            negative_x0_prediction = (
                                xt - t_expanded * implicit_negative_prediction
                            )
                            with torch.no_grad():
                                negative_weight_factor = (
                                    torch.abs(
                                        negative_x0_prediction.double() - x0.double()
                                    )
                                    .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                    .clip(min=0.00001)
                                )
                            negative_loss = (
                                (negative_x0_prediction - x0) ** 2
                                / negative_weight_factor
                            ).mean(dim=tuple(range(1, x0.ndim)))

                            ori_policy_loss = (
                                r
                                * positive_loss
                                / self.config.train.train_policy.kl_beta
                                + (1.0 - r)
                                * negative_loss
                                / self.config.train.train_policy.kl_beta
                            )
                            policy_loss = (
                                ori_policy_loss
                                * self.config.train.train_policy.advantage_high
                            ).mean()

                            kl_loss = (
                                ((forward_prediction - ref_forward_prediction) ** 2)
                                .mean(dim=tuple(range(1, x0.ndim)))
                                .mean()
                            )

                            loss = policy_loss + (
                                self.config.train.train_policy.kl_beta * kl_loss
                            )

                            timestep_scaling = 1.0 / max(self.num_train_timesteps, 1)
                            scaled_factor = loss_scaling_factor * timestep_scaling
                            scaled_loss = loss * scaled_factor
                            scaled_loss.backward()

                            unweighted_policy_loss = ori_policy_loss.mean()

                            report_terms: Dict[str, torch.Tensor] = {}
                            # weighted contributions (so final loss/policy/kl are true averages)
                            report_terms["loss_contrib"] = loss.detach()
                            report_terms["policy_loss_contrib"] = policy_loss.detach()
                            report_terms["unweighted_policy_loss_contrib"] = (
                                unweighted_policy_loss.detach()
                            )
                            report_terms["kl_loss_contrib"] = kl_loss.detach()

                            x0_sq = x0**2
                            cur_x0_norm = torch.mean(x0_sq)
                            cur_x0_norm_max = torch.max(x0_sq)
                            old_dev_sq = (forward_prediction - old_prediction) ** 2
                            cur_old_deviation = torch.mean(old_dev_sq)
                            cur_old_deviation_max = torch.max(old_dev_sq)

                            # raw (unscaled) metrics for monitoring
                            report_terms["x0_norm"] = cur_x0_norm.detach()
                            report_terms["x0_norm_max"] = cur_x0_norm_max.detach()
                            report_terms["old_deviation"] = cur_old_deviation.detach()
                            report_terms["old_deviation_max"] = (
                                cur_old_deviation_max.detach()
                            )
                            report_terms["old_kl"] = torch.mean(
                                ((old_prediction - ref_forward_prediction) ** 2).mean(
                                    dim=tuple(range(1, x0.ndim))
                                )
                            ).detach()

                            metrics_collection.accumulate_report_terms(
                                info_accumulated, report_terms
                            )

                        local_mini_step += 1
                        if (
                            step_interval > 0
                            and local_mini_step % step_interval == 0
                            and mini_end < optimize_batch_size
                        ):
                            all_reduced = True
                            grad_norm_sum += self.all_reduce_states(inter_policy_nccl)
                            grad_norm_count += 1
                            if self.config.train.ema_enable and self.ema is not None:
                                self.ema.step(self.trainable_params, current_step)
                        else:
                            all_reduced = False

                    if not all_reduced:
                        grad_norm_sum += self.all_reduce_states(inter_policy_nccl)
                        grad_norm_count += 1
                        if self.config.train.ema_enable and self.ema is not None:
                            self.ema.step(self.trainable_params, current_step)

        with torch.no_grad():
            decay = get_weight_copy_decay(current_step, self.weight_copy_decay_type)
            copy_weights_with_decay(
                src_params=self.trainable_params,
                tgt_params=self.old_trainable_params,
                decay=decay,
            )

        end_event.record()

        # Calculate average losses and aggregate metrics for logging
        loss = metrics_collection.mean_or_zero(
            info_accumulated, "loss_contrib", self.device
        )
        kl_loss = metrics_collection.mean_or_zero(
            info_accumulated, "kl_loss_contrib", self.device
        )
        policy_loss = metrics_collection.mean_or_zero(
            info_accumulated, "policy_loss_contrib", self.device
        )
        unweighted_policy_loss = metrics_collection.mean_or_zero(
            info_accumulated, "unweighted_policy_loss_contrib", self.device
        )

        x0_norm = metrics_collection.mean_or_zero(
            info_accumulated, "x0_norm", self.device
        )
        x0_norm_max = metrics_collection.mean_or_zero(
            info_accumulated, "x0_norm_max", self.device
        )
        old_deviation = metrics_collection.mean_or_zero(
            info_accumulated, "old_deviation", self.device
        )
        old_deviation_max = metrics_collection.mean_or_zero(
            info_accumulated, "old_deviation_max", self.device
        )
        kl = metrics_collection.mean_or_zero(info_accumulated, "kl", self.device)
        old_kl = metrics_collection.mean_or_zero(
            info_accumulated, "old_kl", self.device
        )

        dist_enabled = (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        )
        mesh = self.parallel_dims.mesh["dp_cp"] if dist_enabled else None

        avg_metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "unweighted_policy_loss": unweighted_policy_loss,
            "kl": kl,
            "old_kl": old_kl,
            "x0_norm": x0_norm,
            "old_deviation": old_deviation,
            "x0_norm_max": x0_norm_max,
            "old_deviation_max": old_deviation_max,
        }
        max_metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "unweighted_policy_loss": unweighted_policy_loss,
        }
        if self.config.train.train_policy.kl_beta != 0.0:
            avg_metrics["kl_loss"] = kl_loss
            max_metrics["kl_loss"] = kl_loss

        if dist_enabled:
            reduced_avg = {
                k: dist_util.dist_mean(v, mesh) for k, v in avg_metrics.items()
            }
            reduced_max = {
                k: dist_util.dist_max(v, mesh) for k, v in max_metrics.items()
            }
        else:
            reduced_avg = {k: v.item() for k, v in avg_metrics.items()}
            reduced_max = {k: v.item() for k, v in max_metrics.items()}

        report_data = {}
        if self.config.logging.logger:
            if is_master_rank(self.parallel_dims, self.global_rank):
                report_data = {"train_step": current_step}
                # Calculate the iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds
                # TODO(dinghaoy): support lr schedulers
                report_data["train/learning_rate"] = self.config.train.optm_lr
                report_data["train/iteration_time"] = iter_time

                for k, v in reduced_avg.items():
                    report_data[f"train/{k}_avg"] = v
                for k, v in reduced_max.items():
                    report_data[f"train/{k}_max"] = v

                report_data["train/grad_norm"] = (
                    grad_norm_sum.item() / grad_norm_count
                    if grad_norm_count > 0
                    else 0.0
                )
                if self.config.logging.multi_modal_log_interval is not None and (
                    current_step % self.config.logging.multi_modal_log_interval == 0
                    or current_step == 1
                ):
                    if not is_wandb_available():
                        logger.warning(
                            "Wandb is not available. Skipping multimodal logging."
                        )
                    else:
                        logger.info(
                            f"Logging multimodal data to Wandb at step {current_step}..."
                        )
                        mm_report_data, modality = self.report_mm_wandb(
                            mm_datas=packed_train_batch["completions"],
                            prompts=packed_train_batch["prompts"],
                            rewards=packed_train_batch["rewards"],
                            current_step=current_step,
                        )
                        report_data[f"rollout_{modality}s"] = mm_report_data

        # checkpointing
        if is_master_replica and (do_save_checkpoint):
            self.save_checkpoint(
                current_step=current_step,
                total_steps=total_steps,
                remain_samples_num=remain_samples_num,
            )

        return report_data
