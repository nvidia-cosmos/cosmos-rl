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
import tempfile
from functools import partial
from PIL import Image
from typing import Optional, Dict, Any, List

import torch

import cosmos_rl.utils.distributed as dist_util
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
from cosmos_rl.utils.util import copy_weights, is_master_rank
from cosmos_rl.utils.wandb_logger import log_wandb
from cosmos_rl.utils.logging import logger


def weight_copy_decay(step, decay_type):
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
        self.weight_copy_decay_type = (
            self.config.policy.diffusers.weight_copy_decay_type
        )
        self.model.transformer.set_adapter("default")
        # Create ref model for RL training
        self.trainable_params = self.model.trainable_params
        self.model.transformer.set_adapter("ref")
        self.ref_trainable_params = self.model.trainable_params
        self.model.transformer.set_adapter("default")
        copy_weights(
            src_params=self.trainable_params,
            tgt_params=self.ref_trainable_params,
        )

        # Create ema if needed
        if self.config.train.ema_enable:
            self.ema = EMAModuleWrapper(
                parameters=self.trainable_params,
                decay=self.config.train.ema_decay,
                update_step_interval=self.config.train.ema_update_step_interval,
                device=self.device,
            )

        # For GRPO
        self.mu_iterations = self.config.train.train_policy.mu_iterations
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
        self.model.train()

        return False

    def update_lr_schedulers(self, total_steps: Optional[int] = None):
        pass

    def all_reduce_states(self, inter_policy_nccl: HighAvailabilitylNccl) -> float:
        """
        # Add nccl allreduce operations for all parameters and necessary states.
        """
        with torch.cuda.stream(self.train_stream):
            for model_part in self.model_parts:
                # Model part may use same physical mesh for different logical mesh,
                # which is not supported by DTensor operands like `torch.nn.utils.get_total_norm`
                # So we need to do allreduce for each model part
                if model_part is not None:
                    dist_util.gradient_reduce_across_dp_replicas_(
                        [p for p in model_part.parameters()], inter_policy_nccl
                    )
            """
            Compute the global grad norm on all parameters and then apply
            gradient clipping using the global grad norm.
            """
            # Must pass empty list even if model_part is None,
            # GradNorm across pp stages will fail if some rank does not join the barrier
            all_params = [
                p
                for m in [model for model in self.model_parts if model is not None]
                for p in m.parameters()
            ]
            grad_norm = dist_util.gradient_norm_clipping(
                all_params,
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
        mm_datas,
        prompts,
        rewards,
        step: int,
    ):
        """Log multimodal data to Weights & Biases (WandB) for visualization."""

        import wandb

        # TODO(dinghaoy): support video data
        images_to_log = mm_datas.cpu()
        prompts_to_log = prompts
        rewards_to_log = rewards.cpu()

        with tempfile.TemporaryDirectory() as tmpdir:
            num_to_log = min(15, len(images_to_log))
            for idx in range(num_to_log):  # log first N
                img_data = images_to_log[idx]
                pil = Image.fromarray(
                    (img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((self.width, self.height))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            data = {
                "images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{idx}.jpg"),
                        caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                    )
                    for idx in range(num_to_log)
                ],
            }
            log_wandb(data, step)

    def set_neg_prompt_embed(self):
        neg_text_embedding_dict = self.model.text_embedding(
            [""],
            device=self.device,
            built_in=False,
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
        loss_sum = torch.tensor(0.0, device=self.device)
        kl_loss_sum = torch.tensor(0.0, device=self.device)
        grad_norm_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0
        grad_norm_count = 0

        if current_step == 1:
            self.set_neg_prompt_embed()
        # Pack the list of rollouts into a batch for training
        packed_train_batch = self.data_packer.get_policy_input(sample=rollouts)

        self.model.transformer.train()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i_mu in range(self.mu_iterations):
            logger.info(
                f"Mu iteration {i_mu + 1}/{self.mu_iterations} for training step {current_step}/{total_steps}"
                if self.mu_iterations > 1
                else f"Training step: {current_step}/{total_steps}"
            )
            # TODO(dinghaoy): support micro batching if needed
            batch_size, num_timesteps = packed_train_batch["timesteps"].shape

            # shuffle within the batch for better training stability
            perm = torch.randperm(batch_size, device=self.device)

            # TODO(dinghaoy): need better parallelism for diffusers pipeline (e.g., VAE, text encoder), need reduce D2D copies
            for k, v in packed_train_batch.items():
                if isinstance(v, torch.Tensor) and v.device != self.device:
                    logger.debug(f"Moving tensor {k} to device {self.device}")
                    packed_train_batch[k] = v.to(self.device)

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
            for key in ["timesteps", "next_timesteps"]:
                train_sample_batch[key] = train_sample_batch[key][
                    torch.arange(batch_size, device=self.device)[:, None], perms_time
                ]

            if self.config.policy.diffusers.sample.guidance_scale > 1.0:
                embeds = torch.cat(
                    [
                        self.neg_prompt_embed.repeat(batch_size, 1, 1),
                        train_sample_batch["prompt_embeds"],
                    ]
                )
                pooled_embeds = torch.cat(
                    [
                        self.neg_pooled_prompt_embed.repeat(batch_size, 1),
                        train_sample_batch["pooled_prompt_embeds"],
                    ]
                )
            else:
                embeds = train_sample_batch["prompt_embeds"]
                pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

            # Loop over timesteps for this micro-batch
            for j_idx, j_timestep_orig_idx in enumerate(
                range(self.num_train_timesteps)
            ):
                assert j_idx == j_timestep_orig_idx
                x0 = train_sample_batch["latents_clean"]

                t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                noise = torch.randn_like(x0.float())

                xt = (1 - t_expanded) * x0 + t_expanded * noise

                self.model.transformer.set_adapter("ref")
                with torch.no_grad():
                    # prediction v
                    ref_prediction = self.model.transformer(
                        hidden_states=xt,
                        timestep=train_sample_batch["timesteps"][:, j_idx],
                        encoder_hidden_states=embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0].detach()
                self.model.transformer.set_adapter("default")

                # prediction v
                forward_prediction = self.model.transformer(
                    hidden_states=xt,
                    timestep=train_sample_batch["timesteps"][:, j_idx],
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

                with torch.no_grad():  # Reference model part
                    # For LoRA, disable adapter.
                    if self.is_lora:
                        self.model.transformer.disable_adapters()
                        ref_forward_prediction = self.model.transformer(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]
                        self.model.transformer.enable_adapters()
                        self.model.transformer.set_adapter("default")
                    else:  # Full model - this requires a frozen copy of the model
                        assert False
                loss_terms = {}
                # Policy Gradient Loss
                advantages_clip = torch.clamp(
                    train_sample_batch["advantages"],
                    self.config.train.train_policy.advantage_low,
                    self.config.train.train_policy.advantage_high,
                )

                # normalize advantage
                normalized_advantages_clip = (
                    advantages_clip / self.config.train.train_policy.advantage_high
                ) / 2.0 + 0.5
                r = torch.clamp(normalized_advantages_clip, 0, 1)
                loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                loss_terms["ref_deviate"] = torch.mean(
                    (forward_prediction - ref_prediction) ** 2
                ).detach()
                loss_terms["ref_deviate_max"] = torch.max(
                    (forward_prediction - ref_prediction) ** 2
                ).detach()
                positive_prediction = (
                    self.config.train.train_policy.kl_beta * forward_prediction
                    + (1 - self.config.train.train_policy.kl_beta)
                    * ref_prediction.detach()
                )
                implicit_negative_prediction = (
                    (1.0 + self.config.train.train_policy.kl_beta)
                    * ref_prediction.detach()
                    - self.config.train.train_policy.kl_beta * forward_prediction
                )

                # adaptive weighting
                x0_prediction = xt - t_expanded * positive_prediction
                with torch.no_grad():
                    weight_factor = (
                        torch.abs(x0_prediction.double() - x0.double())
                        .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                        .clip(min=0.00001)
                    )
                positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
                    dim=tuple(range(1, x0.ndim))
                )
                negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                with torch.no_grad():
                    negative_weight_factor = (
                        torch.abs(negative_x0_prediction.double() - x0.double())
                        .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                        .clip(min=0.00001)
                    )
                negative_loss = (
                    (negative_x0_prediction - x0) ** 2 / negative_weight_factor
                ).mean(dim=tuple(range(1, x0.ndim)))

                ori_policy_loss = (
                    r * positive_loss / self.config.train.train_policy.kl_beta
                    + (1.0 - r) * negative_loss / self.config.train.train_policy.kl_beta
                )
                policy_loss = (
                    ori_policy_loss * self.config.train.train_policy.advantage_high
                ).mean()

                loss = policy_loss
                loss_terms["policy_loss"] = policy_loss.detach()
                loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                    dim=tuple(range(1, x0.ndim))
                )

                loss += self.config.train.train_policy.kl_beta * torch.mean(kl_div_loss)
                kl_div_loss = torch.mean(kl_div_loss)
                loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                loss_terms["kl_div"] = torch.mean(
                    ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )
                ).detach()
                loss_terms["ref_kl_div"] = torch.mean(
                    ((ref_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )
                ).detach()

                loss_terms["total_loss"] = loss.detach()

                loss.backward()
                loss_sum += loss
                kl_loss_sum += kl_div_loss
                loss_count += 1

                if os.environ.get("COSMOS_GRPO_STEP_INTERVAL", None) is not None:
                    grad_norm_sum += self.all_reduce_states(inter_policy_nccl)
                    grad_norm_count += 1

            if self.config.train.ema_enable and self.ema is not None:
                self.ema.step(self.trainable_params, current_step)

        with torch.no_grad():
            decay = weight_copy_decay(current_step, self.weight_copy_decay_type)
            for src_param, tgt_param in zip(
                self.trainable_params, self.ref_trainable_params, strict=True
            ):
                tgt_param.data.copy_(
                    tgt_param.detach().data * decay
                    + src_param.detach().clone().data * (1.0 - decay)
                )
        end_event.record()

        loss = (loss_sum / loss_count) if loss_count > 0 else loss_sum
        kl_loss = (kl_loss_sum / loss_count) if loss_count > 0 else kl_loss_sum
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(loss, self.parallel_dims.mesh["dp_cp"]),
            )
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss, global_max_kl_loss = (  # noqa: F841
                    dist_util.dist_mean(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                    dist_util.dist_max(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                )
        else:
            global_avg_loss = global_max_loss = loss.item()  # noqa: F841
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()  # noqa: F841

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
                report_data["train/loss_avg"] = global_avg_loss
                report_data["train/loss_max"] = global_max_loss
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss
                report_data["train/grad_norm"] = (
                    grad_norm_sum.item() / grad_norm_count
                    if grad_norm_count > 0
                    else 0.0
                )

        # checkpointing
        if is_master_replica and (do_save_checkpoint):
            self.save_checkpoint(
                current_step=current_step,
                total_steps=total_steps,
                remain_samples_num=remain_samples_num,
            )

        return report_data
