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


import functools
import os
import signal
import pynvml
import math

from typing import Callable

import torch
import torch.distributed as dist
import torch.utils.data

from contextlib import nullcontext

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig
from cosmos_rl.utils.wfm import callback, distributed
from cosmos_rl.utils.wfm.ema import ema_scope
from cosmos_rl.utils.wfm.model_loader import (
    create_model_from_consolidated_checkpoint_with_fsdp,
)
from cosmos_rl.utils.wfm.utils import (
    arch_invariant_rand,
    to,
    print_environ_variables,
    set_random_seed,
    timeout_handler,
    TrainingTimer,
)
from cosmos_rl.utils.wfm.checkpointer import (
    Checkpointer,
    DistributedCheckpointer,
)
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.model.wfm.models.v2v_model import Vid2VidModel
from cosmos_rl.policy.model.wfm.models.t2v_model import InferenceInfos
from cosmos_rl.rollout.wfm_rollout.wfm_rollout import WFMRollout

from cosmos_rl.utils.wfm.ref import (
    ref_scope,
)
from cosmos_rl.utils.wfm.distributed import (
    is_tp_cp_pp_rank0,
)
from cosmos_rl.utils.wfm.utils import (
    NUM_EMBEDDING_PADDING_TOKENS,
)

from cosmos_rl.policy.config.wfm import (
    EmbeddingConcatStrategy,
)

from cosmos_rl.utils.wfm.context_parallel import (
    split_inputs_cp,
    cat_outputs_cp,
)
from abc import ABC


class CosmosVisionGenTrainer(ABC):
    """The base trainer class of CosmosVisionGen.

    All trainers in CosmosVisionGen should inherit CosmosVisionGenTrainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (Timer): Timer object to time code blocks and functions.
    """

    def __init__(
        self, config: CosmosVisionGenConfig, parallel_dims: ParallelDims, **kwargs
    ):
        super().__init__()
        self.config = config
        self.trainer_init(self.config, parallel_dims)

        # Initialize the dataloaders.
        assert "dataloader" in kwargs
        assert isinstance(kwargs["dataloader"], Callable)
        dataloader_fn = kwargs["dataloader"]
        self.dataloader_train = dataloader_fn(self.config.dataloader_train)
        self.dataloader_val = dataloader_fn(self.config.dataloader_val)

    def trainer_init(self, config: CosmosVisionGenConfig, parallel_dims: ParallelDims):
        """
        Overide the trainer_init method to add the wfm specific initialization.
        """
        self.config = (
            config  # TODO(chenhsuanl): not ideal, should be only config_trainer
        )
        # init pynvml
        pynvml.nvmlInit()

        # Set up the distributed computing environment.
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Set up NCCL communication.
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        # Set up parallel states.
        if hasattr(config.model, "context_parallel_size"):
            if config.model_parallel.context_parallel_size > 1:
                raise ValueError(
                    "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                    "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                )
            else:
                logger.critical(
                    "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                )
                config.model_parallel.context_parallel_size = (
                    config.model.context_parallel_size
                )
        distributed.initialize_global_parallelism(config)
        self.parallel_dims = distributed.get_global_parallel_dims()

        if config.model_parallel.sequence_parallel:
            # FIXME: (lms/dinghao) check sequence_parallel is used or not. Seems not used in VideoGen.
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Create the local job directory, save the config file, and pipe to a local logger.
        if distributed.get_rank() == 0:
            os.makedirs(config.job.path_local, exist_ok=True)
            config_json = config.model_dump_json()
            with open(os.path.join(config.job.path_local, "config.json"), "w") as f:
                f.write(config_json)
        dist.barrier()
        if distributed.get_rank() == 0:
            # Print important environment variables and the effective config.
            logger.info(f"Config of CosmosVisionGenTrainer:\n{config.model_dump()}")
        print_environ_variables(
            ["TORCH_HOME", "COSMOS_VISION_GEN_OUTPUT_ROOT", "ENABLE_ONELOGGER"]
        )
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        set_random_seed(
            seed=config.trainer.seed, by_rank=True
        )  # FIXME(yenchenl): reinvestigate
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        # Initialize the callback functions.
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # Initialize the timer for speed benchmarking.
        self.training_timer = TrainingTimer()
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        signal.signal(
            signal.SIGALRM,
            functools.partial(timeout_handler, config.trainer.timeout_period),
        )  # type: ignore

        # Create the model and load the consolidated checkpoint if provided.
        # If the checkpoint is in DCP format, checkpoint loading will be handled by the DCP checkpointer.
        if config.checkpoint.load_path and (
            config.checkpoint.load_path.endswith(".pt") or config.checkpoint.model_id
        ):
            self.model = create_model_from_consolidated_checkpoint_with_fsdp(config)
        else:
            self.model = Vid2VidModel(config.model)
        # Initialize the model checkpointer.
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(
                config.checkpoint, config.job, callbacks=self.callbacks
            )
        elif config.checkpoint.type == "distributed":
            self.checkpointer: Checkpointer = DistributedCheckpointer(
                config.checkpoint, config.job, callbacks=self.callbacks
            )
        else:
            raise ValueError(
                f"Unknown checkpoint type: {config.checkpoint.type}. "
                "Please use 'distributed' or None."
            )
        # Initialize the rollout runner.
        self.rollout_runner = WFMRollout(config)
        self.rollout_runner.init_engine(self.model)

    def train(self) -> None:
        """The training function."""
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        memory_format = None
        if self.config.trainer.memory_format == "preserve_format":
            memory_format = torch.preserve_format
        elif self.config.trainer.memory_format == "channels_last":
            memory_format = torch.channels_last
        elif self.config.trainer.memory_format == "channels_last_3d":
            memory_format = torch.channels_last_3d
        elif self.config.trainer.memory_format == "contiguous_format":
            memory_format = torch.contiguous_format
        else:
            raise ValueError(
                f"Unknown memory format: {self.config.trainer.memory_format}"
            )
        self.model = self.model.to("cuda", memory_format=memory_format)  # type: ignore
        self.model.on_train_start(memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = self.model.init_optimizer_scheduler(
            self.config.optimizer, self.config.scheduler
        )
        grad_scaler = torch.amp.GradScaler(
            "cuda", **self.config.trainer.grad_scaler_args
        )
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(
            self.model, optimizer, scheduler, grad_scaler
        )
        grad_accum_iter = 0
        logger.critical(
            f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}"
        )
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            self.model_ddp = distributed.parallel_model_wrapper(
                self.config.trainer.ddp, self.model
            )
        elif self.config.trainer.distributed_parallelism == "fsdp":
            self.model_ddp = self.model
        else:
            raise ValueError(
                f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}"
            )
        # TODO: Pytorch has DistributedOptimizer, is this something we want to consider?
        logger.info("Starting training...")
        self.callbacks.on_train_start(self.model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(self.model, self.dataloader_val, iteration=iteration)
        _end_training = False
        while True:
            dataloader_train_iter = iter(self.dataloader_train)
            while True:
                self.callbacks.on_before_dataloading(iteration)
                try:
                    with (
                        self.training_timer("dataloader_train"),
                    ):
                        data_batch = next(dataloader_train_iter)
                except StopIteration:
                    break
                finally:
                    self.callbacks.on_after_dataloading(iteration)
                # If max_iter is reached, exit the training loop.
                if iteration >= self.config.trainer.max_iter:
                    _end_training = True
                    break
                # Move all tensors in the data batch to GPU device.
                data_batch = to(data_batch, device="cuda")
                # The actual training step.
                self.callbacks.on_training_step_start(
                    self.model, data_batch, iteration=iteration
                )
                self.callbacks.on_training_step_batch_start(
                    self.model, data_batch, iteration=iteration
                )
                if not self.model.training:
                    self.model_ddp.train()
                assert self.model_ddp.training, "model_ddp is not in training mode."
                assert self.model.training, "model is not in training mode."
                output_batch, loss, grad_accum_iter = self.training_step(
                    self.model_ddp,
                    optimizer,
                    scheduler,
                    grad_scaler,
                    data_batch,
                    iteration=iteration,
                    grad_accum_iter=grad_accum_iter,
                )
                self.callbacks.on_training_step_batch_end(
                    self.model, data_batch, output_batch, loss, iteration=iteration
                )
                # If the gradients are still being accumulated, continue to load the next training batch.
                if grad_accum_iter != 0:
                    continue
                # Do the following when an actual optimizer (update) step has been made.
                iteration += 1
                # Save checkpoint.
                if iteration % self.config.checkpoint.save_iter == 0:
                    self.checkpointer.save(
                        self.model,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        iteration=iteration,
                    )
                self.callbacks.on_training_step_end(
                    self.model, data_batch, output_batch, loss, iteration=iteration
                )
                # Validation.
                if (
                    self.config.trainer.run_validation
                    and iteration % self.config.trainer.validation_iter == 0
                ):
                    self.validate(self.model, self.dataloader_val, iteration=iteration)
                # This iteration is successful; reset the timeout signal.
                signal.alarm(self.config.trainer.timeout_period)
            if _end_training:
                break
        logger.info("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(
                self.model, optimizer, scheduler, grad_scaler, iteration=iteration
            )
        self.callbacks.on_train_end(self.model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    def _training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the wfm model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Obtain text embeddings online
        if (
            self.config.model.text_encoder_config is not None
            and self.config.model.text_encoder_config.compute_online
        ):
            text_embeddings = self.compute_text_embeddings_online(data_batch)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        if self.config.model.rl.enabled:
            # Grpo training step
            return self.rl_training_step(data_batch, iteration)

        self.model._update_train_stats(data_batch)

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.model.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.model.draw_training_sigma_and_epsilon(
            x0_B_C_T_H_W.size(), condition
        )

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = (
            self.model.broadcast_split_for_model_parallelsim(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
            )
        )
        output_batch, kendall_loss, _, _ = (
            self.model.compute_loss_with_epsilon_and_sigma(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
            )
        )

        if self.model.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.model.loss_scale
        elif self.model.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.model.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.model.loss_reduce}")

        return output_batch, kendall_loss * self.loss_scale

    def rl_training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single GRPO training step that computes log-probability ratios.

        This method follows the same data processing as training_step but computes
        log-probability ratios for GRPO training instead of standard wfm loss.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed log-probability ratio for the GRPO training step.
        """
        # Update reference model if needed
        if (
            self.model.net_ref is not None
            and self.config.model.rl.update_ref_every_iter > 0
            and iteration % self.config.model.rl.update_ref_every_iter == 0
        ):
            self.model.net_ref_worker.copy_to(
                src_model=self.model.net, tgt_model=self.model.net_ref
            )
            logger.info(f"[WFM RL] Updated reference model at iteration {iteration}")

        # Data synchronization
        data_batch = self.model._sync_data_across_rollouts(data_batch)

        # Check if we need to generate rollout trajectory
        if self.model.inference_infos.rl_cached_trajectory is None:
            # Generate new rollout trajectory
            self.model.inference_infos.x0_fn = self.model.get_x0_fn_from_batch(
                data_batch, guidance=self.config.model.rl.guidance
            )
            self.get_rollout_samples(data_batch, iteration)

        self.model._update_train_stats(data_batch)

        logger.info(
            f"[WFM RL] Performing GRPO training step at iteration {iteration} timesteps {self.model.inference_infos.timesteps}."
        )

        # Get trajectory data for current step
        rl_inference_result = self.model.inference_infos.rl_cached_trajectory[
            self.model.inference_infos.timesteps
        ]

        # Extract trajectory information
        noise_x = rl_inference_result["noise_x"]  # x_t
        sigma_cur = rl_inference_result["sigma"]  # σ_t
        sigma_next = rl_inference_result["sigma_next"]  # σ_{t-1}
        mu_old = rl_inference_result["mu_old"]  # reference mean μ_ref
        sample = rl_inference_result["sample"]  # x_{t-1} (actual next sample)

        # Compute squared differences
        mu_current = self.model.get_mu_from_model(
            self.model.inference_infos.x0_fn,
            noise_x,
            sigma_cur,
            sigma_next,
            **rl_inference_result["kwargs"],
        )
        diff_current = (sample - mu_current) ** 2
        diff_old = (sample - mu_old) ** 2

        # eta control the variance of the gaussian distribution
        eta = min(
            self.model.rl_solver_cfg.s_churn / (self.config.model.rl.sample_steps + 1),
            math.sqrt(1.2) - 1,
        )

        std = (
            sigma_next * math.sqrt(eta**2 + 2 * eta) * self.model.rl_solver_cfg.s_noise
        )

        # Compute log-probability ratio
        # we do not gather here to avoid gradient lost.
        # log_prob_ratio = -(diff_current - diff_old) / (2 * std ** 2)
        log_prob_ratio = -(diff_current - diff_old)  # TODO: rethink correct balance
        ratio = torch.exp(
            log_prob_ratio.mean(dim=[1, 2, 3, 4])
        )  # log p_theta / log p_ref

        if self.config.model.rl.kl_beta > 0:
            kl_loss = (
                (rl_inference_result["mu_ref"].to(mu_current.dtype) - mu_current)
                .pow(2)
                .mean()
            )
        else:
            kl_loss = torch.zeros_like(ratio).mean()

        if self.config.model.rl.data_beta > 0 and (
            not self.config.model.rl.data_on_first_only
            or self.model.inference_infos.timesteps == 0
        ):
            _, x0_B_C_T_H_W, condition = self.model.get_data_and_condition(data_batch)

            if self.config.model.rl.use_rl_sigma_and_noise:
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, _ = (
                    self.model.broadcast_split_for_model_parallelsim(
                        x0_B_C_T_H_W,
                        condition,
                        self.model.inference_infos.gaussian_noise,
                        None,
                    )
                )
                sigma = sigma_cur.reshape(x0_B_C_T_H_W.shape[0], 1)

            else:
                sigma_B_T, epsilon_B_C_T_H_W = (
                    self.model.draw_training_sigma_and_epsilon(
                        x0_B_C_T_H_W.size(), condition
                    )
                )
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma = (
                    self.model.broadcast_split_for_model_parallelsim(
                        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
                    )
                )

            forward_batch, kendall_loss, _, _ = (
                self.model.compute_loss_with_epsilon_and_sigma(
                    x0_B_C_T_H_W,
                    condition,
                    epsilon_B_C_T_H_W,
                    sigma,
                )
            )
            data_loss = kendall_loss.mean()
        else:
            forward_batch = None
            data_loss = torch.zeros_like(ratio).mean()

        # Start loss computation
        rewards = self.model._sync_rewards_across_rollouts()
        N_RM = len(rewards)
        advantages = sum([r["advantage"] for r in rewards.values()]) / N_RM

        if self.config.model.rl.exp_reward:
            advantages = -torch.exp(-advantages)

        advantages = torch.clamp(
            advantages,
            min=self.config.model.rl.reward_config.adv_clip_min,
            max=self.config.model.rl.reward_config.adv_clip_max,
        )

        unclipped_loss = -advantages * ratio
        clipped_loss = (
            -torch.clamp(
                ratio,
                min=1 - self.config.model.rl.clip_ratio,
                max=1 + self.config.model.rl.clip_ratio,
            )
            * advantages
        )
        policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()

        loss = (
            policy_loss
            + self.config.model.rl.kl_beta * kl_loss
            + self.config.model.rl.data_beta * data_loss
        )

        # Prepare output batch for logging/debugging
        output_batch = {
            "sigma": sigma_next,
            "log_prob_ratio": log_prob_ratio.mean(),
            "kl_loss": kl_loss,
            "data_loss": data_loss,
            "policy_loss": policy_loss,
            "std": std.mean(),
            "eta": eta,
            "trajectory_step": rl_inference_result["denoising_step"],
            "mse_loss": loss,
            "edm_loss": log_prob_ratio.mean(),
            "edm_loss_per_frame": log_prob_ratio.mean(dim=[1, 3, 4]),
            "x0": self.model.inference_infos.final_sample,
        }

        if forward_batch is not None:
            output_batch.update(forward_batch)

        inference_out = self.model.inference_infos.final_sample
        if self.model.net.is_context_parallel_enabled:
            inference_out = cat_outputs_cp(
                inference_out,
                seq_dim=2,
                cp_group=self.model.get_context_parallel_group(),
            )

        output_batch["inference_out"] = inference_out

        is_image_batch = self.model.is_image_batch(data_batch)
        output_batch["raw_data"] = data_batch[
            self.model.input_image_key if is_image_batch else self.model.input_data_key
        ]
        output_batch["ai_caption"] = data_batch["ai_caption"]

        output_batch["reward_instance"] = {
            k: r["reward"].mean() for k, r in rewards.items()
        }
        output_batch["reward_instance"].update(
            {k + "_std": r["std_reward"].mean() for k, r in rewards.items()}
        )
        output_batch["reward_instance"].update(
            {k + "_valid_rate": r["valid_rate"].mean() for k, r in rewards.items()}
        )

        output_batch["reward_mean"] = (
            sum([r["reward"].mean() for r in rewards.values()]) / N_RM
        )

        output_batch["data_batch"] = data_batch

        # Update trajectory step
        self.model.inference_infos.timesteps += 1
        if self.model.inference_infos.timesteps >= len(self.config.model.rl.train_on):
            # lms: reset the inference info and for the next step, rollout will populate it again
            self.model.inference_infos = InferenceInfos()  # reset the inference infos

        logger.info(
            f"iteration: {iteration}, denoising_step: {rl_inference_result['denoising_step']}, loss: {loss.mean()}, advantage: {advantages.mean()}, ratio: {ratio.mean()}, kl_loss: {kl_loss.mean()}, data_loss: {data_loss.mean()}"
        )
        return output_batch, loss * self.model.loss_scale

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # Calculate effective gradient accumulation for GRPO
        effective_grad_accum_iter = self.config.trainer.grad_accum_iter

        # Check for GRPO configuration in the model
        if (
            hasattr(model_ddp, "config")
            and hasattr(model_ddp.config, "rl")
            and model_ddp.config.rl.enabled
        ):
            assert (
                model_ddp.config.rl.sample_steps > 0
            ), "GRPO sample steps must be greater than 0"
            assert (
                len(model_ddp.config.rl.train_on) > 0
            ), "GRPO train on must be set to a non-empty list"
            effective_grad_accum_iter *= len(model_ddp.config.rl.train_on)

        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(
            model_ddp, grad_accum_iter == effective_grad_accum_iter - 1
        ):
            self.callbacks.on_before_forward(iteration=iteration)
            with self.training_timer("forward"):
                output_batch, loss = self._training_step(data, iteration)
                # output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                loss_scaled = grad_scaler.scale(loss / effective_grad_accum_iter)
                loss_scaled.backward()
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_after_backward()
                else:
                    model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == effective_grad_accum_iter:
            with self.training_timer("optimizer_step"):
                self.callbacks.on_before_optimizer_step(
                    model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                self.callbacks.on_before_zero_grad(
                    model_ddp, optimizer, scheduler, iteration=iteration
                )
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_before_zero_grad(
                        optimizer, scheduler, iteration=iteration
                    )
                else:
                    model_ddp.on_before_zero_grad(
                        optimizer, scheduler, iteration=iteration
                    )
                optimizer.zero_grad(set_to_none=True)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

    @torch.no_grad()
    def validate(
        self, model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        """Validate on the full validation dataset.

        Args:
            model (CosmosVisionGenModel): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        # Evaluate on the full validation set.
        with ema_scope(model, enabled=model.config.ema.enabled):
            for val_iter, data_batch in enumerate(dataloader_val):
                if (
                    self.config.trainer.max_val_iter is not None
                    and val_iter >= self.config.trainer.max_val_iter
                ):
                    break
                data_batch = to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(
                    model, data_batch, iteration=iteration
                )
                output_batch, loss = model.validation_step(data_batch, iteration)
                self.callbacks.on_validation_step_end(
                    model, data_batch, output_batch, loss, iteration=iteration
                )
        self.callbacks.on_validation_end(model, iteration=iteration)

    def main_loop(self):
        self.train()

    def compute_text_embeddings_online(
        self, data_batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute text embeddings for the given prompts.
        """
        assert self.model.text_encoder is not None, "Text encoder is not initialized"

        # Tokenize prompts
        input_ids_batch = []

        for sample_idx in range(len(data_batch[self.model.input_caption_key])):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": data_batch[self.model.input_caption_key][
                                sample_idx
                            ],
                        }
                    ],
                },
            ]
            tokenizer_output = self.model.text_encoder.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            input_ids = tokenizer_output["input_ids"]
            pad_id = self.model.text_encoder.tokenizer.pad_id

            # Do padding or truncation
            if NUM_EMBEDDING_PADDING_TOKENS > len(input_ids):
                # Do padding:
                pad_len = NUM_EMBEDDING_PADDING_TOKENS - len(input_ids)
                input_ids = input_ids.tolist() + [pad_id] * pad_len
            else:
                # Do truncation:
                input_ids = input_ids.tolist()[:NUM_EMBEDDING_PADDING_TOKENS]
            input_ids = torch.LongTensor(input_ids).to(device="cuda")
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        # Compute text embeddings
        with torch.no_grad():
            _, outputs_batch = self.model.text_encoder(input_ids_batch, {})
        hidden_states = outputs_batch["hidden_states"]

        # # Skip the embeddings of the system prompt
        # hidden_states = hidden_states[:, num_system_prompt_tokens:]

        # Now compute the normalized embeddings
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = self.model.mean_normalize(hidden_states[layer_idx])
            normalized_hidden_states.append(normalized_state)

        text_embeddings_for_wfm = None
        if self.config.model.text_encoder_config.embedding_concat_strategy == str(
            EmbeddingConcatStrategy.FULL_CONCAT
        ):
            text_embeddings_for_wfm = torch.cat(normalized_hidden_states, dim=-1)
        elif self.config.model.text_encoder_config.embedding_concat_strategy == str(
            EmbeddingConcatStrategy.MEAN_POOLING
        ):
            # Stack the normalized hidden states and calculate the mean
            text_embeddings_for_wfm = torch.stack(normalized_hidden_states)
            text_embeddings_for_wfm = text_embeddings_for_wfm.mean(dim=0)
        elif self.config.model.text_encoder_config.embedding_concat_strategy == str(
            EmbeddingConcatStrategy.POOL_EVERY_N_LAYERS_AND_CONCAT
        ):
            # Split the l
            n_layers_per_group = (
                self.config.model.text_encoder_config.n_layers_per_group
            )
            text_embeddings_for_wfm = []
            for i in range(0, len(normalized_hidden_states), n_layers_per_group):
                group_embeddings = normalized_hidden_states[i : i + n_layers_per_group]
                group_embedding = torch.stack(group_embeddings)
                group_embedding = group_embedding.mean(dim=0)
                text_embeddings_for_wfm.append(group_embedding)
            text_embeddings_for_wfm = torch.cat(text_embeddings_for_wfm, dim=-1)
        else:
            raise ValueError(
                f"Invalid embedding_concat_strategy: {self.config.model.text_encoder_config.embedding_concat_strategy}"
            )

        return text_embeddings_for_wfm

    def get_rollout_samples(self, data_batch: dict[str, torch.Tensor], iteration: int):
        """
        GRPO rollout generation with distributed data sync and trajectory caching.
        """
        _, x0_B_C_T_H_W, condition = self.model.get_data_and_condition(data_batch)
        logger.info(
            f"[WFM RL] Performing rollout inference at iteration {iteration}. Data shape to broadcast: {x0_B_C_T_H_W.shape}"
        )

        context = (
            nullcontext()
            if self.config.model.rl.on_policy
            else ref_scope(
                model=self.model.net,
                ref_model=self.model.net_ref,
                ref_worker=self.model.net_ref_worker,
                enabled=True,
                context="GRPO_rollout",
            )
        )
        logger.info(f"[WFM RL] Using {context} for rollout inference")

        with torch.no_grad():
            with context:
                trajs = []

                def callback_fn(**kwargs):
                    # Notice that the saved tensors are context-paralleled
                    # We do not cat them here because we will split them in the training step
                    data = {
                        "noise_x": kwargs["input_x_B_StateShape"].to(torch.float32),
                        "sigma": kwargs["sigma_cur_0"].to(torch.float32),
                        "sigma_next": kwargs["sigma_next_0"].to(torch.float32),
                        "x0_pred": kwargs["x0_pred_B_StateShape"].to(torch.float32),
                        "mu_old": kwargs["output_x_B_StateShape"].to(torch.float32),
                    }
                    trajs.append(data)

                assert not self.model.is_image_batch(
                    data_batch
                ), "GRPO only supports video data"
                x_sigma = arch_invariant_rand(
                    x0_B_C_T_H_W.shape,
                    torch.float32,
                    self.model.tensor_kwargs["device"],
                    data_batch["seed"],
                )
                x_sigma_max = x_sigma * self.model.sde.sigma_max
                sample = self.rollout_runner.rollout_generation(
                    data_batch,
                    guidance=self.config.model.rl.guidance,
                    num_steps=self.config.model.rl.sample_steps,
                    x_sigma_max=x_sigma_max,
                    callback_fns=[callback_fn],
                )
                # synchronize sample across context parallel group
                # TODO: current implementation supports video data only, because for image, the sample is different within one context parallel group (net.is_context_parallel_enabled is False)
                # As a result, cp-saved trajectories correspond to different samples in one context parallel group.
                # Future change: we should broadcast the entire trajectory instead of sample.

                if hasattr(self.model, "decode"):
                    generated_samples = self.model.decode(sample)
                else:
                    generated_samples = sample

                # update reward meta info
                logger.info("[WFM RL] Start computing rewards ...")
                self.model.inference_infos.rewards = {}
                self.model.inference_infos.is_rewards_synced = False

                for k, v in self.model.reward_models.items():
                    valid = is_tp_cp_pp_rank0()
                    if valid:
                        try:
                            reward, bg_id = v.compute_reward(
                                data_batch,
                                generated_samples,
                                latents=sample,
                                is_async=True,
                            )
                        except Exception as e:
                            logger.warning(
                                f"[WFM RL] Reward model {k} failed: {e}. Ignore current rollout.",
                            )
                            valid = False
                    if not valid:
                        reward, bg_id = (
                            torch.zeros((sample.shape[0],), device=sample.device),
                            None,
                        )
                    self.model.inference_infos.rewards[k] = {
                        "reward": reward,
                        "valid": valid,
                        "bg_id": bg_id,
                    }

                logger.info("[WFM RL] Finished computing rewards ...")
                torch.cuda.empty_cache()

                if self.model.net.is_context_parallel_enabled:
                    sample = split_inputs_cp(
                        x=sample,
                        seq_dim=2,
                        cp_group=self.model.get_context_parallel_group(),
                    )
        self.model.inference_infos.gaussian_noise = x_sigma
        self.model.inference_infos.final_sample = sample.to(torch.float32)
        # Add sample to trajectories (next step's noise_x)
        for i in range(len(trajs)):
            trajs[i]["denoising_step"] = i
            trajs[i]["sample"] = (
                trajs[i + 1]["noise_x"]
                if i < len(trajs) - 1
                else self.model.inference_infos.final_sample
            )
            if self.model.rl_solver_cfg.is_multi:
                trajs[i]["kwargs"] = {
                    "x0_preds": [(trajs[i - 1]["x0_pred"], trajs[i - 1]["sigma"])]
                    if i > 0
                    else None
                }
            else:
                trajs[i]["kwargs"] = {}

        self.model.inference_infos.rl_cached_trajectory = [
            w for i, w in enumerate(trajs) if i in self.config.model.rl.train_on
        ]
        logger.info(
            "[WFM RL] Finished rolling out samples. Preparing to cache trajectory ..."
        )

        self.model.inference_infos.rl_cached_trajectory = [
            w for i, w in enumerate(trajs) if i in data_batch["train_on"]
        ]

        if (
            self.config.model.rl.kl_beta > 0
        ):  # compute mu_ref here to reduec memory peak
            logger.info(
                f"[WFM RL] Computing KL loss with reference model, kl_beta: {self.config.model.rl.kl_beta}"
            )
            with ref_scope(
                model=self.model.net,
                ref_model=self.model.net_ref,
                ref_worker=self.model.net_ref_worker,
                enabled=True,
                context="reference_model_KL_loss",
            ):
                for (
                    rl_inference_result
                ) in self.model.inference_infos.rl_cached_trajectory:
                    mu_ref = self.model.get_mu_from_model(
                        self.model.inference_infos.x0_fn,
                        rl_inference_result["noise_x"],
                        rl_inference_result["sigma"],
                        rl_inference_result["sigma_next"],
                        **rl_inference_result["kwargs"],
                    ).detach()
                    rl_inference_result["mu_ref"] = mu_ref

        self.model.inference_infos.timesteps = 0
        torch.cuda.empty_cache()

        logger.info(
            f"[WFM RL] Cached trajectory with {len(self.model.inference_infos.rl_cached_trajectory)} steps at {data_batch['train_on']}, sample shape (with parallelism): {sample.shape}"
        )
