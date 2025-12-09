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

from __future__ import annotations

import gc
import numpy as np
import math
import os
import pandas as pd
import psutil
import pynvml
import pytz
import sys
import time
import tqdm
import wandb
import warnings
from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from einops import rearrange, repeat
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms.functional as torchvision_F
from torch.distributed import ProcessGroup

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.wandb_logger import init_wandb
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig
from cosmos_rl.utils.wfm import distributed, utils
from cosmos_rl.utils.wfm.io.easy_io import easy_io
from cosmos_rl.utils.wfm.visualize.video import save_img_or_video
from cosmos_rl.tools.dataset.wfm.data_sources.item_datasets_for_validation import (
    get_itemdataset_option,
)

from cosmos_rl.utils.wfm.distributed import get_global_parallel_dims


class Callback:
    """The base class for all callbacks.

    All callbacks should inherit from this class and adhere to the established method names and signatures.
    """

    def __init__(self, config: CosmosVisionGenConfig = None, trainer=None):
        """Initializes a Callback object.

        Args:
            config (Optional[Config]): The configuration object for the CosmosVisionGen codebase, if available.
            trainer (Optional[CosmosVisionGenTrainer]): The main trainer handling the training loop, if available.

        Notes:
            The config and trainer parameters are optional to maintain backward compatibility.
            In future releases, these parameters will be removed. Upon using these parameters, a deprecation
            warning will be issued.

        """
        if config is not None or trainer is not None:
            warnings.warn(
                "The 'config' and 'trainer' parameters are deprecated and will be removed in a future release. "
                "Please update your code to create Callback instances without these parameters.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.parallel_dims = get_global_parallel_dims()
        del config, trainer

    def on_train_start(self, model, iteration: int = 0) -> None:
        pass

    def on_training_step_start(
        self, model, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        """
        Called before the training step, for each batch. This is paired with on_training_step_end() but note that
        when using gradient accumulation, while on_training_step_end() is only called when the optimizer is updated,
        this function is called for every batch.
        Use on_training_step_batch_start and on_training_step_batch_end if you need callbacks that are called
        for every batch, albeit with the same iteration number.
        FIXME - should this either be deprecated, or called only when a new training step is started after having updated
        the optimizer?
        """
        pass

    def on_training_step_batch_start(
        self, model, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        """
        Called before the training step, for each batch, similarly to on_training_step_start(). This function is paired with
        on_training_step_batch_end(), and both functions are called for every batch even when using gradient accumulation.
        Note that the iteration is only updated when the optimizer is updated, and therefore it may be the same for multiple invocations.
        """
        pass

    def on_before_forward(self, iteration: int = 0) -> None:
        pass

    def on_after_forward(self, iteration: int = 0) -> None:
        pass

    def on_before_backward(
        self,
        model_ddp: distributed.DistributedDataParallel,
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        pass

    def on_after_backward(
        self, model_ddp: distributed.DistributedDataParallel, iteration: int = 0
    ) -> None:
        pass

    def on_before_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_after_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_optimizer_init_start(self) -> None:
        pass

    def on_optimizer_init_end(self) -> None:
        pass

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_training_step_batch_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """
        Called at the end of a training step for every batch even when using gradient accumulation.
        This is paired with on_training_step_batch_start(). Note that the iteration is only updated when the optimizer is updated,
        and therefore it may be the same for multiple batches.
        """
        pass

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """
        Called at the end of a training step, but note that when using gradient accumulation, this is only called
        when the optimizer is updated, and the iteration incremented, whereas on_training_step_start is called every time.
        Use on_training_step_batch_start and on_training_step_batch_end if you need callbacks that are called
        for every batch.
        """
        pass

    def on_validation_start(
        self, model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        pass

    def on_validation_step_start(
        self, model, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        pass

    def on_validation_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        pass

    def on_validation_end(self, model, iteration: int = 0) -> None:
        pass

    def on_load_checkpoint_start(self, model) -> None:
        pass

    def on_load_checkpoint_end(
        self, model, iteration: int = 0, checkpoint_path: Optional[str] = None
    ) -> None:
        pass

    def on_load_checkpoint(self, model, state_dict: dict[Any]) -> None:
        """
        Called when checkpoint loading is about to start, but after on_save_checkpoint_start().
        FIXME - why do we need this callback, can't we just use on_save_checkpoint_start()?
        """
        pass

    def on_save_checkpoint_start(self, model, iteration: int = 0) -> None:
        """
        Called when checkpoint saving is about to start.
        """
        pass

    def on_save_checkpoint_end(self, model, iteration: int = 0) -> None:
        """
        Called when the synchronous part of checkpointing is finished, this function can be used
        along with on_save_checkpoint_start() to measure the exposed (synchronous) checkpoint time.
        Note that for asynchronous checkpoint, the checkpoint may still be ongoing, so this function
        does not mean the checkpoint is finished for the asynchronous case, use on_save_checkpoint_success()
        for that.
        """
        pass

    def on_save_checkpoint_success(
        self, iteration: int = 0, elapsed_time: float = 0
    ) -> None:
        """
        Called when checkpoint saving is fully finished, and succeeded. Not called if checkpoint failed.
        For synchronous checkpoint, it is called at the same time as on_save_checkpoint_end(), but for asynchronous
        checkpoint, it is called after the asynchronous part has also finished. For checkpointers with out-of-process
        checkpointing, this function is called as soon as the notification is received from the checkpointer process,
        which may not be immediately after the checkpoint has completed but later on. Therefore, if you need to measure
        the full checkpoint duration for the asynchronous part, use the elapsed_time parameter, do not measure it directly
        as this would be a significant overestimate.
        """
        pass

    def on_save_checkpoint(self, model, state_dict: dict[Any]) -> None:
        pass

    def on_train_end(self, model, iteration: int = 0) -> None:
        pass

    def on_app_end(self) -> None:
        pass


class EMAModelCallback(Callback):
    """The callback class for tracking EMA model weights."""

    def on_train_start(self, model, iteration: int = 0) -> None:
        # Set up the EMA model weight tracker.
        if model.config.ema.enabled:
            assert hasattr(
                model, "ema"
            ), "EMA should be initialized from CosmosVisionGenModel"
            # EMA model must be kept in FP32 precision.
            model.ema = model.ema.to(dtype=torch.float32)
        else:
            assert not hasattr(model, "ema"), "There should be no EMA initialized."

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Update the EMA model with the new regular weights.
        if model.config.ema.enabled:
            model.ema.update_average(model, iteration)


class ProgressBarCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        self.train_pbar = tqdm.trange(
            self.config.trainer.max_iter, initial=iteration, desc="Training"
        )

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.train_pbar.update()

    @distributed.rank0_only
    def on_validation_start(
        self, model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        if self.config.trainer.max_val_iter is not None:
            num_iter = self.config.trainer.max_val_iter
        else:
            num_iter = len(dataloader_val)
        assert (
            num_iter is not None and num_iter > 0
        ), f"Invalid number of validation iterations: {num_iter}"
        self.val_pbar = tqdm.trange(
            num_iter, desc="Validating", position=1, leave=False
        )

    @distributed.rank0_only
    def on_validation_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.val_pbar.update()

    @distributed.rank0_only
    def on_validation_end(self, model, iteration: int = 0) -> None:
        self.val_pbar.close()

    @distributed.rank0_only
    def on_train_end(self, model, iteration: int = 0) -> None:
        self.trainer.checkpointer.finalize()  # FIXME(chenhsuanl): where does this belong?
        self.train_pbar.close()


class IterationLoggerCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        # self.train_pbar = tqdm.trange(self.config.trainer.max_iter, initial=iteration, desc="Training")
        self.start_iteration_time = time.time()
        self.elapsed_iteration_time = 0

    @distributed.rank0_only
    def on_training_step_start(
        self, model, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        self.start_iteration_time = time.time()

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # FIXME - this is not correct when using gradient accumulation since self.start_iteration_time is updated every batch
        # but this is only called when the optimizer is updated, so it's only the time for the last batch.
        self.elapsed_iteration_time += time.time() - self.start_iteration_time

        if iteration % self.config.trainer.logging_iter == 0:
            avg_time = self.elapsed_iteration_time / self.config.trainer.logging_iter
            logger.info(
                f"Iteration: {iteration}, average iter time: {avg_time:2f}, total loss {loss.item():4f}"
            )

            self.elapsed_iteration_time = 0


class BaseWandBCallback(Callback):
    """The callback class for logging to Weights and Biases (W&B).

    By default, WandBCallback logs the following training stats to W&B every config.trainer.logging_iter:
    - iteration: The current iteration number (useful for visualizing the training progress over time).
    - train/loss: The computed overall loss in the training batch.
    - optim/lr: The current learning rate and weight decay for each optimizer group
    - timer/*: The averaged timing results of each code block recorded by trainer.training_timer.
    For validation, WandBCallback logs:
    - val/loss: The computed overall loss in the validation dataset.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_loss = 0  # variable to store the accumulated training loss

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        init_wandb(self.config)
        config = self.config
        job_local_path = config.job.path_local
        # read optional job_env saved by `log_reproducible_setup`
        if os.path.exists(f"{job_local_path}/job_env.yaml"):
            job_info = easy_io.load(f"{job_local_path}/job_env.yaml")
            if wandb.run:
                wandb.run.config.update(
                    {f"JOB_INFO/{k}": v for k, v in job_info.items()},
                    allow_val_change=True,
                )

        if (
            os.path.exists(f"{config.job.path_local}/config.yaml")
            and "SLURM_LOG_DIR" in os.environ
        ):
            easy_io.copyfile(
                f"{config.job.path_local}/config.yaml",
                os.path.join(os.environ["SLURM_LOG_DIR"], "config.yaml"),
            )

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            info = {}
            info["sample_counter"] = getattr(self.trainer, "sample_counter", iteration)

            for i, param_group in enumerate(optimizer.param_groups):
                info[f"optim/lr_{i}"] = param_group["lr"]
                info[f"optim/weight_decay_{i}"] = param_group["weight_decay"]

            wandb.log(info, step=iteration)

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> (
        None
    ):  # Log the timing results (over a number of iterations) and the training loss.
        self.training_loss += loss.detach().float()
        if iteration % self.config.trainer.logging_iter == 0:
            timer_results = self.trainer.training_timer.compute_average_results()
            avg_loss = self.training_loss / self.config.trainer.logging_iter
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            if distributed.is_rank0():
                info = {f"timer/{key}": value for key, value in timer_results.items()}
                info["train/loss"] = avg_loss.item()
                info["iteration"] = iteration
                info["sample_counter"] = getattr(
                    self.trainer, "sample_counter", iteration
                )
                wandb.log(info, step=iteration)
            self.trainer.training_timer.reset()
            self.training_loss = 0

    def on_validation_start(
        self, model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
        )

    def on_validation_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        batch_size = utils.get_data_batch_size(data_batch)
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

    def on_validation_end(self, model, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            logger.info(f"Validation loss (iteration {iteration}): {loss}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_train_end(self, model, iteration: int = 0) -> None:
        wandb.finish()


@dataclass
class _LossRecord:
    loss: float = 0
    iter_count: int = 0
    edm_loss: float = 0

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0
        self.edm_loss = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_loss = self.loss / self.iter_count
            avg_edm_loss = self.edm_loss / self.iter_count
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_edm_loss, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss.item()
            avg_edm_loss = avg_edm_loss.item()
        else:
            avg_loss = 0
            avg_edm_loss = 0
        self.reset()
        return avg_loss, avg_edm_loss


class WandbCallback(Callback):
    """
    This callback is used to log the loss, average loss over logging_iter_multipler, and unstable counts of image and video to wandb.
    """

    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.train_image_log = _LossRecord()
        self.train_video_log = _LossRecord()
        self.final_loss_log = _LossRecord()

        self.img_unstable_count = torch.zeros(1, device="cuda")
        self.video_unstable_count = torch.zeros(1, device="cuda")

        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert (
            self.logging_iter_multipler > 0
        ), "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = (
            f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        )
        self.name = "wandb_loss_log" + self.wandb_extra_tag

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        init_wandb(self.config)
        config = self.config
        job_local_path = config.job.path_local
        # read optional job_env saved by `log_reproducible_setup`
        if os.path.exists(f"{job_local_path}/job_env.yaml"):
            job_info = easy_io.load(f"{job_local_path}/job_env.yaml")
            if wandb.run:
                wandb.run.config.update(
                    {f"JOB_INFO/{k}": v for k, v in job_info.items()},
                    allow_val_change=True,
                )

        if (
            os.path.exists(f"{config.job.path_local}/config.yaml")
            and "SLURM_LOG_DIR" in os.environ
        ):
            easy_io.copyfile(
                f"{config.job.path_local}/config.yaml",
                os.path.join(os.environ["SLURM_LOG_DIR"], "config.yaml"),
            )

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            info = {}
            info["sample_counter"] = getattr(self.trainer, "sample_counter", iteration)

            for i, param_group in enumerate(optimizer.param_groups):
                info[f"optim/lr_{i}"] = param_group["lr"]
                info[f"optim/weight_decay_{i}"] = param_group["weight_decay"]

            wandb.log(info, step=iteration)

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            logger.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}"
            )

        if not skip_update_due_to_unstable_loss:
            if model.is_image_batch(data_batch):
                self.train_image_log.loss += loss.detach().float()
                self.train_image_log.iter_count += 1
                self.train_image_log.edm_loss += (
                    output_batch["edm_loss"].detach().float()
                )
            else:
                self.train_video_log.loss += loss.detach().float()
                self.train_video_log.iter_count += 1
                self.train_video_log.edm_loss += (
                    output_batch["edm_loss"].detach().float()
                )

            self.final_loss_log.loss += loss.detach().float()
            self.final_loss_log.iter_count += 1
            self.final_loss_log.edm_loss += output_batch["edm_loss"].detach().float()
        else:
            if model.is_image_batch(data_batch):
                self.img_unstable_count += 1
            else:
                self.video_unstable_count += 1

        if (
            iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler)
            == 0
        ):
            if self.logging_iter_multipler > 1:
                timer_results = {}
            else:
                timer_results = self.trainer.training_timer.compute_average_results()
            avg_image_loss, avg_image_edm_loss = self.train_image_log.get_stat()
            avg_video_loss, avg_video_edm_loss = self.train_video_log.get_stat()
            avg_final_loss, avg_final_edm_loss = self.final_loss_log.get_stat()

            dist.all_reduce(self.img_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.video_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                info = {f"timer/{key}": value for key, value in timer_results.items()}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/image_loss": avg_image_loss,
                        f"train{self.wandb_extra_tag}/image_edm_loss": avg_image_edm_loss,
                        f"train{self.wandb_extra_tag}/video_loss": avg_video_loss,
                        f"train{self.wandb_extra_tag}/video_edm_loss": avg_video_edm_loss,
                        f"train{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"train{self.wandb_extra_tag}/edm_loss": avg_final_edm_loss,
                        f"train{self.wandb_extra_tag}/img_unstable_count": self.img_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/video_unstable_count": self.video_unstable_count.item(),
                        "iteration": iteration,
                        "sample_counter": getattr(
                            self.trainer, "sample_counter", iteration
                        ),
                    }
                )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)
            if self.logging_iter_multipler == 1:
                self.trainer.training_timer.reset()

            # reset unstable count
            self.img_unstable_count.zero_()
            self.video_unstable_count.zero_()

    def on_validation_start(
        self, model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
        )

    def on_validation_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        batch_size = utils.get_data_batch_size(data_batch)
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

    def on_validation_end(self, model, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            logger.info(f"Validation loss (iteration {iteration}): {loss}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_train_end(self, model, iteration: int = 0) -> None:
        wandb.finish()


class BaseLowPrecisionCallback(Callback):
    """The callback class handling low precision training"""

    def __init__(self, update_iter: int):
        self.update_iter = update_iter

    def on_train_start(self, model, iteration: int = 0) -> None:
        assert model.precision in [
            torch.bfloat16,
            torch.float16,
            torch.half,
        ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision

    def on_training_step_start(
        self, model, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_validation_step_start(
        self, model, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        if iteration % self.update_iter == 0:
            if getattr(optimizer, "master_weights", False):
                params, master_params = [], []
                for group, group_master in zip(
                    optimizer.param_groups, optimizer.param_groups_master
                ):
                    for p, p_master in zip(group["params"], group_master["params"]):
                        params.append(distributed.get_local_tensor_if_DTensor(p.data))
                        master_params.append(p_master.data)
                torch._foreach_copy_(params, master_params)


class LowPrecisionCallback(BaseLowPrecisionCallback):
    """
    Config with non-primitive type makes it difficult to override the option.
    The callback gets precision from model.precision instead.
    It also auto disabled when using fp32.
    """

    def __init__(self, config=None, trainer=None, update_iter: int = 1):
        self.config = config
        self.trainer = trainer
        self.update_iter = update_iter

    def on_train_start(self, model, iteration: int = 0) -> None:
        if model.precision == torch.float32:
            logger.critical("Using fp32, should disable master weights.")
            self.update_iter = sys.maxsize
        else:
            assert model.precision in [
                torch.bfloat16,
                torch.float16,
                torch.half,
            ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision


@torch.jit.script
def _fused_nan_to_num(params: List[torch.Tensor]):
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)


@dataclass
class _MagnitudeRecord:
    state: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.state = 0
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        self.state += cur_state
        self.iter_count += 1

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_state = self.state / self.iter_count
            avg_state = avg_state.item()
        else:
            avg_state = 0
        self.reset()
        return avg_state


class GradClip(Callback):
    def __init__(self, clip_norm=1.0, force_finite: bool = True):
        self.clip_norm = clip_norm
        self.force_finite = force_finite

        self.img_mag_log = _MagnitudeRecord()
        self.video_mag_log = _MagnitudeRecord()
        self._cur_state = None

    def on_training_step_start(
        self, model, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if model.is_image_batch(data_batch):
            self._cur_state = self.img_mag_log
        else:
            self._cur_state = self.video_mag_log

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler
        if isinstance(model_ddp, distributed.DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp
        params = []
        # TODO: (qsh) can merge two into one iteration. possible faster
        if self.force_finite:
            for param in model.parameters():
                if param.grad is not None:
                    params.append(param.grad)
                    # torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            _fused_nan_to_num(params)

        total_norm = model.clip_grad_norm_(self.clip_norm)

        self._cur_state.update(total_norm)
        if iteration % self.config.trainer.logging_iter == 0:
            avg_img_mag, avg_video_mag = (
                self.img_mag_log.get_stat(),
                self.video_mag_log.get_stat(),
            )
            if wandb.run:
                wandb.log(
                    {
                        "clip_grad_norm/image": avg_img_mag,
                        "clip_grad_norm/video": avg_video_mag,
                        "iteration": iteration,
                    },
                    step=iteration,
                )


class EveryN(Callback):
    def __init__(
        self,
        every_n: Optional[int] = None,
        step_size: int = 1,
        barrier_after_run: bool = True,
        run_at_start: bool = False,
    ) -> None:
        """Constructor for `EveryN`.

        Args:
            every_n (int): Frequency with which callback is run during training.
            step_size (int): Size of iteration step count. Default 1.
            barrier_after_run (bool): Whether to have a distributed barrier after each execution. Default True, to avoid timeouts.
            run_at_start (bool): Whether to run at the beginning of training. Default False.
        """
        self.every_n = every_n
        if self.every_n == 0:
            logger.warning(
                f"every_n is set to 0. Callback {self.__class__.__name__} will be invoked only once in the beginning of the training. Calls happens on_training_step_end will be skipped."
            )

        self.step_size = step_size
        self.barrier_after_run = barrier_after_run
        self.run_at_start = run_at_start

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # every_n = 0 is a special case which means every_n_impl will be called only once in the beginning of the training
        if self.every_n != 0:
            trainer = self.trainer
            global_step = iteration // self.step_size
            should_run = (iteration == 1 and self.run_at_start) or (
                global_step % self.every_n == 0
            )  # (self.every_n - 1)
            if should_run:
                logger.debug(
                    f"Callback {self.__class__.__name__} fired on train_batch_end step {global_step}"
                )
                self.every_n_impl(
                    trainer, model, data_batch, output_batch, loss, iteration
                )
                logger.debug(
                    f"Callback {self.__class__.__name__} finished on train_batch_end step {global_step}"
                )
                # add necessary barrier to avoid timeout
                if self.barrier_after_run:
                    distributed.barrier()

    @abstractmethod
    def every_n_impl(
        self,
        trainer,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None: ...


class IterSpeed(EveryN):
    """
    Args:
        hit_thres (int): Number of iterations to wait before logging.
        save_s3 (bool): Whether to save to S3.
        save_s3_every_log_n (int): Save to S3 every n log iterations, which means save_s3_every_log_n n * every_n global iterations.
    """

    def __init__(
        self,
        *args,
        hit_thres: int = 5,
        save_s3: bool = True,
        save_s3_every_log_n: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time = None
        self.hit_counter = 0
        self.hit_thres = hit_thres
        self.save_s3 = save_s3
        self.save_s3_every_log_n = save_s3_every_log_n
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            logger.info(
                f"Iteration {iteration}: "
                f"Hit counter: {self.hit_counter + 1}/{self.hit_thres} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {time.time() - self.last_hit_time:.2f}s"
            )
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @distributed.rank0_only
    def every_n_impl(
        self,
        trainer,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if self.time is None:
            self.time = time.time()
            return
        cur_time = time.time()
        iter_speed = (cur_time - self.time) / self.every_n / self.step_size

        logger.info(
            f"{iteration} : iter_speed {iter_speed:.2f} seconds per iteration | Loss: {loss.item():.4f}"
        )

        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            wandb.log(
                {
                    "timer/iter_speed": iter_speed,
                    "sample_counter": sample_counter,
                },
                step=iteration,
            )
        self.time = cur_time
        if self.save_s3:
            if iteration % (self.save_s3_every_log_n * self.every_n) == 0:
                easy_io.dump(
                    {
                        "iter_speed": iter_speed,
                        "iteration": iteration,
                    },
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )


class ModelParamStats(Callback):
    def __init__(
        self,
        save_s3: bool = False,
    ):
        self.save_s3 = save_s3
        self.name = self.__class__.__name__

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        try:
            model_stat: Dict = model.model_param_stats()
        except AttributeError:
            raise AttributeError(
                "Model does not have model_param_stats method. Please implement it."
            )

        log_str = ""
        for k, v in model_stat.items():
            log_str += f"{k}: {v}\n"
        logger.info(f"Model param Stats:\n{log_str}")

        if self.save_s3:
            easy_io.dump(model_stat, f"s3://rundir/{self.name}.yaml")


class HeartBeat(EveryN):
    """
    A callback that logs a heartbeat message at regular intervals to indicate that the training process is still running.

    Args:
        every_n (int): The frequency at which the callback is invoked.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        update_interval_in_minute (int, optional): The interval in minutes for logging the heartbeat. Defaults to 20 minutes.
        save_s3 (bool, optional): Whether to save the heartbeat information to S3. Defaults to False.
    """

    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        update_interval_in_minute: int = 20,
        save_s3: bool = False,
    ):
        super().__init__(every_n=every_n, step_size=step_size)
        self.name = self.__class__.__name__
        self.update_interval_in_minute = update_interval_in_minute
        self.save_s3 = save_s3
        self.pst = pytz.timezone("America/Los_Angeles")
        self.is_hitted = False

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        self.time = time.time()
        if self.save_s3:
            current_time_pst = datetime.now(self.pst).strftime("%Y_%m_%d-%H_%M_%S")
            info = {
                "iteration": iteration,
                "time": current_time_pst,
            }
            easy_io.dump(info, f"s3://rundir/{self.name}_start.yaml")
            easy_io.dump(info, f"s3://timestamps_rundir/{self.name}_start.yaml")

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if not self.is_hitted:
            self.is_hitted = True
            if distributed.is_rank0():
                self.report(iteration)
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @distributed.rank0_only
    def every_n_impl(
        self,
        trainer,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if time.time() - self.time > 60 * self.update_interval_in_minute:
            self.report(iteration)

    def report(self, iteration: int = 0):
        self.time = time.time()
        if self.save_s3:
            current_time_pst = datetime.now(self.pst).strftime("%Y_%m_%d-%H_%M_%S")
            info = {
                "iteration": iteration,
                "time": current_time_pst,
            }
            easy_io.dump(info, f"s3://rundir/{self.name}.yaml")

    @distributed.rank0_only
    def on_train_end(self, model, iteration: int = 0) -> None:
        if self.save_s3:
            current_time_pst = datetime.now(self.pst).strftime("%Y_%m_%d-%H_%M_%S")
            info = {
                "iteration": iteration,
                "time": current_time_pst,
            }
            easy_io.dump(info, f"s3://rundir/{self.name}_end.yaml")
            easy_io.dump(info, f"s3://timestamps_rundir/{self.name}_end.yaml")


def log_prof_data(
    data_list: List[Dict[str, Any]],
    iteration: int,
) -> Tuple[pd.DataFrame]:
    # Create a table to log data with rank information
    columns = ["iteration", "rank"] + list(data_list[0].keys())
    data = []

    # Initialize dictionaries to store min and max values for each metric
    min_values = {key: float("inf") for key in columns[2:]}
    max_values = {key: float("-inf") for key in columns[2:]}
    sum_values = {key: 0.0 for key in columns[2:]}

    count = 0

    for _rank, prof_data in enumerate(data_list):
        row = [iteration, _rank] + [prof_data[key] for key in columns[2:]]
        data.append(row)
        count += 1

        # Update min, max, and sum values
        for key in columns[2:]:
            min_values[key] = min(min_values[key], prof_data[key])
            max_values[key] = max(max_values[key], prof_data[key])
            sum_values[key] += prof_data[key]

    # Calculate average values
    avg_values = {key: sum_values[key] / count for key in columns[2:]}

    df = pd.DataFrame(data, columns=columns)
    summary_df = pd.DataFrame({"Avg": avg_values, "Max": max_values, "Min": min_values})

    if wandb.run:
        # Log the table
        table = wandb.Table(dataframe=df)
        wandb.log({"DeviceMonitor/prof_data": table}, step=iteration)

        # Log summary statistics
        summary = {}
        for key in columns[2:]:
            summary[f"DeviceMonitor/min_{key}"] = min_values[key]
            summary[f"DeviceMonitor/max_{key}"] = max_values[key]
            summary[f"DeviceMonitor/avg_{key}"] = avg_values[key]

        wandb.log(summary, step=iteration)
    return df, summary_df


class DeviceMonitor(EveryN):
    """
    A callback to monitor device (CPU/GPU) usage and log it at regular intervals.

    Args:
        every_n (int, optional): The frequency at which the callback is invoked. Defaults to 200.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        save_s3 (bool, optional): Whether to save the monitoring data to S3. Defaults to False.
    """

    def __init__(
        self,
        every_n: int = 200,
        step_size: int = 1,
        save_s3: bool = False,
        upload_every_n_mul: int = 1,
        log_memory_detail: bool = True,
    ):
        super().__init__(every_n=every_n, step_size=step_size)
        self.name = self.__class__.__name__
        self.save_s3 = save_s3
        self.s3_save_fp = f"s3://rundir/{self.name}"
        self.upload_every_n = upload_every_n_mul * every_n

        self.log_memory_detail = log_memory_detail

    def on_train_start(self, model, iteration=0):
        torch.cuda.reset_peak_memory_stats()
        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            logger.info(f"{self.name} callback: local_dir: {self.local_dir}")

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    def every_n_impl(
        self,
        trainer,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        cur_process = psutil.Process(os.getpid())
        cpu_memory_usage = sum(
            p.memory_info().rss
            for p in [cur_process] + cur_process.children(recursive=True)
        )
        cpu_mem_gb = cpu_memory_usage / (1024**3)

        peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        temp = torch.cuda.temperature()
        try:
            power = torch.cuda.power_draw()
        except Exception as e:
            logger.warning(f"Failed to get power draw with error {e}")
            power = 0
        util = torch.cuda.utilization()
        clock = torch.cuda.clock_rate()

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        nvml_used_gpu_mem_gb = memory_info.used / (1024**3)
        nvml_free_gpu_mem_gb = memory_info.free / (1024**3)

        prof_data = {
            "cpu_mem_gb": cpu_mem_gb,
            "peak_gpu_mem_gb": peak_gpu_mem_gb,
            "peak_gpu_mem_reserved_gb": peak_gpu_mem_reserved_gb,
            "nvml_used_gpu_mem_gb": nvml_used_gpu_mem_gb,
            "nvml_free_gpu_mem_gb": nvml_free_gpu_mem_gb,
            "temp": temp,
            "power": power,
            "util": util,
            "clock": clock,
        }

        data_list = [prof_data] * self.world_size
        # this is blocking by default
        if self.world_size > 1:
            torch.distributed.all_gather_object(data_list, prof_data)
            torch.distributed.barrier()

        df, summary_df = log_prof_data(data_list, iteration)
        if self.save_s3 and self.rank == 0:
            global_step = iteration // self.step_size
            should_run = global_step % self.upload_every_n == 0
            if should_run:
                df.to_csv(
                    os.path.join(self.local_dir, f"prof_data_{iteration:09d}.csv"),
                    index=False,
                )
                summary_df.to_csv(
                    os.path.join(self.local_dir, f"summary_{iteration:09d}.csv"),
                    index=True,
                )
                easy_io.copyfile_from_local(
                    os.path.join(self.local_dir, f"prof_data_{iteration:09d}.csv"),
                    os.path.join(self.s3_save_fp, f"prof_data_{iteration:09d}.csv"),
                )
                easy_io.copyfile_from_local(
                    os.path.join(self.local_dir, f"summary_{iteration:09d}.csv"),
                    os.path.join(self.s3_save_fp, f"summary_{iteration:09d}.csv"),
                )
        if self.rank == 0:
            logger.info(f"{self.name} Stats:\n{summary_df.to_string()}")
            if self.log_memory_detail:
                memory_stats = torch.cuda.memory_stats()
                if wandb.run:
                    wandb_memory_info = {
                        f"mem/{key}": memory_stats[key] for key in memory_stats.keys()
                    }
                    wandb.log(wandb_memory_info, step=iteration)
                if self.save_s3:
                    global_step = iteration // self.step_size
                    should_run = global_step % self.upload_every_n == 0
                    if should_run:
                        easy_io.dump(
                            memory_stats,
                            os.path.join(
                                self.s3_save_fp, f"memory_stats_{iteration:09d}.yaml"
                            ),
                        )

        torch.cuda.reset_peak_memory_stats()


def _verify_keys_same(
    keys: List[str],
    pg: ProcessGroup,
    pg_name: str,
):
    if pg.size() > 1:
        # Prepare a list to store gathered keys from all ranks
        gathered_keys = [None] * dist.get_world_size(pg)

        # Gather data_dict_keys from all ranks in the group
        dist.all_gather_object(gathered_keys, keys, group=pg)

        # Check if all ranks have the same keys
        if not all(keys == keys for keys in gathered_keys):
            raise ValueError(
                f"Inconsistent data dictionary keys across {pg_name} ranks. "
                f"Rank {dist.get_rank(pg)} keys: {keys}"
            )


class VerifyDataBatchKeys(Callback):
    """Callback to verify consistency of data batch keys across distributed ranks.

    This callback checks that all ranks in a distributed training setup have the
    same keys in their data dictionaries for the first N training steps.

    It is useful for mp_rank_0 only to do real data loading while other ranks do dummy data loading.
    Useful for large scale training to ease s3 read load.

    Attributes:
        count (int): Counter for the number of checks performed.
        check_first_n (int): Number of initial training steps to perform the check.
    """

    def __init__(self, check_first_n: int = 10):
        """Initializes the VerifyDataBatchKeys callback.

        Args:
            check_first_n (int): Number of initial training steps to perform the
                key consistency check. Defaults to 10.
        """
        super().__init__()
        self.count = 0
        self.check_first_n = check_first_n

    def on_training_step_start(
        self, model, data: Dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        """Performs the key consistency check at the start of each training step.

        This method is called at the beginning of each training step. It checks
        that all ranks have the same keys in their data dictionaries for the
        first `check_first_n` steps.

        Args:
            model (CosmosVisionGenModel): The model being trained.
            data (Dict[str, torch.Tensor]): The input data dictionary for the
                current training step.
            iteration (int): The current iteration number. Defaults to 0.

        Raises:
            ValueError: If the data dictionary keys are inconsistent across ranks.
        """
        if self.count < self.check_first_n:
            data_dict_keys: List[str] = list(sorted(data.keys()))
            if (
                self.parallel_dims is not None
                and "cp" in self.parallel_dims.mesh.mesh_dim_names
            ):
                cp_group: ProcessGroup = self.parallel_dims.mesh["cp"].get_group()
                _verify_keys_same(data_dict_keys, cp_group, "cp")

            if (
                self.parallel_dims is not None
                and "mp" in self.parallel_dims.mesh.mesh_dim_names
            ):
                mp_group: ProcessGroup = self.parallel_dims.mesh["mp"].get_group()
                _verify_keys_same(data_dict_keys, mp_group, "mp")

            self.count += 1


class ManualGarbageCollection(EveryN):
    """
    Disable auto gc and manually trigger garbage collection every N iterations
    It is super useful for large scale training to reduce gpu sync time!
    Can reach 50% speedup.

    It is important to note that this callback only disables gc in main process and have auto gc enabled in subprocesses.

    We start disable gc after warm_up iterations to avoid disabling gc in subprocesses, such as dataloader, which can cause OOM
    """

    def __init__(self, *args, warm_up: int = 5, **kwargs):
        kwargs["barrier_after_run"] = False
        super().__init__(*args, **kwargs)

        self.counter = 0
        self.warm = warm_up

    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        del trainer, model, data_batch, output_batch, loss
        self.counter += 1
        if self.counter < self.warm:
            return
        if self.counter == self.warm:
            gc.disable()
            logger.critical("Garbage collection disabled")

        gc.collect(1)


class CompileTokenizer(Callback):
    def __init__(
        self,
        enabled: bool = False,
        compile_after_iterations: int = 4,
        dynamic: bool = False,
    ):
        super().__init__()
        self.enabled = enabled
        self.compiled = False
        self.compile_after_iterations = compile_after_iterations
        self.skip_counter = 0
        self.dynamic = dynamic  # If there are issues with constant recompilations you may set this value to None or True

    def on_training_step_start(
        self, model, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if not self.enabled or self.compiled:
            return

        if isinstance(model.tokenizer, torch.jit.ScriptModule):
            logger.critical(
                f"The Tokenizer model {type(model.tokenizer)} is a JIT model, which is not compilable. The Tokenizer will not be compiled."
            )

        if self.skip_counter == self.compile_after_iterations:
            try:
                # PyTorch >= 2.7
                torch._dynamo.config.recompile_limit = 32
            except AttributeError:
                try:
                    torch._dynamo.config.cache_size_limit = 32
                except AttributeError:
                    logger.warning(
                        "Tokenizer compilation requested, but Torch Dynamo is unavailable – skipping compilation."
                    )
                    self.enabled = False
                    return

            model.tokenizer.encode = torch.compile(
                model.tokenizer.encode, dynamic=self.dynamic
            )
            self.compiled = True
        self.skip_counter += 1


@dataclass
class _FrameLossRecord:
    iter_count: int = 0
    edm_loss_m1: float = 0
    edm_loss_m2: float = 0

    def reset(self) -> None:
        self.iter_count = 0
        self.edm_loss_m1 = 0
        self.edm_loss_m2 = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            edm_loss_m1 = self.edm_loss_m1 / self.iter_count
            edm_loss_m2 = self.edm_loss_m2 / self.iter_count
            dist.all_reduce(edm_loss_m1, op=dist.ReduceOp.AVG)
            dist.all_reduce(edm_loss_m2, op=dist.ReduceOp.AVG)
        else:
            edm_loss_m1 = torch.ones(1)
            edm_loss_m2 = torch.ones(1)
        iter_count = self.iter_count
        self.reset()
        return edm_loss_m1.tolist(), edm_loss_m2.tolist(), iter_count


class FrameLossLog(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.save_s3 = save_s3
        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        self.name = self.__class__.__name__

        self.train_image_log = _FrameLossRecord()
        self.train_video_log = _FrameLossRecord()

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ):
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            logger.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
            )

        if not skip_update_due_to_unstable_loss:
            _loss = output_batch["edm_loss_per_frame"].detach().mean(dim=0)
            if model.is_image_batch(data_batch):
                self.train_image_log.iter_count += 1
                self.train_image_log.edm_loss_m1 += _loss
                self.train_image_log.edm_loss_m2 += _loss**2

            else:
                self.train_video_log.iter_count += 1
                self.train_video_log.edm_loss_m1 += _loss
                self.train_video_log.edm_loss_m2 += _loss**2

        if (
            iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler)
            == 0
        ):
            world_size = dist.get_world_size()
            image_edm_loss_m1, image_edm_loss_m2, img_iter_count = (
                self.train_image_log.get_stat()
            )
            video_edm_loss_m1, video_edm_loss_m2, vid_iter_count = (
                self.train_video_log.get_stat()
            )
            img_iter_count *= world_size
            vid_iter_count *= world_size

            if distributed.is_rank0():
                info = {}
                if vid_iter_count > 0:
                    info["frame_loss_log/video_sample"] = vid_iter_count
                    for i, (m1, m2) in enumerate(
                        zip(video_edm_loss_m1, video_edm_loss_m2)
                    ):
                        info[f"frame_loss_log/video_edm_loss_{i}"] = m1
                        info[f"frame_loss_log_sq/video_edm_loss_{i}"] = m2
                if img_iter_count > 0:
                    info["frame_loss_log/image_sample"] = img_iter_count
                    for i, (m1, m2) in enumerate(
                        zip(image_edm_loss_m1, image_edm_loss_m2)
                    ):
                        info[f"frame_loss_log/image_edm_loss_{i}"] = m1
                        info[f"frame_loss_log_sq/image_edm_loss_{i}"] = m2

                if info:
                    if self.save_s3:
                        if (
                            iteration
                            % (
                                self.config.trainer.logging_iter
                                * self.logging_iter_multipler
                                * self.save_logging_iter_multipler
                            )
                            == 0
                        ):
                            easy_io.dump(
                                info,
                                f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                            )

                    if wandb.run:
                        wandb.log(info, step=iteration)


def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [
            convert_to_primitive(v)
            for v in value
            if is_primitive(v) or isinstance(v, (list, dict))
        ]
    elif isinstance(value, dict):
        return {
            k: convert_to_primitive(v)
            for k, v in value.items()
            if is_primitive(v) or isinstance(v, (list, dict))
        }
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [3.0, 7.0, 9.0, 13.0],
        is_x0: bool = True,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        show_all_frames: bool = False,
        fps: int = 16,
    ):
        super().__init__(every_n, step_size, run_at_start=True)
        self.fix_batch = fix_batch_fp
        self.n_x0_level = n_x0_level
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.is_x0 = is_x0
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.show_all_frames = show_all_frames
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self.parallel_dims = get_global_parallel_dims()

    def on_train_start(self, model, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            logger.info(f"Callback: local_dir: {self.local_dir}")

        if self.fix_batch is not None:
            with utils.timer(f"loading fix_batch {self.fix_batch}"):
                self.fix_batch = utils.to(easy_io.load(self.fix_batch), "cpu")

        if self.parallel_dims is not None:
            self.data_parallel_id = self.parallel_dims.dp_shard_coord[1]
        else:
            self.data_parallel_id = self.rank

        if self.use_negative_prompt:
            self.negative_prompt_data = easy_io.load(
                get_itemdataset_option("negative_prompt_v0_s3").path
            )

    @utils.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.fix_batch is not None:
            data_batch = utils.to(self.fix_batch, **model.tensor_kwargs)
        tag = "ema" if self.is_ema else "reg"

        logger.debug("starting data and condition model")
        # TODO: (qsh 2024-07-01) this may be problematic due to sometimes we have uncondition, some times we have condition due to cfg dropout
        # TODO: (qsh 2025-02-25) we need to broadcast raw_data for correct visualization
        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        _, condition, x0, _ = model.broadcast_split_for_model_parallelsim(
            None, condition, x0, None
        )

        logger.debug("done data and condition model")
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(
                math.log(model.sde.sigma_min),
                math.log(model.sde.sigma_max),
                self.n_x0_level + 1,
            )[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(
            *x0.shape, generator=generator, **model.tensor_kwargs
        )
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            logger.debug(f"starting denoising {sigma}")
            sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            logger.debug(f"done denoising {sigma}")
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)
            # TODO: (qsh 2025-02-25) buggy for cp code. need to gather before decode if we split xt
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        base_fp_wo_ext = (
            f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"
        )

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        # override with output given data_batch if it exists
        if "data_batch" in output_batch:
            data_batch = output_batch["data_batch"]

        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if "reward_instance" in output_batch.keys():
            batch_info["reward_instance"] = {
                k: v.item() for k, v in output_batch["reward_instance"].items()
            }
            batch_info["reward_mean"] = output_batch["reward_mean"].item()

        if distributed.is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        logger.debug("entering, every_n_impl")
        with context():
            logger.debug("entering, ema")
            # we only use rank0 and rank to generate images and save
            # other rank run forward pass to make sure it works for FSDP
            logger.debug("entering, fsdp")
            if self.is_x0:
                logger.debug("entering, x0_pred")
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                logger.debug("done, x0_pred")
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )
            if self.is_sample:
                logger.debug("entering, sample")
                sample_img_fp = self.sample(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                logger.debug("done, sample")
            if self.fix_batch is not None:
                utils.to(self.fix_batch, "cpu")

            logger.debug("waiting for all ranks to finish")
            dist.barrier()
        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.is_x0:
                info[f"{self.name}/{tag}_x0"] = wandb.Image(
                    x0_img_fp, caption=f"{sample_counter}"
                )
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update(
                    {
                        f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i]
                        for i in range(len(mse_loss))
                    }
                )

            if self.is_sample:
                info[f"{self.name}/{tag}_sample"] = wandb.Image(
                    sample_img_fp, caption=f"{sample_counter}"
                )
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @utils.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            data_batch = utils.to(self.fix_batch, **model.tensor_kwargs)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        if self.use_negative_prompt:
            batch_size = x0.shape[0]
            data_batch["neg_t5_text_embeddings"] = utils.to(
                repeat(
                    self.negative_prompt_data["t5_text_embeddings"],
                    "... -> b ...",
                    b=batch_size,
                ),
                **model.tensor_kwargs,
            )
            assert (
                data_batch["neg_t5_text_embeddings"].shape
                == data_batch["t5_text_embeddings"].shape
            ), f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]

        to_show = []
        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=self.num_sampling_step,
                is_negative_prompt=True if self.use_negative_prompt else False,
            )
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())

        if "inference_out" in output_batch and hasattr(model, "decode"):
            inference_out = model.decode(output_batch["inference_out"])
            to_show.append(inference_out.float().cpu())

        to_show.append(raw_data.float().cpu())

        base_fp_wo_ext = (
            f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"
        )

        batch_size = output_batch["x0"].shape[0]
        if distributed.is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
            return local_path
        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (
            1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)
        ) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if self.rank == 0 and wandb.run:
            if is_single_frame:  # image case
                to_show = rearrange(
                    to_show[:, :n_viz_sample],
                    "n b c t h w -> t c (n h) (b w)",
                )
                image_grid = torchvision.utils.make_grid(
                    to_show, nrow=1, padding=0, normalize=False
                )
                # resize so that wandb can handle it
                torchvision.utils.save_image(
                    resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True
                )
            else:
                to_show = to_show[:, :n_viz_sample]  # [n, b, c, 3, h, w]
                if not self.show_all_frames:
                    # resize 3 frames frames so that we can display them on wandb
                    _T = to_show.shape[3]
                    three_frames_list = [0, _T // 2, _T - 1]
                    to_show = to_show[:, :, :, three_frames_list]
                    log_image_size = 1024
                else:
                    log_image_size = 512 * to_show.shape[3]
                to_show = rearrange(
                    to_show,
                    "n b c t h w -> 1 c (n h) (b t w)",
                )

                # resize so that wandb can handle it
                image_grid = torchvision.utils.make_grid(
                    to_show, nrow=1, padding=0, normalize=False
                )
                torchvision.utils.save_image(
                    resize_image(image_grid, log_image_size),
                    local_path,
                    nrow=1,
                    scale_each=True,
                )

            return local_path
        return None


class DetailedDataLoadingSpeedMonitor(Callback):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        save_s3: bool = False,
    ):
        self.every_n = every_n
        self.step_size = step_size
        self.should_run = False
        self.start_dataloading_time = None
        self.dataloading_time = None
        self.name = self.__class__.__name__
        self.save_s3 = save_s3
        self.time_delta_list = []

    def on_before_dataloading(self, iteration: int = 0) -> None:
        # We want to run it one iteration before on_training_step_start should_run is set to True.
        global_step = iteration // self.step_size
        self.should_run = (global_step + 1) % self.every_n == 0
        self.start_dataloading_time = time.time()

    def on_after_dataloading(self, iteration: int = 0) -> None:
        self.time_delta_list.append(time.time() - self.start_dataloading_time)

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.should_run:
            self.should_run = False
            cur_rank_mean, cur_rank_max = (
                np.mean(self.time_delta_list),
                np.max(self.time_delta_list),
            )
            self.time_delta_list = []  # Reset the list

            dataloading_time_gather_list = distributed.all_gather_tensor(
                torch.tensor([cur_rank_mean, cur_rank_max]).cuda()
            )
            wandb_info = {
                f"{self.name}_mean/dataloading_{k:03d}": v[0].item()
                for k, v in enumerate(dataloading_time_gather_list)
            }
            wandb_info.update(
                {
                    f"{self.name}_max/dataloading_{k:03d}": v[1].item()
                    for k, v in enumerate(dataloading_time_gather_list)
                }
            )
            mean_times = torch.stack(dataloading_time_gather_list)[:, 0]
            slowest_dataloading_rank_id = torch.argmax(mean_times)
            max_dataloading = torch.max(mean_times)
            wandb_info.update(
                {
                    "slowest_rank/slowest_dataloading_rank": slowest_dataloading_rank_id.item(),
                    "slowest_rank/slowest_dataloading_time": max_dataloading.item(),
                }
            )

            if wandb.run:
                wandb.log(wandb_info, step=iteration)

            if self.save_s3 and distributed.is_rank0():
                easy_io.dump(
                    wandb_info,
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )


@dataclass
class _MetricRecord:
    """Records and tracks metrics during training."""

    # Initialize metrics with default values
    value: torch.Tensor = 0
    iteration_count: int = 0

    def reset(self) -> None:
        """Reset all metrics to their default values."""
        self.value = 0
        self.iteration_count = 0

    def update(
        self,
        value: torch.Tensor,
    ) -> None:
        """Update the metric record with new values.

        Args:
            value: The metric value to add
        """
        self.value += value.detach().float()
        self.iteration_count += 1

    def get_stats(self) -> float:
        """Calculate and return statistics for the metric.

        Returns:
            Averaged metric value
        """
        if self.iteration_count > 0:
            # Calculate average for metrics
            avg_value = self.value / self.iteration_count

            # Distribute across processes
            if isinstance(avg_value, torch.Tensor):
                dist.all_reduce(avg_value, op=dist.ReduceOp.AVG)
                result = avg_value.item()
            else:
                result = avg_value
        else:
            # Default values if no iterations
            result = 0.0

        # Reset after collecting stats
        self.reset()
        return result


class RewardCallback(BaseWandBCallback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()

        self.PRINT_KEY = [
            "reward_mean",
            "log_prob_ratio",
            "kl_loss",
            "policy_loss",
            "data_loss",
        ]
        self.INSTANCE_REWARD_KEY = []
        # Initialize metric logs based on PRINT_KEY
        for key in self.PRINT_KEY:
            setattr(self, f"{key}_log", _MetricRecord())

        self.logging_iter_multiplier = logging_iter_multipler
        self.save_logging_iter_multiplier = save_logging_iter_multipler
        assert (
            self.logging_iter_multiplier > 0
        ), "logging_iter_multiplier should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = (
            f"@{self.logging_iter_multiplier}"
            if self.logging_iter_multiplier > 1
            else ""
        )

        self.setup_instance_reward_logging = False

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Log at specified intervals
        if (
            iteration
            % (self.config.trainer.logging_iter * self.logging_iter_multiplier)
            == 0
        ):
            info = {}
            # Add statistics for all metrics
            for key in self.PRINT_KEY + self.INSTANCE_REWARD_KEY:
                metric_log = getattr(self, f"{key}_log")
                metric_stats = metric_log.get_stats()
                info[f"GRPO{self.wandb_extra_tag}/{key}"] = metric_stats
            if distributed.is_rank0() and wandb.run:
                # Log to wandb
                wandb.log(info, step=iteration)

    def on_training_step_batch_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Log all metrics that exist in output_batch
        if not self.setup_instance_reward_logging:
            instance_reward_keys = list(output_batch["reward_instance"].keys())
            for key in instance_reward_keys:
                setattr(self, f"{key}_log", _MetricRecord())
            self.setup_instance_reward_logging = True
            self.INSTANCE_REWARD_KEY = instance_reward_keys
            logger.info(
                f"Setup instance reward logging for keys: {instance_reward_keys}"
            )

        for key in self.PRINT_KEY:
            metric_value = output_batch[key]
            getattr(self, f"{key}_log").update(metric_value)

        for key in self.INSTANCE_REWARD_KEY:
            metric_value = output_batch["reward_instance"][key]
            getattr(self, f"{key}_log").update(metric_value)


CALLBACK_CLS_MAPPING = {
    "ema_model": EMAModelCallback,
    "progress_bar": ProgressBarCallback,
    "iteration_logger": IterationLoggerCallback,
    "wandb": WandbCallback,
    "wandb_10x": WandbCallback,
    "low_prec": LowPrecisionCallback,
    "grad_clip": GradClip,
    "every_n": EveryN,
    "iter_speed": IterSpeed,
    "param_count": ModelParamStats,
    "heart_beat": HeartBeat,
    "device_monitor": DeviceMonitor,
    "verify_data_batch_keys": VerifyDataBatchKeys,
    "manual_gc": ManualGarbageCollection,
    "compile_tokenizer": CompileTokenizer,
    "frame_loss_log": FrameLossLog,
    "every_n_sample_reg": EveryNDrawSample,
    "every_n_sample_ema": EveryNDrawSample,
    "dataloader_speed": DetailedDataLoadingSpeedMonitor,
    "reward_callback": RewardCallback,
}


class CallBackGroup:
    """A class for hosting a collection of callback objects.

    It is used to execute callback functions of multiple callback objects with the same method name.
    When callbackgroup.func(args) is executed, internally it loops through the objects in self._callbacks and runs
    self._callbacks[0].func(args), self._callbacks[1].func(args), etc. The method name and arguments should match.

    Attributes:
        _callbacks (list[Callback]): List of callback objects.
    """

    def __init__(self, config: CosmosVisionGenConfig, trainer) -> None:
        """Initializes the list of callback objects.

        Args:
            config (Config): The config object for the CosmosVisionGen codebase.
            trainer (CosmosVisionGenTrainer): The main trainer.
        """
        self._callbacks = []
        callback_configs = config.trainer.callbacks
        if callback_configs:
            for current_callback_cfg in callback_configs:
                logger.critical(
                    f"Instantiating callback: {current_callback_cfg.model_dump()}"
                )
                _callback_cls = CALLBACK_CLS_MAPPING.get(
                    current_callback_cfg.name, None
                )
                if _callback_cls is None:
                    raise ValueError(
                        f"Callback {current_callback_cfg.name} is not defined in CALLBACK_CLS_MAPPING."
                    )
                _callback = _callback_cls(**current_callback_cfg.args)
                assert isinstance(
                    _callback, Callback
                ), f"{current_callback_cfg.model_dump()} is not a valid callback."
                _callback.config = config
                _callback.trainer = trainer
                self._callbacks.append(_callback)

    def __getattr__(self, method_name: str) -> Callable:
        """Loops through the callback objects to call the corresponding callback function.

        Args:
            method_name (str): Callback method name.
        """

        def multi_callback_wrapper(*args, **kwargs) -> None:
            for callback in self._callbacks:
                assert hasattr(callback, method_name)
                method = getattr(callback, method_name)
                assert callable(method)
                _ = method(*args, **kwargs)

        return multi_callback_wrapper
