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
import re
import json
import heapq
import torch
import random
import shutil
import numpy as np
import concurrent.futures as futures
from cosmos_rl.utils.util import is_master_rank
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.s3_utils import upload_file_to_s3
from cosmos_rl.policy.config import Config as CosmosConfig
from typing import List, Callable, Union, Optional, Dict


class CheckpointMananger:
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims = None,
        global_rank: int = 0,
        metric: str = "val_loss",
    ):
        self.config = config
        self.parallel_dims = parallel_dims
        self.global_rank = global_rank
        self.max_keep = config.train.ckpt.max_keep
        self.metric = metric
        self.save_mode = config.train.ckpt.save_mode
        self.ckpt_output_dir = os.path.join(config.train.output_dir, "checkpoints")
        if self.config.train.ckpt.upload_s3:
            self.ckpt_s3_output_dir = os.path.join(
                config.train.ckpt.s3_prefix, "checkpoints"
            )
        if self.config.train.ckpt.enable_checkpoint:
            if not os.path.exists(self.ckpt_output_dir):
                os.makedirs(self.ckpt_output_dir, exist_ok=True)
            if self.save_mode == "async":
                self.executor = futures.ThreadPoolExecutor(max_workers=4)
        self.pre_save_futures = []
        self.saved_steps = []
        for step, _ in self._get_saved_step_to_timestamp_map().items():
            heapq.heappush(self.saved_steps, step)
        # Load best score from file if exists (persists across resumes)
        self.best_score = self._load_best_score()
        self.best_step = self._get_best_step_from_link()

    def _get_num_saving_ranks(self) -> int:
        """
        Calculate the number of ranks that save checkpoints based on parallel_dims.

        The checkpoint saving condition is: dp_replicate_coord[0] == 0
        So the number of saving ranks = world_size / dp_replicate

        Different parallelism configurations examples:
        - Pure DP (dp_replicate=8, dp_shard=1): 1 rank saves (rank 0)
        - Pure FSDP (dp_replicate=1, dp_shard=8): 8 ranks save (rank 0-7)
        - DP + FSDP (dp_replicate=2, dp_shard=4): 4 ranks save (rank 0-3)
        - TP/PP/CP: These are within the saving group, so they add to the count

        Returns:
            int: Number of ranks that save checkpoints.
        """
        if self.parallel_dims is None:
            return 1  # Default to 1 rank (pure DP or single GPU)

        # Ranks with dp_replicate_coord[0] == 0 will save
        # This equals: world_size / dp_replicate
        # Note: dp_replicate is guaranteed to be >= 1 by ParallelDims._validate()
        return self.parallel_dims.world_size // self.parallel_dims.dp_replicate

    def ckpt_path_check(self, ckpt_path: str) -> bool:
        """
        Check if a checkpoint path is valid and complete.

        A checkpoint is considered complete if:
        1. The cosmos_config file exists
        2. All expected rank complete markers (.rank_<rank_id>_complete) exist

        The expected ranks are determined by self.parallel_dims:
        - Ranks with dp_replicate_coord[0] == 0 save checkpoints
        - Number of saving ranks = world_size / dp_replicate

        Args:
            ckpt_path: Path to the checkpoint directory (e.g., step_100/policy)

        Returns:
            bool: True if checkpoint is complete, False otherwise.
        """
        # Check cosmos_config exists
        if not os.path.exists(os.path.join(ckpt_path, "cosmos_config")):
            return False

        # Calculate expected number of saving ranks based on parallel_dims
        num_saving_ranks = self._get_num_saving_ranks()

        # Check complete markers for all expected ranks (0 to num_saving_ranks-1)
        for rank in range(num_saving_ranks):
            if not os.path.exists(os.path.join(ckpt_path, f".rank_{rank}_complete")):
                return False
        return True

    def _get_saved_step_to_timestamp_map(self) -> Dict[int, str]:
        """
        Get the map of saved step to timestamp.

        Returns:
            Dict[int, str]: A dictionary mapping saved steps to their corresponding timestamps.
        """
        saved_step_to_timestamp_map = {}
        if self.config.train.resume == True:  # noqa: E712
            root_output_dir = self._get_root_output_dir()
            timestamps = os.listdir(root_output_dir)
            timestamps.sort()

            for timestamp in timestamps:
                # Skip the 'best' directory which contains symlinks
                if timestamp == "best":
                    continue
                ckpt_base = os.path.join(root_output_dir, timestamp, "checkpoints")
                if not os.path.isdir(ckpt_base):
                    continue
                for step_dir in os.listdir(ckpt_base):
                    # validate step_dir format: step_<number>
                    match = re.match(r"^step_(\d+)$", step_dir)
                    if match:
                        saved_step_to_timestamp_map[int(match.group(1))] = timestamp
        return saved_step_to_timestamp_map

    def get_ckpt_path(self) -> List[str]:
        # find the latest checkpoint under output_dir
        if self.config.train.resume == True:  # noqa: E712
            root_output_dir = self._get_root_output_dir()
            saved_step_to_timestamp = self._get_saved_step_to_timestamp_map()
            steps = sorted(saved_step_to_timestamp.keys())
            return [
                os.path.join(
                    root_output_dir,
                    saved_step_to_timestamp[step],
                    "checkpoints",
                    f"step_{step}",
                    "policy",
                )
                for step in reversed(steps)
            ]
        else:
            return [self.config.train.resume]

    def _get_root_output_dir(self) -> str:
        """
        Get the root output directory.

        We assume self.config.train.output_dir directory is structured like:
            /path/to/output_dir/<cur_timestamp>
        This method returns the /path/to/output_dir
        """
        return os.path.dirname(self.config.train.output_dir)

    def _get_best_dir(self) -> str:
        """Get the path to the best model directory at root level."""
        return os.path.join(self._get_root_output_dir(), "best")

    def _get_best_score_path(self) -> str:
        """Get the path to the best score file."""
        return os.path.join(self._get_best_dir(), "best_score.json")

    def _load_best_score(self) -> float:
        """
        Load the best score from file if exists.
        Returns the default value if file doesn't exist.
        """
        default_score = float("inf") if "loss" in self.metric else -float("inf")
        best_score_path = self._get_best_score_path()
        if os.path.exists(best_score_path):
            try:
                with open(best_score_path, "r") as f:
                    data = json.load(f)
                    score = data.get("best_score", default_score)
                    logger.info(f"Loaded best score from {best_score_path}: {score}")
                    return score
            except Exception as e:
                logger.warning(f"Failed to load best score from {best_score_path}: {e}")
        return default_score

    def _save_best_score(self, score: float, step: int):
        """Save the best score to file."""
        best_dir = self._get_best_dir()
        os.makedirs(best_dir, exist_ok=True)
        best_score_path = self._get_best_score_path()
        with open(best_score_path, "w") as f:
            json.dump(
                {"best_score": score, "best_step": step, "metric": self.metric}, f
            )
        logger.info(f"Saved best score to {best_score_path}: {score}")

    def _get_best_step_from_link(self) -> Optional[int]:
        """
        Get the best step number from the existing best checkpoint link.
        Returns None if no best link exists.
        """
        best_ckpt_link = os.path.join(self._get_best_dir(), "checkpoints")
        if os.path.islink(best_ckpt_link):
            try:
                target = os.readlink(best_ckpt_link)
                basename = os.path.basename(target)
                match = re.match(r"^step_(\d+)$", basename)
                if match:
                    step = int(match.group(1))
                    logger.info(f"Found existing best checkpoint at step {step}")
                    return step
            except Exception as e:
                logger.warning(f"Failed to read best checkpoint link: {e}")
        return None

    def _is_step_linked_as_best(self, step: int) -> bool:
        """Check if the given step is currently linked as the best checkpoint."""
        return self.best_step is not None and self.best_step == step

    def _delete_step_checkpoint(self, step: int):
        """Delete checkpoint and safetensors for a given step."""
        if self.config.train.resume == True:  # noqa: E712
            # Resume case: checkpoint may be in a different timestamp directory
            step_to_timestamp = self._get_saved_step_to_timestamp_map()
            timestamp = step_to_timestamp.get(step)
            if timestamp is None:
                # Checkpoint directory not found, skip deletion
                logger.warning(
                    f"Checkpoint step_{step} not found in any timestamp directory"
                )
                return
            root_output_dir = self._get_root_output_dir()
            ckpt_dir = os.path.join(
                root_output_dir, timestamp, "checkpoints", f"step_{step}"
            )
            safetensors_dir = os.path.join(
                root_output_dir, timestamp, "safetensors", f"step_{step}"
            )
        else:
            # Non-resume case: checkpoint is in current output_dir
            ckpt_dir = os.path.join(self.ckpt_output_dir, f"step_{step}")
            safetensors_dir = os.path.join(
                self.config.train.output_dir, "safetensors", f"step_{step}"
            )
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            logger.info(f"Removed old checkpoint: {ckpt_dir}")
        if os.path.exists(safetensors_dir):
            shutil.rmtree(safetensors_dir)
            logger.info(f"Removed old safetensors: {safetensors_dir}")

    @staticmethod
    def get_rng_state():
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        return rng_state

    @staticmethod
    def set_rng_state(rng_state):
        torch.set_rng_state(rng_state["torch"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"])

    @staticmethod
    def load_extra_info(extra_info_path: str):
        if os.path.exists(extra_info_path):
            with open(extra_info_path, "rb") as f:
                extra_info = torch.load(f, weights_only=False)
            return extra_info
        else:
            logger.warning(f"Extra info file {extra_info_path} does not exist.")
            return {}

    def offload_state_dict_cpu(self, state_dict: dict):
        state_dict_cpu = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict_cpu[key] = value.cpu()
            else:
                state_dict_cpu[key] = value
        return state_dict

    def finalize(self) -> None:
        """Wait for any pending async checkpoint saves/uploads to finish.
        This should be called before process exit to avoid losing uploads when
        `save_mode == "async"`.
        """
        if self.save_mode != "async" or not hasattr(self, "executor"):
            return
        if self.pre_save_futures:
            for future in futures.as_completed(self.pre_save_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Async checkpoint save/upload failed: {e}")
            self.pre_save_futures = []
        self.executor.shutdown(wait=True)

    def save_checkpoint(
        self,
        model: Union[torch.nn.Module, Dict],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        step: int,
        total_steps: int,
        **kwargs,
    ):
        """
        Save the model, optimizer, scheduler state dicts and extra info to disk.
        Also upload the checkpoint to S3 if configured.
        Args:
            model (Union[torch.nn.Module, Dict]): The model or state_dict to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save.
            step (int): The current training step.
            **kwargs: Additional information to save, e.g., is_final.
        """

        def _save_upload(state_dict, local_rel_path, is_final=False):
            local_abs_path = os.path.join(self.ckpt_output_dir, local_rel_path)
            torch.save(state_dict, local_abs_path)
            if self.config.train.ckpt.upload_s3:
                if (self.config.train.ckpt.upload_s3 == "final" and is_final) or (
                    self.config.train.ckpt.upload_s3 == "all"
                ):
                    s3_path = os.path.join(self.ckpt_s3_output_dir, local_rel_path)
                    upload_file_to_s3(
                        local_file_path=local_abs_path,
                        bucket_name=self.config.train.ckpt.s3_bucket,
                        s3_file_path=s3_path,
                    )

        is_final = kwargs.get("is_final", False)
        cur_step_ckpt_dir = os.path.join(f"step_{step}", "policy")
        os.makedirs(
            os.path.join(self.ckpt_output_dir, cur_step_ckpt_dir), exist_ok=True
        )

        # construct the extra info dict
        with open(
            os.path.join(self.ckpt_output_dir, cur_step_ckpt_dir, "cosmos_config"), "w"
        ) as f:
            f.write(json.dumps(self.config.model_dump(), indent=4))
        extra_info = {
            "rng_state": self.get_rng_state(),
            "step": step,
            "total_steps": total_steps,
        }
        for key, value in kwargs.items():
            if key in extra_info:
                extra_info[key] = value
            else:
                extra_info[key] = value

        # paths for saving the state dicts
        model_ckpt_path = os.path.join(
            cur_step_ckpt_dir, f"model_rank_{self.global_rank}.pth"
        )
        optimizer_ckpt_path = os.path.join(
            cur_step_ckpt_dir, f"optimizer_rank_{self.global_rank}.pth"
        )
        scheduler_ckpt_path = os.path.join(
            cur_step_ckpt_dir, f"scheduler_rank_{self.global_rank}.pth"
        )
        extra_info_ckpt_path = os.path.join(
            cur_step_ckpt_dir, f"extra_info_rank_{self.global_rank}.pth"
        )

        if isinstance(model, torch.nn.Module):
            state_dict = model.state_dict()
        elif isinstance(model, dict):
            state_dict = model
        else:
            raise ValueError(
                "Unsupport model type, should either be a torch.nn.Module or dict"
            )

        # Path for the complete marker file
        complete_marker_path = os.path.join(
            self.ckpt_output_dir,
            cur_step_ckpt_dir,
            f".rank_{self.global_rank}_complete",
        )

        if self.save_mode == "async":

            def _write_complete_marker_after_saves(futures_to_wait, marker_path):
                """Wait for all save futures to complete, then write the complete marker."""
                for f in futures_to_wait:
                    f.result()  # Block until each future completes
                # All saves completed, write the complete marker
                with open(marker_path, "w") as f:
                    f.write("")

            # wait for the previous save to finish
            if len(self.pre_save_futures) > 0:
                for future in futures.as_completed(self.pre_save_futures):
                    future.result()
                self.pre_save_futures = []

            # offload the state dict to CPU
            model_state_dict_cpu = self.offload_state_dict_cpu(state_dict)
            optimizer_state_dict_cpu = self.offload_state_dict_cpu(
                optimizer.state_dict()
            )
            scheduler_state_dict_cpu = self.offload_state_dict_cpu(
                scheduler.state_dict()
            )
            extra_info_state_dict_cpu = self.offload_state_dict_cpu(extra_info)

            # save the state dicts to disk
            save_futures = []
            save_futures.append(
                self.executor.submit(
                    _save_upload, model_state_dict_cpu, model_ckpt_path, is_final
                )
            )
            save_futures.append(
                self.executor.submit(
                    _save_upload,
                    optimizer_state_dict_cpu,
                    optimizer_ckpt_path,
                    is_final,
                )
            )
            save_futures.append(
                self.executor.submit(
                    _save_upload,
                    scheduler_state_dict_cpu,
                    scheduler_ckpt_path,
                    is_final,
                )
            )
            save_futures.append(
                self.executor.submit(
                    _save_upload,
                    extra_info_state_dict_cpu,
                    extra_info_ckpt_path,
                    is_final,
                )
            )

            # Submit a task that waits for all saves and then writes the complete marker
            complete_marker_future = self.executor.submit(
                _write_complete_marker_after_saves, save_futures, complete_marker_path
            )

            # Track all futures (saves + complete marker)
            self.pre_save_futures = save_futures + [complete_marker_future]

            if is_final:
                # wait for all futures to complete before returning for final save
                futures.wait(self.pre_save_futures)
                self.pre_save_futures = []
        else:  # sync
            _save_upload(state_dict, model_ckpt_path, is_final)
            _save_upload(optimizer.state_dict(), optimizer_ckpt_path, is_final)
            _save_upload(scheduler.state_dict(), scheduler_ckpt_path, is_final)
            _save_upload(extra_info, extra_info_ckpt_path, is_final)
            # Write complete marker after all saves are done
            with open(complete_marker_path, "w") as f:
                f.write("")

        logger.info(
            f"[Policy] Step: {step}, checkpoint saved successfully at {os.path.join(self.ckpt_output_dir, cur_step_ckpt_dir)}."
        )

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, Callable],
        model_name_or_path: str,
        revision: Optional[str] = None,
        strict: bool = True,
    ) -> tuple[Dict, torch.optim.lr_scheduler._LRScheduler]:
        extra_vars = {}
        base_paths: List[str] = self.get_ckpt_path()
        # check whether checkpoint existing
        for base_path in base_paths:
            try:
                logger.info(f"Trying to load checkpoint from {base_path}...")
                if self.ckpt_path_check(base_path):
                    logger.info(
                        f"Cosmos checkpoint found at {self.config.train.resume}. Resuming..."
                    )
                    model_path = os.path.join(
                        base_path, f"model_rank_{self.global_rank}.pth"
                    )
                    optimizer_path = os.path.join(
                        base_path, f"optimizer_rank_{self.global_rank}.pth"
                    )
                    scheduler_path = os.path.join(
                        base_path, f"scheduler_rank_{self.global_rank}.pth"
                    )
                    extra_info_path = os.path.join(
                        base_path, f"extra_info_rank_{self.global_rank}.pth"
                    )
                    extra_info = self.load_extra_info(extra_info_path)
                    for key in extra_info:
                        if key == "rng_state":
                            self.set_rng_state(extra_info["rng_state"])
                        else:
                            extra_vars[key] = extra_info[key]

                    if isinstance(scheduler, Callable):
                        # Create a new scheduler upon ``training_steps``
                        new_scheduler = scheduler(
                            training_steps=extra_vars["total_steps"]
                        )
                        new_scheduler.load_state_dict(
                            torch.load(scheduler_path, weights_only=False)
                        )
                    else:
                        scheduler.load_state_dict(
                            torch.load(scheduler_path, weights_only=False)
                        )
                        new_scheduler = scheduler

                    model.load_state_dict(
                        torch.load(model_path, weights_only=False), strict=strict
                    )
                    optimizer.load_state_dict(
                        torch.load(optimizer_path, weights_only=False)
                    )
                    logger.info(
                        f"[Policy] Checkpoint loaded successfully from {base_path}."
                    )
                    return extra_vars, new_scheduler
            except Exception as e:
                logger.error(
                    f"Error loading checkpoint from {base_path}: {e}, try next checkpoint..."
                )

        raise FileNotFoundError(f"No checkpoint found at {base_paths}")

    def load_extra_info_from_checkpoint(self):
        extra_vars = {}
        base_paths = self.get_ckpt_path()
        # check whether checkpoint existing

        for base_path in base_paths:
            try:
                is_ckpt_path = self.ckpt_path_check(base_path)
                if is_ckpt_path:
                    logger.info(
                        f"Cosmos checkpoint found at {self.config.train.resume}. Loading extra info..."
                    )
                    extra_info_path = os.path.join(
                        base_path, f"extra_info_rank_{self.global_rank}.pth"
                    )
                    extra_info = self.load_extra_info(extra_info_path)
                    for key in extra_info:
                        if key == "rng_state":
                            self.set_rng_state(extra_info["rng_state"])
                        else:
                            extra_vars[key] = extra_info[key]
                    logger.info(
                        f"[Policy] Checkpoint extra info loaded successfully from {base_path}."
                    )
                    return extra_vars
                else:
                    raise FileNotFoundError(f"No checkpoint found at {base_path}")
            except Exception as e:
                logger.error(
                    f"Error loading checkpoint from {base_path}: {e}, try next checkpoint..."
                )

        raise FileNotFoundError(f"No checkpoint found at {base_paths}")

    def save_check(self, step: int, **kwargs):
        if is_master_rank(self.parallel_dims, self.global_rank):
            heapq.heappush(self.saved_steps, step)
            # remove the old checkpoints
            # expected behavior:
            # Keep the best checkpoint, and delete the oldest checkpoint if the number of
            # checkpoints exceeds the max_keep.
            # If the best checkpoint is the oldest checkpoint, delete the second oldest checkpoint.
            if len(self.saved_steps) > self.max_keep and self.max_keep != -1:
                oldest = self.saved_steps[0]  # peek
                step_to_delete = None

                if self._is_step_linked_as_best(oldest) and len(self.saved_steps) > 1:
                    # Best is oldest, delete second oldest instead
                    heapq.heappop(self.saved_steps)  # remove best temporarily
                    step_to_delete = heapq.heappop(self.saved_steps)
                    heapq.heappush(self.saved_steps, oldest)  # put best back
                    logger.info(
                        f"Best checkpoint is at step_{oldest}, "
                        f"deleting step_{step_to_delete} instead"
                    )
                else:
                    step_to_delete = heapq.heappop(self.saved_steps)
                    logger.info(f"Deleting step_{step_to_delete}")

                if step_to_delete is not None:
                    self._delete_step_checkpoint(step_to_delete)

            val_score = kwargs.get("val_score", None)
            if val_score is not None:
                if ("loss" in self.metric and val_score < self.best_score) or (
                    "loss" not in self.metric and val_score > self.best_score
                ):
                    self.best_score = val_score
                    self.best_step = step

                    best_dir = self._get_best_dir()
                    os.makedirs(best_dir, exist_ok=True)

                    # Create symlink for checkpoint at root/best/checkpoints
                    best_ckpt_link = os.path.join(best_dir, "checkpoints")
                    # assume the best checkpoint is at self.ckpt_output_dir/step_<step>
                    step_ckpt_path = os.path.join(self.ckpt_output_dir, f"step_{step}")
                    if os.path.islink(best_ckpt_link):
                        os.unlink(best_ckpt_link)
                    os.symlink(step_ckpt_path, best_ckpt_link)
                    logger.info(
                        f"Best checkpoint updated to step_{step} with score: {val_score}"
                    )

                    # Create symlink for safetensors at root/best/safetensors
                    if self.config.train.ckpt.export_safetensors:
                        best_safetensors_link = os.path.join(best_dir, "safetensors")
                        step_safetensors_path = os.path.join(
                            self.config.train.output_dir, "safetensors", f"step_{step}"
                        )
                        if os.path.islink(best_safetensors_link):
                            os.unlink(best_safetensors_link)
                        os.symlink(step_safetensors_path, best_safetensors_link)
                        logger.info(f"Best safetensors updated to step_{step}")

                    # Save best score to file for persistence across resumes
                    self._save_best_score(val_score, step)
