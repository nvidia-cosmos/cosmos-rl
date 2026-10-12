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

import copy
import os
import re
import json
import torch
import random
import shutil
import numpy as np
import concurrent.futures as futures
from cosmos_rl.utils.util import is_master_rank
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.resume import NoCheckpointFound, controller_checkpoint_metadata
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.s3_utils import upload_file_to_s3
from cosmos_rl.policy.config import Config as CosmosConfig
from typing import List, Callable, Tuple, Union, Optional, Dict


def _step_dir_sort_key(dir_path: str) -> int:
    """
    Key function for sorting directories by step number.

    Args:
        dir_path: Directory path (e.g., '/path/to/step_100' or 'step_100')

    Returns:
        step_id if matches 'step_xxx' pattern
        -9999 if doesn't match - these will sort first
    """
    basename = os.path.basename(dir_path)
    match = re.match(r"^step_(\d+)$", basename)
    if match:
        return int(match.group(1))
    return -9999


def _same_checkpoint_state(saved, current) -> bool:
    """Exact local-state comparison; never issue a DTensor collective.

    A committed step is immutable. Promotion may reuse it only if the caller's
    model, optimizer, scheduler and shared progress still describe that save.
    """
    if isinstance(saved, torch.Tensor) and isinstance(current, torch.Tensor):
        from torch.distributed.tensor import DTensor

        if isinstance(saved, DTensor) != isinstance(current, DTensor):
            return False
        if saved.shape != current.shape or saved.dtype != current.dtype:
            return False
        if isinstance(saved, DTensor):
            if (
                saved.placements != current.placements
                or saved.device_mesh.mesh_dim_names
                != current.device_mesh.mesh_dim_names
                or not torch.equal(saved.device_mesh.mesh, current.device_mesh.mesh)
            ):
                return False
            saved, current = saved.to_local(), current.to_local()
        return torch.equal(saved.detach().cpu(), current.detach().cpu())
    if isinstance(saved, dict) and isinstance(current, dict):
        return (
            saved.keys() == current.keys()
            and all(_same_checkpoint_state(saved[key], current[key]) for key in saved)
            and _same_checkpoint_state(
                getattr(saved, "_metadata", None), getattr(current, "_metadata", None)
            )
        )
    if isinstance(saved, (list, tuple)) and type(saved) is type(current):
        return len(saved) == len(current) and all(
            _same_checkpoint_state(a, b) for a, b in zip(saved, current, strict=True)
        )
    if isinstance(saved, np.ndarray) and isinstance(current, np.ndarray):
        return saved.dtype == current.dtype and np.array_equal(saved, current)
    return type(saved) is type(current) and bool(saved == current)


class CheckpointMananger:
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: Optional[ParallelDims] = None,
        global_rank: int = 0,
        metric: str = "val_loss",
        hook_fns: Dict[str, Callable] = {},
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
        # Housekeeping is caller-owned, never concurrent with another writer.
        # A rank's save completion is not the complete checkpoint's commit.
        self._pending_save_checks = {}
        self.selected_checkpoint_path = None
        if self._is_master_rank():
            self.saved_ckpt_step_dirs = sorted(
                self._get_all_saved_ckpt_step_dirs(),
                key=_step_dir_sort_key,
            )
            self._filter_owned_ckpts()
            # Load best score from file if exists (persists across resumes)
            self.best_score, self.best_ckpt_abs_dir = self._load_best_score()
        if "save_checkpoint_hook" in hook_fns:
            self.save_checkpoint_hook = hook_fns["save_checkpoint_hook"]
        else:
            self.save_checkpoint_hook = None

        if "load_checkpoint_hook" in hook_fns:
            self.load_checkpoint_hook = hook_fns["load_checkpoint_hook"]
        else:
            self.load_checkpoint_hook = None

    def _is_master_rank(self) -> bool:
        return (self.parallel_dims is None and self.global_rank == 0) or (
            self.parallel_dims is not None
            and is_master_rank(self.parallel_dims, self.global_rank)
        )

    def _filter_owned_ckpts(self):
        """Discovery is read-only; retention only owns this run's checkpoints.

        A missing marker can indicate an active writer or another topology,
        not corruption. Never delete a directory merely because discovery
        cannot load it. Cross-run candidates remain available to resume through
        get_latest_ckpt_paths(), but are not eligible for max_keep deletion.
        """
        owned_dirs = []
        output_dir = os.path.realpath(self.ckpt_output_dir)
        for ckpt_dir in self.saved_ckpt_step_dirs:
            if os.path.dirname(os.path.realpath(ckpt_dir)) != output_dir:
                continue
            if os.path.islink(ckpt_dir):
                continue
            policy_path = os.path.join(ckpt_dir, "policy")
            if self.ckpt_path_check(policy_path):
                owned_dirs.append(ckpt_dir)
        self.saved_ckpt_step_dirs = owned_dirs

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
            logger.warning(
                f"Checkpoint config not found at {ckpt_path}. Marking checkpoint as incomplete."
            )
            return False

        # Calculate expected number of saving ranks based on parallel_dims
        num_saving_ranks = self._get_num_saving_ranks()

        # Check complete markers for all expected ranks (0 to num_saving_ranks-1)
        for rank in range(num_saving_ranks):
            if not os.path.exists(os.path.join(ckpt_path, f".rank_{rank}_complete")):
                logger.warning(
                    f"Checkpoint complete marker for rank {rank} not found at {ckpt_path}. Marking checkpoint as incomplete."
                )
                return False
        return True

    def _get_all_saved_ckpt_step_dirs(self) -> List[str]:
        """
        Get the list of all saved checkpoint step directories.

        Returns:
            List[str]: A list of paths to the all saved checkpoint step directories.
        """
        saved_ckpt_step_dirs = []
        if self.config.train.resume == True:  # noqa: E712
            root_output_dir = self._root_output_dir
            try:
                timestamps = os.listdir(root_output_dir)
            except FileNotFoundError:
                # A never-created output root is an ordinary discovery miss.
                return []
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
                        saved_ckpt_step_dirs.append(os.path.join(ckpt_base, step_dir))
        return saved_ckpt_step_dirs

    def get_latest_ckpt_paths(self) -> List[str]:
        """
        Get the paths to the all saved checkpoint directories ordered by step number in descending order.

        Returns:
            List[str]: A list of paths to the all saved checkpoint directories ordered by step number in descending order.
        """
        if isinstance(self.config.train.resume, str):
            return [self.config.train.resume]
        else:
            saved_step_dirs = sorted(
                self._get_all_saved_ckpt_step_dirs(),
                key=_step_dir_sort_key,
                reverse=True,
            )
            return [os.path.join(step_dir, "policy") for step_dir in saved_step_dirs]

    @property
    def _root_output_dir(self) -> str:
        """
        Get the root output directory.

        We assume self.config.train.output_dir directory is structured like:
            /path/to/output_dir/<cur_timestamp>
        This method returns the /path/to/output_dir
        """
        return os.path.dirname(self.config.train.output_dir)

    @property
    def _best_dir(self) -> str:
        """Get the path to the best model directory at root level."""
        return os.path.join(self._root_output_dir, "best")

    @property
    def _best_score_path(self) -> str:
        """Get the path to the best score file."""
        return os.path.join(self._best_dir, "best_score.json")

    def _load_best_score(self) -> Tuple[float, Optional[str]]:
        """
        Load the best score from file if exists.
        Returns the default value if file doesn't exist.
        """
        default_score = float("inf") if "loss" in self.metric else -float("inf")
        best_score_path = self._best_score_path
        if os.path.exists(best_score_path):
            try:
                with open(best_score_path, "r") as f:
                    data = json.load(f)
                    score = data.get("best_score", default_score)
                    best_ckpt_abs_dir = data.get("best_ckpt_abs_dir", None)
                    if best_ckpt_abs_dir is not None:
                        # Older metadata may retain a symlinked output-root
                        # spelling; compare artifact identity, not path aliases.
                        best_ckpt_abs_dir = os.path.realpath(best_ckpt_abs_dir)
                    if (
                        best_ckpt_abs_dir is None
                        or best_ckpt_abs_dir != self._get_best_ckpt_abs_dir()
                    ):
                        raise ValueError(
                            f"Best checkpoint directory mismatch: {best_ckpt_abs_dir} != {self._get_best_ckpt_abs_dir()}"
                        )
                    logger.info(f"Loaded best score from {best_score_path}: {score}")
                    return score, best_ckpt_abs_dir
            except Exception as e:
                logger.warning(f"Failed to load best score from {best_score_path}: {e}")
        return default_score, None

    def _save_best_score(self, score: float, best_ckpt_dir: str):
        """Save the best score to file."""
        best_score_path = self._best_score_path
        os.makedirs(os.path.dirname(best_score_path), exist_ok=True)
        with open(best_score_path, "w") as f:
            json.dump(
                {
                    "best_score": score,
                    "best_ckpt_abs_dir": os.path.realpath(best_ckpt_dir),
                    "metric": self.metric,
                },
                f,
            )
        logger.info(f"Saved best score to {best_score_path}: {score}")

    def _get_best_step_from_link(self) -> Optional[int]:
        """
        Get the best step number from the existing best checkpoint link.
        Returns None if no best link exists.
        """
        best_ckpt_link = os.path.join(self._best_dir, "checkpoints")
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

    def _get_best_ckpt_abs_dir(self) -> Optional[str]:
        """
        Get the path to the best checkpoint directory.

        Returns the final resolved path (not a symlink) only if:
        1. The symlink can be resolved to a final target (not a link)
        2. The target directory exists
        3. The checkpoint is valid (self.ckpt_path_check(dir/policy) returns True)

        Returns:
            str | None: The resolved checkpoint directory path, or None if invalid.
        """
        best_ckpt_link = os.path.join(self._best_dir, "checkpoints")
        if not os.path.islink(best_ckpt_link):
            return None

        # Resolve to the final target (not a link)
        resolved_path = os.path.realpath(best_ckpt_link)

        # Ensure it's not still a symlink (broken chain)
        if os.path.islink(resolved_path):
            logger.warning(
                f"Best checkpoint link could not be fully resolved: {best_ckpt_link}"
            )
            return None

        # Check that the directory exists
        if not os.path.isdir(resolved_path):
            logger.warning(f"Best checkpoint directory does not exist: {resolved_path}")
            return None

        # Validate the step directory format
        basename = os.path.basename(resolved_path)
        match = re.match(r"^step_(\d+)$", basename)
        if not match:
            logger.warning(f"Best checkpoint directory has invalid format: {basename}")
            return None

        # Validate the checkpoint is complete
        policy_path = os.path.join(resolved_path, "policy")
        if not self.ckpt_path_check(policy_path):
            logger.warning(f"Best checkpoint is incomplete or invalid: {policy_path}")
            return None

        return os.path.abspath(resolved_path)

    def _is_ckpt_dir_linked_as_best(self, ckpt_dir: str) -> bool:
        """Check if the given ckpt_dir is currently linked as the best checkpoint."""
        return self.best_ckpt_abs_dir is not None and os.path.realpath(
            self.best_ckpt_abs_dir
        ) == os.path.realpath(ckpt_dir)

    def _delete_checkpoint(self, ckpt_dir: str):
        """Delete checkpoint and safetensors for a given step.

        Args:
            ckpt_dir: The directory of the checkpoint to delete, which is expected to be like:
                /path/to/output_dir/<timestamp>/checkpoints/step_<step_number>
        """
        try:
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)
                logger.info(f"Removed old checkpoint: {ckpt_dir}")
            safetensors_dir = os.path.join(
                os.path.dirname(os.path.dirname(ckpt_dir)),
                "safetensors",
                os.path.basename(ckpt_dir),
            )
            if os.path.exists(safetensors_dir):
                shutil.rmtree(safetensors_dir)
                logger.info(f"Removed old safetensors: {safetensors_dir}")
        except Exception as e:
            logger.error(f"Error deleting checkpoint {ckpt_dir}: {e}")

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
                extra_info = torch.load(f, weights_only=False, map_location="cpu")
            return extra_info
        else:
            raise FileNotFoundError(
                f"Checkpoint metadata is missing: {extra_info_path}"
            )

    def offload_state_dict_cpu(self, state_dict: dict):
        """Own all snapshot storage before handing it to the async writer.

        The caller must hold its training/sampling snapshot boundary while this
        runs. This isolates later mutations; it cannot make concurrently changing
        model, optimizer, and application state a coherent checkpoint.
        """

        def snapshot(value):
            if isinstance(value, torch.Tensor):
                # .cpu() aliases an already-CPU tensor (including Adam's step).
                # A blocking copy also completes CUDA staging before training
                # can reuse the source storage. Detach avoids retaining graphs.
                return value.detach().to(device="cpu", copy=True)
            if isinstance(value, dict):
                result = copy.copy(value)
                result.clear()
                for key, child in value.items():
                    # Preserve the existing omission of nonresident parameters.
                    if isinstance(child, torch.Tensor) and child.is_meta:
                        continue
                    result[copy.deepcopy(key)] = snapshot(child)
                # Module state_dict OrderedDicts carry versioning metadata.
                if hasattr(value, "__dict__"):
                    result.__dict__ = snapshot(value.__dict__)
                return result
            if isinstance(value, list):
                return [snapshot(child) for child in value]
            if isinstance(value, tuple):
                children = [snapshot(child) for child in value]
                if hasattr(value, "_fields"):
                    return type(value)(*children)
                return tuple(children)
            # Includes NumPy RNG arrays and mutable application metadata.
            return copy.deepcopy(value)

        return snapshot(state_dict)

    def _wait_for_pending_async_saves(self) -> None:
        """Wait for pending saves and propagate any background failure."""
        while self.pre_save_futures:
            futures.wait(self.pre_save_futures)
            for future in self.pre_save_futures:
                future.result()
            self.pre_save_futures = []
            self._publish_completed_checkpoints()

    def invalidate_completion_marker(self, step: int) -> None:
        """Compatibility hook before final promotion: join, never invalidate.

        Completed rank artifacts and their markers are immutable. The historical
        hook name remains for worker/custom-trainer compatibility; conflicting
        same-step state is rejected by save_checkpoint without touching files.
        """
        if self.save_mode == "async" and self.pre_save_futures:
            self._wait_for_pending_async_saves()

    def finalize(self) -> None:
        """Wait for any pending async checkpoint saves/uploads to finish.
        This should be called before process exit to avoid losing uploads when
        `save_mode == "async"`.
        """
        if self.save_mode != "async" or not hasattr(self, "executor"):
            self._publish_completed_checkpoints()
            return
        # Observe failures after joining every writer. A failed final save must
        # not be logged and converted into a successful training exit.
        try:
            self._wait_for_pending_async_saves()
            self._publish_completed_checkpoints()
        finally:
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
        Returns:
            str: The path to the saved checkpoint directory.
        """

        def _upload(local_rel_path, is_final=False):
            local_abs_path = os.path.join(self.ckpt_output_dir, local_rel_path)
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

        def _save_upload(state_dict, local_rel_path, is_final=False):
            local_abs_path = os.path.join(self.ckpt_output_dir, local_rel_path)
            torch.save(state_dict, local_abs_path)
            _upload(local_rel_path, is_final)

        is_final = kwargs.get("is_final", False)
        cur_step_ckpt_dir = os.path.join(f"step_{step}", "policy")

        complete_marker_path = os.path.join(
            self.ckpt_output_dir,
            cur_step_ckpt_dir,
            f".rank_{self.global_rank}_complete",
        )
        self.invalidate_completion_marker(step)

        # construct the extra info dict
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

        reused = os.path.exists(complete_marker_path)
        if reused:
            # Verify all components before any upload, hook or file mutation.
            # Final promotion preserves the original coherent RNG/progress
            # snapshot; is_final selects export/upload, not new training state.
            states = (
                (model_ckpt_path, state_dict),
                (optimizer_ckpt_path, optimizer.state_dict()),
                (scheduler_ckpt_path, scheduler.state_dict()),
                (extra_info_ckpt_path, extra_info),
            )
            for relative, current in states:
                saved = torch.load(
                    os.path.join(self.ckpt_output_dir, relative),
                    weights_only=False,
                    map_location="cpu",
                )
                if relative == extra_info_ckpt_path:
                    ignored = {"rng_state", "is_final"}
                    saved = {k: v for k, v in saved.items() if k not in ignored}
                    current = {k: v for k, v in current.items() if k not in ignored}
                if not _same_checkpoint_state(saved, current):
                    raise ValueError(
                        f"Checkpoint step {step} is immutable: conflicting {relative}"
                    )
            for relative, _ in states:
                _upload(relative, is_final)
        else:
            # A failed attempt is not permission to overwrite partly saved
            # artifacts. Other saving ranks may already own committed shards.
            for relative in (
                model_ckpt_path,
                optimizer_ckpt_path,
                scheduler_ckpt_path,
                extra_info_ckpt_path,
            ):
                if os.path.exists(os.path.join(self.ckpt_output_dir, relative)):
                    raise ValueError(
                        f"Checkpoint step {step} has incomplete rank artifacts; "
                        "refusing an in-place retry"
                    )
            os.makedirs(
                os.path.join(self.ckpt_output_dir, cur_step_ckpt_dir), exist_ok=True
            )
            with open(
                os.path.join(self.ckpt_output_dir, cur_step_ckpt_dir, "cosmos_config"),
                "w",
            ) as f:
                f.write(json.dumps(self.config.model_dump(), indent=4))

        if not reused and self.save_mode == "async":

            def _write_complete_marker_after_saves(futures_to_wait, marker_path):
                """Wait for all save futures to complete, then write the complete marker."""
                for f in futures_to_wait:
                    f.result()  # Block until each future completes
                # All saves completed, write the complete marker
                with open(marker_path, "w") as f:
                    f.write("")

            # Freeze every component before submitting any background writer.
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
                self._wait_for_pending_async_saves()
        elif not reused:  # sync
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
        if self.save_checkpoint_hook is not None:
            self.save_checkpoint_hook(
                self,
                data={
                    "checkpoint_path": os.path.join(
                        self.ckpt_output_dir, cur_step_ckpt_dir
                    ),
                    "step": step,
                    "total_steps": total_steps,
                    **kwargs,
                },
            )
        return os.path.join(self.ckpt_output_dir, cur_step_ckpt_dir)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, Callable],
        model_name_or_path: str,
        revision: Optional[str] = None,
        strict: bool = True,
        pp_model_parts: Optional[List] = None,
        pp_model_module_paths: Optional[
            List[str]
        ] = None,  # dotted module paths for each PP stage
    ) -> tuple[Dict, torch.optim.lr_scheduler._LRScheduler]:
        self.selected_checkpoint_path = None
        extra_vars = {}
        base_paths: List[str] = self.get_latest_ckpt_paths()
        # check whether checkpoint existing
        for base_path in base_paths:
            try:
                logger.info(f"Trying to load checkpoint from {base_path}...")
                if self.ckpt_path_check(base_path):
                    self.selected_checkpoint_path = os.path.abspath(base_path)
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
                            torch.load(
                                scheduler_path, weights_only=False, map_location="cpu"
                            )
                        )
                    else:
                        scheduler.load_state_dict(
                            torch.load(
                                scheduler_path, weights_only=False, map_location="cpu"
                            )
                        )
                        new_scheduler = scheduler

                    saved_state = torch.load(
                        model_path, weights_only=False, map_location="cpu"
                    )
                    if pp_model_parts is not None or pp_model_module_paths is not None:
                        if (
                            pp_model_parts is None
                            or pp_model_module_paths is None
                            or not pp_model_parts
                            or len(pp_model_parts) != len(pp_model_module_paths)
                        ):
                            raise ValueError(
                                "Pipeline checkpoint requires matching parts and prefixes"
                            )
                        stage_keys = []
                        expected_keys = set()
                        for part, prefix in zip(
                            pp_model_parts, pp_model_module_paths, strict=True
                        ):
                            if not isinstance(prefix, str):
                                raise ValueError(
                                    "Pipeline checkpoint prefixes must be strings"
                                )
                            keys = {
                                key: f"{prefix}.{key}" if prefix else key
                                for key in part.state_dict()
                            }
                            if expected_keys.intersection(keys.values()):
                                raise ValueError(
                                    "Pipeline checkpoint has duplicate stage keys"
                                )
                            expected_keys.update(keys.values())
                            stage_keys.append(keys)
                        if expected_keys != saved_state.keys():
                            raise ValueError(
                                "Incomplete pipeline checkpoint: "
                                f"missing={sorted(expected_keys - saved_state.keys())}, "
                                f"unexpected={sorted(saved_state.keys() - expected_keys)}"
                            )
                        # Validate ownership for every part before loading any.
                        # Repeated root prefixes are valid when their keys differ.
                        for part, keys in zip(pp_model_parts, stage_keys, strict=True):
                            part.load_state_dict(
                                {
                                    key: saved_state[full_key]
                                    for key, full_key in keys.items()
                                },
                                strict=True,
                            )
                    else:
                        model.load_state_dict(saved_state, strict=strict)
                    optimizer.load_state_dict(
                        torch.load(
                            optimizer_path, weights_only=False, map_location="cpu"
                        )
                    )

                    # Release CUDA cached memory to avoid fragmentation-induced
                    # OOM during the first training step after resume.
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(
                        f"[Policy] Checkpoint loaded successfully from {base_path}."
                    )
                    if self.load_checkpoint_hook is not None:
                        self.load_checkpoint_hook(
                            self,
                            data={
                                "checkpoint_path": base_path,
                                "extra_vars": extra_vars,
                            },
                        )
                    return extra_vars, new_scheduler
            except Exception as e:
                import traceback

                logger.error(
                    f"Error loading selected checkpoint from {base_path}: {e}; continuation is unsafe.\n{traceback.format_exc()}"
                )
                raise

        raise NoCheckpointFound(f"No complete checkpoint found at {base_paths}")

    def load_extra_info_from_checkpoint(self):
        self.selected_checkpoint_path = None
        extra_vars = {}
        base_paths = self.get_latest_ckpt_paths()
        # check whether checkpoint existing

        for base_path in base_paths:
            try:
                is_ckpt_path = self.ckpt_path_check(base_path)
                if is_ckpt_path:
                    self.selected_checkpoint_path = os.path.abspath(base_path)
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
                    return controller_checkpoint_metadata(extra_vars)
            except Exception as e:
                logger.error(
                    f"Error loading selected checkpoint from {base_path}: {e}; continuation is unsafe."
                )
                raise

        raise NoCheckpointFound(f"No complete checkpoint found at {base_paths}")

    def save_check(self, step: int, **kwargs):
        if self._is_master_rank():
            self._pending_save_checks[step] = dict(kwargs)
            self._publish_completed_checkpoints()

    def _publish_completed_checkpoints(self):
        """Defer pruning/best links until all saving ranks have committed.

        Never wait for another rank here. Incomplete saves retain the previous
        checkpoint; a later save_check/finalize can observe their completion.
        """
        for step, options in list(self._pending_save_checks.items()):
            path = os.path.join(self.ckpt_output_dir, f"step_{step}", "policy")
            if self.ckpt_path_check(path):
                self._commit_save_check(step, **options)
                del self._pending_save_checks[step]

    def _commit_save_check(self, step: int, **kwargs):
        if self._is_master_rank():
            step_ckpt_path = os.path.join(self.ckpt_output_dir, f"step_{step}")
            step_ckpt_abs_path = os.path.abspath(step_ckpt_path)
            self.saved_ckpt_step_dirs = [
                saved_dir
                for saved_dir in self.saved_ckpt_step_dirs
                if os.path.abspath(saved_dir) != step_ckpt_abs_path
            ]
            self.saved_ckpt_step_dirs.append(step_ckpt_path)
            # remove the old checkpoints
            # expected behavior:
            # Keep the best checkpoint, and delete the oldest checkpoint if the number of
            # checkpoints exceeds the max_keep.
            # Never delete the checkpoint that was just saved. If all older checkpoints
            # are protected as best, temporarily exceed max_keep.
            if len(self.saved_ckpt_step_dirs) > self.max_keep and self.max_keep != -1:
                step_to_delete = next(
                    (
                        saved_dir
                        for saved_dir in self.saved_ckpt_step_dirs
                        if os.path.abspath(saved_dir) != step_ckpt_abs_path
                        and not self._is_ckpt_dir_linked_as_best(saved_dir)
                    ),
                    None,
                )

                if step_to_delete is not None:
                    self.saved_ckpt_step_dirs.remove(step_to_delete)
                    logger.info(f"Deleting {step_to_delete}")
                    if self.save_mode == "async" and hasattr(self, "executor"):
                        self.pre_save_futures.append(
                            self.executor.submit(
                                self._delete_checkpoint, step_to_delete
                            )
                        )
                    else:
                        self._delete_checkpoint(step_to_delete)
                else:
                    logger.info(
                        "Keeping both the current and best checkpoints; "
                        "checkpoint count temporarily exceeds max_keep"
                    )

            val_score = kwargs.get("val_score", None)
            if val_score is not None:
                if ("loss" in self.metric and val_score < self.best_score) or (
                    "loss" not in self.metric and val_score > self.best_score
                ):
                    self.best_score = val_score
                    self.best_ckpt_abs_dir = os.path.realpath(step_ckpt_path)

                    best_dir = self._best_dir
                    os.makedirs(best_dir, exist_ok=True)

                    # Create symlink for checkpoint at root/best/checkpoints
                    best_ckpt_link = os.path.join(best_dir, "checkpoints")
                    # assume the best checkpoint is at self.ckpt_output_dir/step_<step>
                    if os.path.islink(best_ckpt_link):
                        os.unlink(best_ckpt_link)
                    # Relative output paths are relative to the process cwd,
                    # not the best/ directory containing this link.
                    os.symlink(step_ckpt_abs_path, best_ckpt_link)
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
                        os.symlink(
                            os.path.abspath(step_safetensors_path),
                            best_safetensors_link,
                        )
                        logger.info(f"Best safetensors updated to step_{step}")

                    # Save best score to file for persistence across resumes
                    self._save_best_score(val_score, step_ckpt_path)
