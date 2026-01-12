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

import enum
import functools
import multiprocessing
import os
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from multiprocessing import get_context
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config.wfm import CheckpointConfig
from cosmos_rl.utils.wfm import distributed
from cosmos_rl.utils.wfm.utils import timer, to
from cosmos_rl.utils.wfm.io import object_store
from cosmos_rl.utils.wfm.io.s3_filesystem import (
    S3StorageWriter,
    S3StorageReader,
)
from cosmos_rl.utils.wfm.io.easy_io import easy_io


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

try:
    from torch.distributed.checkpoint.default_planner import (
        DefaultLoadPlanner as _DefaultLoadPlanner,
    )
    from torch.distributed.checkpoint.default_planner import (
        DTensor,
        LoadPlan,
        _create_read_items,
        _version,
        flatten_state_dict,
    )
    from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

    def create_default_local_load_plan(
        state_dict: dict[str, Any],
        metadata: Metadata,
        strict: bool = True,
        dcp_allow_mismatched_size: bool = False,
    ) -> LoadPlan:
        requests = []
        """
        Create the ``LoadPlan`` used by DefaultLoadPlanner.

        It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

        The default behavior is to match key exactly between state_dict and metadata.
        It handles resharding by issuing multiple read requests against storage in order to match
        load requirements.
        """

        for fqn, obj in state_dict.items():
            if fqn.endswith("._extra_state"):  # dirty TE attention package!
                continue
            # ignore state_dict keys which do not exist in `state_dict` if strict=False
            if fqn not in metadata.state_dict_metadata:
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
                else:
                    continue

            md = metadata.state_dict_metadata[fqn]

            if not dcp_allow_mismatched_size:
                if (
                    isinstance(md, TensorStorageMetadata)
                    and getattr(obj, "size", None) is not None
                    and md.size != obj.size()
                ):
                    if not strict:
                        logger.critical(
                            f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}"
                        )
                        continue
                    else:
                        raise ValueError(
                            f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}",
                        )
            # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
            # gets called only when the current rank is part of the mesh for the corresponding DTensor.
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    requests += _create_read_items(fqn, md, obj)
            else:
                requests += _create_read_items(fqn, md, obj)

        return LoadPlan(requests)

    class DefaultLoadPlanner(_DefaultLoadPlanner):
        def set_partial_channel_weight(self, dcp_allow_mismatched_size: bool):
            self.dcp_allow_mismatched_size = dcp_allow_mismatched_size

        def create_local_plan(self) -> LoadPlan:
            assert self.metadata is not None
            if self.flatten_state_dict:
                # To support checkpoints that are saved before v2.4, we have to
                # differentiate if the missing keys are due to old checkpoints.
                # The contracts are:
                # 1. There are 3 cases when we found a missing key.
                #    1.1 Actual missing key, but allow_partial_load is False
                #    1.2 Actual missing key, but allow_partial load is True
                #    1.3 Old checkpoint, but allow_partial_load is False
                #    1.4 Old checkpoint, but allow_partial_load is True
                # 2. If we found a missing key, we first convert the keys back to
                #    the key format of v2.3
                # 3. If the previous missing keys are in the v2.3 keys, we assume
                #    this is a old checkpoint.
                # 4. Pass the state_dict to `create_default_local_load_plan()`,
                #    which has the logic to check missing for allow_partial_load.
                # So for 1.2 and 1.4 cases, we delegate allow_partial_load check to
                # `create_default_local_load_plan()`. The logic here is to determine
                # whether the checkpoint belong to 2.3 (or before) or 2.4 (or after).
                current_keys = set(self.state_dict.keys())
                load_keys = set(self.metadata.state_dict_metadata.keys())
                missing_keys = load_keys - current_keys
                if missing_keys:
                    _version._derived_version = "2_3"
                    old_state_dict, old_mappings = flatten_state_dict(
                        self.original_state_dict
                    )
                    old_keys = set(old_state_dict.keys())
                    if old_keys & missing_keys:
                        self.state_dict, self.mappings = old_state_dict, old_mappings
                    # _derived_version is only used by flatten_state_dict now.
                    # Set it back to None so that later we can save to a new version.
                    _version._derived_version = None

            return create_default_local_load_plan(
                self.state_dict,
                self.metadata,
                not self.allow_partial_load,
                getattr(self, "dcp_allow_mismatched_size", False),
            )

    logger.critical(
        "for the back comptiable pytorch! New DefaultLoadPlanner class is created."
    )
except ImportError as e:
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    logger.critical(f"{e}, using default planner")


StateDictItemPath = namedtuple("StateDictItemPath", ["state_dict", "save_path"])


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


class AbstractCheckpointer(ABC):
    """The checkpointer class. Supports checkpoint saving/loading to both local disk or object store."""

    def __init__(
        self,
        config_checkpoint,
        config_job,
        callbacks=None,
    ):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        self.config_checkpoint = config_checkpoint
        # Set the callback functions.
        self.callbacks = callbacks
        self.save_to_object_store = config_checkpoint.save_to_object_store.enabled
        self.load_from_object_store = config_checkpoint.load_from_object_store.enabled

        # Set checkpoint directories for local and object store paths
        self._local_dirname = os.path.join(config_job.path_local, "checkpoints")
        self._object_store_dirname = os.path.join(config_job.path, "checkpoints")

        self.strict_resume = config_checkpoint.strict_resume
        self.load_path = config_checkpoint.load_path or None
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None
        self.verbose = config_checkpoint.verbose
        self.keys_not_to_resume = config_checkpoint.keys_not_to_resume
        self.broadcast_via_filesystem = config_checkpoint.broadcast_via_filesystem
        # Create the object store client interface.
        if config_checkpoint.load_from_object_store.enabled:
            self.load_s3_backend_key = "_ckpt_s3_loader"
            easy_io.set_s3_backend(
                key="_ckpt_s3_loader",
                backend_args={
                    "backend": "s3",
                    "path_mapping": {
                        "s3://ckpt/": f"s3://{config_checkpoint.load_from_object_store.bucket}/",
                    },
                    "s3_credential_path": config_checkpoint.load_from_object_store.credentials,
                },
            )
        else:
            self.load_s3_backend_key = None

        if config_checkpoint.save_to_object_store.enabled:
            self.save_s3_backend_key = "_ckpt_s3_saver"
            easy_io.set_s3_backend(
                key="_ckpt_s3_saver",
                backend_args={
                    "backend": "s3",
                    "path_mapping": {
                        "s3://ckpt/": f"s3://{config_checkpoint.save_to_object_store.bucket}/",
                    },
                    "s3_credential_path": config_checkpoint.save_to_object_store.credentials,
                },
            )
        else:
            self.save_s3_backend_key = None

    @abstractmethod
    def save(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        pass

    @abstractmethod
    def load(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
    ) -> int:
        pass

    @property
    def save_bucket(self):
        """Get the bucket name for saving checkpoints."""
        return (
            self.config_checkpoint.save_to_object_store.bucket
            if self.save_to_object_store
            else None
        )

    @property
    def load_bucket(self):
        """Get the bucket name for loading checkpoints."""
        return (
            self.config_checkpoint.load_from_object_store.bucket
            if self.load_from_object_store
            else None
        )

    @property
    def save_dirname(self):
        return (
            f"s3://{self.save_bucket}/{self._object_store_dirname}"
            if self.save_to_object_store
            else self._local_dirname
        )

    @property
    def load_dirname(self):
        return (
            f"s3://{self.load_bucket}/{self._object_store_dirname}"
            if self.load_from_object_store
            else self._local_dirname
        )

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        checkpoint_path = os.path.join(self.load_dirname, "latest_checkpoint.txt")
        if easy_io.exists(f"{checkpoint_path}", backend_key=self.load_s3_backend_key):
            checkpoint_file = easy_io.load(
                f"{checkpoint_path}", backend_key=self.load_s3_backend_key
            ).strip()

        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        checkpoint_path = os.path.join(self.save_dirname, "latest_checkpoint.txt")
        easy_io.dump(
            content,
            checkpoint_path,
            backend_key=self.save_s3_backend_key,
        )

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if not easy_io.exists(
            f"{checkpoint_path}", backend_key=self.load_s3_backend_key
        ):
            raise FileNotFoundError(f"File not found (object store): {checkpoint_path}")


class Checkpointer:
    """The checkpointer class. Supports checkpoint saving/loading to both local disk or object store."""

    def __init__(self, config_checkpoint, config_job, callbacks):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        # Set the callback functions.
        self.callbacks = callbacks
        # TODO(snah): multi-EMA
        # TODO(snah): (skip mismatch)
        # TODO(snah): load_master_and_broadcast
        self.checkpoint_dir_local = f"{config_job.path_local}/checkpoints"
        self.checkpoint_dir_object_store = f"{config_job.path}/checkpoints"
        self.save_to_object_store = config_checkpoint.save_to_object_store.enabled
        self.load_from_object_store = config_checkpoint.load_from_object_store.enabled
        self.strict_resume = config_checkpoint.strict_resume
        self.load_path = config_checkpoint.load_path or None
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None
        # Create the object store client interface.
        if self.save_to_object_store:
            self.object_store_saver = object_store.ObjectStore(
                config_checkpoint.save_to_object_store
            )
        if self.load_from_object_store:
            self.object_store_loader = object_store.ObjectStore(
                config_checkpoint.load_from_object_store
            )

    def save(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (CosmosVisionGenModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}.pt"

        if distributed.get_rank() == 0:
            state_dict = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                grad_scaler=grad_scaler.state_dict(),
                iteration=iteration,
            )
            state_dict = to(state_dict, device="cpu")
            self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
            # Wait for previous saver thread to end.
            if self.save_thread:
                self.save_thread.join()
            # Run the checkpoint saver in a separate thread.
            self.save_thread = threading.Thread(
                target=self._save_worker_object_store
                if self.save_to_object_store
                else self._save_worker_local,
                daemon=False,
                args=(state_dict, checkpoint_file, distributed.get_rank()),
            )
            self.save_thread.start()

        # Note: Checkpoints are saved on a separate thread and this callback is not accurate.
        # Please check logs from on_save_checkpoint_success() for better accuracy
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    @timer("checkpoint saving (local)")
    def _save_worker_local(
        self, state_dict: dict[str, torch.Tensor], checkpoint_file: str, rank: int = 0
    ) -> None:
        """Worker to save checkpoint to local disk, spawned with a child thread (runs in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir_local, checkpoint_file)
        os.makedirs(self.checkpoint_dir_local, exist_ok=True)
        try:
            torch.save(state_dict, checkpoint_path)
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            logger.info(f"Saved checkpoint (local): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Checkpoint failed to save (local): {e}")

    @timer("checkpoint saving (object store)")
    def _save_worker_object_store(
        self, state_dict: dict[str, torch.Tensor], checkpoint_file: str, rank: int = 0
    ) -> None:
        """Worker to upload checkpoint to object store, spawned with a child thread (in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir_object_store, checkpoint_file
        )
        try:
            self.object_store_saver.save_object(
                state_dict, key=checkpoint_path, type="torch"
            )
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            logger.info(f"Saved checkpoint (object store): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Checkpoint failed to upload (object store): {e}")

    @timer("checkpoint loading")
    def load(
        self,
        model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (CosmosVisionGenModel): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        self.callbacks.on_load_checkpoint_start(model)

        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = (
                self.checkpoint_dir_object_store
                if self.load_from_object_store
                else self.checkpoint_dir_local
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
            only_resume_scheduler = True
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                resume = self.load_training_state
                only_resume_scheduler = self.only_load_scheduler_state
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
                only_resume_scheduler = False
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            if self.load_from_object_store:
                logger.info(f"Loading checkpoint (object store): {checkpoint_path}")
                state_dict = self.object_store_loader.load_object(
                    key=checkpoint_path, type="torch", max_attempts=20
                )
                logger.info(
                    f"Complete loading checkpoint (object store): {checkpoint_path}"
                )
            else:
                logger.info(f"Loading checkpoint (local): {checkpoint_path}")
                state_dict = torch.load(
                    checkpoint_path, map_location=lambda storage, loc: storage
                )
                logger.info(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            # Load the state dicts.
            logger.info("- Loading the model...")
            model.load_state_dict(state_dict["model"], strict=self.strict_resume)
            if resume or only_resume_scheduler:
                iteration = state_dict["iteration"]
                assert scheduler
                logger.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
            else:
                iteration = 0
            if resume:
                assert optimizer
                logger.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optimizer"])
                logger.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(state_dict["grad_scaler"])
                logger.info(
                    f"Done with loading the checkpoint (iteration {iteration})."
                )
            else:
                logger.info("Done with loading the checkpoint.")
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            iteration = 0
            logger.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(
            model, iteration=iteration, checkpoint_path=checkpoint_path
        )

        return iteration

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        if self.load_from_object_store:
            latest_path = os.path.join(
                self.checkpoint_dir_object_store, "latest_checkpoint.txt"
            )
            if self.object_store_loader.object_exists(key=latest_path):
                checkpoint_file = self.object_store_loader.load_object(
                    key=latest_path, type="text"
                ).strip()
        else:
            latest_path = os.path.join(
                self.checkpoint_dir_local, "latest_checkpoint.txt"
            )
            if os.path.isfile(latest_path):
                checkpoint_file = open(latest_path).read().strip()
        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        if self.save_to_object_store:
            latest_path = os.path.join(
                self.checkpoint_dir_object_store, "latest_checkpoint.txt"
            )
            self.object_store_saver.save_object(content, key=latest_path, type="text")
        else:
            latest_path = os.path.join(
                self.checkpoint_dir_local, "latest_checkpoint.txt"
            )
            with open(latest_path, "w") as file:
                file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if self.load_from_object_store:
            if not self.object_store_loader.object_exists(key=checkpoint_path):
                raise FileNotFoundError(
                    f"File not found (object store): {checkpoint_path}"
                )
        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"File not found (local): {checkpoint_path}")

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()


class MultiRankCheckpointer(Checkpointer):
    def save(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (CosmosVisionGenModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        # checkpoint_file = f"iter_{iteration:09}.pt"
        postfix, _, total_ema_num = model.get_ckpt_postfix()
        checkpoint_file = f"iter_{iteration:09}{postfix}.pt"
        save_ranks = list(range(total_ema_num))
        for _rank in save_ranks:
            if distributed.get_rank() == _rank:
                state_dict = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    grad_scaler=grad_scaler.state_dict(),
                    iteration=iteration,
                )
                state_dict = to(state_dict, device="cpu")
                self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
                # Wait for previous saver thread to end.
                if self.save_thread:
                    self.save_thread.join()
                # Run the checkpoint saver in a separate thread.
                self.save_thread = threading.Thread(
                    target=self._save_worker_object_store
                    if self.save_to_object_store
                    else self._save_worker_local,
                    daemon=False,
                    args=(state_dict, checkpoint_file, distributed.get_rank()),
                )
                self.save_thread.start()

    @timer("checkpoint loading")
    def load(
        self,
        model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (CosmosVisionGenModel): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # different from base checkpointer, this support multi-EMA
            postfix, _, total_ema_num = model.get_ckpt_postfix()
            latest_checkpoint_file = latest_checkpoint_file.replace(
                ".pt", f"{postfix}.pt"
            )
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = (
                self.checkpoint_dir_object_store
                if self.load_from_object_store
                else self.checkpoint_dir_local
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                # different from base checkpointer, this support multi-EMA
                postfix, _, total_ema_num = model.get_ckpt_postfix()
                checkpoint_path = checkpoint_path.replace(".pt", f"{postfix}.pt")
                resume = self.load_training_state
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            if self.load_from_object_store:
                logger.info(f"Loading checkpoint (object store): {checkpoint_path}")
                state_dict = self.object_store_loader.load_object(
                    key=checkpoint_path, type="torch"
                )
                logger.info(
                    f"Complete loading checkpoint (object store): {checkpoint_path}"
                )
            else:
                logger.info(f"Loading checkpoint (local): {checkpoint_path}")
                state_dict = torch.load(
                    checkpoint_path, map_location=lambda storage, loc: storage
                )
                logger.info(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            # Load the state dicts.
            logger.info("- Loading the model...")
            logger.critical(
                model.load_state_dict(state_dict["model"], strict=self.strict_resume)
            )
            if resume:
                iteration = state_dict["iteration"]
                assert optimizer and scheduler
                logger.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optimizer"])
                logger.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
                logger.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(state_dict["grad_scaler"])
                logger.info(
                    f"Done with loading the checkpoint (iteration {iteration})."
                )
            else:
                iteration = 0
                logger.info("Done with loading the checkpoint.")
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            iteration = 0
            logger.info("Training from scratch.")
        torch.cuda.empty_cache()
        return iteration


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(
        self, model: Union[nn.Module, List[nn.Module]], load_ema_to_reg: bool = False
    ):
        self.model = [model] if isinstance(model, nn.Module) else model
        self.load_ema_to_reg = load_ema_to_reg

    def state_dict(self) -> Dict[str, Any]:
        _state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        if self.load_ema_to_reg:
            assert (
                not self.model[0].config.ema.enabled
            ), "EMA is enabled, can not load EMA weights to regular model weights"
            all_keys = list(_state_dict.keys())
            for k in all_keys:
                if k.startswith("net."):
                    _state_dict[k.replace("net.", "net_ema.")] = _state_dict.pop(k)
                else:
                    _state_dict.pop(k)
        return _state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.load_ema_to_reg:
            assert (
                not self.model[0].config.ema.enabled
            ), "EMA is enabled, can not load EMA weights to regular model weights"
            all_keys = list(state_dict.keys())
            assert all(
                k.startswith("net_ema.") for k in all_keys
            ), "All keys must start with net_ema."
            for k in all_keys:
                state_dict[k.replace("net_ema.", "net.")] = state_dict.pop(k)
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    def __init__(self, iteration: int, elapsed_time: float, succeeded: bool):
        self.iteration = iteration
        self.elapsed_time = elapsed_time
        self.succeeded = succeeded

    def __str__(self):
        return f"SaveDone(iteration={self.iteration}, elapsed_time={self.elapsed_time}, succeeded={self.succeeded})"


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    checkpoint_config,
    job_config,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        checkpoint_config: Configuration settings for checkpoint saving behavior
        job_config: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(
        checkpoint_config, job_config, None, disable_async=True
    )

    try:
        while True:
            logger.debug(
                "Checkpoint background process is ready for next task, waiting for new state_dict"
            )
            received_data = receiver_queue.get()
            logger.debug("Received new state_dict")

            if isinstance(received_data, Terminate):
                logger.info(
                    "Received termination signal in checkpoint background process, closing sender queue"
                )
                sender_queue.put(Terminate())
                sender_queue.close()
                return

            assert isinstance(
                received_data, tuple
            ), "Received data must be a tuple of (state_dict, checkpoint_path)"
            state_dict, checkpoint_path = received_data

            # Save checkpoint and measure time taken
            start_time = time.monotonic()
            iteration = state_dict["trainer"][0]["iteration"]
            elapsed_time = 0
            succeeded = False
            try:
                checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)
                elapsed_time = time.monotonic() - start_time
                logger.info(
                    f"Checkpoint saved successfully in background process. Time taken: {elapsed_time:.2f} seconds, iteration: {iteration}"
                )
                succeeded = True
            except Exception as e:
                logger.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
                # continue because if the thread exits, the main thread keeps on adding to the queue
            finally:
                if elapsed_time == 0:
                    elapsed_time = time.monotonic() - start_time
                sender_queue.put(SaveDone(iteration, elapsed_time, succeeded))

    finally:
        logger.info("Cleaning up: destroying distributed process group")
        torch.distributed.destroy_process_group()


class DistributedCheckpointer(AbstractCheckpointer):
    KEYS_TO_SAVE = ["model", "optim", "scheduler", "trainer"]

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job,
        callbacks=None,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.dcp_async_mode_enabled:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if disable_async:
            self.async_mode = AsyncMode.DISABLED

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()

    def keys_to_resume_during_load(self) -> Tuple[Set, Union[str, None]]:
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []

        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.KEYS_TO_SAVE)
        else:
            if self.load_path and not str(self.load_path).endswith(".pt"):
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://{self.config_checkpoint.load_from_object_store.bucket}/{checkpoint_path}"
                    if not re.search(r"/checkpoints/iter_\d{9}/?$", checkpoint_path):
                        # If path doesn't end with specific checkpoint, read latest checkpoint file
                        latest_ckpt_path = os.path.join(
                            checkpoint_path, "checkpoints/latest_checkpoint.txt"
                        )
                        if easy_io.exists(
                            latest_ckpt_path, backend_key=self.load_s3_backend_key
                        ):
                            checkpoint_file = easy_io.load(
                                latest_ckpt_path, backend_key=self.load_s3_backend_key
                            ).strip()
                            checkpoint_path = (
                                f"{checkpoint_path}/checkpoints/{checkpoint_file}"
                            )
                        else:
                            logger.warning(
                                f"Latest checkpoint file {latest_ckpt_path} not found, use the load_path instead: {checkpoint_path}"
                            )

                if self.load_training_state:
                    resume_keys.extend(self.KEYS_TO_SAVE)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None
        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert (
                    key in self.KEYS_TO_SAVE
                ), f"Invalid key to resume: {key} not in {self.KEYS_TO_SAVE}"
            resume_keys = [
                key for key in resume_keys if key not in self.keys_not_to_resume
            ]
        return set(resume_keys), checkpoint_path

    @timer("checkpoint loading")
    def load(
        self,
        model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        logger.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        # TODO: (qsh 2025-01-01) the current interface design can not load callback states.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                load_planner = DefaultLoadPlanner(allow_partial_load=True)
                if hasattr(load_planner, "set_partial_channel_weight"):
                    logger.critical(
                        f"set_partial_channel_weight: {self.config_checkpoint.dcp_allow_mismatched_size}"
                    )
                    load_planner.set_partial_channel_weight(
                        self.config_checkpoint.dcp_allow_mismatched_size
                    )
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                logger.critical(f"Start loading checkpoint from {checkpoint_path}")
                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                torch.distributed.barrier()
                logger.critical(f"starting {cur_key_ckpt_full_path}")
                if key == "model":
                    logger.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()
                    # TODO: (qsh 2025-01-23) set flag `allow_partial_load`
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    if model.config.rl.enabled:  # deal with referene model
                        if not model.config.ema.enabled:
                            logger.info(
                                "- Loading the model... load EMA weights if possible"
                            )
                            loading_keys = storage_reader.read_metadata().state_dict_metadata.keys()
                            all_keys = list(_state_dict.keys())
                            for k in all_keys:
                                # if the key is loaded from the checkpoint, then we convert it to the regular model weights
                                if k.startswith("net_ema.") and k in loading_keys:
                                    _state_dict[k.replace("net_ema.", "net.")] = (
                                        _state_dict.pop(k)
                                    )
                                elif k not in loading_keys:
                                    _state_dict.pop(k)
                        else:
                            logger.info(
                                "- Loading the model... Remove keys that are not in the checkpoint"
                            )
                            loading_keys = storage_reader.read_metadata().state_dict_metadata.keys()
                            all_keys = list(_state_dict.keys())
                            for k in all_keys:
                                if k not in loading_keys:
                                    _state_dict.pop(k)

                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    logger.info("- Loading the optimizer...")
                    _optim_wrapper = OptimizerWrapper(model, optimizer)
                    _state_dict = _optim_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    logger.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    scheduler.load_state_dict(_state_dict)
                elif key == "trainer":
                    logger.info("- Loading the trainer...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            logger.critical(
                f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}"
            )
        else:
            logger.info("Training from scratch.")
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(
                model, iteration=iteration, checkpoint_path=checkpoint_path
            )
        return iteration

    def _async_with_pinned_memory(
        self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]
    ) -> None:
        try:
            from torch.distributed._state_dict_utils import (
                _copy_state_dict,
                _create_cpu_state_dict,
            )
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_ckpt_file = checkpoint_file

        # TODO: (qsh 2025-01-02) a better name?
        self.maybe_wait_for_staging()

    def maybe_wait_for_staging(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM and self.staging:
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait(
                    (self.cpu_offload_state_dict, self.staging_ckpt_file)
                )

            sync_func()
            self.staging = False

    def get_previous_checkpoint_results(self, wait_for: int = 0) -> None:
        """Get the results of previously submitted checkpoints and pass them to callbacks if checkpoint succeeded"""
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            try:
                start_time = time.monotonic()
                while not self.mp_queue_recv.empty() or wait_for > 0:
                    try:
                        ret = self.mp_queue_recv.get(timeout=1)
                        if isinstance(ret, Terminate):
                            logger.info(
                                "Received termination event from checkpoint background process"
                            )
                            break
                        save_done: SaveDone = ret
                        logger.logger.info(
                            f"Received checkpoint save result: {save_done}"
                        )
                        if self.callbacks is not None and save_done.succeeded:
                            self.callbacks.on_save_checkpoint_success(
                                iteration=save_done.iteration,
                                elapsed_time=save_done.elapsed_time,
                            )
                    except queue.Empty:
                        elapsed_time = time.monotonic() - start_time
                        if elapsed_time > wait_for:
                            break
            except (EOFError, BrokenPipeError):
                logger.info("Queue was closed by checkpoint background process")

    def get_storage_writer(
        self, checkpoint_path: str
    ) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.save_to_object_store:
            return S3StorageWriter(
                credential_path=self.config_checkpoint.save_to_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(
        self, checkpoint_path: str
    ) -> Union[S3StorageReader, FileSystemReader]:
        if self.load_from_object_store:
            return S3StorageReader(
                credential_path=self.config_checkpoint.load_from_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemReader(checkpoint_path)

    def save_state_dict_worker(
        self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str
    ) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            storage_writer = self.get_storage_writer(full_checkpoint_path)
            dcp.save(
                v,
                storage_writer=storage_writer,
                planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
            )

        self._write_latest_checkpoint_file(checkpoint_file)
        logger.critical(
            f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}"
        )

    def save(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (CosmosVisionGenModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.get_previous_checkpoint_results(wait_for=0)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": OptimizerWrapper(model, optimizer).state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            try:
                self.save_state_dict_worker(to_save_dict, checkpoint_file)
            finally:
                if self.callbacks is not None:
                    self.callbacks.on_save_checkpoint_success(
                        iteration=iteration, elapsed_time=time.monotonic() - start_time
                    )

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.get_previous_checkpoint_results(wait_for=60)
                self.mp.join()


# https://github.com/facebookresearch/fvcore/blob/9d683aae73fb899dd35d6cf6720e5ef567761c57/fvcore/common/checkpoint.py
def non_strict_load_model(
    model: torch.nn.Module, checkpoint_state_dict: dict
) -> _IncompatibleKeys:
    # workaround https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            if "_extra_state" in k:  # Key introduced by TransformerEngine for FP8
                logger.warning(
                    f"Skipping key {k} introduced by TransformerEngine for FP8 in the checkpoint."
                )
                continue
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(
                model_param, torch.nn.parameter.UninitializedParameter
            ):
                continue
            if not isinstance(model_param, torch.Tensor):
                raise ValueError(
                    f"Find non-tensor parameter {k} in the model. type: {type(model_param)} {type(checkpoint_state_dict[k])}, please check if this key is safe to skip or not."
                )

            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    def _get_module_for_key(
                        model: torch.nn.Module, key: str
                    ) -> torch.nn.Module:
                        # foo.bar.param_or_buffer_name -> [foo, bar]
                        key_parts = key.split(".")[:-1]
                        cur_module = model
                        for key_part in key_parts:
                            cur_module = getattr(cur_module, key_part)
                        return cur_module

                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(model, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    # Remove keys with "_extra_state" suffix, which are non-parameter items introduced by TransformerEngine for FP8 handling
    missing_keys = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    unexpected_keys = [
        k for k in incompatible.unexpected_keys if "_extra_state" not in k
    ]
    return _IncompatibleKeys(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        incorrect_shapes=incorrect_shapes,
    )


def load_checkpoint(
    model_parts: list[nn.Module],
    ckpt_dir,
    checkpoint_cred="./credentials/s3_training.secret",
    model_ckpt_key_map: dict[str, str] = {},
):
    logger.info(f"Loading checkpoint from {ckpt_dir}.")

    _model_wrapper = ModelWrapper(model_parts)
    state_dict = _model_wrapper.state_dict()
    # remove _extra_state
    state_dict = {
        k: v for k, v in state_dict.items() if not k.endswith("._extra_state")
    }

    # remap keys if needed
    if model_ckpt_key_map:
        for model_key, checkpoint_key in model_ckpt_key_map.items():
            state_dict[checkpoint_key] = state_dict.pop(model_key)
            logger.info(f"Re-mapping {model_key} to {checkpoint_key}")

    if ckpt_dir.startswith("s3://"):
        storage_reader = S3StorageReader(
            credential_path=checkpoint_cred,
            path=ckpt_dir,
        )
        dcp.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
        )
    else:
        fs_storage_reader = dcp.FileSystemReader(ckpt_dir)
        dcp.load(
            state_dict=state_dict,
            storage_reader=fs_storage_reader,
        )

    # inverse the remapping if needed
    if model_ckpt_key_map:
        for model_key, checkpoint_key in model_ckpt_key_map.items():
            state_dict[model_key] = state_dict.pop(checkpoint_key)
            logger.info(f"Inverse re-mapping {checkpoint_key} to {model_key}")

    _model_wrapper.load_state_dict(state_dict)

    logger.info(f"Finished loading checkpoint from {ckpt_dir}.")
