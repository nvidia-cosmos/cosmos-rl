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
import torch
import numpy as np
from typing import Type, Dict, Optional

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.util import (
    msgpack_c_long,
    msgunpack_c_long,
    fix_data_type_size,
)
import msgpack
from abc import ABC, abstractmethod


def wrap_to_cuda_tensor(device, key, obj, in_place=False):
    """
    wrap the object to cuda tensor for sync parameters using nccl.
    """
    if isinstance(obj, torch.Tensor):
        if isinstance(obj, torch.distributed.tensor.DTensor):
            obj = obj.to_local()

        if obj.device != device:
            if in_place:
                raise ValueError(
                    f"Object {key} is not on the same device as the model. Please set in_place to False."
                )
            obj = obj.to(device)
        return obj
    elif isinstance(obj, np.ndarray):
        if in_place:
            raise ValueError(
                f"Object {key} is not a tensor. Please set in_place to False."
            )
        obj = torch.from_numpy(obj).to(device)
        return obj
    else:
        if in_place:
            raise ValueError(
                f"Object {key} is not a tensor. Please set in_place to False."
            )
        if isinstance(obj, tuple):
            obj = tuple([x.tolist() if isinstance(x, np.ndarray) else x for x in obj])
            obj = fix_data_type_size(obj)
        bytes = msgpack.packb(obj, default=msgpack_c_long)
        obj = torch.frombuffer(bytes, dtype=torch.uint8).to(device)
        return obj


def extract_from_cuda_tensor(device, key, obj, tensor):
    """
    Extract the object from cuda tensor for sync parameters using nccl.
    """
    if isinstance(obj, torch.distributed.tensor.DTensor):
        if obj.device != device:
            local_shard = obj.to_local()
            local_shard.copy_(tensor)
    elif isinstance(obj, torch.Tensor):
        if obj.device != device:
            obj.copy_(tensor)
    elif isinstance(obj, np.ndarray):
        if obj.shape != tensor.shape:
            raise ValueError(
                f"Object {key} is not the same shape as the tensor. Please check the data consistency."
            )
        x = tensor.cpu()
        obj.copy_(x.numpy())
    else:
        np_arr = tensor.cpu()
        obj_new = msgpack.unpackb(bytes(np_arr.numpy()), ext_hook=msgunpack_c_long)
        if isinstance(obj, tuple):
            assert len(obj) == len(obj_new)
            obj = tuple(
                [
                    np.array(obj_new[idx])
                    if isinstance(x, np.ndarray)
                    else tuple(obj_new[idx])
                    if isinstance(x, tuple)
                    else obj_new[idx]
                    for idx, x in enumerate(obj)
                ]
            )
        else:
            obj = obj_new
    return obj


class Trainer(ABC):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: torch.cuda.Stream,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.parallel_dims = parallel_dims
        self.train_stream = train_stream
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        # Set data packer
        self.data_packer = kwargs.get("data_packer", None)
        self.val_data_packer = kwargs.get("val_data_packer", None)
        assert self.data_packer is not None, "data_packer is required"
        if self.config.validation.enable:
            if self.val_data_packer is None:
                self.val_data_packer = self.data_packer

    @abstractmethod
    def build_optimizers(self):
        """
        Build the optimizers for the trainer.
        """
        raise NotImplementedError("build_optimizers method must be implemented")

    @abstractmethod
    def build_lr_schedulers(self):
        """
        Build the lr schedulers for the trainer.
        """
        raise NotImplementedError("build_lr_schedulers method must be implemented")

    @abstractmethod
    def step_training(self):
        """
        One step of training.
        """
        raise NotImplementedError("train_step method must be implemented")

    @abstractmethod
    def step_validation(self):
        """
        One step of validation.
        """
        raise NotImplementedError("step_validation method must be implemented")

    @abstractmethod
    def export_safetensors(
        self,
        output_dir: str,
        rel_path: str,
        trainable_only: bool = False,
        is_final=False,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Export the model to safetensors or checkpoint.
        """
        raise NotImplementedError("export_safetensors method must be implemented")

    @abstractmethod
    def model_load_from_hf(self):
        """
        Load the model from HuggingFace.
        """
        raise NotImplementedError("model_load_from_hf method must be implemented")

    @abstractmethod
    def model_resume_from_checkpoint(self):
        """
        Resume the model from checkpoint.
        """
        raise NotImplementedError(
            "model_resume_from_checkpoint method must be implemented"
        )

    @property
    def pp_loss_fn(self):
        raise NotImplementedError("pp_loss_fn must be provided by subclass")


class TrainerRegistry:
    _TRAINER_REGISTRY: Dict[str, Type] = {}

    @classmethod
    def register_trainer_backend(cls, trainer_cls: Type, trainer_type: str):
        TrainerRegistry._TRAINER_REGISTRY[trainer_type] = trainer_cls

    @classmethod
    def register(
        x,
        trainer_type: str,
        *,
        allow_override: bool = False,
    ):
        def decorator(cls: Type) -> Type:
            assert issubclass(
                cls, Trainer
            ), "Registered trainer must be a subclass of Trainer."
            assert isinstance(trainer_type, str), "trainer_type must be a string."
            if (
                not allow_override
                and trainer_type in TrainerRegistry._TRAINER_REGISTRY
                and TrainerRegistry._TRAINER_REGISTRY[trainer_type] != cls
            ):
                raise ValueError(f"Trainer {trainer_type} is already registered.")
            TrainerRegistry.register_trainer_backend(
                cls,
                trainer_type,
            )
            return cls

        return decorator

    @classmethod
    def check_trainer_type_supported(cls, trainer_type: str) -> bool:
        return trainer_type in TrainerRegistry._TRAINER_REGISTRY

    @classmethod
    def get_trainer_cls(cls, trainer_type: str) -> Type:
        if trainer_type not in TrainerRegistry._TRAINER_REGISTRY:
            raise ValueError(f"Trainer {trainer_type} is not supported.")
        return TrainerRegistry._TRAINER_REGISTRY[trainer_type]
