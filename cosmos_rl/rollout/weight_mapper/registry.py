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

from typing import Dict, Type, Tuple, List, Union
import torch
from torch import nn
from cosmos_rl.utils.parallelism import ParallelismConfig
from cosmos_rl.utils.parallelism_map import (
    DimRankInfo,
)
from abc import ABC, abstractmethod
from cosmos_rl.utils.pynccl import nccl_recv
from transformers import AutoConfig
import cosmos_rl.utils.util as util
from cosmos_rl.utils.logging import logger


class WeightMapper(ABC):
    _MODEL_WEIGHT_MAPPER_REGISTRY: Dict[str, Tuple[Type["WeightMapper"], int]] = {}

    def __init__(self, hf_config_path: str):
        logger.info(f"WeightMapper: {type(self).__name__} is being initialized.")
        self.config = util.retry(AutoConfig.from_pretrained)(hf_config_path)
        self.vllm_weight_inplace_view_map: Dict[str, torch.Tensor] = None
        self.recv_key_n_rank_list: List[Tuple[str, Tuple[int]]]

    def rollout_prepare_recv(self, model: nn.Module):
        if (
            self.vllm_weight_inplace_view_map is None
            or self.recv_key_n_rank_list is None
        ):
            self.vllm_weight_inplace_view_map, self.recv_key_n_rank_list = (
                self.rollout_prepare_recv_impl(model)
            )
            self.recv_key_n_rank_list.sort(key=lambda x: x[0])

        return self.recv_key_n_rank_list

    def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        inst: Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]],
        communicator_index: Dict[int, int],
        do_weight_sync_check: bool = False,
    ):
        p_rank, r_rank, tensor_split_strategys, dest_name, shape = inst
        assert r_rank == global_rank_of_rollout

        target_tensor = self.vllm_weight_inplace_view_map[dest_name]

        view = target_tensor.cosmos_slice(tensor_split_strategys)

        if do_weight_sync_check:
            cloned_target_tensor = target_tensor.clone()
            # clear the current view
            view.zero_()

        recv_tensor = None
        if view.is_contiguous():
            recv_tensor = view
        else:
            # new a temp tensor
            recv_tensor = torch.empty_like(view)

        # logger.info(
        #     f"[Rollout] rank {global_rank_of_rollout} recv tensor: {dest_name} from rank {p_rank} with shape: {view.shape} out of {target_tensor.shape} with dtype {view.dtype} on device {view.device}"
        # )
        nccl_recv(recv_tensor, p_rank, communicator_index[p_rank])

        # inplace copy
        if not view.is_contiguous():
            view.copy_(recv_tensor)

        if do_weight_sync_check:
            # If the weight sync between Policy and Rollout is correct, the
            # `target_tensor` would have no change.
            # TODO: (lms) When we support quantization in rollout side,
            # we should handle the numerical error of quantized weight, not
            # just apply `torch.allclose` simply.
            if not torch.allclose(cloned_target_tensor, target_tensor):
                raise ValueError(
                    f"Weight sync check failed after weight sync instruction: {inst}"
                )

        return recv_tensor.numel() * recv_tensor.element_size()

    @abstractmethod
    def rollout_prepare_recv_impl(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, int]]]:
        """
        Rollout prepare recv list for P2R weight sync:
            - vllm_weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[Tuple[str, int]]: the list of recv key and its tensor rank
        """
        pass

    def name_to_model_index(self, dest_name: str) -> int:
        return 0

    @abstractmethod
    def get_rollout_parallelism(self, replica_parallelism: ParallelismConfig):
        pass

    @abstractmethod
    def get_policy_parallelism(self, replica_parallelism: ParallelismConfig):
        pass

    @abstractmethod
    def get_policy_parallelism_strategy(self):
        pass

    @abstractmethod
    def get_rollout_parallelism_strategy(self):
        pass

    @classmethod
    def register_class(
        cls,
        reg_key: Union[str, List[str]],
        *,
        allow_override: bool = False,
        n_model: int = 1,
    ):
        if isinstance(reg_key, str):
            reg_key = [reg_key]

        def decorator(cls: Type) -> Type:
            for key in reg_key:
                if not allow_override and key in cls._MODEL_WEIGHT_MAPPER_REGISTRY:
                    raise ValueError(f"Class '{key}' is already registered.")
                cls._MODEL_WEIGHT_MAPPER_REGISTRY[key] = (cls, n_model)
            return cls

        return decorator

    @classmethod
    def get_weight_mapper(cls, model_type: str) -> Tuple[Type["WeightMapper"], int]:
        if model_type not in cls._MODEL_WEIGHT_MAPPER_REGISTRY:
            raise ValueError(f"ModelType '{model_type}' is not supported now.")

        return cls._MODEL_WEIGHT_MAPPER_REGISTRY[model_type]
