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

from typing import Dict, Type, Tuple, List, Union, Any
import torch
from cosmos_rl.utils.parallelism import ParallelismConfig
from abc import ABC, abstractmethod
from transformers import AutoConfig
from cosmos_rl.utils.logging import logger


class WeightMapper(ABC):
    _MODEL_WEIGHT_MAPPER_REGISTRY: Dict[str, Tuple[Type["WeightMapper"], int]] = {}

    def __init__(self, hf_config: AutoConfig):
        logger.info(f"WeightMapper: {type(self).__name__} is being initialized.")
        self.config = hf_config

    @torch.no_grad()
    def policy_maybe_decompose_weights_to_hf_naming(self, name, param):
        """
        Decompose the weights of the model parameters into fine-grained weights
        This is especially useful for models with non-symmetric parameter layout than the original HuggingFace one
        For example, MoE experts' weights are stacked in the 0th dimension,
        while they are stored in different keys in the original HuggingFace naming convention
        """
        yield name, param

    def policy_pre_P2R_gather_required_for_sync(self, name: str) -> bool:
        """
        For P->R weight sync, some weights need to be pre-collected before first `nccl_send/recv` instruction.
        To not be messed up with the following `nccl_send/recv` instructions,
        pre-collect those weights before first `nccl_send/recv` instruction.

        Args:
            name (str): The name of the tensor.
        Returns:
            bool: True if the tensor sync precollect is required, False otherwise.
        """
        return False

    @abstractmethod
    def rollout_prepare_recv(
        self, vllm_model: Any
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, int]]]:
        """
        Rollout prepare recv list for P2R weight sync:
            - vllm_weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[Tuple[str, int]]: the list of recv key and its tensor rank
        """
        pass

    def name_to_model_part_index(self, dest_name: str) -> int:
        return 0

    @abstractmethod
    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        pass

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
        x,
        reg_key: Union[str, List[str]],
        *,
        allow_override: bool = False,
    ):
        if isinstance(reg_key, str):
            reg_key = [reg_key]

        def decorator(cls: Type) -> Type:
            for key in reg_key:
                if (
                    not allow_override
                    and key in WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY
                ):
                    raise ValueError(f"Class '{key}' is already registered.")
                WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY[key] = cls
            return cls

        return decorator

    @classmethod
    def get_weight_mapper(cls, model_type: str) -> Type["WeightMapper"]:
        if model_type not in WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY:
            raise ValueError(f"ModelType '{model_type}' is not supported now.")

        return WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY[model_type]
