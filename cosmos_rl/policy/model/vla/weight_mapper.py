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

import torch
from typing import List, Tuple, Dict, Any
from transformers import AutoConfig

from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism_registry import (
    ParallelismStrategyRole,
    register_parallelism_strategy,
    get_policy_parallelism_strategy as get_policy_strategy,
    get_rollout_parallelism_strategy as get_rollout_strategy,
)


class VLAWeightMapper(WeightMapper):
    """
    Weight mapper for VLA (Vision-Language-Action) models

    Handles weight mapping and tensor parallelism for VLA models including:
    - OpenVLA: Standard OpenVLA models
    - OpenVLA-OFT: OpenVLA with Orthogonal Fine-Tuning

    The mapper handles the complex architecture of VLA models which typically include:
    - Vision backbone (e.g., DINOv2, SigLIP)
    - Vision-language projector
    - Language model (e.g., Llama, Vicuna)
    - Action head for robotic control
    """

    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)

    def rollout_prepare_recv(
        self,
        model: Any,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[str, int]]]]:
        """
        Rollout prepare recv list for P2R weight sync:
            - weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[List[Tuple[str, int]]]: the list of grouped recv key and its tensor rank

        For VLA models, we need to handle the multimodal components properly.
        """

        weight_inplace_view_map = {}
        recv_key_n_rank_list = []

        for param_name, param_tensor in model.state_dict().items():
            if (
                isinstance(param_tensor, torch.distributed.tensor.DTensor)
                and param_tensor.to_local().numel() == 0
            ):
                continue
            weight_inplace_view_map[param_name] = param_tensor
            recv_key_n_rank_list.append([(param_name, param_tensor.dim())])

        logger.info(
            f"VLA rollout_prepare_recv: prepared {len(weight_inplace_view_map)} parameters "
            f"in {len(recv_key_n_rank_list)} groups"
        )

        return weight_inplace_view_map, recv_key_n_rank_list

    def get_unsplited_weight_name(self, weight_key: str) -> str:
        return weight_key

    def get_policy_parallelism_strategy(self):
        """
        Define parallelism strategies for VLA model components.

        All VLA components (vision_backbone, projector, language_model, action_head)
        are now sharded uniformly by FSDP, so they're all in parallelism_info_for_params.

        We use automatic inference for all parameters - no special handling needed!

        Returns:
            List containing the registered "openvla" strategy function
        """

        # Register VLA policy parallelism strategy
        @register_parallelism_strategy(
            "openvla", role=ParallelismStrategyRole.POLICY, allow_override=True
        )
        def vla_policy_strategy(shape, dest_name, parallelism, hf_config):
            """
            Use automatic inference for all VLA parameters.

            All components are FSDP-sharded uniformly, so automatic inference
            will correctly detect the sharding from parallelism_info_for_params.

            Args:
                shape: Tensor shape (tuple of ints)
                dest_name: Parameter name (str)
                parallelism: ParallelDims configuration
                hf_config: HuggingFace model config

            Returns:
                Tuple of (None, None, None) to trigger automatic inference
            """
            # All parameters: use automatic inference
            return None, None, None

        logger.info(
            "[VLAWeightMapper] Registered policy parallelism strategy for P2R weight sync"
        )
        return [get_policy_strategy("openvla")]

    def get_rollout_parallelism_strategy(self):
        """
        Define parallelism strategies for VLA rollout workers.

        Rollout workers receive weights from policy workers via P2R sync.
        Since all parameters are now uniformly sharded on policy side,
        we use automatic inference for all parameters.

        Returns:
            List of strategy functions for rollout recv instructions
        """

        # Register VLA rollout parallelism strategy
        @register_parallelism_strategy(
            "openvla", role=ParallelismStrategyRole.ROLLOUT, allow_override=True
        )
        def vla_rollout_strategy(shape, dest_name, parallelism, hf_config):
            """
            Use automatic inference for all VLA parameters.

            Args:
                shape: Tensor shape (tuple of ints)
                dest_name: Parameter name (str)
                parallelism: ParallelDims configuration
                hf_config: HuggingFace model config

            Returns:
                Tuple of (None, None, None) to trigger automatic inference
            """
            # All parameters: use automatic inference
            return None, None, None

        logger.info(
            "[VLAWeightMapper] Registered rollout parallelism strategy for P2R weight sync"
        )
        return [get_rollout_strategy("openvla")]

    def policy_decompose_param_1_to_n_for_sync(self, name):
        """
        Override to prevent parameter decomposition for weight sync.

        VLA models do NOT decompose parameters (like qkv -> q,k,v) for P2R weight sync.
        All parameters are synced as complete tensors.

        Without this override, the base WeightMapper might try to decompose parameters
        like 'vision_backbone.*.attn.qkv.weight', causing them to be skipped from
        weight_sync_transforms and never synced to rollout workers.

        Args:
            name: Parameter name

        Returns:
            Empty list [] means no decomposition
        """
        # VLA models: no parameter decomposition for weight sync
        return []
