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
from typing import List, Tuple
from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils import util
from transformers import AutoConfig


class GPTWeightMapper(WeightMapper):
    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        self.kv_head_ratio = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        # Happen to be the same as policy name mapping.
        return self.policy_map_local_key_to_hf_key(rollout_weight_name)

    def _rollout_split_qkv_weight(self, name, weight: torch.Tensor):
        # weight has shape [q_num_heads * head_dim + k_num_heads * head_dim + v_num_heads * head_dim, hidden_dim]
        shares = self.kv_head_ratio + 2
        dim_0 = weight.shape[0]  # for both weight and bias
        unit_dim = dim_0 // shares

        q_weight = weight[: unit_dim * self.kv_head_ratio]
        k_weight = weight[
            unit_dim * self.kv_head_ratio : unit_dim * (self.kv_head_ratio + 1)
        ]
        v_weight = weight[unit_dim * (self.kv_head_ratio + 1) :]
        return q_weight, k_weight, v_weight

    def _split_gate_proj_weight(self, name, weight: torch.Tensor):
        # weight has shape [2 * x, hidden_dim]
        dim_0 = weight.shape[0]
        gate_proj_weight = weight[: dim_0 // 2]
        up_proj_weight = weight[dim_0 // 2 :]
        return gate_proj_weight, up_proj_weight

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        group_keys = []
        compatible_key = self.rollout_map_local_key_to_hf_key(param_name)
        if "qkv_proj" in compatible_key:
            # must be inplace slicing.
            # split qkv weight
            q_weight, k_weight, v_weight = self._rollout_split_qkv_weight(
                compatible_key, param
            )
            q_proj_weight_key = compatible_key.replace("qkv_proj", "q_proj")
            k_proj_weight_key = compatible_key.replace("qkv_proj", "k_proj")
            v_proj_weight_key = compatible_key.replace("qkv_proj", "v_proj")
            group_keys.append((q_proj_weight_key, q_weight))
            group_keys.append((k_proj_weight_key, k_weight))
            group_keys.append((v_proj_weight_key, v_weight))
        elif "gate_up_proj" in compatible_key:
            # split gate and up proj
            gate_proj_weight, up_proj_weight = self._split_gate_proj_weight(
                compatible_key, param
            )
            gate_proj_weight_key = compatible_key.replace("gate_up_proj", "gate_proj")
            group_keys.append((gate_proj_weight_key, gate_proj_weight))
            up_proj_weight_key = compatible_key.replace("gate_up_proj", "up_proj")
            group_keys.append((up_proj_weight_key, up_proj_weight))
        else:
            group_keys.append((compatible_key, param))
        return group_keys

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = util.clear_weight_name(name)
        if not name == "lm_head.weight":
            if not name.startswith("model."):
                name = "model." + name
        return name
