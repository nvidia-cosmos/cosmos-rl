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

import re
import torch
from typing import List, Tuple, Dict

from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils.parallelism_registry import (
    get_policy_parallelism_strategy,
    get_rollout_parallelism_strategy,
)
from cosmos_rl.utils import util
from cosmos_rl.utils.logging import logger
from transformers import AutoConfig


class Qwen3MoeWeightMapper(WeightMapper):
    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        self.kv_head_ratio = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

    def _rollout_vllm_name_to_hf(self, rollout_weight_name: str) -> str:
        if not rollout_weight_name == "lm_head.weight":
            if "experts.w13_weight" in rollout_weight_name:
                return rollout_weight_name.replace(
                    "experts.w13_weight", "experts.gate_up_proj.weight"
                )
            elif "experts.w2_weight" in rollout_weight_name:
                return rollout_weight_name.replace(
                    "experts.w2_weight", "experts.down_proj.weight"
                )
            # Below are for trtllm weight for gate_up_proj and input_layernorm.
            elif "experts.w3_w1_weight" in rollout_weight_name:
                return rollout_weight_name.replace(
                    "experts.w3_w1_weight", "experts.gate_up_proj.weight"
                )
            elif "next_layer_layernorm" in rollout_weight_name:
                # For trtllm, next_layer_layernorm is:
                #   `model.norm` when layer_id == self.config.num_hidden_layers - 1
                #   `model.layers.${layer_id + 1}.input_layernorm` when layer_id < self.config.num_hidden_layers - 1
                layer_id = int(rollout_weight_name.split(".")[2])
                if layer_id == self.config.num_hidden_layers - 1:
                    return "model.norm.weight"
                else:
                    return f"model.layers.{layer_id + 1}.input_layernorm.weight"
        return rollout_weight_name

    def _rollout_split_qkv_weight(self, weight: torch.Tensor):
        # Note that this follows GQA (Qwen3) design where:
        # 1) k_num_heads = v_num_heads.
        # 2) q_num_heads = kv_head_ratio * k_num_heads
        # Weight has shape [(q_num_heads + k_num_heads + v_num_heads) * head_dim, hidden_dim]
        shares = self.kv_head_ratio + 2
        dim_0 = weight.shape[0]  # for both weight and bias
        unit_dim = dim_0 // shares  # unit_dim = k_num_heads * head_dim

        q_weight = weight[
            : unit_dim * self.kv_head_ratio
        ]
        k_weight = weight[
            unit_dim * self.kv_head_ratio : unit_dim * (self.kv_head_ratio + 1)
        ]
        v_weight = weight[
            unit_dim * (self.kv_head_ratio + 1) :
        ]
        return q_weight, k_weight, v_weight

    def _rollout_split_gate_up_proj_weight(self, weight: torch.Tensor):
        # weight has shape [num_experts, 2 * x, hidden_dim], first gate_proj, then up_proj
        # if backend is trtllm,  [num_experts, 2 * x, hidden_dim], first up_proj, then gate_proj
        dim_1 = weight.shape[1]
        gate_proj_weight = weight[:, : dim_1 // 2]
        up_proj_weight = weight[:, dim_1 // 2 :]
        if self.backend == "trtllm":
            gate_proj_weight, up_proj_weight = up_proj_weight, gate_proj_weight
        return gate_proj_weight, up_proj_weight

    def rollout_prepare_recv(
        self,
        vllm_model,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        List[List[Tuple[str, torch.Size]]],
    ]:
        """
        Given the rollout (e.g. VLLM) nn.Module, return the list of HuggingFace-compatible
        parameter names along with their tensor views and shapes. Param names produced this
        method should exactly match policy_model.sorted_params.

        Args:
           vllm_model: rollout nn.Module

        Returns:
          Tuple[
            Dict[
              str,           # compatible_key
              Tensor         # param Tensor
            ],               # compatible weight map
            List[
              List[
                Tuple[
                  str,         # compatible_key
                  int,         # param tensor ndim
                ]
              ]              # param group (?)
            ]                # compatible weight list

            Where "compatible_key" is the HuggingFace param name
        """
        recv_key_n_rank_list = []
        compatible_key_map = {}
        for param_name, param in vllm_model.named_parameters():
            group_keys = []
            compatible_key = self._rollout_vllm_name_to_hf(param_name)
            logger.info(f"[Rollout] param vllm_name {param_name} hf_name: {compatible_key}")

            if "qkv_proj" in compatible_key:
                # must be inplace slicing.
                # split qkv weight
                q_weight, k_weight, v_weight = self._rollout_split_qkv_weight(param)

                q_proj_weight_key = compatible_key.replace("qkv_proj", "q_proj")
                compatible_key_map[q_proj_weight_key] = q_weight
                group_keys.append((q_proj_weight_key, q_weight.ndim))

                k_proj_weight_key = compatible_key.replace("qkv_proj", "k_proj")
                compatible_key_map[k_proj_weight_key] = k_weight
                group_keys.append((k_proj_weight_key, k_weight.ndim))

                v_proj_weight_key = compatible_key.replace("qkv_proj", "v_proj")
                compatible_key_map[v_proj_weight_key] = v_weight
                group_keys.append((v_proj_weight_key, v_weight.ndim))

            elif "gate_up_proj" in compatible_key:
                # split gate and up proj
                gate_proj_weight, up_proj_weight = self._rollout_split_gate_up_proj_weight(param)

                gate_proj_weight_key = compatible_key.replace("gate_up_proj", "gate_proj")
                compatible_key_map[gate_proj_weight_key] = gate_proj_weight
                group_keys.append((gate_proj_weight_key, gate_proj_weight.ndim))

                up_proj_weight_key = compatible_key.replace("gate_up_proj", "up_proj")
                compatible_key_map[up_proj_weight_key] = up_proj_weight
                group_keys.append((up_proj_weight_key, up_proj_weight.ndim))

            else:
                compatible_key_map[compatible_key] = param
                group_keys.append((compatible_key, param.ndim))
            recv_key_n_rank_list.append(group_keys)

        return compatible_key_map, recv_key_n_rank_list

    @torch.no_grad()
    def policy_maybe_decompose_weights_to_hf_naming(
        self, name, expert_weight: torch.Tensor
    ):
        if match := re.search(
            r"model\.layers\.(\d+)\.mlp\.experts\.(up_proj|gate_proj|down_proj)\.weight", name
        ):
            layer_id = int(match.group(1))
            w_name = match.group(2)
            n_experts = expert_weight.shape[0]
            for expert_id in range(n_experts):
                single_expert_weight = expert_weight[expert_id].contiguous()
                yield (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.{w_name}.weight",
                    single_expert_weight,
                )
        else:
            yield name, expert_weight

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = util.clear_weight_name(name)
        if not name == "lm_head.weight":
            # The policy weights do not have the "model." prefix, but the hf weights do.
            if not name.startswith("model."):
                name = "model." + name

            if re.search(
                r"model\.layers\.(\d+)\.mlp\.experts\.(up_projs|gate_projs|down_projs)", name
            ):
                name = name.replace("projs", "proj.weight")
        return name

    def get_policy_parallelism_strategy(self):
        return [get_policy_parallelism_strategy("qwen3_moe")]

    def get_rollout_parallelism_strategy(self):
        return [get_rollout_parallelism_strategy("qwen3_moe")]

    def get_unsplited_weight_name(self, weight_key: str) -> str:
        for key in ["q_proj", "k_proj", "v_proj"]:
            if key in weight_key:
                return weight_key.replace(key, "qkv_proj")
        for key in ["gate_proj", "up_proj"]:
            if key in weight_key:
                return weight_key.replace(key, "gate_up_proj")
        return weight_key  # return full weight key
