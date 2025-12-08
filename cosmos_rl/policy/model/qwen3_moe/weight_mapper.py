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
from cosmos_rl.utils.parallelism_registry import get_rollout_parallelism_strategy
from cosmos_rl.utils import util
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
                    "experts.w13_weight", "experts.gate_and_up_proj.weight"
                )
            elif "experts.w2_weight" in rollout_weight_name:
                return rollout_weight_name.replace(
                    "experts.w2_weight", "experts.down_proj.weight"
                )
            # below are for trtllm weight for gate_and_up_proj and input_layernorm.
            elif "experts.w3_w1_weight" in rollout_weight_name:
                return rollout_weight_name.replace(
                    "experts.w3_w1_weight", "experts.gate_and_up_proj.weight"
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
        recv_key_n_rank_list = []
        vllm_weight_inplace_view_map = {}
        for param_name, param in vllm_model.named_parameters():
            group_keys = []
            param_name_hf = self._rollout_vllm_name_to_hf(param_name)
            # logger.info(f"[Rollout] param_name_hf: {param_name_hf}")
            if "qkv_proj" in param_name_hf:
                # only for language model
                # must be inplace slicing.
                # split qkv weight
                q_weight, k_weight, v_weight = self._rollout_split_qkv_weight(
                    param_name_hf, param
                )
                q_proj_weight_key = param_name_hf.replace("qkv_proj", "q_proj")
                k_proj_weight_key = param_name_hf.replace("qkv_proj", "k_proj")
                v_proj_weight_key = param_name_hf.replace("qkv_proj", "v_proj")
                vllm_weight_inplace_view_map[q_proj_weight_key] = q_weight
                group_keys.append((q_proj_weight_key, q_weight.ndim))
                vllm_weight_inplace_view_map[k_proj_weight_key] = k_weight
                group_keys.append((k_proj_weight_key, k_weight.ndim))
                vllm_weight_inplace_view_map[v_proj_weight_key] = v_weight
                group_keys.append((v_proj_weight_key, v_weight.ndim))
            else:
                vllm_weight_inplace_view_map[param_name_hf] = param
                group_keys.append((param_name_hf, param.ndim))
            recv_key_n_rank_list.append(group_keys)
        return vllm_weight_inplace_view_map, recv_key_n_rank_list

    @torch.no_grad()
    def policy_maybe_decompose_weights_to_hf_naming(
        self, name, expert_weight: torch.Tensor
    ):
        # name is HF naming convention
        def yield_weight(n_experts, expert_weight, w_name, layer_id):
            for expert_id in range(n_experts):
                single_expert_weight = expert_weight[expert_id].contiguous()
                yield (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.{w_name}.weight",
                    single_expert_weight,
                )

        if match := re.search(
            r"model\.layers\.(\d+)\.mlp\.experts\.(gate_and_up_proj|down_proj)\.(weight)",
            name,
        ):
            layer_id = int(match.group(1))
            w_name = match.group(2)
            n_experts = expert_weight.shape[0]
            if w_name == "gate_and_up_proj":
                # for qwen3 moe, gate_and_up_proj is split into gate_proj and up_proj and stored.
                # shape: [experts, 2 * ffn_dim, hidden_dim]
                part = expert_weight.shape[1] // 2
                gate_proj_weight = expert_weight[:, :part, :]
                up_proj_weight = expert_weight[:, part:, :]
                yield from yield_weight(
                    n_experts, gate_proj_weight, "gate_proj", layer_id
                )
                yield from yield_weight(n_experts, up_proj_weight, "up_proj", layer_id)
            else:
                yield from yield_weight(n_experts, expert_weight, w_name, layer_id)
        else:
            yield name, expert_weight

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = util.clear_weight_name(name)
        if not name == "lm_head.weight":
            if not name.startswith("model."):
                name = "model." + name
        if re.search(
            r"model\.layers\.(\d+)\.mlp\.experts\.(gate_and_up_projs|down_projs)", name
        ):
            name = name.replace("projs", "proj.weight")
        return name

    def get_rollout_parallelism_strategy(self):
        return [get_rollout_parallelism_strategy("qwen3_moe")]

    def get_unsplited_weight_name(self, weight_key: str) -> str:
        for key in ["q_proj", "k_proj", "v_proj"]:
            if key in weight_key:
                return weight_key.replace(key, "qkv_proj")
        for key in ["gate_proj", "up_proj"]:
            if key in weight_key:
                return weight_key.replace(key, "gate_and_up_proj")
        return weight_key  # return full weight key
