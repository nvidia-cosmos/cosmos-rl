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

from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoConfig

from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils import util
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism_registry import (
    get_rollout_parallelism_strategy,
)


class DeepseekV3MoEWeightMapper(WeightMapper):
    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)

    def _rollout_vllm_name_to_hf(self, rollout_weight_name: str) -> str:
        # TODO(aazzolini): 2.2. Implement name_to_hf correctly.
        # Why is .experts skipped here for HF weights?
        if not rollout_weight_name == "lm_head.weight":
            if "experts.w13_weight" in rollout_weight_name:
                return rollout_weight_name.replace(
                    "experts.w13_weight", "experts.gate_up_proj.weight"
                )
            elif "experts.w2_weight" in rollout_weight_name:
                return rollout_weight_name.replace(
                    "experts.w2_weight", "experts.down_proj.weight"
                )
        return rollout_weight_name

    def _split_qkv_a_proj_weight(self, weight: torch.Tensor):
        # Weight has shape [q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden_dim]
        # Split it into [q_lora_rank, hidden_dim] and [kv_lora_rank + qk_rope_head_dim, hidden_dim]

        q_lora_rank = self.config.q_lora_rank
        kv_lora_rank = self.config.kv_lora_rank
        qk_rope_head_dim = self.config.qk_rope_head_dim
        assert weight.shape[0] == q_lora_rank + kv_lora_rank + qk_rope_head_dim, (
            f"weight.shape[0] {weight.shape[0]} != q_lora_rank + kv_lora_rank + qk_rope_head_dim {q_lora_rank + kv_lora_rank + qk_rope_head_dim}"
        )
        
        q_a_proj = weight[q_lora_rank :]
        kv_a_proj_with_mqa = weight[: q_lora_rank]
        return q_a_proj, kv_a_proj_with_mqa

    def _split_gate_up_proj_weight(self, weight: torch.Tensor):
        # Weight has shape [n_experts, 2 * inter_dim, hidden_dim]
        # Split it into [n_experts, :inter_dim, hidden_dim] and [n_experts, inter_dim:, hidden_dim]

        # Note that for MoE we have an extra dimension on the left for for dense we dont.
        split_idx = weight.shape[-2] // 2
        gate_proj_weight = weight[..., :split_idx, :]
        up_proj_weight = weight[..., split_idx:, :]
        return gate_proj_weight, up_proj_weight

    def rollout_prepare_recv(
        self, vllm_model: Any
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, torch.Size]]]:
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
        vllm_weight_inplace_view_map = {}
        for param_name, param in vllm_model.named_parameters():
            group_keys = []
            compatible_key = self._rollout_vllm_name_to_hf(param_name)
            logger.info(f"[Rollout] param vllm_name {param_name} hf_name: {compatible_key}")

            if "fused_qkv_a_proj" in compatible_key:
                # Split q_a and kv_a_proj weights.

                # self_attn.fused_qkv_a_proj.weight
                # 1) self_attn.q_a_proj.weight
                # 2) self_attn.kv_a_proj_with_mqa.weight
                q_a_proj, kv_a_proj_with_mqa = self._split_qkv_a_proj_weight(param)

                q_a_proj_key = compatible_key.replace("fused_qkv_a_proj", "q_a_proj")
                vllm_weight_inplace_view_map[q_a_proj_key] = q_a_proj
                group_keys.append((q_a_proj_key, q_a_proj.ndim))

                kv_a_proj_with_mqa_key = compatible_key.replace(
                    "fused_qkv_a_proj", "kv_a_proj_with_mqa"
                )
                vllm_weight_inplace_view_map[kv_a_proj_with_mqa_key] = kv_a_proj_with_mqa
                group_keys.append((kv_a_proj_with_mqa_key, kv_a_proj_with_mqa.ndim))

            if "gate_up_proj" in compatible_key:
                # Split gate and up proj weights.
                gate_proj_weight, up_proj_weight = self._split_gate_proj_weight(param)

                gate_proj_weight_key = compatible_key.replace("gate_up_proj", "gate_proj")
                vllm_weight_inplace_view_map[gate_proj_weight_key] = gate_proj_weight
                group_keys.append((gate_proj_weight_key, gate_proj_weight.ndim))

                up_proj_weight_key = compatible_key.replace("gate_up_proj", "up_proj")
                vllm_weight_inplace_view_map[up_proj_weight_key] = up_proj_weight
                group_keys.append((up_proj_weight_key, up_proj_weight.ndim))

            else:
                vllm_weight_inplace_view_map[compatible_key] = param
                group_keys.append((compatible_key, param.ndim))

            recv_key_n_rank_list.append(group_keys)

        return vllm_weight_inplace_view_map, recv_key_n_rank_list

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = util.clear_weight_name(name)

        name = name.replace("model.", "")
        name = name.replace("mlp.experts.down_projs", "mlp.experts.down_proj.weight")
        name = name.replace("mlp.experts.up_projs", "mlp.experts.up_proj.weight")
        name = name.replace("mlp.experts.gate_projs", "mlp.experts.gate_proj.weight")

        return name

    def get_rollout_parallelism_strategy(self):
        return [get_rollout_parallelism_strategy("deepseek_v3")]
