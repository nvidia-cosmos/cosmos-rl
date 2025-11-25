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
from cosmos_rl.utils import util
from transformers import AutoConfig
from typing import Dict, List, Tuple
from functools import cached_property
from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils.parallelism_registry import (
    get_rollout_parallelism_strategy,
)


class Qwen3VLMoeWeightMapper(WeightMapper):
    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)

        self.kv_head_ratio = (
            self.config.text_config.num_attention_heads
            // self.config.text_config.num_key_value_heads
        )
        self.head_dim = (
            self.config.text_config.hidden_size
            // self.config.text_config.num_attention_heads
        )

    def _rollout_vllm_name_to_hf(self, rollout_weight_name: str) -> str:
        converted_name = None

        if rollout_weight_name.startswith("language_model.model."):
            converted_name = rollout_weight_name.replace(
                "language_model.model.", "model.language_model."
            )

        if rollout_weight_name.startswith("visual."):
            converted_name = rollout_weight_name.replace("visual.", "model.visual.")

        if rollout_weight_name == "language_model.lm_head.weight":
            converted_name = "lm_head.weight"

        if "experts.w13_weight" in converted_name:
            converted_name = converted_name.replace(
                "experts.w13_weight", "experts.gate_up_proj"
            )
        elif "experts.w2_weight" in converted_name:
            converted_name = converted_name.replace(
                "experts.w2_weight", "experts.down_proj"
            )

        assert (
            converted_name is not None
        ), f"{rollout_weight_name} is not mapped successfully."
        return converted_name

    def __rollout_split_qkv_weight(self, name, weight: torch.Tensor):
        # visual
        if "visual" in name:
            # split qkv weight for visual
            # weight has shape [3 * head_dim, hidden_dim]
            # kv head ratio is 1, so we can split it into q, k, v
            assert (
                weight.shape[0] % 3 == 0
            ), "Weight shape is not compatible for splitting."
            unit_dim = weight.shape[0] // 3  # for both weight and bias
            q_weight = weight[:unit_dim]
            k_weight = weight[unit_dim : unit_dim * 2]
            v_weight = weight[unit_dim * 2 :]
            return q_weight, k_weight, v_weight
        # language
        # split qkv weight
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
        # gate_proj and up_proj in vllm is already split.
        # weight has shape [num_experts, 2 * x, hidden_dim]
        split_size = weight.shape[1] // 2
        gate_proj_weight = weight[:, :split_size, :]
        up_proj_weight = weight[:, split_size:, :]
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
            compatible_key = self._rollout_vllm_name_to_hf(param_name)

            if "qkv_proj" in compatible_key:
                q_weight, k_weight, v_weight = self.__rollout_split_qkv_weight(
                    compatible_key, param
                )
                q_proj_weight_key = compatible_key.replace("qkv_proj", "q_proj")
                k_proj_weight_key = compatible_key.replace("qkv_proj", "k_proj")
                v_proj_weight_key = compatible_key.replace("qkv_proj", "v_proj")
                vllm_weight_inplace_view_map[q_proj_weight_key] = q_weight
                group_keys.append((q_proj_weight_key, q_weight.ndim))
                vllm_weight_inplace_view_map[k_proj_weight_key] = k_weight
                group_keys.append((k_proj_weight_key, k_weight.ndim))
                vllm_weight_inplace_view_map[v_proj_weight_key] = v_weight
                group_keys.append((v_proj_weight_key, v_weight.ndim))
            # elif "gate_up_proj" in compatible_key:
            #     # split gate and up proj
            #     gate_proj_weight, up_proj_weight = self._split_gate_proj_weight(
            #         compatible_key, param
            #     )
            #     gate_proj_weight_key = compatible_key.replace(
            #         "gate_up_proj", "gate_proj"
            #     )
            #     vllm_weight_inplace_view_map[gate_proj_weight_key] = gate_proj_weight
            #     group_keys.append((gate_proj_weight_key, gate_proj_weight.ndim))

            #     up_proj_weight_key = compatible_key.replace("gate_up_proj", "up_proj")
            #     vllm_weight_inplace_view_map[up_proj_weight_key] = up_proj_weight
            #     group_keys.append((up_proj_weight_key, up_proj_weight.ndim))
            elif "qkv" in compatible_key and "visual" in compatible_key:
                q_weight, k_weight, v_weight = self.__rollout_split_qkv_weight(
                    compatible_key, param
                )
                q_visual_proj_weight_key = compatible_key.replace("qkv", "q")
                k_visual_proj_weight_key = compatible_key.replace("qkv", "k")
                v_visual_proj_weight_key = compatible_key.replace("qkv", "v")
                vllm_weight_inplace_view_map[q_visual_proj_weight_key] = q_weight
                group_keys.append((q_visual_proj_weight_key, q_weight.ndim))
                vllm_weight_inplace_view_map[k_visual_proj_weight_key] = k_weight
                group_keys.append((k_visual_proj_weight_key, k_weight.ndim))
                vllm_weight_inplace_view_map[v_visual_proj_weight_key] = v_weight
                group_keys.append((v_visual_proj_weight_key, v_weight.ndim))
            else:
                vllm_weight_inplace_view_map[compatible_key] = param
                group_keys.append((compatible_key, param.ndim))
            recv_key_n_rank_list.append(group_keys)
        return vllm_weight_inplace_view_map, recv_key_n_rank_list

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = util.clear_weight_name(name)
        if name.startswith("model.") and "visual" not in name:
            name = name.replace("model.", "model.language_model.")
        if name.startswith("visual."):
            name = name.replace("visual.", "model.visual.")

        if "lm_head.weight" in name:
            name = "lm_head.weight"

        if re.search(
            r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.gate_and_up_projs",
            name,
        ):
            name = name.replace("gate_and_up_projs", "gate_up_proj")
        elif re.search(
            r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.down_projs",
            name,
        ):
            name = name.replace("down_projs", "down_proj")

        return name

    def name_to_model_part_index(self, dest_name: str) -> int:
        if dest_name in ["lm_head.weight", "lm_head.bias"]:
            return 0
        elif dest_name.startswith("model.visual."):
            return 1
        elif dest_name.startswith("model.language_model."):
            return 0
        else:
            raise ValueError(f"Unsupported weight: {dest_name}")

    def policy_decompose_param_1_to_n_for_sync(self, name):
        if match := re.search(  # noqa: F841
            r"visual\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)",
            name,
        ):
            split_strategy = []
            # The first part of the split:
            # the dictionary means at dimension 0, extract the part of offset 0 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv", "q"),
                    {0: {"offset": 0, "total_size": 3, "length": 1}},
                )
            )
            # The second part of the split:
            # the dictionary means at dimension 0, extract the part of offset 1 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv", "k"),
                    {0: {"offset": 1, "total_size": 3, "length": 1}},
                )
            )
            # The third part of the split:
            # the dictionary means at dimension 0, extract the part of offset 2 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv", "v"),
                    {0: {"offset": 2, "total_size": 3, "length": 1}},
                )
            )
            return split_strategy
        return []

    @cached_property
    def packed_modules_mapping(self):
        mapping_dict = {
            "qkv": [
                "q",
                "k",
                "v",
            ],
            "gate_up_proj": [
                "gate_proj",
                "up_proj",
            ],
            "qkv_proj": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
        }
        return mapping_dict

    def get_rollout_parallelism_strategy(self):
        return [get_rollout_parallelism_strategy("qwen3_vl_moe")]

    def get_unsplited_weight_name(self, weight_key: str) -> str:
        for key in ["q_proj", "k_proj", "v_proj"]:
            if key in weight_key:
                return weight_key.replace(key, "qkv_proj")
        for key in ["gate_proj", "up_proj"]:
            if key in weight_key:
                return weight_key.replace(key, "gate_and_up_proj")
        for key in ["q", "k", "v"]:
            if "visual" in weight_key and key in weight_key:
                return weight_key.replace(key, "qkv")
        return weight_key  # return full weight key
