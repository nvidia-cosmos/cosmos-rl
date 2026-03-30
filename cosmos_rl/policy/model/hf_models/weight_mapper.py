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
import re
from typing import List, Tuple
from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils import util
from transformers import AutoConfig
from functools import cached_property


class HFModelWeightMapper(WeightMapper):
    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        self.kv_head_ratio = 1
        self.head_dim = 1
        self.attn_output_gate = False
        self.text_config = None

        if getattr(self.config, "num_key_value_heads", None) is not None:
            self.kv_head_ratio = (
                self.config.num_attention_heads // self.config.num_key_value_heads
            )
            self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        elif getattr(self.config, "text_config", None) is not None:
            # VLM models like Gemma3-12b-it has num_attention_heads in text_config
            self.text_config = self.config.text_config
            self.kv_head_ratio = (
                self.text_config.num_attention_heads
                // self.text_config.num_key_value_heads
            )
            self.head_dim = (
                self.text_config.hidden_size // self.text_config.num_attention_heads
            )
            # Qwen3-5 enabled attn_output_gate which fused q+gate in q_proj
            self.attn_output_gate = getattr(self.text_config, "attn_output_gate", False)
        elif getattr(self.config, "llm_config", None) is not None:
            # VLM models like InternVL could has num_attention_heads in llm_config
            self.text_config = self.config.llm_config
            self.kv_head_ratio = (
                self.text_config.num_attention_heads
                // self.text_config.num_key_value_heads
            )
            self.head_dim = (
                self.text_config.hidden_size // self.text_config.num_attention_heads
            )
        else:
            raise ValueError(
                f"Can not determine kv_head_ratio and head_dim from config: {self.config}"
            )
        self.is_vlm = getattr(self.config, "vision_config", None) is not None
        self.reverse_hf_conversion_mapping = None

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        # Happen to be the same as policy name mapping.
        model_type = self.config.model_type
        if model_type == "gpt_oss":
            # Some special cases for GPT-OSS.
            gpt_oss_rename_mapping = {
                # Please do not change the order of the keys.
                "attn": "self_attn",
                "embedding": "embed_tokens",
            }
            for key, value in gpt_oss_rename_mapping.items():
                if key in rollout_weight_name:
                    return rollout_weight_name.replace(key, value)
            # gate_up_proj
            if "w13_weight" in rollout_weight_name:
                return rollout_weight_name.replace("w13_weight", "gate_up_proj")
            elif "w2_weight" in rollout_weight_name:
                return rollout_weight_name.replace("w2_weight", "down_proj")
            elif "w13_bias" in rollout_weight_name:
                return rollout_weight_name.replace("w13_bias", "gate_up_proj_bias")
            elif "w2_bias" in rollout_weight_name:
                return rollout_weight_name.replace("w2_bias", "down_proj_bias")
            else:
                pass
        elif model_type in ["qwen3_vl", "qwen3_5", "qwen3_5_moe"]:
            # Special case for Qwen3-VL
            if rollout_weight_name.startswith("language_model.model."):
                rollout_weight_name = rollout_weight_name.replace(
                    "language_model.model.", "language_model."
                )
            if (
                not rollout_weight_name.startswith("model.")
                and "lm_head" not in rollout_weight_name
            ):
                rollout_weight_name = "model." + rollout_weight_name
            if rollout_weight_name.startswith("language_model.lm_head."):
                # lm_head exists at language_model level, remove the language_model prefix
                rollout_weight_name = rollout_weight_name.replace(
                    "language_model.lm_head.", "lm_head."
                )

            # gate_up_proj
            if "w13_weight" in rollout_weight_name:
                return rollout_weight_name.replace("w13_weight", "gate_up_proj")
            elif "w2_weight" in rollout_weight_name:
                return rollout_weight_name.replace("w2_weight", "down_proj")

            return rollout_weight_name

        return self.policy_map_local_key_to_hf_key(rollout_weight_name)

    def _vllm_qkv_column_parallel_shard_lens(
        self, dim_0: int, total_q_heads: int
    ) -> Tuple[int, int, int]:
        """Row splits for one TP rank of vLLM ``QKVParallelLinear`` fused ``[Q(+gate) | K | V]``.

        ``total_q_heads`` is the logical Q row count in **head** units: ``Nq`` without
        ``attn_output_gate``, or ``Nq * (1 + gate)`` when the gate doubles the Q block.

        When ``tp_size >= num_key_value_heads``, K/V are replicated (``num_kv_heads==1`` per
        rank); the uniform ``kv_head_ratio:1:1`` split does **not** match local row sizes.
        We infer ``tp`` from ``dim_0`` and ``head_dim`` (same layout as vLLM).

        ``tc`` is ``text_config`` / ``llm_config`` when present, otherwise the root HF
        ``config`` (models with ``num_key_value_heads`` on the top-level config).
        """
        tc = self.text_config if self.text_config is not None else self.config
        head_dim = getattr(tc, "head_dim", None) or (
            tc.hidden_size // tc.num_attention_heads
        )
        n_kv = max(1, tc.num_key_value_heads)
        if head_dim <= 0 or dim_0 % head_dim != 0:
            raise ValueError(
                f"vLLM qkv split: dim_0={dim_0} not divisible by head_dim={head_dim}"
            )
        units = dim_0 // head_dim
        for tp in range(1, total_q_heads + 1):
            if total_q_heads % tp != 0:
                continue
            num_heads_local = total_q_heads // tp
            if tp >= n_kv:
                num_kv_local = 1
            else:
                if n_kv % tp != 0:
                    continue
                num_kv_local = n_kv // tp
            if num_heads_local + 2 * num_kv_local == units:
                q_len = num_heads_local * head_dim
                k_len = num_kv_local * head_dim
                v_len = num_kv_local * head_dim
                assert q_len + k_len + v_len == dim_0
                return q_len, k_len, v_len
        raise ValueError(
            "vLLM qkv: cannot infer TP shard layout from "
            f"dim_0={dim_0}, head_dim={head_dim}, total_q_heads={total_q_heads}, n_kv={n_kv}"
        )

    def _rollout_split_qkv_weight(self, name, weight: torch.Tensor):
        if "visual" in name or "vision_tower" in name:
            # split qkv weight for visual
            # weight has shape [3 * head_dim, hidden_dim]
            # kv head ratio is 1, so we can split it into q, k, v
            assert weight.shape[0] % 3 == 0, (
                "Weight shape is not compatible for splitting."
            )
            unit_dim = weight.shape[0] // 3  # for both weight and bias
            q_weight = weight[:unit_dim]
            k_weight = weight[unit_dim : unit_dim * 2]
            v_weight = weight[unit_dim * 2 :]
            return q_weight, k_weight, v_weight

        # vLLM QKVParallelLinear: local [Q(+gate) | K | V]. Same layout for nested
        # ``text_config`` / ``llm_config`` and top-level GQA configs (Llama, etc.).
        tc = self.text_config if self.text_config is not None else self.config
        dim_0 = weight.shape[0]
        total_q = tc.num_attention_heads * (1 + int(self.attn_output_gate))
        q_len, k_len, _ = self._vllm_qkv_column_parallel_shard_lens(dim_0, total_q)
        q_weight = weight[:q_len]
        k_weight = weight[q_len : q_len + k_len]
        v_weight = weight[q_len + k_len :]
        return q_weight, k_weight, v_weight

    def _split_gate_proj_weight(self, name, weight: torch.Tensor, is_moe: bool = False):
        if is_moe:
            # weight has shape [num_experts, 2 * intermediate_size, hidden_dim]
            intermediate_size = weight.shape[1] // 2
            gate_proj_weight = weight[:, :intermediate_size]
            up_proj_weight = weight[:, intermediate_size:]
            return gate_proj_weight, up_proj_weight
        else:
            # weight has shape [2 * x, hidden_dim]
            dim_0 = weight.shape[0]
            gate_proj_weight = weight[: dim_0 // 2]
            up_proj_weight = weight[dim_0 // 2 :]
            return gate_proj_weight, up_proj_weight

    # For Qwen3-5 and Qwen3-5-MoE, we need to split the in_proj_qkvz weight into in_proj_q/k/v and in_proj_z
    # self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim)
    # self.in_proj_z   = nn.Linear(self.hidden_size, self.value_dim)
    def _split_in_proj_qkvz_weight(self, name, weight: torch.Tensor):
        # weight has shape [2 * head_dim, hidden_dim]
        num_v_heads = self.config.text_config.linear_num_value_heads
        num_k_heads = self.config.text_config.linear_num_key_heads
        head_v_dim = self.config.text_config.linear_value_head_dim
        value_dim = head_v_dim * num_v_heads
        head_k_dim = self.config.text_config.linear_key_head_dim
        key_dim = head_k_dim * num_k_heads

        total_size = 2 * key_dim + 2 * value_dim
        output_sizes_ratio = [
            key_dim / total_size,
            key_dim / total_size,
            value_dim / total_size,
            value_dim / total_size,
        ]
        output_dim = weight.shape[0]
        split_outputs = []
        for i in range(4):
            output_size = int(output_dim * output_sizes_ratio[i])
            if output_size > 0:
                split_outputs.append(weight[:output_size])
                weight = weight[output_size:]
        assert len(split_outputs) == 4, "_split_in_proj_qkvz_weight outputs should be 4"
        return split_outputs

    # For Qwen3-5 and Qwen3-5-MoE, we need to split the in_proj_ba weight into in_proj_b and in_proj_a
    def _split_in_proj_ba_weight(self, name, weight: torch.Tensor):
        # weight has shape [2 * head_dim, hidden_dim]
        dim_0 = weight.shape[0]
        in_proj_b_weight = weight[: dim_0 // 2]
        in_proj_a_weight = weight[dim_0 // 2 :]
        return in_proj_b_weight, in_proj_a_weight

    def _linear_attn_key_value_dims(self) -> Tuple[int, int]:
        """Linear attention Q/K/V channel dims (same as conv_dim = 2 * key_dim + value_dim)."""
        tc = self.config.text_config
        num_v_heads = tc.linear_num_value_heads
        num_k_heads = tc.linear_num_key_heads
        head_v_dim = tc.linear_value_head_dim
        head_k_dim = tc.linear_key_head_dim
        value_dim = head_v_dim * num_v_heads
        key_dim = head_k_dim * num_k_heads
        return key_dim, value_dim

    def _split_linear_attn_conv1d_weight(self, weight: torch.Tensor):
        """Split fused conv1d along dim 0 into Q, K, V blocks.

        Matches vLLM `mamba_v2_sharded_weight_loader` layout: checkpoint order is
        Q | K | V; each block is column-sharded by TP independently so that
        policy/rollout sync slices align with `in_proj_qkv`.
        """
        key_dim, value_dim = self._linear_attn_key_value_dims()
        conv_dim = 2 * key_dim + value_dim
        dim0 = weight.shape[0]
        if conv_dim % dim0 != 0:
            raise ValueError(
                f"linear_attn conv1d dim0 {dim0} is not a divisor of conv_dim {conv_dim}"
            )
        tp_equiv = conv_dim // dim0
        q_len = key_dim // tp_equiv
        k_len = key_dim // tp_equiv
        v_len = value_dim // tp_equiv
        if q_len + k_len + v_len != dim0:
            raise ValueError(
                f"linear_attn conv1d split lengths {q_len},{k_len},{v_len} do not sum to {dim0}"
            )
        q_w = weight[:q_len]
        k_w = weight[q_len : q_len + k_len]
        v_w = weight[q_len + k_len :]
        return q_w, k_w, v_w

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        models_do_not_split_gate_up_proj = ["gpt_oss"]
        # For some models like Qwen3-VL, vLLM just soft link lm_head with embed_tokens
        # Like: `self.lm_head = self.model.embed_tokens`
        # If we don't set `remove_duplicate` to False, the `lm_head` will not be included in the named_parameters.
        group_keys = []
        compatible_key = self.rollout_map_local_key_to_hf_key(param_name)
        if ("qkv_proj" in compatible_key) or (
            "qkv" in compatible_key and not self.is_vlm
        ):
            # must be inplace slicing.
            # split qkv weight
            rule = "qkv_proj" if "qkv_proj" in compatible_key else "qkv"
            q_weight, k_weight, v_weight = self._rollout_split_qkv_weight(
                compatible_key, param
            )
            q_proj_weight_key = compatible_key.replace(rule, "q_proj")
            k_proj_weight_key = compatible_key.replace(rule, "k_proj")
            v_proj_weight_key = compatible_key.replace(rule, "v_proj")
            group_keys.append((q_proj_weight_key, q_weight))
            group_keys.append((k_proj_weight_key, k_weight))
            group_keys.append((v_proj_weight_key, v_weight))
        elif "in_proj_qkvz" in compatible_key:
            [in_proj_q_weight, in_proj_k_weight, in_proj_v_weight, in_proj_z_weight] = (
                self._split_in_proj_qkvz_weight(compatible_key, param)
            )
            in_proj_q_weight_key = compatible_key.replace("in_proj_qkvz", "in_proj_q")
            in_proj_k_weight_key = compatible_key.replace("in_proj_qkvz", "in_proj_k")
            in_proj_v_weight_key = compatible_key.replace("in_proj_qkvz", "in_proj_v")
            in_proj_z_weight_key = compatible_key.replace("in_proj_qkvz", "in_proj_z")
            group_keys.append((in_proj_q_weight_key, in_proj_q_weight))
            group_keys.append((in_proj_k_weight_key, in_proj_k_weight))
            group_keys.append((in_proj_v_weight_key, in_proj_v_weight))
            group_keys.append((in_proj_z_weight_key, in_proj_z_weight))
        elif (
            "gate_up_proj" in compatible_key
            and self.config.model_type not in models_do_not_split_gate_up_proj
        ):
            # split gate and up proj
            gate_proj_weight, up_proj_weight = self._split_gate_proj_weight(
                compatible_key, param, is_moe=".experts." in compatible_key
            )
            gate_proj_weight_key = compatible_key.replace("gate_up_proj", "gate_proj")
            group_keys.append((gate_proj_weight_key, gate_proj_weight))
            up_proj_weight_key = compatible_key.replace("gate_up_proj", "up_proj")
            group_keys.append((up_proj_weight_key, up_proj_weight))
        elif "in_proj_ba" in compatible_key:
            in_proj_b_weight, in_proj_a_weight = self._split_in_proj_ba_weight(
                compatible_key, param
            )
            in_proj_b_weight_key = compatible_key.replace("in_proj_ba", "in_proj_b")
            in_proj_a_weight_key = compatible_key.replace("in_proj_ba", "in_proj_a")
            group_keys.append((in_proj_b_weight_key, in_proj_b_weight))
            group_keys.append((in_proj_a_weight_key, in_proj_a_weight))
        elif "linear_attn.conv1d." in compatible_key:
            q_w, k_w, v_w = self._split_linear_attn_conv1d_weight(param)
            group_keys.append((compatible_key.replace("conv1d", "conv1d_q"), q_w))
            group_keys.append((compatible_key.replace("conv1d", "conv1d_k"), k_w))
            group_keys.append((compatible_key.replace("conv1d", "conv1d_v"), v_w))
        elif "qkv" in compatible_key:
            q_weight, k_weight, v_weight = self._rollout_split_qkv_weight(
                compatible_key, param
            )
            q_visual_proj_weight_key = compatible_key.replace("qkv", "q")
            k_visual_proj_weight_key = compatible_key.replace("qkv", "k")
            v_visual_proj_weight_key = compatible_key.replace("qkv", "v")
            group_keys.append((q_visual_proj_weight_key, q_weight))
            group_keys.append((k_visual_proj_weight_key, k_weight))
            group_keys.append((v_visual_proj_weight_key, v_weight))
        else:
            group_keys.append((compatible_key, param))
        return group_keys

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = util.clear_weight_name(name)
        if self.is_vlm:
            # Policy model is HFModel, so we need to reverse the hf checkpoint conversion mapping
            if self.reverse_hf_conversion_mapping:
                for pattern, replacement in self.reverse_hf_conversion_mapping.items():
                    if re.match(pattern, name):
                        name = re.sub(pattern, replacement, name)
                        break
            else:
                # Rollout model is vllm model, so we don't need to reverse the hf checkpoint conversion mapping
                pass
        else:
            if not name == "lm_head.weight":
                if not name.startswith("model."):
                    name = "model." + name
        return name

    def name_to_model_part_index(self, dest_name: str) -> int:
        if dest_name in ["lm_head.weight", "lm_head.bias"]:
            return 0
        elif dest_name.startswith("visual.") or dest_name.startswith("vision_tower."):
            return 1
        elif dest_name.startswith("model.") or dest_name.startswith("language_model."):
            return 0
        else:
            raise ValueError(f"Unsupported weight: {dest_name}")

    def policy_decompose_param_1_to_n_for_sync(self, name):
        """
        Map a parameter of the policy model to set of transformed parameters that need to be synchronized.
        This method returns a list containing tuples of the new parameter name and the corresponding new tensor transformed from the original tensor of the given name.
        Each tuple element includes a transformed tensor and its corresponding slice strategy to derive from the original tensor.
        """
        # Addedd for models with qkv_proj and gate_up_proj like phi4
        if match := re.search(  # noqa: F841
            r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)",
            name,
        ):
            total_size = self.kv_head_ratio + 2
            split_strategy = []
            # The first part of the split:
            # the dictionary means at dimension 0, extract the part of offset 0 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv_proj", "q_proj"),
                    {
                        0: {
                            "offset": 0,
                            "total_size": total_size,
                            "length": self.kv_head_ratio,
                        }
                    },
                )
            )
            # The second part of the split:
            # the dictionary means at dimension 0, extract the part of offset 1 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv_proj", "k_proj"),
                    {
                        0: {
                            "offset": self.kv_head_ratio,
                            "total_size": total_size,
                            "length": 1,
                        }
                    },
                )
            )
            # The third part of the split:
            # the dictionary means at dimension 0, extract the part of offset 2 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv_proj", "v_proj"),
                    {
                        0: {
                            "offset": self.kv_head_ratio + 1,
                            "total_size": total_size,
                            "length": 1,
                        }
                    },
                )
            )
            return split_strategy
        elif match := re.search(  # noqa: F841
            r"(visual|vision_tower)\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)",
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
        elif match := re.search(  # noqa: F841
            r"model\.layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)",
            name,
        ):
            split_strategy = []
            split_strategy.append(
                (
                    name.replace("gate_up_proj", "gate_proj"),
                    {0: {"offset": 0, "total_size": 2, "length": 1}},
                )
            )
            split_strategy.append(
                (
                    name.replace("gate_up_proj", "up_proj"),
                    {0: {"offset": 1, "total_size": 2, "length": 1}},
                )
            )
            return split_strategy
        elif match := re.search(  # noqa: F841
            r"model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight",
            name,
        ):
            key_dim, value_dim = self._linear_attn_key_value_dims()
            total_size = 2 * key_dim + value_dim

            split_strategy = []
            # The first part of the split:
            # the dictionary means at dimension 0, extract the part of offset 0 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv", "q"),
                    {0: {"offset": 0, "total_size": total_size, "length": key_dim}},
                )
            )
            # The second part of the split:
            # the dictionary means at dimension 0, extract the part of offset 1 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv", "k"),
                    {
                        0: {
                            "offset": key_dim,
                            "total_size": total_size,
                            "length": key_dim,
                        }
                    },
                )
            )
            # The third part of the split:
            # the dictionary means at dimension 0, extract the part of offset 2 and length 1 when regarding the whole 0 dimension as length 3.
            split_strategy.append(
                (
                    name.replace("qkv", "v"),
                    {
                        0: {
                            "offset": key_dim * 2,
                            "total_size": total_size,
                            "length": value_dim,
                        }
                    },
                )
            )
            return split_strategy
        elif match := re.search(  # noqa: F841
            r"model\.language_model\.layers\.(\d+)\.linear_attn\.conv1d\.(weight|bias)",
            name,
        ):
            key_dim, value_dim = self._linear_attn_key_value_dims()
            total_size = 2 * key_dim + value_dim
            split_strategy = []
            split_strategy.append(
                (
                    name.replace("conv1d", "conv1d_q"),
                    {0: {"offset": 0, "total_size": total_size, "length": key_dim}},
                )
            )
            split_strategy.append(
                (
                    name.replace("conv1d", "conv1d_k"),
                    {
                        0: {
                            "offset": key_dim,
                            "total_size": total_size,
                            "length": key_dim,
                        }
                    },
                )
            )
            split_strategy.append(
                (
                    name.replace("conv1d", "conv1d_v"),
                    {
                        0: {
                            "offset": key_dim * 2,
                            "total_size": total_size,
                            "length": value_dim,
                        }
                    },
                )
            )
            return split_strategy
        elif match := re.search(  # noqa: F841
            r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj",
            name,
        ):
            split_strategy = []
            split_strategy.append(
                (
                    name.replace("gate_up_proj", "gate_proj"),
                    {1: {"offset": 0, "total_size": 2, "length": 1}},
                )
            )
            split_strategy.append(
                (
                    name.replace("gate_up_proj", "up_proj"),
                    {1: {"offset": 1, "total_size": 2, "length": 1}},
                )
            )
            return split_strategy
        return []

    @cached_property
    def packed_modules_mapping(self):
        mapping_dict = {
            "qkv.": [
                "q.",
                "k.",
                "v.",
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
        if self.config.model_type == "gpt_oss":
            mapping_dict["qkv."] = ["q_proj.", "k_proj.", "v_proj."]
        elif self.config.model_type in ["qwen3_5", "qwen3_5_moe"]:
            mapping_dict["in_proj_qkvz"] = [
                "in_proj_q",
                "in_proj_k",
                "in_proj_v",
                "in_proj_z",
            ]
            mapping_dict["in_proj_ba"] = ["in_proj_b", "in_proj_a"]
            mapping_dict["linear_attn.conv1d"] = [
                "linear_attn.conv1d_q",
                "linear_attn.conv1d_k",
                "linear_attn.conv1d_v",
            ]

        return mapping_dict

    def update_tensor_view(
        self,
        tensor_view: torch.Tensor,
        recv_tensor: torch.Tensor,
        inst_dest_name: str,
        **kwargs,
    ):
        tmp_recv_tensor = recv_tensor.to(tensor_view.dtype)
        if self.config.model_type == "gpt_oss" and "down_proj_bias" in inst_dest_name:
            assert "parallel_dims" in kwargs, (
                "parallel_dims is required for update_tensor_view"
            )
            tp_rank, _ = kwargs["parallel_dims"].tp_coord
            if tp_rank != 0:
                tmp_recv_tensor.zero_()
        tensor_view.copy_(tmp_recv_tensor)
