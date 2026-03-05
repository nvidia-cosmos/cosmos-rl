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

from cosmos_rl.utils.parallelism import ParallelDims
import torch
from typing import Tuple, Optional, Any, Dict
from functools import partial
import re

def fsdp_sharding(shard: torch.Tensor, dp_shard_rank: int, dp_shard_size: int) -> torch.Tensor:
    row_size = shard.shape[0]
    if row_size % dp_shard_size != 0:
        average_row_size = (row_size + dp_shard_size - 1) // dp_shard_size
        start_idx = dp_shard_rank * average_row_size
        end_idx = min(start_idx + average_row_size, row_size)
        shard = shard[start_idx:end_idx]
    else:
        shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return shard.contiguous()


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    parallel_dims: ParallelDims,
    tp_slice_dim: Optional[int] = None,
    ignore_unknown_weights: bool = False,
    hf_config: Optional[Any] = None,
) -> Tuple[str, torch.Tensor]:
    # Since TP is not used in this model, we use tp_coord to represent the expert rank and size
    ep_rank, ep_size = parallel_dims.tp_coord

    language_model_config = hf_config.text_config if 'text_config' in hf_config else hf_config
    n_experts = language_model_config.n_routed_experts

    if ep_size > 1 and "mixer.experts." in name:
        # for example, `backbone.layers.1.mixer.experts.32`
        dp_shard_mesh = parallel_dims.mesh[tuple(("dp_shard_cp",))]
        dp_shard_rank, dp_shard_size = dp_shard_mesh.get_local_rank(), dp_shard_mesh.size()
        if (
            match := re.search(
            r"model.language_model.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)$",
            name)
        ) is not None:
            # shard = tensor.tensor_split(ep_size, dim=0)[ep_rank]
            # Check whether this expert belongs to the current process
            # Groups example (with 32 experts, and 4 EP groups):
            #  EP=0: 0, 1, 2, 3, 4, 5, 6, 7
            #  EP=1: 8, 9, 10, 11, 12, 13, 14, 15
            #  EP=2: 16, 17, 18, 19, 20, 21, 22, 23
            #  EP=3: 24, 25, 26, 27, 28, 29, 30, 31
            n_expert_per_ep = n_experts // ep_size
            belongs_to_current_ep = (
                ep_rank * n_expert_per_ep
                <= int(match.group(2))  # Expert index
                < (ep_rank + 1) * n_expert_per_ep
            )
            belongs_to_current_dp_shard = (
                int(match.group(2)) - ep_rank * n_expert_per_ep
            ) // (n_expert_per_ep // dp_shard_size) == dp_shard_rank
            if belongs_to_current_ep and belongs_to_current_dp_shard:
                # No fsdp applied to expert weights of shape [out_features, in_features]
                # since the sharding dim is only on dim 0 of shape [n_experts, out_features, in_features]

                def transform_name_fn(model_state_dict: Dict[str, torch.Tensor], name: str) -> Optional[torch.Tensor]:
                    m = re.search(
                        r"^(model\.language_model\.layers\.\d+\.mixer\.experts)\.\d+\.((?:up_proj|gate_proj|down_proj)\.(?:weight|bias))$",
                        name,
                    )
                    assert m is not None, f"Unsupported weight: {name}"

                    # `prefix` is the prefix of the weight name, e.g. "backbone.layers.1.mixer.experts"
                    # `suffix` is the suffix of the weight name, e.g. "up_proj.weight"
                    prefix, suffix = m.group(1), m.group(2)
                    suffix = suffix.replace("up_proj.weight", "gate_and_up_projs").replace("down_proj.weight", "down_projs")

                    local_expert_idx = int(match.group(2)) % (n_experts // (ep_size * dp_shard_size))
                    target_tensor = model_state_dict[prefix + "." + suffix]
                    if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                        target_tensor = target_tensor.to_local()

                    return target_tensor[local_expert_idx]

                return partial(transform_name_fn, name=name), tensor
            else:
                # If the expert does not belong to the current process, return None to skip this weight
                return None, None
        elif (
            match := re.search(
            r"backbone.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)$",
            name)
        ) is not None:
            # shard = tensor.tensor_split(ep_size, dim=0)[ep_rank]
            # Check whether this expert belongs to the current process
            # Groups example (with 32 experts, and 4 EP groups):
            #  EP=0: 0, 1, 2, 3, 4, 5, 6, 7
            #  EP=1: 8, 9, 10, 11, 12, 13, 14, 15
            #  EP=2: 16, 17, 18, 19, 20, 21, 22, 23
            #  EP=3: 24, 25, 26, 27, 28, 29, 30, 31
            n_expert_per_ep = n_experts // ep_size
            belongs_to_current_ep = (
                ep_rank * n_expert_per_ep
                <= int(match.group(2))  # Expert index
                < (ep_rank + 1) * n_expert_per_ep
            )
            belongs_to_current_dp_shard = (
                int(match.group(2)) - ep_rank * n_expert_per_ep
            ) // (n_expert_per_ep // dp_shard_size) == dp_shard_rank
            if belongs_to_current_ep and belongs_to_current_dp_shard:
                # No fsdp applied to expert weights of shape [out_features, in_features]
                # since the sharding dim is only on dim 0 of shape [n_experts, out_features, in_features]

                def transform_name_fn(model_state_dict: Dict[str, torch.Tensor], name: str) -> Optional[torch.Tensor]:
                    m = re.search(
                        r"^(backbone\.layers\.\d+\.mixer\.experts)\.\d+\.((?:up_proj|gate_proj|down_proj)\.(?:weight|bias))$",
                        name,
                    )
                    assert m is not None, f"Unsupported weight: {name}"

                    # `prefix` is the prefix of the weight name, e.g. "backbone.layers.1.mixer.experts"
                    # `suffix` is the suffix of the weight name, e.g. "up_proj.weight"
                    prefix, suffix = m.group(1), m.group(2)
                    suffix = suffix.replace("up_proj.weight", "gate_and_up_projs").replace("down_proj.weight", "down_projs")

                    local_expert_idx = int(match.group(2)) % (n_experts // (ep_size * dp_shard_size))
                    target_tensor = model_state_dict[prefix + "." + suffix]
                    if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                        target_tensor = target_tensor.to_local()

                    return target_tensor[local_expert_idx]
                return partial(transform_name_fn, name=name), tensor
            else:
                # If the expert does not belong to the current process, return None to skip this weight
                return None, None
        else:
            raise ValueError(f"Unsupported weight: {name}")
    else:
        # When vision_replicate is enabled, vision and projector weights are
        # replicated (not FSDP-sharded), so load them in full on every rank.
        if getattr(hf_config, '_vision_replicate', False) and _is_vision_or_projector_weight(name):
            return name, tensor.contiguous()

        dp_group_names = ["dp_shard"]
        if parallel_dims.cp_enabled:
            dp_group_names.append("cp")
        if ep_size > 1:
            # although EP is enabled, the weights other than experts are still sharded via EP+DP_SHARD fused mesh
            dp_group_names.append("tp")
        flattened_dp_group = parallel_dims.mesh[tuple(dp_group_names)]._flatten(mesh_dim_name="fsdp_no_moe")

        if hasattr(parallel_dims, "mesh"):
            dp_shard_rank = flattened_dp_group.get_local_rank()
            dp_shard_size = flattened_dp_group.size()
        else:
            dp_shard_rank = 0
            dp_shard_size = 1
        return name, fsdp_sharding(tensor.contiguous(), dp_shard_rank, dp_shard_size)


# Known vision model and multi-modal projector weight name prefixes across VLM architectures.
_VISION_REPLICATE_PREFIXES = (
    "model.visual.",                  # NemotronH_Nano_VL
    "model.vision_model.",            # Generic VLMs
    "model.vision_tower.",            # LLaVA-style
    "model.multi_modal_projector.",   # Generic projector
    "model.mlp1.",                    # InternVL projector
    "visual.",                        # Without model. prefix
)

def _is_vision_or_projector_weight(name: str) -> bool:
    return any(name.startswith(p) for p in _VISION_REPLICATE_PREFIXES)

