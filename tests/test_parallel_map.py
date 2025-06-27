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

import unittest

from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapper,
)
from cosmos_rl.utils.parallelism import ParallelismConfig
import torch
from cosmos_rl.utils.parallelism_registry import (
    get_policy_parallelism_strategy,
    get_rollout_parallelism_strategy,
)
from cosmos_rl.policy.model.gpt.weight_converter import map_weight_parallel_dims


class TestModelType:
    num_hidden_layers = 12


class TestParallelMap(unittest.TestCase):
    def test_parallel_topo_mapper(self):
        # Create a mock ParallelismConfig object
        policy_parallelism_config = ParallelismConfig(
            dp_shard_size=-1, cp_size=1, tp_size=2, pp_size=1
        )
        rollout_parallelism_config = ParallelismConfig(
            dp_shard_size=-1, cp_size=1, tp_size=4, pp_size=1
        )
        p_world_size = 8
        r_world_size = 4

        assert map_weight_parallel_dims is not None
        model_type = "gpt"
        mapper = ParallelTopoMapper(
            policy_parallelism_config,
            rollout_parallelism_config,
            poilcy_world_size=p_world_size,
            rollout_world_size=r_world_size,
            policy_parallelism_strategy=get_policy_parallelism_strategy(model_type),
            rollout_parallelism_strategy=get_rollout_parallelism_strategy(model_type),
            hf_config=TestModelType(),
        )
        layers = [
            ("model.layers.9.input_layernorm.weight", torch.Size([1024])),
            ("model.layers.9.mlp.down_proj.weight", torch.Size([1024, 11008])),
            ("model.layers.9.mlp.gate_proj.weight", torch.Size([5504, 2048])),
            ("model.layers.9.mlp.up_proj.weight", torch.Size([5504, 2048])),
            ("model.layers.9.post_attention_layernorm.weight", torch.Size([1024])),
            ("model.layers.9.self_attn.k_proj.bias", torch.Size([128])),
            ("model.layers.9.self_attn.k_proj.weight", torch.Size([128, 2048])),
            ("model.layers.9.self_attn.o_proj.weight", torch.Size([1024, 2048])),
            ("model.layers.9.self_attn.q_proj.bias", torch.Size([1024])),
            ("model.layers.9.self_attn.q_proj.weight", torch.Size([1024, 2048])),
            ("model.layers.9.self_attn.v_proj.bias", torch.Size([128])),
            ("model.layers.9.self_attn.v_proj.weight", torch.Size([128, 2048])),
            ("lm_head.weight", torch.Size([75968, 2048])),
            ("model.norm.weight", torch.Size([1024])),
            ("model.embed_tokens.weight", torch.Size([75968, 2048])),
        ]
        global_rank = 5
        insts = mapper.policy_to_rollout_manifest(
            params=[(x[0], len(x[1])) for x in layers], global_rank=int(global_rank)
        )
        r_rank_max = 0
        layer_idx = 0
        for inst in insts:
            p_rank, r_rank, tensor_split_strategys, dest_name, _ = inst
            assert p_rank == global_rank
            while layers[layer_idx][0] != dest_name:
                r_rank_max = 0
                layer_idx += 1
            assert layers[layer_idx][0] == dest_name
            assert r_rank >= r_rank_max
            if r_rank > r_rank_max:
                r_rank_max = r_rank

        global_rank = 2
        insts = mapper.rollout_from_policy_manifest(
            params=[(x[0], len(x[1])) for x in layers], rollout_rank=int(global_rank)
        )
        p_rank_max = 0
        layer_idx = 0
        for inst in insts:
            p_rank, r_rank, tensor_split_strategys, dest_name, _ = inst
            assert r_rank == global_rank
            while layers[layer_idx][0] != dest_name:
                p_rank_max = 0
                layer_idx += 1
            assert layers[layer_idx][0] == dest_name
            assert p_rank >= p_rank_max
            if p_rank > p_rank_max:
                p_rank_max = p_rank


if __name__ == "__main__":
    unittest.main()
