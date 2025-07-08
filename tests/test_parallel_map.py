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
    ParallelTopoMapperGroup,
    ParallelTopoMapper,
    ParallelizedShardMapper,
)
from cosmos_rl.policy.model.gpt.weight_mapper import GPTWeightMapper
from cosmos_rl.utils.parallelism import ParallelismConfig, ParallelDims
import os
import sys
import subprocess


class TestModelType:
    num_hidden_layers = 12
    num_attention_heads = 32
    num_key_value_heads = 32
    hidden_size = 1024
    num_attention_heads = 32
    model_type = "gpt"


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

        policy_parallel_dims = ParallelDims.from_config_for_analysis(
            policy_parallelism_config, p_world_size
        )
        rollout_parallel_dims = ParallelDims.from_config_for_analysis(
            rollout_parallelism_config, r_world_size
        )

        policy_weight_mapper = GPTWeightMapper(
            hf_config=TestModelType  # Assuming a mock config for testing
        )
        rollout_weight_mapper = GPTWeightMapper(
            hf_config=TestModelType  # Assuming a mock config for testing
        )

        def dummy(*args, **kwargs):
            return None

        ParallelTopoMapper.parallelism_info_for_dtensor_params = dummy
        ParallelTopoMapper.parallelism_info_for_vllm_params = dummy

        policy_mapper = ParallelTopoMapperGroup(
            global_parallelism=policy_parallel_dims,
            hf_config=TestModelType,
            is_policy=True,
            underlying_model=None,
            weight_mapper=policy_weight_mapper,
        )
        rollout_mapper = ParallelTopoMapperGroup(
            global_parallelism=rollout_parallel_dims,
            hf_config=TestModelType,
            is_policy=False,
            underlying_model=None,
            weight_mapper=rollout_weight_mapper,
        )

        assert len(policy_mapper.mapper_group) == 1
        assert len(rollout_mapper.mapper_group) == 1

        def name_to_hf(name: str) -> str:
            return name

        #  insert_to_parallelism_info(
        # self,
        # param_name: str,
        # dims_map: Dict[str, int],
        # name_to_hf: Callable,

        layers = [
            ("model.layers.9.input_layernorm.weight", {}),
            ("model.layers.9.mlp.down_proj.weight", {"tp": 1}),
            ("model.layers.9.mlp.gate_proj.weight", {"tp": 0}),
            ("model.layers.9.mlp.up_proj.weight", {"tp": 0}),
            ("model.layers.9.post_attention_layernorm.weight", {}),
            ("model.layers.9.self_attn.k_proj.bias", {"tp": 0}),
            ("model.layers.9.self_attn.k_proj.weight", {"tp": 0}),
            ("model.layers.9.self_attn.o_proj.weight", {"tp": 0}),
            ("model.layers.9.self_attn.q_proj.bias", {"tp": 0}),
            ("model.layers.9.self_attn.q_proj.weight", {"tp": 0}),
            ("model.layers.9.self_attn.v_proj.bias", {"tp": 0}),
            ("model.layers.9.self_attn.v_proj.weight", {"tp": 0}),
            ("lm_head.weight", {"tp": 0}),
            ("model.norm.weight", {}),
            ("model.embed_tokens.weight", {"tp": 0}),
        ]
        policy_mapper.mapper_group[0].parallelism_info_for_params = {}
        for k, v in layers:
            policy_mapper.mapper_group[0].insert_to_parallelism_info(
                param_name=k,
                dims_map=v | {"dp_shard_cp": 0},
                name_to_hf=name_to_hf,
            )

        rollout_mapper.mapper_group[0].parallelism_info_for_params = {}
        for k, v in layers:
            rollout_mapper.mapper_group[0].insert_to_parallelism_info(
                param_name=k,
                dims_map=v | {"dp_shard_cp": 0},
                name_to_hf=name_to_hf,
            )

        local_shards_p = [
            policy_mapper.prepare_local_shard_infos(
                hf_key_n_rank=[[x] for x in layers], global_rank=p_rank
            )
            for p_rank in range(p_world_size)
        ]
        local_shards_r = [
            rollout_mapper.prepare_local_shard_infos(
                hf_key_n_rank=[[x] for x in layers], global_rank=r_rank
            )
            for r_rank in range(r_world_size)
        ]

        generator = ParallelizedShardMapper()
        generator.set_shard_infos_of_policy(local_shards_p)
        generator.set_shard_infos_of_rollout(local_shards_r)

        global_rank = 5
        insts = generator.generate_parallelized_shard_send_insts_for_policy(global_rank)
        r_rank_max = 0
        layer_idx = 0

        layers.sort(key=lambda x: x[0])
        for inst_group in insts:
            for inst in inst_group:
                dest_name = inst["name"]
                for i in inst["insts"]:
                    p_rank, r_rank, tensor_split_strategys = i
                    assert p_rank == global_rank
                    while layers[layer_idx][0] != dest_name:
                        r_rank_max = 0
                        layer_idx += 1
                    assert layers[layer_idx][0] == dest_name
                    assert r_rank >= r_rank_max
                    if r_rank > r_rank_max:
                        r_rank_max = r_rank

        global_rank = 2
        insts = generator.generate_parallelized_shard_recv_insts_for_rollout(
            global_rank
        )

        p_rank_max = 0
        layer_idx = 0
        for inst_group in insts:
            for inst in inst_group:
                dest_name = inst["name"]
                for i in inst["insts"]:
                    p_rank, r_rank, tensor_split_strategys = i
                    assert r_rank == global_rank
                    while layers[layer_idx][0] != dest_name:
                        p_rank_max = 0
                        layer_idx += 1
                    assert layers[layer_idx][0] == dest_name
                    assert p_rank >= p_rank_max
                    if p_rank > p_rank_max:
                        p_rank_max = p_rank

    def test_policy_parallelism_extract(self):
        """Test policy parallelism extraction using torchrun."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        cases = [
            [2, 1, 2],  # fsdp: 2, tp: 1, pp: 2
            [2, 2, 1],  # fsdp: 2, tp: 2, pp: 1
            [1, 1, 1],  # fsdp: 1, tp: 1, pp: 1
        ]

        for case in cases:
            fsdp = case[0]
            tp = case[1]
            pp = case[2]

            world_size = fsdp * tp * pp  # Total number of processes

            # Create the Python command for torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={world_size}",  # Use 4 GPUs
                "--role=rank",
                "--tee=3",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                os.path.join(cur_dir, "launch_test_worker.py"),
                "-1",  # Use -1 to indicate no need for shared memory
                "-1",  # Use -1 to indicate no need for shared memory size
                "policy_parallelism_extract",
                f"fsdp:{fsdp};tp:{tp};pp:{pp}",
            ]

            env = dict(os.environ)
            # Start the process
            policy_process = subprocess.Popen(
                cmd,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=env,
            )
            try:
                # Wait for process to complete
                for process in [policy_process]:
                    stdout, stderr = process.communicate()

                    # Check if process completed successfully
                    assert process.returncode == 0, f"Process failed: {stderr.decode()}"

            finally:
                # Ensure process is terminated
                for process in [policy_process]:
                    process.wait()

    def test_rollout_parallelism_extract(self):
        """Test rollout parallelism extraction using torchrun."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        cases = [[4, 1], [1, 1]]

        for case in cases:
            fsdp = 1  # always FSDP to be 1 for rollout parallelism.
            tp = case[0]
            pp = case[1]
            world_size = fsdp * tp * pp  # Total number of processes

            # Create the Python command for torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={world_size}",  # Use 4 GPUs
                "--role=rank",
                "--tee=3",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                os.path.join(cur_dir, "launch_test_worker.py"),
                "-1",  # Use -1 to indicate no need for shared memory
                "-1",  # Use -1 to indicate no need for shared memory size
                "rollout_parallelism_extract",
                f"fsdp:{fsdp};tp:{tp};pp:{pp}",
            ]

            env = dict(os.environ)
            # Start the process
            policy_process = subprocess.Popen(
                cmd,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=env,
            )
            try:
                # Wait for process to complete
                for process in [policy_process]:
                    stdout, stderr = process.communicate()

                    # Check if process completed successfully
                    assert process.returncode == 0, f"Process failed: {stderr.decode()}"

            finally:
                # Ensure process is terminated
                for process in [policy_process]:
                    process.wait()


if __name__ == "__main__":
    unittest.main()
