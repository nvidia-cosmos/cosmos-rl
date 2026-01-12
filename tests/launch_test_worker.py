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

import os
import torch
import time
from typing import Any, List, Dict
import argparse
from multiprocessing import shared_memory, Event as mp_Event
from unittest.mock import Mock
import numpy as np
import torch.distributed as dist
import toml
from transformers import AutoConfig
import cosmos_rl.utils.util as util
from cosmos_rl.policy.model import ModelRegistry, WeightMapper
import msgpack
import threading
from queue import Queue
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from cosmos_rl.policy.trainer import GRPOTrainer, SFTTrainer, LLMTrainer
from cosmos_rl.policy.worker import SFTPolicyWorker, RLPolicyWorker
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)
from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from cosmos_rl.rollout import State
import types
from cosmos_rl.dispatcher.command import (
    PolicyToRolloutUnicastCommand,
    PolicyToPolicyUnicastCommand,
    PolicyToPolicyBroadcastCommand,
)
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
    ParallelTopoMapper,
    ParallelizedShardMapper,
    WeightSyncInstructionsGroup,
)
from cosmos_rl.utils.parallelism import ParallelismConfig, ParallelDims
from cosmos_rl.utils.distributed import (
    init_distributed,
    destroy_distributed,
)
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.policy.model.gpt.weight_converter import convert_weight_from_hf
from cosmos_rl.policy.model.gpt.weight_mapper import GPTWeightMapper
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.comm.base import CommMixin
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_send,
    nccl_recv,
    nccl_broadcast,
)
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.utils.logging import logger
import asyncio
from cosmos_rl.dispatcher.data.packer import (
    DecoderOnlyLLMDataPacker,
)
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.policy.config import GrpoConfig
import uuid
from cosmos_rl.utils.ulysses import (
    slice_inputs_for_ulysses,
)
from cosmos_rl.utils.sequence_packing import (
    pack_sequences_info_collect,
    pack_sequences_for_masks,
    pack_sequences_for_labels,
)
from torch.utils.data import DataLoader, DistributedSampler, Sampler, BatchSampler
from cosmos_rl.policy.worker.sft_worker import collate_fn, construct_dataset
from torch.utils.data import Dataset
from datasets import concatenate_datasets
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.dispatcher.algo.reward import boxed_math_reward_fn
import multiprocessing as mp
from cosmos_rl.dispatcher.replica import Rollout

POLICY_WORLD_SIZE = 4
ROLLOUT_WORLD_SIZE = 4


class TestDataset(Dataset):
    def __init__(self, config: CosmosConfig):
        pass

    def setup(
        self,
        config: CosmosConfig,
    ):
        dataset = util.load_data_from_disk_or_hf(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
            config.train.train_policy.dataset.revision or None,
        )
        dataset_list = []
        for split_name in config.train.train_policy.dataset.split:
            dataset_list.append(dataset[split_name])
        self.response_column = None
        if isinstance(config.train.train_policy, GrpoConfig):
            self.response_column = config.train.train_policy.response_column_name
        self.dataset = concatenate_datasets(dataset_list)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_reference_answer(self, idx: int) -> Any:
        if self.response_column is None:
            raise ValueError(
                "You are under SFT config, but trying to get reference answer for GRPO."
            )
        return self.dataset[idx][self.response_column]


def load_simple_grpo_config():
    config_name = "test_simple_grpo.toml"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cur_dir, "configs", config_name)
    with open(config_path, "r") as f:
        config_dict = toml.load(f)
        config_dict["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, config_dict["train"]["train_policy"]["dataset"]["name"]
        )
        return config_dict


def load_simple_sft_config():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cur_dir, "configs", "test_simple_sft.toml")
    with open(config_path, "r") as f:
        config_dict = toml.load(f)
        return config_dict


class TestModel:
    model_type = "qwen2"
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    num_hidden_layers = 16

    def __init__(self, device, parallel_dims, freeze_params: bool = False):
        self.sorted_hf_key_n_rank = [
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

        self.parallel_spec = [
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

        self.keys_to_freeze = (
            [
                "model.layers.9.mlp.down_proj.weight",
                "model.layers.9.mlp.gate_proj.weight",
                "model.layers.9.mlp.up_proj.weight",
            ]
            if freeze_params
            else []
        )

        self.sorted_hf_key_n_rank.sort(key=lambda x: x[0])

        self.config = AutoConfig.from_pretrained(self.model_path)
        self.device = device
        self.parallel_dims = parallel_dims
        self.tensors = [
            (
                k,
                (
                    torch.arange(v.numel(), dtype=torch.float32, device=self.device)
                    .reshape(v)
                    .to(self.device)
                    * 0.001
                ).requires_grad_(True)
                if k not in self.keys_to_freeze
                else (
                    torch.arange(v.numel(), dtype=torch.float32, device=self.device)
                    .reshape(v)
                    .to(self.device)
                    * 0.001
                ).requires_grad_(False),
            )
            for k, v in self.sorted_hf_key_n_rank
        ]
        self.sharded_tensors = {}
        for k, v in self.tensors:
            self.sharded_tensors[k] = convert_weight_from_hf(
                v, k, self.model_type, self.parallel_dims
            )[1]
        self.sorted_sharded_params = [
            (k, self.sharded_tensors[k].ndim) for k, _ in self.sorted_hf_key_n_rank
        ]
        self.weight_mapper = GPTWeightMapper(self.config)

    def get_trainable_params(self):
        trainable_params = set()
        for k, v in self.sharded_tensors.items():
            if v.requires_grad:
                trainable_params.add(k)
        return trainable_params


class TestPolicyTrainer:
    def __init__(
        self, config, parallel_dims, train_stream: torch.cuda.Stream, **kwargs
    ):
        self.config = config
        self.parallel_dims = parallel_dims
        self.train_stream = train_stream
        freeze_params = kwargs.get("freeze_params", False)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.global_rank = int(os.environ.get("RANK", 0))
        self.model = TestModel(self.device, self.parallel_dims, freeze_params)
        self.map_w_from_policy_to_rollout = self.model.sharded_tensors
        self.model.sorted_hf_key_n_rank = self.model.sorted_sharded_params

    def build_optimizers(self):
        pass

    def build_lr_schedulers(self):
        pass

    def prepare_trainable_params(self):
        self.trainable_params = self.model.get_trainable_params()

    def train(self):
        pass


class TestPolicyWorker:
    def __init__(
        self, name, policy_world_size, rollouts_comm, freeze_params: bool = False
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.global_rank = int(os.environ.get("RANK", 0))
        self.role = Role.POLICY
        self.world_size = policy_world_size
        policy_parallelism_dims = ParallelismConfig(
            dp_shard_size=2, cp_size=1, tp_size=2, pp_size=1
        )
        self.parallel_dims = ParallelDims.from_config(
            policy_parallelism_dims,
        )
        self.parallel_dims.build_mesh(device_type="cuda")
        self.replica_name = name
        self.rollouts_comm = rollouts_comm
        self.policy_to_rollout_insts = None

        self.p2r_nccl_uuids = rollouts_comm
        self.train_stream = torch.cuda.Stream()
        self.config = CosmosConfig()
        self.config.train.param_dtype = "float32"
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.config.train.train_policy.dataset.name = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        self.trainer = TestPolicyTrainer(
            config=self.config,
            parallel_dims=self.parallel_dims,
            train_stream=self.train_stream,
            freeze_params=freeze_params,
        )
        self.parallel_mapper = ParallelTopoMapperGroup(
            self.parallel_dims,
            self.trainer.model.config,
            True,
            self.trainer.model,
            self.trainer.model.weight_mapper,
        )

        self.prepare_trainable_params()

    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        pass

    def pre_P2R_collect_parameters(self):
        return {}

    def prepare_trainable_params(self):
        self.trainable_params = self.trainer.model.get_trainable_params()

    def execute(self):
        assert self.trainer is not None, "[Policy] Trainer has not been built."
        try:
            self.main_loop()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.destroy_worker()

    def main_loop(self):
        pass

    def destroy_worker(self):
        destroy_distributed()


class TestRollout:
    def __init__(
        self, name, rollout_world_size, policies_comm, freeze_params: bool = False
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.global_rank = int(os.environ.get("RANK", 0))
        self.role = Role.ROLLOUT
        self.world_size = rollout_world_size
        self.policy_to_rollout_nccl_communicators = policies_comm
        rollout_parallelism_config = ParallelismConfig(
            dp_shard_size=1, cp_size=1, tp_size=4, pp_size=1
        )
        self.replica_name = name
        self.parallel_dims = ParallelDims.from_config(
            rollout_parallelism_config,
        )
        self.parallel_dims.build_mesh(device_type="cuda")
        self.model = TestModel(self.device, self.parallel_dims, freeze_params)
        self.parallel_mapper = ParallelTopoMapperGroup(
            self.parallel_dims,
            self.model.config,
            False,
            self.model,
            self.model.weight_mapper,
        )
        self.weight_mapper = self.parallel_mapper.weight_mapper
        compatibale_map = self.model.sharded_tensors
        compatibale_list = self.model.sorted_sharded_params
        operate_compatibale_map = {
            k: torch.zeros(v.shape, dtype=v.dtype).to(self.device)
            for k, v in compatibale_map.items()
        }
        self.ref_compatibale_map = compatibale_map
        self.quantization_type = None
        self.config = CosmosConfig()
        self.config.train.param_dtype = "float32"  # keep the same as policy above.

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.config.train.train_policy.dataset.name = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )

        self.weight_inplace_view_map = compatibale_map
        self.recv_param_key_n_rank_list = compatibale_list
        self.quantized_weight_map = {}
        self.hp_weight_map = {}

        self.operate_compatibale_map = operate_compatibale_map
        self.inference_stream = torch.cuda.Stream()
        self.state = State()

        self.recv_weight_shard = types.MethodType(
            DisaggregatedRolloutControlWorker.recv_weight_shard, self
        )
        # change the default parallelism config
        self.config.rollout.parallelism.tp_size = 4
        self.config.rollout.parallelism.pp_size = 1

        self.consume_command = types.MethodType(
            DisaggregatedRolloutControlWorker.consume_command, self
        )

        self.rollout = vLLMRollout(self.config, None, torch.cuda.current_device())

        self.temp_recv_tensor_queue = Queue()
        self.prepare_trainable_params()
        self.validation_flag = threading.Event()
        self.non_trainable_params_received = True

    def get_underlying_model(self):
        return None

    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        pass

    def prepare_trainable_params(self):
        self.trainable_params = self.model.get_trainable_params()

    def lazy_initialize_rollout_engine(self, load_format):
        pass


async def generate_send_recv_insts(model: TestModel, is_send: bool, global_rank: int):
    policy_parallelism_config = ParallelismConfig(
        dp_shard_size=2, cp_size=1, tp_size=2, pp_size=1
    )
    rollout_parallelism_config = ParallelismConfig(
        dp_shard_size=1, cp_size=1, tp_size=4, pp_size=1
    )
    p_world_size = 4
    r_world_size = 4

    policy_parallel_dims = ParallelDims.from_config_for_analysis(
        policy_parallelism_config, p_world_size
    )
    rollout_parallel_dims = ParallelDims.from_config_for_analysis(
        rollout_parallelism_config, r_world_size
    )

    policy_weight_mapper = GPTWeightMapper(hf_config=model.config)
    rollout_weight_mapper = GPTWeightMapper(hf_config=model.config)

    def dummy(*args, **kwargs):
        return None

    ParallelTopoMapper.parallelism_info_for_dtensor_params = dummy
    ParallelTopoMapper.parallelism_info_for_vllm_params = dummy

    policy_mapper = ParallelTopoMapperGroup(
        global_parallelism=policy_parallel_dims,
        hf_config=model.config,
        is_policy=True,
        underlying_model=None,
        weight_mapper=policy_weight_mapper,
    )
    rollout_mapper = ParallelTopoMapperGroup(
        global_parallelism=rollout_parallel_dims,
        hf_config=model.config,
        is_policy=False,
        underlying_model=None,
        weight_mapper=rollout_weight_mapper,
    )

    def name_to_hf(name: str) -> str:
        return name

    policy_mapper.mapper_group[0].parallelism_info_for_params = {}
    for k, v in model.parallel_spec:
        policy_mapper.mapper_group[0].insert_to_parallelism_info(
            param_name=k, dims_map=v | {"dp_shard_cp": 0}, name_to_hf=name_to_hf
        )

    rollout_mapper.mapper_group[0].parallelism_info_for_params = {}
    for k, v in model.parallel_spec:
        rollout_mapper.mapper_group[0].insert_to_parallelism_info(
            param_name=k,
            dims_map=v | {"dp_shard_cp": 0},
            name_to_hf=name_to_hf,
        )

    local_shards_p = [
        policy_mapper.prepare_local_shard_infos(
            hf_key_n_rank=model.sorted_hf_key_n_rank, global_rank=p_rank
        )
        for p_rank in range(p_world_size)
    ]
    local_shards_r = [
        rollout_mapper.prepare_local_shard_infos(
            hf_key_n_rank=model.sorted_hf_key_n_rank, global_rank=r_rank
        )
        for r_rank in range(r_world_size)
    ]
    config_dict = load_simple_grpo_config()
    cosmos_config = CosmosConfig.from_dict(config_dict)
    cosmos_config.policy.parallelism = policy_parallelism_config
    cosmos_config.rollout.parallelism = rollout_parallelism_config
    generator = ParallelizedShardMapper.get_instance(cosmos_config)
    p_params = [[x[0] for x in model.sorted_hf_key_n_rank] for _ in range(p_world_size)]
    r_params = [[x[0] for x in model.sorted_hf_key_n_rank] for _ in range(r_world_size)]
    p_body = {
        "shard_infos": local_shards_p,
        "param_groups": [],
        "sorted_params": p_params,
        "trainable_params": list(model.get_trainable_params()),
    }
    p_data = msgpack.packb(p_body)
    r_body = {
        "shard_infos": local_shards_r,
        "param_groups": [],
        "sorted_params": r_params,
    }
    r_data = msgpack.packb(r_body)

    await generator.set_shard_infos_of_policy(p_data, p_world_size)
    await generator.set_shard_infos_of_rollout(r_data, r_world_size)
    await generator.scheme_generation_done.wait()
    if is_send:
        insts_meta = await generator.get_send_insts_for_policy(global_rank)
    else:
        insts_meta = await generator.get_recv_insts_for_rollout(global_rank)
    insts = msgpack.unpackb(insts_meta, strict_map_key=False)
    policy_to_rollout_insts = [
        WeightSyncInstructionsGroup.from_dict(inst) for inst in insts
    ]
    return policy_to_rollout_insts


async def run_policy_send_to_rollout(shm_name, shm_size, rank, trainable_param_sync):
    """Run as a test policy process to send to rollout process"""
    # Set up NCCL communicator
    policy_name = "policy"
    rollout_name = "rollout"

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    command = PolicyToRolloutUnicastCommand(
        policy_name,
        rollout_name,
        POLICY_WORLD_SIZE,
        ROLLOUT_WORLD_SIZE,
        trainable_only=trainable_param_sync,
    )

    try:
        if rank == 0:
            nccl_uid = create_nccl_uid()
            # Create shared memory for NCCL UID
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            # Copy NCCL UID to shared memory
            uid_array[:-1] = nccl_uid
            uid_array[-1] = 1
        else:
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            while uid_array[-1] == 0:
                time.sleep(0.001)
            assert uid_array[-1] == 1, "Sender process did not set UID correctly"
            nccl_uid = uid_array[:-1].tolist()

        # Create NCCL communicator after UID is shared
        comm_idx = create_nccl_comm(
            nccl_uid, rank, POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE
        )
        policy_worker = TestPolicyWorker(
            policy_name,
            POLICY_WORLD_SIZE,
            {policy_name + "_" + rollout_name: comm_idx},
            trainable_param_sync,
        )
        policy_worker.policy_to_rollout_insts = await generate_send_recv_insts(
            policy_worker.trainer.model, True, rank
        )
        policy_worker.execute_policy_to_rollout_unicast = types.MethodType(
            RLPolicyWorker.execute_policy_to_rollout_unicast, policy_worker
        )
        policy_worker.execute_policy_to_rollout_unicast(command)
        policy_worker.train_stream.synchronize()

    finally:
        # Detach from shared memory
        shm.close()


async def run_rollout_recv_from_policy(shm_name, shm_size, rank, trainable_param_sync):
    """Run as a rollout process to receive from policy process"""
    # Set up NCCL communicator
    policy_name = "policy"
    rollout_name = "rollout"

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    command = PolicyToRolloutUnicastCommand(
        policy_name,
        rollout_name,
        POLICY_WORLD_SIZE,
        ROLLOUT_WORLD_SIZE,
        trainable_only=trainable_param_sync,
    )
    try:
        # Get NCCL UID from shared memory
        uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
        while uid_array[-1] == 0:
            time.sleep(0.001)
        assert uid_array[-1] == 1, "Sender process did not set UID correctly"
        nccl_uid = uid_array[:-1].tolist()
        # Create NCCL communicator with shared UID
        comm_idx = create_nccl_comm(
            nccl_uid, rank + POLICY_WORLD_SIZE, POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE
        )

        rollout = TestRollout(
            rollout_name,
            ROLLOUT_WORLD_SIZE,
            {policy_name + "_" + rollout_name: comm_idx},
            trainable_param_sync,
        )
        rollout.policy_to_rollout_recv_insts = await generate_send_recv_insts(
            rollout.model, False, rank
        )
        rollout.weight_mapper.map_to_unsplited_weight_name = {}
        rollout.policy_to_rollout_unicast = types.MethodType(
            DisaggregatedRolloutControlWorker.policy_to_rollout_unicast, rollout
        )
        rollout.prepare_shard_infos_for_weight_sync_insts = lambda: None
        rollout.policy_to_rollout_unicast(command)
        rollout.inference_stream.synchronize()

        for k, v in rollout.operate_compatibale_map.items():
            torch.allclose(v, rollout.ref_compatibale_map[k])

    finally:
        # Detach from shared memory
        shm.close()


def policy_to_policy_sync_common(
    shm_names,
    shm_size,
    rank,
    send,
    nccl_rank,
    nccl_size,
    policy_name,
    replica_name_to_rank,
    command,
):
    """Run as a policy process to perform unicast to another policy process or broadcast to all policy processes"""
    # Attach to shared memory
    shm_names = shm_names.split(",")
    shm_name = shm_names[rank]
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        # Get NCCL UID from shared memory
        if send:
            nccl_uid = create_nccl_uid()
            # Create shared memory for NCCL UID
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            # Copy NCCL UID to shared memory
            uid_array[:-1] = nccl_uid
            uid_array[-1] = 1
        else:
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            # Wait for sender process to set UID
            while uid_array[-1] == 0:
                time.sleep(0.001)
            assert uid_array[-1] == 1, "Sender process did not set UID correctly"
            nccl_uid = uid_array[:-1].tolist()

        # Create NCCL communicator with shared UID
        comm_idx = create_nccl_comm(nccl_uid, nccl_rank, nccl_size)

        # Construct the model and trainer
        config_dict = load_simple_grpo_config()

        cosmos_config = CosmosConfig.from_dict(
            config_dict,
        )
        parallel_dims = ParallelDims.from_config(cosmos_config.policy.parallelism)
        parallel_dims.build_mesh(device_type="cuda")

        def dummy(self, *args, **kwargs):
            pass

        def dummy_init_comm(self):
            self.api_client = APIClient(self.role, ["localhost"], 8000)
            self.shutdown_mp_signal = mp_Event()  # Must be a multiprocessing event

        def dummy_init_nccl(self, replica_name, global_rank, api_client):
            pass

        HighAvailabilitylNccl.__init__ = dummy_init_nccl

        class FakeNCCL:
            def __init__(self, comm_idx):
                self.comm_idx = comm_idx

            def get_replica_rank(self, replica_name: str):
                if replica_name in replica_name_to_rank:
                    return replica_name_to_rank[replica_name]
                else:
                    raise ValueError(
                        f"Replica name {replica_name} not found in mapping."
                    )

            def broadcast(self, tensor: torch.Tensor, src_replica: str):
                src_rank = self.get_replica_rank(src_replica)
                nccl_broadcast(tensor, src_rank, self.comm_idx)

            def send(self, tensor: torch.Tensor, dst_replica: str):
                dst_rank = self.get_replica_rank(dst_replica)
                nccl_send(tensor, dst_rank, self.comm_idx)

            def recv(self, tensor: torch.Tensor, src_replica: str):
                src_rank = self.get_replica_rank(src_replica)
                nccl_recv(tensor, src_rank, self.comm_idx)

            def shutdown(self):
                pass

        CommMixin.init_comm = dummy_init_comm
        CommMixin.init_redis = dummy
        CommMixin.start_heartbeat = dummy
        CommMixin.replica_name = policy_name
        CommMixin.shutdown_signal = threading.Event()
        RLPolicyWorker.prepare_shard_infos_for_weight_sync_insts = dummy
        # Fake the dataset

        policy_worker = RLPolicyWorker(cosmos_config, parallel_dims)
        policy_worker.replica_name = policy_name
        policy_worker.inter_policy_nccl = FakeNCCL(comm_idx)
        policy_worker.mesh_ready = True
        policy_worker.replica_name_to_rank = replica_name_to_rank

        def sample_tensor():
            sample_tensors = []
            self_state_dict = policy_worker.trainer.model.state_dict()
            sample_tensors.append(self_state_dict[sorted(self_state_dict.keys())[0]])
            sample_tensors.append(self_state_dict[sorted(self_state_dict.keys())[-1]])

            optimizer_state = policy_worker.trainer.optimizers.state_dict()
            sample_tensors.append(optimizer_state[sorted(optimizer_state.keys())[0]])
            sample_tensors.append(optimizer_state[sorted(optimizer_state.keys())[-1]])

            lr_sheduler_state = policy_worker.trainer.lr_schedulers.state_dict()
            sample_tensors.append(
                lr_sheduler_state[sorted(lr_sheduler_state.keys())[0]]
            )
            sample_tensors.append(
                lr_sheduler_state[sorted(lr_sheduler_state.keys())[-1]]
            )
            sample_tensors = [
                tensor.to_local().cpu()
                if isinstance(tensor, torch.distributed.tensor.DTensor)
                else tensor.cpu()
                if isinstance(tensor, torch.Tensor)
                else tensor
                for tensor in sample_tensors
            ]
            return sample_tensors

        if not send:
            sample_tensors = sample_tensor()

        if isinstance(command, PolicyToPolicyUnicastCommand):
            policy_worker.execute_policy_to_policy_unicast(command)
        elif isinstance(command, PolicyToPolicyBroadcastCommand):
            policy_worker.execute_policy_to_policy_broadcast(command)

        if not send:
            origin_sample_tensors = sample_tensors
            sample_tensors = sample_tensor()
            for tensor, origin_tensor in zip(sample_tensors, origin_sample_tensors):
                if isinstance(tensor, torch.Tensor):
                    assert torch.allclose(
                        tensor, origin_tensor
                    ), f"Tensor values do not match {tensor} {origin_tensor}"
                elif isinstance(tensor, bool):
                    assert (
                        tensor == origin_tensor
                    ), f"Tensor values do not match {tensor} {origin_tensor}"
    finally:
        # Detach from shared memory
        shm.close()


def run_policy_unicast_to_policy(shm_names, shm_size, rank, send):
    """Run as a policy process to perform unicast to another policy process"""
    policy_src_name = "policy_src"
    policy_dst_name = "policy_dst"
    command = PolicyToPolicyUnicastCommand(policy_src_name, policy_dst_name)
    nccl_rank = 0 if send else 1
    nccl_size = 2
    replica_name_to_rank = {policy_src_name: 0, policy_dst_name: 1}
    policy_name = policy_src_name if send else policy_dst_name
    # Call the common function to handle both send and receive
    policy_to_policy_sync_common(
        shm_names,
        shm_size,
        rank,
        send,
        nccl_rank,
        nccl_size,
        policy_name,
        replica_name_to_rank,
        command,
    )


def run_policy_broadcast_to_policy(shm_names, shm_size, rank, total_rep, self_rep):
    """Run as a policy process to perform broadcast to all policy processes"""
    policy_name = "policy_" + str(self_rep)
    policy_src = "policy_0"
    policy_dsts = ["policy_" + str(rep) for rep in range(total_rep)]
    command = PolicyToPolicyBroadcastCommand(policy_src, policy_dsts)
    nccl_rank = self_rep
    nccl_size = total_rep
    replica_name_to_rank = {"policy_" + str(i): i for i in range(total_rep)}
    send = policy_src == policy_name
    # Call the common function to handle both send and receive
    policy_to_policy_sync_common(
        shm_names,
        shm_size,
        rank,
        send,
        nccl_rank,
        nccl_size,
        policy_name,
        replica_name_to_rank,
        command,
    )


def run_overfitting_policy(args: argparse.Namespace):
    from cosmos_rl.policy.train import main as policy_main
    from cosmos_rl.utils.ulysses import slice_inputs_for_ulysses

    def _log_in_master(worker_or_trainer, msg):
        if (
            worker_or_trainer.config.logging.logger
            and util.is_master_rank(
                worker_or_trainer.parallel_dims, worker_or_trainer.global_rank
            )
            and "console" in worker_or_trainer.config.logging.logger
        ):
            logger.info(msg)

    N_STEPS = 30
    training_loss = []

    def train(
        self,
        global_batch,
        total_steps: int,
        train_step: int,
        save_freq: int,
        pp_last_stage: bool = False,
    ):
        if self.lr_schedulers is None:
            assert (
                train_step == 0
            ), "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            self.lr_schedulers = build_lr_schedulers(
                self.optimizers, self.config, total_steps
            )
        max_len = min(
            self.config.policy.model_max_length,
            self.data_packer.sft_compute_max_len(global_batch),
        )

        if self.seq_len_multiple > 1:
            max_len = (
                (max_len + self.seq_len_multiple - 1)
                // self.seq_len_multiple
                * self.seq_len_multiple
            )
        global_batch = self.data_packer.sft_collate_fn(
            global_batch,
            computed_max_len=max_len,
            ignore_label_id=-100,
        )

        for k, v in global_batch.items():
            global_batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

        labels = global_batch.pop("label_ids")

        position_ids, input_ids, _ = self.model.get_position_ids(**global_batch)

        global_batch["position_ids"] = position_ids
        padding_mask = global_batch.get("padding_mask", None)

        if self.parallel_dims.cp_enabled:
            [input_ids, position_ids, padding_mask] = slice_inputs_for_ulysses(
                [input_ids, position_ids, padding_mask],
                self.parallel_dims.mesh["cp"],
            )

            global_batch["input_ids"] = input_ids
            global_batch["position_ids"] = position_ids
            if padding_mask is not None:
                global_batch["padding_mask"] = padding_mask

        assert not self.parallel_dims.pp_enabled

        self.optimizers.zero_grad()
        self.model.train()
        logits = self.model(**global_batch)
        loss = self.loss_fn(
            logits,
            labels,
        )
        loss.backward()
        acc_loss = loss.detach()

        """
        Compute the global grad norm on all parameters and then apply
        gradient clipping using the global grad norm.
        """
        grad_norm = None
        if self.config.train.optm_grad_norm_clip > 0:
            # Must pass empty list even if model_part is None,
            # GradNorm across pp stages will fail if some rank does not join the barrier
            all_params = [
                p
                for m in [model for model in self.model_parts if model is not None]
                for p in m.parameters()
            ]
            grad_norm = dist_util.gradient_norm_clipping(
                all_params,
                self.config.train.optm_grad_norm_clip,
                foreach=True,
                pp_mesh=self.parallel_dims.mesh["pp"]
                if self.parallel_dims.pp_enabled
                else None,
            )

        self.optimizers.step()
        self.lr_schedulers.step()

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, _ = (
                dist_util.dist_mean(acc_loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(acc_loss, self.parallel_dims.mesh["dp_cp"]),
            )
        else:
            global_avg_loss = acc_loss.item()

        _log_in_master(
            self,
            f"Step: {train_step}/{N_STEPS}, Loss: {global_avg_loss:.5f}, Grad norm: {grad_norm:.5f}, Learning rate: {self.lr_schedulers.get_last_lr()[0]:.5e}",
        )
        return global_avg_loss, grad_norm

    def main_loop(self):
        global_batch = next(iter(self.train_data_loader))
        raw_batch = global_batch[0 : self.config.train.train_policy.mini_batch]

        for step in range(N_STEPS):
            _log_in_master(self, f"Training step {step + 1}/{N_STEPS}")

            global_avg_loss, grad_norm = self.trainer.step_training(
                raw_batch, N_STEPS, step, 0, False
            )

            training_loss.append(global_avg_loss)

        self.unregister_from_controller()

    SFTTrainer.step_training = train
    SFTPolicyWorker.main_loop = main_loop

    assert args is not None
    policy_main(args=args)

    # check the loss has been decreasing over time
    x = np.arange(len(training_loss))
    m, _ = np.polyfit(x, training_loss, 1)
    print(f"slope is {m}")
    assert m < 0


def run_dummy_policy(args: argparse.Namespace):
    """Run as a dummy policy process for testing"""
    from cosmos_rl.policy.train import main as policy_main

    def dummy_train_grpo(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        do_save_checkpoint: bool,
        inter_policy_nccl: Any,
        is_master_replica: bool,
    ):
        return {}

    def dummy(self):
        pass

    def dummy_model_load_from_hf(self):
        self.model_ready = True

    def dummy_execute_policy_to_rollout_unicast(self, command):
        return False

    GRPOTrainer.step_training = dummy_train_grpo
    GRPOTrainer.model_load_from_hf = dummy_model_load_from_hf

    def get_policy_command_handler(cls, command_type):
        if command_type == PolicyToRolloutUnicastCommand:
            return dummy_execute_policy_to_rollout_unicast
        return cls.policy_command_handler_registry.get_command_handler(command_type)

    RLPolicyWorker.prepare_shard_infos_for_weight_sync_insts = dummy
    RLPolicyWorker.get_policy_command_handler = get_policy_command_handler
    SFTTrainer.step_training = dummy
    SFTPolicyWorker.main_loop = dummy
    assert args is not None
    policy_main(args=args)


def run_dummy_rollout(args: argparse.Namespace):
    """Run as a dummy rollout process for testing purposes"""
    from cosmos_rl.rollout.rollout_entry import run_rollout

    def dummy_sync_weight_from_policy(self, command):
        self.state.set_weight_synced()

    def dummy_rollout2rollout_broadcast(self, broadcast_command):
        self.current_weight_version = broadcast_command.weight_step
        if broadcast_command.replica_should_stop():
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()

    def dummy(self):
        pass

    def get_rollout_command_handler(cls, command_type):
        if command_type == PolicyToRolloutUnicastCommand:
            return dummy_sync_weight_from_policy
        elif command_type == PolicyToPolicyUnicastCommand:
            return dummy_rollout2rollout_broadcast
        return cls.rollout_command_handler_registry.get_command_handler(command_type)

    DisaggregatedRolloutControlWorker.get_rollout_command_handler = (
        get_rollout_command_handler
    )
    DisaggregatedRolloutControlWorker.prepare_shard_infos_for_weight_sync_insts = dummy

    def dummy_init(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        device: torch.device,
        **kwargs,
    ):
        class Llm_engine:
            def step(self, *args, **kwargs):
                pass

        class Rollout_engine:
            llm_engine = Llm_engine()

        self.rollout_config = config.rollout
        self.rollout_engine = Rollout_engine()
        self.eos_token_ids = [0]
        self._engine_initialized = True

        def rollout_generation(
            self,
            payloads: List[RLPayload],
            stream,
            data_packer,
            data_fetcher,
            is_validation: bool = False,
            *args,
            **kwargs,
        ) -> List[RolloutResult]:
            completions_per_prompt = [
                RolloutResult(
                    prompt=payload.prompt,
                    completions=[payload.prompt] * config.rollout.n_generation,
                )
                for payload in payloads
            ]
            return completions_per_prompt

        self.rollout_generation = types.MethodType(rollout_generation, self)

    vLLMRollout.__init__ = dummy_init
    assert args is not None
    run_rollout(args=args)


def run_policy_parallelism_extract(rank, fsdp, tp, pp):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_dict = load_simple_grpo_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.policy.parallelism.dp_shard_size = fsdp
    config.policy.parallelism.tp_size = tp
    config.policy.parallelism.pp_size = pp
    hf_config = util.retry(AutoConfig.from_pretrained)(
        config.policy.model_name_or_path,
        trust_remote_code=True,
    )
    parallel_dims = ParallelDims.from_config(config.policy.parallelism)
    parallel_dims.build_mesh(device_type="cuda")
    model = ModelRegistry.build_model(config)
    try:
        # Apply parallelism to the model
        parallelize_fn, _ = model.parallelize_fn
        parallelize_fn(model, parallel_dims, config, pp_loss_fn=GRPOTrainer.pp_loss_fn)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e

    mapper = ParallelTopoMapperGroup(
        parallel_dims,
        hf_config=hf_config,
        is_policy=True,
        underlying_model=model,
        weight_mapper=model.weight_mapper,
    )

    assert len(mapper.mapper_group) == 1, "Only one mapper group expected"
    keys_n_ranks = []
    for name, tensor_or_callable in model.weight_sync_transforms:
        if isinstance(tensor_or_callable, torch.Tensor):
            keys_n_ranks.append((name, tensor_or_callable.ndim))
        else:
            tensor_or_callable = tensor_or_callable()
            keys_n_ranks.append((name, tensor_or_callable.ndim))
    hf_key_n_rank = sorted(keys_n_ranks, key=lambda x: x[0])
    local_shard_infos = mapper.prepare_local_shard_infos(hf_key_n_rank, rank)
    all_rank_local_shard_infos = dist_util.all_gather_object_cpu(local_shard_infos)
    if rank == 0:
        config_path = os.path.join(
            cur_dir, "data", f"test_policy_extract_pp_{pp}_fsdp_{fsdp}_tp_{tp}.npy"
        )
        gt = np.load(config_path, allow_pickle=True)
        np.testing.assert_array_equal(
            np.array(all_rank_local_shard_infos, dtype=object), gt
        )


def run_rollout_parallelism_extract(rank, fsdp, tp, pp):
    config_dict = load_simple_grpo_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.rollout.parallelism.tp_size = tp
    config.rollout.parallelism.pp_size = pp

    hf_config = util.retry(AutoConfig.from_pretrained)(
        config.policy.model_name_or_path,
        trust_remote_code=True,
    )
    rollout = vLLMRollout(config, None, torch.cuda.current_device())

    rollout.init_engine(seed=config.rollout.seed, load_format="dummy")
    parallel_dims = ParallelDims.from_config(config.rollout.parallelism)
    parallel_dims.build_mesh(device_type="cuda")

    weight_mapper = WeightMapper.get_weight_mapper(hf_config.model_type)(hf_config)
    mapper = ParallelTopoMapperGroup(
        parallel_dims,
        hf_config=hf_config,
        is_policy=False,
        underlying_model=rollout.get_underlying_model(),
        weight_mapper=weight_mapper,
    )

    assert len(mapper.mapper_group) == 1, "Only one mapper group expected"

    recv_param_key_n_rank_list = []
    _, grouped_recv_param_key_n_rank_list = weight_mapper.cosmos_rollout_prepare_recv(
        rollout.get_underlying_model()
    )
    for group in grouped_recv_param_key_n_rank_list:
        recv_param_key_n_rank_list.extend(group)

    local_shard_infos = mapper.prepare_local_shard_infos(
        recv_param_key_n_rank_list, rank
    )
    all_rank_local_shard_infos = dist_util.all_gather_object_cpu(local_shard_infos)
    if rank == 0:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(
            cur_dir, "data", f"test_rollout_extract_pp_{pp}_fsdp_{fsdp}_tp_{tp}.npy"
        )
        gt = np.load(config_path, allow_pickle=True)
        np.testing.assert_array_equal(
            np.array(all_rank_local_shard_infos, dtype=object), gt
        )


class TestModelType:
    num_hidden_layers = 12
    num_attention_heads = 32
    num_key_value_heads = 32
    hidden_size = 1024
    num_attention_heads = 32
    model_type = "gpt"


async def parallel_map_check():
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
    layers.sort(key=lambda x: x[0])
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
            hf_key_n_rank=layers, global_rank=p_rank
        )
        for p_rank in range(p_world_size)
    ]
    local_shards_r = [
        rollout_mapper.prepare_local_shard_infos(
            hf_key_n_rank=layers, global_rank=r_rank
        )
        for r_rank in range(r_world_size)
    ]

    p_params = [[x[0] for x in layers] for _ in range(p_world_size)]
    r_params = [[x[0] for x in layers] for _ in range(r_world_size)]
    p_body = {
        "shard_infos": local_shards_p,
        "param_groups": [],
        "sorted_params": p_params,
        "trainable_params": [x[0] for x in layers],
    }
    p_data = msgpack.packb(p_body)
    r_body = {
        "shard_infos": local_shards_r,
        "param_groups": [],
        "sorted_params": r_params,
    }
    r_data = msgpack.packb(r_body)

    config_dict = load_simple_grpo_config()
    cosmos_config = CosmosConfig.from_dict(config_dict)
    cosmos_config.policy.parallelism = policy_parallelism_config
    cosmos_config.rollout.parallelism = rollout_parallelism_config
    generator = ParallelizedShardMapper.get_instance(cosmos_config)
    await generator.set_shard_infos_of_policy(p_data, p_world_size)
    await generator.set_shard_infos_of_rollout(r_data, r_world_size)

    await generator.scheme_generation_done.wait()
    global_rank = 5
    # generator.rollout_from_policy_insts_meta = [
    #     [{} for _ in range(generator.rollout_world_size)]
    #     for _ in range(generator.policy_world_size)
    # ]

    insts = await generator.get_send_insts_for_policy(global_rank)
    r_rank_max = 0
    layer_idx = 0

    layers.sort(key=lambda x: x[0])
    insts = msgpack.unpackb(insts, strict_map_key=False)
    for inst_group in insts:
        for inst in inst_group["param_instructions"]:
            dest_name = inst["param_name"]
            for i in inst["instructions"]:
                p_rank = i["policy_rank"]
                r_rank = i["rollout_rank"]
                assert p_rank == global_rank
                while layers[layer_idx][0] != dest_name:
                    r_rank_max = 0
                    layer_idx += 1
                assert layers[layer_idx][0] == dest_name
                assert r_rank >= r_rank_max
                if r_rank > r_rank_max:
                    r_rank_max = r_rank

    global_rank = 2
    insts = await generator.get_recv_insts_for_rollout(global_rank)
    insts = msgpack.unpackb(insts, strict_map_key=False)

    p_rank_max = 0
    layer_idx = 0
    for inst_group in insts:
        for inst in inst_group["param_instructions"]:
            dest_name = inst["param_name"]
            for i in inst["instructions"]:
                p_rank = i["policy_rank"]
                r_rank = i["rollout_rank"]
                assert r_rank == global_rank
                while layers[layer_idx][0] != dest_name:
                    p_rank_max = 0
                    layer_idx += 1
                assert layers[layer_idx][0] == dest_name
                assert p_rank >= p_rank_max
                if p_rank > p_rank_max:
                    p_rank_max = p_rank


def run_sft_for_sequence_packing(fsdp, tp, cp):
    def train_test(sft_worker, packing_seq):
        train_dataset, _ = construct_dataset(
            config,
            data_packer=sft_worker.data_packer,
            user_provided_dataset=None,
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=sft_worker.dp_world_size,
            rank=sft_worker.dp_rank,
            shuffle=False,
            drop_last=False,
        )

        def get_train_data_loader(sampler: Sampler[int]):
            return DataLoader(
                train_dataset,
                batch_size=config.train.train_batch_per_replica,
                shuffle=False,
                num_workers=config.train.train_policy.dataloader_num_workers,
                prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                sampler=sampler,
                collate_fn=collate_fn,
                drop_last=False,
            )

        sft_worker.train_data_loader = get_train_data_loader(train_sampler)
        losses = []
        for global_batch in sft_worker.train_data_loader:
            acc_loss = torch.zeros(1, device=sft_worker.device)
            sft_worker.trainer.optimizers.zero_grad()
            global_batch_size = len(global_batch)
            # split global_batch into mini_batches
            mini_batch_begin_idxs = list(
                range(
                    0,
                    global_batch_size,
                    sft_worker.config.train.train_policy.mini_batch,
                )
            )
            for i in mini_batch_begin_idxs:
                raw_batch = global_batch[
                    i : i + sft_worker.config.train.train_policy.mini_batch
                ]
                max_len = min(
                    sft_worker.config.policy.model_max_length,
                    sft_worker.data_packer.sft_compute_max_len(raw_batch),
                )
                if sft_worker.trainer.seq_len_multiple > 1:
                    max_len = (
                        (max_len + sft_worker.trainer.seq_len_multiple - 1)
                        // sft_worker.trainer.seq_len_multiple
                        * sft_worker.trainer.seq_len_multiple
                    )
                batch = sft_worker.data_packer.sft_collate_fn(
                    raw_batch,
                    computed_max_len=max_len,
                    ignore_label_id=-100,
                )
                sft_worker.trainer.model.train()
                for k, v in batch.items():
                    batch[k] = (
                        v.to(sft_worker.device) if isinstance(v, torch.Tensor) else v
                    )
                labels = batch.pop("label_ids")
                position_ids, input_ids, pos_seq_dim = (
                    sft_worker.trainer.model.get_position_ids(**batch)
                )
                batch["position_ids"] = position_ids
                padding_mask = batch.get("padding_mask", None)
                if packing_seq:
                    # Prepare for the sequence packing information.
                    packed_args = pack_sequences_info_collect(
                        batch["input_ids"],
                        pad_token_id=sft_worker.data_packer.pad_token_id,
                        label_ids=labels,
                        ignore_label_id=-100,
                        seq_len_multiple=sft_worker.trainer.seq_len_multiple,
                    )
                    batch.update(packed_args)
                    labels = pack_sequences_for_labels(labels, batch["valid_input_len"])
                    packed_args = pack_sequences_for_masks(
                        batch["valid_input_len"], batch["valid_input_len"]
                    )
                    batch.update(packed_args)

                if sft_worker.parallel_dims.cp_enabled and not packing_seq:
                    [input_ids, position_ids, padding_mask] = slice_inputs_for_ulysses(
                        [input_ids, position_ids, padding_mask],
                        sft_worker.parallel_dims.mesh["cp"],
                        seq_dims=[1, pos_seq_dim, 1],
                    )
                    batch["input_ids"] = input_ids
                    batch["position_ids"] = position_ids
                    if padding_mask is not None:
                        batch["padding_mask"] = padding_mask

                if sft_worker.parallel_dims.cp_enabled and packing_seq:
                    # Slice for cp after embedding generation and sequence packing in the model forward later.
                    batch["cp_mesh"] = sft_worker.parallel_dims.mesh["cp"]
                logits = sft_worker.trainer.model(**batch)

                loss = sft_worker.trainer.loss_fn(
                    logits,
                    labels,
                    output_packing_mask=batch.get("input_packing_mask", None),
                    target_packing_mask=batch.get("label_packing_mask", None),
                    loss_scaling_factor=1.0 / len(mini_batch_begin_idxs),
                )
                loss.backward()
                acc_loss += loss.detach()
                all_params = [
                    p
                    for m in [
                        model
                        for model in sft_worker.trainer.model_parts
                        if model is not None
                    ]
                    for p in m.parameters()
                ]
                dist_util.gradient_norm_clipping(
                    all_params,
                    sft_worker.config.train.optm_grad_norm_clip,
                    foreach=True,
                    pp_mesh=sft_worker.parallel_dims.mesh["pp"]
                    if sft_worker.parallel_dims.pp_enabled
                    else None,
                    return_norm_only=(
                        sft_worker.config.train.optm_grad_norm_clip <= 0.0
                    ),
                )
                sft_worker.trainer.optimizers.step()
                if sft_worker.trainer.lr_schedulers is None:
                    assert (
                        sft_worker.train_step == 0
                    ), "lr_schedulers should be None if training is from scratch"
                    sft_worker.trainer.lr_schedulers = build_lr_schedulers(
                        sft_worker.trainer.optimizers,
                        sft_worker.config,
                        training_steps=sft_worker.total_steps,
                    )
                sft_worker.trainer.lr_schedulers.step()
                sft_worker.train_step += 1
                if (
                    sft_worker.parallel_dims.dp_replicate_enabled
                    or sft_worker.parallel_dims.dp_shard_enabled
                    or sft_worker.parallel_dims.cp_enabled
                ):
                    global_avg_loss, global_max_loss = (  # noqa: F841
                        dist_util.dist_mean(
                            acc_loss, sft_worker.parallel_dims.mesh["dp_cp"]
                        ),
                        dist_util.dist_max(
                            acc_loss, sft_worker.parallel_dims.mesh["dp_cp"]
                        ),
                    )
                else:
                    global_avg_loss = global_max_loss = acc_loss.item()  # noqa: F841

                if util.is_master_rank(
                    sft_worker.parallel_dims, sft_worker.global_rank
                ):
                    losses.append(global_avg_loss)
                if sft_worker.train_step >= 8:
                    return losses
        return losses

    config_dict = load_simple_sft_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.policy.parallelism.dp_shard_size = fsdp
    config.policy.parallelism.tp_size = tp
    config.policy.parallelism.cp_size = cp
    logger.info(f"[Test] sequence packing with fsdp {fsdp}, tp {tp}, cp {cp}")
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        model_type = hf_config.model_type
        logger.info(f"model type {model_type}")
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        pass

    CommMixin.init_comm = dummy
    sft_worker = SFTPolicyWorker(config=config, parallel_dims=parallel_dims)
    non_packing_losses = train_test(sft_worker, False)
    sft_worker = SFTPolicyWorker(config=config, parallel_dims=parallel_dims)
    packing_losses = train_test(sft_worker, True)
    if util.is_master_rank(sft_worker.parallel_dims, sft_worker.global_rank):
        assert (
            len(non_packing_losses) == 8
        ), f"Expected non-packing losses to be 8, but got {len(non_packing_losses)}"
        assert (
            len(packing_losses) == 8
        ), f"Expected packing losses to be 8, but got {len(packing_losses)}"
        double_actual = torch.tensor(non_packing_losses).double().view(-1)
        double_expected = torch.tensor(packing_losses).double().view(-1)
        cosine_similarity = torch.nn.functional.cosine_similarity(
            double_actual, double_expected, dim=0, eps=1e-5
        )
        assert (
            np.allclose(non_packing_losses, packing_losses, atol=1e-2, rtol=1e-2)
            or cosine_similarity > 0.999
        )


def run_sft_validation():
    config_dict = load_simple_sft_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        model_type = hf_config.model_type
        logger.info(f"model type {model_type}")
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        pass

    CommMixin.init_comm = dummy
    sft_worker = SFTPolicyWorker(config=config, parallel_dims=parallel_dims)
    assert len(sft_worker.val_data_loader) == 29195

    class TestDatasetSFTVal(TestDataset):
        def setup(
            self,
            config: CosmosConfig,
        ):
            dataset = util.load_data_from_disk_or_hf(
                config.validation.dataset.name,
                config.validation.dataset.subset,
                config.validation.dataset.revision or None,
            )
            dataset_list = []
            for split_name in config.validation.dataset.split:
                dataset_list.append(dataset[split_name])
            self.dataset = concatenate_datasets(dataset_list)

        def __len__(self):
            return 1

    # hooks
    pre_validation_hook_mock = Mock()
    pre_per_step_validation_hook_mock = Mock()
    post_per_step_validation_hook_mock = Mock()
    post_validation_hook_mock = Mock()

    def pre_validation_hook(trainer, report_data: Dict[str, Any]):
        pre_validation_hook_mock(trainer, report_data)

    def pre_per_step_validation_hook(trainer, report_data: Dict[str, Any]):
        pre_per_step_validation_hook_mock(trainer, report_data)

    def post_per_step_validation_hook(trainer, report_data: Dict[str, Any]):
        post_per_step_validation_hook_mock(trainer, report_data)

    def post_validation_hook(trainer, report_data: Dict[str, Any]):
        post_validation_hook_mock(trainer, report_data)

    hook_fns = {
        "pre_validation_hook": pre_validation_hook,
        "pre_per_step_validation_hook": pre_per_step_validation_hook,
        "post_per_step_validation_hook": post_per_step_validation_hook,
        "post_validation_hook": post_validation_hook,
    }

    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        val_dataset=TestDatasetSFTVal,
        val_data_packer=DecoderOnlyLLMDataPacker(),
        hook_fns=hook_fns,
    )
    assert len(sft_worker.val_data_loader) == 1
    sft_worker.validate(current_epoch=0, is_last_step=True)
    assert pre_validation_hook_mock.call_count >= 1
    assert pre_per_step_validation_hook_mock.call_count >= 1
    assert post_per_step_validation_hook_mock.call_count >= 1
    assert post_validation_hook_mock.call_count >= 1


def run_reward_check():
    config_dict = load_simple_grpo_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    logger.info(f"Using model from {config.policy.model_name_or_path}")
    # config.rollout.n_generation = 2
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.rollout.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        self.val_data_packer = None
        self.shutdown_signal = threading.Event()
        self.shutdown_mp_signal = mp.Event()  # Must be a multiprocessing event
        self.heartbeat_thread = None

    def report_rollouts(self, block=False):
        while True:
            payloads, is_validation, step, empty = (
                self.reward_dispatcher.dequeue_rewards_cal()
            )
            if not hasattr(self, "_cnt"):
                self._cnt = 0
            if payloads is not None:
                self._cnt += 1
                assert len(payloads) == 1
                assert len(payloads[0].completions) == config.rollout.n_generation
                assert len(payloads[0].rewards) == config.rollout.n_generation
                assert len(payloads[0].advantages) == config.rollout.n_generation
                logger.info(
                    f"Got {payloads[0].rewards} {payloads[0].advantages} from reward calculation at {self._cnt}"
                )
                if is_validation:
                    break
            elif not block or empty:
                break
        if self._cnt >= 1:
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()

        shutdown_signal = dist_util.broadcast_object_cpu(self.shutdown_signal.is_set())
        if shutdown_signal:
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()

        return payloads, is_validation, step, empty

    DisaggregatedRolloutControlWorker.report_rollouts = report_rollouts
    DisaggregatedRolloutControlWorker.send_end_signal = lambda self: None

    def consume_command(
        self,
        cmd_pred=None,
    ):
        pass

    CommMixin.init_comm = dummy
    CommMixin.init_redis = lambda self: None
    DisaggregatedRolloutControlWorker.prepare_shard_infos_for_weight_sync_insts = (
        lambda self: None
    )
    DisaggregatedRolloutControlWorker.consume_command = consume_command
    rollout = DisaggregatedRolloutControlWorker(config, parallel_dims=parallel_dims)

    class TestDatasetReward(TestDataset):
        def __len__(self):
            return 1

    def custom_reward_fn(to_be_evaluated, reference, *args, **kwargs) -> float:
        assert isinstance(reference, str), "Reference answer should be a string"
        reward = boxed_math_reward_fn(to_be_evaluated, reference, *args, **kwargs)
        # Add more reward functions here
        # ...
        return reward

    rollout.setup(
        dataset=TestDatasetReward,
        reward_fns=[custom_reward_fn],
    )
    rollout.lazy_initialize_rollout_engine("auto")

    dataset = TestDatasetReward(config)
    dataset.setup(config)
    for idx in range(len(dataset)):
        prompts = [
            RLPayload(
                prompt=dataset[idx]["prompt"],
                prompt_idx=idx,
                reference_answer=dataset[idx]["result"],
            ),
        ]
        rollout._prompt_queue.put(prompts)

    rollout.state.set_weight_synced()
    rollout.state.set_prompt_fetch_end()
    rollout.main_loop()
    rollout.handle_shutdown()


def run_sft_custom_sampler():
    config_dict = load_simple_sft_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.train.train_policy.dataloader_shuffle = False
    config.validation.enable = True
    config.validation.freq = 1
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        model_type = hf_config.model_type
        logger.info(f"model type {model_type}")
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        pass

    CommMixin.init_comm = dummy

    class TestDatasetSampler(TestDataset):
        def setup(
            self,
            config: CosmosConfig,
        ):
            dataset = util.load_data_from_disk_or_hf(
                config.validation.dataset.name,
                config.validation.dataset.subset,
                config.validation.dataset.revision or None,
            )
            dataset_list = []
            for split_name in config.validation.dataset.split:
                dataset_list.append(dataset[split_name])
            self.dataset = concatenate_datasets(dataset_list)

        def __getitem__(self, idx):
            return super().__getitem__(idx)["conversation"]

        def __len__(self):
            return 16

    class TestSampler(Sampler[int]):
        def __init__(
            self,
            dataset: Dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = False,
        ):
            self.base = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )

        def __iter__(self):
            it = iter(self.base)
            dp_rank = dist.get_rank() // 2
            if not hasattr(self, "checked"):
                cnt = 0
                for i in it:
                    assert (i - dp_rank) % 2 == 0
                    cnt += 1
                assert cnt == 8
                self.checked = True
            it = iter(self.base)
            return it

        def __len__(self) -> int:
            base_len = len(self.base)
            return base_len

        def set_epoch(self, epoch: int):
            self.base.set_epoch(epoch)

    dataset = TestDatasetSampler(config)
    dataset.setup(config)

    dp_rank, dp_world_size = 0, 1
    if parallel_dims.dp_enabled:
        dp_rank = parallel_dims.mesh["dp"].get_local_rank()
        dp_world_size = parallel_dims.mesh["dp"].size()

    test_sampler = TestSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=False,
        drop_last=False,
    )

    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        dataset=dataset,
        val_dataset=dataset,
        val_data_packer=DecoderOnlyLLMDataPacker(),
        sampler=test_sampler,
        val_sampler=test_sampler,
    )
    cnt = 0
    for it in sft_worker.train_data_loader:
        assert len(it) == 8
        cnt += 1
    assert cnt == 1
    cnt = 0
    for it in sft_worker.val_data_loader:
        assert len(it) == 1
        cnt += 1
    assert cnt == 8

    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        dataset=dataset,
        val_dataset=dataset,
        val_data_packer=DecoderOnlyLLMDataPacker(),
        sampler=TestSampler,
        val_sampler=TestSampler,
    )
    cnt = 0
    for it in sft_worker.train_data_loader:
        assert len(it) == 8
        cnt += 1
    assert cnt == 1
    cnt = 0
    for it in sft_worker.val_data_loader:
        assert len(it) == 1
        cnt += 1
    assert cnt == 8

    batch_sampler = BatchSampler(
        test_sampler,
        batch_size=config.train.train_batch_per_replica,
        drop_last=False,
    )
    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        dataset=dataset,
        val_dataset=dataset,
        val_data_packer=DecoderOnlyLLMDataPacker(),
        batch_sampler=batch_sampler,
        val_batch_sampler=batch_sampler,
    )
    cnt = 0
    for it in sft_worker.train_data_loader:
        assert len(it) == 8
        cnt += 1
    assert cnt == 1
    cnt = 0
    for it in sft_worker.val_data_loader:
        assert len(it) == 8
        cnt += 1
    assert cnt == 1

    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        dataset=dataset,
        val_dataset=dataset,
        val_data_packer=DecoderOnlyLLMDataPacker(),
        sampler=TestSampler,
        val_sampler=TestSampler,
        batch_sampler=BatchSampler,
        val_batch_sampler=BatchSampler,
    )
    cnt = 0
    for it in sft_worker.train_data_loader:
        assert len(it) == 8
        cnt += 1
    assert cnt == 1
    cnt = 0
    for it in sft_worker.val_data_loader:
        assert len(it) == 1
        cnt += 1
    assert cnt == 8

    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        dataset=dataset,
        val_dataset=dataset,
        val_data_packer=DecoderOnlyLLMDataPacker(),
        sampler=test_sampler,
        val_sampler=test_sampler,
        batch_sampler=BatchSampler,
        val_batch_sampler=BatchSampler,
    )
    cnt = 0
    for it in sft_worker.train_data_loader:
        assert len(it) == 8
        cnt += 1
    assert cnt == 1
    cnt = 0
    for it in sft_worker.val_data_loader:
        assert len(it) == 1
        cnt += 1
    assert cnt == 8


def run_gspo_test():
    config_dict = load_simple_grpo_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.train.train_policy.variant = "gspo"
    config.logging.logger = ["console"]
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        model_type = hf_config.model_type
        logger.info(f"model type {model_type}")
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        self.shutdown_signal = threading.Event()
        self.shutdown_mp_signal = mp.Event()  # Must be a multiprocessing event
        pass

    CommMixin.init_comm = dummy
    CommMixin.init_redis = lambda self: None
    RLPolicyWorker.prepare_shard_infos_for_weight_sync_insts = lambda self: None

    rl_worker = RLPolicyWorker(config=config, parallel_dims=parallel_dims)
    rl_worker.replica_batch_for_this_step = 8
    rl_worker.inter_policy_nccl.is_single_peer.set()
    rl_worker.inter_policy_nccl.is_comm_ready.set()
    total_steps = 8
    dataset = TestDataset(config)
    dataset.setup(config=config)
    length = []
    for i in range(total_steps * rl_worker.replica_batch_for_this_step):
        index = i % len(dataset)
        prompt = dataset[index][config.train.train_policy.prompt_column_name]
        completion = dataset[index][config.train.train_policy.response_column_name]
        completion_ids = rl_worker.trainer.tokenizer(
            completion, add_special_tokens=False
        ).input_ids
        if (
            i % 2 == 0
            and rl_worker.global_rank // 2 == 0
            or i % 2 == 1
            and rl_worker.global_rank // 2 == 1
        ):
            length.append(len(completion_ids))
        rollout = Rollout(
            prompt=prompt,
            completion=completion,
            advantage=0.05 * (i % 20),
            prompt_idx=index,
        )
        rl_worker.data_queue.put(rollout)

    def hooked_execute_all_reduce(self, inter_policy_nccl):
        ret = GRPOTrainer.all_reduce_states(self, inter_policy_nccl)
        if not hasattr(self, "test_hooked_cnt"):
            self.test_hooked_cnt = 0
        for old in self.old_per_token_logps:
            assert (
                old.shape[0]
                == length[self.test_hooked_cnt] + length[self.test_hooked_cnt + 1]
            )
            self.test_hooked_cnt += 2
        return ret

    rl_worker.trainer.all_reduce_states = types.MethodType(
        hooked_execute_all_reduce, rl_worker.trainer
    )
    rl_worker.total_steps = total_steps

    for i in range(total_steps):
        rollouts = rl_worker.dispatch_rollouts()
        report = rl_worker.trainer.step_training(
            rollouts=rollouts,
            current_step=i,
            total_steps=total_steps,
            remain_samples_num=-1,
            do_save_checkpoint=False,
            inter_policy_nccl=rl_worker.inter_policy_nccl,
            is_master_replica=rl_worker.is_master_replica,
        )
        if rl_worker.global_rank == 0:
            logger.info(f"Step {i} report {report['train/loss_avg']}")
            assert (
                report["train/loss_avg"] < -0.1 and report["train/loss_avg"] > -0.8
            ), f"Expected loss avg to be between -0.1 and -0.8, but got {report['train/loss_avg']} for step {i}"
    rl_worker.handle_shutdown()
    destroy_distributed()


def run_reference_reset_test():
    config_dict = load_simple_grpo_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.logging.logger = ["console"]
    config.train.train_policy.kl_beta = 100
    config.train.train_policy.reference_reset_interval = 2
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        model_type = hf_config.model_type
        logger.info(f"model type {model_type}")
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        self.shutdown_signal = threading.Event()
        self.shutdown_mp_signal = mp.Event()  # Must be a multiprocessing event

    CommMixin.init_comm = dummy
    CommMixin.init_redis = lambda self: None
    RLPolicyWorker.prepare_shard_infos_for_weight_sync_insts = lambda self: None

    rl_worker = RLPolicyWorker(config=config, parallel_dims=parallel_dims)
    rl_worker.trainer.model_load_from_hf()
    state_dict = rl_worker.trainer.model.state_dict()
    for key, value in state_dict.items():
        rl_worker.trainer.reference_state_dict[key] = value.detach().cpu()

    rl_worker.replica_batch_for_this_step = 8
    rl_worker.inter_policy_nccl.is_single_peer.set()
    rl_worker.inter_policy_nccl.is_comm_ready.set()
    total_steps = 8
    dataset = TestDataset(config)
    dataset.setup(config=config)
    for i in range(total_steps * rl_worker.replica_batch_for_this_step):
        index = i % len(dataset)
        prompt = dataset[index][config.train.train_policy.prompt_column_name]
        completion = dataset[index][config.train.train_policy.response_column_name]
        rollout = Rollout(
            prompt=prompt, completion=completion, advantage=1.0, prompt_idx=index
        )
        rl_worker.data_queue.put(rollout)

    for i in range(total_steps):
        rollouts = rl_worker.dispatch_rollouts()
        report = rl_worker.trainer.step_training(
            rollouts=rollouts,
            current_step=i + 1,
            total_steps=total_steps,
            remain_samples_num=-1,
            do_save_checkpoint=False,
            inter_policy_nccl=rl_worker.inter_policy_nccl,
            is_master_replica=rl_worker.is_master_replica,
        )
        if rl_worker.global_rank == 0:
            logger.info(
                f"Step {i} report train/kl_loss_avg {report['train/kl_loss_avg']}, train/kl_loss_max {report['train/kl_loss_max']}"
            )
            if i % 2 == 0:
                assert report["train/kl_loss_avg"] == 0.0
                assert report["train/kl_loss_max"] == 0.0
            else:
                assert report["train/kl_loss_avg"] > 0.0
                assert report["train/kl_loss_max"] > 0.0
    rl_worker.handle_shutdown()
    destroy_distributed()


def run_dynamic_batchsize_test(
    max_token_len_per_mini_batch: int = 2048, batch_size_per_optimize: int = 2
):
    config_dict = load_simple_grpo_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.logging.logger = ["console"]
    config.train.train_policy.batch_size_per_optimize = 16
    config.train.train_policy.mini_batch = 4
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        self.shutdown_signal = threading.Event()
        self.shutdown_mp_signal = mp.Event()  # Must be a multiprocessing event
        pass

    CommMixin.init_comm = dummy
    CommMixin.init_redis = lambda self: None
    RLPolicyWorker.prepare_shard_infos_for_weight_sync_insts = lambda self: None

    rl_worker = RLPolicyWorker(config=config, parallel_dims=parallel_dims)
    state_dict = rl_worker.trainer.model.state_dict()
    for key, value in state_dict.items():
        rl_worker.trainer.reference_state_dict[key] = value.detach().cpu()

    rl_worker.replica_batch_for_this_step = 8
    rl_worker.inter_policy_nccl.is_single_peer.set()
    rl_worker.inter_policy_nccl.is_comm_ready.set()
    total_steps = 8
    dataset = TestDataset(config)
    dataset.setup(config=config)
    for i in range(total_steps * rl_worker.replica_batch_for_this_step):
        index = i % len(dataset)
        prompt = dataset[index][config.train.train_policy.prompt_column_name]
        completion = dataset[i % len(dataset)][
            config.train.train_policy.response_column_name
        ]
        rollout = Rollout(
            prompt=prompt, completion=completion, advantage=1.0, prompt_idx=index
        )
        rl_worker.data_queue.put(rollout)

    def hooked_execute_all_reduce(self, inter_policy_nccl):
        ret = GRPOTrainer.all_reduce_states(self, inter_policy_nccl)
        self.test_hooked_all_reduce_cnt += 1
        return ret

    def hooked_compute_logprobs(self, minibatch, logits, is_full_logits):
        ret = GRPOTrainer.compute_logprobs(self, minibatch, logits, is_full_logits)
        self.test_hooked_compute_logprobs_cnt += 1
        return ret

    rl_worker.trainer.all_reduce_states = types.MethodType(
        hooked_execute_all_reduce, rl_worker.trainer
    )
    rl_worker.trainer.compute_logprobs = types.MethodType(
        hooked_compute_logprobs, rl_worker.trainer
    )
    rl_worker.trainer.test_hooked_compute_logprobs_cnt = 0
    rl_worker.trainer.test_hooked_all_reduce_cnt = 0
    for i in range(total_steps // 4):
        rl_worker.trainer.step_training(
            rollouts=rl_worker.dispatch_rollouts(),
            current_step=i + 1,
            total_steps=total_steps,
            remain_samples_num=-1,
            do_save_checkpoint=False,
            inter_policy_nccl=rl_worker.inter_policy_nccl,
            is_master_replica=rl_worker.is_master_replica,
        )

    assert rl_worker.trainer.test_hooked_compute_logprobs_cnt == 2
    assert rl_worker.trainer.test_hooked_all_reduce_cnt == 2
    rl_worker.trainer.batch_size_per_optimize = 4
    rl_worker.trainer.mini_batch = 1
    rl_worker.trainer.test_hooked_compute_logprobs_cnt = 0
    rl_worker.trainer.test_hooked_all_reduce_cnt = 0
    for i in range(total_steps // 4, total_steps // 2):
        rl_worker.trainer.step_training(
            rollouts=rl_worker.dispatch_rollouts(),
            current_step=i + 1,
            total_steps=total_steps,
            remain_samples_num=-1,
            do_save_checkpoint=False,
            inter_policy_nccl=rl_worker.inter_policy_nccl,
            is_master_replica=rl_worker.is_master_replica,
        )
    assert rl_worker.trainer.test_hooked_compute_logprobs_cnt == 8
    assert rl_worker.trainer.test_hooked_all_reduce_cnt == 2
    rl_worker.trainer.batch_size_per_optimize = 2
    rl_worker.trainer.test_hooked_compute_logprobs_cnt = 0
    rl_worker.trainer.test_hooked_all_reduce_cnt = 0
    for i in range(total_steps // 2, total_steps * 3 // 4):
        rl_worker.trainer.step_training(
            rollouts=rl_worker.dispatch_rollouts(),
            current_step=i + 1,
            total_steps=total_steps,
            remain_samples_num=-1,
            do_save_checkpoint=False,
            inter_policy_nccl=rl_worker.inter_policy_nccl,
            is_master_replica=rl_worker.is_master_replica,
        )
    assert rl_worker.trainer.test_hooked_compute_logprobs_cnt == 16
    assert rl_worker.trainer.test_hooked_all_reduce_cnt == 4
    rl_worker.trainer.batch_size_per_optimize = 4
    rl_worker.trainer.config.train.train_policy.max_token_len_per_mini_batch = 4096
    rl_worker.trainer.test_hooked_compute_logprobs_cnt = 0
    rl_worker.trainer.test_hooked_all_reduce_cnt = 0
    for i in range(total_steps * 3 // 4, total_steps * 7 // 8):
        rl_worker.trainer.step_training(
            rollouts=rl_worker.dispatch_rollouts(),
            current_step=i + 1,
            total_steps=total_steps,
            remain_samples_num=-1,
            do_save_checkpoint=False,
            inter_policy_nccl=rl_worker.inter_policy_nccl,
            is_master_replica=rl_worker.is_master_replica,
        )
    assert rl_worker.trainer.test_hooked_all_reduce_cnt == 1
    assert rl_worker.trainer.test_hooked_compute_logprobs_cnt == 1
    rl_worker.trainer.batch_size_per_optimize = 4
    rl_worker.trainer.config.train.train_policy.max_token_len_per_mini_batch = 2048
    rl_worker.trainer.test_hooked_compute_logprobs_cnt = 0
    rl_worker.trainer.test_hooked_all_reduce_cnt = 0
    for i in range(total_steps * 7 // 8, total_steps):
        rl_worker.trainer.step_training(
            rollouts=rl_worker.dispatch_rollouts(),
            current_step=i + 1,
            total_steps=total_steps,
            remain_samples_num=-1,
            do_save_checkpoint=False,
            inter_policy_nccl=rl_worker.inter_policy_nccl,
            is_master_replica=rl_worker.is_master_replica,
        )
    assert rl_worker.trainer.test_hooked_all_reduce_cnt == 1
    assert rl_worker.trainer.test_hooked_compute_logprobs_cnt == 2

    rl_worker.handle_shutdown()
    destroy_distributed()


def run_sft_ddp_load_check():
    config_dict = load_simple_sft_config()
    config = CosmosConfig.from_dict(
        config_dict,
    )
    config.policy.parallelism.dp_replicate_size = 2
    config.policy.parallelism.tp_size = 1
    config.policy.parallelism.dp_shard_size = 2
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    def dummy(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        self.data_packer = DecoderOnlyLLMDataPacker()
        self.data_packer.setup(self.config)
        pass

    CommMixin.init_comm = dummy

    def dummy_sync_all_states(
        self,
        is_send: bool,
        send_hook: callable,
        recv_hook: callable,
        reference_model: bool = False,
    ):
        if not hasattr(self, "sync_cnt"):
            self.sync_cnt = 0

        def offload_state_dict_cpu(state_dict: dict):
            state_dict_cpu = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.distributed.tensor.DTensor):
                    state_dict_cpu[key] = value.to_local().cpu()
                elif isinstance(value, torch.Tensor):
                    state_dict_cpu[key] = value.cpu()
            return state_dict_cpu

        if self.parallel_dims.dp_replicate_coord[0] != 0:
            pre_model_state_dict_cpu = offload_state_dict_cpu(self.model.state_dict())
            pre_model_state_dict_cpu_for_load = (
                self.ckpt_manager.offload_state_dict_cpu(self.model.state_dict())
            )
            pre_optimizer_state_dict_cpu = offload_state_dict_cpu(
                self.optimizers.state_dict()
            )
            self.model.load_hf_weights(
                self.config.policy.model_name_or_path,
                self.parallel_dims,
                self.device,
                revision=self.config.policy.model_revision,
            )
            gt_model_state_dict_cpu = offload_state_dict_cpu(self.model.state_dict())
            self.model.load_state_dict(pre_model_state_dict_cpu_for_load)
        ret = self.original_sync_all_states(
            is_send, send_hook, recv_hook, reference_model
        )
        if self.parallel_dims.dp_replicate_coord[0] != 0:
            new_model_state_dict_cpu = offload_state_dict_cpu(self.model.state_dict())
            new_optimizer_state_dict_cpu = offload_state_dict_cpu(
                self.optimizers.state_dict()
            )
            results1 = []
            results2 = []
            cnt = 0
            for key in pre_model_state_dict_cpu.keys():
                results1.append(
                    torch.allclose(
                        gt_model_state_dict_cpu[key], new_model_state_dict_cpu[key]
                    )
                )
                results2.append(
                    torch.allclose(
                        pre_model_state_dict_cpu[key], new_model_state_dict_cpu[key]
                    )
                )
                cnt += 1
                if cnt > 10:
                    break
            assert all(
                results1
            ), "DDP load failed for some model parameters, they should match hf loaded weights"
            assert not all(
                results2
            ), "DDP load failed for some model parameters, they should be different after load"
            results1 = []
            cnt = 0
            for key in pre_optimizer_state_dict_cpu.keys():
                results1.append(
                    torch.allclose(
                        pre_optimizer_state_dict_cpu[key],
                        new_optimizer_state_dict_cpu[key],
                    )
                )
                cnt += 1
                if cnt > 10:
                    break
            assert all(results1), "DDP load failed for some optimizer parameters"
        self.sync_cnt += 1
        return ret

    LLMTrainer.original_sync_all_states = LLMTrainer.sync_all_states
    LLMTrainer.sync_all_states = dummy_sync_all_states
    sft_worker = SFTPolicyWorker(
        config=config, parallel_dims=parallel_dims, dataset=TestDataset
    )

    assert sft_worker.trainer.sync_cnt == 1
    destroy_distributed()


async def main():
    # Get shared memory name and size from command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shm_name", type=str)  # 1st arg
    parser.add_argument("--shm_size", type=int)  # 2nd arg
    parser.add_argument("--mode", type=str, required=True)  # 3rd arg
    parser.add_argument(
        "--parallel_config",
        type=str,
        required=False,
        default="",
        help="Parallel dimensions to use for the test. Format: fsdp;tp;pp",
    )
    parser.add_argument(
        "--trainable_param_sync",
        type=bool,
        required=False,
        default=False,
        help="If only trainable params are synced. If set, part of the params will be frozen.",
    )  # 4th arg
    args = parser.parse_args()
    mode = args.mode
    shm_name = args.shm_name
    shm_size = args.shm_size
    mode = args.mode
    trainable_param_sync = (
        args.trainable_param_sync
    )  # If only trainable params are synced

    if mode == "dummy_policy":
        os.environ["COSMOS_ROLE"] = "Policy"
        # Dummy policy process for testing
        run_dummy_policy(args=args)
        exit(0)
    elif mode == "dummy_rollout":
        os.environ["COSMOS_ROLE"] = "Rollout"
        # Dummy rollout process for testing
        run_dummy_rollout(args=args)
        exit(0)
    elif mode == "test_overfit":
        run_overfitting_policy(args=args)
        exit(0)
    elif mode == "sft_for_sequence_packing":
        sepc = args.parallel_config
        fsdp, tp, cp = sepc.split(";")
        fsdp = int(fsdp.split(":")[1])
        tp = int(tp.split(":")[1])
        cp = int(cp.split(":")[1])
        run_sft_for_sequence_packing(fsdp, tp, cp)
        exit(0)
    elif mode == "sft_for_validation":
        run_sft_validation()
        exit(0)
    elif mode == "sft_for_custom_sampler":
        run_sft_custom_sampler()
        exit(0)
    elif mode == "reward_execution_check":
        run_reward_check()
        exit(0)
    elif mode == "gspo_test":
        run_gspo_test()
        exit(0)
    elif mode == "reference_reset_test":
        run_reference_reset_test()
        exit(0)
    elif mode == "dynamic_batchsize_test":
        run_dynamic_batchsize_test()
        exit(0)
    elif mode == "sft_ddp_load_check":
        run_sft_ddp_load_check()
        exit(0)

    # Initialize distributed environment
    init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        rank = 0
        world_size = 1
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    print(f"Rank {rank} started with mode {mode} {torch.cuda.current_device()}")

    if mode == "policy_send_to_rollout":
        assert (
            world_size == POLICY_WORLD_SIZE
        ), "World size must match POLICY_WORLD_SIZE for policy process"
        await run_policy_send_to_rollout(shm_name, shm_size, rank, trainable_param_sync)
    elif mode == "rollout_recv_from_policy":
        assert (
            world_size == ROLLOUT_WORLD_SIZE
        ), "World size must match ROLLOUT_WORLD_SIZE for rollout process"
        await run_rollout_recv_from_policy(
            shm_name, shm_size, rank, trainable_param_sync
        )
    elif mode == "policy_send_to_policy":
        run_policy_unicast_to_policy(shm_name, shm_size, rank, True)
    elif mode == "policy_recv_from_policy":
        run_policy_unicast_to_policy(shm_name, shm_size, rank, False)
    elif mode.startswith("policy_broadcast_to_policy"):
        total_rep = int(mode.split(",")[1])
        self_rep = int(mode.split(",")[2])
        run_policy_broadcast_to_policy(shm_name, shm_size, rank, total_rep, self_rep)
    elif mode == "policy_parallelism_extract":
        sepc = args.parallel_config
        fsdp, tp, pp = sepc.split(";")
        fsdp = int(fsdp.split(":")[1])
        tp = int(tp.split(":")[1])
        pp = int(pp.split(":")[1])
        run_policy_parallelism_extract(rank, fsdp, tp, pp)
    elif mode == "rollout_parallelism_extract":
        sepc = args.parallel_config
        fsdp, tp, pp = sepc.split(";")
        fsdp = int(fsdp.split(":")[1])
        tp = int(tp.split(":")[1])
        pp = int(pp.split(":")[1])
        run_rollout_parallelism_extract(rank, fsdp, tp, pp)
    elif mode == "parallel_map_check":
        await parallel_map_check()
    else:
        raise ValueError("Invalid mode.")
    # Clean up distributed environment
    destroy_distributed()


if __name__ == "__main__":
    asyncio.run(main())
