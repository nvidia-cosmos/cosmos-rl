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
import unittest
import asyncio
import toml
import torch
import threading
import uuid
import functools
from typing import Optional, Tuple, List, Any, Dict
import numpy as np
from transformers import AutoTokenizer

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout.vllm_rollout.vllm_rollout_async import vLLMRolloutAsync
from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
    DecoderOnlyLLMDataPacker,
    DataPacker,
)
from cosmos_rl.dispatcher.protocol import RolloutRequest, ValidationReportRequest
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed
from cosmos_rl.utils import async_utils
from cosmos_rl.reward.reward_calculator import RewardCalculator


def override_environment(port: int = 29500) -> dict[str, str]:
    async_utils.unsafe_enable_nest_asyncio()

    old_env = os.environ.copy()
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return old_env


class MockAPIClient(APIClient):
    def __init__(self, config: CosmosConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        # load test dataset
        self.data_fetcher = ControllerDataFetcher(
            config=self.config,
            dataset=None,
            val_dataset=None,
            is_rl=True,
        )
        self.max_iter = 2
        self.cur_iter = 0

        # rollout_completion_payloads cache 1 batch of payloads for testing
        self.rollout_completion_payloads: List[RLPayload] = []
        self.validation_completion_payloads: List[RLPayload] = []

    def post_rollout_shard_info(
        self,
        shard_infos: List[Dict[str, Any]],
        param_groups: List[List[str]],
        sorted_params: List[List[str]],
    ):
        pass

    def register(
        self,
        replica_name: str,
        role: str,
        mesh_names: List[str],
        ranks: List[int],
        group_size: int,
        global_rank: int,
        host_ip: str,
        host_name: str,
    ):
        logger.info(
            f"[MockAPIClient] Register: {replica_name}, {role}, {mesh_names}, {ranks}, {group_size}, {global_rank}, {host_ip}, {host_name}"
        )

    def unregister(self, replica_name: str):
        logger.info(f"[MockAPIClient] Unregister: {replica_name}")

    def get_next_prompt(
        self, batch_size: int, validation_step: Optional[int] = None
    ) -> Tuple[List[Tuple[int, str]], bool]:
        # masked validation_step for testing
        validation_step = None
        payloads_list, is_end = self.data_fetcher.get_batched_prompt(
            batch_size, validation_step
        )
        self.cur_iter += 1

        payloads_list = [pl.model_dump() for pl in payloads_list]
        return payloads_list, is_end or self.cur_iter == self.max_iter

    def post_rollout_completion(self, response: RolloutRequest):
        logger.info(
            f"[MockAPIClient] Post rollout completion: {len(response.payloads)} results"
        )
        self.rollout_completion_payloads.extend(response.payloads)

    def post_validation_report(self, report: ValidationReportRequest):
        logger.info(
            f"[MockAPIClient] Post validation report: {len(report.payloads)} results"
        )
        self.validation_completion_payloads.extend(report.payloads)


class MockDeviceMesh:
    def __init__(self, mesh: List[int], mesh_dim_names: List[str] = None):
        self.mesh = np.array(mesh)
        self.mesh_dim_names = mesh_dim_names
        self._mesh_name_to_mesh = {dn: d for dn, d in zip(mesh_dim_names, mesh)}

    def size(self):
        return np.prod(self.mesh)

    def __getitem__(self, mesh_dim_name: str) -> "MockDeviceMesh":
        submesh = [self._mesh_name_to_mesh[mesh_dim_name]]
        return MockDeviceMesh(mesh=submesh, mesh_dim_names=[mesh_dim_name])


def getMockConfig():
    # Construct the model and trainer
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cur_dir, "configs", "test_simple_grpo.toml")

    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    config = CosmosConfig.from_dict(config_dict)
    config.rollout.async_config.enable = True
    config.rollout.async_config.max_concurrent_requests = 10
    config.rollout.backend = "vllm_async"
    return config


class TestAsyncVLLMRollout(unittest.TestCase):
    """Test AsyncVLLMRollout."""

    def setUp(self):
        self.old_env = override_environment()
        init_distributed()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        destroy_distributed()

    def get_rollout_engine_and_data_packer(
        self, config: CosmosConfig
    ) -> Tuple[vLLMRolloutAsync, DataPacker]:
        # initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.policy.model_name_or_path)
        parallel_dims = ParallelDims.from_config(config.rollout.parallelism)
        # initialize rollout engine
        rollout_engine = vLLMRolloutAsync(
            config, parallel_dims=parallel_dims, device=torch.device("cuda")
        )
        rollout_engine.init_engine(quantization="none", seed=42, load_format="auto")

        # create data packer
        data_packer = DecoderOnlyLLMDataPacker()
        data_packer.setup(config=config, tokenizer=tokenizer)
        return rollout_engine, data_packer

    def test_rollout_engine_rpc(self):
        """Test rollout engine rpc."""
        cosmos_config = getMockConfig()
        cosmos_config.rollout.parallelism.tp_size = 1

        async def test_helper():
            rollout_engine, _ = self.get_rollout_engine_and_data_packer(cosmos_config)
            # Test use sync function to call the async function
            results = asyncio.run(
                rollout_engine.rollout_engine.collective_rpc("get_state_dict_ipc")
            )
            rollout_engine.get_engine().shutdown()
            return results

        results = asyncio.run(test_helper())
        self.assertGreater(len(results), 0)

    def test_async_rollout_single_generate(self):
        """Test async rollout."""
        cosmos_config = getMockConfig()

        # force try tp1, pp1
        cosmos_config.rollout.parallelism.tp_size = 1

        payloads = [
            RLPayload(prompt="What is 2+2?", weight_version=0),
            # RLPayload(prompt="Explain AI in one sentence.", weight_version=0),
        ]

        async def test_helper():
            rollout_engine, data_packer = self.get_rollout_engine_and_data_packer(
                cosmos_config
            )
            results = await rollout_engine.rollout_generation(
                payloads=payloads,
                stream=None,
                data_packer=data_packer,
                data_fetcher=None,
                is_validation=False,
            )
            rollout_engine.get_engine().shutdown()
            return results

        results = asyncio.run(test_helper())

        # check results
        self.assertEqual(len(results), len(payloads))
        for i, result in enumerate(results):
            self.assertEqual(
                len(result.completions), cosmos_config.rollout.n_generation
            )


class TestAsyncRolloutWorker(unittest.TestCase):
    """Test AsyncRolloutWorker."""

    def setUp(self):
        self.old_env = override_environment(port=29501)
        init_distributed()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        destroy_distributed()

        # clean singleton instance of RewardCalculator
        if hasattr(RewardCalculator, "_instance"):
            delattr(RewardCalculator, "_instance")

    def test_async_rollout_worker_1gpu(self):
        """Test async rollout worker."""
        cosmos_config = getMockConfig()
        cosmos_config.rollout.parallelism.dp_shard_size = 1
        cosmos_config.rollout.parallelism.tp_size = 1
        cosmos_config.rollout.parallelism.pp_size = 1
        cosmos_config.rollout.batch_size = 4

        parallel_dims = ParallelDims.from_config(cosmos_config.rollout.parallelism)
        parallel_dims.mesh = MockDeviceMesh(mesh=[1], mesh_dim_names=["dp"])

        from cosmos_rl.rollout.worker.asynchronous.rollout_control import (
            AsyncDisaggregatedRolloutControlWorker,
        )

        # here dummy some functions to make the worker work
        def dummy_init_comm(self):
            self.api_client = MockAPIClient(
                config=cosmos_config,
                role="ROLLOUT",
                remote_ips=["localhost"],
                remote_port=8000,
            )
            self.data_packer = DecoderOnlyLLMDataPacker()
            self.val_data_packer = self.data_packer

        def dummy(self):
            pass

        AsyncDisaggregatedRolloutControlWorker.init_comm = dummy_init_comm
        AsyncDisaggregatedRolloutControlWorker.init_redis = dummy

        worker = AsyncDisaggregatedRolloutControlWorker(cosmos_config, parallel_dims)
        worker.query_command_from_controller = functools.partial(dummy, worker)
        worker.replica_name = str(uuid.uuid4())
        worker.shutdown_signal = threading.Event()
        worker.shutdown_mp_signal = threading.Event()
        worker.heartbeat_thread = None
        # Skip weight sync preparation in test since we don't need it
        worker.state.set_weight_synced()
        worker.work()

        self.assertEqual(
            len(worker.api_client.rollout_completion_payloads),
            cosmos_config.rollout.batch_size * worker.api_client.max_iter,
        )

        # clean the test environment
        worker.handle_shutdown()

    def test_async_rollout_worker_validation(self):
        """Test async rollout worker validation."""
        cosmos_config = getMockConfig()
        cosmos_config.rollout.parallelism.dp_shard_size = 1
        cosmos_config.rollout.parallelism.tp_size = 1
        cosmos_config.rollout.parallelism.pp_size = 1
        cosmos_config.validation.enable = True
        cosmos_config.validation.batch_size = 4
        cosmos_config.validation.dataset = cosmos_config.train.train_policy.dataset

        parallel_dims = ParallelDims.from_config(cosmos_config.rollout.parallelism)
        parallel_dims.mesh = MockDeviceMesh(mesh=[1], mesh_dim_names=["dp"])
        from cosmos_rl.rollout.worker.asynchronous.rollout_control import (
            AsyncDisaggregatedRolloutControlWorker,
        )

        # here dummy some functions to make the worker work
        def dummy_init_comm(self):
            self.api_client = MockAPIClient(
                config=cosmos_config,
                role="ROLLOUT",
                remote_ips=["localhost"],
                remote_port=8000,
            )
            self.data_packer = DecoderOnlyLLMDataPacker()
            self.val_data_packer = self.data_packer

        def dummy(self):
            pass

        AsyncDisaggregatedRolloutControlWorker.init_comm = dummy_init_comm
        AsyncDisaggregatedRolloutControlWorker.init_redis = dummy

        worker = AsyncDisaggregatedRolloutControlWorker(cosmos_config, parallel_dims)
        worker.query_command_from_controller = functools.partial(dummy, worker)
        worker.replica_name = str(uuid.uuid4())
        worker.shutdown_signal = threading.Event()
        worker.shutdown_mp_signal = threading.Event()
        worker.heartbeat_thread = None
        # Skip weight sync preparation in test since we don't need it
        worker.state.set_weight_synced()
        worker.init_scheduler()

        async def test_helper():
            worker.current_step = 1
            worker.lazy_initialize_rollout_engine(load_format="auto")
            try:
                await worker.scheduler.start_async()
                await worker.do_validation()
            finally:
                await worker.scheduler.stop_async()
                worker.handle_shutdown()

        asyncio.run(test_helper())
        self.assertEqual(
            len(worker.api_client.validation_completion_payloads),
            cosmos_config.validation.batch_size * worker.api_client.max_iter,
        )

        # clean the test environment
        worker.handle_shutdown()


if __name__ == "__main__":
    unittest.main()
