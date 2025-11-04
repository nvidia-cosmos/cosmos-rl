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
import threading
import uuid
import functools
from typing import Optional, Tuple, List, Any, Dict

import datasets
from transformers import AutoTokenizer
from vllm import SamplingParams


from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.schema import ChatMessage
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout.vllm_rollout.vllm_rollout_async import vLLMRolloutAsync
from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
    DecoderOnlyLLMDataPacker,
    DataPacker,
)
from cosmos_rl.dispatcher.protocol import RolloutRequest, ValidationReportRequest
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed


def override_environment(port: int = 29500) -> dict[str, str]:
    old_env = os.environ.copy()
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"
    return old_env


class MockAPIClient(APIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = getMockConfig()

        # initialize the data_packer
        tokenizer = AutoTokenizer.from_pretrained(self.config.policy.model_name_or_path)
        data_packer = DecoderOnlyLLMDataPacker()
        data_packer.setup(config=self.config, tokenizer=tokenizer)
        self.data_packer = data_packer

        # load test dataset
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset = datasets.load_from_disk(
            os.path.join(cur_dir, "data_fixtures", "sharegpt52k_small")
        )["train"]
        self.data_iter = iter(self.dataset)
        self.cur_epoch = 0

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
        def _collect_batch(self):
            batch = []
            for i in range(batch_size):
                dat = next(self.data_iter)
                conversation = dat["conversation"]
                prompt = self.data_packer.get_rollout_input(conversation)
                payload = RLPayload(prompt=prompt, conversation=conversation)
                batch.append((i, payload))
            return batch

        try:
            batch = _collect_batch(self)
        except StopIteration:
            self.data_iter = iter(self.dataset)
            batch = _collect_batch(self)
            self.cur_epoch += 1

        return batch, self.cur_epoch == 2

    def post_rollout_completion(self, response: RolloutRequest):
        logger.info(f"[MockAPIClient] Post rollout completion: {response}")

    def post_validation_report(self, report: ValidationReportRequest):
        logger.info(f"[MockAPIClient] Post validation report: {report}")


def getMockConfig():
    # Construct the model and trainer
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cur_dir, "configs", "test_simple_grpo.toml")

    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    return CosmosConfig.from_dict(config_dict)


class TestVLLMRolloutAsync(unittest.TestCase):
    """Test vLLMRolloutAsync."""

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
        # initialize rollout engine
        rollout_engine = vLLMRolloutAsync(config, tokenizer)
        rollout_engine.init_engine(quantization="none", seed=42, load_format="auto")

        # create data packer
        data_packer = DecoderOnlyLLMDataPacker()
        data_packer.setup(config=config, tokenizer=tokenizer)
        return rollout_engine, data_packer

    def test_async_rollout_single_generate(self):
        """Test async rollout."""
        cosmos_config = getMockConfig()

        # force try tp1, pp1
        cosmos_config.rollout.parallelism.tp_size = 1

        rollout_engine, data_packer = self.get_rollout_engine_and_data_packer(
            cosmos_config
        )

        payloads = [
            RLPayload(prompt="What is 2+2?", weight_version=0),
            # RLPayload(prompt="Explain AI in one sentence.", weight_version=0),
        ]

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            n=2,  # 2 responses for each prompt
        )

        results = asyncio.run(
            rollout_engine.rollout_generation(
                payloads=payloads,
                stream=None,
                data_packer=data_packer,
                sampling_params=sampling_params,
            )
        )

        rollout_engine.get_engine().shutdown()

        # check results
        self.assertEqual(len(results), len(payloads))
        for i, result in enumerate(results):
            print(f"Result {i}: {result}")

    def test_async_rollout_multi_turn_generate(self):
        """Test async rollout multi turn."""
        cosmos_config = getMockConfig()
        cosmos_config.rollout.multi_turn_config.enable = True
        cosmos_config.rollout.multi_turn_config.max_assistant_turns = 2
        cosmos_config.rollout.multi_turn_config.enable_thinking = True

        # force try tp1, pp1
        cosmos_config.rollout.parallelism.tp_size = 1

        rollout_engine, data_packer = self.get_rollout_engine_and_data_packer(
            cosmos_config
        )

        payloads = [
            RLPayload(
                conversation=[ChatMessage(role="user", content="What is 2+2?")],
                weight_version=0,
            ),
        ]

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            n=2,  # 2 responses for each prompt
        )

        results = asyncio.run(
            rollout_engine.rollout_generation(
                payloads=payloads,
                stream=None,
                data_packer=data_packer,
                sampling_params=sampling_params,
            )
        )

        rollout_engine.get_engine().shutdown()

        # check results
        self.assertEqual(len(results), len(payloads))
        for i, result in enumerate(results):
            print(f"Result {i}: {result}")

    def test_async_rollout_get_underlying_model_state_dict(self):
        """Test async rollout get underlying model state dict."""
        cosmos_config = getMockConfig()
        cosmos_config.rollout.parallelism.tp_size = 1
        rollout_engine, _ = self.get_rollout_engine_and_data_packer(cosmos_config)
        state_dict = asyncio.run(rollout_engine.get_underlying_model_state_dict())
        print(f"State dict: {state_dict}")

        self.assertIn("model.layers.0.self_attn.q_proj.weight", state_dict)
        self.assertGreater(
            state_dict["model.layers.0.self_attn.q_proj.weight"].sum(), 0
        )
        rollout_engine.shutdown()


class TestVLLMRolloutWorkerAsync(unittest.TestCase):
    """Test vLLMRolloutWorkerAsync."""

    def setUp(self):
        self.old_env = override_environment(port=29501)
        init_distributed()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        destroy_distributed()

    def test_async_rollout_worker_1gpu(self):
        """Test async rollout worker."""
        cosmos_config = getMockConfig()
        cosmos_config.rollout.parallelism.dp_shard_size = 1
        cosmos_config.rollout.parallelism.tp_size = 1
        cosmos_config.rollout.parallelism.pp_size = 1

        parallel_dims = ParallelDims.from_config(cosmos_config.rollout.parallelism)

        from cosmos_rl.rollout.vllm_rollout.vllm_rollout_worker_async import (
            vLLMRolloutWorkerAsync,
        )

        # here dummy some functions to make the worker work
        def dummy_init_comm(self):
            self.api_client = MockAPIClient(
                role="ROLLOUT", remote_ips=["localhost"], remote_port=8000
            )
            self.data_packer = DecoderOnlyLLMDataPacker()
            self.val_data_packer = self.data_packer

        def dummy(self):
            pass

        vLLMRolloutWorkerAsync.init_comm = dummy_init_comm
        vLLMRolloutWorkerAsync.init_redis = dummy

        worker = vLLMRolloutWorkerAsync(cosmos_config, parallel_dims)
        worker.query_command_from_controller = functools.partial(dummy, worker)
        worker.replica_name = str(uuid.uuid4())
        worker.shutdown_signal = threading.Event()
        worker.shutdown_mp_signal = threading.Event()
        worker.heartbeat_thread = None
        # Skip weight sync preparation in test since we don't need it
        worker.lazy_initialize_rollout_engine(load_format="auto")

        worker.state.set_weight_synced()
        worker.setup()
        worker.work()

        # clean the test environment
        worker.handle_shutdown()


if __name__ == "__main__":
    unittest.main()
