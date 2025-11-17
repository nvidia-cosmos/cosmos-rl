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
import os
import subprocess
import sys
from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
    DecoderOnlyLLMDataPacker,
)
import uuid
import toml
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.utils import util
from cosmos_rl.utils.payload import extract_rollouts
from datasets import concatenate_datasets


class TestCustomSampler(unittest.TestCase):
    def test_custom_sampler(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 4
        # Create the Python command for torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--mode",
            "sft_for_custom_sampler",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
        )
        processes = [process]

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert (
                process.returncode == 0
            ), f"Process failed with code: {process.returncode}"


class TestCustomRolloutOutput(unittest.TestCase):
    def test_custom_rollout(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_grpo.toml",
        )
        with open(config_path, "r") as f:
            config_dict = toml.load(f)
        config = CosmosConfig.from_dict(
            config_dict,
        )
        config.train.train_policy.dataset.name = os.path.join(
            cur_dir, config.train.train_policy.dataset.name
        )

        class TestDataPacker(DecoderOnlyLLMDataPacker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.kv_store = {}

            def get_policy_input(self, item, rollout_output, n_ignore_prefix_tokens=0):
                id = rollout_output
                rollout_output = self.kv_store.pop(id, None)
                return rollout_output

            def get_rollout_output(
                self,
                items,
                completed_conversations=None,
                logprobs=None,
                token_ids=None,
                **kwargs,
            ):
                uuids = []
                if not items:
                    return items
                if all([not i for i in items]):
                    return items
                for item in items:
                    id = uuid.uuid4()
                    self.kv_store[str(id)] = item
                    uuids.append(str(id))
                return uuids, completed_conversations, logprobs, token_ids, kwargs

        data_packer = TestDataPacker()

        dataset = util.load_data_from_disk_or_hf(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
            config.train.train_policy.dataset.revision or None,
        )
        dataset_list = []
        for split_name in config.train.train_policy.dataset.split:
            dataset_list.append(dataset[split_name])
        dataset = concatenate_datasets(dataset_list)

        payloads = []
        for i in range(1):
            payloads.append(
                RLPayload(
                    prompt=dataset[i]["prompt"],
                    prompt_idx=0,  # Mock the prompt index
                    completions=[dataset[i]["result"] for _ in range(16)],
                    completed_conversations=[[]] * 16,
                    rewards=[0.5] * 16,
                    advantages=[0.5] * 16,
                    filter_rewards=[0.5] * 16,
                    n_ignore_prefix_tokens=[0] * 16,
                    valid=True,
                )
            )
        if payloads is not None:
            for i in range(len(payloads)):
                (
                    payloads[i].completions,
                    payloads[i].completed_conversations,
                    _,
                    _,
                    _,
                ) = data_packer.get_rollout_output(
                    payloads[i].completions,
                    payloads[i].completed_conversations,
                )

        valid_rollouts_list = extract_rollouts(payloads, False)

        assert len(valid_rollouts_list) == 1
        assert len(valid_rollouts_list[0]) == 16

        rollouts = [r for rs in valid_rollouts_list for r in rs]

        samples = [rollout.prompt for rollout in rollouts]

        completions_list = [rollout.completion for rollout in rollouts]
        n_ignore_prefix_tokens_list = [
            rollout.n_ignore_prefix_tokens for rollout in rollouts
        ]
        data_packer.config = config
        processed_samples = [
            data_packer.get_policy_input(
                samples[i],
                completions_list[i],
                n_ignore_prefix_tokens_list[i],
            )
            for i in range(len(samples))
        ]
        for sample in processed_samples:
            assert sample is not None
            assert sample == dataset[0]["result"]


if __name__ == "__main__":
    unittest.main()
