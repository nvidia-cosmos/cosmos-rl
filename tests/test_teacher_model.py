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

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.data.data_fetcher import WorkerDataFetcher
from cosmos_rl.dispatcher.data.packer import DecoderOnlyLLMDataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.redis_stream import RedisStreamHandler

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import unittest
import subprocess
import sys
from cosmos_rl.utils import util
import toml
import tempfile
from transformers import AutoTokenizer


class TestTeacherModel(unittest.TestCase):
    def test_teacher_model(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 4
        port = util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_grpo.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)

        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["logger"] = ["console"]
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["epoch"] = 0

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name

        append_config = """
        [distillation]
        enable = true
        model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
        model_max_length = 1024
        compile = true
        batch_size_per_replica = 8

        [distillation.parallelism]
        n_init_replicas = 1
        tp_size = 1
        cp_size = 1
        dp_shard_size = 4
        pp_size = 1
        dp_replicate_size = 1
        """
        with open(tmpfile_toml, "a") as f:
            f.write(append_config)

        controller_cmd = f"{sys.executable} -m cosmos_rl.dispatcher.run_web_panel --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        env_dict = os.environ.copy()
        env_dict["COSMOS_ROLE"] = "Controller"
        subprocess.Popen(
            controller_cmd,
            shell=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        # Create the Python command for torchrun
        reference_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            "-m",
            "cosmos_rl.reference.reference_entry",
            "--config",
            tmpfile_toml,
        ]
        reference_env = dict(os.environ)
        reference_env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        reference_process = subprocess.Popen(
            reference_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=reference_env,
        )
        processes = [reference_process]
        api_client = APIClient(role="POLICY")
        metadata = api_client.get_controller_metadata()

        if metadata["config"] is None:
            raise RuntimeError(
                f"[Policy] Please first go to http://{api_client.remote_ips}:{api_client.remote_port} to configure training parameters."
            )
        cosmos_config = Config.from_dict(metadata["config"])
        logger.info(f"[Reference] Loaded configuration: {cosmos_config.model_dump()}")
        redis_controller = RedisStreamHandler(
            ips=api_client.remote_ips, port=int(cosmos_config.redis)
        )
        data_fetcher = WorkerDataFetcher(
            config=cosmos_config,
            data_packer=DecoderOnlyLLMDataPacker(),
            val_data_packer=DecoderOnlyLLMDataPacker(),
        )
        prompt_idx = 0
        prompt = data_fetcher.get_payload_by_index(prompt_idx)
        reference_answer = data_fetcher.query_reference_answer(prompt_idx)
        tokenizer = AutoTokenizer.from_pretrained(
            cosmos_config.distillation.model_name_or_path, trust_remote_code=True
        )
        tokenizer_prompt = tokenizer(prompt, add_special_tokens=False).input_ids
        tokenizer_reference_answer = tokenizer(
            reference_answer, add_special_tokens=False
        ).input_ids
        data = {
            "prompt_idx": prompt_idx,
            "completion_token_ids": [
                tokenizer_reference_answer
                for _ in range(cosmos_config.rollout.n_generation)
            ],
        }
        uuids = redis_controller.publish_teacher_request(data, "test_client")
        for uuid in uuids:
            teacher_result = redis_controller.get_teacher_result(uuid)
            assert (
                len(teacher_result["teacher_logprobs"])
                == len(tokenizer_prompt) + len(tokenizer_reference_answer)
            ), f"Teacher logprobs must be the same length as the prompt and reference answer, got {len(teacher_result['teacher_logprobs'])} != {len(tokenizer_prompt) + len(tokenizer_reference_answer)}"

        data["is_end"] = True
        redis_controller.publish_teacher_request(data, "test_client")
        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert (
                process.returncode == 0
            ), f"Process failed with code: {process.returncode}"


class TestDistillationFlow(unittest.TestCase):
    def test_distillation_flow(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        port = util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "..",
            "configs",
            "qwen3",
            "qwen3-8b-distill.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)

        config["train"]["epoch"] = 1
        config["rollout"]["parallelism"]["tp_size"] = 2
        config["rollout"]["parallelism"]["dp_shard_size"] = 1
        config["policy"]["parallelism"]["tp_size"] = 1
        config["policy"]["parallelism"]["dp_shard_size"] = 2
        config["rollout"]["parallelism"]["n_init_replicas"] = 1
        config["policy"]["parallelism"]["n_init_replicas"] = 1
        config["distillation"]["parallelism"]["n_init_replicas"] = 2
        config["distillation"]["parallelism"]["dp_shard_size"] = 2
        config["validation"]["freq"] = 1
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["train_policy"]["prompt_column_name"] = "prompt"
        config["train"]["train_policy"]["response_column_name"] = "result"
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["train_policy"]["dataset"]["subset"] = ""
        config["validation"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["validation"]["dataset"]["subset"] = ""
        config["validation"]["dataset"]["split"] = "train"
        config["train"]["ckpt"]["enable_checkpoint"] = False
        config["rollout"]["n_generation"] = 2
        config["train"]["train_batch_per_replica"] = 16

        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["logger"] = ["console"]

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name

        controller_cmd = f"cosmos-rl --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        env_dict = os.environ.copy()
        controller_process = subprocess.Popen(
            controller_cmd,
            shell=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        processes = [controller_process]

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert (
                process.returncode == 0
            ), f"Process failed with code: {process.returncode}"


if __name__ == "__main__":
    unittest.main()
