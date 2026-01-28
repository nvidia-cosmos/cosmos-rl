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

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import unittest
import subprocess
import sys
import toml
import tempfile

from cosmos_rl.utils import network_util
from launch_test_worker import load_simple_grpo_config


class TestColocatedSeparated(unittest.TestCase):
    def test_colocated_separated(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # 2 policy replicas and 2 rollout replicas, they share the same devices.
        world_size = 2
        cuda_visible_devices = ",".join(str(i) for i in range(world_size))
        port = network_util.find_available_port(8123)

        config = load_simple_grpo_config()

        config["mode"] = "colocated_separated"
        # lower the gpu memory utilization for rollout to avoid OOM when Policy and Rollout share the same devices.
        config["rollout"]["gpu_memory_utilization"] = 0.3

        config["train"]["epoch"] = 1
        config["train"]["train_batch_per_replica"] = 64
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["force_use_hf"] = True
        config["train"]["train_policy"]["mini_batch"] = 1

        config["rollout"]["n_generation"] = 8
        config["rollout"]["batch_size"] = 1
        config["rollout"]["backend"] = "vllm"
        config["rollout"]["max_response_length"] = 128

        config["rollout"]["parallelism"]["tp_size"] = 2
        config["rollout"]["parallelism"]["n_init_replicas"] = 1

        config["policy"]["parallelism"]["tp_size"] = 1
        config["policy"]["parallelism"]["dp_shard_size"] = 2
        config["policy"]["parallelism"]["n_init_replicas"] = 1
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["logger"] = ["console"]

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name
        controller_cmd = f"{sys.executable} -m cosmos_rl.dispatcher.run_web_panel --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        env_dict = os.environ.copy()
        env_dict["COSMOS_ROLE"] = "Controller"
        controller_process = subprocess.Popen(
            controller_cmd,
            shell=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "utils", "mock_policy_entrance.py"),
        ]

        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )

        # rollout
        rollout_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "utils", "mock_rollout_entrance.py"),
        ]

        rollout_env = dict(os.environ)
        rollout_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        # Start the process
        rollout_process = subprocess.Popen(
            rollout_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=rollout_env,
        )

        processes = [controller_process, policy_process, rollout_process]

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert (
                process.returncode == 0
            ), f"Process failed with code: {process.returncode}"


if __name__ == "__main__":
    unittest.main()
