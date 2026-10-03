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
from cosmos_rl.utils import network_util
import toml
import tempfile
from pathlib import Path
import torch
from subprocess_helpers import wait_all_or_fail, wait_for_controller_ready


class TestColocated(unittest.TestCase):
    def test_colocated(self):
        self.run_colocated()

    def test_final_step_healthy(self):
        self.run_colocated("healthy")

    def test_final_step_quality_refill(self):
        self.run_colocated("reject-final-group")

    def test_genuine_exhaustion_checkpoints_last_completed_update(self):
        self.run_colocated("exhaust-final-group")

    def run_colocated(self, refill_case=None):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 1 if refill_case else 4
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_grpo.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)

        config["train"]["epoch"] = 1
        config["train"]["train_batch_per_replica"] = 32
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["train_policy"]["mini_batch"] = 1
        config["rollout"]["n_generation"] = 8
        config["rollout"]["batch_size"] = 1
        config["rollout"]["max_response_length"] = 128
        config["mode"] = "colocated"
        config["train"]["force_use_hf"] = True
        config["rollout"]["backend"] = "example_hf"
        config["rollout"]["parallelism"]["tp_size"] = 1
        config["rollout"]["parallelism"]["dp_shard_size"] = 4
        config["policy"]["parallelism"]["tp_size"] = 1
        config["policy"]["parallelism"]["dp_shard_size"] = 4
        config["rollout"]["parallelism"]["n_init_replicas"] = 2
        config["policy"]["parallelism"]["n_init_replicas"] = 2
        config["redis"] = str(network_util.find_available_port(12808))
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["logger"] = ["console"]
        if refill_case:
            config["train"]["max_num_steps"] = 3
            config["train"]["train_batch_per_replica"] = 16
            config["rollout"]["parallelism"]["dp_shard_size"] = world_size
            config["policy"]["parallelism"]["dp_shard_size"] = world_size
            config["validation"] = {"enable": False}
            if refill_case == "exhaust-final-group":
                config["train"]["output_dir"] = tempfile.mkdtemp(
                    prefix="colocated-exhaustion-"
                )
                config["train"]["ckpt"]["enable_checkpoint"] = True
                config["train"]["ckpt"]["save_freq"] = 100

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
            start_new_session=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        wait_for_controller_ready(
            self,
            controller_process,
            port,
            timeout_s=120,
            context="test_colocated",
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "utils", "mock_policy_entrance.py"),
            "--test",
            "colocated_final_step_refill" if refill_case else "colocated",
        ]

        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0" if refill_case else "0,1,2,3"
        if refill_case:
            policy_env["COLOCATED_REFILL_CASE"] = refill_case
            policy_env["COLOCATED_REFILL_TARGET"] = "1"
        # Start the process
        policy_process0 = subprocess.Popen(
            policy_cmd,
            start_new_session=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )

        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "1" if refill_case else "4,5,6,7"
        if refill_case:
            policy_env["COLOCATED_REFILL_CASE"] = refill_case
            policy_env["COLOCATED_REFILL_TARGET"] = "0"
        # Start the process
        policy_process1 = subprocess.Popen(
            policy_cmd,
            start_new_session=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )

        processes = [controller_process, policy_process0, policy_process1]
        # This deadline covers model loading plus both training steps, not just
        # shutdown. CI can still be making step-1 progress near five minutes.
        # Keep a finite budget and the existing failure/process-tree cleanup.
        wait_all_or_fail(self, processes, timeout_s=600, context="test_colocated")
        if refill_case == "exhaust-final-group":
            output = Path(config["train"]["output_dir"])
            metadata = list(output.rglob("extra_info_rank_0.pth"))
            self.assertTrue(metadata, "early completion must save a checkpoint")
            for path in metadata:
                state = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(state["step"], 2)
                self.assertEqual(state["total_steps"], 3)
                self.assertTrue((path.parent / ".rank_0_complete").is_file())
                self.assertTrue((path.parent / "model_rank_0.pth").is_file())
                self.assertTrue((path.parent / "optimizer_rank_0.pth").is_file())
            self.assertFalse(list(output.rglob("step_3")))
            print("COLOCATED_EXHAUSTION_ARTIFACT_PASS step=2 horizon=3", flush=True)


if __name__ == "__main__":
    unittest.main()
