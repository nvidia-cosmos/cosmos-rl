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
from cosmos_rl.utils import util
import tempfile
import toml


class TestGspo(unittest.TestCase):
    def test_gspo(self):
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
            "gspo_test",
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


class TestReferenceReset(unittest.TestCase):
    def test_reference_reset(self):
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
            "reference_reset_test",
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


class TestPolicyBatchSize(unittest.TestCase):
    def test_dynamic_batchsize(self):
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
            "dynamic_batchsize_test",
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


class TestDecoupledLoss(unittest.TestCase):
    def test_decoupled_loss(self):
        """Test grpo all processes exit cleanly."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_grpo.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)

        config["train"]["epoch"] = 1
        config["train"]["train_batch_per_replica"] = 16
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["rollout"]["n_generation"] = 2
        config["rollout"]["n_generation_to_batch"] = True
        config["rollout"]["batch_size"] = 8
        config["train"]["train_policy"]["use_decoupled_loss"] = True
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
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "utils", "mock_policy_entrance.py"),
            "--test",
            "decoupled_loss",
        ]
        rollout_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "utils", "mock_rollout_entrance.py"),
            "--test",
            "decoupled_loss",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )
        rollout_env = dict(os.environ)
        rollout_env["CUDA_VISIBLE_DEVICES"] = "2,3"
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
