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
# Enable faulthandler at interpreter startup in every child process (controller,
# torchrun workers, launchers) that copies this env. With it set, a SIGABRT makes
# each Python process dump *all* thread stacks to stderr before dying, so a
# teardown hang is pinpointed in the logs. Test-only; no production code touched.
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
import unittest
import subprocess
import sys
import time
import signal
import toml
import tempfile

from cosmos_rl.utils import network_util


# Bound for how long a process-flow scenario may take before a non-exiting
# process is treated as hung. Generous (covers model load + a few steps + the
# heartbeat-timeout teardown) but finite, so a hang fails fast and LOUD here
# instead of silently burning the outer CI `timeout 2h`.
_PROC_EXIT_TIMEOUT = float(os.environ.get("COSMOS_TEST_PROC_EXIT_TIMEOUT", "1200"))


def _signal_tree(proc, sig):
    """Send `sig` to a child's whole process group when it leads its own group.

    Children run with PYTHONFAULTHANDLER=1 (set at module import), so SIGABRT
    makes every Python process dump *all* thread stacks to stderr before dying
    -- pinpointing exactly where a hang is. We only use killpg when
    the child is its own session/group leader (started with start_new_session);
    otherwise we fall back to a per-process signal so we never hit the test
    runner's own group.
    """
    try:
        pgid = os.getpgid(proc.pid)
        if pgid == proc.pid:
            os.killpg(pgid, sig)
            return
    except Exception:
        pass
    try:
        proc.send_signal(sig)
    except Exception:
        pass


def _diagnose_and_kill(processes):
    for proc in processes:
        if proc.poll() is None:
            _signal_tree(proc, signal.SIGABRT)
    # Let faulthandler flush all-thread stacks to stderr before the hard kill.
    time.sleep(5.0)
    for proc in processes:
        if proc.poll() is None:
            _signal_tree(proc, signal.SIGKILL)


def _await_processes(processes, timeout=_PROC_EXIT_TIMEOUT):
    """Wait for all processes to exit within `timeout`; diagnose + fail on hang.

    On timeout we dump every process's thread stacks (via SIGABRT/faulthandler),
    tear the tree down, and fail with a clear message -- instead of blocking
    forever in communicate() as the previous loop did.
    """
    deadline = time.time() + timeout
    for process in processes:
        remaining = max(1.0, deadline - time.time())
        try:
            process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            _diagnose_and_kill(processes)
            raise AssertionError(
                f"Process pid={process.pid} did not exit within {timeout:.0f}s -- "
                "likely a teardown/finalize hang. All-thread stack dumps were "
                "emitted to stderr above (SIGABRT via faulthandler)."
            )
        assert process.returncode == 0, (
            f"Process failed with code: {process.returncode}"
        )


class TestProcessFlow(unittest.TestCase):
    def test_process_exit_grpo(self):
        """Test grpo all processes exit cleanly."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_grpo.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)
        config["train"]["epoch"] = 1
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["train_policy"]["allowed_outdated_steps"] = 100
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
            # Own session so a hang can be group-signaled (SIGABRT -> faulthandler
            # all-thread dump) without touching the test runner's own group.
            start_new_session=True,
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
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",
            "--shm_size",
            "-1",
            "--mode",
            "dummy_policy",
        ]
        rollout_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",
            "--shm_size",
            "-1",
            "--mode",
            "dummy_rollout",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
            start_new_session=True,
        )
        rollout_env = dict(os.environ)
        rollout_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        rollout_process = subprocess.Popen(
            rollout_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=rollout_env,
            start_new_session=True,
        )

        processes = [controller_process, policy_process, rollout_process]

        # Wait for processes to complete (bounded; diagnoses hangs).
        _await_processes(processes)

    def test_process_exit_sft(self):
        """Test sft all processes exit cleanly."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_sft.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)
        config["train"]["epoch"] = 1
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
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
            # Own session so a hang can be group-signaled (SIGABRT -> faulthandler
            # all-thread dump) without touching the test runner's own group.
            start_new_session=True,
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
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",
            "--shm_size",
            "-1",
            "--mode",
            "dummy_policy",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
            start_new_session=True,
        )
        processes = [controller_process, policy_process]

        # Wait for processes to complete (bounded; diagnoses hangs).
        _await_processes(processes)


class TestValidationFlow(unittest.TestCase):
    def test_train_validation(self):
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
            "sft_for_validation",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
            start_new_session=True,
        )
        processes = [process]

        # Wait for processes to complete (bounded; diagnoses hangs).
        _await_processes(processes)


class TestRewardFlow(unittest.TestCase):
    def test_check_reward(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
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
            "reward_execution_check",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
            start_new_session=True,
        )
        processes = [process]

        # Wait for processes to complete (bounded; diagnoses hangs).
        _await_processes(processes)


class TestSFTDDPLoadFlow(unittest.TestCase):
    def test_sft_ddp_load(self):
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
            "sft_ddp_load_check",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
            start_new_session=True,
        )
        processes = [process]

        # Wait for processes to complete (bounded; diagnoses hangs).
        _await_processes(processes)


class TestMultiReplicaSFT(unittest.TestCase):
    def test_multi_replica_sft(self):
        """Test the multi-replica SFT process flow."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_sft.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)

        config["train"]["epoch"] = 16
        config["train"]["train_batch_per_replica"] = 4
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "sharegpt52k_small"
        )
        config["policy"]["parallelism"]["tp_size"] = 1
        config["policy"]["parallelism"]["dp_shard_size"] = 2
        config["policy"]["parallelism"]["n_init_replicas"] = 4

        config["validation"]["batch_size"] = 2
        config["validation"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "sharegpt52k_small"
        )

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
            # Own session so a hang can be group-signaled (SIGABRT -> faulthandler
            # all-thread dump) without touching the test runner's own group.
            start_new_session=True,
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
            "multi_replica_sft",
        ]
        rollout_processes = []
        for dev in ["0,1", "2,3", "4,5", "6,7"]:
            rollout_env = dict(os.environ)
            rollout_env["CUDA_VISIBLE_DEVICES"] = dev
            rollout_processes.append(
                subprocess.Popen(
                    policy_cmd,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    env=rollout_env,
                    start_new_session=True,
                )
            )

        processes = [controller_process] + rollout_processes

        # Wait for processes to complete (bounded; diagnoses hangs).
        _await_processes(processes)


if __name__ == "__main__":
    unittest.main()
