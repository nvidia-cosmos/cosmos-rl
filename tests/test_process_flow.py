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
import signal
import sys
import time
import toml
import tempfile

from cosmos_rl.utils import network_util
from subprocess_helpers import (
    kill_process_group,
    wait_all_or_fail,
    wait_for_controller_ready,
)

# Bounded wait for the end-of-data shutdown tests.  Sized to cover a healthy
# run (model load + vLLM init + a short single-epoch GRPO/SFT drain, which
# completes well under a minute on CI) plus generous margin -- but far below
# the CI job's ``timeout 2h`` wall, so a shutdown deadlock fails fast and
# attributed instead of burning hours.  A wedge here manifests within seconds
# of end-of-data, so this is pure backstop headroom, not expected runtime.
_END_OF_DATA_TIMEOUT_S = 300

_kill_process_group = kill_process_group
_wait_all_or_fail = wait_all_or_fail

# Default bounded wait for non-end-of-data scenarios (validation, reward check,
# ddp load).  Sized above healthy runtime with margin, but far below the CI
# job's ``timeout 2h`` wall.  Override via COSMOS_TEST_PROC_EXIT_TIMEOUT.
_PROC_EXIT_TIMEOUT = float(os.environ.get("COSMOS_TEST_PROC_EXIT_TIMEOUT", "300"))
# Long multi-replica / multi-epoch runs (e.g. test_multi_replica_sft) need more
# headroom than the default; keep a separate cap with faulthandler diagnostics.
_LONG_PROC_EXIT_TIMEOUT = float(
    os.environ.get("COSMOS_TEST_LONG_PROC_EXIT_TIMEOUT", "1200")
)


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
        config["redis"] = str(network_util.find_available_port(12808))
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
        wait_for_controller_ready(
            self,
            controller_process,
            port,
            timeout_s=120,
            context="grpo end-of-data shutdown (tp=2)",
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

        _wait_all_or_fail(
            self,
            processes,
            timeout_s=_END_OF_DATA_TIMEOUT_S,
            context="grpo end-of-data shutdown (tp=2)",
        )

    def _make_multirank_end_of_data_config(self, rollout_tp_size, world_size):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(cur_dir, "configs", "test_simple_grpo.toml")
        with open(config_path, "r") as f:
            config = toml.load(f)
        config["train"]["epoch"] = 1
        # Small batch so the last step of the single epoch leaves an
        # uneven prompt tail across the worker's ranks.  GRPO's default
        # mini_batch=2 requires train_batch_per_replica to be divisible by it,
        # so this test uses mini_batch=1 to keep the odd tail intentional.
        config["train"]["train_batch_per_replica"] = 3
        config["train"]["train_policy"]["mini_batch"] = 1
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["train_policy"]["allowed_outdated_steps"] = 100
        config["redis"] = str(network_util.find_available_port(12808))
        config["rollout"]["parallelism"]["tp_size"] = rollout_tp_size
        # Auto-infer rollout dp_shard from WORLD_SIZE // tp_size.  Required
        # for the tp=1 layout (dp=2 on a 2-GPU worker); explicit dp_shard_size
        # values are rejected for vLLM rollout configs.
        config["rollout"]["parallelism"]["dp_shard_size"] = -1
        return config

    def test_multirank_end_of_data_config_is_valid(self):
        """CPU guard for the CI-only GRPO end-of-data test configs."""
        from cosmos_rl.policy.config import Config
        from cosmos_rl.utils.parallelism import ParallelDims

        world_size = 2
        old_world_size = os.environ.get("WORLD_SIZE")
        try:
            os.environ["WORLD_SIZE"] = str(world_size)
            for rollout_tp_size in (1, 2):
                with self.subTest(rollout_tp_size=rollout_tp_size):
                    config = self._make_multirank_end_of_data_config(
                        rollout_tp_size, world_size
                    )
                    loaded = Config.from_dict(config)
                    train_batch_per_replica = config["train"]["train_batch_per_replica"]
                    mini_batch = config["train"]["train_policy"]["mini_batch"]
                    policy_parallel = config["policy"]["parallelism"]
                    policy_dp_shard = policy_parallel["dp_shard_size"]
                    if policy_dp_shard == -1:
                        policy_dp_shard = world_size // policy_parallel["tp_size"]
                    rollout_parallel = config["rollout"]["parallelism"]
                    rollout_dp_shard = world_size // rollout_parallel["tp_size"]
                    parallel_dims = ParallelDims.from_config(loaded.rollout.parallelism)
                    self.assertEqual(
                        train_batch_per_replica % (policy_dp_shard * mini_batch),
                        0,
                    )
                    self.assertEqual(rollout_parallel["dp_shard_size"], -1)
                    self.assertEqual(parallel_dims.dp_shard, rollout_dp_shard)
                    self.assertEqual(
                        rollout_parallel["tp_size"] * rollout_dp_shard,
                        world_size,
                    )
        finally:
            if old_world_size is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = old_world_size

    def _run_multirank_end_of_data(self, rollout_tp_size, context):
        """Drive a multi-rank rollout worker to genuine end-of-data and
        assert every process exits cleanly within a bounded wait.

        Shared body for the ``tp`` (dp==1) and ``dp`` (dp>1) variants
        below.  ``rollout_tp_size`` selects the rollout DP layout for the
        2-GPU rollout worker: ``2`` -> pure TP (all ranks drain on the
        same iteration); ``1`` -> dp_shard=2, so the final round-robin
        batch leaves an *uneven* tail and ranks reach end-of-data on
        different iterations -- the case the cross-rank ``StopCommand``
        broadcast must handle without stranding a peer.
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config = self._make_multirank_end_of_data_config(rollout_tp_size, world_size)
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
            start_new_session=True,
        )
        wait_for_controller_ready(
            self,
            controller_process,
            port,
            timeout_s=120,
            context=context,
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        worker_args = [
            "torchrun",
            f"--nproc_per_node={world_size}",
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
        ]
        policy_cmd = worker_args + ["dummy_policy"]
        rollout_cmd = worker_args + ["dummy_rollout"]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
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
        # Generous bound (model load + a short single-epoch GRPO run) but
        # far below the 2h CI wall, so the consume-end deadlock surfaces
        # as a fast, attributed failure.
        _wait_all_or_fail(
            self, processes, timeout_s=_END_OF_DATA_TIMEOUT_S, context=context
        )

    def test_process_exit_grpo_multirank_end_of_data(self):
        """Regression: a multi-rank rollout worker driven to genuine
        end-of-data must shut down without deadlocking.

        ``tp`` layout (rollout tp_size=2 -> world_size=2, dp==1): all
        ranks reach end-of-data on the same iteration.  The controller's
        explicit ``StopCommand`` -- broadcast across ranks by
        ``consume_one_command`` -- self-terminates them in lockstep,
        without the old stop-carrying R2R weight-sync broadcast that raced
        the end-of-data signal (see rollout_multirank_shutdown.md).
        """
        self._run_multirank_end_of_data(
            rollout_tp_size=2,
            context="grpo multi-rank end-of-data shutdown (tp=2, dp=1)",
        )

    def test_process_exit_grpo_multirank_dp_end_of_data(self):
        """Regression (corner E): rollout tp_size=1 -> dp_shard=2, so the
        final round-robin batch is scattered *unevenly* and the two ranks
        reach ``prompt_consume_end`` on *different* iterations.

        This is the case a rank-local ``shutdown_signal.set()`` cannot
        handle (the first rank to drain would strand the other in the next
        collective); the ``StopCommand`` is delivered to every rank at the
        same ``consume_one_command`` broadcast, so they leave ``main_loop``
        together regardless of the uneven tail.
        """
        self._run_multirank_end_of_data(
            rollout_tp_size=1,
            context="grpo multi-rank end-of-data shutdown (tp=1, dp=2)",
        )

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

        _wait_all_or_fail(
            self,
            processes,
            timeout_s=_END_OF_DATA_TIMEOUT_S,
            context="sft end-of-data shutdown",
        )


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

        _await_processes(processes, timeout=_LONG_PROC_EXIT_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
