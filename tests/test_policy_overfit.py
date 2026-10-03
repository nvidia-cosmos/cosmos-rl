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
import subprocess
import sys
import tempfile
import unittest

import toml

from cosmos_rl.utils import network_util
from cosmos_rl.utils.model_config import load_model_config
from subprocess_helpers import kill_process_group


class TestPolicyOverfit(unittest.TestCase):
    def test_policy_overfit(self):
        """Tests if policy trains on fixed samples, overfits, and decreases training loss."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 8
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "sft_integration_deepseek_simple.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "sharegpt52k_small"
        )
        # Publish the immutable fixture's dynamic configuration before ranks
        # race to copy/import it into the shared Transformers module cache.
        # A half-written import can otherwise stay poisoned across retries.
        load_model_config(config["policy"]["model_name_or_path"])
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name
        controller_cmd = [
            sys.executable,
            "-m",
            "cosmos_rl.dispatcher.run_web_panel",
            "--config",
            tmpfile_toml,
            "--port",
            str(port),
        ]
        env_dict = os.environ.copy()
        env_dict["COSMOS_ROLE"] = "Controller"
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
            "test_overfit",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        # NCCL allocates its buffers with raw cudaMalloc, outside torch's
        # caching allocator.  With the default allocator this model fills the
        # card with retained, fragmented segments (~99% of an 80GB H100), and
        # the next NCCL collective fails in include/alloc.h with
        # "Cuda failure 2 'out of memory'" -- never a torch "Tried to allocate",
        # because torch is the one holding it.  Expandable segments release
        # physical pages back, so NCCL can allocate.
        #
        # cosmos_rl/launcher/utility.py sets this for every replica it starts,
        # which is why real runs are unaffected; this test spawns torchrun
        # directly and so is the one path that misses it.  Set it here rather
        # than in the CI harness so the test carries its own requirement.
        policy_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Wait on the POLICY, not the controller.  The controller is an HTTP
        # server: it exits only when training completes normally, so waiting on
        # it first turns *any* policy-side failure into an unbounded hang --
        # the policy dies, the controller never sees completion, never exits,
        # and communicate() blocks forever.  Observed twice on an 8-GPU node,
        # each time consuming the suite's entire remaining budget (a 2h
        # ceiling, and 29 later suites that never ran).
        timeout_s = float(os.environ.get("COSMOS_TEST_OVERFIT_TIMEOUT_S", "1800"))
        controller_process = subprocess.Popen(
            controller_cmd,
            start_new_session=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        policy_process = None
        try:
            policy_process = subprocess.Popen(
                policy_cmd,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=policy_env,
                start_new_session=True,
            )
            policy_process.communicate(timeout=timeout_s)
            assert policy_process.returncode == 0, (
                f"policy process failed with code: {policy_process.returncode}"
            )
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"policy did not finish within {timeout_s:.0f}s; killed it. "
                "Set COSMOS_TEST_OVERFIT_TIMEOUT_S to raise the budget."
            )
        finally:
            # A shell-only terminate leaves the controller holding tee's pipe,
            # even after the test and its timeout process have already exited.
            try:
                if policy_process is not None:
                    kill_process_group(policy_process, owned_session=True)
            finally:
                kill_process_group(controller_process, owned_session=True)


if __name__ == "__main__":
    unittest.main()
