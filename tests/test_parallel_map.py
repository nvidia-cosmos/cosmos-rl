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
import sys
import subprocess
import unittest


class TestParallelMap(unittest.TestCase):
    def test_parallel_topo_mapper(self):
        """Test ParallelTopoMapper with a mock ParallelismConfig."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # Create the Python command for torchrun
        cmd = [
            "torchrun",
            "--nproc_per_node=1",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "-1",  # Use -1 to indicate no need for shared memory
            "-1",  # Use -1 to indicate no need for shared memory size
            "parallel_map_check",
        ]

        env = dict(os.environ)
        # Start the process
        policy_process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
        )
        try:
            # Wait for process to complete
            for process in [policy_process]:
                stdout, stderr = process.communicate()

                # Check if process completed successfully
                assert process.returncode == 0, f"Process failed: {stderr.decode()}"

        finally:
            # Ensure process is terminated
            for process in [policy_process]:
                process.wait()

    def test_policy_parallelism_extract(self):
        """Test policy parallelism extraction using torchrun."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        cases = [
            [2, 1, 2],  # fsdp: 2, tp: 1, pp: 2
            [2, 2, 1],  # fsdp: 2, tp: 2, pp: 1
            [1, 1, 1],  # fsdp: 1, tp: 1, pp: 1
        ]

        for case in cases:
            fsdp = case[0]
            tp = case[1]
            pp = case[2]

            world_size = fsdp * tp * pp  # Total number of processes

            # Create the Python command for torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={world_size}",  # Use 4 GPUs
                "--role=rank",
                "--tee=3",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                os.path.join(cur_dir, "launch_test_worker.py"),
                "-1",  # Use -1 to indicate no need for shared memory
                "-1",  # Use -1 to indicate no need for shared memory size
                "policy_parallelism_extract",
                f"fsdp:{fsdp};tp:{tp};pp:{pp}",
            ]

            env = dict(os.environ)
            # Start the process
            policy_process = subprocess.Popen(
                cmd,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=env,
            )
            try:
                # Wait for process to complete
                for process in [policy_process]:
                    stdout, stderr = process.communicate()

                    # Check if process completed successfully
                    assert process.returncode == 0, f"Process failed: {stderr.decode()}"

            finally:
                # Ensure process is terminated
                for process in [policy_process]:
                    process.wait()

    def test_rollout_parallelism_extract(self):
        """Test rollout parallelism extraction using torchrun."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        cases = [[4, 1], [1, 1]]

        for case in cases:
            fsdp = 1  # always FSDP to be 1 for rollout parallelism.
            tp = case[0]
            pp = case[1]
            world_size = fsdp * tp * pp  # Total number of processes

            # Create the Python command for torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={world_size}",  # Use 4 GPUs
                "--role=rank",
                "--tee=3",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                os.path.join(cur_dir, "launch_test_worker.py"),
                "-1",  # Use -1 to indicate no need for shared memory
                "-1",  # Use -1 to indicate no need for shared memory size
                "rollout_parallelism_extract",
                f"fsdp:{fsdp};tp:{tp};pp:{pp}",
            ]

            env = dict(os.environ)
            # Start the process
            policy_process = subprocess.Popen(
                cmd,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=env,
            )
            try:
                # Wait for process to complete
                for process in [policy_process]:
                    stdout, stderr = process.communicate()

                    # Check if process completed successfully
                    assert process.returncode == 0, f"Process failed: {stderr.decode()}"

            finally:
                # Ensure process is terminated
                for process in [policy_process]:
                    process.wait()


if __name__ == "__main__":
    unittest.main()
