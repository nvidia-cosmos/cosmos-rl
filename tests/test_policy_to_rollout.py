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
import torch
import subprocess
import sys

from subprocess_helpers import wait_all_or_fail
from p2r_test_support import P2RReadinessServer
from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
from multiprocessing import shared_memory
import numpy as np
from launch_test_worker import POLICY_WORLD_SIZE, ROLLOUT_WORLD_SIZE
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
)


class TestPolicyToRollout(unittest.TestCase):
    def policy_to_rollout_wieght_sync(
        self,
        trainable_param_sync: bool = False,
        *,
        p2r_groups_per_round: int = 0,
        policy_tp_size: int = 2,
        policy_pp_size: int = 1,
        rollout_tp_size: int = 4,
        rollout_dp_shard_size: int = 1,
    ):
        """Test NCCL communication between multiple ranks using torchrun."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # Create NCCL UID and shared memory
        nccl_uid = create_nccl_uid()
        nccl_uid_tensor = torch.tensor(nccl_uid, dtype=torch.int64)
        shm = shared_memory.SharedMemory(
            create=True,
            size=(nccl_uid_tensor.numel() + 1) * nccl_uid_tensor.element_size(),
        )
        uid_array = np.ndarray(
            (nccl_uid_tensor.numel() + 1,), dtype=np.int64, buffer=shm.buf
        )
        uid_array[-1] = 0
        trainable_param_sync_str = "True" if trainable_param_sync else "False"
        readiness = None

        try:
            readiness = P2RReadinessServer(
                PolicyToRolloutUnicastCommand(
                    "policy",
                    "rollout",
                    POLICY_WORLD_SIZE,
                    ROLLOUT_WORLD_SIZE,
                    trainable_only=trainable_param_sync,
                )
            )
            # Create the Python command for torchrun
            policy_cmd = [
                "torchrun",
                f"--nproc_per_node={POLICY_WORLD_SIZE}",  # Use 4 GPUs
                "--role=rank",
                "--tee=3",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                os.path.join(cur_dir, "launch_test_worker.py"),
                "--shm_name",
                shm.name,
                "--shm_size",
                str(nccl_uid_tensor.numel()),
                "--mode",
                "policy_send_to_rollout",
                "--trainable_param_sync",
                trainable_param_sync_str,
            ]
            rollout_cmd = [
                "torchrun",
                f"--nproc_per_node={ROLLOUT_WORLD_SIZE}",  # Use 4 GPUs
                "--role=rank",
                "--tee=3",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                os.path.join(cur_dir, "launch_test_worker.py"),
                "--shm_name",
                shm.name,
                "--shm_size",
                str(nccl_uid_tensor.numel()),
                "--mode",
                "rollout_recv_from_policy",
                "--trainable_param_sync",
                trainable_param_sync_str,
            ]
            policy_env = dict(os.environ)
            policy_env.update(readiness.environment())
            policy_env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            policy_env.pop("COSMOS_P2R_NCCL_GROUP_SIZE", None)
            policy_env.update(
                {
                    "COSMOS_TEST_P2R_GROUPS_PER_ROUND": str(p2r_groups_per_round),
                    "COSMOS_TEST_P2R_POLICY_TP_SIZE": str(policy_tp_size),
                    "COSMOS_TEST_P2R_POLICY_PP_SIZE": str(policy_pp_size),
                    "COSMOS_TEST_P2R_ROLLOUT_TP_SIZE": str(rollout_tp_size),
                    "COSMOS_TEST_P2R_ROLLOUT_DP_SHARD_SIZE": str(rollout_dp_shard_size),
                }
            )
            # Start the process
            policy_process = subprocess.Popen(
                policy_cmd,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=policy_env,
            )
            rollout_env = dict(os.environ)
            rollout_env.update(readiness.environment())
            rollout_env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
            rollout_env.pop("COSMOS_P2R_NCCL_GROUP_SIZE", None)
            rollout_env.update(
                {
                    "COSMOS_TEST_P2R_GROUPS_PER_ROUND": str(p2r_groups_per_round),
                    "COSMOS_TEST_P2R_POLICY_TP_SIZE": str(policy_tp_size),
                    "COSMOS_TEST_P2R_POLICY_PP_SIZE": str(policy_pp_size),
                    "COSMOS_TEST_P2R_ROLLOUT_TP_SIZE": str(rollout_tp_size),
                    "COSMOS_TEST_P2R_ROLLOUT_DP_SHARD_SIZE": str(rollout_dp_shard_size),
                }
            )
            rollout_process = subprocess.Popen(
                rollout_cmd,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=rollout_env,
            )

            wait_all_or_fail(
                self,
                [policy_process, rollout_process],
                timeout_s=600,
                context="policy_to_rollout_wieght_sync",
            )
            readiness.assert_all_ready()
        finally:
            if readiness is not None:
                readiness.close()
            # Clean up shared memory
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                # Ignore if shared memory is already unlinked
                pass

    def test_policy_to_rollout_wieght_sync_all_params(self):
        self.policy_to_rollout_wieght_sync(trainable_param_sync=False)

    def test_policy_to_rollout_wieght_sync_trainable_params(self):
        self.policy_to_rollout_wieght_sync(trainable_param_sync=True)

    def test_policy_to_rollout_grouped_with_pp2_fsdp2(self):
        self.policy_to_rollout_wieght_sync(
            p2r_groups_per_round=4,
            policy_tp_size=1,
            policy_pp_size=2,
            rollout_tp_size=2,
        )

    def test_policy_to_rollout_grouped_with_tp2_fsdp2(self):
        self.policy_to_rollout_wieght_sync(
            p2r_groups_per_round=4,
            policy_tp_size=2,
            policy_pp_size=1,
            rollout_tp_size=4,
        )

    def test_policy_to_rollout_grouped_with_policy_and_rollout_tp2_fsdp2(self):
        self.policy_to_rollout_wieght_sync(
            p2r_groups_per_round=4,
            policy_tp_size=2,
            policy_pp_size=1,
            rollout_tp_size=2,
            rollout_dp_shard_size=2,
        )


if __name__ == "__main__":
    unittest.main()
