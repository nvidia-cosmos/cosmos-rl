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

import torch
import torch.multiprocessing as mp

from cosmos_rl.utils.ipc.tensor_util import (
    named_tensors_from_serialize,
    named_tensors_to_serialize,
)


class TestTensorIPCUtils(unittest.TestCase):
    """Test tensor IPC utils."""

    @staticmethod
    def _child_process_restore_tensor(state_dict_ipc, expected_mean, result_queue):
        """Child process function to restore tensor from IPC handle."""
        try:
            # Initialize CUDA in child process
            torch.cuda.set_device(0)

            # Restore state dict from IPC handle
            state_dict = named_tensors_from_serialize(state_dict_ipc)

            # Verify the restored tensor
            restored_tensor = state_dict["model.layers.0.self_attn.q_proj.weight"]
            actual_mean = restored_tensor.mean().item()

            # Check if the mean is close to expected (indicates successful restoration)
            is_close = abs(actual_mean - expected_mean) < 1e-5
            result_queue.put(("success", is_close, actual_mean))
        except Exception as e:
            result_queue.put(("error", str(e), None))

    def test_state_dict_ipc_to_state_dict(self):
        """Test state dict IPC to state dict across processes."""
        # Ensure 'spawn' start method for CUDA compatibility
        ctx = mp.get_context("spawn")

        device = torch.device("cuda:0")
        demo_tensor = torch.randn(10, 10, device=device)
        expected_mean = demo_tensor.mean().item()

        demo_state_dict = {"model.layers.0.self_attn.q_proj.weight": demo_tensor}
        state_dict_ipc = named_tensors_to_serialize(demo_state_dict)

        # Create queue for result
        result_queue = ctx.Queue()

        # Start child process to restore tensor
        p = ctx.Process(
            target=self._child_process_restore_tensor,
            args=(state_dict_ipc, expected_mean, result_queue),
            name="child_process_restore_tensor",
        )
        p.start()
        p.join(timeout=30)  # 10 second timeout

        # Check result
        if p.is_alive():
            p.terminate()
            self.fail("Child process timed out")

        self.assertEqual(p.exitcode, 0, "Child process failed")

        # Get result from queue
        status, result, actual_mean = result_queue.get(timeout=5)

        if status == "error":
            self.fail(f"Child process error: {result}")

        self.assertTrue(
            result, f"Tensor mean mismatch: expected {expected_mean}, got {actual_mean}"
        )

    @staticmethod
    def _child_process_modify_tensor(state_dict_ipc, result_queue):
        """Child process function to modify tensor."""
        try:
            # Initialize CUDA in child process
            torch.cuda.set_device(0)

            # Restore state dict from IPC handle
            state_dict = named_tensors_from_serialize(state_dict_ipc)

            # Modify tensor
            state_dict["model.layers.0.self_attn.q_proj.weight"] += 1
            modified_mean = (
                state_dict["model.layers.0.self_attn.q_proj.weight"].mean().item()
            )

            # Put result into queue
            result_queue.put(("success", modified_mean))
        except Exception as e:
            result_queue.put(("error", str(e), None))

    def test_shared_tensor_modification(self):
        """Test shared tensor modification."""
        # Ensure 'spawn' start method for CUDA compatibility
        ctx = mp.get_context("spawn")

        device = torch.device("cuda:0")
        demo_tensor = torch.randn(10, 10, device=device)
        expected_mean_old = demo_tensor.mean().item()

        demo_state_dict = {"model.layers.0.self_attn.q_proj.weight": demo_tensor}
        state_dict_ipc = named_tensors_to_serialize(demo_state_dict)

        # Create queue for result
        result_queue = ctx.Queue()

        # Start child process to modify tensor
        p = ctx.Process(
            target=self._child_process_modify_tensor,
            args=(state_dict_ipc, result_queue),
            name="child_process_modify_tensor",
        )
        p.start()
        p.join(timeout=30)  # 10 second timeout

        # Check result
        if p.is_alive():
            p.terminate()
            self.fail("Child process timed out")

        self.assertEqual(p.exitcode, 0, "Child process failed")

        # Get result from queue
        # status, result = result_queue.get(timeout=5)
        status, result = result_queue.get(block=True)

        if status == "error":
            self.fail(f"Child process error: {result}")

        # check the modified mean
        expected_mean_new = demo_tensor.mean().item()
        self.assertTrue(
            abs(expected_mean_new - (expected_mean_old + 1)) < 1e-5,
            f"Tensor mean mismatch: expected {expected_mean_old + 1}, got {expected_mean_new}",
        )
        self.assertTrue(
            result, f"Tensor mean mismatch: expected {expected_mean_new}, got {result}"
        )


if __name__ == "__main__":
    unittest.main()
