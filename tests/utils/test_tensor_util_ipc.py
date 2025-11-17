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

from cosmos_rl.utils.ipc import (
    ModuleLike,
    named_tensors_from_serialize,
    named_tensors_to_serialize,
)


if __name__ == "__main__":
    unittest.main()


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


class TestModuleLike(unittest.TestCase):
    """Test ModuleLike."""

    def get_module(self):
        """Get the module."""
        state_dict = {
            "model.embed_tokens.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(10, 10),
            "model.layers.0.self_attn.o_proj.bias": torch.randn(10),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(10, 10),
            "model.layers.1.self_attn.k_proj.weight": torch.randn(10, 10),
            "model.layers.1.self_attn.v_proj.weight": torch.randn(10, 10),
            "model.layers.1.self_attn.o_proj.weight": torch.randn(10, 10),
            "model.layers.1.self_attn.o_proj.bias": torch.randn(10),
            "lm_head.weight": torch.randn(10, 10),
        }

        module_names = [
            "",
            "model",
            "model.embed_tokens",
            "model.layers",
            "model.layers.0",
            "model.layers.0.self_attn",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.1",
            "model.layers.1.self_attn",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.self_attn.k_proj",
            "model.layers.1.self_attn.v_proj",
            "model.layers.1.self_attn.o_proj",
            "lm_head",
        ]

        not_parameter_names = set(["lm_head.weight"])
        return ModuleLike(state_dict, not_parameter_names), state_dict, module_names

    def test_getattribute(self):
        """Test getattribute of ModuleLike."""
        fake_module, _, _ = self.get_module()

        self.assertTrue(hasattr(fake_module, "model"))
        self.assertTrue(isinstance(fake_module.model, ModuleLike))
        self.assertTrue(hasattr(fake_module.model, "embed_tokens"))
        self.assertTrue(isinstance(fake_module.model.embed_tokens.weight, torch.Tensor))
        self.assertTrue(fake_module.model.embed_tokens.weight.shape, (10, 10))
        self.assertTrue(isinstance(fake_module.model.embed_tokens, ModuleLike))
        self.assertTrue(hasattr(fake_module.model, "layers"))
        self.assertTrue(isinstance(fake_module.model.layers, ModuleLike))
        self.assertTrue(hasattr(fake_module.model.layers[0], "self_attn"))
        self.assertTrue(isinstance(fake_module.model.layers[0], ModuleLike))
        self.assertTrue(hasattr(fake_module.model.layers[0].self_attn, "q_proj"))
        self.assertTrue(isinstance(fake_module.model.layers[0].self_attn, ModuleLike))
        self.assertTrue(hasattr(fake_module.model.layers[1], "self_attn"))
        self.assertTrue(isinstance(fake_module.model.layers[1], ModuleLike))
        self.assertTrue(hasattr(fake_module.model.layers[1].self_attn, "q_proj"))

    def test_get_state_dict(self):
        """Test get state dict of ModuleLike."""
        fake_module, original_state_dict, _ = self.get_module()
        state_dict = fake_module.state_dict()

        self.assertEqual(len(state_dict), len(original_state_dict))
        for name, value in original_state_dict.items():
            self.assertTrue(name in state_dict)
            self.assertTrue(isinstance(state_dict[name], torch.Tensor))
            self.assertTrue(state_dict[name].shape, value.shape)
            self.assertTrue(state_dict[name].dtype, value.dtype)

    def test_named_modules(self):
        """Test named modules of ModuleLike."""
        fake_module, _, original_module_names = self.get_module()

        module_names = []
        for name, _ in fake_module.named_modules():
            module_names.append(name)

        assert len(module_names) == len(original_module_names)
        assert set(module_names) == set(original_module_names)

    def test_named_parameters(self):
        """Test named parameters of ModuleLike."""
        fake_module, original_state_dict, _ = self.get_module()

        parameter_names = []
        for name, _ in fake_module.named_parameters():
            parameter_names.append(name)

        assert len(parameter_names) == len(original_state_dict) - 1
        assert "lm_head.weight" not in parameter_names
        assert set(parameter_names) == set(original_state_dict.keys()) - set(
            ["lm_head.weight"]
        )
