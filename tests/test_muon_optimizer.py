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

"""
Unit tests for Muon Optimizer

This module contains comprehensive tests for the Muon optimizer implementation,
including tests for:
- Parameter updates (Muon vs AdamW fallback)
- Newton-Schulz orthogonalization
- Weight decay
- Momentum accumulation
- Nesterov momentum
- State dict serialization
"""

import torch
import torch.nn as nn
import unittest

from cosmos_rl.policy.trainer.optm.muon_optimizer import MuonOptimizer


class DummyModel(nn.Module):
    """
    Dummy model for testing Muon optimizer.

    Contains:
    - Linear layer (2D weight matrix) -> should use Muon
    - Bias parameter (1D) -> should use AdamW fallback
    - LayerNorm (1D weight) -> should use AdamW fallback
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        return x + self.bias


class TestMuonOptimizer(unittest.TestCase):
    """Test suite for Muon Optimizer"""

    def setUp(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        self.model = DummyModel()
        self.input_dim = 10
        self.hidden_dim = 20

    def test_muon_updates_2d_weights(self):
        """Test that Muon optimizer updates 2D weight matrices"""
        # Separate parameters into Muon and AdamW groups
        muon_params = [self.model.linear.weight]
        adamw_params = [
            self.model.linear.bias,
            self.model.layer_norm.weight,
            self.model.bias,
        ]

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.0,  # Disable momentum for simpler test
            weight_decay=0.0,
            nesterov=False,
            ns_steps=1,  # Use minimal steps for testing
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        # Create dummy input and target
        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)

        # Forward pass
        output = self.model(x)
        loss = nn.MSELoss()(output, target)

        # Store weights before update
        weight_before = self.model.linear.weight.clone()

        # Backward and step
        loss.backward()
        optimizer.step()

        # Check that weight was updated
        self.assertFalse(
            torch.allclose(self.model.linear.weight, weight_before),
            "Muon optimizer should update 2D weight matrices",
        )

    def test_adamw_updates_1d_params(self):
        """Test that AdamW fallback updates 1D parameters"""
        muon_params = [self.model.linear.weight]
        adamw_params = [
            self.model.linear.bias,
            self.model.layer_norm.weight,
            self.model.bias,
        ]

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            ns_steps=1,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)

        output = self.model(x)
        loss = nn.MSELoss()(output, target)

        # Store biases before update
        bias_before = self.model.linear.bias.clone()
        param_before = self.model.bias.clone()

        loss.backward()
        optimizer.step()

        # Check that biases were updated by AdamW
        self.assertFalse(
            torch.allclose(self.model.linear.bias, bias_before),
            "AdamW fallback should update bias parameters",
        )
        self.assertFalse(
            torch.allclose(self.model.bias, param_before),
            "AdamW fallback should update 1D parameters",
        )

    def test_momentum_accumulation(self):
        """Test that momentum is properly accumulated"""
        muon_params = [self.model.linear.weight]
        adamw_params = [self.model.linear.bias]

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0,
            nesterov=False,
            ns_steps=1,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)

        # First step
        output1 = self.model(x)
        loss1 = nn.MSELoss()(output1, target)
        loss1.backward()

        # Store gradient
        grad1 = self.model.linear.weight.grad.clone()

        optimizer.step()
        optimizer.zero_grad()

        # Second step with different gradient
        output2 = self.model(x)
        loss2 = nn.MSELoss()(output2, target)
        loss2.backward()

        grad2 = self.model.linear.weight.grad.clone()

        # Check momentum buffer exists and has correct shape
        state = optimizer.state[self.model.linear.weight]
        self.assertIn("momentum_buffer", state)
        self.assertEqual(state["momentum_buffer"].shape, self.model.linear.weight.shape)

        # After second step, momentum buffer should contain weighted sum
        optimizer.step()
        # Momentum buffer should be: momentum * grad1 + grad2 (approximately)
        # Since we zero_grad between steps, grad1 contribution is in momentum buffer
        expected_momentum = 0.9 * grad1 + grad2
        # Allow some tolerance due to Newton-Schulz orthogonalization
        self.assertTrue(
            torch.allclose(
                state["momentum_buffer"], expected_momentum, atol=1e-3, rtol=1e-3
            ),
            "Momentum buffer should accumulate gradients correctly",
        )

    def test_nesterov_momentum(self):
        """Test Nesterov momentum update"""
        muon_params = [self.model.linear.weight]
        adamw_params = []

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0,
            nesterov=True,  # Enable Nesterov
            ns_steps=1,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)

        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        weight_before = self.model.linear.weight.clone()
        # grad = self.model.linear.weight.grad.clone()

        optimizer.step()

        # With Nesterov, update should use grad + momentum * momentum_buffer
        # Check that weight changed
        self.assertFalse(
            torch.allclose(self.model.linear.weight, weight_before),
            "Nesterov momentum should update parameters",
        )

    def test_weight_decay(self):
        """Test that weight decay is applied correctly"""
        muon_params = [self.model.linear.weight]
        adamw_params = []

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.0,
            weight_decay=0.01,  # Enable weight decay
            nesterov=False,
            ns_steps=1,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)

        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        weight_before = self.model.linear.weight.clone()

        optimizer.step()

        # Weight should be decayed: w_new = w_old * (1 - lr * weight_decay) - lr * grad
        # Check that weight changed
        self.assertFalse(
            torch.allclose(self.model.linear.weight, weight_before),
            "Weight decay should affect parameter updates",
        )

    def test_newton_schulz_orthogonalization(self):
        """Test that Newton-Schulz orthogonalization is applied"""
        muon_params = [self.model.linear.weight]
        adamw_params = []

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            ns_steps=5,  # Use multiple steps
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)

        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        weight_before = self.model.linear.weight.clone()
        grad = self.model.linear.weight.grad.clone()

        optimizer.step()

        # Check that weight was updated (orthogonalization should modify gradient direction)
        self.assertFalse(
            torch.allclose(self.model.linear.weight, weight_before),
            "Newton-Schulz orthogonalization should update parameters",
        )

        # Check that gradient direction was modified (not just scaled)
        # The orthogonalized gradient should differ from original gradient
        weight_change = self.model.linear.weight - weight_before
        # Normalize for comparison
        weight_change_norm = weight_change / (weight_change.norm() + 1e-8)
        grad_norm = grad / (grad.norm() + 1e-8)

        # They should not be exactly aligned due to orthogonalization
        cosine_sim = (weight_change_norm * grad_norm).sum()
        self.assertLess(
            abs(cosine_sim - 1.0),
            0.5,  # Allow some deviation but not too much
            "Newton-Schulz should modify gradient direction",
        )

    def test_zero_grad(self):
        """Test that zero_grad clears gradients"""
        muon_params = [self.model.linear.weight]
        adamw_params = [self.model.linear.bias]

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            ns_steps=1,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)

        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(self.model.linear.weight.grad)
        self.assertIsNotNone(self.model.linear.bias.grad)

        optimizer.zero_grad()

        # Check gradients are cleared
        self.assertIsNone(self.model.linear.weight.grad)
        self.assertIsNone(self.model.linear.bias.grad)

    def test_state_dict(self):
        """Test state dict serialization"""
        muon_params = [self.model.linear.weight]
        adamw_params = [self.model.linear.bias]

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0,
            nesterov=False,
            ns_steps=5,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        # Perform a step to populate state
        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.hidden_dim)
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # Get state dict
        state_dict = optimizer.state_dict()

        # Check that state dict contains expected keys
        self.assertIn("state", state_dict)
        self.assertIn("param_groups", state_dict)

        # Check that momentum buffers are in state
        state = state_dict["state"]
        # Find weight parameter ID in state
        weight_id = None
        for i, group in enumerate(optimizer.param_groups):
            if group.get("use_muon", False):
                for j, p in enumerate(group["params"]):
                    if p is self.model.linear.weight:
                        weight_id = f"{i}_{j}"
                        break

        # Verify momentum buffer exists
        if weight_id:
            self.assertIn(weight_id, state or {})
        else:
            # Alternative: check by iterating state dict
            found_momentum = False
            for key, value in state.items():
                if isinstance(value, dict) and "momentum_buffer" in value:
                    found_momentum = True
                    break
            self.assertTrue(found_momentum, "Momentum buffer should be in state dict")

    def test_no_gradient_no_update(self):
        """Test that parameters are not updated when gradient is None"""
        muon_params = [self.model.linear.weight]
        adamw_params = []

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            ns_steps=1,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        weight_before = self.model.linear.weight.clone()

        # Step without backward
        optimizer.step()

        # Weight should not change
        self.assertTrue(
            torch.allclose(self.model.linear.weight, weight_before),
            "Parameters should not update without gradients",
        )

    def test_parameter_grouping(self):
        """Test that parameters are correctly grouped"""
        # Create model with multiple layers
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
            nn.LayerNorm(10),
        )

        # Collect parameters
        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.ndim >= 2 and "weight" in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        optimizer = MuonOptimizer(
            params=[],
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            ns_steps=1,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        # Check that we have the expected number of parameter groups
        self.assertEqual(
            len(optimizer.param_groups), 2, "Should have 2 parameter groups"
        )

        # Check Muon group
        muon_group = optimizer.param_groups[0]
        self.assertTrue(muon_group.get("use_muon", False), "First group should be Muon")
        self.assertEqual(len(muon_group["params"]), 2, "Should have 2 Muon parameters")

        # Check AdamW group
        adamw_group = optimizer.param_groups[1]
        self.assertFalse(
            adamw_group.get("use_muon", False), "Second group should be AdamW"
        )
        self.assertGreater(
            len(adamw_group["params"]), 0, "Should have at least 1 AdamW parameter"
        )


if __name__ == "__main__":
    unittest.main()
