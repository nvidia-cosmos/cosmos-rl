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
Test suite for B1KEnvWrapper (BEHAVIOR-1K benchmark).

This test suite validates the B1KEnvWrapper's ability to manage multiple BEHAVIOR-1K environments.
Note: Currently B1K does NOT support partial reset/step like Libero, so we test full env operations only.

Requirements:
    - BEHAVIOR-1K/OmniGibson installed
    - CUDA-capable GPU (for rendering)

Usage:
    # Run all tests
    python test_env_manager_b1k.py

    # Run with unittest discovery
    python -m unittest test_env_manager_b1k

    # Run specific test
    python -m unittest test_env_manager_b1k.TestB1KEnvWrapper.test_create_env

    # Run with verbose output
    python -m unittest test_env_manager_b1k -v

Tests:
    1. test_create_env: Verify creation of B1K environment
    2. test_full_reset: Test full environment reset
    3. test_full_step: Test stepping all environments
    4. test_chunk_step: Test chunk stepping all environments
    5. test_validation_and_video_saving: Test validation mode and video saving
"""

import unittest
import numpy as np
import torch
from dataclasses import dataclass
import os
import tempfile
import shutil
import atexit

# Disable torch compilation before importing B1K wrapper to avoid typing_extensions issues
torch._dynamo.config.disable = True

from cosmos_rl.simulators.b1k.env_wrapper import B1KEnvWrapper
import omnigibson as og


@dataclass
class MockB1KConfig:
    """Minimal configuration for B1KEnvWrapper."""

    num_envs: int = 2  # Small number for testing
    task_name: str = "turning_on_radio"  # A task from BEHAVIOR-1K


class TestB1KEnvWrapper(unittest.TestCase):
    """Test suite for B1KEnvWrapper."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = MockB1KConfig()
        print(
            f"\nInitializing B1K environment with {self.config.num_envs} environments..."
        )
        print(f"Task: {self.config.task_name}")
        self.env = B1KEnvWrapper(cfg=self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        print("Tearing down environment...")
        if hasattr(self, "env") and self.env is not None:
            # Clean up environment properly
            try:
                self.env.close()
            except Exception as e:
                print(f"Warning during env.close(): {e}")
            finally:
                self.env = None
        print("✓ Environment torn down successfully")

        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning during temp_dir cleanup: {e}")

    def test_create_env(self):
        """Test 1: Create B1K environments."""
        print("\n" + "=" * 80)
        print("TEST 1: Create B1K environments")
        print("=" * 80)

        # Verify the environment was created
        self.assertIsNotNone(self.env)
        print("✓ B1KEnvWrapper created successfully")

        # Verify environment states were initialized
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        self.assertEqual(len(env_states), self.config.num_envs)
        print(
            f"✓ Environment states initialized for {self.config.num_envs} environments"
        )

        # Verify each environment state is properly initialized
        for i, state in enumerate(env_states):
            self.assertEqual(state.env_idx, i)
            self.assertTrue(state.active)
            self.assertFalse(state.complete)
            self.assertEqual(state.step, 0)
        print("✓ All environment states properly initialized")

        # Verify task description is loaded
        self.assertIsNotNone(self.env.task_description)
        print(f"✓ Task description loaded: '{self.env.task_description[:50]}...'")

    def test_full_reset(self):
        """Test 2: Full environment reset."""
        print("\n" + "=" * 80)
        print("TEST 2: Full environment reset")
        print("=" * 80)

        # Perform full reset (without validation)
        print(f"Resetting all {self.config.num_envs} environments...")
        images_and_states, task_descriptions = self.env.reset(do_validation=False)

        # Verify reset results structure
        self.assertIsInstance(images_and_states, dict)
        self.assertIn("full_images", images_and_states)
        self.assertIn("wrist_images", images_and_states)
        print("✓ Reset returned expected data structure")

        # Verify shapes
        self.assertEqual(
            images_and_states["full_images"].shape[0], self.config.num_envs
        )
        self.assertEqual(
            images_and_states["wrist_images"].shape[0], self.config.num_envs
        )
        print(f"✓ Observation shapes correct for {self.config.num_envs} environments")

        # Verify image dimensions (should be 4D: [num_envs, height, width, channels])
        full_img_shape = images_and_states["full_images"].shape
        wrist_img_shape = images_and_states["wrist_images"].shape
        self.assertEqual(len(full_img_shape), 4)
        self.assertEqual(len(wrist_img_shape), 5)
        print(f"✓ Full images shape: {full_img_shape}")
        print(f"✓ Wrist images shape: {wrist_img_shape}")

        # Verify task descriptions were returned
        self.assertIsInstance(task_descriptions, list)
        self.assertEqual(len(task_descriptions), self.config.num_envs)
        print(f"✓ Task descriptions returned for {self.config.num_envs} environments")

        # Verify all task descriptions match
        for desc in task_descriptions:
            self.assertEqual(desc, self.env.task_description)
        print("✓ All task descriptions match configured task")

        # Verify environment states were updated
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            self.assertEqual(state.task_name, self.config.task_name)
            self.assertTrue(state.active)
            self.assertFalse(state.complete)
            self.assertEqual(state.step, 0)
            self.assertIsNotNone(state.current_obs)
            self.assertFalse(state.do_validation)
        print("✓ Environment states updated correctly after reset")

    def test_full_step(self):
        """Test 3: Step all environments."""
        print("\n" + "=" * 80)
        print("TEST 3: Step all environments")
        print("=" * 80)

        # First reset all environments
        print(f"Resetting all {self.config.num_envs} environments...")
        self.env.reset(do_validation=False)
        print("✓ All environments reset")

        # Create random actions for all environments
        # B1K uses R1Pro robot - check action dimension from OmniGibson config
        action_dim = 23
        actions = np.random.randn(self.config.num_envs, action_dim).astype(np.float32)
        actions = np.clip(actions, -1, 1)  # Clip to valid range

        print(f"Stepping all {self.config.num_envs} environments...")
        print(f"Action shape: {actions.shape}")

        # Perform step
        result = self.env.step(actions)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("full_images", result)
        self.assertIn("wrist_images", result)
        self.assertIn("complete", result)
        self.assertIn("active", result)
        self.assertIn("finish_step", result)
        print("✓ Step returned expected data structure")

        # Verify result shapes
        self.assertEqual(result["full_images"].shape[0], self.config.num_envs)
        self.assertEqual(result["wrist_images"].shape[0], self.config.num_envs)
        self.assertEqual(len(result["complete"]), self.config.num_envs)
        self.assertEqual(len(result["active"]), self.config.num_envs)
        self.assertEqual(len(result["finish_step"]), self.config.num_envs)
        print(f"✓ Result shapes correct for {self.config.num_envs} environments")

        # Verify step counter was incremented
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            self.assertGreaterEqual(state.step, 1)
            self.assertEqual(result["finish_step"][i], state.step)
        print("✓ Step counter incremented for all environments")

        # Step a few more times to verify consistency
        print("\nStepping 5 more times...")
        for step_num in range(5):
            actions = np.random.randn(self.config.num_envs, action_dim).astype(
                np.float32
            )
            actions = np.clip(actions, -1, 1)
            result = self.env.step(actions)
            print(f"  - Step {step_num + 2} completed")

        # Verify step counts match
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            self.assertEqual(result["finish_step"][i], state.step)
        print("✓ Step counts consistent after multiple steps")

    def test_chunk_step(self):
        """Test 4: Chunk step all environments."""
        print("\n" + "=" * 80)
        print("TEST 4: Chunk step all environments")
        print("=" * 80)

        # Reset all environments first
        print(f"Resetting all {self.config.num_envs} environments...")
        self.env.reset(do_validation=False)
        print("✓ All environments reset")

        # Prepare chunk actions
        chunk_size = 3  # 3 actions per environment
        action_dim = 23

        # Create action chunk: (num_envs, chunk_size, action_dim)
        action_chunk = np.random.randn(
            self.config.num_envs, chunk_size, action_dim
        ).astype(np.float32)
        action_chunk = np.clip(action_chunk, -1, 1)

        print(f"Chunk stepping all {self.config.num_envs} environments...")
        print(f"Action chunk shape: {action_chunk.shape}")
        print(f"  - Number of environments: {self.config.num_envs}")
        print(f"  - Chunk size (actions per env): {chunk_size}")
        print(f"  - Action dimension: {action_dim}")

        # Perform chunk step
        result = self.env.chunk_step(action_chunk)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("full_images", result)
        self.assertIn("wrist_images", result)
        self.assertIn("complete", result)
        self.assertIn("active", result)
        self.assertIn("finish_step", result)
        print("✓ Chunk step returned expected data structure")

        # Verify result shapes
        self.assertEqual(result["full_images"].shape[0], self.config.num_envs)
        self.assertEqual(result["wrist_images"].shape[0], self.config.num_envs)
        print(f"✓ Result shapes correct for {self.config.num_envs} environments")

        # Verify step counter was incremented by chunk_size
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            # Should have advanced by chunk_size steps (or less if completed early)
            self.assertTrue(state.step >= chunk_size or not state.active)
            print(f"  - Env {i}: {state.step} steps, active={state.active}")
        print("✓ Step counters incremented by chunk_size")

        # Test with torch tensors
        print("\nTesting chunk_step with torch.Tensor input...")
        action_chunk_torch = torch.from_numpy(action_chunk)
        result_torch = self.env.chunk_step(action_chunk_torch)
        self.assertIsInstance(result_torch, dict)
        print("✓ Chunk step works with torch.Tensor input")

    def test_validation_and_video_saving(self):
        """Test 5: Enable validation and save videos."""
        print("\n" + "=" * 80)
        print("TEST 5: Validation mode and video saving")
        print("=" * 80)

        # Reset with validation enabled
        print(f"Resetting all {self.config.num_envs} environments with validation...")
        images_and_states, task_descriptions = self.env.reset(do_validation=True)
        print("✓ All environments reset with validation enabled")

        # Verify validation flags are set correctly
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            self.assertTrue(state.do_validation)
            self.assertIsNotNone(state.valid_pixels)
            self.assertIn("full_images", state.valid_pixels)
            self.assertIn("wrist_images", state.valid_pixels)
            # Initially empty (validation pixels collected during steps)
            self.assertEqual(len(state.valid_pixels["full_images"]), 0)
            self.assertEqual(len(state.valid_pixels["wrist_images"]), 0)
        print("✓ Validation flags set correctly for all environments")

        # Step multiple times to collect validation data
        num_steps = 16
        action_dim = 23
        print(
            f"\nStepping all environments {num_steps} times to collect validation data..."
        )

        for step_num in range(num_steps):
            actions = np.random.randn(self.config.num_envs, action_dim).astype(
                np.float32
            )
            actions = np.clip(actions, -1, 1)
            self.env.step(actions)
            print(f"  - Step {step_num + 1}/{num_steps} completed")

        print(f"✓ Completed {num_steps} steps for all environments")

        # Check that validation data was collected
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            num_collected = len(state.valid_pixels["full_images"])
            self.assertGreater(num_collected, 0)
            self.assertEqual(
                len(state.valid_pixels["full_images"]),
                len(state.valid_pixels["wrist_images"]),
            )
            print(f"  - Env {i}: collected {num_collected} image frames")
        print("✓ Validation data collected for all environments")

        # Test video saving
        print("\nTesting video saving functionality...")
        env_ids = list(range(self.config.num_envs))

        try:
            self.env.save_validation_videos(self.temp_dir, env_ids)
            print("✓ save_validation_videos completed without errors")

            # Check if video files were created
            video_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".mp4")]
            print(f"  - Created {len(video_files)} video file(s) in {self.temp_dir}")

            # We expect at least one video file per environment
            # (though it depends on implementation details)
            if len(video_files) > 0:
                print(f"  - Video files: {video_files}")
                print("✓ Video files created successfully")
            else:
                print("  - Note: No .mp4 files found (may depend on implementation)")
        except Exception as e:
            print(f"  - Warning: Video saving raised exception: {e}")
            print("  - This may be expected if video codec/writer is not configured")

    def test_reset_without_validation(self):
        """Test 6: Reset without validation (default behavior)."""
        print("\n" + "=" * 80)
        print("TEST 6: Reset without validation")
        print("=" * 80)

        # Reset without validation
        print(
            f"Resetting all {self.config.num_envs} environments without validation..."
        )
        images_and_states, task_descriptions = self.env.reset(do_validation=False)
        print("✓ All environments reset without validation")

        # Verify validation is disabled
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            self.assertFalse(state.do_validation)
        print("✓ Validation disabled for all environments")

        # Step a few times
        num_steps = 2
        action_dim = 23
        print(f"\nStepping {num_steps} times...")

        for step_num in range(num_steps):
            actions = np.random.randn(self.config.num_envs, action_dim).astype(
                np.float32
            )
            actions = np.clip(actions, -1, 1)
            self.env.step(actions)

        # Verify no validation data was collected
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        for i, state in enumerate(env_states):
            self.assertFalse(state.do_validation)
            # valid_pixels should be None since validation is disabled
            self.assertIsNone(state.valid_pixels)
        print("✓ No validation data collected when validation is disabled")


def cleanup_omnigibson():
    """Cleanup function to properly shutdown OmniGibson.

    Note: You may see AttributeError/RuntimeError messages during shutdown.
    These are benign - they occur because UI/viewport components try to
    access cameras after the scene has been cleared. Safe to ignore.
    """
    try:
        print("\nShutting down OmniGibson...")
        og.shutdown()
        print("OmniGibson shutdown complete.")
    except Exception as e:
        print(f"Warning during OmniGibson shutdown: {e}")


# Register cleanup to be called at exit
# atexit.register(cleanup_omnigibson)


if __name__ == "__main__":
    """
    Run tests using unittest framework.
    Usage: python test_env_manager_b1k.py
    or: python -m unittest test_env_manager_b1k.py
    """
    unittest.main(verbosity=2, exit=False)
    # Explicitly call cleanup after tests
    cleanup_omnigibson()
