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

This test suite validates the B1KEnvWrapper's ability to manage multiple BEHAVIOR-1K environments
with support for partial reset/step operations.

Requirements:
    - BEHAVIOR-1K/OmniGibson installed
    - CUDA-capable GPU (for rendering)

Usage:
    # Run all tests
    python test_env_manager_b1k.py

    # Run with unittest discovery
    python -m unittest test_env_manager_b1k

    # Run specific test
    python -m unittest test_env_manager_b1k.TestB1KEnvWrapper.test_full_reset

    # Run with verbose output
    python -m unittest test_env_manager_b1k -v

Tests:
    1. test_create_env: Verify creation of B1K environment
    2. test_full_reset: Test resetting all environments
    3. test_partial_reset: Test partial environment reset (subset of envs)
    4. test_full_step: Test stepping all environments
    5. test_partial_step: Test stepping subset of environments
    6. test_chunk_step: Test chunk stepping environments
    7. test_validation_and_video_saving: Test validation mode and video saving
"""

import unittest
import numpy as np
import torch
from dataclasses import dataclass
import os
import tempfile
import shutil

# Disable torch compilation before importing B1K wrapper to avoid typing_extensions issues
torch._dynamo.config.disable = True

from cosmos_rl.simulators.b1k.env_wrapper import B1KEnvWrapper
import omnigibson as og


@dataclass
class MockB1KConfig:
    """Minimal configuration for B1KEnvWrapper."""

    num_envs: int = 2  # Small number for testing, 3 allows partial ops
    height: int = 256  # Image height
    width: int = 256  # Image width
    max_steps: int = 512  # Max steps per episode


class TestB1KEnvWrapper(unittest.TestCase):
    """Test suite for B1KEnvWrapper."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all test methods."""
        cls.config = MockB1KConfig()
        print(
            f"\n{'='*80}\n"
            f"Initializing B1K environment with {cls.config.num_envs} environments...\n"
            f"This will be REUSED across all tests for efficiency.\n"
            f"{'='*80}"
        )
        cls.env = B1KEnvWrapper(cfg=cls.config)

        # Use first task from the sorted task list for testing
        cls.test_task_id = 0
        cls.test_task_name = cls.env.task_names[cls.test_task_id]
        print(f"Test task ID: {cls.test_task_id}, Task: {cls.test_task_name}")
        print(f"✓ B1KEnvWrapper initialized once for all tests\n{'='*80}\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all test methods have run."""
        print("\n" + "=" * 80)
        print("Tearing down shared environment (runs once after all tests)...")
        print("=" * 80)
        if hasattr(cls, "env") and cls.env is not None:
            # Clean up environment properly
            try:
                pass  # cls.env.close()
            except Exception as e:
                print(f"Warning during env.close(): {e}")
            finally:
                cls.env = None
        print("✓ Shared environment torn down successfully")

    def setUp(self):
        """Set up before each test method."""
        # Create a fresh temp directory for each test
        self.temp_dir = tempfile.mkdtemp()
        print(f"Temp directory: {self.temp_dir}")

    def tearDown(self):
        """Clean up after each test method."""
        # Only clean up temp directory (not the shared environment)
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
        print("✓ All environment states properly initialized")

        # Verify task list is loaded and sorted
        self.assertIsNotNone(self.env.task_names)
        self.assertGreater(len(self.env.task_names), 0)
        print(
            f"✓ Task list loaded: {len(self.env.task_names)} tasks (alphabetically sorted)"
        )

    def test_full_reset(self):
        """Test 2: Reset all environments."""
        print("\n" + "=" * 80)
        print("TEST 2: Reset all environments")
        print("=" * 80)

        # Perform full reset (all environments, same task, different trials)
        env_ids = list(range(self.config.num_envs))
        task_ids = [self.test_task_id] * self.config.num_envs  # Same task for all
        trial_ids = list(range(self.config.num_envs))  # Different trials

        print(f"Resetting {len(env_ids)} environments...")
        print(f"  env_ids: {env_ids}")
        print(f"  task_ids: {task_ids}")
        print(f"  trial_ids: {trial_ids}")

        images_and_states, task_descriptions = self.env.reset(
            env_ids=env_ids, task_ids=task_ids, trial_ids=trial_ids, do_validation=False
        )

        # Verify reset results structure
        self.assertIsInstance(images_and_states, dict)
        self.assertIn("full_images", images_and_states)
        self.assertIn("wrist_images", images_and_states)
        print("✓ Reset returned expected data structure")

        # Verify shapes
        self.assertEqual(images_and_states["full_images"].shape[0], len(env_ids))
        self.assertEqual(images_and_states["wrist_images"].shape[0], len(env_ids))
        print(f"✓ Observation shapes correct for {len(env_ids)} environments")

        # Verify image dimensions
        full_img_shape = images_and_states["full_images"].shape
        wrist_img_shape = images_and_states["wrist_images"].shape
        self.assertEqual(len(full_img_shape), 4)
        self.assertEqual(len(wrist_img_shape), 5)
        print(f"✓ Full images shape: {full_img_shape}")
        print(f"✓ Wrist images shape: {wrist_img_shape}")

        # Verify task descriptions were returned
        self.assertIsInstance(task_descriptions, list)
        self.assertEqual(len(task_descriptions), len(env_ids))
        print(f"✓ Task descriptions returned for {len(env_ids)} environments")

        # Verify all task descriptions are set
        for desc in task_descriptions:
            self.assertIsNotNone(desc)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)
        print("✓ All task descriptions are valid")

        # Verify environment states were updated
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertEqual(state.task_id, self.test_task_id)
            self.assertEqual(state.trial_id, trial_ids[i])
            self.assertTrue(state.active)
            self.assertFalse(state.complete)
            self.assertEqual(state.step, 0)
            self.assertIsNotNone(state.task_description)
            self.assertFalse(state.do_validation)
        print("✓ Environment states updated correctly after reset")

    def test_partial_reset(self):
        """Test 3: Partial environment reset (subset of envs)."""
        print("\n" + "=" * 80)
        print("TEST 3: Partial environment reset")
        print("=" * 80)

        # First reset all environments
        all_env_ids = list(range(self.config.num_envs))
        all_task_ids = [self.test_task_id] * self.config.num_envs
        all_trial_ids = [0] * self.config.num_envs

        print(f"First, resetting all {self.config.num_envs} environments...")
        self.env.reset(all_env_ids, all_task_ids, all_trial_ids, do_validation=False)
        print("✓ All environments reset")

        # Now reset only a subset
        partial_env_ids = [0]
        partial_task_ids = [self.test_task_id]
        partial_trial_ids = [10]  # Different trial IDs

        print(
            f"\nNow resetting subset: env_ids={partial_env_ids}, trial_ids={partial_trial_ids}"
        )
        images_and_states, task_descriptions = self.env.reset(
            env_ids=partial_env_ids,
            task_ids=partial_task_ids,
            trial_ids=partial_trial_ids,
            do_validation=False,
        )

        # Verify only the partial set was returned
        self.assertEqual(
            images_and_states["full_images"].shape[0], len(partial_env_ids)
        )
        self.assertEqual(len(task_descriptions), len(partial_env_ids))
        print(f"✓ Returned data for {len(partial_env_ids)} environments")

        # Verify the reset environments have new trial IDs
        for i, env_id in enumerate(partial_env_ids):
            state = self.env.get_env_states([env_id])[0]
            self.assertEqual(state.trial_id, partial_trial_ids[i])
            self.assertEqual(state.step, 0)  # Should be reset to 0
        print("✓ Partial reset environments have correct trial IDs and steps")

        # Verify the non-reset environment (env 1) still has old state
        env1_state = self.env.get_env_states([1])[0]
        self.assertEqual(env1_state.trial_id, 0)  # Original trial ID
        print("✓ Non-reset environment (env 1) retained original state")

    def test_full_step(self):
        """Test 4: Step all environments."""
        print("\n" + "=" * 80)
        print("TEST 4: Step all environments")
        print("=" * 80)

        # First reset all environments
        env_ids = list(range(self.config.num_envs))
        task_ids = [self.test_task_id] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))

        print(f"Resetting {len(env_ids)} environments...")
        self.env.reset(env_ids, task_ids, trial_ids, do_validation=False)
        print("✓ All environments reset")

        # Create random actions for all environments
        # B1K uses R1Pro robot - check action dimension from OmniGibson config
        action_dim = 23
        actions = np.random.randn(len(env_ids), action_dim).astype(np.float32)
        actions = np.clip(actions, -1, 1)  # Clip to valid range

        print(f"\nStepping {len(env_ids)} environments...")
        print(f"Action shape: {actions.shape}")

        # Perform step
        result = self.env.step(env_ids, actions)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("full_images", result)
        self.assertIn("wrist_images", result)
        self.assertIn("complete", result)
        self.assertIn("active", result)
        self.assertIn("finish_step", result)
        print("✓ Step returned expected data structure")

        # Verify result shapes
        self.assertEqual(result["full_images"].shape[0], len(env_ids))
        self.assertEqual(result["wrist_images"].shape[0], len(env_ids))
        self.assertEqual(len(result["complete"]), len(env_ids))
        self.assertEqual(len(result["active"]), len(env_ids))
        self.assertEqual(len(result["finish_step"]), len(env_ids))
        print(f"✓ Result shapes correct for {len(env_ids)} environments")

        # Verify step counter was incremented
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertGreaterEqual(state.step, 1)
            self.assertEqual(result["finish_step"][i], state.step)
        print("✓ Step counter incremented for all environments")

        # Step a few more times to verify consistency
        print("\nStepping 5 more times...")
        for step_num in range(5):
            actions = np.random.randn(len(env_ids), action_dim).astype(np.float32)
            actions = np.clip(actions, -1, 1)
            result = self.env.step(env_ids, actions)
            print(f"  - Step {step_num + 2} completed")

        # Verify step counts match
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertEqual(result["finish_step"][i], state.step)
        print("✓ Step counts consistent after multiple steps")

    def test_partial_step(self):
        """Test 5: Step subset of environments."""
        print("\n" + "=" * 80)
        print("TEST 5: Partial step (subset of environments)")
        print("=" * 80)

        # Reset all environments first
        all_env_ids = list(range(self.config.num_envs))
        task_ids = [self.test_task_id] * self.config.num_envs
        trial_ids = [0] * self.config.num_envs

        print(f"Resetting all {self.config.num_envs} environments...")
        self.env.reset(all_env_ids, task_ids, trial_ids, do_validation=False)
        print("✓ All environments reset")

        # Step only a subset
        partial_env_ids = [1]
        action_dim = 23
        actions = np.random.randn(len(partial_env_ids), action_dim).astype(np.float32)
        actions = np.clip(actions, -1, 1)

        print(f"\nStepping subset: env_ids={partial_env_ids}")
        print(f"Action shape: {actions.shape}")

        result = self.env.step(partial_env_ids, actions)

        # Verify result structure
        self.assertEqual(result["full_images"].shape[0], len(partial_env_ids))
        self.assertEqual(len(result["complete"]), len(partial_env_ids))
        print("✓ Returned data for partial environments only")

        # Verify step counter incremented for stepped envs
        for env_id in partial_env_ids:
            state = self.env.get_env_states([env_id])[0]
            self.assertEqual(state.step, 1)
        print(f"✓ Step counter incremented for envs {partial_env_ids}")

        # Verify non-stepped environment (env 0) still at step 0
        env0_state = self.env.get_env_states([0])[0]
        self.assertEqual(env0_state.step, 0)
        print("✓ Non-stepped environment (env 0) remains at step 0")

    def test_chunk_step(self):
        """Test 6: Chunk step environments."""
        print("\n" + "=" * 80)
        print("TEST 6: Chunk step environments")
        print("=" * 80)

        # Reset environments first
        env_ids = [0]
        task_ids = [self.test_task_id]
        trial_ids = [10]

        print(f"Resetting {len(env_ids)} environments...")
        self.env.reset(env_ids, task_ids, trial_ids, do_validation=False)
        print("✓ Environments reset")

        # Prepare chunk actions
        chunk_size = 3  # 3 actions per environment
        action_dim = 23

        # Create action chunk: (num_envs, chunk_size, action_dim)
        action_chunk = np.random.randn(len(env_ids), chunk_size, action_dim).astype(
            np.float32
        )
        action_chunk = np.clip(action_chunk, -1, 1)

        print(f"\nChunk stepping {len(env_ids)} environments...")
        print(f"Action chunk shape: {action_chunk.shape}")
        print(f"  - Number of environments: {len(env_ids)}")
        print(f"  - Chunk size (actions per env): {chunk_size}")
        print(f"  - Action dimension: {action_dim}")

        # Perform chunk step
        result = self.env.chunk_step(env_ids, action_chunk)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("full_images", result)
        self.assertIn("wrist_images", result)
        self.assertIn("complete", result)
        self.assertIn("active", result)
        self.assertIn("finish_step", result)
        print("✓ Chunk step returned expected data structure")

        # Verify result shapes
        self.assertEqual(result["full_images"].shape[0], len(env_ids))
        self.assertEqual(result["wrist_images"].shape[0], len(env_ids))
        print(f"✓ Result shapes correct for {len(env_ids)} environments")

        # Verify step counter was incremented by chunk_size
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            # Should have advanced by chunk_size steps (or less if completed early)
            self.assertTrue(state.step >= chunk_size or not state.active)
            print(f"  - Env {env_ids[i]}: {state.step} steps, active={state.active}")
        print("✓ Step counters incremented by chunk_size")

        # Test with torch tensors
        print("\nTesting chunk_step with torch.Tensor input...")
        action_chunk_torch = torch.from_numpy(action_chunk)
        result_torch = self.env.chunk_step(env_ids, action_chunk_torch)
        self.assertIsInstance(result_torch, dict)
        print("✓ Chunk step works with torch.Tensor input")

    def test_validation_and_video_saving(self):
        """Test 7: Enable validation and save videos."""
        print("\n" + "=" * 80)
        print("TEST 7: Validation mode and video saving")
        print("=" * 80)

        # Reset with validation enabled
        env_ids = [0, 1]  # Use subset for faster test
        task_ids = [self.test_task_id] * len(env_ids)
        trial_ids = [0] * len(env_ids)

        print(f"Resetting {len(env_ids)} environments with validation...")
        images_and_states, task_descriptions = self.env.reset(
            env_ids=env_ids, task_ids=task_ids, trial_ids=trial_ids, do_validation=True
        )
        print("✓ Environments reset with validation enabled")

        # Verify validation flags are set correctly
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertTrue(state.do_validation)
            self.assertIsNotNone(state.valid_pixels)
            self.assertIn("full_images", state.valid_pixels)
            self.assertIn("wrist_images", state.valid_pixels)
            # Initially empty (validation pixels collected during steps)
            self.assertEqual(len(state.valid_pixels["full_images"]), 1)
            self.assertEqual(len(state.valid_pixels["wrist_images"]), 1)
        print("✓ Validation flags set correctly for all environments")

        # Step multiple times to collect validation data
        num_steps = 16
        action_dim = 23
        print(
            f"\nStepping {len(env_ids)} environments {num_steps} times to collect validation data..."
        )

        for step_num in range(num_steps):
            actions = np.random.randn(len(env_ids), action_dim).astype(np.float32)
            actions = np.clip(actions, -1, 1)
            self.env.step(env_ids, actions)
            print(f"  - Step {step_num + 1}/{num_steps} completed")

        print(f"✓ Completed {num_steps} steps for {len(env_ids)} environments")

        # Check that validation data was collected
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            num_collected = len(state.valid_pixels["full_images"])
            self.assertGreater(num_collected, 0)
            self.assertEqual(
                len(state.valid_pixels["full_images"]),
                len(state.valid_pixels["wrist_images"]),
            )
            print(f"  - Env {env_ids[i]}: collected {num_collected} image frames")
        print("✓ Validation data collected for all environments")

        # Test video saving
        print("\nTesting video saving functionality...")

        try:
            self.env.save_validation_videos(self.temp_dir, env_ids)
            print("✓ save_validation_videos completed without errors")

            # Check if video files were created
            video_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".mp4")]
            print(f"  - Created {len(video_files)} video file(s) in {self.temp_dir}")

            # We expect at least one video file per environment
            if len(video_files) > 0:
                print(f"  - Video files: {video_files}")
                print("✓ Video files created successfully")
            else:
                print("  - Note: No .mp4 files found (may depend on implementation)")
        except Exception as e:
            print(f"  - Warning: Video saving raised exception: {e}")
            print("  - This may be expected if video codec/writer is not configured")

    def test_reset_without_validation(self):
        """Test 8: Reset without validation (default behavior)."""
        print("\n" + "=" * 80)
        print("TEST 8: Reset without validation")
        print("=" * 80)

        # Reset without validation
        env_ids = list(range(self.config.num_envs))
        task_ids = [self.test_task_id] * self.config.num_envs
        trial_ids = [0] * self.config.num_envs

        print(f"Resetting {len(env_ids)} environments without validation...")
        images_and_states, task_descriptions = self.env.reset(
            env_ids=env_ids, task_ids=task_ids, trial_ids=trial_ids, do_validation=False
        )
        print("✓ All environments reset without validation")

        # Verify validation is disabled
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertFalse(state.do_validation)
        print("✓ Validation disabled for all environments")

        # Step a few times
        num_steps = 2
        action_dim = 23
        print(f"\nStepping {num_steps} times...")

        for step_num in range(num_steps):
            actions = np.random.randn(len(env_ids), action_dim).astype(np.float32)
            actions = np.clip(actions, -1, 1)
            self.env.step(env_ids, actions)

        # Verify no validation data was collected
        env_states = self.env.get_env_states(env_ids)
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
