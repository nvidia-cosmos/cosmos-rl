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
Test suite for EnvManager wrapping LiberoEnvWrapper.

This test suite validates the EnvManager's ability to manage multiple Libero environments,
including creating vectorized environments, resetting subsets, stepping subsets, and 
chunk stepping subsets.

Requirements:
    - libero package installed
    - CUDA-capable GPU (for rendering)

Usage:
    # Run all tests
    python test_env_manager_libero.py
    
    # Run with unittest discovery
    python -m unittest test_env_manager_libero
    
    # Run specific test
    python -m unittest test_env_manager_libero.TestEnvManagerLibero.test_create_vector_envs
    
    # Run with verbose output
    python -m unittest test_env_manager_libero -v

Tests:
    1. test_create_vector_envs: Verify creation of a vector of environments
    2. test_reset_subset_envs: Test resetting a subset of environments
    3. test_step_subset_envs: Test stepping a subset of environments
    4. test_chunk_step_subset_envs: Test chunk stepping a subset of environments
    5. test_validation_subset_and_get_valid_pixels: Test validation in subset and get_valid_pixels
"""

import unittest
import numpy as np
import torch
from dataclasses import dataclass

from cosmos_rl.simulators.env_manager import EnvManager
from cosmos_rl.simulators.libero.env_wrapper import LiberoEnvWrapper


@dataclass
class MockLiberoConfig:
    """Minimal configuration for LiberoEnvWrapper."""
    num_envs: int = 4
    seed: int = 42
    task_suite_name: str = "libero_10"  # Smallest suite for testing
    height: int = 256
    width: int = 256
    max_steps: int = 100


class TestEnvManagerLibero(unittest.TestCase):
    """Test suite for EnvManager wrapping LiberoEnvWrapper."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = MockLiberoConfig()
        self.env_manager = EnvManager(
            cfg=self.config,
            rank=0,
            num_envs=self.config.num_envs,
            seed_offset=0,
            total_num_processes=1,
            env_cls=LiberoEnvWrapper,
        )
        self.env_manager.start_simulator()
    
    def tearDown(self):
        """Clean up after each test method."""
        self.env_manager.stop_simulator()

    
    def test_create_vector_envs(self):
        """Test 1: Create a vector of environments."""
        print("\n" + "="*80)
        print("TEST 1: Create a vector of environments")
        print("="*80)
        
        # Verify the environment manager was created
        self.assertIsNotNone(self.env_manager)
        print(f"✓ EnvManager created successfully")
        
        # Verify number of environments
        self.assertEqual(self.env_manager.num_envs, self.config.num_envs)
        print(f"✓ Number of environments: {self.config.num_envs}")
        
        # Verify task states were initialized
        env_states = self.env_manager.get_env_states(list(range(self.config.num_envs)))
        self.assertEqual(len(env_states), self.config.num_envs)
        print(f"✓ Task states initialized for {self.config.num_envs} environments")
        
        # Verify each environment state is properly initialized
        for i, state in enumerate(env_states):
            self.assertEqual(state.env_idx, i)
            self.assertEqual(state.task_id, -1)  # Not yet configured
            self.assertTrue(state.active)
            self.assertFalse(state.complete)
        print(f"✓ All environment states properly initialized")
    
    def test_reset_subset_envs(self):
        """Test 2: Reset a subset of environments."""
        print("\n" + "="*80)
        print("TEST 2: Reset a subset of environments")
        print("="*80)
        
        # Reset first 2 environments (subset)
        subset_env_ids = [0, 1]
        task_ids = [0, 0]  # Use same task for simplicity
        trial_ids = [0, 1]  # Different trials
        do_validation = [False, False]
        
        print(f"Resetting environment IDs: {subset_env_ids}")
        print(f"Task IDs: {task_ids}")
        print(f"Trial IDs: {trial_ids}")
        
        # Perform reset
        images_and_states, task_descriptions = self.env_manager.reset(
            env_ids=subset_env_ids,
            task_ids=task_ids,
            trial_ids=trial_ids,
            do_validataion=do_validation,
        )
        
        # Verify reset results structure
        self.assertIsInstance(images_and_states, dict)
        self.assertIn("full_images", images_and_states)
        self.assertIn("wrist_images", images_and_states)
        self.assertIn("states", images_and_states)
        print(f"✓ Reset returned expected data structure")
        
        # Verify shapes
        self.assertEqual(images_and_states["full_images"].shape[0], len(subset_env_ids))
        self.assertEqual(images_and_states["wrist_images"].shape[0], len(subset_env_ids))
        self.assertEqual(images_and_states["states"].shape[0], len(subset_env_ids))
        print(f"✓ Observation shapes correct for subset of {len(subset_env_ids)} envs")
        
        # Verify image dimensions
        self.assertEqual(images_and_states["full_images"].shape[1:], (self.config.height, self.config.width, 3))
        self.assertEqual(images_and_states["wrist_images"].shape[1:], (self.config.height, self.config.width, 3))
        print(f"✓ Image dimensions: {self.config.height}x{self.config.width}x3")
        
        # Verify task descriptions were returned
        self.assertIsInstance(task_descriptions, list)
        self.assertEqual(len(task_descriptions), len(subset_env_ids))
        print(f"✓ Task descriptions returned for {len(subset_env_ids)} environments")
        
        # Verify task states were updated for reset environments
        env_states = self.env_manager.get_env_states(subset_env_ids)
        for i, env_id in enumerate(subset_env_ids):
            state = env_states[i]
            self.assertEqual(state.task_id, task_ids[i])
            self.assertEqual(state.trial_id, trial_ids[i])
            self.assertTrue(state.active)
            self.assertEqual(state.step, 0)
            self.assertIsNotNone(state.current_obs)
        print(f"✓ Task states updated correctly for reset environments")
        
        # Verify non-reset environments were not affected
        non_reset_env_states = self.env_manager.get_env_states(list(range(self.config.num_envs)))
        for env_id in range(self.config.num_envs):
            if env_id not in subset_env_ids:
                state = self.env_manager.get_env_states([env_id])[0]
                self.assertEqual(state.task_id, -1)  # Still uninitialized
        print(f"✓ Non-reset environments remain unchanged")
    
    def test_step_subset_envs(self):
        """Test 3: Step a subset of environments."""
        print("\n" + "="*80)
        print("TEST 3: Step a subset of environments")
        print("="*80)
        
        # First reset all environments
        all_env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        do_validation = [False] * self.config.num_envs
        
        print(f"Resetting all {self.config.num_envs} environments...")
        self.env_manager.reset(
            env_ids=all_env_ids,
            task_ids=task_ids,
            trial_ids=trial_ids,
            do_validataion=do_validation,
        )
        print(f"✓ All environments reset")
        
        # Step a subset of environments
        step_env_ids = [0, 2, 3]  # Step envs 0, 2, 3 (not 1)
        
        # Random actions in [0, 1], gripper is binary [-1, 1]
        actions = np.random.rand(len(step_env_ids), 7).astype(np.float32)
        actions[:, -1] = np.sign(actions[:, -1] - 0.5)
        
        print(f"Stepping environment IDs: {step_env_ids}")
        print(f"Action shape: {actions.shape}")
        
        # Perform step
        result = self.env_manager.step(env_ids=step_env_ids, action=actions)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("full_images", result)
        self.assertIn("wrist_images", result)
        self.assertIn("states", result)
        self.assertIn("completes", result)
        self.assertIn("active", result)
        self.assertIn("finish_steps", result)
        print(f"✓ Step returned expected data structure")
        
        # Verify result shapes match the subset
        self.assertEqual(result["full_images"].shape[0], len(step_env_ids))
        self.assertEqual(result["wrist_images"].shape[0], len(step_env_ids))
        self.assertEqual(result["states"].shape[0], len(step_env_ids))
        self.assertEqual(len(result["completes"]), len(step_env_ids))
        self.assertEqual(len(result["active"]), len(step_env_ids))
        self.assertEqual(len(result["finish_steps"]), len(step_env_ids))
        print(result["finish_steps"])
        print(f"✓ Result shapes correct for subset of {len(step_env_ids)} envs")
        
        # Verify step counter was incremented for stepped environments
        env_states = self.env_manager.get_env_states(step_env_ids)
        for i, env_id in enumerate(step_env_ids):
            state = env_states[i]
            self.assertGreaterEqual(state.step, 1)
        print(f"✓ Step counter incremented for stepped environments")
        
        # Verify non-stepped environment (env 1) remains at step 0
        non_stepped_env_state = self.env_manager.get_env_states([1])[0]
        self.assertEqual(non_stepped_env_state.step, 0)
        print(f"✓ Non-stepped environment remains at step 0")
        
        # Verify active flags are correct
        for i, env_id in enumerate(step_env_ids):
            # Should be active if not done and below max_steps
            state = env_states[i]
            if state.step < self.config.max_steps and not state.complete:
                self.assertTrue(result["active"][i])
        print(f"✓ Active flags correctly set")
    
    def test_chunk_step_subset_envs(self):
        """Test 4: Chunk step a subset of environments."""
        print("\n" + "="*80)
        print("TEST 4: Chunk step a subset of environments")
        print("="*80)
        
        # Reset all environments first
        all_env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        do_validation = [False] * self.config.num_envs
        
        print(f"Resetting all {self.config.num_envs} environments...")
        self.env_manager.reset(
            env_ids=all_env_ids,
            task_ids=task_ids,
            trial_ids=trial_ids,
            do_validataion=do_validation,
        )
        print(f"✓ All environments reset")
        
        # Prepare chunk actions
        chunk_env_ids = [0, 1, 2]  # Subset of 3 environments
        chunk_size = 5  # 5 actions per environment
        
        # Create action chunk: (num_envs, chunk_size, action_dim)
        action_chunk = np.random.randn(len(chunk_env_ids), chunk_size, 7).astype(np.float32)
        action_chunk[:, :, -1] = np.clip(action_chunk[:, :, -1], -1, 1)  # Gripper
        
        print(f"Chunk stepping environment IDs: {chunk_env_ids}")
        print(f"Action chunk shape: {action_chunk.shape}")
        print(f"  - Number of environments: {len(chunk_env_ids)}")
        print(f"  - Chunk size (actions per env): {chunk_size}")
        print(f"  - Action dimension: 7")
        
        # Perform chunk step
        result = self.env_manager.chunk_step(env_ids=chunk_env_ids, actions=action_chunk)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("full_images", result)
        self.assertIn("wrist_images", result)
        self.assertIn("states", result)
        self.assertIn("completes", result)
        self.assertIn("active", result)
        self.assertIn("finish_steps", result)
        print(f"✓ Chunk step returned expected data structure")
        
        # Verify result shapes
        self.assertEqual(result["full_images"].shape[0], len(chunk_env_ids))
        self.assertEqual(result["wrist_images"].shape[0], len(chunk_env_ids))
        self.assertEqual(result["states"].shape[0], len(chunk_env_ids))
        print(f"✓ Result shapes correct for {len(chunk_env_ids)} environments")
        
        # Verify step counter was incremented by chunk_size for active environments
        env_states = self.env_manager.get_env_states(chunk_env_ids)
        for i, env_id in enumerate(chunk_env_ids):
            state = env_states[i]
            # Should have advanced by chunk_size steps (or less if completed early)
            self.assertTrue(state.step >= chunk_size or not state.active)
            print(f"  - Env {env_id}: {state.step} steps, active={state.active}")
        print(f"✓ Step counters incremented by chunk_size")
        
        # Verify non-chunked environment remains unchanged
        non_chunked_env_id = 3
        non_chunked_env_state = self.env_manager.get_env_states([non_chunked_env_id])[0]
        self.assertEqual(non_chunked_env_state.step, 0)
        print(f"✓ Non-chunked environment (ID {non_chunked_env_id}) remains at step 0")
        
        # Test with torch tensors
        print(f"\nTesting chunk_step with torch.Tensor input...")
        action_chunk_torch = torch.from_numpy(action_chunk)
        result_torch = self.env_manager.chunk_step(
            env_ids=chunk_env_ids, 
            actions=action_chunk_torch
        )
        self.assertIsInstance(result_torch, dict)
        print(f"✓ Chunk step works with torch.Tensor input")
    
    def test_validation_subset_and_get_valid_pixels(self):
        """Test 5: Enable validation in subset of environments and check get_valid_pixels."""
        print("\n" + "="*80)
        print("TEST 5: Validation in subset and get_valid_pixels")
        print("="*80)
        
        # Reset all environments, but enable validation only for subset
        all_env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        
        # Enable validation for envs 0 and 2, disable for envs 1 and 3
        validation_enabled_envs = [0, 2]
        validation_disabled_envs = [1, 3]
        do_validation = [i in validation_enabled_envs for i in all_env_ids]
        
        print(f"Resetting all {self.config.num_envs} environments...")
        print(f"  - Validation enabled for: {validation_enabled_envs}")
        print(f"  - Validation disabled for: {validation_disabled_envs}")
        
        self.env_manager.reset(
            env_ids=all_env_ids,
            task_ids=task_ids,
            trial_ids=trial_ids,
            do_validataion=do_validation,
        )
        print(f"✓ All environments reset with selective validation")
        
        # Verify validation flags are set correctly
        env_states = self.env_manager.get_env_states(all_env_ids)
        for i, env_id in enumerate(all_env_ids):
            state = env_states[i]
            expected_validation = env_id in validation_enabled_envs
            self.assertEqual(state.do_validation, expected_validation)
            if expected_validation:
                self.assertIsNotNone(state.valid_pixels)
                self.assertIn("full_images", state.valid_pixels)
                self.assertIn("wrist_images", state.valid_pixels)
                self.assertEqual(len(state.valid_pixels["full_images"]), 0)
                self.assertEqual(len(state.valid_pixels["wrist_images"]), 0)
        print(f"✓ Validation flags set correctly for each environment")
        
        # Step all environments multiple times to collect validation data
        num_steps = 3
        print(f"\nStepping all environments {num_steps} times to collect validation data...")
        
        for step_num in range(num_steps):
            # Random actions in [0, 1], gripper is binary [-1, 1]
            actions = np.random.rand(len(all_env_ids), 7).astype(np.float32)
            actions[:, -1] = np.sign(actions[:, -1] - 0.5)
            
            result = self.env_manager.step(env_ids=all_env_ids, action=actions)
            print(f"  - Step {step_num + 1}/{num_steps} completed")
        
        print(f"✓ Completed {num_steps} steps for all environments")
        
        # Check that validation-enabled environments have collected pixels
        env_states = self.env_manager.get_env_states(validation_enabled_envs)
        for i, env_id in enumerate(validation_enabled_envs):
            state = env_states[i]
            # Should have collected images for each step
            num_collected = len(state.valid_pixels["full_images"])
            self.assertGreater(num_collected, 0)
            self.assertEqual(
                len(state.valid_pixels["full_images"]),
                len(state.valid_pixels["wrist_images"])
            )
            print(f"  - Env {env_id}: collected {num_collected} image frames")
        print(f"✓ Validation-enabled environments collected pixel data")
        
        # Check that validation-disabled environments have NOT collected pixels
        env_states_disabled = self.env_manager.get_env_states(validation_disabled_envs)
        for i, env_id in enumerate(validation_disabled_envs):
            state = env_states_disabled[i]
            self.assertFalse(state.do_validation)
            # valid_pixels should be None or empty for non-validation envs
            if state.valid_pixels is not None:
                self.assertEqual(len(state.valid_pixels.get("full_images", [])), 0)
        print(f"✓ Validation-disabled environments did NOT collect pixel data")
        
        # Test get_valid_pixels function
        print(f"\nTesting get_valid_pixels function...")
        valid_pixels = self.env_manager.get_valid_pixels(env_ids=validation_enabled_envs)
        
        # Verify structure
        self.assertIsInstance(valid_pixels, dict)
        self.assertIn("full_images", valid_pixels)
        self.assertIn("wrist_images", valid_pixels)
        print(f"✓ get_valid_pixels returned expected data structure")
        
        # Verify shapes
        # Shape should be (num_envs, num_frames, height, width, channels)
        full_images_shape = valid_pixels["full_images"].shape
        wrist_images_shape = valid_pixels["wrist_images"].shape
        
        self.assertEqual(full_images_shape[0], len(validation_enabled_envs))
        self.assertEqual(wrist_images_shape[0], len(validation_enabled_envs))
        
        print(f"  - Full images shape: {full_images_shape}")
        print(f"  - Wrist images shape: {wrist_images_shape}")
        
        # Verify each environment's validation data
        for i, env_id in enumerate(validation_enabled_envs):
            num_frames = full_images_shape[1]
            self.assertGreater(num_frames, 0)
            
            # Check image dimensions
            self.assertEqual(full_images_shape[2:], (self.config.height, self.config.width, 3))
            self.assertEqual(wrist_images_shape[2:], (self.config.height, self.config.width, 3))
            
            print(f"  - Env {env_id}: {num_frames} frames, "
                  f"images {self.config.height}x{self.config.width}x3")
        
        print(f"✓ get_valid_pixels returned correct shapes and data")
        
        # Test that we can retrieve validation pixels for individual environments
        print(f"\nTesting get_valid_pixels for individual environments...")
        for env_id in validation_enabled_envs:
            single_env_pixels = self.env_manager.get_valid_pixels(env_ids=[env_id])
            self.assertEqual(single_env_pixels["full_images"].shape[0], 1)
            self.assertEqual(single_env_pixels["wrist_images"].shape[0], 1)
            print(f"  - Env {env_id}: retrieved validation data successfully")
        print(f"✓ Individual environment validation retrieval works")
        
        # Verify pixel data types
        self.assertTrue(np.issubdtype(valid_pixels["full_images"].dtype, np.number))
        self.assertTrue(np.issubdtype(valid_pixels["wrist_images"].dtype, np.number))
        print(f"✓ Validation pixels have correct data type")


if __name__ == "__main__":
    """
    Run tests using unittest framework.
    Usage: python test_env_manager_libero.py
    or: python -m unittest test_env_manager_libero.py
    """
    unittest.main(verbosity=2)

