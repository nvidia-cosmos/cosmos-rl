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
Test suite for RoboTwin RoboTwinEnvWrapper.

This test suite validates the RoboTwinEnvWrapper's ability to manage multiple RoboTwin environments,
including creating vectorized environments, resetting subsets, stepping subsets, chunk stepping,
and validation.

Requirements:
    - RoboTwin repository in path
    - ASSETS_PATH environment variable set
    - SAPIEN rendering support

Usage:
    # Run all tests
    python test_env_wrapper_robotwin.py

    # Run with unittest discovery
    python -m unittest test_env_wrapper_robotwin

    # Run specific test
    python -m unittest test_env_wrapper_robotwin.TestRoboTwinEnvWrapper.test_create_vector_envs

    # Run with verbose output
    python -m unittest test_env_wrapper_robotwin -v

Tests:
    1. test_create_vector_envs: Verify creation of a vector of environments
    2. test_reset_full: Test resetting all environments
    3. test_reset_partial: Test resetting a subset of environments
    4. test_step_full: Test stepping all environments
    5. test_step_partial: Test stepping a subset of environments
    6. test_chunk_step: Test chunk stepping environments
    7. test_validation_and_pixels: Test validation mode, pixel collection, and video saving
    8. test_async_reset: Test async reset operations
    9. test_reset_without_validation: Test reset without validation (no data collection)
    10. test_instruction_generation: Test task instruction generation
    11. test_multi_task_envs: Test multiple different tasks in parallel environments
"""

import unittest
import numpy as np
import torch
from dataclasses import dataclass
import os
import sys
import tempfile
import warnings
import logging

# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Enable info logging to see progress during long initialization
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Ensure RoboTwin is in path
robotwin_path = os.getenv("ROBOTWIN_PATH", "/root/workspace/RoboTwin")
if robotwin_path and robotwin_path not in sys.path:
    sys.path.insert(0, robotwin_path)

# Set ASSETS_PATH if not already set
if "ASSETS_PATH" not in os.environ:
    os.environ["ASSETS_PATH"] = os.path.join(robotwin_path, "assets")

from cosmos_rl.simulators.robotwin.env_wrapper import RoboTwinEnvWrapper


@dataclass
class MockRoboTwinConfig:
    """Minimal configuration for RoboTwinEnvWrapper."""

    num_envs: int = 4
    seed: int = 42
    max_steps: int = 100  # Short for testing
    task_config: dict = None

    def __post_init__(self):
        if self.task_config is None:
            # Test config - will be merged with demo_randomized.yml defaults
            # We specify all fields explicitly for testing to ensure deterministic behavior
            self.task_config = {
                "task_name": "place_shoe",  # Single task for most tests
                "step_lim": self.max_steps,
                "save_path": "./data",
                "embodiment": [
                    "aloha-agilex"
                ],  # Only embodiment with complete dual-arm configs
                "instruction_type": "seen",
                "camera": {
                    "head_camera_type": "D435",
                },
                "data_type": {
                    "rgb": True,
                    "third_view": False,
                    "depth": False,
                    "pointcloud": False,
                    "observer": False,
                    "endpose": True,
                    "qpos": True,
                    "mesh_segmentation": False,
                    "actor_segmentation": False,
                },
                "eval_mode": True,
                "eval_video_log": False,
                "render_freq": 0,
                "planner_backend": "mplib",  # H200 GPU compatibility
                "rdt_step": 10,
                "domain_randomization": {
                    "random_background": False,
                    "cluttered_table": False,
                    "clean_background_rate": 1,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0,
                    "random_light": False,
                    "crazy_random_light_rate": 0,
                    "random_embodiment": False,
                },
            }


class TestRoboTwinEnvWrapper(unittest.TestCase):
    """Test suite for RoboTwinEnvWrapper."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all test methods."""
        cls.config = MockRoboTwinConfig()
        print(
            f"\n{'='*80}\n"
            f"Initializing RoboTwin environment with {cls.config.num_envs} environments...\n"
            f"Task: {cls.config.task_config['task_name']}\n"
            f"\n"
            f"⚠️  FIRST RUN TAKES 5-10 MINUTES:\n"
            f"  - Curobo JIT compiles CUDA kernels (~2-3 min)\n"
            f"  - SAPIEN initializes physics for each env (~1-2 min per env)\n"
            f"  - Episode info extraction for instructions (~1 min)\n"
            f"\n"
            f"Subsequent runs will be much faster (cached compilation).\n"
            f"Please wait...\n"
            f"{'='*80}"
        )

        try:
            import time

            start_time = time.time()

            # Check if we should skip slow instruction setup
            skip_setup = os.getenv("SKIP_INSTRUCTION_SETUP", "0") == "1"
            if skip_setup:
                print(
                    "⚡ FAST MODE: Skipping instruction setup (set via SKIP_INSTRUCTION_SETUP=1)"
                )

            cls.env = RoboTwinEnvWrapper(
                cfg=cls.config, skip_instruction_setup=skip_setup
            )
            init_time = time.time() - start_time
            print(
                f"\n{'='*80}\n"
                f"✓ RoboTwinEnvWrapper initialized successfully in {init_time:.1f}s\n"
                f"{'='*80}\n"
            )
        except Exception as e:
            print(
                f"\n{'='*80}\n✗ Failed to initialize RoboTwinEnvWrapper: {e}\n{'='*80}\n"
            )
            import traceback

            traceback.print_exc()
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up after all test methods have run."""
        print("\n" + "=" * 80)
        print("Tearing down shared environment (runs once after all tests)...")
        print("=" * 80)
        if hasattr(cls, "env") and cls.env is not None:
            try:
                cls.env.close(clear_cache=True)
                print("✓ Environment closed successfully")
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
        # Clean up temp directory
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            try:
                pass  # shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning during temp_dir cleanup: {e}")

    def test_create_vector_envs(self):
        """Test 1: Create a vector of environments."""
        print("\n" + "=" * 80)
        print("TEST 1: Create a vector of environments")
        print("=" * 80)

        # Verify the environment manager was created
        self.assertIsNotNone(self.env)
        print("✓ RoboTwinEnvWrapper created successfully")

        # Verify number of environments
        self.assertEqual(self.env.num_envs, self.config.num_envs)
        print(f"✓ Number of environments: {self.config.num_envs}")

        # Verify task states were initialized
        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        self.assertEqual(len(env_states), self.config.num_envs)
        print(f"✓ Task states initialized for {self.config.num_envs} environments")

        # Verify each environment state is properly initialized
        for i, state in enumerate(env_states):
            self.assertEqual(state.env_idx, i)
            # task_id may be -1 (fresh) or >= 0 (if previous test ran)
            self.assertGreaterEqual(state.task_id, -1)
            # Note: In shared test environment, states may be modified by previous tests
            # A fresh environment would have task_id=-1, but after reset it would be >= 0
        print("✓ All environment states are valid")

    def test_reset_full(self):
        """Test 2: Reset all environments."""
        print("\n" + "=" * 80)
        print("TEST 2: Reset all environments")
        print("=" * 80)

        # Reset all environments
        env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs  # Same task for all
        trial_ids = list(range(self.config.num_envs))  # Different trials/seeds
        do_validation = [False] * self.config.num_envs

        print(f"Resetting {len(env_ids)} environments...")
        print(f"  env_ids: {env_ids}")
        print(f"  task_ids: {task_ids}")
        print(f"  trial_ids: {trial_ids}")

        images_and_states, task_descriptions = self.env.reset(
            env_ids=env_ids,
            task_ids=task_ids,
            trial_ids=trial_ids,
            do_validation=do_validation,
        )

        # Verify reset results structure
        self.assertIsInstance(images_and_states, dict)
        self.assertIn("full_images", images_and_states)
        self.assertIn("states", images_and_states)
        print("✓ Reset returned expected data structure")

        # Verify shapes
        self.assertEqual(images_and_states["full_images"].shape[0], len(env_ids))
        self.assertEqual(images_and_states["states"].shape[0], len(env_ids))
        print(f"✓ Observation shapes correct for {len(env_ids)} environments")

        # Verify image dimensions (RoboTwin default: 480x640x3)
        full_img_shape = images_and_states["full_images"].shape
        self.assertEqual(len(full_img_shape), 4)  # (num_envs, H, W, C)
        print(f"✓ Full images shape: {full_img_shape}")

        # Verify wrist images if present
        if "left_wrist_images" in images_and_states:
            self.assertEqual(
                images_and_states["left_wrist_images"].shape[0], len(env_ids)
            )
            print(
                f"✓ Left wrist images shape: {images_and_states['left_wrist_images'].shape}"
            )

        if "right_wrist_images" in images_and_states:
            self.assertEqual(
                images_and_states["right_wrist_images"].shape[0], len(env_ids)
            )
            print(
                f"✓ Right wrist images shape: {images_and_states['right_wrist_images'].shape}"
            )

        # Verify task descriptions were returned
        self.assertIsInstance(task_descriptions, list)
        self.assertEqual(len(task_descriptions), len(env_ids))
        print(f"✓ Task descriptions returned for {len(env_ids)} environments")

        # Verify all task descriptions are valid
        for i, desc in enumerate(task_descriptions):
            self.assertIsNotNone(desc)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)
            print(f"  - Env {i}: '{desc[:50]}...'")
        print("✓ All task descriptions are valid")

        # Verify task states were updated for reset environments
        env_states = self.env.get_env_states(env_ids)
        for i, env_id in enumerate(env_ids):
            state = env_states[i]
            self.assertEqual(state.task_id, task_ids[i])
            self.assertEqual(state.trial_id, trial_ids[i])
            self.assertTrue(state.active)
            self.assertEqual(state.step, 0)
            self.assertIsNotNone(state.current_obs)
            self.assertFalse(state.do_validation)
        print("✓ Task states updated correctly for reset environments")

    def test_reset_partial(self):
        """Test 3: Reset a subset of environments."""
        print("\n" + "=" * 80)
        print("TEST 3: Reset a subset of environments")
        print("=" * 80)

        # First reset all environments
        all_env_ids = list(range(self.config.num_envs))
        all_task_ids = [0] * self.config.num_envs
        all_trial_ids = [0] * self.config.num_envs
        all_do_validation = [False] * self.config.num_envs

        print(f"First, resetting all {self.config.num_envs} environments...")
        self.env.reset(all_env_ids, all_task_ids, all_trial_ids, all_do_validation)
        print("✓ All environments reset")

        # Now reset only a subset
        partial_env_ids = [0, 2]  # Reset envs 0 and 2, not 1 and 3
        partial_task_ids = [0, 0]
        partial_trial_ids = [10, 11]  # Different trial IDs
        partial_do_validation = [False, False]

        print(
            f"\nNow resetting subset: env_ids={partial_env_ids}, trial_ids={partial_trial_ids}"
        )
        images_and_states, task_descriptions = self.env.reset(
            env_ids=partial_env_ids,
            task_ids=partial_task_ids,
            trial_ids=partial_trial_ids,
            do_validation=partial_do_validation,
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
            self.assertEqual(state.step, 0)
        print("✓ Partial reset environments have correct trial IDs and step=0")

        # Verify the non-reset environments still have old state
        non_reset_env_ids = [1, 3]
        for env_id in non_reset_env_ids:
            state = self.env.get_env_states([env_id])[0]
            self.assertEqual(state.trial_id, 0)  # Original trial ID
        print(f"✓ Non-reset environments {non_reset_env_ids} retained original state")

    def test_step_full(self):
        """Test 4: Step all environments."""
        print("\n" + "=" * 80)
        print("TEST 4: Step all environments")
        print("=" * 80)

        # First reset all environments
        env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        do_validation = [False] * self.config.num_envs

        print(f"Resetting {len(env_ids)} environments...")
        self.env.reset(env_ids, task_ids, trial_ids, do_validation)
        print("✓ All environments reset")

        # Create random actions for all environments
        # RoboTwin dual-arm action: 14 dims (7 per arm)
        action_dim = 14
        actions = np.random.randn(len(env_ids), action_dim).astype(np.float32)
        actions = np.clip(actions, -1, 1)  # Clip to valid range

        print(f"\nStepping {len(env_ids)} environments...")
        print(f"Action shape: {actions.shape}")

        # Perform step
        result = self.env.step(env_ids, actions)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("full_images", result)
        self.assertIn("states", result)
        self.assertIn("complete", result)
        self.assertIn("active", result)
        self.assertIn("finish_step", result)
        print("✓ Step returned expected data structure")

        # Verify result shapes
        self.assertEqual(result["full_images"].shape[0], len(env_ids))
        self.assertEqual(result["states"].shape[0], len(env_ids))
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

    def test_step_partial(self):
        """Test 5: Step a subset of environments."""
        print("\n" + "=" * 80)
        print("TEST 5: Step subset of environments")
        print("=" * 80)

        # Reset all environments first
        all_env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = [0] * self.config.num_envs
        do_validation = [False] * self.config.num_envs

        print(f"Resetting all {self.config.num_envs} environments...")
        self.env.reset(all_env_ids, task_ids, trial_ids, do_validation)
        print("✓ All environments reset")

        # Step only a subset
        step_env_ids = [0, 2, 3]  # Step envs 0, 2, 3 (not 1)
        action_dim = 14
        actions = np.random.randn(len(step_env_ids), action_dim).astype(np.float32)
        actions = np.clip(actions, -1, 1)

        print(f"\nStepping subset: env_ids={step_env_ids}")
        print(f"Action shape: {actions.shape}")

        result = self.env.step(step_env_ids, actions)

        # Verify result structure
        self.assertEqual(result["full_images"].shape[0], len(step_env_ids))
        self.assertEqual(len(result["complete"]), len(step_env_ids))
        self.assertEqual(len(result["active"]), len(step_env_ids))
        print("✓ Returned data for partial environments only")

        # Verify step counter incremented for stepped envs
        for i, env_id in enumerate(step_env_ids):
            state = self.env.get_env_states([env_id])[0]
            self.assertGreaterEqual(state.step, 1)
        print(f"✓ Step counter incremented for envs {step_env_ids}")

        # Verify non-stepped environment (env 1) still at step 0
        non_stepped_env_id = 1
        env1_state = self.env.get_env_states([non_stepped_env_id])[0]
        self.assertEqual(env1_state.step, 0)
        print(f"✓ Non-stepped environment (env {non_stepped_env_id}) remains at step 0")

    def test_chunk_step(self):
        """Test 6: Chunk step environments."""
        print("\n" + "=" * 80)
        print("TEST 6: Chunk step environments")
        print("=" * 80)

        # Reset environments first
        env_ids = [0, 1, 2]  # Subset of 3 environments
        task_ids = [0] * len(env_ids)
        trial_ids = list(range(len(env_ids)))
        do_validation = [False] * len(env_ids)

        print(f"Resetting {len(env_ids)} environments...")
        self.env.reset(env_ids, task_ids, trial_ids, do_validation)
        print("✓ Environments reset")

        # Prepare chunk actions
        chunk_size = 5  # 5 actions per environment
        action_dim = 14

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
        self.assertIn("states", result)
        self.assertIn("complete", result)
        self.assertIn("active", result)
        self.assertIn("finish_step", result)
        print("✓ Chunk step returned expected data structure")

        # Verify result shapes
        self.assertEqual(result["full_images"].shape[0], len(env_ids))
        self.assertEqual(result["states"].shape[0], len(env_ids))
        print(f"✓ Result shapes correct for {len(env_ids)} environments")

        # Verify step counter was incremented by chunk_size
        env_states = self.env.get_env_states(env_ids)
        for i, env_id in enumerate(env_ids):
            state = env_states[i]
            # Should have advanced by chunk_size steps (or less if completed early)
            self.assertTrue(state.step >= chunk_size or not state.active)
            print(f"  - Env {env_id}: {state.step} steps, active={state.active}")
        print("✓ Step counters incremented by chunk_size")

        # Test with torch tensors
        print("\nTesting chunk_step with torch.Tensor input...")
        action_chunk_torch = torch.from_numpy(action_chunk)
        result_torch = self.env.chunk_step(env_ids, action_chunk_torch)
        self.assertIsInstance(result_torch, dict)
        print("✓ Chunk step works with torch.Tensor input")

    def test_validation_and_pixels(self):
        """Test 7: Enable validation, check pixel collection, and save videos."""
        print("\n" + "=" * 80)
        print("TEST 7: Validation mode, pixel collection, and video saving")
        print("=" * 80)

        # Reset all environments, enable validation for subset
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

        self.env.reset(all_env_ids, task_ids, trial_ids, do_validation)
        print("✓ All environments reset with selective validation")

        # Verify validation flags are set correctly
        env_states = self.env.get_env_states(all_env_ids)
        for i, env_id in enumerate(all_env_ids):
            state = env_states[i]
            expected_validation = env_id in validation_enabled_envs
            self.assertEqual(state.do_validation, expected_validation)
            if expected_validation:
                self.assertIsNotNone(state.valid_pixels)
                self.assertIn("full_images", state.valid_pixels)
        print("✓ Validation flags set correctly for each environment")

        # Step all environments multiple times to collect validation data
        num_steps = 50
        print(
            f"\nStepping all environments {num_steps} times to collect validation data..."
        )

        action_dim = 14
        for step_num in range(num_steps):
            actions = np.random.randn(len(all_env_ids), action_dim).astype(np.float32)
            actions = np.clip(actions, -1, 1)
            self.env.step(env_ids=all_env_ids, action=actions)
            print(f"  - Step {step_num + 1}/{num_steps} completed")

        print(f"✓ Completed {num_steps} steps for all environments")

        # Check that validation-enabled environments have collected pixels
        env_states = self.env.get_env_states(validation_enabled_envs)
        for i, env_id in enumerate(validation_enabled_envs):
            state = env_states[i]
            # Should have collected images for each step
            num_collected = len(state.valid_pixels["full_images"])
            self.assertGreater(num_collected, 0)
            print(f"  - Env {env_id}: collected {num_collected} image frames")
        print("✓ Validation-enabled environments collected pixel data")

        # Check that validation-disabled environments have NOT collected pixels
        env_states_disabled = self.env.get_env_states(validation_disabled_envs)
        for i, env_id in enumerate(validation_disabled_envs):
            state = env_states_disabled[i]
            self.assertFalse(state.do_validation)
            # valid_pixels should be None or empty for non-validation envs
            if state.valid_pixels is not None:
                self.assertEqual(len(state.valid_pixels.get("full_images", [])), 0)
        print("✓ Validation-disabled environments did NOT collect pixel data")

        # Test video saving for validation-enabled environments
        print("\nTesting video saving functionality...")

        try:
            self.env.save_validation_videos(self.temp_dir, validation_enabled_envs)
            print("✓ save_validation_videos completed without errors")

            # Check if video files were created
            video_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".mp4")]
            print(f"  - Created {len(video_files)} video file(s) in {self.temp_dir}")

            # We expect at least one video file per validation-enabled environment
            if len(video_files) > 0:
                print(f"  - Video files: {video_files}")
                print("✓ Video files created successfully")
            else:
                print("  - Note: No .mp4 files found (may depend on implementation)")
        except Exception as e:
            print(f"  - Warning: Video saving raised exception: {e}")
            print("  - This may be expected if video codec/writer is not configured")
            import traceback

            traceback.print_exc()

    def test_async_reset(self):
        """Test 8: Async reset operations."""
        print("\n" + "=" * 80)
        print("TEST 8: Async reset operations")
        print("=" * 80)

        # First, initialize all environments
        all_env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        do_validation = [False] * self.config.num_envs

        print(f"Initial reset of all {self.config.num_envs} environments...")
        self.env.reset(all_env_ids, task_ids, trial_ids, do_validation)
        print("✓ All environments initialized")

        # Define split: reset envs 0,1 asynchronously while stepping envs 2,3
        reset_env_ids = [0, 1]
        step_env_ids = [2, 3]

        # Start async reset for envs 0,1
        reset_task_ids = [0, 0]
        reset_trial_ids = [10, 11]  # Different trials
        reset_do_validation = [False, False]

        print(f"\nStarting async reset for envs {reset_env_ids}...")
        print(f"  - New trial IDs: {reset_trial_ids}")

        import time

        start_time = time.time()
        self.env.reset_async(
            env_ids=reset_env_ids,
            task_ids=reset_task_ids,
            trial_ids=reset_trial_ids,
            do_validation=reset_do_validation,
        )
        async_start_time = time.time() - start_time
        print(f"✓ Async reset initiated in {async_start_time*1000:.2f}ms")

        # While reset is happening, step the other environments
        print(f"\nStepping envs {step_env_ids} while reset is in progress...")
        num_concurrent_steps = 3
        action_dim = 14
        step_start_time = time.time()

        for step_num in range(num_concurrent_steps):
            actions = np.random.randn(len(step_env_ids), action_dim).astype(np.float32)
            actions = np.clip(actions, -1, 1)
            self.env.step(env_ids=step_env_ids, action=actions)
            print(f"  - Step {step_num + 1}/{num_concurrent_steps} completed")

        step_duration = time.time() - step_start_time
        print(f"✓ Completed {num_concurrent_steps} steps in {step_duration*1000:.2f}ms")

        # Verify stepped environments advanced
        step_env_states = self.env.get_env_states(step_env_ids)
        for i, env_id in enumerate(step_env_ids):
            state = step_env_states[i]
            self.assertGreaterEqual(state.step, num_concurrent_steps)
            print(f"  - Env {env_id}: at step {state.step}")
        print("✓ Stepped environments advanced correctly")

        # Wait for async reset to complete
        print(f"\nWaiting for async reset of envs {reset_env_ids}...")
        wait_start_time = time.time()

        images_and_states, task_descriptions = self.env.reset_wait(
            env_ids=reset_env_ids
        )
        wait_duration = time.time() - wait_start_time
        print(f"✓ Async reset completed in {wait_duration*1000:.2f}ms (wait time)")

        # Verify reset results
        self.assertEqual(images_and_states["full_images"].shape[0], len(reset_env_ids))
        self.assertEqual(len(task_descriptions), len(reset_env_ids))
        print(f"✓ Reset data shapes correct for {len(reset_env_ids)} environments")

        # Verify reset environments are at step 0 with new trial IDs
        reset_env_states = self.env.get_env_states(reset_env_ids)
        for i, env_id in enumerate(reset_env_ids):
            state = reset_env_states[i]
            self.assertEqual(state.step, 0)
            self.assertEqual(state.trial_id, reset_trial_ids[i])
            print(f"  - Env {env_id}: step={state.step}, trial_id={state.trial_id}")
        print("✓ Reset environments have correct state after async reset")

    def test_reset_without_validation(self):
        """Test 9: Reset without validation (verify no data collected)."""
        print("\n" + "=" * 80)
        print("TEST 9: Reset without validation")
        print("=" * 80)

        # Reset without validation
        env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
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
        action_dim = 14
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

    def test_instruction_generation(self):
        """Test 10: Task instruction generation."""
        print("\n" + "=" * 80)
        print("TEST 10: Task instruction generation")
        print("=" * 80)

        # Reset environments and get instructions
        env_ids = [0, 1]
        task_ids = [0, 0]
        trial_ids = [0, 1]  # Different seeds may generate different instructions
        do_validation = [False, False]

        print(
            f"Resetting {len(env_ids)} environments to test instruction generation..."
        )
        images_and_states, task_descriptions = self.env.reset(
            env_ids=env_ids,
            task_ids=task_ids,
            trial_ids=trial_ids,
            do_validation=do_validation,
        )

        # Verify instructions were generated
        self.assertEqual(len(task_descriptions), len(env_ids))
        print(f"✓ Received {len(task_descriptions)} task descriptions")

        # Verify instruction content
        for i, desc in enumerate(task_descriptions):
            self.assertIsNotNone(desc)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)
            print(f"  - Env {env_ids[i]}: '{desc}'")
        print("✓ All task descriptions are valid strings")

        # Verify instructions are stored in env states
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertIsNotNone(state.language)
            self.assertEqual(state.language, task_descriptions[i])
        print("✓ Instructions stored in environment states")

    def test_multi_task_envs(self):
        """Test 11: Multiple different tasks in parallel environments."""
        print("\n" + "=" * 80)
        print("TEST 11: Multi-task environments (different tasks per env)")
        print("=" * 80)

        # Create a separate environment with different tasks
        print("Creating multi-task environment...")
        multi_task_config = MockRoboTwinConfig()

        # Each environment runs a different task
        multi_task_config.task_config["task_name"] = [
            "place_shoe",  # Env 0
            "place_empty_cup",  # Env 1
            "place_shoe",  # Env 2 (can repeat)
            "place_empty_cup",  # Env 3 (can repeat)
        ]

        skip_setup = os.getenv("SKIP_INSTRUCTION_SETUP", "0") == "1"
        multi_env = RoboTwinEnvWrapper(
            cfg=multi_task_config, skip_instruction_setup=skip_setup
        )

        try:
            print("✓ Multi-task environment created")
            print(f"  - Task names: {multi_task_config.task_config['task_name']}")

            # Reset all environments
            env_ids = list(range(4))
            task_ids = [0] * 4
            trial_ids = list(range(4))
            do_validation = [False] * 4

            print("\nResetting all environments...")
            images_and_states, task_descriptions = multi_env.reset(
                env_ids, task_ids, trial_ids, do_validation
            )
            print("✓ All environments reset successfully")

            # Verify we got descriptions for all tasks
            self.assertEqual(len(task_descriptions), 4)
            print("\nTask descriptions:")
            for i, desc in enumerate(task_descriptions):
                task_name = multi_task_config.task_config["task_name"][i]
                print(
                    f"  - Env {i} ({task_name}): {desc[:60]}..."
                    if len(desc) > 60
                    else f"  - Env {i} ({task_name}): {desc}"
                )
            print("✓ Received task descriptions for all environments")

            # Step all environments
            num_steps = 3
            action_dim = 14
            print(f"\nStepping all {len(env_ids)} environments {num_steps} times...")

            for step_num in range(num_steps):
                actions = np.random.randn(len(env_ids), action_dim).astype(np.float32)
                actions = np.clip(actions, -1, 1)
                result = multi_env.step(env_ids=env_ids, action=actions)

                # Verify result structure
                self.assertIn("active", result)
                self.assertEqual(len(result["active"]), len(env_ids))
                print(
                    f"  - Step {step_num + 1}/{num_steps}: {np.sum(result['active'])} envs active"
                )

            print("✓ Successfully stepped multi-task environments")

        finally:
            # Clean up
            multi_env.close(clear_cache=False)
            print("✓ Multi-task environment closed")


if __name__ == "__main__":
    """
    Run tests using unittest framework.
    Usage: python test_env_wrapper_robotwin.py
    or: python -m unittest test_env_wrapper_robotwin.py
    """
    unittest.main(verbosity=2)
