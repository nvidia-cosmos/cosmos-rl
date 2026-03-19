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
Test suite for ManiSkillEnvWrapper.

Validates the wrapper's ability to manage ManiSkill3 vectorized environments
including full reset/step, chunk stepping, validation video recording, task
switching, and rejection of partial operations.

Requirements:
    - mani_skill package installed (pip install mani_skill)
    - CUDA-capable GPU (for num_envs > 1)

Usage:
    python tests/test_env_manager_maniskill.py
    python -m pytest tests/test_env_manager_maniskill.py -v

Tests:
    1. test_create_vector_envs: Verify creation of vectorized environments
    2. test_reset_full: Test full reset of all environments
    3. test_step_full: Test full step of all environments
    4. test_chunk_step: Test chunk stepping all environments
    5. test_validation_and_pixels: Test validation mode and video saving
    6. test_partial_reset_rejected: Verify partial reset raises AssertionError
    7. test_partial_step_rejected: Verify partial step raises AssertionError
    8. test_nonuniform_task_ids_rejected: Verify mixed task_ids raise AssertionError
    9. test_task_switching: Test switching between different ManiSkill3 tasks
   10. test_episode_to_completion: Run full episode until truncation
"""

import os
import tempfile
import unittest
import warnings
import logging

import numpy as np
import torch
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

try:
    import mani_skill  # noqa: F401

    HAS_MANISKILL = True
except ImportError:
    HAS_MANISKILL = False

from cosmos_rl.simulators.maniskill.env_wrapper import (
    ManiSkillEnvWrapper,
    get_maniskill_task_suite,
)


@dataclass
class MockManiSkillConfig:
    """Minimal configuration for ManiSkillEnvWrapper."""

    num_envs: int = 2
    seed: int = 42
    max_steps: int = 50
    obs_mode: str = "rgb"
    shader: str = "default"
    task_suite_name: str = "PickCube-v1"
    main_camera: str = None
    wrist_camera: str = None
    env_kwargs: dict = None

    def __post_init__(self):
        if self.env_kwargs is None:
            self.env_kwargs = {}


@unittest.skipUnless(HAS_MANISKILL, "mani_skill not installed")
class TestManiSkillEnvWrapper(unittest.TestCase):
    """Test suite for ManiSkillEnvWrapper."""

    @classmethod
    def setUpClass(cls):
        cls.config = MockManiSkillConfig()
        print(
            f"\n{'=' * 80}\n"
            f"Initializing ManiSkill environment with {cls.config.num_envs} environments...\n"
            f"Task suite: {cls.config.task_suite_name}\n"
            f"{'=' * 80}"
        )
        try:
            import time

            start_time = time.time()
            cls.env = ManiSkillEnvWrapper(cfg=cls.config)
            init_time = time.time() - start_time
            print(
                f"\n{'=' * 80}\n"
                f"ManiSkillEnvWrapper initialized in {init_time:.1f}s\n"
                f"{'=' * 80}\n"
            )
        except Exception as e:
            print(f"\nFailed to initialize ManiSkillEnvWrapper: {e}")
            import traceback

            traceback.print_exc()
            raise

    @classmethod
    def tearDownClass(cls):
        print(f"\n{'=' * 80}")
        print("Tearing down ManiSkill environment...")
        print("=" * 80)
        if hasattr(cls, "env") and cls.env is not None:
            try:
                cls.env.close()
                print("Environment closed successfully")
            except Exception as e:
                print(f"Warning during env.close(): {e}")
            finally:
                cls.env = None

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def _reset_all(self, do_validation=False):
        """Helper: full reset with task_id=0."""
        env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        do_val = [do_validation] * self.config.num_envs
        return self.env.reset(env_ids, task_ids, trial_ids, do_val)

    def _sample_actions(self):
        """Helper: sample random actions from the underlying env.

        ManiSkill3 vectorized envs return batched samples of shape
        ``(num_envs, action_dim)`` from a single ``sample()`` call.
        """
        sample = self.env.env.action_space.sample()
        if isinstance(sample, np.ndarray):
            return torch.from_numpy(sample).float()
        return sample

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_create_vector_envs(self):
        """Test 1: Verify creation of vectorized environments."""
        print("\n" + "=" * 80)
        print("TEST 1: Create vectorized environments")
        print("=" * 80)

        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.num_envs, self.config.num_envs)
        print(f"num_envs = {self.config.num_envs}")

        env_states = self.env.get_env_states(list(range(self.config.num_envs)))
        self.assertEqual(len(env_states), self.config.num_envs)
        for i, state in enumerate(env_states):
            self.assertEqual(state.env_idx, i)
        print("All environment states properly initialized")

    def test_reset_full(self):
        """Test 2: Full reset of all environments."""
        print("\n" + "=" * 80)
        print("TEST 2: Full reset")
        print("=" * 80)

        env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        do_validation = [False] * self.config.num_envs

        images_and_states, task_descriptions = self.env.reset(
            env_ids, task_ids, trial_ids, do_validation
        )

        # Structure checks
        self.assertIsInstance(images_and_states, dict)
        self.assertIn("full_images", images_and_states)
        self.assertIn("states", images_and_states)
        print("Reset returned expected data structure")

        # Shape checks
        full_img = images_and_states["full_images"]
        self.assertEqual(full_img.shape[0], self.config.num_envs)
        self.assertEqual(len(full_img.shape), 4)  # (B, H, W, C)
        self.assertEqual(full_img.dtype, np.uint8)
        print(f"full_images shape: {full_img.shape}  dtype: {full_img.dtype}")

        states = images_and_states["states"]
        if states is not None:
            self.assertEqual(states.shape[0], self.config.num_envs)
            print(f"states shape: {states.shape}")

        # Task descriptions
        self.assertEqual(len(task_descriptions), self.config.num_envs)
        for i, desc in enumerate(task_descriptions):
            self.assertIsInstance(desc, str)
            print(f"  env {i}: '{desc[:80]}'")

        # Env states
        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertEqual(state.task_id, 0)
            self.assertEqual(state.trial_id, trial_ids[i])
            self.assertTrue(state.active)
            self.assertFalse(state.complete)
            self.assertEqual(state.step, 0)
            self.assertIsNotNone(state.current_obs)
        print("Env states updated correctly after reset")

    def test_step_full(self):
        """Test 3: Full step of all environments."""
        print("\n" + "=" * 80)
        print("TEST 3: Full step")
        print("=" * 80)

        self._reset_all()
        print("Environments reset")

        env_ids = list(range(self.config.num_envs))
        actions = self._sample_actions()
        print(f"Action shape: {actions.shape}")

        result = self.env.step(env_ids, actions)

        self.assertIn("full_images", result)
        self.assertIn("states", result)
        self.assertIn("complete", result)
        self.assertIn("active", result)
        self.assertIn("finish_step", result)
        print("Step returned expected keys")

        self.assertEqual(result["full_images"].shape[0], self.config.num_envs)
        self.assertEqual(len(result["complete"]), self.config.num_envs)
        self.assertEqual(len(result["active"]), self.config.num_envs)
        self.assertEqual(len(result["finish_step"]), self.config.num_envs)
        print(f"Result shapes correct for {self.config.num_envs} envs")

        env_states = self.env.get_env_states(env_ids)
        for state in env_states:
            self.assertGreaterEqual(state.step, 1)
        print("Step counters incremented")

    def test_chunk_step(self):
        """Test 4: Chunk step all environments."""
        print("\n" + "=" * 80)
        print("TEST 4: Chunk step")
        print("=" * 80)

        self._reset_all()

        env_ids = list(range(self.config.num_envs))
        chunk_size = 5
        action_dim = self.env.env.action_space.shape[-1]
        action_chunk = np.random.randn(
            self.config.num_envs, chunk_size, action_dim
        ).astype(np.float32)
        action_chunk = np.clip(action_chunk, -1, 1)

        print(f"Action chunk shape: {action_chunk.shape}")

        result = self.env.chunk_step(env_ids, action_chunk)

        self.assertIn("full_images", result)
        self.assertIn("complete", result)
        self.assertEqual(result["full_images"].shape[0], self.config.num_envs)
        print("Chunk step returned expected structure")

        env_states = self.env.get_env_states(env_ids)
        for i, state in enumerate(env_states):
            self.assertTrue(state.step >= chunk_size or not state.active)
            print(f"  env {i}: step={state.step}, active={state.active}")
        print("Step counters advanced by chunk_size")

        # Also verify torch tensor input works
        action_chunk_torch = torch.from_numpy(action_chunk)
        result_torch = self.env.chunk_step(env_ids, action_chunk_torch)
        self.assertIsInstance(result_torch, dict)
        print("Chunk step works with torch.Tensor input")

    def test_validation_and_pixels(self):
        """Test 5: Validation mode, pixel collection, and video saving."""
        print("\n" + "=" * 80)
        print("TEST 5: Validation and video saving")
        print("=" * 80)

        env_ids = list(range(self.config.num_envs))
        task_ids = [0] * self.config.num_envs
        trial_ids = list(range(self.config.num_envs))
        do_validation = [True if i == 0 else False for i in range(self.config.num_envs)]

        self.env.reset(env_ids, task_ids, trial_ids, do_validation)
        print(
            f"Validation enabled for env 0, disabled for envs 1..{self.config.num_envs - 1}"
        )

        # Verify flags
        env_states = self.env.get_env_states(env_ids)
        self.assertTrue(env_states[0].do_validation)
        self.assertIsNotNone(env_states[0].valid_pixels)
        for s in env_states[1:]:
            self.assertFalse(s.do_validation)
        print("Validation flags set correctly")

        # Step several times to collect frames
        num_steps = 5
        for _ in range(num_steps):
            actions = self._sample_actions()
            self.env.step(env_ids, actions)

        state0 = self.env.get_env_states([0])[0]
        n_frames = len(state0.valid_pixels["full_images"])
        self.assertGreater(n_frames, 0)
        print(f"Env 0 collected {n_frames} validation frames")

        state1 = self.env.get_env_states([1])[0]
        self.assertIsNone(state1.valid_pixels)
        print("Env 1 has no validation pixels (as expected)")

        # Save video
        try:
            self.env.save_validation_videos(self.temp_dir, env_ids)
            video_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".mp4")]
            print(f"Created {len(video_files)} video file(s) in {self.temp_dir}")
            if video_files:
                print(f"  {video_files}")
        except Exception as e:
            print(f"Video saving raised: {e} (may need imageio)")

    def test_partial_reset_rejected(self):
        """Test 6: Partial reset must raise AssertionError."""
        print("\n" + "=" * 80)
        print("TEST 6: Partial reset rejected")
        print("=" * 80)

        partial_ids = [0]  # Only 1 out of num_envs
        with self.assertRaises(AssertionError):
            self.env.reset(partial_ids, [0], [0], [False])
        print(f"Partial reset with env_ids={partial_ids} correctly rejected")

    def test_partial_step_rejected(self):
        """Test 7: Partial step must raise AssertionError."""
        print("\n" + "=" * 80)
        print("TEST 7: Partial step rejected")
        print("=" * 80)

        self._reset_all()

        partial_ids = [0]
        action_dim = self.env.env.action_space.shape[-1]
        actions = np.zeros((1, action_dim), dtype=np.float32)

        with self.assertRaises(AssertionError):
            self.env.step(partial_ids, actions)
        print(f"Partial step with env_ids={partial_ids} correctly rejected")

    def test_nonuniform_task_ids_rejected(self):
        """Test 8: Non-uniform task_ids must raise AssertionError."""
        print("\n" + "=" * 80)
        print("TEST 8: Non-uniform task_ids rejected")
        print("=" * 80)

        # Create a wrapper with a multi-task suite
        multi_cfg = MockManiSkillConfig(task_suite_name="PickCube-v1,StackCube-v1")
        multi_env = ManiSkillEnvWrapper(cfg=multi_cfg)

        try:
            env_ids = list(range(multi_cfg.num_envs))
            mixed_task_ids = [0, 1]  # different tasks
            trial_ids = [0] * multi_cfg.num_envs

            with self.assertRaises(AssertionError):
                multi_env.reset(env_ids, mixed_task_ids, trial_ids, False)
            print("Mixed task_ids correctly rejected")
        finally:
            multi_env.close()

    def test_task_switching(self):
        """Test 9: Switch between different ManiSkill3 tasks."""
        print("\n" + "=" * 80)
        print("TEST 9: Task switching")
        print("=" * 80)

        multi_cfg = MockManiSkillConfig(task_suite_name="PickCube-v1,StackCube-v1")
        multi_env = ManiSkillEnvWrapper(cfg=multi_cfg)

        try:
            env_ids = list(range(multi_cfg.num_envs))
            trial_ids = [0] * multi_cfg.num_envs

            # Task 0: PickCube-v1
            imgs0, descs0 = multi_env.reset(
                env_ids, [0] * multi_cfg.num_envs, trial_ids, False
            )
            self.assertEqual(multi_env._current_task_env_id, "PickCube-v1")
            print(f"Task 0: {multi_env._current_task_env_id}")
            print(f"  full_images shape: {imgs0['full_images'].shape}")

            # Task 1: StackCube-v1
            imgs1, descs1 = multi_env.reset(
                env_ids, [1] * multi_cfg.num_envs, trial_ids, False
            )
            self.assertEqual(multi_env._current_task_env_id, "StackCube-v1")
            print(f"Task 1: {multi_env._current_task_env_id}")
            print(f"  full_images shape: {imgs1['full_images'].shape}")

            # Switch back to 0 — env should be recreated
            imgs0b, _ = multi_env.reset(
                env_ids, [0] * multi_cfg.num_envs, trial_ids, False
            )
            self.assertEqual(multi_env._current_task_env_id, "PickCube-v1")
            print("Switched back to PickCube-v1 successfully")
        finally:
            multi_env.close()

    def test_episode_to_completion(self):
        """Test 10: Run a full episode until max_steps truncation."""
        print("\n" + "=" * 80)
        print("TEST 10: Run episode to completion")
        print("=" * 80)

        short_cfg = MockManiSkillConfig(max_steps=10)
        short_env = ManiSkillEnvWrapper(cfg=short_cfg)

        try:
            env_ids = list(range(short_cfg.num_envs))
            short_env.reset(
                env_ids,
                [0] * short_cfg.num_envs,
                list(range(short_cfg.num_envs)),
                False,
            )

            for step_num in range(short_cfg.max_steps + 5):
                sample = short_env.env.action_space.sample()
                actions = (
                    torch.from_numpy(sample).float()
                    if isinstance(sample, np.ndarray)
                    else sample
                )
                result = short_env.step(env_ids, actions)

                n_active = int(result["active"].sum())
                if n_active == 0:
                    print(f"All envs inactive at step {step_num + 1}")
                    break

            # After max_steps, all envs should be inactive
            env_states = short_env.get_env_states(env_ids)
            for state in env_states:
                self.assertFalse(state.active)
                self.assertGreaterEqual(state.step, 1)
            print(
                f"All envs finished. Steps: "
                f"{[s.step for s in env_states]}, "
                f"complete: {[s.complete for s in env_states]}"
            )
        finally:
            short_env.close()


class TestManiSkillTaskSuite(unittest.TestCase):
    """Unit tests for get_maniskill_task_suite (no GPU needed)."""

    def test_named_suite(self):
        suite = get_maniskill_task_suite("bridge")
        self.assertIsInstance(suite, list)
        self.assertGreater(len(suite), 0)
        self.assertIn("PutCarrotOnPlateInScene-v1", suite)
        print(f"bridge suite: {suite}")

    def test_csv_suite(self):
        suite = get_maniskill_task_suite("PickCube-v1, StackCube-v1")
        self.assertEqual(suite, ["PickCube-v1", "StackCube-v1"])
        print(f"CSV suite: {suite}")

    def test_single_env_id(self):
        suite = get_maniskill_task_suite("PushCube-v1")
        self.assertEqual(suite, ["PushCube-v1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
