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

import copy
import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.parallelism import ParallelDims


def create_test_parallel_dims():
    """Create a minimal ParallelDims for single-GPU testing."""
    return ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        world_size=1,
        pp_dynamic_shape=False,
    )


def create_test_config(
    output_dir, resume=False, max_keep=3, export_safetensors=False, save_mode="sync"
):
    """Create a CosmosConfig for testing with minimal required settings.

    Args:
        output_dir: Full path including timestamp, e.g., /tmp/test/20250101000000
        resume: Whether to resume from checkpoint
        max_keep: Maximum number of checkpoints to keep
        export_safetensors: Whether to export safetensors
        save_mode: "sync" or "async"
    """
    # from_dict expects parent dir and appends timestamp, so we split them
    timestamp = os.path.basename(output_dir)

    config_dict = {
        "train": {
            "output_dir": output_dir,
            "resume": resume,
            "timestamp": timestamp,
            "ckpt": {
                "enable_checkpoint": True,
                "save_mode": save_mode,
                "max_keep": max_keep,
                "upload_s3": False,
                "export_safetensors": export_safetensors,
            },
        }
    }
    return CosmosConfig.from_dict(config_dict)


class SimpleModel(nn.Module):
    """Simple model for testing checkpoint save/load."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class SimpleScheduler:
    """Simple scheduler mock for testing."""

    def __init__(self, step=0):
        self._step = step

    def state_dict(self):
        return {"step": self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict["step"]

    def step(self):
        self._step += 1


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.timestamp1 = "20250101000000"
        self.timestamp2 = "20250102000000"

    def tearDown(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_resume_best_score(self):
        """Test that best_score persists across resume sessions.

        Simulates:
        1. First training session saves checkpoints with val_score
        2. Second session resumes and should have the best_score from first session
        """
        parallel_dims = create_test_parallel_dims()

        # === First training session ===
        output_dir1 = os.path.join(self.test_dir, self.timestamp1)
        config1 = create_test_config(output_dir=output_dir1, resume=False, max_keep=5)
        manager1 = CheckpointMananger(
            config1, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        # Create model, optimizer, scheduler
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler(step=0)

        # Save checkpoint at step 100 with val_score
        manager1.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=1000,
        )
        manager1.save_check(100, val_score=1.0)

        # Save checkpoint at step 200 with better val_score
        manager1.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=200,
            total_steps=1000,
        )
        manager1.save_check(200, val_score=0.5)

        # Verify best score and step are set
        self.assertEqual(manager1.best_score, 0.5)
        self.assertEqual(os.path.basename(manager1.best_ckpt_abs_dir), "step_200")

        # Verify best symlink exists
        best_ckpt_link = os.path.join(self.test_dir, "best", "checkpoints")
        self.assertTrue(os.path.islink(best_ckpt_link))

        # Verify best checkpoint symlink points to step 200
        target = os.readlink(best_ckpt_link)
        self.assertTrue(target.endswith("step_200"))

        # Verify best_score.json was saved
        best_score_path = os.path.join(self.test_dir, "best", "best_score.json")
        self.assertTrue(os.path.exists(best_score_path))

        # Verify the content of best_score.json
        with open(best_score_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["best_score"], 0.5)
        self.assertEqual(os.path.basename(data["best_ckpt_abs_dir"]), "step_200")
        self.assertEqual(data["metric"], "val_loss")

        # === Second training session (resume) ===
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        # Should have loaded best_score and best_step from first session
        self.assertEqual(manager2.best_score, 0.5)
        self.assertEqual(os.path.basename(manager2.best_ckpt_abs_dir), "step_200")

    def test_export_safetensors(self):
        """Test that export_safetensors behavior is correct."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(
            output_dir=output_dir, resume=False, max_keep=3, export_safetensors=True
        )
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        # Save checkpoint at step 100 with val_score
        manager.save_checkpoint(model, optimizer, scheduler, step=100, total_steps=1000)
        manager.save_check(100, val_score=0.5)

        # Verify best safetensors symlink exists
        best_safetensors_link = os.path.join(self.test_dir, "best", "safetensors")
        self.assertTrue(os.path.islink(best_safetensors_link))

        # Verify best safetensors symlink points to step 100
        target = os.readlink(best_safetensors_link)
        self.assertTrue(target.endswith("step_100"))

    def test_save_check_protects_best_checkpoint_from_deletion(self):
        """Test that save_check does not delete the best checkpoint when max_keep is exceeded.

        Scenario:
        - max_keep = 3
        - Steps 100, 200, step 100 is best
        - resume from previous session
        - save step 300 with worse score, step 100 still the best
        - When step 400 is added, step 100 still the best, step 200 (second oldest)
          should be deleted.
        """
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=3)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()
        # Save step 100 with best score
        manager.save_checkpoint(model, optimizer, scheduler, step=100, total_steps=1000)
        manager.save_check(100, val_score=0.3)  # Best score

        # Save step 200 with worse score
        # Second training session (resume)
        output_dir = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir, resume=True, max_keep=3)
        manager = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        # Save step 300 with worse score
        manager.save_checkpoint(model, optimizer, scheduler, step=300, total_steps=1000)
        manager.save_check(300, val_score=0.6)

        # Now we have 3 checkpoints, best is step 100
        self.assertEqual(os.path.basename(manager.best_ckpt_abs_dir), "step_100")

        # Save step 400, should trigger deletion
        manager.save_checkpoint(model, optimizer, scheduler, step=400, total_steps=1000)
        manager.save_check(400, val_score=0.7)

        # Verify step 100 still exists (protected as best)
        step_100_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100"
        )
        self.assertTrue(os.path.exists(step_100_path))

        # Verify step 200 was deleted (oldest non-best)
        step_200_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_200"
        )
        self.assertFalse(os.path.exists(step_200_path))

        # Verify we can resume from the latest checkpoint
        original_state = copy.deepcopy(model.state_dict())
        original_optimizer_state = copy.deepcopy(optimizer.state_dict())
        # make the model and optimizer different from the original
        loss = torch.nn.functional.mse_loss(
            model(torch.randn(4, 10)), torch.randn(4, 5)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Assert new model weights are different from original
        self.assertFalse(self._state_dicts_equal(original_state, model.state_dict()))
        self.assertFalse(
            self._state_dicts_equal(original_optimizer_state, optimizer.state_dict())
        )

        # Load the checkpoint from the previous session
        manager.load_checkpoint(model, optimizer, scheduler, model_name_or_path="dummy")

        # Assert new model weights are now the same as original
        self.assertTrue(self._state_dicts_equal(original_state, model.state_dict()))
        self.assertTrue(
            self._state_dicts_equal(original_optimizer_state, optimizer.state_dict())
        )

    def test_save_check_does_not_update_on_worse_score(self):
        """Test that save_check does not update best checkpoint on worse score."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        # Save step 100 with good score
        manager.save_checkpoint(model, optimizer, scheduler, step=100, total_steps=1000)
        manager.save_check(100, val_score=0.5)

        # Save step 200 with worse score
        manager.save_checkpoint(model, optimizer, scheduler, step=200, total_steps=1000)
        manager.save_check(200, val_score=0.8)

        # Best should still be step 100
        self.assertEqual(manager.best_score, 0.5)
        self.assertEqual(os.path.basename(manager.best_ckpt_abs_dir), "step_100")

        # Check the actual symlink points to step 100
        best_ckpt_link = os.path.join(self.test_dir, "best", "checkpoints")
        self.assertTrue(os.path.islink(best_ckpt_link))
        target = os.readlink(best_ckpt_link)
        self.assertTrue(target.endswith("step_100"))

    def _state_dicts_equal(self, sd1, sd2):
        """Helper to compare two state dicts."""
        if sd1.keys() != sd2.keys():
            return False
        for key in sd1:
            if isinstance(sd1[key], torch.Tensor):
                if not torch.equal(sd1[key], sd2[key]):
                    return False
            elif isinstance(sd1[key], dict):
                if not self._state_dicts_equal(sd1[key], sd2[key]):
                    return False
            elif sd1[key] != sd2[key]:
                return False
        return True

    def test_resume_from_previous_session(self):
        """Test that resume mode finds checkpoints from previous session."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        # Save original model state for later comparison
        torch.manual_seed(42)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        # Run forward pass and compute gradients (input: batch=4, dim=10 -> output: batch=4, dim=5)
        loss = torch.nn.functional.mse_loss(
            model(torch.randn(4, 10)), torch.randn(4, 5)
        )
        loss.backward()
        optimizer.step()

        original_state = copy.deepcopy(model.state_dict())
        original_optimizer_state = copy.deepcopy(optimizer.state_dict())

        manager.save_checkpoint(model, optimizer, scheduler, step=100, total_steps=1000)
        manager.save_check(100)

        # Second session: resume and verify it sees checkpoints from first session
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0
        )

        # Create new model with different weights
        torch.manual_seed(123)
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = SimpleScheduler()

        # Assert new model weights are different from original
        self.assertFalse(
            self._state_dicts_equal(original_state, new_model.state_dict())
        )
        self.assertFalse(
            self._state_dicts_equal(
                original_optimizer_state, new_optimizer.state_dict()
            )
        )

        # Load the checkpoint from the previous session
        manager2.load_checkpoint(
            new_model, new_optimizer, new_scheduler, model_name_or_path="dummy"
        )

        # Assert new model weights are now the same as original
        self.assertTrue(self._state_dicts_equal(original_state, new_model.state_dict()))
        self.assertTrue(
            self._state_dicts_equal(
                original_optimizer_state, new_optimizer.state_dict()
            )
        )

    def test_best_score_default_for_loss_metric(self):
        """Test default best_score is inf for loss metrics."""
        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False)
        parallel_dims = create_test_parallel_dims()
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )
        self.assertEqual(manager.best_score, float("inf"))

    def test_best_score_default_for_non_loss_metric(self):
        """Test default best_score is -inf for non-loss metrics (like accuracy)."""
        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False)
        parallel_dims = create_test_parallel_dims()
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="accuracy"
        )
        self.assertEqual(manager.best_score, -float("inf"))

    def test_load_checkpoint_restores_extra_info(self):
        """Test that load_checkpoint correctly restores extra info (step, total_steps)."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler(step=50)

        # Save checkpoint at step 100 with total_steps=1000
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=1000,
        )
        manager.save_check(100)

        # Resume from checkpoint
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = SimpleScheduler(step=0)

        extra_vars, loaded_scheduler = manager2.load_checkpoint(
            new_model, new_optimizer, new_scheduler, model_name_or_path="dummy"
        )

        # Verify extra info was restored
        self.assertEqual(extra_vars["step"], 100)
        self.assertEqual(extra_vars["total_steps"], 1000)
        # Verify scheduler state was restored
        self.assertEqual(loaded_scheduler._step, 50)

    def test_load_checkpoint_restores_rng_state(self):
        """Test that load_checkpoint correctly restores RNG state."""
        import numpy as np
        import random

        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        # Set specific RNG state and generate some random numbers
        torch.manual_seed(12345)
        np.random.seed(12345)
        random.seed(12345)

        # Generate random numbers to advance the RNG state
        _ = torch.randn(10)
        _ = np.random.rand(10)
        _ = [random.random() for _ in range(10)]

        # Save checkpoint (this captures current RNG state)
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=1000,
        )
        manager.save_check(100)

        # Generate reference random numbers after saving
        torch_ref = torch.randn(5).tolist()
        np_ref = np.random.rand(5).tolist()
        py_ref = [random.random() for _ in range(5)]

        # Reset RNG to different state
        torch.manual_seed(99999)
        np.random.seed(99999)
        random.seed(99999)

        # Resume from checkpoint
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = SimpleScheduler()

        manager2.load_checkpoint(
            new_model, new_optimizer, new_scheduler, model_name_or_path="dummy"
        )

        # Generate random numbers after loading - should match reference
        torch_after = torch.randn(5).tolist()
        np_after = np.random.rand(5).tolist()
        py_after = [random.random() for _ in range(5)]

        # Verify RNG state was restored
        self.assertEqual(torch_ref, torch_after)
        self.assertEqual(np_ref, np_after)
        self.assertEqual(py_ref, py_after)

    def test_load_checkpoint_with_scheduler_callable(self):
        """Test that load_checkpoint works with scheduler as a callable factory."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler(step=25)

        # Save checkpoint
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=500,
        )
        manager.save_check(100)

        # Resume using a scheduler factory
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Define a scheduler factory that creates scheduler based on training_steps
        def scheduler_factory(training_steps):
            return SimpleScheduler(step=0)

        extra_vars, loaded_scheduler = manager2.load_checkpoint(
            new_model, new_optimizer, scheduler_factory, model_name_or_path="dummy"
        )

        # Verify scheduler was created and state loaded
        self.assertIsInstance(loaded_scheduler, SimpleScheduler)
        self.assertEqual(loaded_scheduler._step, 25)
        self.assertEqual(extra_vars["total_steps"], 500)

    def test_load_checkpoint_skips_incomplete_checkpoint(self):
        """Test that load_checkpoint skips incomplete checkpoints and tries next."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler(step=10)

        # Save first checkpoint at step 100
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=1000,
        )
        manager.save_check(100)
        original_model_state = copy.deepcopy(model.state_dict())

        # add some gradients to the model
        loss = torch.nn.functional.mse_loss(
            model(torch.randn(4, 10)), torch.randn(4, 5)
        )
        loss.backward()
        optimizer.step()

        # Advance scheduler and save second checkpoint at step 200
        scheduler._step = 20
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=200,
            total_steps=1000,
        )
        manager.save_check(200)

        # Make step 200 checkpoint incomplete by removing the complete marker
        step_200_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_200", "policy"
        )
        complete_marker = os.path.join(step_200_path, ".rank_0_complete")
        os.remove(complete_marker)

        # Resume - should skip step 200 and load step 100
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = SimpleScheduler(step=0)

        extra_vars, _ = manager2.load_checkpoint(
            new_model, new_optimizer, new_scheduler, model_name_or_path="dummy"
        )

        # Should have loaded step 100 (scheduler step=10), not step 200 (scheduler step=20)
        self.assertEqual(extra_vars["step"], 100)
        self.assertEqual(new_scheduler._step, 10)
        # check model weights
        self.assertTrue(
            self._state_dicts_equal(original_model_state, new_model.state_dict())
        )

    def test_load_checkpoint_raises_when_no_checkpoint_found(self):
        """Test that load_checkpoint raises FileNotFoundError when no checkpoint exists."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=True, max_keep=5)

        # Create the output directory structure but no checkpoints
        os.makedirs(output_dir, exist_ok=True)

        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        with self.assertRaises(FileNotFoundError):
            manager.load_checkpoint(
                model, optimizer, scheduler, model_name_or_path="dummy"
            )

    def test_load_extra_info_from_checkpoint(self):
        """Test that load_extra_info_from_checkpoint correctly loads extra info."""
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        # Save checkpoint with custom extra info
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=150,
            total_steps=2000,
            custom_key="custom_value",
        )
        manager.save_check(150)

        # Resume and load only extra info
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        extra_vars = manager2.load_extra_info_from_checkpoint()

        # Verify extra info was loaded
        self.assertEqual(extra_vars["step"], 150)
        self.assertEqual(extra_vars["total_steps"], 2000)
        self.assertEqual(extra_vars["custom_key"], "custom_value")

    def test_save_and_load_with_dp_shard_greater_than_one(self):
        """Test save and load when dp_shard > 1 (FSDP mode with multiple ranks saving).

        When dp_shard > 1, multiple ranks save their own model/optimizer shards.
        The checkpoint is only considered complete when all rank markers exist.
        """
        # Create ParallelDims with dp_shard=2 (simulates 2 FSDP ranks)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            world_size=2,
            pp_dynamic_shape=False,
        )

        output_dir = os.path.join(self.test_dir, self.timestamp1)

        # Create models with different weights for each rank
        torch.manual_seed(42)
        model_rank0 = SimpleModel()
        optimizer_rank0 = torch.optim.Adam(model_rank0.parameters(), lr=0.001)
        scheduler_rank0 = SimpleScheduler(step=10)

        torch.manual_seed(123)
        model_rank1 = SimpleModel()
        optimizer_rank1 = torch.optim.Adam(model_rank1.parameters(), lr=0.001)
        scheduler_rank1 = SimpleScheduler(step=10)

        # Save original states for later comparison
        original_state_rank0 = copy.deepcopy(model_rank0.state_dict())
        original_state_rank1 = copy.deepcopy(model_rank1.state_dict())

        # Verify the two models have different weights
        self.assertFalse(
            self._state_dicts_equal(original_state_rank0, original_state_rank1)
        )

        # Create checkpoint managers for each rank
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager_rank0 = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )
        manager_rank1 = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=1, metric="val_loss"
        )

        # Save checkpoints from both ranks (simulating distributed saving)
        manager_rank0.save_checkpoint(
            model=model_rank0,
            optimizer=optimizer_rank0,
            scheduler=scheduler_rank0,
            step=100,
            total_steps=1000,
        )
        manager_rank1.save_checkpoint(
            model=model_rank1,
            optimizer=optimizer_rank1,
            scheduler=scheduler_rank1,
            step=100,
            total_steps=1000,
        )
        manager_rank0.save_check(100, val_score=0.5)

        # Verify checkpoint path check requires all rank markers
        ckpt_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100", "policy"
        )
        self.assertTrue(manager_rank0.ckpt_path_check(ckpt_path))

        # Verify both rank files exist
        self.assertTrue(os.path.exists(os.path.join(ckpt_path, "model_rank_0.pth")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_path, "model_rank_1.pth")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_path, ".rank_0_complete")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_path, ".rank_1_complete")))

        # === Resume and load ===
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)

        # Create new models with different weights
        torch.manual_seed(999)
        new_model_rank0 = SimpleModel()
        new_optimizer_rank0 = torch.optim.Adam(new_model_rank0.parameters(), lr=0.001)
        new_scheduler_rank0 = SimpleScheduler(step=0)

        torch.manual_seed(888)
        new_model_rank1 = SimpleModel()
        new_optimizer_rank1 = torch.optim.Adam(new_model_rank1.parameters(), lr=0.001)
        new_scheduler_rank1 = SimpleScheduler(step=0)

        # Verify new models are different from originals
        self.assertFalse(
            self._state_dicts_equal(original_state_rank0, new_model_rank0.state_dict())
        )
        self.assertFalse(
            self._state_dicts_equal(original_state_rank1, new_model_rank1.state_dict())
        )

        # Create resume managers for each rank
        resume_manager_rank0 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )
        resume_manager_rank1 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=1, metric="val_loss"
        )

        # Load checkpoints
        extra_vars_rank0, _ = resume_manager_rank0.load_checkpoint(
            new_model_rank0,
            new_optimizer_rank0,
            new_scheduler_rank0,
            model_name_or_path="dummy",
        )
        extra_vars_rank1, _ = resume_manager_rank1.load_checkpoint(
            new_model_rank1,
            new_optimizer_rank1,
            new_scheduler_rank1,
            model_name_or_path="dummy",
        )

        # Verify each rank loaded its own shard correctly
        self.assertTrue(
            self._state_dicts_equal(original_state_rank0, new_model_rank0.state_dict())
        )
        self.assertTrue(
            self._state_dicts_equal(original_state_rank1, new_model_rank1.state_dict())
        )

        # Verify extra vars
        self.assertEqual(extra_vars_rank0["step"], 100)
        self.assertEqual(extra_vars_rank1["step"], 100)

        # Verify scheduler state restored
        self.assertEqual(new_scheduler_rank0._step, 10)
        self.assertEqual(new_scheduler_rank1._step, 10)

    def test_ckpt_path_check_fails_with_missing_rank_marker(self):
        """Test that ckpt_path_check returns False when a rank marker is missing.

        When dp_shard > 1, all rank markers must exist for checkpoint to be valid.
        """
        # Create ParallelDims with dp_shard=2
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            world_size=2,
            pp_dynamic_shape=False,
        )

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)

        # Only save from rank 0 (rank 1 doesn't save)
        manager_rank0 = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        manager_rank0.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=1000,
        )

        # Check that checkpoint is incomplete (rank 1 marker missing)
        ckpt_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100", "policy"
        )
        self.assertFalse(manager_rank0.ckpt_path_check(ckpt_path))

        # Verify rank 0 marker exists but rank 1 doesn't
        self.assertTrue(os.path.exists(os.path.join(ckpt_path, ".rank_0_complete")))
        self.assertFalse(os.path.exists(os.path.join(ckpt_path, ".rank_1_complete")))

    def test_prune_corrupted_checkpoints(self):
        """Test that _prune_corrupted_checkpoints removes incomplete checkpoints.

        Scenario:
        - Save checkpoints at step 100, 200, 300
        - Corrupt checkpoint at step 200 by removing the complete marker
        - Resume with new manager
        - Verify step 200 was pruned from saved_step_dirs
        """
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        # Save checkpoints at step 100, 200, 300
        for step in [100, 200, 300]:
            manager.save_checkpoint(
                model, optimizer, scheduler, step=step, total_steps=1000
            )
            manager.save_check(step)

        # Verify all three checkpoints exist
        step_100_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100"
        )
        step_200_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_200"
        )
        step_300_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_300"
        )

        self.assertTrue(os.path.exists(step_100_path))
        self.assertTrue(os.path.exists(step_200_path))
        self.assertTrue(os.path.exists(step_300_path))

        # Corrupt step 200 by removing the complete marker
        complete_marker = os.path.join(step_200_path, "policy", ".rank_0_complete")
        os.remove(complete_marker)

        # Resume with a new manager - this should prune corrupted checkpoints during init
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        # Verify step 200 was pruned (directory deleted)
        self.assertFalse(os.path.exists(step_200_path))

        # Verify step 100 and 300 still exist
        self.assertTrue(os.path.exists(step_100_path))
        self.assertTrue(os.path.exists(step_300_path))

        # Verify saved_step_dirs only contains step 100 and 300
        saved_steps = [os.path.basename(d) for d in manager2.saved_ckpt_step_dirs]
        self.assertEqual(len(manager2.saved_ckpt_step_dirs), 2)
        self.assertIn("step_100", saved_steps)
        self.assertIn("step_300", saved_steps)
        self.assertNotIn("step_200", saved_steps)

    def test_best_ckpt_corrupted_resets_to_default(self):
        """Test that if best checkpoint is corrupted, init resets best_score to default and best_ckpt_abs_dir to None.

        Scenario:
        - Save checkpoints at step 100 and 200, with step 200 being the best
        - Corrupt step 200 (the best checkpoint) by removing the complete marker
        - Resume with new manager
        - Verify best_score is default (inf for loss metric) and best_ckpt_abs_dir is None
        """
        parallel_dims = create_test_parallel_dims()

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)
        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler()

        # Save checkpoint at step 100 with val_score
        manager.save_checkpoint(model, optimizer, scheduler, step=100, total_steps=1000)
        manager.save_check(100, val_score=1.0)

        # Save checkpoint at step 200 with better val_score (lower is better for loss)
        manager.save_checkpoint(model, optimizer, scheduler, step=200, total_steps=1000)
        manager.save_check(200, val_score=0.5)

        # Verify step 200 is the best checkpoint
        self.assertEqual(manager.best_score, 0.5)
        self.assertEqual(os.path.basename(manager.best_ckpt_abs_dir), "step_200")

        # Verify best symlink exists and points to step 200
        best_ckpt_link = os.path.join(self.test_dir, "best", "checkpoints")
        self.assertTrue(os.path.islink(best_ckpt_link))
        target = os.readlink(best_ckpt_link)
        self.assertTrue(target.endswith("step_200"))

        # Corrupt step 200 (the best checkpoint) by removing the complete marker
        step_200_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_200"
        )
        complete_marker = os.path.join(step_200_path, "policy", ".rank_0_complete")
        os.remove(complete_marker)

        # Resume with a new manager
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        # Verify best_score is reset to default (inf for loss metric)
        self.assertEqual(manager2.best_score, float("inf"))

        # Verify best_ckpt_abs_dir is None
        self.assertIsNone(manager2.best_ckpt_abs_dir)

        # Verify the corrupted checkpoint was pruned
        self.assertFalse(os.path.exists(step_200_path))

        # Verify step 100 still exists
        step_100_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100"
        )
        self.assertTrue(os.path.exists(step_100_path))

    def test_save_and_load_with_dp_replicate_greater_than_one(self):
        """Test save and load when dp_replicate > 1 (pure DP mode).

        When dp_replicate > 1, only one rank per replicate group saves.
        With dp_replicate=2, world_size=2: num_saving_ranks = 2/2 = 1 rank saves.
        """
        # Create ParallelDims with dp_replicate=2 (pure DP, only rank 0 saves)
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            world_size=2,
            pp_dynamic_shape=False,
        )

        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=False, max_keep=5)

        # Only rank 0 saves in pure DP mode
        manager_rank0 = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        torch.manual_seed(42)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler(step=15)

        original_state = copy.deepcopy(model.state_dict())

        manager_rank0.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=1000,
        )
        manager_rank0.save_check(100)

        # Verify checkpoint is complete with only rank 0 marker
        ckpt_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100", "policy"
        )
        self.assertTrue(manager_rank0.ckpt_path_check(ckpt_path))
        self.assertTrue(os.path.exists(os.path.join(ckpt_path, ".rank_0_complete")))
        # rank 1 marker should NOT exist in pure DP mode
        self.assertFalse(os.path.exists(os.path.join(ckpt_path, ".rank_1_complete")))

        # Resume from rank 0
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        resume_manager = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        torch.manual_seed(999)
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = SimpleScheduler(step=0)

        self.assertFalse(
            self._state_dicts_equal(original_state, new_model.state_dict())
        )

        extra_vars, _ = resume_manager.load_checkpoint(
            new_model, new_optimizer, new_scheduler, model_name_or_path="dummy"
        )

        # Verify model restored
        self.assertTrue(self._state_dicts_equal(original_state, new_model.state_dict()))
        self.assertEqual(extra_vars["step"], 100)
        self.assertEqual(new_scheduler._step, 15)


def _async_save_worker(
    rank: int,
    world_size: int,
    init_file: str,
    test_dir: str,
    timestamp: str,
    result_queue: mp.Queue,
):
    """Worker function for distributed async save test.

    Each worker:
    1. Initializes distributed process group
    2. Creates a model with rank-specific weights
    3. Saves checkpoint using async mode
    4. Waits for all ranks to finish saving
    5. Puts the original model state in the result queue for verification
    """
    try:
        # Initialize distributed process group
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        # Create ParallelDims for FSDP mode (all ranks save)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=world_size,
            cp=1,
            tp=1,
            pp=1,
            world_size=world_size,
            pp_dynamic_shape=False,
        )

        output_dir = os.path.join(test_dir, timestamp)
        config = create_test_config(
            output_dir=output_dir, resume=False, max_keep=5, save_mode="async"
        )

        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=rank, metric="val_loss"
        )

        # Create model with rank-specific weights
        torch.manual_seed(42 + rank)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = SimpleScheduler(step=10 + rank)
        scheduler_step = scheduler._step

        # Barrier to ensure all ranks are ready before saving
        dist.barrier()

        # Save checkpoint (async mode)
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            total_steps=1000,
        )

        # Finalize to wait for async save to complete
        manager.finalize()

        # Only rank 0 calls save_check
        if rank == 0:
            manager.save_check(100, val_score=0.5)

        # Barrier to ensure all ranks finished saving
        dist.barrier()

        # Put result in queue
        result_queue.put(
            {
                "rank": rank,
                "scheduler_step": scheduler_step,
                "success": True,
            }
        )

        dist.destroy_process_group()

    except Exception as e:
        result_queue.put(
            {
                "rank": rank,
                "error": str(e),
                "success": False,
            }
        )


def _async_load_worker(
    rank: int,
    world_size: int,
    init_file: str,
    test_dir: str,
    timestamp: str,
    result_queue: mp.Queue,
):
    """Worker function for distributed async load test.

    Each worker:
    1. Initializes distributed process group
    2. Regenerates expected model state using same seed as save worker
    3. Loads checkpoint
    4. Verifies the loaded state matches the expected
    """
    try:
        # Initialize distributed process group
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        # Create ParallelDims for FSDP mode
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=world_size,
            cp=1,
            tp=1,
            pp=1,
            world_size=world_size,
            pp_dynamic_shape=False,
        )

        output_dir = os.path.join(test_dir, timestamp)
        config = create_test_config(
            output_dir=output_dir, resume=True, max_keep=5, save_mode="sync"
        )

        manager = CheckpointMananger(
            config, parallel_dims=parallel_dims, global_rank=rank, metric="val_loss"
        )

        # Regenerate expected state using same seed as save worker
        torch.manual_seed(42 + rank)
        expected_model = SimpleModel()
        expected_state = copy.deepcopy(expected_model.state_dict())

        # Create new model with different weights for loading
        torch.manual_seed(999 + rank)
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = SimpleScheduler(step=0)

        # Barrier before loading
        dist.barrier()

        # Load checkpoint
        extra_vars, loaded_scheduler = manager.load_checkpoint(
            new_model, new_optimizer, new_scheduler, model_name_or_path="dummy"
        )

        # Barrier after loading
        dist.barrier()

        # Verify loaded state matches expected
        loaded_state = new_model.state_dict()

        # Compare states
        states_match = True
        for key in expected_state:
            if not torch.equal(expected_state[key], loaded_state[key]):
                states_match = False
                break

        result_queue.put(
            {
                "rank": rank,
                "states_match": states_match,
                "step": extra_vars.get("step"),
                "scheduler_step": loaded_scheduler._step,
                "success": True,
            }
        )

        dist.destroy_process_group()

    except Exception as e:
        result_queue.put(
            {
                "rank": rank,
                "error": str(e),
                "success": False,
            }
        )


class TestDistributedCheckpoint(unittest.TestCase):
    """Test checkpoint save/load with real multi-process distributed training."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.timestamp1 = "20250101000000"
        self.timestamp2 = "20250102000000"

    def tearDown(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_async_save_and_load_multiprocess(self):
        """Test async checkpoint save and load with multiple processes.

        This test simulates real distributed training with:
        - Multiple processes (world_size=2)
        - Async save mode
        - Each rank saves its own model shard
        - Resume and load from saved checkpoints
        """
        world_size = 2

        # Create init file for distributed rendezvous
        init_file = os.path.join(self.test_dir, "init_file_save")
        open(init_file, "w").close()

        # Queue to collect results from workers
        result_queue = mp.Queue()

        # === Phase 1: Async Save ===
        save_processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=_async_save_worker,
                args=(
                    rank,
                    world_size,
                    init_file,
                    self.test_dir,
                    self.timestamp1,
                    result_queue,
                ),
            )
            p.start()
            save_processes.append(p)

        # Wait for all save processes to complete
        for p in save_processes:
            p.join(timeout=30)
            self.assertFalse(p.is_alive(), "Save process timed out")

        # Collect save results
        save_results = {}
        for _ in range(world_size):
            result = result_queue.get(timeout=5)
            self.assertTrue(result["success"], f"Save failed: {result.get('error')}")
            save_results[result["rank"]] = result

        # Verify checkpoint files exist
        ckpt_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100", "policy"
        )
        self.assertTrue(os.path.exists(ckpt_path))
        for rank in range(world_size):
            self.assertTrue(
                os.path.exists(os.path.join(ckpt_path, f"model_rank_{rank}.pth"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(ckpt_path, f".rank_{rank}_complete"))
            )

        # === Phase 2: Load ===
        # Create new init file for load phase
        init_file_load = os.path.join(self.test_dir, "init_file_load")
        open(init_file_load, "w").close()

        load_result_queue = mp.Queue()
        load_processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=_async_load_worker,
                args=(
                    rank,
                    world_size,
                    init_file_load,
                    self.test_dir,
                    self.timestamp2,
                    load_result_queue,
                ),
            )
            p.start()
            load_processes.append(p)

        # Wait for all load processes to complete
        for p in load_processes:
            p.join(timeout=30)
            self.assertFalse(p.is_alive(), "Load process timed out")

        # Collect and verify load results
        for _ in range(world_size):
            result = load_result_queue.get(timeout=5)
            self.assertTrue(result["success"], f"Load failed: {result.get('error')}")
            self.assertTrue(
                result["states_match"],
                f"Rank {result['rank']}: loaded state doesn't match original",
            )
            self.assertEqual(result["step"], 100)
            # Each rank had scheduler step = 10 + rank
            expected_scheduler_step = 10 + result["rank"]
            self.assertEqual(result["scheduler_step"], expected_scheduler_step)

    def test_async_save_with_barrier_sync(self):
        """Test that async save completes correctly with barrier synchronization.

        Ensures that all ranks complete their async saves before any rank proceeds.
        """
        world_size = 3

        # Create init file
        init_file = os.path.join(self.test_dir, "init_file_barrier")
        open(init_file, "w").close()

        result_queue = mp.Queue()

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=_async_save_worker,
                args=(
                    rank,
                    world_size,
                    init_file,
                    self.test_dir,
                    self.timestamp1,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=30)
            self.assertFalse(p.is_alive(), "Process timed out")

        # Verify all succeeded
        for _ in range(world_size):
            result = result_queue.get(timeout=5)
            self.assertTrue(result["success"], f"Failed: {result.get('error')}")

        # Verify all rank markers exist
        ckpt_path = os.path.join(
            self.test_dir, self.timestamp1, "checkpoints", "step_100", "policy"
        )
        for rank in range(world_size):
            self.assertTrue(
                os.path.exists(os.path.join(ckpt_path, f".rank_{rank}_complete")),
                f"Rank {rank} complete marker missing",
            )


if __name__ == "__main__":
    unittest.main()
