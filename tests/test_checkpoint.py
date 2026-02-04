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


def create_test_config(output_dir, resume=False, max_keep=3, export_safetensors=False):
    """Create a CosmosConfig for testing with minimal required settings.

    Args:
        output_dir: Full path including timestamp, e.g., /tmp/test/20250101000000
        resume: Whether to resume from checkpoint
        max_keep: Maximum number of checkpoints to keep
        export_safetensors: Whether to export safetensors
    """
    # from_dict expects parent dir and appends timestamp, so we split them
    parent_dir = os.path.dirname(output_dir)
    timestamp = os.path.basename(output_dir)

    config_dict = {
        "train": {
            "output_dir": parent_dir,
            "resume": resume,
            "timestamp": timestamp,
            "ckpt": {
                "enable_checkpoint": True,
                "save_mode": "sync",
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
        self.assertEqual(manager1.best_step, 200)

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
        self.assertEqual(data["best_step"], 200)
        self.assertEqual(data["metric"], "val_loss")

        # === Second training session (resume) ===
        output_dir2 = os.path.join(self.test_dir, self.timestamp2)
        config2 = create_test_config(output_dir=output_dir2, resume=True, max_keep=5)
        manager2 = CheckpointMananger(
            config2, parallel_dims=parallel_dims, global_rank=0, metric="val_loss"
        )

        # Should have loaded best_score and best_step from first session
        self.assertEqual(manager2.best_score, 0.5)
        self.assertEqual(manager2.best_step, 200)

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
        self.assertEqual(manager.best_step, 100)

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
        self.assertEqual(manager.best_step, 100)

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
        config = create_test_config(output_dir=output_dir, resume=True)

        manager = CheckpointMananger(config, global_rank=0, metric="val_loss")
        self.assertEqual(manager.best_score, float("inf"))

    def test_best_score_default_for_non_loss_metric(self):
        """Test default best_score is -inf for non-loss metrics (like accuracy)."""
        output_dir = os.path.join(self.test_dir, self.timestamp1)
        config = create_test_config(output_dir=output_dir, resume=True)

        manager = CheckpointMananger(config, global_rank=0, metric="accuracy")
        self.assertEqual(manager.best_score, -float("inf"))


if __name__ == "__main__":
    unittest.main()
