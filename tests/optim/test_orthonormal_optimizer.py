# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for orthonormal optimizer utils.
# Run: python -m unittest tests.optim.test_orthonormal_optimizer

import math
import unittest
from unittest.mock import MagicMock, patch

import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model with embed, linear (matrix + optional bias), and optional lm_head."""

    def __init__(self, with_lm_head=True, with_bias=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.linear = nn.Linear(4, 4, bias=with_bias)
        if with_lm_head:
            self.lm_head = nn.Linear(4, 10, bias=False)


class FakeMesh:
    """DeviceMesh-like object with ndim and __getitem__."""

    def __init__(self, mapping=None, ndim=1):
        self._mapping = dict(mapping or {})
        self.ndim = ndim

    def __getitem__(self, key):
        if key not in self._mapping:
            raise KeyError(key)
        return self._mapping[key]


class FakeConfig:
    """Minimal config with .train holding optimizer-related attributes."""

    def __init__(self, **train_kw):
        self.train = MagicMock()
        for k, v in train_kw.items():
            setattr(self.train, k, v)


# ---------------------------------------------------------------------------
# Tests for is_orthonormal_optimizer()
# ---------------------------------------------------------------------------


class TestIsOrthonormalOptimizer(unittest.TestCase):
    def test_returns_true_for_known_names(self):
        from cosmos_rl.policy.trainer.optm.utils import is_orthonormal_optimizer

        for name in ("Muon", "NorMuon", "Dion", "Dion2"):
            self.assertTrue(is_orthonormal_optimizer(name), f"Expected True for {name}")

    def test_returns_false_for_non_orthonormal_optimizer(self):
        from cosmos_rl.policy.trainer.optm.utils import is_orthonormal_optimizer

        for name in ("AdamW", "Adam", "Adam8bit", "SGD", ""):
            self.assertFalse(is_orthonormal_optimizer(name))


# ---------------------------------------------------------------------------
# Tests for separate_param_groups_for_orthonormal_optim()
# ---------------------------------------------------------------------------


class TestSeparateParamGroupsForOrthonormalOptim(unittest.TestCase):
    def _call(self, **kwargs):
        from cosmos_rl.policy.trainer.optm.utils import (
            separate_param_groups_for_orthonormal_optim,
        )

        return separate_param_groups_for_orthonormal_optim(**kwargs)

    def test_basic_grouping(self):
        model = TinyModel()
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.01
        )
        self.assertEqual(len(groups), 4)
        self.assertNotIn("algorithm", groups[0])
        self.assertEqual(groups[1]["algorithm"], "adamw")
        self.assertEqual(groups[2]["algorithm"], "adamw")
        self.assertEqual(groups[2]["weight_decay"], 0.0)
        self.assertEqual(groups[3]["algorithm"], "adamw")
        self.assertEqual(groups[3]["weight_decay"], 0.0)

    def test_no_lm_head(self):
        model = TinyModel(with_lm_head=False)
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.01
        )
        self.assertEqual(len(groups), 3)

    def test_auto_lm_head_lr(self):
        model = TinyModel()
        base_lr = 1e-3
        groups = self._call(
            model=model, base_lr=base_lr, scalar_opt="adamw", weight_decay=0.0
        )
        lm_head_group = groups[3]
        expected_lr = base_lr / math.sqrt(4.0)
        self.assertAlmostEqual(lm_head_group["lr"], expected_lr)

    def test_explicit_lm_head_lr(self):
        model = TinyModel()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="adamw",
            weight_decay=0.0,
            lm_head_lr=5e-5,
        )
        self.assertAlmostEqual(groups[3]["lr"], 5e-5)

    def test_scalar_lr_and_embed_lr_overrides(self):
        model = TinyModel()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="lion",
            weight_decay=0.0,
            scalar_lr=2e-4,
            embed_lr=3e-4,
        )
        self.assertAlmostEqual(groups[1]["lr"], 2e-4)
        self.assertAlmostEqual(groups[2]["lr"], 3e-4)

    def test_scalar_lr_defaults_embed_lr(self):
        model = TinyModel()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="adamw",
            weight_decay=0.0,
            scalar_lr=7e-4,
        )
        self.assertAlmostEqual(groups[2]["lr"], 7e-4)

    def test_scalar_betas_and_eps(self):
        model = TinyModel()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="adamw",
            weight_decay=0.0,
            scalar_betas=(0.9, 0.999),
            scalar_eps=1e-8,
        )
        for g in groups[1:]:
            self.assertAlmostEqual(g["beta1"], 0.9)
            self.assertAlmostEqual(g["beta2"], 0.999)
            self.assertAlmostEqual(g["epsilon"], 1e-8)

    def test_requires_grad_false_skipped(self):
        model = TinyModel()
        for p in model.linear.parameters():
            p.requires_grad = False
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0
        )
        self.assertEqual(len(groups[0]["params"]), 0)

    def test_bias_goes_to_vector_group(self):
        model = TinyModel(with_bias=True)
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0
        )
        vector_shapes = [p.shape for p in groups[1]["params"]]
        self.assertTrue(any(len(s) == 1 for s in vector_shapes))


# ---------------------------------------------------------------------------
# Tests for get_orthonormal_optimizer_mesh()
# ---------------------------------------------------------------------------


class TestGetOrthonormalOptimizerMesh(unittest.TestCase):
    """None, 1D passthrough, Cosmos RL dp_shard_cp, Automodel 2D, fallback."""

    def _call(self, mesh_or_parallel_dims):
        from cosmos_rl.policy.trainer.optm.utils import get_orthonormal_optimizer_mesh

        return get_orthonormal_optimizer_mesh(mesh_or_parallel_dims)

    def test_none_returns_none(self):
        self.assertIsNone(self._call(None))

    def test_1d_mesh_returned_as_is(self):
        mesh = FakeMesh(ndim=1)
        self.assertIs(self._call(mesh), mesh)

    def test_no_ndim_returned_as_is(self):
        mesh = object()
        self.assertIs(self._call(mesh), mesh)

    def test_cosmos_rl_dp_shard_cp_returns_1d_submesh(self):
        """Cosmos RL: mesh[('dp_shard_cp',)] is 1D submesh."""
        dp_shard_cp_mesh = FakeMesh(ndim=1)
        mesh = FakeMesh({("dp_shard_cp",): dp_shard_cp_mesh}, ndim=2)
        result = self._call(mesh)
        self.assertIs(result, dp_shard_cp_mesh)

    def test_automodel_style_2d_extracts_submesh(self):
        inner = FakeMesh(ndim=1)
        dp_2d = FakeMesh({"dp_shard_cp": inner}, ndim=2)
        mesh = FakeMesh({("dp_replicate", "dp_shard_cp"): dp_2d}, ndim=2)
        result = self._call(mesh)
        self.assertIs(result, inner)

    def test_automodel_style_fallback_on_key_error(self):
        mesh = FakeMesh({}, ndim=2)
        result = self._call(mesh)
        self.assertIs(result, mesh)

    def test_parallel_dims_with_mesh_uses_mesh(self):
        """When passed ParallelDims-like object with .mesh, use that mesh."""
        dp_shard_cp_mesh = FakeMesh(ndim=1)
        mesh = FakeMesh({("dp_shard_cp",): dp_shard_cp_mesh}, ndim=2)
        parallel_dims = MagicMock()
        parallel_dims.mesh = mesh
        result = self._call(parallel_dims)
        self.assertIs(result, dp_shard_cp_mesh)


# ---------------------------------------------------------------------------
# Tests for build_orthonormal_optimizer()
# ---------------------------------------------------------------------------


class TestBuildOrthonormalOptimizer(unittest.TestCase):
    """Builder uses param groups, config, and get_orthonormal_optimizer_mesh."""

    def _build(self, optimizer_name, config_dict, model=None, mesh=None):
        from cosmos_rl.policy.trainer.optm import utils as optm_utils
        from cosmos_rl.policy.trainer.optm import orthonormal_optimizers

        if model is None:
            model = TinyModel()
        config = FakeConfig(**config_dict)

        captured = {}

        class FakeOpt:
            # Include distributed_mesh (and common kwargs) so builder's filter keeps them
            def __init__(self, param_groups, distributed_mesh=None, lr=None, **kwargs):
                captured["param_groups"] = param_groups
                captured["kwargs"] = {
                    "distributed_mesh": distributed_mesh,
                    "lr": lr,
                    **kwargs,
                }

        with (
            patch.object(orthonormal_optimizers, "_import_error", None),
            patch.object(orthonormal_optimizers, "Muon", FakeOpt),
            patch.object(orthonormal_optimizers, "NorMuon", FakeOpt),
            patch.object(orthonormal_optimizers, "Dion", FakeOpt),
            patch.object(orthonormal_optimizers, "Dion2", FakeOpt),
        ):
            result = optm_utils.build_orthonormal_optimizer(
                optimizer_name, model, config, distributed_mesh=mesh
            )
        return result, captured

    def test_param_groups_structure(self):
        model = TinyModel()
        _, captured = self._build(
            "Muon",
            {"optm_lr": 1e-3, "optm_weight_decay": 0.01, "optm_betas": (0.9, 0.95)},
            model=model,
        )
        groups = captured["param_groups"]
        self.assertEqual(len(groups), 4)
        for g in groups:
            self.assertIn("params", g)
            self.assertIsInstance(g["params"], list)

    def test_passes_distributed_mesh_when_given(self):
        mesh = FakeMesh(ndim=1)
        _, captured = self._build(
            "Muon",
            {"optm_lr": 1e-3},
            mesh=mesh,
        )
        self.assertIs(captured["kwargs"].get("distributed_mesh"), mesh)

    def test_none_mesh(self):
        _, captured = self._build("Muon", {"optm_lr": 1e-3}, mesh=None)
        self.assertIsNone(captured["kwargs"].get("distributed_mesh"))

    def test_unknown_optimizer_raises(self):
        from cosmos_rl.policy.trainer.optm import utils as optm_utils
        from cosmos_rl.policy.trainer.optm import orthonormal_optimizers

        with patch.object(orthonormal_optimizers, "_import_error", None):
            with self.assertRaises(ValueError) as ctx:
                optm_utils.build_orthonormal_optimizer(
                    "UnknownOpt", TinyModel(), FakeConfig(optm_lr=1e-3)
                )
            self.assertIn("Unknown orthonormal optimizer", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
