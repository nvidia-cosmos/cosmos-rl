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

import os
import math
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch._dynamo

from accelerate import init_on_device
from transformers import AutoConfig, AutoProcessor

from cosmos_rl.policy.config import Config as CosmosConfig, ParallelismConfig
from cosmos_rl.policy.model.hf_models import HFModel
from cosmos_rl.utils.parallelism import ParallelDims


# Configure torch.compile to handle dynamic shapes in Dion optimizer
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.force_parameter_static_shapes = False

QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"


@contextmanager
def _cosmos_default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


_cached_qwen3_vl_model = None
_cached_qwen3_vl_device = None


def get_qwen3_vl_cosmos_model():
    global _cached_qwen3_vl_model, _cached_qwen3_vl_device
    if _cached_qwen3_vl_model is not None:
        return _cached_qwen3_vl_model, _cached_qwen3_vl_device
    if not torch.cuda.is_available():
        raise unittest.SkipTest("Qwen3-VL loader requires CUDA.")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    max_position_embeddings = 1024
    config = AutoConfig.from_pretrained(QWEN3_VL_MODEL_ID, trust_remote_code=True)
    config.max_position_embeddings = max_position_embeddings
    config.torch_dtype = dtype
    with init_on_device("meta", include_buffers=False):
        with _cosmos_default_dtype(dtype):
            cosmos_hf_model = HFModel.from_pretrained(
                config,
                QWEN3_VL_MODEL_ID,
                max_position_embeddings=max_position_embeddings,
            )
    cosmos_hf_model._apply(
        lambda t: torch.empty_like(t, device=device)
        if t.device.type == "meta"
        else t.to(device),
        recurse=True,
    )
    cosmos_hf_model.post_to_empty_hook(CosmosConfig())
    parallel_dims = ParallelDims.from_config(ParallelismConfig(tp_size=1))
    cosmos_hf_model.load_hf_weights(
        QWEN3_VL_MODEL_ID, parallel_dims, device, revision=None
    )
    _cached_qwen3_vl_model = cosmos_hf_model
    _cached_qwen3_vl_device = device
    return cosmos_hf_model, device


class NoLmHeadModel(nn.Module):
    """Minimal model without lm_head; used only for test_no_lm_head (Qwen3-VL has lm_head)."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.linear = nn.Linear(4, 4, bias=True)


class FakeMesh:
    """DeviceMesh-like object with ndim and __getitem__.
    When ndim==1, key ("dp_cp_tp",) returns self so get_orthonormal_optimizer_mesh
    succeeds without raising and without triggering the fallback log.
    """

    def __init__(self, mapping=None, ndim=1):
        self._mapping = dict(mapping or {})
        self.ndim = ndim

    def __getitem__(self, key):
        if key == ("dp_cp_tp",) and self.ndim == 1:
            return self
        if key in self._mapping:
            return self._mapping[key]
        raise KeyError(key)


class FakeConfig:
    """Minimal config with .train holding optimizer-related attributes. Uses a plain object so unset attributes return None from getattr(..., default), not MagicMock (which would break real optimizer constructors e.g. Muon comparing mu < 0)."""

    def __init__(self, **train_kw):
        self.train = type("Train", (), {})()
        for k, v in train_kw.items():
            setattr(self.train, k, v)


class TestIsOrthonormalOptimizer(unittest.TestCase):
    def test_returns_true_for_known_names(self):
        from cosmos_rl.policy.trainer.optm.utils import is_orthonormal_optimizer

        for name in ("Muon", "NorMuon", "Dion", "Dion2"):
            self.assertTrue(is_orthonormal_optimizer(name), f"Expected True for {name}")

    def test_returns_false_for_non_orthonormal_optimizer(self):
        from cosmos_rl.policy.trainer.optm.utils import is_orthonormal_optimizer

        for name in ("AdamW", "Adam", "Adam8bit", "SGD", ""):
            self.assertFalse(is_orthonormal_optimizer(name))


class TestSeparateParamGroupsForOrthonormalOptim(unittest.TestCase):
    def _call(self, **kwargs):
        from cosmos_rl.policy.trainer.optm.utils import (
            separate_param_groups_for_orthonormal_optim,
        )

        return separate_param_groups_for_orthonormal_optim(**kwargs)

    def test_basic_grouping(self):
        model, _ = get_qwen3_vl_cosmos_model()
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
        model = NoLmHeadModel()
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.01
        )
        self.assertEqual(len(groups), 3)

    def test_auto_lm_head_lr(self):
        model, _ = get_qwen3_vl_cosmos_model()
        base_lr = 1e-3
        groups = self._call(
            model=model, base_lr=base_lr, scalar_opt="adamw", weight_decay=0.0
        )
        lm_head_group = groups[3]
        d_in = lm_head_group["params"][0].shape[-1]
        expected_lr = base_lr / math.sqrt(float(d_in))
        self.assertAlmostEqual(lm_head_group["lr"], expected_lr)

    def test_explicit_lm_head_lr(self):
        model, _ = get_qwen3_vl_cosmos_model()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="adamw",
            weight_decay=0.0,
            lm_head_lr=5e-5,
        )
        self.assertAlmostEqual(groups[3]["lr"], 5e-5)

    def test_scalar_lr_and_embed_lr_overrides(self):
        model, _ = get_qwen3_vl_cosmos_model()
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
        model, _ = get_qwen3_vl_cosmos_model()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="adamw",
            weight_decay=0.0,
            scalar_lr=7e-4,
        )
        self.assertAlmostEqual(groups[2]["lr"], 7e-4)

    def test_scalar_betas_and_eps(self):
        model, _ = get_qwen3_vl_cosmos_model()
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
        model, _ = get_qwen3_vl_cosmos_model()
        # Disable one matrix param (first layer q_proj weight) so it is skipped from group 0.
        first_q_proj = model.model.language_model.layers[0].self_attn.q_proj.weight
        first_q_proj.requires_grad = False
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0
        )
        matrix_params = groups[0]["params"]
        # Use id() to avoid tensor comparison (in/== on parameters can raise RuntimeError).
        matrix_param_ids = {id(p) for p in matrix_params}
        self.assertNotIn(id(first_q_proj), matrix_param_ids)
        self.assertGreater(len(matrix_params), 0)

    def test_bias_goes_to_vector_group(self):
        model, _ = get_qwen3_vl_cosmos_model()
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0
        )
        vector_shapes = [p.shape for p in groups[1]["params"]]
        self.assertTrue(any(len(s) == 1 for s in vector_shapes))


class TestGetOrthonormalOptimizerMesh(unittest.TestCase):
    def _call(self, mesh_or_parallel_dims):
        from cosmos_rl.policy.trainer.optm.utils import get_orthonormal_optimizer_mesh

        return get_orthonormal_optimizer_mesh(mesh_or_parallel_dims)

    def test_none_returns_none(self):
        self.assertIsNone(self._call(None))

    def test_1d_mesh_returned_as_is(self):
        mesh = FakeMesh(ndim=1)
        self.assertIs(self._call(mesh), mesh)

    def test_cosmos_rl_dp_cp_tp_returns_1d_submesh(self):
        """Cosmos RL: mesh[('dp_cp_tp',)] is 1D submesh."""
        dp_cp_tp_mesh = FakeMesh(ndim=1)
        mesh = FakeMesh({("dp_cp_tp",): dp_cp_tp_mesh}, ndim=2)
        result = self._call(mesh)
        self.assertIs(result, dp_cp_tp_mesh)

    def test_parallel_dims_with_mesh_uses_mesh(self):
        """When passed ParallelDims-like object with .mesh, use that mesh."""
        dp_cp_tp_mesh = FakeMesh(ndim=1)
        mesh = FakeMesh({("dp_cp_tp",): dp_cp_tp_mesh}, ndim=2)
        parallel_dims = MagicMock()
        parallel_dims.mesh = mesh
        result = self._call(parallel_dims)
        self.assertIs(result, dp_cp_tp_mesh)


class TestBuildOrthonormalOptimizer(unittest.TestCase):
    """Builder uses param groups, config, and get_orthonormal_optimizer_mesh."""

    def _build(self, optimizer_name, config_dict, model=None, mesh=None):
        from cosmos_rl.policy.trainer.optm import utils as optm_utils
        from cosmos_rl.policy.trainer.optm import orthonormal_optimizers

        if model is None:
            model, _ = get_qwen3_vl_cosmos_model()
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
            patch.object(optm_utils, "use_builtin_torch_muon", return_value=False),
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
        model, _ = get_qwen3_vl_cosmos_model()
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

        with self.assertRaises(ValueError) as ctx:
            model, _ = get_qwen3_vl_cosmos_model()
            optm_utils.build_orthonormal_optimizer(
                "UnknownOpt", model, FakeConfig(optm_lr=1e-3)
            )
        self.assertIn("Unknown orthonormal optimizer", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests for backward + optimizer.step()
# ---------------------------------------------------------------------------


class TestOrthonormalOptimizerStep(unittest.TestCase):
    """Run forward, backward, and optimizer.step() with Qwen3-VL."""

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "backward/step test uses CUDA.",
    )
    def test_backward_and_optimizer_step_muon(self):
        """Forward -> loss -> backward -> optimizer.step() with Qwen3-VL and Muon."""
        from PIL import Image

        from qwen_vl_utils import process_vision_info

        from cosmos_rl.policy.trainer.optm import utils as optm_utils
        from cosmos_rl.policy.trainer.optm import orthonormal_optimizers

        if (
            orthonormal_optimizers.Muon is None
            and not optm_utils.use_builtin_torch_muon()
        ):
            self.skipTest(
                "dion not installed and PyTorch built-in Muon unavailable; cannot build Muon optimizer."
            )

        model, device = get_qwen3_vl_cosmos_model()
        model.train()
        config = FakeConfig(
            optm_lr=1e-5,
            optm_weight_decay=0.01,
            optm_betas=(0.9, 0.999),
            optm_scalar_opt="adamw",
        )
        built = optm_utils.build_orthonormal_optimizer(
            "Muon", model, config, distributed_mesh=None
        )
        optimizers = list(built) if isinstance(built, (list, tuple)) else [built]

        processor = AutoProcessor.from_pretrained(
            QWEN3_VL_MODEL_ID, trust_remote_code=True, use_fast=True
        )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(
            os.path.dirname(current_dir), "data", "test_hf_model.jpg"
        )
        if not os.path.isfile(test_image_path):
            self.skipTest(f"test image not found: {test_image_path}")
        image = Image.open(test_image_path)
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "describe the image"},
                    ],
                }
            ]
        ]
        text = processor.apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits
        loss = logits[:, -1, :].sum()

        # Snapshot one weight before step to assert it changes.
        param_to_track = model.model.language_model.layers[0].self_attn.q_proj.weight
        weight_before = param_to_track.data.clone()

        for o in optimizers:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in optimizers:
            o.step()

        self.assertTrue(loss.requires_grad)
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(
            torch.equal(weight_before, param_to_track.data),
            "Weights should have been updated by optimizer.step()",
        )
        # Assert update is in a reasonable range (meaningful step, no explosion).
        max_abs_diff = (param_to_track.data - weight_before).abs().max().item()
        self.assertGreater(
            max_abs_diff,
            1e-12,
            "Weight update too small; optimizer may not be stepping.",
        )
        # Loose upper bound: one step with lr=1e-5 gives small updates; 100 catches NaNs/explosion only.
        max_allowed_update = 100.0
        self.assertLess(
            max_abs_diff,
            max_allowed_update,
            "Weight update too large; possible numerical instability.",
        )


if __name__ == "__main__":
    unittest.main()
