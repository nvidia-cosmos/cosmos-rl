# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for async R2R weight sync in rollout/worker/weight_sync.py.

Tests the config parsing, enum values, buffer model helpers, and
install_inference_sync wiring — all without a real GPU or NCCL.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from cosmos_rl.rollout.worker.weight_sync import (
    AsyncR2RSyncMode,
    create_buffer_model,
    get_async_r2r_sync_mode,
    get_broadcast_all_params,
    install_inference_sync,
    redirect_view_map_to_buffer,
    sync_buffer_to_live,
)


# ---------------------------------------------------------------------------
# AsyncR2RSyncMode enum
# ---------------------------------------------------------------------------


class TestAsyncR2RSyncMode:
    """Verify enum values match the TOML config strings."""

    def test_disabled_value(self):
        assert AsyncR2RSyncMode.DISABLED.value == "disabled"

    def test_generation_value(self):
        assert AsyncR2RSyncMode.GENERATION.value == "generation"

    def test_inference_value(self):
        assert AsyncR2RSyncMode.INFERENCE.value == "inference"

    def test_roundtrip_from_string(self):
        for mode in AsyncR2RSyncMode:
            assert AsyncR2RSyncMode(mode.value) is mode

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AsyncR2RSyncMode("bogus")


# ---------------------------------------------------------------------------
# get_async_r2r_sync_mode — config parsing
# ---------------------------------------------------------------------------


def _make_worker(async_r2r_sync="disabled", broadcast_all=False):
    """Build a minimal stub that looks like DisaggregatedRolloutControlWorker."""
    worker = SimpleNamespace()
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            async_r2r_sync=async_r2r_sync,
            broadcast_all_params=broadcast_all,
        ),
    )
    return worker


class TestGetAsyncR2RSyncMode:
    """Test config parsing from worker.config.rollout."""

    def test_defaults_to_disabled(self):
        worker = _make_worker()
        assert get_async_r2r_sync_mode(worker) == AsyncR2RSyncMode.DISABLED

    def test_parses_disabled(self):
        worker = _make_worker("disabled")
        assert get_async_r2r_sync_mode(worker) == AsyncR2RSyncMode.DISABLED

    def test_parses_generation(self):
        worker = _make_worker("generation")
        assert get_async_r2r_sync_mode(worker) == AsyncR2RSyncMode.GENERATION

    def test_parses_inference(self):
        worker = _make_worker("inference")
        assert get_async_r2r_sync_mode(worker) == AsyncR2RSyncMode.INFERENCE

    def test_invalid_value_raises(self):
        worker = _make_worker("banana")
        with pytest.raises(ValueError):
            get_async_r2r_sync_mode(worker)


class TestGetBroadcastAllParams:
    """Test broadcast_all_params config parsing."""

    def test_defaults_to_false(self):
        worker = _make_worker()
        assert get_broadcast_all_params(worker) is False

    def test_returns_true(self):
        worker = _make_worker(broadcast_all=True)
        assert get_broadcast_all_params(worker) is True


# ---------------------------------------------------------------------------
# create_buffer_model — CPU-only tests
# ---------------------------------------------------------------------------


class TestCreateBufferModel:
    """Test buffer model creation from a mock underlying model."""

    def _make_model_worker(self):
        """Create a worker with a simple 2-param model stub."""
        p1 = torch.randn(4, 4)
        p2 = torch.randn(3)
        model = MagicMock()
        model.state_dict.return_value = {"layer.weight": p1, "layer.bias": p2}
        model.parameters.return_value = iter([p1, p2])

        rollout = SimpleNamespace(get_underlying_model=lambda: model)
        worker = SimpleNamespace(rollout=rollout)
        return worker, p1, p2

    def test_creates_buffer_state_dict(self):
        worker, p1, p2 = self._make_model_worker()
        create_buffer_model(worker, device="cpu")

        assert hasattr(worker, "_buffer_state_dict")
        assert set(worker._buffer_state_dict.keys()) == {"layer.weight", "layer.bias"}

    def test_buffer_is_clone_not_alias(self):
        worker, p1, _ = self._make_model_worker()
        create_buffer_model(worker, device="cpu")

        buf_w = worker._buffer_state_dict["layer.weight"]
        assert torch.equal(buf_w, p1)
        assert buf_w.data_ptr() != p1.data_ptr()

    def test_initializes_version_counters(self):
        worker, _, _ = self._make_model_worker()
        create_buffer_model(worker, device="cpu")
        assert worker._buffer_version == 0
        assert worker._buffer_synced_version == 0


# ---------------------------------------------------------------------------
# redirect_view_map_to_buffer
# ---------------------------------------------------------------------------


class TestRedirectViewMapToBuffer:
    """Test that view map entries are redirected to buffer tensors."""

    def test_redirects_matching_keys(self):
        p1 = torch.randn(4, 4)
        buf_p1 = p1.clone()
        model = MagicMock()
        model.state_dict.return_value = {"layer.weight": p1}

        worker = SimpleNamespace(
            rollout=SimpleNamespace(get_underlying_model=lambda: model),
            weight_inplace_view_map={"layer.weight": p1},
            _buffer_state_dict={"layer.weight": buf_p1},
        )

        redirect_view_map_to_buffer(worker)

        assert worker.weight_inplace_view_map["layer.weight"] is buf_p1
        assert worker.weight_inplace_view_map["layer.weight"] is not p1

    def test_raises_on_unmatched_keys(self):
        """An entry that cannot be buffered must fail setup, not fall back.

        A receive target left on the live model would race inference, and
        its received updates would be overwritten with stale buffer data
        by the next ``sync_buffer_to_live``.
        """
        p1 = torch.randn(4)
        model = MagicMock()
        model.state_dict.return_value = {}

        worker = SimpleNamespace(
            rollout=SimpleNamespace(get_underlying_model=lambda: model),
            weight_inplace_view_map={"unknown_key": p1},
            _buffer_state_dict={},
        )

        with pytest.raises(RuntimeError, match="unknown_key"):
            redirect_view_map_to_buffer(worker)

    def test_rebuilds_fused_qkv_views_over_buffer(self):
        """q/k/v views of a fused parameter must land in the buffer.

        Regression test: matching views by ``data_ptr`` mapped the
        offset-zero q view to the full fused buffer tensor (wrong shape)
        and left the k/v views pointing at the live model, so the first
        P2R receive zeroed and checked the wrong tensors.
        """
        heads, dim = 4, 6
        qkv = torch.arange(3 * heads * dim, dtype=torch.float32).reshape(3 * heads, dim)
        buf_qkv = qkv.detach().clone()
        model = MagicMock()
        model.state_dict.return_value = {"visual.attn.qkv.weight": qkv}

        worker = SimpleNamespace(
            rollout=SimpleNamespace(get_underlying_model=lambda: model),
            weight_inplace_view_map={
                "visual.attn.q.weight": qkv[0:heads],
                "visual.attn.k.weight": qkv[heads : 2 * heads],
                "visual.attn.v.weight": qkv[2 * heads : 3 * heads],
            },
            _buffer_state_dict={"visual.attn.qkv.weight": buf_qkv},
        )

        redirect_view_map_to_buffer(worker)
        new_map = worker.weight_inplace_view_map

        for name in ("q", "k", "v"):
            view = new_map[f"visual.attn.{name}.weight"]
            assert view.shape == (heads, dim)
            assert (
                view.untyped_storage().data_ptr()
                == buf_qkv.untyped_storage().data_ptr()
            )
        assert new_map["visual.attn.v.weight"].storage_offset() == 2 * heads * dim

        # Writing through the redirected view mutates the buffer, not the
        # live model.
        new_map["visual.attn.k.weight"].zero_()
        assert buf_qkv[heads : 2 * heads].abs().sum() == 0
        assert qkv[heads : 2 * heads].abs().sum() > 0

    def test_redirects_flat_backed_disjoint_params(self):
        """Two params sharing one flat storage must both resolve correctly.

        Regression test: a single-candidate storage index kept only the
        last parameter per storage, so the other parameter (and any view
        of it) failed to redirect.
        """
        flat = torch.arange(24, dtype=torch.float32)
        a = flat[:8].view(2, 4)
        b = flat[8:].view(4, 4)
        sd = {"a.weight": a, "b.weight": b}
        buf = {k: v.detach().clone() for k, v in sd.items()}
        model = MagicMock()
        model.state_dict.return_value = sd

        worker = SimpleNamespace(
            rollout=SimpleNamespace(get_underlying_model=lambda: model),
            weight_inplace_view_map={
                "hf.a.weight": a,
                "hf.b.weight": b,
                "hf.b.rows12": b[1:3],
            },
            _buffer_state_dict=buf,
        )

        redirect_view_map_to_buffer(worker)
        new_map = worker.weight_inplace_view_map

        assert new_map["hf.a.weight"] is buf["a.weight"]
        assert new_map["hf.b.weight"] is buf["b.weight"]
        rows = new_map["hf.b.rows12"]
        assert rows.shape == (2, 4)
        assert (
            rows.untyped_storage().data_ptr()
            == buf["b.weight"].untyped_storage().data_ptr()
        )
        # b[1:3] sits at offset 12 in the flat storage; rebased against the
        # clone of b (whose storage starts at zero) it must land at 4.
        assert rows.storage_offset() == 4
        rows.zero_()
        assert buf["b.weight"][1:3].abs().sum() == 0
        assert b[1:3].abs().sum() > 0

    def test_redirects_renamed_nondense_param(self):
        """A renamed non-dense param maps to its same-shape buffer clone.

        Regression test: the clone of a non-dense parameter is contiguous,
        so a stride-equality guard rejected it even though the whole
        buffer tensor is a valid receive target.
        """
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        p = base[:, :3]  # non-dense: stride (6, 1) over a (4, 3) shape
        buf_p = p.detach().clone()  # contiguous clone: stride (3, 1)
        assert p.stride() != buf_p.stride()
        model = MagicMock()
        model.state_dict.return_value = {"w": p}

        worker = SimpleNamespace(
            rollout=SimpleNamespace(get_underlying_model=lambda: model),
            weight_inplace_view_map={"hf.w": p},
            _buffer_state_dict={"w": buf_p},
        )

        redirect_view_map_to_buffer(worker)
        assert worker.weight_inplace_view_map["hf.w"] is buf_p

    def test_dtensor_state_dict_exact_key_redirect(self):
        """DTensor state-dict entries must not crash the storage index.

        Regression test: building the index called
        ``untyped_storage().data_ptr()`` on every state-dict value, which
        raises on DTensors before any per-entry guard can run.  Exact-name
        DTensor entries must still redirect to their buffer clone.
        """
        import torch.distributed as dist

        if not dist.is_available():
            pytest.skip("torch.distributed not available")
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, distribute_tensor

        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cpu", (1,))
            dt = distribute_tensor(torch.randn(4, 4), mesh, [Replicate()])
            buf_dt = dt.detach().clone()
            plain = torch.randn(3)
            buf_plain = plain.detach().clone()
            model = MagicMock()
            model.state_dict.return_value = {"dt.weight": dt, "plain.bias": plain}

            worker = SimpleNamespace(
                rollout=SimpleNamespace(get_underlying_model=lambda: model),
                weight_inplace_view_map={
                    "dt.weight": dt,
                    "plain.bias": plain,
                },
                _buffer_state_dict={"dt.weight": buf_dt, "plain.bias": buf_plain},
            )

            redirect_view_map_to_buffer(worker)
            new_map = worker.weight_inplace_view_map
            assert new_map["dt.weight"] is buf_dt
            assert new_map["plain.bias"] is buf_plain
        finally:
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# sync_buffer_to_live — version gating (CPU-only)
# ---------------------------------------------------------------------------


class TestSyncBufferToLive:
    """Test version-gated sync logic without CUDA."""

    def _make_sync_worker(self, buf_ver=0, synced_ver=0):
        """Build a worker stub for sync_buffer_to_live tests."""
        worker = SimpleNamespace()
        worker._buffer_version = buf_ver
        worker._buffer_synced_version = synced_ver
        worker._buffer_state_dict = {}
        return worker

    def test_noop_when_versions_equal(self):
        worker = self._make_sync_worker(buf_ver=3, synced_ver=3)
        sync_buffer_to_live(worker)
        assert worker._buffer_synced_version == 3

    def test_noop_when_synced_ahead(self):
        worker = self._make_sync_worker(buf_ver=2, synced_ver=5)
        sync_buffer_to_live(worker)
        assert worker._buffer_synced_version == 5


# ---------------------------------------------------------------------------
# install_inference_sync — policy_fn wrapping
# ---------------------------------------------------------------------------


class TestInstallInferenceSync:
    """Test that install_inference_sync wraps the servicer's policy_fn."""

    def test_wraps_policy_fn(self):
        call_log = []

        def original_fn(obs):
            call_log.append(("original", obs))
            return {"action": 1}

        servicer = SimpleNamespace(policy_fn=original_fn)
        rollout = SimpleNamespace(_servicer=servicer)
        worker = SimpleNamespace(rollout=rollout)

        install_inference_sync(worker)

        assert servicer.policy_fn is not original_fn
        with patch("cosmos_rl.rollout.worker.weight_sync.sync_buffer_to_live"):
            result = servicer.policy_fn({"obs": 42})
        assert result == {"action": 1}
        assert call_log == [("original", {"obs": 42})]

    def test_wrapped_fn_calls_sync_buffer_to_live(self):
        def original_fn(obs):
            return {"action": 1}

        servicer = SimpleNamespace(policy_fn=original_fn)
        rollout = SimpleNamespace(_servicer=servicer)
        worker = SimpleNamespace(rollout=rollout)

        install_inference_sync(worker)

        with patch(
            "cosmos_rl.rollout.worker.weight_sync.sync_buffer_to_live"
        ) as mock_sync:
            servicer.policy_fn({"obs": 1})
            mock_sync.assert_called_once_with(worker)

    def test_warns_when_no_servicer(self):
        rollout = SimpleNamespace()
        worker = SimpleNamespace(rollout=rollout)

        with patch("cosmos_rl.rollout.worker.weight_sync.logger") as mock_logger:
            install_inference_sync(worker)
            mock_logger.warning.assert_called_once()
            assert "no _servicer" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# Config validation — Literal type enforcement
# ---------------------------------------------------------------------------


class TestConfigLiteralValidation:
    """Test that RolloutConfig validates async_r2r_sync values."""

    def test_valid_values_accepted(self):
        from cosmos_rl.policy.config import RolloutConfig

        for val in ("disabled", "generation", "inference"):
            cfg = RolloutConfig(async_r2r_sync=val)
            assert cfg.async_r2r_sync == val

    def test_invalid_value_rejected(self):
        from cosmos_rl.policy.config import RolloutConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RolloutConfig(async_r2r_sync="banana")
