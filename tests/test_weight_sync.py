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

    def test_keeps_unmatched_keys(self):
        p1 = torch.randn(4)
        model = MagicMock()
        model.state_dict.return_value = {}

        worker = SimpleNamespace(
            rollout=SimpleNamespace(get_underlying_model=lambda: model),
            weight_inplace_view_map={"unknown_key": p1},
            _buffer_state_dict={},
        )

        redirect_view_map_to_buffer(worker)
        assert worker.weight_inplace_view_map["unknown_key"] is p1


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
