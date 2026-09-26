# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual pinned-to-CUDA final-reader tests; require CUDA in the live gate."""

import os
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cosmos_rl.utils.payload_transport.ucxx import operation as module
from cosmos_rl.utils.payload_transport.ucxx.strategy import UCXXTransportStrategy
from cosmos_rl.utils.transport_failure import TransportUnusableError


if os.environ.get("COSMOS_REQUIRE_CUDA") == "1":
    assert torch.cuda.is_available(), "required CUDA gate must not silently skip"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def candidate():
    strategy = UCXXTransportStrategy()
    strategy._device = torch.device("cuda", torch.cuda.current_device())
    strategy._read_timeout = 5
    strategy._client = Mock()
    arrays = {
        "observations": np.arange(20, dtype=np.float32).reshape(5, 4),
        "episode_length": np.array([3], dtype=np.int64),
        "mask": np.array([True, False, True], dtype=np.bool_),
    }
    pinned = torch.empty(
        sum(array.nbytes for array in arrays.values()),
        dtype=torch.uint8,
        pin_memory=True,
    )
    offset = 0
    payload = {}
    for key, array in arrays.items():
        target = np.frombuffer(
            pinned.numpy()[offset : offset + array.nbytes], dtype=array.dtype
        ).reshape(array.shape)
        target[:] = array
        payload[key] = target
        offset += array.nbytes
    payload["_pinned_buf"] = pinned
    return strategy, payload


def test_actual_device_copy_finishes_before_recycle_and_keeps_outputs_owned():
    strategy, payload = candidate()
    pinned = payload["_pinned_buf"]
    strategy._client.return_pinned.side_effect = lambda buf: buf.fill_(255)
    outputs = []
    for version in range(4):
        payload["observations"][:] = np.arange(20).reshape(5, 4) + version
        payload["episode_length"][:] = 3
        payload["mask"][:] = [True, False, True]
        outputs.append(strategy._copy_to_device(payload))
        assert torch.all(pinned == 255)
        for old_version, output in enumerate(outputs):
            torch.testing.assert_close(
                output["observations"],
                torch.arange(20, dtype=torch.float32, device="cuda").reshape(5, 4)[:3]
                + old_version,
            )
            assert output["mask"].tolist() == [True, False, True]
            assert output["episode_length"].item() == 3
    assert strategy._client.return_pinned.call_count == 4


@pytest.mark.parametrize("failure", ["pending", "record"])
def test_uncertain_cuda_completion_retains_host_and_device_storage(
    monkeypatch, failure
):
    strategy, payload = candidate()
    strategy._read_timeout = 0.03
    failures, retained = [], []
    monkeypatch.setattr(module, "fail_transport", failures.append)
    monkeypatch.setattr(module, "_TERMINAL_OPERATIONS", retained)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            # The copy queues behind actual outstanding device work.
            torch.cuda._sleep(1_000_000_000)
            if failure == "record":
                event = Mock()
                event.record.side_effect = RuntimeError("record failed after enqueue")
                monkeypatch.setattr(torch.cuda, "Event", lambda: event)
            with pytest.raises(TransportUnusableError):
                strategy._copy_to_device(payload)
            strategy._client.return_pinned.assert_not_called()
            assert failures and strategy._failure
            assert any(owner is payload["_pinned_buf"] for owner in retained[0].owners)
            assert any(
                isinstance(owner, torch.Tensor) and owner.is_cuda
                for owner in retained[0].owners
            )
    finally:
        # Fixture-only cleanup: the actual worker exits without attempting it.
        stream.synchronize()


def test_allocation_failure_after_copy_issue_is_safe_only_after_device_drain(
    monkeypatch,
):
    strategy, payload = candidate()
    failure = Mock()
    monkeypatch.setattr(module, "fail_transport", failure)
    clone = torch.Tensor.clone

    def fail_device_clone(value, *args, **kwargs):
        if value.is_cuda:
            raise torch.OutOfMemoryError("decoded allocation rejected after H2D issue")
        return clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", fail_device_clone)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            with pytest.raises(ValueError, match="after safe drain"):
                strategy._copy_to_device(payload)
            assert stream.query(), (
                "pinned storage was released before native completion"
            )
            strategy._client.return_pinned.assert_called_once()
            failure.assert_not_called()
            assert not strategy._failure
    finally:
        stream.synchronize()
