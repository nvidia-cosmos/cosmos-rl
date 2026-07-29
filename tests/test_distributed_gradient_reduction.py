# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inter-replica gradient and collective failure semantics."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist

from cosmos_rl.utils import distributed as dist_utils


class _RecordingCommunicator:
    def __init__(self) -> None:
        self.ops: list[dist.ReduceOp] = []
        self.wait_calls = 0

    def wait_comm_ready(self) -> None:
        self.wait_calls += 1

    def world_size(self) -> int:
        return 2

    def allreduce(
        self,
        sendbuff: torch.Tensor,
        recvbuff: torch.Tensor,
        op: dist.ReduceOp,
        timeout_ms: int | None = None,
    ) -> None:
        del timeout_ms
        torch.testing.assert_close(sendbuff, recvbuff)
        self.ops.append(op)
        recvbuff.mul_(2.0)


def test_gradient_reduction_supports_sum_and_requires_every_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RSSM can SUM a complete, deterministic trainable-parameter set."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0])),
        torch.nn.Parameter(torch.tensor([2.0])),
    ]
    for parameter, gradient in zip(parameters, (3.0, 5.0), strict=True):
        parameter.grad = torch.tensor([gradient])
    communicator = _RecordingCommunicator()

    dist_utils.gradient_reduce_across_dp_replicas_(
        parameters,
        communicator,
        reduce_op=dist.ReduceOp.SUM,
        require_all_gradients=True,
    )

    assert communicator.wait_calls == 1
    assert communicator.ops == [dist.ReduceOp.SUM]
    torch.testing.assert_close(parameters[0].grad, torch.tensor([6.0]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([10.0]))


def test_gradient_reduction_rejects_missing_required_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0])),
        torch.nn.Parameter(torch.tensor([2.0])),
    ]
    parameters[0].grad = torch.tensor([3.0])

    communicator = _RecordingCommunicator()
    with pytest.raises(RuntimeError, match="missing gradients across policy replicas"):
        dist_utils.gradient_reduce_across_dp_replicas_(
            parameters,
            communicator,
            reduce_op=dist.ReduceOp.SUM,
            require_all_gradients=True,
        )

    assert communicator.ops == [dist.ReduceOp.SUM]


def test_strict_gradient_reduction_rejects_empty_parameter_set() -> None:
    with pytest.raises(RuntimeError, match="at least one parameter"):
        dist_utils.gradient_reduce_across_dp_replicas_(
            [],
            _RecordingCommunicator(),
            reduce_op=dist.ReduceOp.SUM,
            require_all_gradients=True,
        )


def test_gradient_reduction_accepts_one_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([3.0])
    communicator = _RecordingCommunicator()

    dist_utils.gradient_reduce_across_dp_replicas_(parameter, communicator)

    assert communicator.ops == [dist.ReduceOp.AVG]
    torch.testing.assert_close(parameter.grad, torch.tensor([6.0]))


def test_strict_gradient_reduction_detects_membership_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([3.0])

    with pytest.raises(RuntimeError, match="expected participant"):
        dist_utils.gradient_reduce_across_dp_replicas_(
            [parameter],
            _RecordingCommunicator(),
            reduce_op=dist.ReduceOp.SUM,
            require_all_gradients=True,
            expected_participants=3,
        )


def test_gradient_reduction_keeps_average_as_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing Cosmos trainers retain their historical AVG behavior."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([3.0])
    communicator = _RecordingCommunicator()

    dist_utils.gradient_reduce_across_dp_replicas_([parameter], communicator)

    assert communicator.ops == [dist.ReduceOp.AVG]


def test_default_gradient_reduction_still_skips_missing_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0])),
        torch.nn.Parameter(torch.tensor([2.0])),
    ]
    parameters[0].grad = torch.tensor([3.0])
    communicator = _RecordingCommunicator()

    dist_utils.gradient_reduce_across_dp_replicas_(parameters, communicator)

    assert communicator.ops == [dist.ReduceOp.AVG]
    torch.testing.assert_close(parameters[0].grad, torch.tensor([6.0]))
    assert parameters[1].grad is None


def test_expected_participants_requires_strict_mode() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([3.0])

    with pytest.raises(ValueError, match="requires strict"):
        dist_utils.gradient_reduce_across_dp_replicas_(
            parameter,
            _RecordingCommunicator(),
            expected_participants=2,
        )


def test_ha_nccl_raises_after_exhausting_collective_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collective cannot silently return with unreduced state."""
    attempts = 0

    def fail_allreduce(**_kwargs) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("synthetic NCCL failure")

    error_reports: list[Exception] = []
    communicator = dist_utils.HighAvailabilitylNccl.__new__(
        dist_utils.HighAvailabilitylNccl
    )
    communicator.replica_name = "policy-0"
    communicator.global_rank = 0
    communicator.replica_name_to_rank = {}
    communicator.comm_idx = 7
    communicator.max_retry = 3
    communicator.default_timeout_ms = 10
    communicator.is_single_peer = threading.Event()
    communicator.is_comm_ready = threading.Event()
    communicator.is_comm_ready.set()
    communicator.build_mesh_lock = threading.Lock()
    communicator.api_client = SimpleNamespace(
        post_nccl_comm_error=lambda _name, error: error_reports.append(error)
    )
    communicator.wait_comm_ready = lambda timeout=0: None
    monkeypatch.setattr(dist_utils, "nccl_allreduce", fail_allreduce)
    monkeypatch.setattr(
        dist_utils,
        "nccl_timeout_watchdog",
        lambda **_kwargs: nullcontext(),
    )

    with pytest.raises(RuntimeError, match="failed after 3 attempts") as error:
        communicator.allreduce(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            dist.ReduceOp.SUM,
        )

    assert attempts == 3
    assert len(error_reports) == 3
    assert isinstance(error.value.__cause__, OSError)


def test_ha_nccl_returns_after_a_transient_collective_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def eventually_reduce(
        sendbuff: torch.Tensor,
        recvbuff: torch.Tensor,
        **_kwargs,
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("transient NCCL failure")
        recvbuff.copy_(sendbuff * 2.0)

    error_reports: list[Exception] = []
    communicator = dist_utils.HighAvailabilitylNccl.__new__(
        dist_utils.HighAvailabilitylNccl
    )
    communicator.replica_name = "policy-0"
    communicator.global_rank = 0
    communicator.replica_name_to_rank = {}
    communicator.comm_idx = 7
    communicator.max_retry = 3
    communicator.default_timeout_ms = 10
    communicator.is_single_peer = threading.Event()
    communicator.is_comm_ready = threading.Event()
    communicator.is_comm_ready.set()
    communicator.build_mesh_lock = threading.Lock()
    communicator.api_client = SimpleNamespace(
        post_nccl_comm_error=lambda _name, error: error_reports.append(error)
    )
    communicator.wait_comm_ready = lambda timeout=0: None
    monkeypatch.setattr(dist_utils, "nccl_allreduce", eventually_reduce)
    monkeypatch.setattr(
        dist_utils,
        "nccl_timeout_watchdog",
        lambda **_kwargs: nullcontext(),
    )
    send = torch.tensor([2.0])
    receive = torch.zeros_like(send)

    communicator.allreduce(send, receive, dist.ReduceOp.SUM)

    assert attempts == 3
    assert len(error_reports) == 2
    torch.testing.assert_close(receive, torch.tensor([4.0]))
