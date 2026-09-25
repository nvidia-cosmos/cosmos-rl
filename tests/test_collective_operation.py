# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native failure cannot replay a partially committed logical operation."""

import threading
import gc
import weakref
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from cosmos_rl.utils import distributed as dist_utils


def _comm():
    comm = object.__new__(dist_utils.HighAvailabilitylNccl)
    comm.replica_name = "policy-0"
    comm.global_rank = 0
    comm.replica_name_to_rank = {"policy-0": 0, "policy-1": 1}
    comm.comm_idx = 7
    comm.max_retry = 3
    comm.default_timeout_ms = 10
    comm.is_single_peer = threading.Event()
    comm.is_comm_ready = threading.Event()
    comm.is_comm_ready.set()
    comm.build_mesh_lock = threading.RLock()
    comm.api_client = Mock()
    return comm


def test_partially_written_reduction_is_terminal_before_optimizer_commit(monkeypatch):
    comm = _comm()
    # Simulate a controller promptly publishing a rebuilt mesh. A new handle
    # does not make replay of this already modified logical input safe.
    comm.api_client.post_nccl_comm_error.side_effect = (
        lambda *_: comm.is_comm_ready.set()
    )
    weight = torch.nn.Parameter(torch.tensor([2.0]))
    weight.grad = torch.tensor([2.0])
    optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
    calls = []

    def mutate_then_fail(*, recvbuff, **kwargs):
        calls.append(True)
        recvbuff.add_(2)
        raise OSError("completion unknown after partial write")

    monkeypatch.setattr(dist_utils, "nccl_allreduce", mutate_then_fail)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", lambda **kw: nullcontext())
    with pytest.raises(RuntimeError):
        comm.allreduce(weight.grad, weight.grad, dist.ReduceOp.SUM)
        optimizer.step()
    assert calls == [True]
    assert weight.item() == 2.0
    assert weight.grad.item() == 4.0  # damaged operation storage, never replayed
    assert not optimizer.state
    with pytest.raises(dist_utils.CollectiveOperationError):
        comm.wait_comm_ready()


def test_stream_completion_failure_is_terminal_even_after_successful_enqueue(
    monkeypatch,
):
    comm = _comm()
    comm.api_client.post_nccl_comm_error.side_effect = (
        lambda *_: comm.is_comm_ready.set()
    )
    raw = Mock()
    raw.__name__ = "allreduce"

    @contextmanager
    def fail_completion(**kwargs):
        yield
        raise TimeoutError("injected device completion timeout")

    monkeypatch.setattr(dist_utils, "nccl_allreduce", raw)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", fail_completion)
    with pytest.raises(RuntimeError) as caught:
        comm.allreduce(torch.ones(1), torch.ones(1), dist.ReduceOp.SUM)
    assert isinstance(caught.value.__cause__, TimeoutError)
    raw.assert_called_once()


def test_mesh_remains_pinned_through_stream_completion(monkeypatch):
    comm = _comm()
    observations = []

    @contextmanager
    def check_completion(**kwargs):
        yield

        def attempt_rebuild_lock():
            acquired = comm.build_mesh_lock.acquire(blocking=False)
            observations.append(acquired)
            if acquired:
                comm.build_mesh_lock.release()

        thread = threading.Thread(target=attempt_rebuild_lock)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()

    monkeypatch.setattr(dist_utils, "nccl_allreduce", lambda **kw: None)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", check_completion)
    comm.allreduce(torch.ones(1), torch.ones(1), dist.ReduceOp.SUM)
    assert observations == [False]


@pytest.mark.parametrize("phase", ["call", "stream"])
def test_native_success_after_deadline_cannot_commit_optimizer(monkeypatch, phase):
    comm = _comm()
    now = [0.0]
    calls = []
    weight = torch.nn.Parameter(torch.ones(1))
    weight.grad = torch.ones_like(weight)
    optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)

    def late_success(**kwargs):
        calls.append(True)
        if phase == "call":
            now[0] = 0.020  # 10 ms operation budget already elapsed.

    @contextmanager
    def late_stream(**kwargs):
        yield
        if phase == "stream":
            now[0] = 0.020

    monkeypatch.setattr(dist_utils.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(dist_utils, "nccl_allreduce", late_success)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", late_stream)
    with pytest.raises(RuntimeError):
        comm.allreduce(weight.grad, weight.grad, dist.ReduceOp.SUM)
        optimizer.step()
    assert calls == [True]
    assert weight.item() == 1.0
    assert not optimizer.state
    assert not comm.is_ready()


def test_native_batch_consumes_one_deadline_without_issuing_late_calls(monkeypatch):
    comm = _comm()
    now = [0.0]
    budgets = []

    def slow_send(*, timeout_ms, **kwargs):
        budgets.append(timeout_ms)
        now[0] += 0.006

    monkeypatch.setattr(dist_utils.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(dist_utils, "nccl_send", slow_send)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", lambda **kw: nullcontext())
    with pytest.raises(RuntimeError):
        comm.send_batch([torch.ones(1) for _ in range(3)], "policy-1")
    assert budgets == [10, 4]
    assert not comm.is_ready()


@pytest.mark.parametrize(
    "method,raw,key",
    [
        ("broadcast_batch", "nccl_broadcast", "rank"),
        ("send_batch", "nccl_send", "peer"),
        ("recv_batch", "nccl_recv", "peer"),
    ],
)
def test_peer_is_resolved_after_acquiring_current_mesh(monkeypatch, method, raw, key):
    comm = _comm()
    lock = comm.build_mesh_lock

    @contextmanager
    def rebuild_wins_lock():
        with lock:
            comm.replica_name_to_rank = {"policy-0": 1, "policy-1": 0}
            yield

    # Model a completed rebuild between readiness and acquisition. The old
    # implementation resolved policy-1 as rank 1 before acquiring this lock.
    class Lock:
        def __enter__(self):
            self.contexts.append(rebuild_wins_lock())
            return self.contexts[-1].__enter__()

        def __exit__(self, *args):
            return self.contexts.pop().__exit__(*args)

        def __init__(self):
            self.contexts = []

    comm.build_mesh_lock = Lock()
    calls = []
    monkeypatch.setattr(dist_utils, raw, lambda **kwargs: calls.append(kwargs[key]))
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", lambda **kw: nullcontext())
    getattr(comm, method)([torch.ones(1)], "policy-1")
    assert calls == [0]


def test_single_peer_allreduce_copies_separate_send_storage(monkeypatch):
    comm = _comm()
    comm.is_single_peer.set()
    send, recv = torch.tensor([4.0]), torch.zeros(1)
    comm.allreduce(send, recv, dist.ReduceOp.SUM)
    torch.testing.assert_close(send, recv)


def test_later_gradient_bucket_failure_prevents_optimizer_commit(monkeypatch):
    comm = _comm()
    parameters = [torch.nn.Parameter(torch.tensor([float(i)])) for i in (1, 2)]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.9)
    calls = []

    def fail_second_bucket(*, recvbuff, **kwargs):
        calls.append(True)
        recvbuff.mul_(2)
        if len(calls) == 2:
            raise OSError("second bucket partially completed")

    monkeypatch.setattr(dist_utils, "_GRADIENT_BUCKET_BYTES", 8)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    monkeypatch.setattr(dist_utils, "nccl_allreduce", fail_second_bucket)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", lambda **kw: nullcontext())
    with pytest.raises(dist_utils.CollectiveOperationError):
        dist_utils.gradient_reduce_across_dp_replicas_(parameters, comm)
        optimizer.step()
    assert len(calls) == 2
    assert [p.item() for p in parameters] == [1.0, 2.0]
    assert not optimizer.state


def test_uncertain_operands_remain_owned_after_communicator_teardown(monkeypatch):
    def fail(**kwargs):
        raise OSError("native completion is unknown")

    monkeypatch.setattr(dist_utils, "nccl_allreduce", fail)
    monkeypatch.setattr(dist_utils, "nccl_timeout_watchdog", lambda **kw: nullcontext())
    start = len(dist_utils._FAILED_COLLECTIVE_BUFFERS)

    def failed_operation():
        comm = _comm()
        operand = torch.ones(1)
        reference = weakref.ref(operand)
        try:
            comm.allreduce(operand, operand, dist.ReduceOp.SUM)
        except dist_utils.CollectiveOperationError:
            pass
        return reference

    try:
        reference = failed_operation()
        gc.collect()
        assert reference() is not None
        assert len(dist_utils._FAILED_COLLECTIVE_BUFFERS) == start + 1
    finally:
        # The test never issued native work. Production quarantine is terminal
        # and intentionally has no release API without completion proof.
        del dist_utils._FAILED_COLLECTIVE_BUFFERS[start:]
