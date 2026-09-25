# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual receive closures queue only submitted copies, never unrecorded events."""

from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils import constant
from cosmos_rl.utils.parallelism_map import (
    WeightSyncInstruction,
    WeightSyncInstructionsGroup,
    WeightSyncInstructionsPerParam,
)


class Tensor:
    def __init__(self, dtype=torch.float32, device="cuda:0"):
        self.dtype = dtype
        self.device = torch.device(device)
        self.shape = (4,)
        self.record_stream = Mock()

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def cosmos_slice(self, slices):
        return self

    def to(self, dtype):
        return Tensor(dtype)

    def numel(self):
        return 4

    def element_size(self):
        return 2


class Event:
    def __init__(self):
        self.recorded = False
        self.finished = False
        self.waits = 0

    def query(self):
        # Real CUDA also reports an unrecorded event complete. The caller must
        # not use this result to retire a receive whose copy has not been issued.
        return not self.recorded or self.finished

    def record(self):
        self.recorded = True

    def synchronize(self):
        assert self.recorded, "cannot drain an unissued copy inside an open group"
        self.waits += 1
        self.finished = True


def make_worker(monkeypatch, *, off_device=False, fail_copy=False):
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: "copy-stream")
    monkeypatch.setattr(torch, "empty_like", lambda t, **kw: Tensor(t.dtype))
    monkeypatch.setattr(constant, "COSMOS_RECV_TENSOR_QUEUE_SIZE", 2)
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.device = torch.device("cuda:0")
    worker.config = SimpleNamespace(train=SimpleNamespace(transfer_dtype="float16"))
    worker.quantization_type = None
    worker.temp_recv_tensor_queue = Queue()
    worker.get_underlying_model = lambda: None
    worker.parallel_dims = None
    worker.trainable_params = {"a", "b", "c"}
    worker.weight_inplace_view_map = {
        name: Tensor(
            torch.float16 if off_device else torch.float32,
            "cpu" if off_device else "cuda:0",
        )
        for name in worker.trainable_params
    }
    worker.p2r_collective_manager = SimpleNamespace(recv=Mock())
    worker.weight_mapper = SimpleNamespace(
        update_tensor_view=Mock(
            side_effect=RuntimeError("copy failed") if fail_copy else None
        )
    )
    return worker


def receive(worker, names=("a", "b", "c")):
    group = WeightSyncInstructionsGroup(
        [
            WeightSyncInstructionsPerParam(name, [WeightSyncInstruction(0, 0, {})])
            for name in names
        ]
    )
    return worker.recv_weight_shard(0, group, "pair", False)[1]


def test_pending_atomic_round_is_not_published_as_completed(monkeypatch):
    worker = make_worker(monkeypatch)
    complete = receive(worker)
    assert worker.temp_recv_tensor_queue.empty()
    complete()
    entries = list(worker.temp_recv_tensor_queue.queue)
    # An atomic round may exceed the nominal count; do not wait for an event
    # whose copy cannot be issued until the current NCCL group has closed.
    assert len(entries) == 3
    for tensor, event in entries:
        assert event.recorded and not event.finished and event.waits == 0
        tensor.record_stream.assert_called_once_with("copy-stream")


@pytest.mark.parametrize("off_device", [False, True])
def test_next_round_backpressures_only_recorded_previous_copies(
    monkeypatch, off_device
):
    worker = make_worker(monkeypatch, off_device=off_device)
    receive(worker)()
    events = [entry[1] for entry in worker.temp_recv_tensor_queue.queue]
    next_complete = receive(worker, ("a",))
    assert sum(event.waits for event in events) == 1
    assert worker.temp_recv_tensor_queue.qsize() == 1
    next_complete()
    assert worker.temp_recv_tensor_queue.qsize() == 2
    assert all(event.recorded for _, event in worker.temp_recv_tensor_queue.queue)


def test_completed_copies_are_retired_without_blocking(monkeypatch):
    worker = make_worker(monkeypatch)
    receive(worker)()
    events = [entry[1] for entry in worker.temp_recv_tensor_queue.queue]
    for event in events:
        event.finished = True
    complete = receive(worker, ("a",))
    assert worker.temp_recv_tensor_queue.empty()
    assert not any(event.waits for event in events)
    complete()
    assert worker.temp_recv_tensor_queue.qsize() == 1


def test_failed_copy_does_not_publish_false_completion(monkeypatch):
    worker = make_worker(monkeypatch, fail_copy=True)
    complete = receive(worker, ("a",))
    with pytest.raises(RuntimeError, match="copy failed"):
        complete()
    assert worker.temp_recv_tensor_queue.empty()
    tensor = worker.p2r_collective_manager.recv.call_args.args[1]
    tensor.record_stream.assert_called_once_with("copy-stream")
