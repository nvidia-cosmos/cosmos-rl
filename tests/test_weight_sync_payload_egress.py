# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for holding payload egress across a weight sync (CPU-only).

The packers here record when they are held and released; the transport calls
are stubbed, so the ordering of those records against the broadcast is what is
being checked.
"""

import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_rl.dispatcher.command import (
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
from cosmos_rl.rollout.worker import weight_sync
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker.weight_sync import (
    AsyncR2RSyncMode,
    WeightSyncThread,
    payload_egress_held,
)


class _HoldingPacker:
    """A packer that can keep its egress off the device for a whole sync."""

    def __init__(self, events):
        self._events = events

    @contextmanager
    def hold_sends(self):
        self._events.append("hold")
        try:
            yield
        finally:
            self._events.append("release")


class _FlushingPacker:
    """A packer from before ``hold_sends``, which can only drain on demand."""

    def __init__(self, events):
        self._events = events

    def flush_pending_sends(self):
        self._events.append("flush")


def _make_worker(events, packer):
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.data_packer = packer
    worker.replica_name = "rollout-0"
    worker.rank_in_rollout_repicas = 0
    worker.replica_name_to_rank = {"rollout-0": 0, "rollout-1": 1}
    worker.global_commnicator_idex = 7
    worker.inference_stream = None
    worker.weight_mapper = None
    worker.trainable_params = {"a.weight"}
    worker.non_trainable_params_received = True
    worker.current_weight_version = 3
    worker.prepare_trainable_params = lambda: None
    worker.rollout = SimpleNamespace(
        model_param_map=lambda _mapper: {"a.weight": torch.zeros(2)}
    )
    worker.state = SimpleNamespace(
        weight_synced=lambda: True, set_weight_synced=lambda: None
    )
    worker.config = SimpleNamespace(
        validation=SimpleNamespace(enable=False, val_before_train=False, freq=1)
    )
    worker.lazy_initialize_rollout_engine = lambda _load_format: events.append(
        "lazy init"
    )
    return worker


def _broadcast(worker, events):
    command = RolloutToRolloutBroadcastCommand(
        src_replica_name="rollout-0",
        dst_replica_names=["rollout-0", "rollout-1"],
        weight_step=4,
        total_steps=10,
        trainable_only=True,
    )
    with (
        patch(
            "cosmos_rl.rollout.worker.rollout_control.nccl_broadcast",
            lambda *_args: events.append("broadcast"),
        ),
        patch(
            "cosmos_rl.rollout.worker.rollout_control.get_async_r2r_sync_mode",
            lambda _worker: AsyncR2RSyncMode.DISABLED,
        ),
        patch(
            "cosmos_rl.rollout.worker.rollout_control.get_broadcast_all_params",
            lambda _worker: False,
        ),
    ):
        worker.broadcast_to_all_rollout_replica(command)


def test_r2r_holds_egress_for_the_whole_broadcast():
    events = []
    worker = _make_worker(events, _HoldingPacker(events))

    _broadcast(worker, events)

    assert events == ["hold", "broadcast", "release"]


def test_r2r_still_drains_a_packer_that_cannot_hold():
    events = []
    worker = _make_worker(events, _FlushingPacker(events))

    _broadcast(worker, events)

    assert events == ["flush", "broadcast"]


def test_p2r_holds_egress_too():
    events = []
    worker = _make_worker(events, _HoldingPacker(events))
    # Addressed to a peer, so the handler initializes the engine and returns
    # without a transfer of its own.
    command = PolicyToRolloutUnicastCommand(
        src_replica_name="policy-0",
        dst_replica_name="rollout-1",
        src_replica_size=1,
        dst_replica_size=1,
    )

    worker.policy_to_rollout_unicast(command)

    assert events == ["hold", "lazy init", "release"]


def test_the_weight_sync_thread_holds_egress_around_a_command():
    events = []
    wst = object.__new__(WeightSyncThread)
    wst._queue = queue.PriorityQueue()
    wst._stop = threading.Event()
    wst._idle = threading.Event()
    wst._worker = SimpleNamespace(device="cpu", data_packer=_HoldingPacker(events))
    wst._queue.put((0, 0, ("r2r", object())))

    def execute_r2r(_command):
        events.append("r2r")
        wst._stop.set()

    wst._execute_r2r = execute_r2r
    with patch.object(weight_sync.torch.cuda, "set_device"):
        wst._run()

    assert events == ["hold", "r2r", "release"]


def test_a_packer_that_ships_nothing_over_nccl_needs_neither_hook():
    worker = SimpleNamespace(data_packer=SimpleNamespace())

    with payload_egress_held(worker):
        pass


def test_a_worker_without_a_packer_is_left_alone():
    with payload_egress_held(SimpleNamespace()):
        pass
