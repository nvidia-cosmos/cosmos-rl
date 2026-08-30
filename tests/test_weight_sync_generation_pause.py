# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for quiescing generation across a weight sync (CPU-only).

The backends here record when they are parked and released; the transport
calls are stubbed, so the ordering of those records against the transfer is
what is being checked.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_rl.dispatcher.command import (
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker.weight_sync import (
    AsyncR2RSyncMode,
    generation_paused,
)


class _ParkingBackend:
    """A backend that keeps serving until it is asked to park."""

    def __init__(self, events):
        self._events = events

    @contextmanager
    def paused(self):
        self._events.append("park")
        try:
            yield
        finally:
            self._events.append("resume")

    def model_param_map(self, _weight_mapper):
        return {"a.weight": torch.zeros(2)}


def _make_worker(events):
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.rollout = _ParkingBackend(events)
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
    worker.state = SimpleNamespace(
        weight_synced=lambda: True, set_weight_synced=lambda: None
    )
    worker.config = SimpleNamespace(
        validation=SimpleNamespace(enable=False, val_before_train=False, freq=1)
    )
    worker.lazy_initialize_rollout_engine = lambda _load_format: None
    return worker


def _broadcast(worker, events, async_mode=AsyncR2RSyncMode.DISABLED):
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
            "cosmos_rl.rollout.worker.weight_sync.get_async_r2r_sync_mode",
            lambda _worker: async_mode,
        ),
        patch(
            "cosmos_rl.rollout.worker.rollout_control.get_async_r2r_sync_mode",
            lambda _worker: async_mode,
        ),
        patch(
            "cosmos_rl.rollout.worker.rollout_control.get_broadcast_all_params",
            lambda _worker: False,
        ),
    ):
        worker.broadcast_to_all_rollout_replica(command)


def test_r2r_writes_the_served_model_with_generation_parked():
    events = []
    worker = _make_worker(events)

    _broadcast(worker, events)

    assert events == ["park", "broadcast", "resume"]


def test_p2r_receives_with_generation_parked():
    events = []
    worker = _make_worker(events)
    worker.lazy_initialize_rollout_engine = lambda _load_format: events.append(
        "lazy init"
    )
    # Addressed to a peer, so the handler initializes the engine and returns
    # without a transfer of its own.
    command = PolicyToRolloutUnicastCommand(
        src_replica_name="policy-0",
        dst_replica_name="rollout-1",
        src_replica_size=1,
        dst_replica_size=1,
    )

    with patch(
        "cosmos_rl.rollout.worker.weight_sync.get_async_r2r_sync_mode",
        lambda _worker: AsyncR2RSyncMode.DISABLED,
    ):
        worker.policy_to_rollout_unicast(command)

    assert events == ["park", "lazy init", "resume"]


def test_an_async_sync_leaves_generation_running():
    # The async modes write a buffer clone, and the swap into the served model
    # takes its own safe point, so there is nothing to park for.
    events = []
    worker = SimpleNamespace(rollout=_ParkingBackend(events))

    with patch(
        "cosmos_rl.rollout.worker.weight_sync.get_async_r2r_sync_mode",
        lambda _worker: AsyncR2RSyncMode.GENERATION,
    ):
        with generation_paused(worker):
            events.append("sync")

    assert events == ["sync"]


class _BlockingBackend(RolloutBase):
    """A backend whose generation call blocks, as every in-tree one does."""

    def post_init_hook(self, **kwargs):
        pass

    def rollout_generation(self, *args, **kwargs):
        pass

    def init_engine(self, *args, **kwargs):
        pass

    def get_underlying_model(self):
        pass


def test_a_blocking_engine_needs_no_pause_of_its_own():
    # The default: an engine whose generation call blocks the main loop is
    # already quiesced by the time a command can be dequeued.
    backend = object.__new__(_BlockingBackend)

    with backend.paused():
        pass
