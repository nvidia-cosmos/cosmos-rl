# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the synchronous R2R broadcast shape (CPU-only).

No NCCL traffic: the transport calls are stubbed with fakes that record the
call order and, like a real group, only deliver a receive buffer's data once
the group closes.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_rl.dispatcher.command import RolloutToRolloutBroadcastCommand
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker.weight_sync import AsyncR2RSyncMode


class _FakeTransport:
    """Records the call order and defers every receive fill to group end."""

    def __init__(self):
        self.calls: list[str] = []
        self._pending: list[torch.Tensor] = []

    def group_start(self, comm_idx):
        self.calls.append("group_start")

    def broadcast(self, tensor, rank, comm_idx):
        self.calls.append("broadcast")
        self._pending.append(tensor)

    def group_end(self, comm_idx):
        self.calls.append("group_end")
        for tensor in self._pending:
            tensor.fill_(1.0)
        self._pending.clear()


def _make_worker(param_map, trainable_params):
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.replica_name = "rollout-0"
    worker.rank_in_rollout_repicas = 0
    worker.replica_name_to_rank = {"rollout-0": 0, "rollout-1": 1}
    worker.global_commnicator_idex = 7
    worker.inference_stream = None
    worker.weight_mapper = None
    worker.trainable_params = trainable_params
    worker.non_trainable_params_received = True
    worker.current_weight_version = 3
    worker.prepare_trainable_params = lambda: None
    worker.rollout = SimpleNamespace(model_param_map=lambda _mapper: param_map)
    worker.state = SimpleNamespace(
        weight_synced=lambda: True, set_weight_synced=lambda: None
    )
    worker.config = SimpleNamespace(
        validation=SimpleNamespace(enable=False, val_before_train=False, freq=1)
    )
    return worker


def _broadcast(worker, trainable_only):
    command = RolloutToRolloutBroadcastCommand(
        src_replica_name="rollout-0",
        dst_replica_names=["rollout-0", "rollout-1"],
        weight_step=4,
        total_steps=10,
        trainable_only=trainable_only,
    )
    transport = _FakeTransport()
    with (
        patch(
            "cosmos_rl.rollout.worker.rollout_control.nccl_group_start",
            transport.group_start,
        ),
        patch(
            "cosmos_rl.rollout.worker.rollout_control.nccl_broadcast",
            transport.broadcast,
        ),
        patch(
            "cosmos_rl.rollout.worker.rollout_control.nccl_group_end",
            transport.group_end,
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
    return transport


def test_trainable_subset_travels_as_one_group():
    param_map = {name: torch.zeros(2) for name in ("a.weight", "b.weight", "c.weight")}
    worker = _make_worker(param_map, trainable_params={"a.weight", "b.weight"})

    transport = _broadcast(worker, trainable_only=True)

    # One group around the whole subset, and the frozen parameter stays home.
    assert transport.calls == ["group_start", "broadcast", "broadcast", "group_end"]


def test_whole_state_dict_travels_when_not_trainable_only():
    param_map = {name: torch.zeros(2) for name in ("a.weight", "b.weight", "c.weight")}
    worker = _make_worker(param_map, trainable_params={"a.weight"})

    transport = _broadcast(worker, trainable_only=False)

    assert transport.calls.count("broadcast") == len(param_map)


def test_non_contiguous_parameter_is_written_after_the_group_closes():
    # A transposed view: the broadcast lands in a contiguous copy, and the copy
    # back into the parameter is only valid once the group has delivered it.
    parameter = torch.zeros(2, 3).t()
    assert not parameter.is_contiguous()
    worker = _make_worker({"a.weight": parameter}, trainable_params={"a.weight"})

    _broadcast(worker, trainable_only=True)

    assert torch.equal(parameter, torch.ones(3, 2))
