# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.rollout import State
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker import rollout_control as control


def worker(*, synced, frozen_received, broadcast_all=False):
    instance = object.__new__(DisaggregatedRolloutControlWorker)
    instance.replica_name = "rollout-b"
    instance.state = State()
    if synced:
        instance.state.set_weight_synced()
    instance.non_trainable_params_received = frozen_received
    instance.current_weight_version = 0
    instance.config = SimpleNamespace(
        rollout=SimpleNamespace(
            async_r2r_sync="disabled", broadcast_all_params=broadcast_all
        ),
        validation=SimpleNamespace(enable=False, val_before_train=False, freq=1),
    )
    instance.prepare_trainable_params = Mock()
    instance.trainable_params = {"trainable"}
    instance.weight_mapper = None
    instance.rank_in_rollout_repicas = 1
    instance.replica_name_to_rank = {"rollout-a": 0, "rollout-b": 1}
    instance.global_commnicator_idex = 7
    instance.inference_stream = SimpleNamespace(synchronize=Mock())
    parameters = {"trainable": torch.ones(2), "frozen": torch.zeros(3)}
    instance.rollout = SimpleNamespace(model_param_map=lambda mapper: parameters)
    instance.validation_flag = threading.Event()
    return instance, parameters


def command(*, trainable_only):
    return SimpleNamespace(
        src_replica_name="rollout-a",
        dst_replica_names=["rollout-a", "rollout-b"],
        trainable_only=trainable_only,
        weight_step=1,
        total_steps=5,
        replica_should_stop=lambda: False,
    )


@pytest.mark.parametrize("synced", [False, True])
def test_unseeded_trainable_command_rejected_before_native_selection(
    monkeypatch, synced
):
    instance, _ = worker(synced=synced, frozen_received=False)
    native = Mock(return_value=(1, 8))
    monkeypatch.setattr(control, "do_nccl_broadcast_tensors", native)
    error = None
    try:
        instance._execute_rollout_broadcast(command(trainable_only=True))
    except (RuntimeError, AssertionError) as caught:
        error = caught
    assert error is not None, "cannot silently change the command's tensor selection"
    native.assert_not_called()
    assert not instance.non_trainable_params_received
    assert instance.current_weight_version == 0


@pytest.mark.parametrize("synced", [False, True])
@pytest.mark.parametrize("trainable_only", [False, True])
def test_full_state_path_records_frozen_receipt_regardless_of_command_hint(
    monkeypatch, synced, trainable_only
):
    instance, _ = worker(synced=synced, frozen_received=False, broadcast_all=True)
    native = Mock(return_value=(2, 20))
    monkeypatch.setattr(control, "do_nccl_broadcast_grouped", native)
    instance._execute_rollout_broadcast(command(trainable_only=trainable_only))
    native.assert_called_once()
    assert instance.state.weight_synced()
    assert instance.non_trainable_params_received
    assert instance.current_weight_version == 1


@pytest.mark.parametrize("trainable_only", [False, True])
def test_seeded_trainable_path_preserves_authoritative_command_selection(
    monkeypatch, trainable_only
):
    instance, parameters = worker(synced=True, frozen_received=True)
    native = Mock(
        return_value=(1 if trainable_only else 2, 8 if trainable_only else 20)
    )
    monkeypatch.setattr(control, "do_nccl_broadcast_tensors", native)
    instance._execute_rollout_broadcast(command(trainable_only=trainable_only))
    selected = native.call_args.args[1]
    assert len(selected) == (1 if trainable_only else 2)
    assert selected[0] is parameters["trainable"]
    if not trainable_only:
        assert selected[1] is parameters["frozen"]
    assert instance.current_weight_version == 1
