# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""No optimizer enters an incomplete colocated dispatch; no fake success ACK."""

import asyncio
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.dispatcher.dispatch import TrainingDispatch
from cosmos_rl.dispatcher.command import TrainingCompleteCommand
from cosmos_rl.dispatcher.protocol import ColocatedPreparationRequest
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from test_running_dispatch_barrier import manager


def dispatched(*, completed=0, validation=False):
    status, first, second = manager(horizon=10)
    status.config.mode = "colocated"
    status.config.validation.enable = validation
    status.config.validation.freq = 1
    status.current_step = completed
    status.policy_init_done = True
    status.rollout_buffer = Queue()
    status.data_fetcher.validation_activate_dataloader = Mock()
    for replica in (first, second):
        replica.atoms = {0: NS(global_rank=0, report_session_id=replica.name)}
    status.try_trigger_data_fetch_and_training()
    # The generic fixture mocks DataFetcher; give cancellation a real marker.
    status.data_fetcher.activated_val_iter = object() if validation else None
    status.data_fetcher.activated_val_step = completed + 1 if validation else None

    def clear():
        status.data_fetcher.activated_val_iter = None
        status.data_fetcher.activated_val_step = None

    status.data_fetcher.clear_validation_status = Mock(side_effect=clear)
    return status, first, second


def prepare(status, replica, ready, *, step=None, **changes):
    kwargs = dict(
        replica_name=replica.name,
        step=step or status.current_step,
        total_steps=10,
        ready=ready,
        report_session_id=replica.name,
        src_global_rank=0,
    )
    kwargs.update(changes)
    return status.colocated_preparation(**kwargs)


@pytest.mark.parametrize("completed", [0, 4])
@pytest.mark.parametrize("readiness", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("validation", [False, True])
def test_exhaustion_cancels_only_unstarted_step(completed, readiness, validation):
    status, first, second = dispatched(completed=completed, validation=validation)
    step = completed + 1
    count = status.dispatched_rollouts_by_step[step]
    remaining, in_flight = status.remain_samples_num, status.samples_on_the_fly
    with patch.object(TrainingCompleteCommand, "trigger") as publish:
        assert prepare(status, first, readiness[0]) == "wait"
        assert status.current_step == step
        publish.assert_not_called()
        assert prepare(status, second, readiness[1]) == "stop"
        assert status.current_step == completed
        assert status.remain_samples_num == remaining + count
        assert status.samples_on_the_fly == in_flight - count
        assert not status.dispatched_rollouts_by_step
        assert not status.training_dispatches[step].reports
        assert status.training_dispatches[step].cancelled
        assert (
            status.stop_reason
            == "colocated dataset exhausted before a complete training batch"
        )
        assert not status.terminal_complete
        assert publish.call_count == 2
        for call in publish.call_args_list:
            assert call.kwargs["final_step"] == completed
            assert call.kwargs["checkpoint_total_steps"] == 10
            assert call.kwargs["remain_samples_num"] == remaining + count
        # A lost reply cannot cancel or publish terminal commands twice.
        assert prepare(status, first, readiness[0], step=step) == "stop"
        assert prepare(status, second, readiness[1], step=step) == "stop"
        assert publish.call_count == 2
    for replica in (first, second):
        status.record_completion_ack(replica.name, step)
    assert status.terminal_complete
    assert status.current_step == completed


def test_every_replica_must_prepare_before_training_ack():
    status, first, second = dispatched()
    assert prepare(status, first, True) == "wait"
    with pytest.raises(ValueError, match="without preparation agreement"):
        status.train_ack(first.name, 1, 10, False, {}, Mock())
    assert prepare(status, second, True) == "train"
    assert prepare(status, first, True) == "train"
    assert not status.stop_reason
    status.train_ack(first.name, 1, 10, False, {}, Mock())
    assert set(status.dispatched_rollouts_by_step) == {1}


def test_initial_local_dispatch_restores_progress_without_remote_reservations():
    status, first, second = dispatched()
    remaining = status.remain_samples_num
    status.training_dispatches[1].rollout_count = 0
    status.dispatched_rollouts_by_step[1] = 0
    status.samples_on_the_fly = 0
    with patch.object(TrainingCompleteCommand, "trigger"):
        assert prepare(status, first, False) == "wait"
        assert prepare(status, second, False) == "stop"
    assert status.remain_samples_num == remaining + 4
    assert status.samples_on_the_fly == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"report_session_id": "stale"},
        {"src_global_rank": 1},
        {"step": 5},
        {"total_steps": 9},
        {"replica_name": "stranger"},
    ],
)
def test_invalid_preparation_never_changes_accounting(changes):
    status, first, _ = dispatched()
    remaining = status.remain_samples_num
    with pytest.raises(ValueError):
        prepare(status, first, True, **changes)
    assert status.current_step == 1
    assert status.remain_samples_num == remaining
    assert not status.training_dispatches[1].preparation_reports


def test_ready_report_cannot_be_changed_after_agreement():
    dispatch = TrainingDispatch(
        1, 10, frozenset({"a", "b"}), 4, preparation_required=True
    )
    assert dispatch.prepare("a", 1, 10, True) == "wait"
    assert dispatch.prepare("b", 1, 10, True) == "train"
    with pytest.raises(ValueError, match="changed readiness"):
        dispatch.prepare("a", 1, 10, False)
    assert dispatch.preparation_decision == "train"


def test_previous_completed_validation_does_not_block_early_stop():
    status, first, second = dispatched(completed=4, validation=True)
    status.validation_round = NS(step=4, complete=True)
    with patch.object(TrainingCompleteCommand, "trigger"):
        assert prepare(status, first, False) == "wait"
        assert prepare(status, second, True) == "stop"


def test_active_validation_cannot_be_cancelled_as_unstarted():
    status, first, second = dispatched()
    status.validation_round = NS(step=0, complete=False)
    assert prepare(status, first, False) == "wait"
    with pytest.raises(RuntimeError, match="uncertain colocated dispatch"):
        prepare(status, second, True)
    assert status.current_step == 1
    assert not status.training_dispatches[1].settled


def test_local_pending_uses_weakest_rank_not_local_extrapolation():
    controller = object.__new__(ColocatedController)
    controller.config = NS(train=NS(train_policy=NS(uncentralized_training=True)))
    controller.policy = NS(world_size=2, data_queue=Queue())
    controller.policy.data_queue.put(object())
    with patch(
        "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu",
        return_value=[8, 4],
    ):
        assert controller.pending_policy_samples_all_replicas() == 8


@pytest.mark.parametrize("peer_reports", [[["b"], ["d"]], [["b"]], []])
def test_centralized_gather_keeps_callback_then_rank_order(peer_reports):
    controller = object.__new__(ColocatedController)
    controller.config = NS(train=NS(train_policy=NS(uncentralized_training=False)))
    controller.policy = NS(data_queue=Queue())
    controller.rollout = NS(
        parallel_dims=NS(
            mesh={"dp": NS(get_group=lambda: None, get_local_rank=lambda: 0)},
            cp_coord=(0, 1),
        )
    )
    controller._unreported_rollouts = [["a"], ["c"]]
    with patch(
        "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu",
        return_value=[[["a"], ["c"]], peer_reports],
    ):
        controller.synchronize_rollouts()
    expected = (
        ["a", "b", "c", "d"]
        if len(peer_reports) == 2
        else (["a", "b", "c"] if peer_reports else ["a", "c"])
    )
    assert list(controller.policy.data_queue.queue) == expected
    assert not controller._unreported_rollouts


def test_empty_generation_rank_stops_entire_generation_cohort():
    worker = NS(
        batch_size=2,
        _prompt_queue=Queue(),
        current_weight_version=0,
        request_new_prompts=Mock(return_value=True),
        one_step_generation=Mock(),
    )
    worker._prompt_queue.put([NS(weight_version=0)])
    with patch(
        "cosmos_rl.rollout.worker.colocated.rollout_control.dist_util.all_gather_object_cpu",
        return_value=[(True, True), (False, True)],
    ):
        assert ColocatedRolloutControlWorker.rollout_for_one_minor_step(worker) == (
            True,
            0,
        )
    worker.one_step_generation.assert_not_called()


def test_http_preparation_is_serialized_and_conflicts_are_permanent():
    from cosmos_rl.dispatcher import run_web_panel

    status, first, _ = dispatched()
    request = ColocatedPreparationRequest(
        replica_name=first.name,
        step=1,
        total_steps=10,
        ready=True,
        report_session_id=first.name,
        src_global_rank=0,
    )
    controller = NS(life_cycle_lock=asyncio.Lock(), policy_status_manager=status)
    with patch.object(run_web_panel, "controller", controller):
        assert asyncio.run(run_web_panel.colocated_preparation(request)) == {
            "decision": "wait"
        }
        response = asyncio.run(
            run_web_panel.colocated_preparation(
                request.model_copy(update={"ready": False})
            )
        )
        assert response.status_code == 409
