# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real controller receipts and training ACKs during validation-aware drain."""

from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher import command
from cosmos_rl.dispatcher.protocol import ValidationReportRequest
from cosmos_rl.dispatcher.status import JobPhase
import test_terminal_drain_protocol as terminal_fixture
from test_validation_delivery_contract import manager as validation_manager


def setup(monkeypatch, accepted, *, freq=10):
    policies, policy = terminal_fixture.TestTerminalMatrix._manager(accepted)
    validation, _, rollout = validation_manager([])
    validation.data_fetcher.clear_validation_status()
    policies.data_fetcher = validation.data_fetcher
    policies.custom_logger_fns = []
    policies.config.validation = validation.config.validation
    policies.config.validation.freq = freq
    policies.config.validation.val_before_train = False
    policies.config.train.train_policy.type = "grpo"
    policies.config.train.train_policy.on_policy = False
    policies.config.train.sync_weight_interval = 1
    policies.config.train.coalesce_weight_sync = False
    policies.policy_atoms_in_replica = 1
    rollouts = SimpleNamespace(
        all_rollouts_ended=lambda: True,
        get_safe_weight_sync_replicas=lambda **kwargs: [rollout],
        rollout_atoms_in_replica=1,
        replica_scaling_log=[],
    )
    p2r, r2r, complete = Mock(), Mock(), Mock()
    monkeypatch.setattr(command.PolicyToRolloutUnicastCommand, "trigger", p2r)
    monkeypatch.setattr(command.RolloutToRolloutBroadcastCommand, "trigger", r2r)
    monkeypatch.setattr(command.TrainingCompleteCommand, "trigger", complete)
    return policies, rollouts, rollout, p2r, r2r, complete


def finish_round(policies, rollouts):
    round_ = policies.validation_round
    batch = policies.fetch_validation_prompts(
        2, round_.step, None, round_.round_id, "a", 0
    )
    assert not batch.payloads and batch.is_end
    request = ValidationReportRequest(
        src_replica_name="a",
        src_global_rank=0,
        validation_step=round_.step,
        validation_round_id=round_.round_id,
        report_sequence=0,
        payloads=[],
        is_end=True,
    )
    policies.validation_report_validation_results(
        round_.step, [], rollouts, request=request
    )


@pytest.mark.parametrize("accepted,real_steps", [(0, 0), (1, 0), (2, 1), (3, 1)])
def test_early_exhaustion_validates_last_real_version_before_completion(
    monkeypatch, accepted, real_steps
):
    policies, rollouts, _, p2r, r2r, complete = setup(monkeypatch, accepted)
    policies.on_rollout_is_end(rollouts)
    assert policies.job_phase == JobPhase.DRAINING
    complete.assert_not_called()
    if real_steps:
        assert policies.dispatched_rollouts_by_step == {1: 2}
        policies.train_ack("policy-0", 1, 2, False, {}, rollouts)
    assert policies.current_step == real_steps
    assert policies.total_steps == policies.training_horizon() == 2
    assert policies.validation_round.step == real_steps
    assert not policies.validation_round.complete
    assert r2r.call_args.kwargs["total_steps"] == 2
    complete.assert_not_called()
    finish_round(policies, rollouts)
    complete.assert_called_once()
    assert complete.call_args.kwargs["final_step"] == real_steps
    assert complete.call_args.kwargs["checkpoint_total_steps"] == 2
    assert policies.current_step == real_steps
    assert policies.samples_on_the_fly == 0
    assert not policies.validation_drained
    completion_step = policies.completion_step
    policies.train_ack(
        "policy-0", completion_step, completion_step, False, {}, rollouts
    )
    assert policies.validation_drained


def test_active_initial_round_keeps_accepted_tail_until_it_finishes(monkeypatch):
    policies, rollouts, replica, _, r2r, complete = setup(monkeypatch, 3)
    policies.config.validation.val_before_train = True
    policies.prepare_validation_round(0, 2, [replica])
    policies.on_rollout_is_end(rollouts)
    assert policies.job_phase == JobPhase.DRAINING
    assert policies.rollout_buffer.qsize() == 3
    assert policies.current_step == 0
    complete.assert_not_called()
    finish_round(policies, rollouts)
    assert policies.current_step == 1
    assert policies.dispatched_rollouts_by_step == {1: 2}
    policies.train_ack("policy-0", 1, 2, False, {}, rollouts)
    assert r2r.call_args.kwargs["weight_step"] == 1
    finish_round(policies, rollouts)
    complete.assert_called_once()


def test_periodic_validation_sync_survives_exhaustion_and_is_not_repeated(monkeypatch):
    policies, rollouts, _, p2r, r2r, complete = setup(monkeypatch, 3, freq=1)
    policies.on_rollout_is_end(rollouts)
    assert policies.current_step == 1
    assert policies.data_fetcher.activated_val_iter is not None
    assert policies.validation_round is None
    policies.train_ack("policy-0", 1, 2, False, {}, rollouts)
    p2r.assert_called_once()
    r2r.assert_called_once()
    complete.assert_not_called()
    finish_round(policies, rollouts)
    complete.assert_called_once()
    assert r2r.call_count == 1


def test_zero_update_reuses_completed_initial_validation(monkeypatch):
    policies, rollouts, replica, p2r, r2r, complete = setup(monkeypatch, 1)
    policies.config.validation.val_before_train = True
    policies.prepare_validation_round(0, 2, [replica])
    finish_round(policies, rollouts)
    policies.on_rollout_is_end(rollouts)
    complete.assert_called_once()
    assert complete.call_args.kwargs["final_step"] == 0
    p2r.assert_not_called()
    r2r.assert_not_called()


@pytest.mark.parametrize("accepted", [4, 5, 6])
@pytest.mark.parametrize("frequency", [1, 10])
def test_nominal_horizon_never_sends_a_synthetic_extra_step(
    monkeypatch, accepted, frequency
):
    policies, rollouts, _, _, r2r, complete = setup(
        monkeypatch, accepted, freq=frequency
    )
    policies.on_rollout_is_end(rollouts)
    for step in (1, 2):
        assert policies.current_step == step
        assert policies.dispatched_rollouts_by_step == {step: 2}
        policies.train_ack("policy-0", step, 2, False, {}, rollouts)
        if (
            policies.validation_round is not None
            and policies.validation_round.step == step
        ):
            assert not policies.terminal_complete
            finish_round(policies, rollouts)
    assert policies.terminal_complete and policies.validation_drained
    assert (
        policies.current_step
        == policies.total_steps
        == policies.training_horizon()
        == 2
    )
    assert policies.samples_on_the_fly == 0
    assert policies.rollout_buffer.empty()
    assert r2r.call_count == (2 if frequency == 1 else 1)
    complete.assert_not_called()


def test_missing_training_ack_cannot_finish_validation_drain(monkeypatch):
    policies, rollouts, _, p2r, r2r, complete = setup(monkeypatch, 2)
    policies.on_rollout_is_end(rollouts)
    # Even misleading survivor status must not replace the issued ACK set.
    policies.all_ready_or_reduced = lambda: True
    policies.finish_draining_phase(rollouts)
    assert policies.dispatched_rollouts_by_step == {1: 2}
    assert not policies.terminal_complete and not policies.validation_drained
    p2r.assert_not_called()
    r2r.assert_not_called()
    complete.assert_not_called()


@pytest.mark.parametrize("mode", ["sync", "colocated", "weight-thread"])
@pytest.mark.parametrize("step", [0, 3])
def test_controller_requested_final_round_ignores_periodic_frequency(
    monkeypatch, mode, step
):
    from queue import Queue
    from cosmos_rl.rollout import State
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )
    from cosmos_rl.rollout.worker.colocated.rollout_control import (
        ColocatedRolloutControlWorker,
    )
    from cosmos_rl.rollout.worker import weight_sync

    worker = SimpleNamespace(
        replica_name="a",
        current_weight_version=0,
        state=State(),
        validation_flag=threading.Event(),
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        do_validation=Mock(),
        _buffer_version=1,
        config=SimpleNamespace(
            validation=SimpleNamespace(enable=True, val_before_train=False, freq=10),
            rollout=SimpleNamespace(
                async_r2r_sync="disabled", broadcast_all_params=False
            ),
        ),
    )
    message = command.RolloutToRolloutBroadcastCommand(
        "a",
        ["a"],
        step,
        20,
        False,
        validation_round_id="final-round",
        validation_protocol_version=1,
    )
    if mode == "weight-thread":
        thread = object.__new__(weight_sync.WeightSyncThread)
        thread._worker, thread._stream = worker, object()
        thread._executed, thread._queue = 0, Queue()
        monkeypatch.setattr(weight_sync.torch.cuda, "Event", Mock())
        thread._execute_r2r(message)
        assert worker._pending_validation_step == step
    else:
        cls = (
            ColocatedRolloutControlWorker
            if mode == "colocated"
            else DisaggregatedRolloutControlWorker
        )
        cls.broadcast_to_all_rollout_replica(worker, message)
    assert worker.current_step == step
    assert worker.validation_round_id == "final-round"
    assert worker.validation_flag.is_set()
    assert not worker.shutdown_signal.is_set()
    assert (
        not message.replica_should_stop()
    )  # Original checkpoint horizon, no fake final step.
    assert worker.do_validation.call_count == int(mode == "colocated")


@pytest.mark.parametrize("phase", ["validation", "checkpoint", "complete"])
@pytest.mark.parametrize("policy_exited", [False, True])
def test_monitor_stop_requires_validation_completion_ack_and_policy_exit(
    monkeypatch, phase, policy_exited
):
    import asyncio
    from cosmos_rl.dispatcher import run_web_panel

    policies, rollouts, replica, _, _, _ = setup(monkeypatch, 0)
    policies.on_rollout_is_end(rollouts)
    if phase != "validation":
        finish_round(policies, rollouts)
    if phase == "complete":
        step = policies.completion_step
        policies.train_ack("policy-0", step, step, False, {}, rollouts)
    if policy_exited:
        policies.policy_replicas.clear()
    policies.redis_handler._publication_failure = None
    rollouts.rollout_replicas = {"a": replica}
    rollouts.redis_handler = Mock()
    rollouts.maintain_life_status = Mock()
    stop = Mock()
    monkeypatch.setattr(command.StopCommand, "trigger", stop)
    monkeypatch.setattr(
        run_web_panel,
        "controller",
        SimpleNamespace(
            policy_status_manager=policies,
            rollout_status_manager=rollouts,
            config=policies.config,
            life_cycle_lock=asyncio.Lock(),
        ),
    )
    monkeypatch.setattr(run_web_panel, "_policy_replicas_were_registered", True)
    monkeypatch.setattr(run_web_panel, "_maybe_finalize", lambda reason: False)
    monkeypatch.setattr(run_web_panel, "COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS", False)
    monkeypatch.setattr(run_web_panel.os, "_exit", Mock())

    async def scenario():
        observed = asyncio.Event()
        policies.maintain_life_status = observed.set
        async with run_web_panel.lifespan(run_web_panel.app):
            await asyncio.wait_for(observed.wait(), timeout=2)

    asyncio.run(scenario())
    assert stop.call_count == int(phase == "complete" and policy_exited)
