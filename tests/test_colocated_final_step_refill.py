# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Final dispatch must not close prompts needed by colocated batch preparation."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from cosmos_rl.colocated.rl_worker import ColocatedRLControlWorker
from cosmos_rl.dispatcher.command import DataFetchCommand
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import RLPayload
from test_running_dispatch_barrier import manager


def final_dispatch():
    status, first, second = manager(horizon=30)
    status.config.mode = "colocated"
    status.config.rollout = NS(n_generation=8)
    status.config.train.train_batch_per_replica = 16
    status.config.train.train_policy.variant = "grpo"
    status.current_step = 29
    status.samples_on_the_fly = 32
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 30
    assert status.dispatched_rollouts_by_step == {30: 32}
    controller = object.__new__(Controller)
    controller.config = status.config
    controller.policy_status_manager = status
    controller.rollout_status_manager = NS(replica_scaling_log={})
    controller.data_fetcher = NS(
        get_batched_prompt=Mock(return_value=([RLPayload(prompt_idx=0)], False))
    )
    return status, controller, first, second


def prompt_end(controller, *, validation=False):
    _, end = asyncio.run(
        controller._get_batched_prompt_impl(
            1, validation_step=30 if validation else None
        )
    )
    return end


def run_final_batch(controller, contributions, prompts_per_call=1):
    state = NS(pending=0, calls=0, training=[], shutdown=False)
    groups = iter(contributions)

    def generate():
        state.calls += 1
        assert state.calls <= len(contributions), "refill exceeded supplied prompts"
        end = prompt_end(controller)
        state.pending += next(groups)
        return end, prompts_per_call

    def consume(command, **kwargs):
        if command is DataFetchCommand:
            state.training.append(not kwargs.get("no_exec", False))

    def finish():
        state.shutdown = True
        return False

    local = NS(
        init_commands=Mock(),
        prepare_iteration=lambda: True,
        advance_iteration=Mock(),
        pending_policy_samples_all_replicas=lambda: state.pending,
        pending_policy_samples=lambda: state.pending,
        synchronize_rollouts=Mock(),
        agree_prepared_batch=lambda ready: ready,
        rollout_completed_for_data_fetch_n_training=Mock(),
        training_end_ack=Mock(),
        rollout_consume_one_step_commands_util_r2r=finish,
    )
    worker = NS(
        config=controller.config,
        controller=local,
        policy=NS(consume_command=consume),
        rollout=NS(
            consume_command=Mock(),
            parallel_dims=NS(mesh={"dp": NS(size=lambda: 1)}),
            rollout_for_one_minor_step=generate,
            report_rollouts=Mock(),
            shutdown_signal=NS(is_set=lambda: state.shutdown),
        ),
    )
    ColocatedRLControlWorker.main_loop(worker)
    return state


@pytest.mark.parametrize(
    "groups,prompts_per_call",
    [([8, 8], 1), ([8, 0, 8], 1), ([8, 8], 2), ([0, 0, 16], 2)],
)
def test_final_batch_refills_before_training(groups, prompts_per_call):
    status, controller, _, _ = final_dispatch()
    result = run_final_batch(controller, groups, prompts_per_call)
    assert result.training == [True]
    assert result.pending == 16
    assert result.calls == len(groups)
    assert not status.training_finished(), (
        "local preparation is not a completed ACK set"
    )


def test_final_prompt_stream_stays_open_until_every_original_ack():
    status, controller, first, second = final_dispatch()
    assert not prompt_end(controller)
    for replica in (first, second):
        status.training_dispatches[30].prepare(replica.name, 30, 30, True)
    status.train_ack(first.name, 30, 30, False, {}, Mock())
    assert not prompt_end(controller)
    status.train_ack(second.name, 30, 30, False, {}, Mock())
    assert prompt_end(controller)


@pytest.mark.parametrize(
    "validation_run,validation_fetch", [(True, False), (True, True), (False, True)]
)
def test_synthetic_training_end_does_not_close_validation(
    validation_run, validation_fetch
):
    status, controller, _, _ = final_dispatch()
    status.dispatched_rollouts_by_step.clear()
    status.config.validation.enable = validation_run
    assert status.training_finished()
    assert not prompt_end(controller, validation=validation_fetch)


def test_real_dataset_exhaustion_is_preserved_not_claimed_recovered():
    status, controller, _, _ = final_dispatch()
    controller.data_fetcher.get_batched_prompt.return_value = ([], True)
    assert prompt_end(controller)
    assert not status.training_finished()


def test_disaggregated_shutdown_still_closes_after_final_ack_set():
    status, controller, first, second = final_dispatch()
    status.config.mode = "disaggregated"
    policy = status.config.train.train_policy
    policy.allowed_outdated_steps = 100
    policy.max_inflight_steps = None
    policy.max_retry_for_on_policy = 0
    controller._assign_prompt_weight_versions = Mock()
    controller._soft_throttle_engaged_since = None
    controller._soft_throttle_last_log_ts = 0
    assert not prompt_end(controller)
    for replica in (first, second):
        status.training_dispatches[30].prepare(replica.name, 30, 30, True)
    status.train_ack(first.name, 30, 30, False, {}, Mock())
    status.train_ack(second.name, 30, 30, False, {}, Mock())
    assert prompt_end(controller)


def test_previous_dispatched_equals_finished_predicate_reproduces_skip(monkeypatch):
    status, controller, _, _ = final_dispatch()
    monkeypatch.setattr(
        status, "training_finished", lambda: status.current_step >= status.total_steps
    )
    result = run_final_batch(controller, [8, 0, 8])
    assert (result.calls, result.pending, result.training) == (1, 8, [])
