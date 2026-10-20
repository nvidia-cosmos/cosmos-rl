# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Colocated resume and strict on-policy surplus retain real weight identity."""

from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.dispatcher.command import (
    DataFetchCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    TrainingCompleteCommand,
)


def controller(on_policy, step, versions):
    queue = Queue()
    for index, version in enumerate(versions):
        queue.put(NS(weight_version=version, index=index))
    return NS(
        config=NS(train=NS(train_policy=NS(on_policy=on_policy))),
        current_step=step,
        policy=NS(data_queue=queue),
        rollout=NS(
            current_weight_version=step,
            api_client=NS(post_queued_rollout_discard=Mock()),
        ),
        train_report_data={},
    )


@pytest.mark.parametrize("step", [0, 8])
def test_on_policy_retains_current_fifo_only(step):
    obj = controller(True, step, [step - 1, step, step - 1, step])
    ColocatedController.advance_iteration(obj)
    assert obj.current_step == step + 1
    assert [x.index for x in obj.policy.data_queue.queue] == [1, 3]


def test_off_policy_surplus_is_unchanged():
    obj = controller(False, 8, [7, 0, 9, 8])
    ColocatedController.advance_iteration(obj)
    assert [x.weight_version for x in obj.policy.data_queue.queue] == [7, 0, 9, 8]


def test_empty_nonowning_rank_advances_without_fabrication():
    obj = controller(True, 8, [])
    ColocatedController.advance_iteration(obj)
    assert obj.current_step == 9 and obj.policy.data_queue.empty()


@pytest.mark.parametrize("step", [0, 8])
def test_initial_weight_sync_uses_completed_step_and_restored_horizon(step):
    command = object.__new__(DataFetchCommand)
    command.global_step, command.total_steps = step + 1, 16
    obj = NS(
        current_step=0,
        total_steps=100,
        policy_consume_one_step_commands_util_data_fetch=Mock(return_value=command),
    )
    observed = []
    obj.rollout_consume_one_step_commands_util_r2r = lambda **kwargs: (
        observed.append((obj.current_step, obj.total_steps, kwargs)) or True
    )
    assert ColocatedController.init_commands(obj)
    assert observed == [(step, 16, {"initial": True})]
    assert obj.init_data_fetch_command is command


def test_terminal_resume_does_not_synthesize_weight_sync():
    command = object.__new__(TrainingCompleteCommand)
    obj = NS(
        policy_consume_one_step_commands_util_data_fetch=Mock(return_value=command),
        finish_requested_stop=Mock(),
        rollout_consume_one_step_commands_util_r2r=Mock(),
    )
    assert not ColocatedController.init_commands(obj)
    obj.finish_requested_stop.assert_called_once_with(command)
    obj.rollout_consume_one_step_commands_util_r2r.assert_not_called()


def test_resume_between_periodic_sync_steps_still_binds_weights():
    obj = object.__new__(ColocatedController)
    obj.current_step, obj.total_steps = 0, 100
    command = object.__new__(DataFetchCommand)
    command.global_step, command.total_steps = 9, 16
    broadcast = object.__new__(RolloutToRolloutBroadcastCommand)
    broadcast.validation_round_id = None
    broadcast.validation_protocol_version = 1
    obj.config = NS(train=NS(sync_weight_interval=3), validation=NS(enable=False))
    obj.policy = NS(world_size=1)
    obj.rollout = NS(world_size=1)
    obj.policy_replica, obj.rollout_replica = Mock(), Mock()
    obj.command_dispatcher = Mock()
    obj.policy_consume_one_step_commands_util_data_fetch = lambda: command
    obj.wait_for_remote_command = Mock(return_value=broadcast)
    with (
        patch.object(PolicyToRolloutUnicastCommand, "trigger") as p2r,
        patch.object(RolloutToRolloutBroadcastCommand, "trigger") as r2r,
    ):
        assert obj.init_commands()
    for publish in (p2r, r2r):
        publish.assert_called_once()
        assert publish.call_args.kwargs["weight_step"] == 8
        assert publish.call_args.kwargs["total_steps"] == 16


def test_stop_during_initial_weight_wait_does_not_admit_generation():
    command = object.__new__(DataFetchCommand)
    command.global_step, command.total_steps = 9, 16
    obj = NS(
        policy_consume_one_step_commands_util_data_fetch=Mock(return_value=command),
        rollout_consume_one_step_commands_util_r2r=Mock(return_value=False),
    )
    assert not ColocatedController.init_commands(obj)
    assert not hasattr(obj, "init_data_fetch_command")
