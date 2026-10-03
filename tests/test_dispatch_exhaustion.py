# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.dispatcher.command import TrainingCompleteCommand
from cosmos_rl.dispatcher.status import JobPhase, PolicyStatus, RolloutStatusManager
import test_terminal_drain_protocol as _terminal
from test_terminal_drain_protocol import _registered_rollout


@pytest.mark.parametrize("tail_count", [0, 1, 2, 3])
def test_last_generating_departure_drains_accepted_tail(tail_count):
    policies, _ = _terminal.TestTerminalMatrix._manager(tail_count)
    rollouts = RolloutStatusManager()
    survivor = _registered_rollout("ended", ended=True)
    rollouts.rollout_replicas = {
        "departing": _registered_rollout("departing"),
        survivor.name: survivor,
    }
    rollouts._command_participant_ended_replicas = {survivor.name}
    rebuild = Mock()
    rollouts.trigger_rebuild_mesh = rebuild

    with patch.object(TrainingCompleteCommand, "trigger") as complete:
        rollouts.unregister("departing", policies)

    rebuild.assert_called_once_with([survivor])
    assert policies.job_phase == JobPhase.DRAINING
    assert policies.training_horizon() == policies.total_steps == 2
    if tail_count < 2:
        complete.assert_called_once()
        assert complete.call_args.kwargs["final_step"] == 0
        assert complete.call_args.kwargs["checkpoint_total_steps"] == 2
        assert policies.rollout_buffer.empty()
        assert policies.samples_on_the_fly == 0
    else:
        complete.assert_not_called()
        assert policies.current_step == 1
        assert policies.status["policy-0"] == PolicyStatus.RUNNING
        assert policies.dispatched_rollouts_by_step == {1: 2}
        assert policies.samples_on_the_fly == tail_count


def test_departure_drain_does_not_complete_an_unacknowledged_update():
    policies, _ = _terminal.TestTerminalMatrix._manager(2)
    policies.try_trigger_data_fetch_and_training()
    rollouts = RolloutStatusManager()
    rollouts.rollout_replicas = {
        "departing": _registered_rollout("departing"),
        "ended": _registered_rollout("ended", ended=True),
    }
    rollouts._command_participant_ended_replicas = {"ended"}
    rollouts.trigger_rebuild_mesh = Mock()

    with patch.object(TrainingCompleteCommand, "trigger") as complete:
        rollouts.unregister("departing", policies)

    assert policies.job_phase == JobPhase.DRAINING
    complete.assert_not_called()
    assert policies.dispatched_rollouts_by_step == {1: 2}
    assert not policies.training_dispatches[1].settled
    assert policies.samples_on_the_fly == 2


def test_departure_does_not_drain_while_another_rollout_is_generating():
    policies, _ = _terminal.TestTerminalMatrix._manager(0)
    rollouts = RolloutStatusManager()
    rollouts.rollout_replicas = {
        "departing": _registered_rollout("departing"),
        "generating": _registered_rollout("generating"),
    }
    rollouts.trigger_rebuild_mesh = Mock()

    with patch.object(TrainingCompleteCommand, "trigger") as complete:
        rollouts.unregister("departing", policies)

    assert policies.job_phase == JobPhase.RUNNING
    complete.assert_not_called()


def test_drain_cannot_replace_a_departed_participants_missing_ack():
    policies, first = _terminal.TestTerminalMatrix._manager(4)
    second = copy(first)
    second.name = "policy-1"
    second.in_mesh = False
    policies.policy_replicas[second.name] = second
    policies.status[second.name] = PolicyStatus.READY
    policies.get_all_atoms_arrived_replicas = lambda: list(
        policies.policy_replicas.values()
    )
    policies.config.train.train_policy.type = "grpo"
    policies.try_trigger_data_fetch_and_training()
    rollouts = RolloutStatusManager()
    rollouts.rollout_replicas = {
        "departing": _registered_rollout("departing"),
        "ended": _registered_rollout("ended", ended=True),
    }
    rollouts.trigger_rebuild_mesh = Mock()
    policies.train_ack("policy-0", 1, 2, False, {}, rollouts)
    # Conservative containment now rejects the membership loss immediately,
    # before a later rollout departure could try to synthesize completion.
    with pytest.raises(RuntimeError, match="unsettled update"):
        policies.unregister("policy-1")
    assert policies.all_ready_or_reduced()

    with patch.object(TrainingCompleteCommand, "trigger") as complete:
        rollouts.unregister("departing", policies)
        policies.finish_draining_phase(rollouts)

    complete.assert_not_called()
    assert not policies.terminal_complete
    assert not policies.training_finished()
    assert policies.completion_step is None
    assert policies.dispatched_rollouts_by_step == {1: 4}
    assert not policies.training_dispatches[1].settled
    assert policies.samples_on_the_fly == 4
