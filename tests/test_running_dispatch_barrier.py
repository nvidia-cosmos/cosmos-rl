# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sealed receipts, not live status flags, authorize the next optimizer step."""

from copy import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.status import PolicyStatus, PolicyStatusManager
import test_terminal_drain_protocol as fixture


def manager(horizon=2):
    status, first = fixture.TestTerminalMatrix._manager(8)
    status.total_steps = horizon
    second = copy(first)
    second.name = "policy-1"
    first.in_mesh = second.in_mesh = True
    status.policy_replicas[second.name] = second
    status.status[second.name] = PolicyStatus.READY
    status.get_all_atoms_arrived_replicas = lambda: list(
        status.policy_replicas.values()
    )
    status.config.train.train_policy.type = "grpo"
    status.config.train.train_policy.on_policy = False
    status.should_weight_sync_after_train_ack = Mock(return_value=False)
    return status, first, second


@pytest.mark.parametrize("mode", ["disaggregated", "colocated", "colocated_separated"])
def test_no_next_command_while_a_sealed_dispatch_is_missing_any_ack(mode):
    status, first, second = manager()
    status.config.mode = mode
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1
    status.train_ack(first.name, 1, 2, False, {}, Mock())
    # A transient status update cannot certify the missing receipt.
    status.status[second.name] = PolicyStatus.READY
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1
    assert set(status.dispatched_rollouts_by_step) == {1}
    assert status.redis_handler.publish_plan.call_count == 1
    status.train_ack(second.name, 1, 2, False, {}, Mock())
    assert status.current_step == 2
    assert set(status.dispatched_rollouts_by_step) == {2}


def test_participant_departure_seals_failure_before_rebuilding_active_mesh():
    status, first, second = manager()
    status.try_trigger_data_fetch_and_training()
    status.train_ack(first.name, 1, 2, False, {}, Mock())
    status.trigger_rebuild_mesh = Mock()
    with pytest.raises(
        RuntimeError, match="membership changed during an unsettled update"
    ):
        status.unregister(second.name)
    status.trigger_rebuild_mesh.assert_not_called()
    assert status.training_dispatches[1].participants == {first.name, second.name}
    assert not status.training_dispatches[1].settled
    assert not status.training_finished()
    with pytest.raises(RuntimeError, match="unsettled update"):
        status.try_trigger_data_fetch_and_training()
    with pytest.raises(RuntimeError, match="unsettled update"):
        status.train_ack(first.name, 1, 2, False, {}, Mock())
    assert status.current_step == 1


def test_mesh_rebuild_is_rejected_before_any_command_publication():
    status, _, _ = manager()
    status.try_trigger_data_fetch_and_training()
    with pytest.raises(RuntimeError, match="unsettled update"):
        status.trigger_rebuild_mesh(status.get_all_atoms_arrived_replicas())
    status.redis_handler.publish_command.assert_not_called()
    assert not status.training_finished()


def test_new_participant_cannot_join_an_unsettled_update():
    status, _, _ = manager()
    status.try_trigger_data_fetch_and_training()
    with pytest.raises(RuntimeError, match="Cannot join or replace"):
        status.register(SimpleNamespace(replica_name="late"), status.config, Mock())
    assert "late" not in status.policy_replicas
    assert status.terminal_error is None


def test_last_dispatch_is_not_success_until_the_entire_original_cohort_acks():
    status, first, second = manager(horizon=1)
    status.try_trigger_data_fetch_and_training()
    assert not status.training_finished()
    status.train_ack(first.name, 1, 1, False, {}, Mock())
    assert not status.training_finished()
    status.trigger_rebuild_mesh = Mock()
    status.unregister(first.name)
    assert status.terminal_error is None
    status.trigger_rebuild_mesh.assert_not_called()
    status.train_ack(second.name, 1, 1, False, {}, Mock())
    assert status.training_finished()
    status.unregister(second.name)
    assert status.terminal_error is None


def test_terminal_accounting_error_cannot_be_reported_successful():
    status = PolicyStatusManager()
    status.current_step = status.total_steps = 2
    status.terminal_complete = True
    status.terminal_error = RuntimeError("uncertain execution")
    assert not status.training_finished()


def test_terminal_flag_cannot_override_an_outstanding_original_ack_set():
    status, _, _ = manager(horizon=1)
    status.try_trigger_data_fetch_and_training()
    status.terminal_complete = True
    assert not status.training_finished()
