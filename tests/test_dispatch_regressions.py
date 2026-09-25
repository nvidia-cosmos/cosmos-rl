# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Entry-point probes also runnable against the previous implementation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.protocol import MESH_NAMES, Role
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.status import (
    PolicyStatus,
    PolicyStatusManager,
    RolloutStatusManager,
)
import test_terminal_drain_protocol as _terminal
from test_sft_ack_progress import manager as _sft_manager


def _dispatched():
    manager, original = _terminal.TestTerminalMatrix._manager(2)
    manager.config.train.train_policy.type = "grpo"
    manager.config.train.train_policy.on_policy = False
    manager.should_weight_sync_after_train_ack = Mock(return_value=False)
    manager.try_trigger_data_fetch_and_training()
    manager.try_trigger_data_fetch_and_training = Mock()
    return manager, original


def test_late_join_cannot_strand_a_dispatched_steps_reservations():
    manager, original = _dispatched()
    late = SimpleNamespace(name="late", start_time=1)
    manager.policy_replicas["late"] = late
    manager.status["late"] = PolicyStatus.READY
    manager.get_all_atoms_arrived_replicas = lambda: [original, late]
    manager.train_ack(original.name, 1, 2, False, {}, Mock())
    assert manager.samples_on_the_fly == 0
    assert 1 not in manager.dispatched_rollouts_by_step
    manager.try_trigger_data_fetch_and_training.assert_called_once()


@pytest.mark.parametrize("loggers", [[], ["console"]])
def test_missing_logging_metrics_do_not_retain_previous_reports(loggers):
    manager, original = _dispatched()
    manager.config.logging.logger = loggers
    manager.train_ack(original.name, 1, 2, False, {"batching/skipped": 1}, Mock())
    assert manager.report_data_list == []


def test_invalid_ack_is_rejected_before_accounting():
    manager, original = _dispatched()
    with pytest.raises(ValueError):
        manager.train_ack(original.name, 100, 2, False, {}, Mock())
    assert manager.samples_on_the_fly == 2
    assert manager.status[original.name] == PolicyStatus.RUNNING


def test_sft_interleaved_validation_cannot_lose_training_settlement():
    manager = _sft_manager()
    manager.sft_train_ack("a", {}, 3, 10)
    manager.sft_train_ack("a", {"val/avg_loss": 1}, 3, 10)
    manager.sft_train_ack("b", {}, 3, 10)
    assert manager.remain_samples_num == 92


@pytest.mark.parametrize("role", [Role.POLICY, Role.ROLLOUT])
def test_registration_retry_does_not_reenter_initialization(role):
    def atom():
        return Atom(
            global_rank=0,
            host_ip="127.0.0.1",
            host_name="host",
            trace_path="",
            ranks=[0] * len(MESH_NAMES),
            group_size=[1] * len(MESH_NAMES),
            replica_name="r",
        )

    replica = Replica("r", role, [atom()])
    manager = PolicyStatusManager() if role == Role.POLICY else RolloutStatusManager()
    if role == Role.POLICY:
        manager.policy_replicas["r"] = replica
        manager.status["r"] = PolicyStatus.RUNNING
    else:
        manager.rollout_replicas["r"] = replica
    manager.post_register_hook = Mock()
    assert manager.register(atom(), Mock(), Mock()) is replica
    manager.post_register_hook.assert_not_called()


def test_discard_underflow_does_not_mutate_or_spend_report_identity():
    manager = PolicyStatusManager()
    manager.samples_on_the_fly = 1
    with pytest.raises(ValueError):
        manager.settle_discarded_samples("r", "report", 2, 0)
    assert manager.samples_on_the_fly == 1
    assert not manager.filter_records
    assert not manager._applied_discard_report_ids["r"]
