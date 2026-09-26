# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.command import Command, DataFetchCommand
from cosmos_rl.dispatcher.dispatch import TrainingDispatch
from cosmos_rl.dispatcher.status import PolicyStatus
import test_terminal_drain_protocol as _terminal
from test_sft_ack_progress import manager as sft_manager


def test_ack_set_is_sealed_and_retries_are_idempotent():
    dispatch = TrainingDispatch(3, 10, frozenset({"a", "b"}), 8)
    report = {"loss": 1.0}
    assert dispatch.acknowledge("a", 3, 10, report)
    report["loss"] = 7.0
    assert dispatch.reports["a"] == {"loss": 1.0}
    assert not dispatch.acknowledge("a", 3, 10, {"loss": 1.0})
    with pytest.raises(ValueError, match="changed"):
        dispatch.acknowledge("a", 3, 10, report)
    assert not dispatch.complete
    assert dispatch.acknowledge("b", 3, 10, {})
    assert dispatch.complete
    assert dispatch.settle() == [{"loss": 1.0}, {}]
    assert dispatch.reports == {}
    assert not dispatch.acknowledge("a", 3, 10, {"loss": 1.0})
    with pytest.raises(ValueError, match="settled"):
        dispatch.settle()


@pytest.mark.parametrize(
    "replica,step,total", [("late", 3, 10), ("a", 2, 10), ("a", 3, 9)]
)
def test_invalid_ack_cannot_change_ledger(replica, step, total):
    dispatch = TrainingDispatch(3, 10, frozenset({"a", "b"}), 8)
    with pytest.raises(ValueError):
        dispatch.acknowledge(replica, step, total, {})
    assert dispatch.reports == dispatch.report_digests == {}


def dispatched_manager():
    manager, replica = _terminal.TestTerminalMatrix._manager(2)
    manager.config.train.train_policy.type = "grpo"
    manager.config.train.train_policy.on_policy = False
    manager.should_weight_sync_after_train_ack = Mock(return_value=False)
    manager.try_trigger_data_fetch_and_training()
    manager.try_trigger_data_fetch_and_training = Mock()
    return manager, replica


def test_dispatch_publishes_payloads_and_command_as_one_plan():
    manager, _ = dispatched_manager()
    manager.redis_handler.publish_plan.assert_called_once()
    plan = manager.redis_handler.publish_plan.call_args.args[0]
    assert [entry[:2] for entry in plan.entries] == [
        ("policy-0_rollout", "rollout"),
        ("policy-0_rollout", "rollout"),
        ("policy-0_command", "command"),
    ]
    command = Command.depack(plan.entries[-1][2])
    assert isinstance(command, DataFetchCommand)
    assert command.global_step == 1 and command.items_count == 2
    assert manager.training_dispatches[1].participants == {"policy-0"}


def test_partial_dispatch_build_failure_is_terminal_not_retriable():
    manager, _ = _terminal.TestTerminalMatrix._manager(2)
    manager.redis_handler.publish_plan.side_effect = RuntimeError(
        "injected publication failure"
    )
    with pytest.raises(RuntimeError, match="injected") as failure:
        manager.try_trigger_data_fetch_and_training()
    assert manager.terminal_error is failure.value


def test_late_join_does_not_block_previous_ack_or_repeat_accounting():
    manager, original = dispatched_manager()
    late = SimpleNamespace(name="late", start_time=1)
    manager.policy_replicas[late.name] = late
    manager.status[late.name] = PolicyStatus.READY
    manager.get_all_atoms_arrived_replicas = lambda: [original, late]
    manager.train_ack("policy-0", 1, 2, False, {}, Mock())
    assert manager.samples_on_the_fly == 0
    assert manager.training_dispatches[1].settled
    assert manager.training_dispatches[1].reports == {}
    manager.train_ack("policy-0", 1, 2, False, {}, Mock())
    assert manager.samples_on_the_fly == 0
    manager.try_trigger_data_fetch_and_training.assert_called_once()


@pytest.mark.parametrize(
    "step,total,replica", [(0, 2, "policy-0"), (1, 1, "policy-0"), (1, 2, "late")]
)
def test_invalid_controller_ack_leaves_counters_and_status_unchanged(
    step, total, replica
):
    manager, _ = dispatched_manager()
    manager.policy_replicas["late"] = SimpleNamespace(name="late")
    manager.status["late"] = PolicyStatus.READY
    before = dict(manager.status)
    with pytest.raises(ValueError):
        manager.train_ack(replica, step, total, False, {}, Mock())
    assert manager.samples_on_the_fly == 2
    assert manager.status == before
    assert not manager.training_dispatches[1].report_digests


@pytest.mark.parametrize("logging", [[], ["console"]])
def test_bad_or_disabled_logging_does_not_retain_reports(logging):
    manager, _ = dispatched_manager()
    manager.config.logging.logger = logging
    manager.filter_records = {"accepted": 2}
    manager.train_ack("policy-0", 1, 2, False, {"deliberately": "incomplete"}, Mock())
    assert manager.samples_on_the_fly == 0
    assert manager.training_dispatches[1].reports == {}
    assert manager.report_data_list == [] and manager.filter_records == {}
    manager.try_trigger_data_fetch_and_training.assert_called_once()


def test_sft_interleaved_ack_groups_settle_once_and_keep_reports_separate():
    manager = sft_manager()
    manager.sft_train_ack("a", {"loss": 3}, 3, 10)
    manager.sft_train_ack("a", {"val/avg_loss": 0.4}, 3, 10)
    manager.sft_train_ack("a", {"loss": 4}, 4, 10)
    manager.sft_train_ack("b", {"loss": 3}, 3, 10)
    assert manager.remain_samples_num == 92
    manager.sft_train_ack("b", {"loss": 3}, 3, 10)
    assert manager.remain_samples_num == 92
    manager.sft_train_ack("b", {"val/avg_loss": 0.6}, 3, 10)
    manager.sft_train_ack("b", {"loss": 4}, 4, 10)
    assert manager.remain_samples_num == 84
    assert manager.current_step == 4
    calls = manager.sft_report_summary.call_args_list
    assert len(calls) == 3
    assert [c.kwargs["is_validation"] for c in calls] == [False, True, False]
    assert calls[0].kwargs["reports"] == [{"loss": 3}, {"loss": 3}]
    assert calls[1].kwargs["reports"] == [{"val/avg_loss": 0.4}, {"val/avg_loss": 0.6}]
