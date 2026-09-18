# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.dispatcher.command import TrainingCompleteCommand
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.status import PolicyStatus, PolicyStatusManager
from cosmos_rl.policy.config import Config


@pytest.fixture
def manager():
    m = PolicyStatusManager()
    m.config = Config()
    m.config.mode = "disaggregated"
    m.config.train.ckpt.enable_checkpoint = True
    m.current_step = 0
    m.total_steps = 100
    m.remain_samples_num = 200
    m.policy_init_done = True
    m.data_fetcher = SimpleNamespace(activated_val_iter=None)
    m.redis_handler = object()
    m.policy_replicas = {name: SimpleNamespace(name=name) for name in ("p0", "p1")}
    m.status = {name: PolicyStatus.READY for name in m.policy_replicas}
    m.get_all_atoms_arrived_replicas = lambda: list(m.policy_replicas.values())
    m.cleanup_buffered_rollouts = Mock()
    return m


@pytest.mark.parametrize("step", [0, 5])
def test_stop_uses_real_step_and_preserves_horizon(manager, step):
    manager.current_step = step
    with patch.object(TrainingCompleteCommand, "trigger") as publish:
        assert manager.request_stop("application budget reached")
        assert not manager.request_stop("different reason")
        manager.try_trigger_data_fetch_and_training()
    assert manager.current_step == step
    assert manager.total_steps == manager.training_horizon() == 100
    assert manager.stop_reason == "application budget reached"
    assert manager.rollout_admission_closed()
    assert publish.call_count == 2
    for call in publish.call_args_list:
        assert call.kwargs["final_step"] == step
        assert call.kwargs["checkpoint_total_steps"] == 100
        assert call.kwargs["do_save"]
    assert not manager.terminal_complete
    manager.record_completion_ack("p0", step + 1)
    manager.record_completion_ack("p0", step + 1)
    assert not manager.terminal_complete
    manager.record_completion_ack("p1", step + 1)
    assert manager.terminal_complete


def test_waits_for_issued_update_and_validation(manager):
    manager.current_step = 1
    manager.dispatched_rollouts_by_step = {1: 4}
    manager.status = {name: PolicyStatus.RUNNING for name in manager.policy_replicas}
    manager.data_fetcher.activated_val_iter = object()
    with patch.object(TrainingCompleteCommand, "trigger") as publish:
        manager.request_stop("quality gate")
        publish.assert_not_called()
        # Even normalized status does not suffice before complete ACK accounting.
        manager.status = {name: PolicyStatus.READY for name in manager.policy_replicas}
        manager.try_trigger_data_fetch_and_training()
        publish.assert_not_called()
        manager.dispatched_rollouts_by_step.clear()
        manager.try_trigger_data_fetch_and_training()
        publish.assert_not_called()
        manager.data_fetcher.activated_val_iter = None
        manager.try_trigger_data_fetch_and_training()
        assert publish.call_count == 2
        assert publish.call_args.kwargs["final_step"] == 1


def test_stop_after_natural_completion_does_not_send_another_command(manager):
    manager.terminal_complete = True
    with patch.object(TrainingCompleteCommand, "trigger") as publish:
        assert not manager.request_stop("late stop")
        publish.assert_not_called()
    assert manager.stop_reason is None


def test_membership_change_cannot_be_certified_as_success(manager):
    manager.dispatched_rollouts_by_step = {1: 4}
    manager.request_stop("stop")
    manager.policy_replicas.pop("p1")
    with pytest.raises(RuntimeError, match="membership changed"):
        manager.try_trigger_data_fetch_and_training()
    assert not manager.terminal_complete


@pytest.mark.parametrize("reason", [None, "", "  ", 3])
def test_invalid_reason_does_not_close_admission(manager, reason):
    with pytest.raises(ValueError):
        manager.request_stop(reason)
    assert not manager.rollout_admission_closed()


def test_uninitialized_stop_is_rejected_without_mutation(manager):
    manager.policy_init_done = False
    with pytest.raises(RuntimeError, match="initialized"):
        manager.request_stop("stop")
    assert not manager.rollout_admission_closed()


def test_controller_serializes_request_and_closes_training_prompts(manager):
    controller = object.__new__(Controller)
    controller.policy_status_manager = manager
    controller.life_cycle_lock = asyncio.Lock()
    with patch.object(TrainingCompleteCommand, "trigger"):
        assert asyncio.run(controller.request_stop("stop"))
    assert asyncio.run(controller._get_batched_prompt_impl(4)) == ([], True)


def test_zero_horizon_stop_still_requests_step_zero_checkpoint(manager):
    manager.total_steps = 0
    with patch.object(TrainingCompleteCommand, "trigger") as publish:
        manager.request_stop("empty input")
    assert publish.call_args.kwargs["final_step"] == 0
    assert publish.call_args.kwargs["do_save"]


def test_real_ack_path_finishes_issued_update_before_stop(manager):
    manager.current_step = 1
    manager.dispatched_rollouts_by_step = {1: 4}
    manager.samples_on_the_fly = 4
    manager.status = {name: PolicyStatus.RUNNING for name in manager.policy_replicas}
    for replica in manager.policy_replicas.values():
        replica.start_time = 0
    rollout_status = Mock()
    with patch.object(TrainingCompleteCommand, "trigger") as publish:
        manager.request_stop("stop during update")
        manager.train_ack("p0", 1, 100, False, {}, rollout_status)
        publish.assert_not_called()
        manager.train_ack("p1", 1, 100, False, {}, rollout_status)
        assert publish.call_count == 2
        assert publish.call_args.kwargs["final_step"] == 1
    assert manager.samples_on_the_fly == 0
    assert manager.current_step == 1
