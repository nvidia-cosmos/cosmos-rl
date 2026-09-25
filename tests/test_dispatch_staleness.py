# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Admission estimates cannot certify staleness at a later dispatch."""

from copy import copy
from queue import Queue
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.status import PolicyStatus
import test_terminal_drain_protocol as fixture


@pytest.mark.parametrize("allowed", [0, 1])
def test_shrinking_cohort_rechecks_admitted_surplus_at_actual_use(allowed):
    status, first = fixture.TestTerminalMatrix._manager(0)
    second = copy(first)
    second.name = "policy-1"
    status.policy_replicas[second.name] = second
    status.status[second.name] = PolicyStatus.READY
    status.get_all_atoms_arrived_replicas = lambda: list(
        status.policy_replicas.values()
    )
    status.config.train.train_policy.allowed_outdated_steps = allowed
    status.config.train.sync_weight_interval = 1
    status.samples_on_the_fly = 4
    admitted = status.filter_outdated_rollouts(
        [Rollout(prompt_idx=index, weight_version=0) for index in range(4)]
    )
    assert len(admitted) == 4  # All four fit the original first global update.
    for rollout in admitted:
        status.rollout_buffer.put(rollout)
    status.policy_replicas.pop(second.name)
    status.status.pop(second.name)
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1 and status.rollout_buffer.qsize() == 2
    # Complete the original issued update; the next dispatch uses the smaller
    # cohort. Only completion state is controlled; filtering/dispatch are real.
    status.dispatched_rollouts_by_step.clear()
    status.status[first.name] = PolicyStatus.READY
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == (1 if allowed == 0 else 2)


@pytest.mark.parametrize("allowed", [0, 1])
def test_rank_local_imbalance_rechecks_real_use_not_global_queue_estimate(allowed):
    status, first = fixture.TestTerminalMatrix._manager(0)
    second = copy(first)
    second.name = "policy-1"
    status.policy_replicas[second.name] = second
    status.status[second.name] = PolicyStatus.READY
    status.get_all_atoms_arrived_replicas = lambda: list(
        status.policy_replicas.values()
    )
    status.config.train.train_policy.allowed_outdated_steps = allowed
    status.config.train.train_policy.data_dispatch_as_rank_in_mesh = True
    status.config.train.sync_weight_interval = 1
    status.samples_on_the_fly = 8
    status.rollout_buffer_per_rank = [Queue(), Queue()]
    admitted = status.filter_outdated_rollouts(
        [Rollout(prompt_idx=0, weight_version=0) for _ in range(4)]
    )
    assert len(admitted) == 4
    for rollout in admitted:
        status.rollout_buffer_per_rank[0].put(rollout)
    for _ in range(2):
        status.rollout_buffer_per_rank[1].put(Rollout(prompt_idx=1, weight_version=0))
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1
    status.dispatched_rollouts_by_step.clear()
    for name in status.status:
        status.status[name] = PolicyStatus.READY
    for _ in range(2):
        status.rollout_buffer_per_rank[1].put(Rollout(prompt_idx=1, weight_version=1))
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == (1 if allowed == 0 else 2)


@pytest.mark.parametrize("rank_local", [False, True])
def test_drop_stale_without_issuing_partial_batch_or_repeating_accounting(rank_local):
    status, _ = fixture.TestTerminalMatrix._manager(0)
    status.current_step = 1
    status.config.train.train_policy.allowed_outdated_steps = 0
    status.config.train.train_policy.data_dispatch_as_rank_in_mesh = rank_local
    status._publish_payload_transport_cleanup = Mock()
    refill = Mock()
    status.set_discard_refill_hook(refill)
    queue = Queue()
    status.rollout_buffer_per_rank = [queue]
    status.rollout_buffer = queue
    old, fresh = (
        Rollout(prompt_idx=1, weight_version=0),
        Rollout(prompt_idx=2, weight_version=1),
    )
    queue.put(old)
    queue.put(fresh)
    status.samples_on_the_fly = 2
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1
    assert list(queue.queue) == [fresh]
    assert status.samples_on_the_fly == 1 and status.remain_samples_num == 19
    assert status.filter_records["outdated"] == 1
    refill.assert_not_called()
    assert not status.training_dispatches and not status.dispatched_rollouts_by_step
    status.redis_handler.publish_plan.assert_not_called()
    status._publish_payload_transport_cleanup.assert_called_once_with(
        [old, fresh], [fresh]
    )
    status.try_trigger_data_fetch_and_training()
    assert status.samples_on_the_fly == 1 and status.remain_samples_num == 19
    assert status._publish_payload_transport_cleanup.call_count == 1
    queue.put(Rollout(prompt_idx=3, weight_version=1))
    status.samples_on_the_fly += 1
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 2 and queue.empty()
    assert status.training_dispatches[2].rollout_count == 2
    assert status.samples_on_the_fly == 2 and status.remain_samples_num == 17


def test_revalidation_preserves_queue_order_and_protects_retained_payloads():
    status, _ = fixture.TestTerminalMatrix._manager(0)
    status.current_step = 1
    status.config.train.train_policy.allowed_outdated_steps = 0
    status._publish_payload_transport_cleanup = Mock()
    original = [
        Rollout(prompt_idx=i, weight_version=version)
        for i, version in enumerate((1, 0, 1, 0, 1))
    ]
    for rollout in original:
        status.rollout_buffer.put(rollout)
    status.samples_on_the_fly = 5
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 2
    assert list(status.rollout_buffer.queue) == [original[4]]
    assert status.samples_on_the_fly == 3
    assert status.remain_samples_num == 16
    status._publish_payload_transport_cleanup.assert_called_once_with(
        original, original[::2]
    )


def test_running_training_is_not_revalidated_before_its_ack_boundary():
    status, replica = fixture.TestTerminalMatrix._manager(2)
    status.current_step = 1
    status.config.train.train_policy.allowed_outdated_steps = 0
    status.status[replica.name] = PolicyStatus.RUNNING
    status._publish_payload_transport_cleanup = Mock()
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1 and status.rollout_buffer.qsize() == 2
    assert status.samples_on_the_fly == 2
    status._publish_payload_transport_cleanup.assert_not_called()


def test_colocated_local_generation_is_unchanged():
    status, _ = fixture.TestTerminalMatrix._manager(0)
    status.config.mode = "colocated"
    status.config.train.train_policy.allowed_outdated_steps = 0
    status._publish_payload_transport_cleanup = Mock()
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1
    status._publish_payload_transport_cleanup.assert_not_called()
