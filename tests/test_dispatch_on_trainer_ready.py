# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Training steps dispatched on trainer readiness (``dispatch_on_trainer_ready``)."""

from unittest.mock import MagicMock

import msgpack
import pytest
from test_dispatch_balance import KEY, _manager, _per_rank, _replica, _streams, _weight

from cosmos_rl.dispatcher.command import Command
from cosmos_rl.dispatcher.status import PolicyStatus
from cosmos_rl.policy.config import GrpoConfig


def _ready_manager(weights, replicas, *, key=None):
    manager = _manager(weights, replicas, count=2, key=key)
    manager.config.train.train_policy.dispatch_on_trainer_ready = True
    return manager


def _published(manager):
    """Return ``{replica: (items_count, [prompt_idx, ...])}`` of the last plan."""
    plan = manager.redis_handler.publish_plan.call_args.args[0]
    published = {}
    for name, field, data in plan.entries:
        if field == "command":
            replica = name.removesuffix("_command")
            items_count = Command.depack(data).items_count
            published.setdefault(replica, [None, []])[0] = items_count
    for name, field, data in plan.entries:
        if field == "rollout":
            replica = name.removesuffix("_rollout")
            published[replica][1].append(msgpack.unpackb(data)["prompt_idx"])
    return {replica: tuple(value) for replica, value in published.items()}


def test_step_takes_every_pending_rollout_with_uneven_counts():
    replicas = [_replica(f"policy-{i}", i, 1) for i in range(4)]
    manager = _ready_manager([1] * 5, replicas)

    manager.try_trigger_data_fetch_and_training()

    assert _published(manager) == {
        "policy-0": (2, [0, 4]),
        "policy-1": (1, [1]),
        "policy-2": (1, [2]),
        "policy-3": (1, [3]),
    }
    assert manager.rollout_buffer.empty()
    assert manager.dispatched_rollouts_by_step == {1: 5}
    assert manager.training_dispatches[1].rollout_count == 5
    assert manager.remain_samples_num == 95


def test_step_without_pending_rollouts_dispatches_zero_counts():
    replicas = [_replica("policy-0", 0, 1), _replica("policy-1", 1, 1)]
    manager = _ready_manager([], replicas)

    manager.try_trigger_data_fetch_and_training()

    assert _published(manager) == {"policy-0": (0, []), "policy-1": (0, [])}
    assert manager.dispatched_rollouts_by_step == {1: 0}
    assert manager.current_step == 1


def _complete_cohort(manager, replicas):
    """Run the registration hook for the replica that completes the policy cohort."""
    manager.policy_init_done = False
    manager.status = {replica.name: PolicyStatus.UNINITIALIZED for replica in replicas}
    manager.trigger_rebuild_mesh = MagicMock()
    manager.config.policy.parallelism.n_init_replicas = len(replicas)
    manager.post_register_hook(replicas, replicas[-1], manager.config, MagicMock())


def test_completed_cohort_starts_the_first_step_without_rollouts():
    """The first step must not wait for the first finished episode."""
    replicas = [_replica("policy-0", 0, 1)]
    manager = _ready_manager([], replicas)

    _complete_cohort(manager, replicas)

    assert _published(manager) == {"policy-0": (0, [])}
    assert manager.current_step == 1


def test_completed_cohort_waits_for_rollouts_under_fixed_dispatch():
    replicas = [_replica("policy-0", 0, 1)]
    manager = _manager([], replicas, count=2, key=None)

    _complete_cohort(manager, replicas)

    manager.redis_handler.publish_plan.assert_not_called()
    assert manager.current_step == 0


def test_counts_are_whole_rounds_per_data_rank_and_leave_the_rest_queued():
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 2)]
    manager = _ready_manager([1] * 3, replicas)

    assert manager._dispatch_counts(replicas) == [2, 0]

    manager.try_trigger_data_fetch_and_training()

    assert _published(manager) == {"policy-0": (2, [0, 1]), "policy-1": (0, [])}
    assert manager.rollout_buffer.qsize() == 1
    assert manager.dispatched_rollouts_by_step == {1: 2}


def test_mixed_data_parallel_sizes_get_even_rollouts_per_rank():
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 1)]
    manager = _ready_manager([1] * 6, replicas)

    assert manager._dispatch_counts(replicas) == [4, 2]


def test_balancing_applies_to_uneven_replica_counts():
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 1)]
    manager = _ready_manager([177, 177, 177, 46, 46, 46], replicas, key=KEY)

    counts = manager._dispatch_counts(replicas)
    rows = manager._take_dispatch_rows(replicas, counts)

    streams = _streams(rows, replicas)
    assert [len(stream) for stream in streams] == counts == [4, 2]
    for rank_rollouts in [*_per_rank(streams[0], 2), streams[1]]:
        assert sorted(_weight(r) for r in rank_rollouts) == [46, 177]


def test_fixed_dispatch_still_waits_for_a_full_step():
    replicas = [_replica("policy-0", 0, 1), _replica("policy-1", 1, 1)]
    manager = _manager([1] * 3, replicas, count=2, key=None)

    manager.try_trigger_data_fetch_and_training()

    manager.redis_handler.publish_plan.assert_not_called()
    assert manager.rollout_buffer.qsize() == 3


def test_final_step_without_rollouts_completes_training():
    replicas = [_replica("policy-0", 0, 1)]
    manager = _ready_manager([], replicas)
    manager.total_steps = 1

    manager.try_trigger_data_fetch_and_training()
    manager.train_ack("policy-0", 1, 1, False, {}, MagicMock())

    assert manager.terminal_complete
    assert manager.training_finished()


def test_config_rejects_trainer_ready_dispatch_with_fixed_queue_modes():
    assert GrpoConfig(
        type="grpo", dispatch_on_trainer_ready=True
    ).dispatch_on_trainer_ready
    for conflict in ({"on_policy": True}, {"data_dispatch_as_rank_in_mesh": True}):
        with pytest.raises(ValueError, match="dispatch_on_trainer_ready"):
            GrpoConfig(type="grpo", dispatch_on_trainer_ready=True, **conflict)
