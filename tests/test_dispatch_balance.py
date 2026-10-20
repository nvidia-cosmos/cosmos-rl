# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Weight-balanced per-data-rank dispatch selected by ``dispatch_balance_key``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest

from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.protocol import MESH_NAMES, Role
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.status import PolicyStatus, PolicyStatusManager
from cosmos_rl.policy.config import GrpoConfig

KEY = "rows"
HEAVY_LIGHT = [177, 177, 177, 177, 46, 46, 46, 46]


def _replica(name, start_time, dp_size):
    return SimpleNamespace(
        name=name,
        start_time=start_time,
        data_rank_count=lambda: dp_size,
        atoms={
            f"rank{rank}": SimpleNamespace(global_rank=rank, report_session_id="s")
            for rank in range(dp_size)
        },
        sub_profiler_config=SimpleNamespace(
            do_profile=False,
            active_steps=None,
            rank_filter=None,
            record_shape=None,
            profile_memory=None,
            with_stack=None,
            with_modules=None,
        ),
    )


def _manager(weights, replicas, *, count, key=KEY, prefetch=False):
    manager = PolicyStatusManager()
    manager.total_steps = 10
    manager.remain_samples_num = 100
    manager.samples_on_the_fly = len(weights)
    manager.samples_per_epoch = 100
    for index, weight in enumerate(weights):
        extra_info = None if weight is None else {KEY: weight}
        manager.rollout_buffer.put(Rollout(prompt_idx=index, extra_info=extra_info))
    manager.policy_replicas = {replica.name: replica for replica in replicas}
    manager.status = {replica.name: PolicyStatus.READY for replica in replicas}
    manager.get_all_atoms_arrived_replicas = lambda: list(replicas)
    manager.data_fetcher = SimpleNamespace(activated_val_iter=None)
    manager.redis_handler = MagicMock()
    manager.config = SimpleNamespace(
        mode="disaggregated",
        validation=SimpleNamespace(enable=False, freq=1),
        logging=SimpleNamespace(logger=[]),
        distillation=SimpleNamespace(enable=False),
        policy=SimpleNamespace(
            parallelism=SimpleNamespace(tp_size=1, cp_size=1, pp_size=1)
        ),
        custom={"payload_transfer": "nccl"},
        train=SimpleNamespace(
            train_batch_per_replica=count,
            epoch=1,
            non_text=False,
            prefetch_payloads=prefetch,
            sync_weight_interval=4,
            ckpt=SimpleNamespace(
                enable_checkpoint=False, save_freq=1, save_freq_in_epoch=0
            ),
            train_policy=SimpleNamespace(
                type="grpo",
                on_policy=False,
                allowed_outdated_steps=10,
                coalesce_weight_sync=False,
                data_dispatch_as_rank_in_mesh=False,
                rollout_as_token_ids=False,
                dispatch_balance_key=key,
            ),
        ),
    )
    manager._publish_payload_transport_cleanup = lambda *_args: None
    return manager


def _streams(rows, replicas):
    """Per-replica publication order, as each policy worker data queue sees it."""
    return [
        [rollout for recipient, rollout in rows if recipient is replica]
        for replica in replicas
    ]


def _per_rank(stream, dp_size):
    """Mirror ``RLPolicyWorker.dispatch_rollouts``: item ``i`` goes to data rank ``i % dp``."""
    return [stream[rank::dp_size] for rank in range(dp_size)]


def _weight(rollout):
    return (rollout.extra_info or {}).get(KEY, 0)


def test_balanced_rows_give_every_rank_equal_count_and_even_weight():
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 2)]
    manager = _manager(HEAVY_LIGHT, replicas, count=4)

    rows = manager._take_dispatch_rows(replicas)

    assert manager.rollout_buffer.empty()
    assert sorted(rollout.prompt_idx for _, rollout in rows) == list(range(8))
    for stream in _streams(rows, replicas):
        assert len(stream) == 4
        for rank_rollouts in _per_rank(stream, 2):
            assert sorted(_weight(r) for r in rank_rollouts) == [46, 177]


def test_unset_key_keeps_fifo_round_robin():
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 2)]
    manager = _manager(HEAVY_LIGHT, replicas, count=4, key=None)

    rows = manager._take_dispatch_rows(replicas)

    assert [rollout.prompt_idx for _, rollout in rows] == list(range(8))
    assert [[r.prompt_idx for r in s] for s in _streams(rows, replicas)] == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]


def test_equal_weights_take_the_oldest_rollouts_in_arrival_order():
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 2)]
    manager = _manager([3] * 10, replicas, count=4)

    rows = manager._take_dispatch_rows(replicas)

    assert sorted(rollout.prompt_idx for _, rollout in rows) == list(range(8))
    assert manager.rollout_buffer.qsize() == 2
    for stream in _streams(rows, replicas):
        for rank_rollouts in _per_rank(stream, 2):
            prompts = [r.prompt_idx for r in rank_rollouts]
            assert prompts == sorted(prompts)


def test_missing_or_invalid_weight_counts_as_zero_and_warns_once(monkeypatch):
    warning = MagicMock()
    monkeypatch.setattr("cosmos_rl.dispatcher.status.logger.warning", warning)
    replicas = [_replica("policy-0", 0, 2)]
    manager = _manager([None, "many", 5, 5], replicas, count=4)

    rows = manager._take_dispatch_rows(replicas)

    assert warning.call_count == 1
    assert "weight 0" in warning.call_args.args[0]
    for rank_rollouts in _per_rank(_streams(rows, replicas)[0], 2):
        assert [_weight(r) for r in rank_rollouts].count(5) == 1


def test_count_must_divide_the_replica_data_parallel_size():
    replicas = [_replica("policy-0", 0, 2)]
    manager = _manager([1, 2, 3], replicas, count=3)
    with pytest.raises(ValueError, match="divisible"):
        manager._take_dispatch_rows(replicas)


def test_real_dispatch_publishes_balanced_replica_streams():
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 2)]
    manager = _manager(HEAVY_LIGHT, replicas, count=4)

    manager.try_trigger_data_fetch_and_training()

    plan = manager.redis_handler.publish_plan.call_args.args[0]
    for replica in replicas:
        stream = [
            msgpack.unpackb(data)["prompt_idx"]
            for name, field, data in plan.entries
            if name == replica.name + "_rollout" and field == "rollout"
        ]
        assert len(stream) == 4
        for rank_prompts in _per_rank(stream, 2):
            assert sorted(HEAVY_LIGHT[p] for p in rank_prompts) == [46, 177]
    assert manager.dispatched_rollouts_by_step == {1: 8}
    assert manager.remain_samples_num == 92
    assert manager.rollout_buffer.empty()


def test_payload_lookahead_reserves_with_the_same_selection(monkeypatch):
    replicas = [_replica("policy-0", 0, 2), _replica("policy-1", 1, 2)]
    manager = _manager(HEAVY_LIGHT + HEAVY_LIGHT, replicas, count=4, prefetch=True)
    calls = []
    original = manager._take_dispatch_rows

    def spy(target):
        rows = original(target)
        calls.append(rows)
        return rows

    monkeypatch.setattr(manager, "_take_dispatch_rows", spy)

    manager.try_trigger_data_fetch_and_training()

    # One real step plus the admitted lookahead, both through the same selection.
    assert len(calls) == 2
    _, _, reserved = manager._payload_lookahead
    assert reserved == [rollout for _, rollout in calls[1]]
    assert set(manager._payload_notifications) == {"policy-0", "policy-1"}
    assert manager.rollout_buffer.empty()
    for rows in calls:
        for stream in _streams(rows, replicas):
            for rank_rollouts in _per_rank(stream, 2):
                assert sorted(_weight(r) for r in rank_rollouts) == [46, 177]
    assert manager.dispatched_rollouts_by_step == {1: 8}
    assert manager.remain_samples_num == 92


def test_replica_data_rank_count_reads_the_dp_shard_group_size():
    atoms = []
    for rank in range(2):
        ranks = [0] * len(MESH_NAMES)
        group_size = [1] * len(MESH_NAMES)
        ranks[MESH_NAMES.index("dp_shard")] = rank
        group_size[MESH_NAMES.index("dp_shard")] = 2
        atoms.append(
            Atom(
                global_rank=rank,
                host_ip="127.0.0.1",
                host_name="localhost",
                trace_path="",
                ranks=ranks,
                group_size=group_size,
                replica_name="policy-0",
            )
        )
    assert Replica("policy-0", Role.POLICY, atoms).data_rank_count() == 2


def test_config_rejects_balancing_with_rank_affine_dispatch():
    assert GrpoConfig(type="grpo", dispatch_balance_key=KEY).dispatch_balance_key == KEY
    with pytest.raises(ValueError, match="dispatch_balance_key"):
        GrpoConfig(
            type="grpo", dispatch_balance_key=KEY, data_dispatch_as_rank_in_mesh=True
        )
