# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rank-affine prompt loss must reopen the same destination's capacity."""

import asyncio
from queue import Queue

import pytest

from cosmos_rl.dispatcher.data.schema import RLPayload, TrainingCompletionIdentity
from cosmos_rl.dispatcher.reservations import ProducerReservations
from test_dispatch_fetch_bounds import fetcher
from test_zero_staleness_refill_deadlock import _config, _controller


def controller_with_rank_fetcher(*, rollout_size=2, policy_size=2, group=1, quota=1):
    config = _config()
    config.train.train_policy.data_dispatch_as_rank_in_mesh = True
    config.rollout.n_generation = group
    config.train.train_batch_per_replica = quota * group
    item = _controller(config, current_step=0, samples_on_the_fly=0, pending_rollouts=0)
    item.policy_status_manager.__len__.return_value = policy_size
    item.policy_status_manager.pending_rank_training_batches.return_value = 0
    item.weight_version_to_prompt_num = {}
    item.data_fetcher = fetcher(
        rollout_size=rollout_size, policy_size=policy_size, quota=quota * group
    )
    item.data_fetcher.config.rollout.n_generation = group
    return item


def get(item, rank, n=1):
    return asyncio.run(item._get_batched_prompt_impl(n, rank_in_mesh=rank))[0]


@pytest.mark.parametrize("rollout_size,policy_size", [(2, 2), (4, 2), (2, 4), (3, 2)])
def test_rejected_rank_refills_only_its_queue_at_original_version(
    rollout_size, policy_size
):
    item = controller_with_rank_fetcher(
        rollout_size=rollout_size, policy_size=policy_size
    )
    assert [p.index for p in get(item, 0)] == [0]
    for _ in range(policy_size * 2):
        get(item, 0)
    item.policy_status_manager.samples_on_the_fly = 0
    assert item.register_discarded_samples_for_refill(0, 1, "drop", policy_rank=0) == 1
    replacement = []
    for _ in range(10):
        replacement += get(item, 0)
        if replacement:
            break
    assert replacement and replacement[0].index % policy_size == 0
    assert replacement[0].weight_version == 0
    assert item.register_discarded_samples_for_refill(0, 1, "drop", policy_rank=0) == 0
    assert item.data_fetcher._rank_discarded_samples == {(0, 0): 1}
    assert item.weight_version_to_replacement_prompt_num == {}


def test_partial_losses_round_per_policy_not_across_destinations():
    item = controller_with_rank_fetcher(group=4)
    assert get(item, 0) and get(item, 1)
    item.policy_status_manager.samples_on_the_fly = 6
    assert item.register_discarded_samples_for_refill(0, 1, "a", policy_rank=0) == 1
    assert item.register_discarded_samples_for_refill(0, 1, "b", policy_rank=1) == 1
    assert get(item, 0)[0].weight_version == 0
    assert get(item, 1)[0].weight_version == 0
    assert item.register_discarded_samples_for_refill(0, 3, "c", policy_rank=0) == 0
    assert item.data_fetcher._rank_discarded_samples == {(0, 0): 4, (0, 1): 1}


def test_refill_cannot_spend_other_ranks_credit():
    item = controller_with_rank_fetcher()
    assert get(item, 0) and get(item, 1)
    item.policy_status_manager.samples_on_the_fly = 1
    item.register_discarded_samples_for_refill(0, 1, "a", policy_rank=1)
    pulled = list(item.data_fetcher.pulled)
    assert get(item, 0) == []
    assert item.data_fetcher.pulled == pulled
    assert get(item, 1)[0].weight_version == 0


def test_colocated_preparation_can_refill_its_rank_after_prompt_quota():
    item = controller_with_rank_fetcher()
    item.config.mode = "colocated"
    assert [p.index for p in get(item, 0)] == [0]
    # Colocated preparation, not a disaggregated version quota, decides when
    # enough usable completions exist. The rejected local group needs a fetch.
    assert [p.index for p in get(item, 0)] == [2]
    assert item.data_fetcher.data_fetched_for_each_policy_at_step == {}


@pytest.mark.parametrize("rank", [0, 1])
def test_colocated_startup_assigns_rollout_mesh_without_communicator(monkeypatch, rank):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cosmos_rl.colocated.controller import ColocatedController
    from cosmos_rl.dispatcher.command import (
        BuildMeshCommand,
        PolicyToRolloutUnicastCommand,
        RolloutToRolloutBroadcastCommand,
    )

    names = {"rollout-a": 0, "rollout-b": 1}
    build = object.__new__(BuildMeshCommand)
    build.replica_name_to_rank = names
    sync = object.__new__(RolloutToRolloutBroadcastCommand)
    sync.validation_round_id = sync.validation_protocol_version = None
    rollout = SimpleNamespace(
        replica_name=list(names)[rank], rank_in_rollout_repicas=-1, world_size=1
    )
    item = SimpleNamespace(
        config=SimpleNamespace(
            validation=SimpleNamespace(enable=False),
            train=SimpleNamespace(sync_weight_interval=1),
        ),
        current_step=0,
        total_steps=20,
        rollout=rollout,
        policy=SimpleNamespace(world_size=1),
        policy_replica=object(),
        rollout_replica=object(),
        command_dispatcher=object(),
        wait_for_remote_command=Mock(side_effect=[build, sync]),
    )
    for cls in (PolicyToRolloutUnicastCommand, RolloutToRolloutBroadcastCommand):
        monkeypatch.setattr(cls, "trigger", Mock())
    assert ColocatedController.rollout_consume_one_step_commands_util_r2r(
        item, initial=True
    )
    assert rollout.rank_in_rollout_repicas == rank
    assert rollout.replica_name_to_rank == names


def test_colocated_worker_forwards_its_rollout_mesh_rank(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cosmos_rl.rollout.worker.colocated.rollout_control import (
        ColocatedRolloutControlWorker,
    )

    item = SimpleNamespace(
        batch_size=1,
        _prompt_queue=Queue(),
        current_weight_version=0,
        rank_in_rollout_repicas=1,
        request_new_prompts=Mock(return_value=True),
    )
    monkeypatch.setattr(
        "cosmos_rl.rollout.worker.colocated.rollout_control.dist_util.all_gather_object_cpu",
        lambda state: [state],
    )
    assert ColocatedRolloutControlWorker.rollout_for_one_minor_step(item) == (True, 0)
    item.request_new_prompts.assert_called_once_with(
        1, item._prompt_queue, rank_in_mesh=1
    )


@pytest.mark.parametrize("validation", [False, True])
def test_other_ranks_buffered_rows_do_not_suppress_end_of_data(validation):
    item = fetcher(policy_size=2, quota=4, indices=[1, 3])
    item.config.train.epoch = 1
    if validation:
        item.validation_get_dataloader = lambda step: item.train_dataloader_iter
    kwargs = {"validation_step": 0} if validation else {}
    values, end = item.get_batched_prompt(4, rank_in_mesh=0, **kwargs)
    assert values == [] and end
    buffer = (
        item.fetched_data_buffer_for_validation
        if validation
        else item.fetched_data_buffer
    )
    assert [index for index, _ in buffer] == [1, 3]
    values, _ = item.get_batched_prompt(2, rank_in_mesh=1, **kwargs)
    assert [value.index for value in values] == [1, 3]


def test_exhausted_rank_can_drain_its_last_rows_without_advancing_epochs():
    from types import SimpleNamespace

    item = fetcher(policy_size=2, indices=[])
    item.train_iterator_exhausted = True
    item.epoch = 2
    item.fetched_data_buffer = [
        (0, SimpleNamespace(index=0)),
        (1, SimpleNamespace(index=1)),
    ]
    values, end = item.get_batched_prompt(1, rank_in_mesh=0, weight_version=0)
    assert [value.index for value in values] == [0] and end
    assert item.epoch == 2 and [index for index, _ in item.fetched_data_buffer] == [1]
    item.data_fetched_for_each_policy_at_step[0][1] = 1
    assert item.get_batched_prompt(1, rank_in_mesh=1, weight_version=0) == ([], False)
    values, end = item.get_batched_prompt(1, rank_in_mesh=1, weight_version=1)
    assert [value.index for value in values] == [1] and end and item.epoch == 2


def test_full_quota_does_not_hide_known_source_exhaustion():
    item = fetcher(policy_size=2)
    item.train_iterator_exhausted = True
    item.data_fetched_for_each_policy_at_step[0] = {0: 1}
    item.fetched_data_buffer = [(1, None)]
    assert item.get_batched_prompt(1, rank_in_mesh=0, weight_version=0) == ([], True)
    assert item.fetched_data_buffer == [(1, None)] and item.epoch == 1


def test_rank_admission_rejects_changed_destination_before_mutation(monkeypatch):
    from test_training_fetch_receipts import setup, fetch
    from test_producer_reservations import completed_request
    from test_rollout_report_receipts import post

    item, _, identity = setup(monkeypatch)
    item.config.train.train_policy.data_dispatch_as_rank_in_mesh = True
    item.data_fetcher.prune_rank_prompt_capacity = lambda step: None
    item.policy_status_manager.rollout_buffer_per_rank = [Queue(), Queue()]
    item.policy_status_manager.policy_replicas["second"] = next(
        iter(item.policy_status_manager.policy_replicas.values())
    )
    assert len(item.policy_status_manager) == 2
    fetched = RLPayload.model_validate(fetch(1, identity)["payloads_list"][0])
    request = completed_request(fetched)
    request.payloads[0].prompt_idx = 8
    ledger = item.rollout_status_manager["source"].producer_reservations
    before = item.policy_status_manager.samples_on_the_fly
    assert post(request).status_code == 409
    assert ledger.outstanding == 2
    assert item.policy_status_manager.samples_on_the_fly == before
    assert all(q.empty() for q in item.policy_status_manager.rollout_buffer_per_rank)
    assert item.policy_status_manager.terminal_error is None


def test_reservation_plan_retains_original_queue_for_partial_failure():
    from cosmos_rl.dispatcher.protocol import RolloutRequest

    ledger = ProducerReservations()
    first = ledger.issue(0, 4, policy_rank=0)
    second = ledger.issue(0, 4, policy_rank=1)
    request = RolloutRequest(
        src_replica_name="source",
        payloads=[],
        training_rejections=[
            TrainingCompletionIdentity(work_id=first, slot=0),
            TrainingCompletionIdentity(work_id=second, slot=0),
        ],
        metrics={"discarded_samples": 2},
    )
    plan = ledger.prepare_report(request, policy_size=2)
    assert plan.rejected_rank_counts == ((0, 0, 1), (0, 1, 1))
    ledger.commit(plan.ownership)
    assert ledger.outstanding_rank_counts() == {(0, 0): 3, (0, 1): 3}
    assert ledger.retire() == {0: 6}
    assert ledger.outstanding_rank_counts() == {}


def test_unrouted_rejection_cannot_silently_skip_settlement():
    from cosmos_rl.dispatcher.protocol import RolloutRequest

    ledger = ProducerReservations()
    work = ledger.issue(0, 1)
    request = RolloutRequest(
        src_replica_name="source",
        payloads=[],
        training_rejections=[TrainingCompletionIdentity(work_id=work, slot=0)],
        metrics={"discarded_samples": 1},
    )
    with pytest.raises(ValueError, match="original policy rank"):
        ledger.prepare_report(request, policy_size=2)
    assert ledger.outstanding == 1


def http_controller(monkeypatch):
    from test_training_fetch_receipts import setup

    item, _, identity = setup(monkeypatch)
    item.config.train.train_policy.data_dispatch_as_rank_in_mesh = True
    status = item.policy_status_manager
    status.policy_replicas["second"] = next(iter(status.policy_replicas.values()))
    status.rollout_buffer_per_rank = [Queue(), Queue()]
    status.set_discard_refill_hook(item.register_discarded_samples_for_refill)
    actual = fetcher(policy_size=2, quota=status.config.train.train_batch_per_replica)
    actual.config.rollout.n_generation = 2
    fetch = actual.get_batched_prompt

    def convert(*args, **kwargs):
        values, end = fetch(*args, **kwargs)
        return [RLPayload(prompt_idx=value.index) for value in values], end

    actual.get_batched_prompt = convert
    item.data_fetcher = actual
    return item, identity


@pytest.mark.parametrize(
    "remaining",
    [
        "none",
        "queued",
        "owned",
        "unissued",
        "lookahead",
        "unknown-iterator",
        "lost-policy",
        "colocated",
    ],
)
def test_rank_exhaustion_closes_only_provably_unfillable_input(monkeypatch, remaining):
    from cosmos_rl.dispatcher.data.schema import Rollout

    item, _ = http_controller(monkeypatch)
    status = item.policy_status_manager
    status.get_all_atoms_arrived_replicas = lambda: list(
        status.policy_replicas.values()
    )
    status.policy_init_done = True
    status.rollout_buffer_per_rank[0].put(Rollout(prompt_idx=0))
    for _ in range(2):
        status.rollout_buffer_per_rank[1].put(Rollout(prompt_idx=1))
    item.data_fetcher.train_iterator_exhausted = True
    if remaining == "queued":
        status.rollout_buffer_per_rank[0].put(Rollout(prompt_idx=0))
    elif remaining == "owned":
        item.rollout_status_manager["source"].producer_reservations.issue(
            0, 1, policy_rank=0
        )
    elif remaining == "unissued":
        item.data_fetcher.fetched_data_buffer = [(0, None)]
    elif remaining == "lookahead":
        status._payload_lookahead = (
            None,
            None,
            [Rollout(prompt_idx=rank) for rank in [0, 0, 1, 1]],
        )
    elif remaining == "unknown-iterator":
        item.data_fetcher.train_iterator_exhausted = False
    elif remaining == "lost-policy":
        status.policy_replicas.clear()
    elif remaining == "colocated":
        item.config.mode = "colocated"
    assert item.close_exhausted_rank_input() is (remaining == "none")
    # The probe only closes future prompts: no dropped ownership or fake ACK.
    assert [q.qsize() for q in status.rollout_buffer_per_rank] == (
        [2, 2] if remaining == "queued" else [1, 2]
    )
    assert status.stop_reason is None and status.terminal_error is None


def test_closed_rank_input_still_serves_validation(monkeypatch):
    from types import SimpleNamespace

    item, _ = http_controller(monkeypatch)
    item._rank_input_exhausted = True
    assert asyncio.run(item._get_batched_prompt_impl(1, rank_in_mesh=0)) == ([], True)
    item.data_fetcher.validation_get_dataloader = lambda step: iter(
        [
            (
                [0],
                [
                    SimpleNamespace(
                        index=0, prompt=None, conversation=None, reference_answer=None
                    )
                ],
            )
        ]
    )
    values, end = asyncio.run(
        item._get_batched_prompt_impl(1, validation_step=0, rank_in_mesh=0)
    )
    assert len(values) == 1 and not end


@pytest.mark.parametrize("identified", [False, True])
def test_http_failure_reopens_rank_quota_once_and_replacement_can_fetch(
    monkeypatch, identified
):
    from cosmos_rl.dispatcher.protocol import RolloutRequest
    from cosmos_rl.dispatcher.data.admission_state import CompletionAdmissionState
    from cosmos_rl.reward.identity import CompletionReporter
    from test_training_fetch_receipts import fetch
    from test_rollout_report_receipts import post

    item, identity = http_controller(monkeypatch)
    first = RLPayload.model_validate(
        fetch(1, {**identity, "rank_in_mesh": 0})["payloads_list"][0]
    )
    assert first.prompt_idx == 0
    if identified:
        item.completion_admission = CompletionAdmissionState()
        reporter = CompletionReporter("source", 0)
        reporter.reserve([first], 2, 0)
        request = reporter.generation_failure([first], "generation_error")
        request.report_session_id, request.report_sequence = "session", 0
    else:
        request = RolloutRequest(
            src_replica_name="source",
            src_global_rank=0,
            report_session_id="session",
            report_sequence=0,
            payloads=[],
            training_rejections=[
                TrainingCompletionIdentity(work_id=first.training_work_id, slot=slot)
                for slot in [0, 1]
            ],
            metrics={"discarded_samples": 2},
        )
    assert post(request) == post(request)
    assert item.policy_status_manager.samples_on_the_fly == 0
    result = fetch(1, {**identity, "fetch_sequence": 1, "rank_in_mesh": 0})
    assert len(result["payloads_list"]) == 1
    assert result["payloads_list"][0]["prompt_idx"] == 2
    assert result["payloads_list"][0]["weight_version"] == 0


def test_surplus_on_another_queue_cannot_reject_a_needed_completion():
    from test_terminal_drain_protocol import TestTerminalMatrix
    from cosmos_rl.dispatcher.data.schema import Rollout

    status, _ = TestTerminalMatrix._manager(0)
    status.config.train.train_policy.data_dispatch_as_rank_in_mesh = True
    status.config.train.train_policy.allowed_outdated_steps = 0
    status.config.train.sync_weight_interval = 1
    status.rollout_buffer_per_rank = [Queue(), Queue()]
    status.samples_on_the_fly = 5
    for _ in range(4):
        status.rollout_buffer_per_rank[0].put(Rollout(prompt_idx=0))
    needed = Rollout(prompt_idx=1, weight_version=0)
    assert status.filter_outdated_rollouts([needed]) == [needed]
    assert status.pending_rank_training_batches() == 0


@pytest.mark.parametrize(
    "sizes,reserved,expected",
    [([6, 0], False, 0), ([6, 2], False, 1), ([6, 2], True, 2)],
)
def test_only_balanced_batches_advance_rank_version(sizes, reserved, expected):
    from test_terminal_drain_protocol import TestTerminalMatrix

    status, _ = TestTerminalMatrix._manager(0)
    status.rollout_buffer_per_rank = [Queue(), Queue()]
    for queue, size in zip(status.rollout_buffer_per_rank, sizes):
        for _ in range(size):
            queue.put(None)
    status._payload_lookahead = (None, None, [None] * 4) if reserved else None
    assert status.pending_rank_training_batches() == expected


def test_finished_rank_quota_state_is_pruned_without_erasing_future_capacity():
    item = fetcher()
    item.data_fetched_for_each_policy_at_step = {0: {0: 1}, 1: {0: 1}}
    item.reopen_rank_prompt_capacity(0, 0, 1)
    item.reopen_rank_prompt_capacity(1, 0, 1)
    item.prune_rank_prompt_capacity(1)
    assert item.data_fetched_for_each_policy_at_step == {1: {0: 1}}
    assert item._rank_discarded_samples == {(1, 0): 1}


def exercise_partial_rank_refill(monkeypatch):
    """Real fetch/report accounting and rank queues, without an optimizer stub."""
    from test_training_fetch_receipts import fetch
    from test_producer_reservations import completed_request
    from test_rollout_report_receipts import post

    item, identity = http_controller(monkeypatch)
    for rank in range(2):
        payload = RLPayload.model_validate(
            fetch(1, {**identity, "fetch_sequence": rank, "rank_in_mesh": rank})[
                "payloads_list"
            ][0]
        )
        request = completed_request(payload)
        request.report_sequence = rank
        request.training_rejections = [
            TrainingCompletionIdentity(work_id=payload.training_work_id, slot=1)
        ]
        request.metrics = {"discarded_samples": 1}
        assert post(request) == post(request) == {"message": "Rollout put"}
    assert [q.qsize() for q in item.policy_status_manager.rollout_buffer_per_rank] == [
        1,
        1,
    ]
    for rank in range(2):
        payload = RLPayload.model_validate(
            fetch(1, {**identity, "fetch_sequence": rank + 2, "rank_in_mesh": rank})[
                "payloads_list"
            ][0]
        )
        assert payload.weight_version == 0 and payload.prompt_idx % 2 == rank
        request = completed_request(payload, count=2)
        request.report_sequence = rank + 2
        assert post(request) == post(request) == {"message": "Rollout put"}
    queues = item.policy_status_manager.rollout_buffer_per_rank
    assert [q.qsize() for q in queues] == [2, 2]
    assert item.policy_status_manager.samples_on_the_fly == 4
    assert item.policy_status_manager.terminal_error is None
    return [[q.get_nowait() for _ in range(2)] for q in queues]


def test_partial_rank_refill_completes_both_queues_without_loosening_staleness(
    monkeypatch,
):
    assert len(exercise_partial_rank_refill(monkeypatch)) == 2


def test_retirement_reopens_only_original_queues_once(monkeypatch):
    from test_training_fetch_receipts import fetch
    from cosmos_rl.dispatcher.status import RolloutStatusManager

    item, identity = http_controller(monkeypatch)
    for rank in range(2):
        fetch(1, {**identity, "fetch_sequence": rank, "rank_in_mesh": rank})
    replica = item.rollout_status_manager["source"]
    RolloutStatusManager.release_producer_reservations(
        replica, item.policy_status_manager
    )
    RolloutStatusManager.release_producer_reservations(
        replica, item.policy_status_manager
    )
    assert item.policy_status_manager.samples_on_the_fly == 0
    assert item.data_fetcher._rank_discarded_samples == {(0, 0): 2, (0, 1): 2}


def test_retirement_after_last_policy_unregister_uses_reserved_mesh(monkeypatch):
    from test_training_fetch_receipts import fetch
    from cosmos_rl.dispatcher.status import RolloutStatusManager

    item, identity = http_controller(monkeypatch)
    fetch(1, {**identity, "rank_in_mesh": 0})
    # Final policy departure does not invalidate already issued producer work.
    item.policy_status_manager.policy_replicas.clear()
    assert len(item.policy_status_manager) == 0
    replica = item.rollout_status_manager["source"]
    RolloutStatusManager.release_producer_reservations(
        replica, item.policy_status_manager
    )
    assert item.policy_status_manager.samples_on_the_fly == 0
    assert item.policy_status_manager.terminal_error is None
    assert item.data_fetcher._rank_discarded_samples == {(0, 0): 2}


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_rank_refill_is_needed_even_when_generation_may_be_stale(lag):
    item = controller_with_rank_fetcher()
    item.config.train.train_policy.on_policy = False
    item.config.train.train_policy.allowed_outdated_steps = lag
    assert get(item, 0)
    item.policy_status_manager.samples_on_the_fly = 0
    assert item.register_discarded_samples_for_refill(0, 1, "a", policy_rank=0) == 1
    assert get(item, 0)[0].weight_version == 0
