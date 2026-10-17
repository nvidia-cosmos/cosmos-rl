# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import pytest

from cosmos_rl.dispatcher.reservations import ProducerReservations


def test_retirement_releases_only_unreported_slots_at_original_versions():
    ledger = ProducerReservations()
    a, b = ledger.issue(2, 4), ledger.issue(3, 2)
    plan = ledger.prepare([(a, [0, 2]), (b, [0])])
    assert plan.requested_version_counts == ((2, 2), (3, 1))
    assert ledger.outstanding == 6  # Preparing a report is not acceptance.
    ledger.commit(plan)
    assert ledger.outstanding == 3
    assert ledger.retire() == {2: 2, 3: 1}
    assert ledger.retire() == {} and ledger.outstanding == 0


def test_completed_work_is_released_and_cannot_be_settled_again():
    ledger = ProducerReservations()
    work = ledger.issue(0, 2)
    ledger.commit(ledger.prepare([(work, [0, 1])]))
    assert not ledger._work
    with pytest.raises(ValueError, match="already settled"):
        ledger.prepare([(work, [0, 1])])
    assert ledger.retire() == {}


@pytest.mark.parametrize("slots", [[0, 0], [2], [-1], [True], []])
def test_bad_report_cannot_partially_settle_earlier_valid_work(slots):
    ledger = ProducerReservations()
    a, b = ledger.issue(0, 2), ledger.issue(0, 2)
    with pytest.raises(ValueError):
        ledger.prepare([(a, [0]), (b, slots)])
    assert ledger.outstanding == 4 and ledger.retire() == {0: 4}


def test_two_entries_cannot_claim_the_same_slot():
    ledger = ProducerReservations()
    work = ledger.issue(0, 2)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.prepare([(work, [0]), (work, [0])])
    ledger.commit(ledger.prepare([(work, [0]), (work, [1])]))
    assert ledger.outstanding == 0


def test_partial_settlement_does_not_allow_reclaiming_a_finished_slot():
    ledger = ProducerReservations()
    work = ledger.issue(9, 4)
    ledger.commit(ledger.prepare([(work, [0, 2])]))
    with pytest.raises(ValueError, match="settled"):
        ledger.prepare([(work, [2, 3])])
    assert ledger.retire() == {9: 2}


def test_separate_producers_and_replacement_incarnations_never_share_ids():
    first, second = ProducerReservations(), ProducerReservations()
    a, b = first.issue(0, 1), second.issue(0, 1)
    assert a != b
    with pytest.raises(ValueError, match="Unknown"):
        second.prepare([(a, [0])])
    assert first.retire() == {0: 1} and second.outstanding == 1


@pytest.mark.parametrize("mutation", ["issue", "commit", "retire", "foreign"])
def test_prepared_plan_cannot_outlive_its_owned_revision(mutation):
    ledger = ProducerReservations()
    work = ledger.issue(0, 2)
    plan = ledger.prepare([(work, [0])])
    if mutation == "issue":
        ledger.issue(1, 1)
    elif mutation == "commit":
        ledger.commit(ledger.prepare([(work, [1])]))
    elif mutation == "retire":
        ledger.retire()
    else:
        ledger = ProducerReservations()
    with pytest.raises(RuntimeError, match="source/revision"):
        ledger.commit(plan)


def test_live_state_does_not_grow_with_completed_history():
    ledger = ProducerReservations()
    ids = set()
    for _ in range(1000):
        work = ledger.issue(0, 2)
        assert work not in ids
        ids.add(work)
        ledger.commit(ledger.prepare([(work, [0, 1])]))
        assert not ledger._work
    assert ledger.outstanding == 0


def test_http_fetch_replay_issues_one_set_of_controller_owned_slots(monkeypatch):
    from test_training_fetch_receipts import setup, fetch

    controller, _, identity = setup(monkeypatch)
    first = fetch(1, identity)
    assert first == fetch(1, identity)
    payload = first["payloads_list"][0]
    assert payload["training_completion_slots"] == [0, 1]
    ledger = controller.rollout_status_manager["source"].producer_reservations
    assert ledger.outstanding == 2
    plan = ledger.prepare([(payload["training_work_id"], [0, 1])])
    assert plan.requested_version_counts == ((0, 2),)


def test_repeating_dataset_index_gets_distinct_controller_work(monkeypatch):
    from test_training_fetch_receipts import setup, fetch

    controller, _, identity = setup(monkeypatch)
    controller.config.train.train_policy.allowed_outdated_steps = 10
    controller.config.train.train_policy.max_inflight_steps = None
    first = fetch(1, identity)["payloads_list"][0]
    identity["fetch_sequence"] = 1
    second = fetch(1, identity)["payloads_list"][0]
    assert first["prompt_idx"] == second["prompt_idx"] == 7
    assert first["training_work_id"] != second["training_work_id"]
    assert (
        controller.rollout_status_manager["source"].producer_reservations.outstanding
        == 4
    )


@pytest.mark.parametrize("mask", [[True, False, True], [False, False, False]])
def test_quality_selection_preserves_work_and_exact_rejected_slots(mask):
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.reward.admission import (
        resolve_completion_admission,
        select_payload_completions,
    )

    source = RLPayload(
        prompt_idx=7,
        completions=["a", "b", "c"],
        completion_trainable=mask,
        training_work_id="issued:0",
        training_completion_slots=[1, 3, 4],
        training_rejected_slots=[0, 2],
    )
    selected = select_payload_completions(
        source, resolve_completion_admission(source, 1)
    )
    assert selected.training_work_id == "issued:0"
    expected = [slot for slot, valid in zip([1, 3, 4], mask) if valid]
    assert selected.training_completion_slots == expected
    assert sorted(selected.training_rejected_slots + expected) == list(range(5))
    assert source.training_rejected_slots == [0, 2]


def test_report_preserves_quality_drops_and_whole_dapo_group_removal():
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.dispatcher.protocol import RolloutRequest
    from cosmos_rl.reward.reservations import attach_training_rejections

    kept = RLPayload(
        training_work_id="a",
        training_completion_slots=[0, 2],
        training_rejected_slots=[1],
        completions=["a", "c"],
    )
    dropped = RLPayload(
        training_work_id="b", training_completion_slots=[0, 1], completions=["x", "y"]
    )
    report = RolloutRequest(src_replica_name="source", payloads=[kept])
    attach_training_rejections(report, [kept, dropped])
    assert [(r.work_id, r.slot) for r in report.training_rejections] == [
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert report.payloads[0].completions == ["a", "c"]
    # Round-trip the wire; local excluded fields must not be needed remotely.
    restored = RolloutRequest.model_validate(report.model_dump(mode="json"))
    assert restored.training_rejections == report.training_rejections
    assert restored.payloads[0].training_rejected_slots == []


def test_identified_selection_and_late_failure_keep_controller_coordinates():
    from types import SimpleNamespace
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.dispatcher.data.admission import SourceWindow
    from cosmos_rl.reward.admission import (
        resolve_completion_admission,
        select_payload_completions,
    )
    from cosmos_rl.reward.identity import CompletionReporter

    original = RLPayload(
        training_work_id="controller:4",
        training_completion_slots=[0, 1, 2],
        completions=["a", "b", "c"],
        completion_trainable=[True, False, True],
    )
    reporter = CompletionReporter("source", 0)
    reporter.reserve([original], 3, 9)
    selected = select_payload_completions(
        original, resolve_completion_admission(original, 1)
    )
    report = reporter.report(
        [selected], SimpleNamespace(get_rollout_output=lambda *args: (*args, None))
    )
    assert [i.reservation.slot for i in report.completion_identities] == [0, 2]
    rejected = report.completion_failures[0].identity
    assert rejected.reservation.slot == 1
    assert rejected.reservation.work_id == "controller:4"
    assert rejected.weight_version == 9
    assert report.training_rejections == [rejected.reservation]
    failure = reporter.generation_failure([original], "generation_error")
    assert [f.identity.reservation.slot for f in failure.completion_failures] == [
        0,
        1,
        2,
    ]
    window = SourceWindow(2)
    window.commit([rejected])
    assert window.unseen([rejected]) == [False]
    changed = rejected.model_copy(
        update={"reservation": rejected.reservation.model_copy(update={"slot": 0})}
    )
    with pytest.raises(ValueError, match="controller reservation"):
        window.unseen([changed])
    window.commit([rejected.model_copy(update={"sequence": 3})])
    assert set(window.reservations) == {3}


@pytest.mark.parametrize("size", [0, 1, 2])
def test_structural_short_group_reports_only_missing_original_slots(size):
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.rollout.schema import RolloutResult
    from test_discarded_rollout_accounting import _rollout_worker

    worker = _rollout_worker(n_generation=2)
    payload = RLPayload(training_work_id="issued:7", training_completion_slots=[0, 1])
    values = ["unchanged-a", "unchanged-b"][:size]
    kept, results = worker._filter_valid_rollout_results_and_report(
        [RolloutResult(completions=values)], [payload]
    )
    if size < 2:
        report = worker.api_client.post_rollout_completion.call_args.args[0]
        assert report.metrics["discarded_samples"] == 2 - size
        assert [identity.slot for identity in report.training_rejections] == list(
            range(size, 2)
        )
    else:
        worker.api_client.post_rollout_completion.assert_not_called()
    if size:
        queued = worker.reward_dispatcher.enqueue_rewards_cal.call_args.args[0][0]
        assert queued.training_completion_slots == list(range(size))
        assert queued.completions == values
    else:
        assert kept == results == []
    assert payload.training_completion_slots == [0, 1]


def test_structural_filter_cannot_admit_unreserved_surplus_completions():
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.rollout.schema import RolloutResult
    from test_discarded_rollout_accounting import _rollout_worker

    worker = _rollout_worker(n_generation=2)
    payload = RLPayload(training_work_id="issued:7", training_completion_slots=[0, 1])
    with pytest.raises(ValueError, match="controller-reserved"):
        worker._filter_valid_rollout_results_and_report(
            [RolloutResult(completions=["a", "b", "surplus"])], [payload]
        )
    worker.reward_dispatcher.enqueue_rewards_cal.assert_not_called()
    worker.api_client.post_rollout_completion.assert_not_called()


def fetched_controller(monkeypatch):
    from test_training_fetch_receipts import setup, fetch
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.dispatcher.replica import Replica
    from cosmos_rl.dispatcher.protocol import Role
    import test_terminal_drain_protocol as fixture

    controller, _, identity = setup(monkeypatch)
    controller.policy_status_manager.set_discard_refill_hook(
        controller.register_discarded_samples_for_refill
    )
    controller.rollout_status_manager.trigger_rebuild_mesh = lambda *args: None
    other = fixture._rollout_atom("survivor", 0, 0)
    other.report_session_id = "survivor-session"
    controller.rollout_status_manager.rollout_replicas["survivor"] = Replica(
        "survivor", Role.ROLLOUT, [other]
    )
    first = RLPayload.model_validate(fetch(1, identity)["payloads_list"][0])
    return controller, first


def completed_request(payload, count=1):
    from cosmos_rl.dispatcher.protocol import RolloutRequest

    delivered = payload.model_copy(
        update={
            "training_completion_slots": payload.training_completion_slots[:count],
            "completions": ["reply"] * count,
            "rewards": [1.0] * count,
            "advantages": [0.0] * count,
            "completion_token_ids": [[[3]]] * count,
        }
    )
    return RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="session",
        report_sequence=0,
        payloads=[delivered],
    )


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_http_departure_releases_only_unreported_work(monkeypatch, accepted):
    import asyncio
    from test_rollout_report_receipts import post

    controller, fetched = fetched_controller(monkeypatch)
    status = controller.policy_status_manager
    if accepted:
        request = completed_request(fetched, accepted)
        assert post(request) == post(request) == {"message": "Rollout put"}
    replica = controller.rollout_status_manager["source"]
    assert replica.producer_reservations.outstanding == 2 - accepted
    asyncio.run(controller.unregister("source"))
    assert status.samples_on_the_fly == accepted
    assert status.rollout_buffer.qsize() == accepted
    assert replica.producer_reservations.outstanding == 0
    assert replica.producer_reservations.retired
    if accepted:
        assert post(request).status_code == 410
    assert status.terminal_error is None


def test_departure_reopens_strict_original_version_for_surviving_producer(monkeypatch):
    import asyncio
    from test_training_fetch_receipts import fetch

    controller, _ = fetched_controller(monkeypatch)
    asyncio.run(controller.unregister("source"))
    result = fetch(
        1,
        dict(
            src_replica_name="survivor",
            src_global_rank=0,
            fetch_session_id="survivor-session",
            fetch_sequence=0,
        ),
    )
    assert len(result["payloads_list"]) == 1
    assert result["payloads_list"][0]["weight_version"] == 0
    assert controller.policy_status_manager.samples_on_the_fly == 2


def test_discard_and_departure_each_settle_only_their_own_slot(monkeypatch):
    import asyncio
    from test_rollout_report_receipts import post
    from cosmos_rl.dispatcher.data.schema import TrainingCompletionIdentity

    controller, fetched = fetched_controller(monkeypatch)
    request = completed_request(fetched)
    request.payloads = []
    request.training_rejections = [
        TrainingCompletionIdentity(work_id=fetched.training_work_id, slot=0)
    ]
    request.metrics = {"discarded_samples": 1, "discarded_weight_version": 99}
    assert post(request) == post(request) == {"message": "Rollout put"}
    status = controller.policy_status_manager
    assert status.samples_on_the_fly == 1
    assert controller.weight_version_to_replacement_prompt_num == {0: 1}
    asyncio.run(controller.unregister("source"))
    assert status.samples_on_the_fly == 0
    assert controller.weight_version_to_replacement_prompt_num == {0: 1}


def test_unknown_work_rejected_before_any_counter_or_admission_mutation(monkeypatch):
    from test_rollout_report_receipts import post

    controller, fetched = fetched_controller(monkeypatch)
    request = completed_request(fetched)
    request.payloads[0].training_work_id = "unissued"
    assert post(request).status_code == 409
    assert controller.policy_status_manager.samples_on_the_fly == 2
    assert controller.policy_status_manager.rollout_buffer.qsize() == 0
    assert (
        controller.rollout_status_manager["source"].producer_reservations.outstanding
        == 2
    )


def test_report_and_departure_share_one_ownership_boundary(monkeypatch):
    import asyncio
    from cosmos_rl.dispatcher import run_web_panel as web

    controller, fetched = fetched_controller(monkeypatch)
    request = completed_request(fetched)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        original = web._apply_rollout_group

        async def paused(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(web, "_apply_rollout_group", paused)
        report = asyncio.create_task(web.put_rollout_group(request))
        await asyncio.wait_for(entered.wait(), 1)
        departure = asyncio.create_task(controller.unregister("source"))
        await asyncio.sleep(0)
        assert not departure.done()
        release.set()
        result, _ = await asyncio.wait_for(asyncio.gather(report, departure), 1)
        assert result == {"message": "Rollout put"}

    asyncio.run(run())
    assert controller.policy_status_manager.samples_on_the_fly == 1
    assert controller.policy_status_manager.rollout_buffer.qsize() == 1


def test_heartbeat_reaping_uses_the_same_unreported_only_release(monkeypatch):
    import time
    from test_rollout_report_receipts import post

    controller, fetched = fetched_controller(monkeypatch)
    assert post(completed_request(fetched)) == {"message": "Rollout put"}
    manager = controller.rollout_status_manager
    manager["source"].status.heartbeat_timestamp = 0
    manager["survivor"].status.heartbeat_timestamp = time.time()
    manager.maintain_life_status(controller.policy_status_manager)
    assert "source" not in manager and "survivor" in manager
    assert controller.policy_status_manager.samples_on_the_fly == 1
    assert controller.policy_status_manager.rollout_buffer.qsize() == 1


def test_identified_failure_then_late_cleanup_does_not_reclaim_or_resettle(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cosmos_rl.dispatcher.data.admission_state import CompletionAdmissionState
    from cosmos_rl.reward.identity import CompletionReporter
    from test_rollout_report_receipts import post

    controller, fetched = fetched_controller(monkeypatch)
    controller.completion_admission = CompletionAdmissionState()
    cleanup = controller.policy_status_manager._publish_payload_transport_cleanup = (
        Mock()
    )
    reporter = CompletionReporter("source", 0)
    reporter.reserve([fetched], 2, 0)
    failed = reporter.generation_failure([fetched], "generation_error")
    failed.report_session_id, failed.report_sequence = "session", 0
    assert post(failed) == {"message": "Identified rollout report processed"}
    ledger = controller.rollout_status_manager["source"].producer_reservations
    assert (
        ledger.outstanding == controller.policy_status_manager.samples_on_the_fly == 0
    )
    cleanup.assert_not_called()
    fetched.completions, fetched.rewards, fetched.advantages = (
        ["a", "b"],
        [1.0, 1.0],
        [0.0, 0.0],
    )
    fetched.completion_token_ids = [[[3]], [[4]]]
    late = reporter.report(
        [fetched], SimpleNamespace(get_rollout_output=lambda *args: (*args, None))
    )
    late.report_session_id, late.report_sequence = "session", 1
    assert (
        post(late) == post(late) == {"message": "Identified rollout report processed"}
    )
    assert cleanup.call_count == 2
    assert (
        ledger.outstanding == controller.policy_status_manager.samples_on_the_fly == 0
    )
    assert controller.policy_status_manager.rollout_buffer.qsize() == 0
    # Even a fresh HTTP report ID cannot settle these application slots twice.
    late.report_sequence = 2
    assert post(late) == {"message": "Identified rollout report processed"}
    assert cleanup.call_count == 2


def test_terminal_checkout_retires_only_unreported_slots_once(monkeypatch):
    from test_rollout_report_receipts import post

    controller, fetched = fetched_controller(monkeypatch)
    request = completed_request(fetched)
    assert post(request) == {"message": "Rollout put"}
    replica = controller.rollout_status_manager["source"]
    for atom in replica.atoms.values():
        atom.rollout_reporter = atom.global_rank == 0
    request.payloads, request.is_end, request.report_sequence = [], True, 1
    assert post(request) == post(request) == {"message": "Rollout end signal received"}
    assert replica.status.ended and replica.producer_reservations.retired
    assert controller.policy_status_manager.samples_on_the_fly == 1
    assert controller.policy_status_manager.rollout_buffer.qsize() == 1


def allow_another_prompt(controller):
    policy = controller.config.train.train_policy
    policy.on_policy, policy.allowed_outdated_steps = False, 10
    policy.max_inflight_steps = None


def test_departure_cannot_settle_dispatched_work_before_trainer_ack(monkeypatch):
    import asyncio
    from unittest.mock import Mock
    from cosmos_rl.dispatcher.status import PolicyStatusManager
    from test_training_fetch_receipts import fetch
    from test_rollout_report_receipts import post

    controller, first = fetched_controller(monkeypatch)
    allow_another_prompt(controller)
    second = fetch(
        1,
        dict(
            src_replica_name="source",
            src_global_rank=0,
            fetch_session_id="session",
            fetch_sequence=1,
        ),
    )
    assert len(second["payloads_list"]) == 1
    assert controller.policy_status_manager.samples_on_the_fly == 4
    assert post(completed_request(first, 2)) == {"message": "Rollout put"}
    manager = controller.policy_status_manager
    # Use the real dispatch plan/ACK lifecycle, not only a buffered-result check.
    PolicyStatusManager.try_trigger_data_fetch_and_training(manager)
    dispatch = manager.training_dispatches[1]
    assert dispatch.rollout_count == 2 and not dispatch.settled
    assert manager.rollout_buffer.empty()
    manager.redis_handler.publish_plan.assert_called_once()
    asyncio.run(controller.unregister("source"))
    assert manager.samples_on_the_fly == 2 and not dispatch.settled
    assert manager.dispatched_rollouts_by_step == {1: 2}
    manager.should_weight_sync_after_train_ack = Mock(return_value=False)
    for _ in range(2):
        manager.train_ack(
            "policy-0", 1, 2, False, {}, controller.rollout_status_manager
        )
    assert manager.samples_on_the_fly == 0 and dispatch.settled
    assert manager.dispatched_rollouts_by_step == {}
    assert manager.remain_samples_num == 18


def test_departure_does_not_release_another_producers_reservations(monkeypatch):
    import asyncio
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from test_training_fetch_receipts import fetch
    from test_rollout_report_receipts import post

    controller, first = fetched_controller(monkeypatch)
    allow_another_prompt(controller)
    reply = fetch(
        1,
        dict(
            src_replica_name="survivor",
            src_global_rank=0,
            fetch_session_id="survivor-session",
            fetch_sequence=0,
        ),
    )
    second = RLPayload.model_validate(reply["payloads_list"][0])
    # Fetch may reserve a future version; generation reports its actual adopted
    # version. Producer ownership still refers to the original reservation.
    second.weight_version = controller.policy_status_manager.current_step
    assert first.prompt_idx == second.prompt_idx  # Repeated dataset index is legal.
    assert first.training_work_id != second.training_work_id
    assert post(completed_request(second, 2)).status_code == 409
    manager = controller.rollout_status_manager
    assert manager["source"].producer_reservations.outstanding == 2
    assert manager["survivor"].producer_reservations.outstanding == 2
    asyncio.run(controller.unregister("source"))
    assert controller.policy_status_manager.samples_on_the_fly == 2
    assert manager["survivor"].producer_reservations.outstanding == 2
    request = completed_request(second, 2)
    request.src_replica_name, request.report_session_id = "survivor", "survivor-session"
    assert post(request) == {"message": "Rollout put"}
    assert manager["survivor"].producer_reservations.outstanding == 0
    assert controller.policy_status_manager.rollout_buffer.qsize() == 2


def test_reused_name_cannot_reclaim_a_retired_incarnations_work(monkeypatch):
    import asyncio
    from cosmos_rl.dispatcher.data.schema import RLPayload
    from cosmos_rl.dispatcher.replica import Replica
    from cosmos_rl.dispatcher.protocol import Role
    from test_terminal_drain_protocol import _rollout_atom
    from test_training_fetch_receipts import fetch
    from test_rollout_report_receipts import post

    controller, old_payload = fetched_controller(monkeypatch)
    asyncio.run(controller.unregister("source"))
    replacement = _rollout_atom("source", 0, 0)
    replacement.report_session_id = "replacement"
    controller.rollout_status_manager.rollout_replicas["source"] = Replica(
        "source", Role.ROLLOUT, [replacement]
    )
    reply = fetch(
        1,
        dict(
            src_replica_name="source",
            src_global_rank=0,
            fetch_session_id="replacement",
            fetch_sequence=0,
        ),
    )
    fresh = RLPayload.model_validate(reply["payloads_list"][0])
    assert fresh.prompt_idx == old_payload.prompt_idx
    assert fresh.training_work_id != old_payload.training_work_id
    stale = completed_request(old_payload, 2)
    assert post(stale).status_code == 409  # The old registered HTTP source is fenced.
    stale.report_session_id = "replacement"
    assert post(stale).status_code == 409  # A new source cannot claim the old work.
    assert controller.policy_status_manager.samples_on_the_fly == 2
    assert controller.policy_status_manager.rollout_buffer.empty()
    request = completed_request(fresh, 2)
    request.report_session_id, request.report_sequence = "replacement", 1
    assert post(request) == {"message": "Rollout put"}
    assert controller.policy_status_manager.rollout_buffer.qsize() == 2


@pytest.mark.parametrize("outcome", ["payload", "rejection"])
def test_end_with_training_outcomes_cannot_silently_retire_work(monkeypatch, outcome):
    from cosmos_rl.dispatcher.data.schema import TrainingCompletionIdentity
    from test_rollout_report_receipts import post

    controller, fetched = fetched_controller(monkeypatch)
    request = completed_request(fetched)
    request.is_end = True
    if outcome == "rejection":
        request.payloads = []
        request.training_rejections = [
            TrainingCompletionIdentity(work_id=fetched.training_work_id, slot=0)
        ]
    assert post(request).status_code == 409
    replica = controller.rollout_status_manager["source"]
    assert not replica.status.ended and not replica.producer_reservations.retired
    assert replica.producer_reservations.outstanding == 2
    assert controller.policy_status_manager.samples_on_the_fly == 2
    assert controller.policy_status_manager.rollout_buffer.empty()


@pytest.mark.parametrize("backend", ["local", "remote"])
@pytest.mark.parametrize("mask", [None, [True, False, True, True]])
def test_reward_paths_preserve_slots_without_changing_values(
    backend, mask, monkeypatch
):
    from queue import Queue
    from types import SimpleNamespace
    import torch
    from cosmos_rl.reward.local_calculator import LocalRewardCalculator
    from cosmos_rl.reward.remote_calculator import RemoteRewardCalculator
    from test_completion_admission import _payload, _TestAlgo

    baseline = _payload(mask)
    reserved = baseline.model_copy(deep=True)
    reserved.training_work_id, reserved.training_completion_slots = (
        "work:0",
        [0, 1, 2, 3],
    )

    def compute(payload):
        if backend == "local":
            calculator = LocalRewardCalculator()
            calculator.rl_algo = _TestAlgo([1.0, 100.0, 3.0, 5.0])
            calculator.config = SimpleNamespace(
                train=SimpleNamespace(
                    non_text=True,
                    train_policy=SimpleNamespace(min_filter_prefix_tokens=None),
                )
            )
            return calculator.compute_rewards([payload], False, 0)[0][0]
        calculator = RemoteRewardCalculator()
        calculator.minimum_trainable_completions = 2
        calculator.rl_algo = _TestAlgo([])
        calculator.uuid2payload = {"request": [payload]}
        calculator.uuid2replica = {"request": None}
        calculator.uuid2stage = {"request": "training"}
        calculator.uuid2step = {"request": 0}
        calculator.uuid2completions_per_payload = {"request": [4]}
        monkeypatch.setattr(
            calculator, "fetch_reward", lambda *_: torch.tensor([1.0, 100.0, 3.0, 5.0])
        )
        queue = Queue()
        queue.put("request")
        return calculator.get_results(queue)[0][0]

    original, result = compute(baseline), compute(reserved)
    assert result.training_work_id == "work:0"
    assert result.training_completion_slots == ([0, 2, 3] if mask else [0, 1, 2, 3])
    assert result.training_rejected_slots == ([1] if mask else [])
    ignored = {
        "training_work_id",
        "training_completion_slots",
        "training_rejected_slots",
    }
    assert result.model_dump(exclude=ignored) == original.model_dump(exclude=ignored)
    assert reserved.training_completion_slots == [0, 1, 2, 3]
