# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real local ingestion plus source-fenced HTTP settlement, preparation and ACK."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json as jsonlib
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import requests

from cosmos_rl.colocated.api_client import ColocatedAPIClient
from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.dispatcher.algo.grpo import GRPO
from cosmos_rl.dispatcher.command import TrainingCompleteCommand
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.reward.local_calculator import LocalRewardCalculator
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
import test_terminal_drain_protocol as terminal_fixture
from test_rollout_report_receipts import setup as report_setup, post


class Rewards(GRPO):
    def compute_reward(self, completions, reference, **kwargs):
        rewards = [float(i) for i in range(len(completions))]
        return rewards, rewards, [{} for _ in completions]


def system(
    monkeypatch,
    *,
    centralized=False,
    width=2,
    rank=0,
    lost_reply=False,
    steps=1,
    on_policy=False,
):
    remote, _, _ = report_setup(monkeypatch)
    status, replica = terminal_fixture.TestTerminalMatrix._manager(0)
    config = status.config
    config.mode = "colocated"
    config.rollout = NS(n_generation=width, multi_turn_config=NS(enable=False))
    config.train.non_text, config.train.local_dataset = True, False
    config.train.train_batch_per_replica = 4
    policy = config.train.train_policy
    policy.type, policy.variant, policy.on_policy = "grpo", "grpo", on_policy
    policy.bypass_reward, policy.min_filter_prefix_tokens = False, 0
    policy.uncentralized_training = not centralized
    replica.atoms = {0: NS(global_rank=0, report_session_id="policy-session")}
    status.policy_init_done, status.total_steps = True, steps
    status.should_weight_sync_after_train_ack = Mock(return_value=False)
    status.try_trigger_data_fetch_and_training()
    assert status.current_step == 1
    remote.config, remote.policy_status_manager = config, status
    remote.data_fetcher = NS(
        get_batched_prompt=Mock(
            side_effect=lambda n, *a, **kw: (
                [
                    RLPayload(prompt_idx=i, prompt="p", reference_answer="a")
                    for i in range(n)
                ],
                False,
            )
        )
    )
    atom = next(
        a
        for a in remote.rollout_status_manager["source"].atoms.values()
        if a.global_rank == rank
    )
    atom.report_session_id, atom.rollout_reporter = "rollout-session", True

    local = object.__new__(ColocatedController)
    local.config, local.current_step = config, 1
    local.train_report_data, local._unreported_rollouts = {}, []
    local.policy = NS(data_queue=Queue())
    local.rollout = NS(
        parallel_dims=NS(
            mesh={"dp": NS(get_group=lambda: None, get_local_rank=lambda: 0)},
            cp_coord=(0, 1),
        )
    )
    monkeypatch.setattr(
        "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu",
        lambda value, **kw: [value],
    )
    calculator = object.__new__(LocalRewardCalculator)
    calculator.config, calculator.rl_algo = config, Rewards(reward_fn=None)
    pending = Queue()

    def enqueue(payloads, is_validation, step, **kwargs):
        pending.put((*calculator.compute_rewards(payloads, is_validation, step), False))

    client = ColocatedAPIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    client.set_controller(local)
    local.rollout.api_client = client
    client._registered_replica_name, client._registered_global_rank = "source", rank
    client._report_session_id, client.max_retries = atom.report_session_id, 2
    sent = []

    def http_post(url, *, json, **kwargs):
        request = RolloutRequest.model_validate(json)
        sent.append(request)
        result = post(request)
        if lost_reply and len(sent) == 1:
            raise requests.ConnectionError("injected lost settlement ACK")
        response = requests.Response()
        response.status_code = getattr(result, "status_code", 200)
        response._content = (
            result.body if hasattr(result, "body") else jsonlib.dumps(result).encode()
        )
        return response

    monkeypatch.setattr(requests, "post", http_post)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", lambda _: None)
    worker = object.__new__(ColocatedRolloutControlWorker)
    worker.config, worker.replica_name = config, "source"
    worker.global_rank, worker.current_weight_version, worker.should_report = (
        rank,
        0,
        True,
    )
    worker._prompt_queue = Queue()
    worker.inference_stream = worker.data_fetcher = None
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    worker.reward_dispatcher = NS(
        enqueue_rewards_cal=enqueue,
        dequeue_rewards_cal=lambda: pending.get()
        if not pending.empty()
        else (None, False, -1, True),
    )
    worker.data_packer = NS(get_rollout_output=lambda *values: (*values, None))
    worker.api_client = client
    return NS(
        remote=remote,
        status=status,
        replica=replica,
        local=local,
        worker=worker,
        client=client,
        sent=sent,
        pending=pending,
    )


def generate(state, *, mask=None, empty=False, malformed=False):
    payloads, _ = asyncio.run(state.remote._get_batched_prompt_impl(1))
    state.worker._prompt_queue.put(payloads)
    width = state.remote.config.rollout.n_generation
    results = (
        []
        if empty
        else [
            RolloutResult(
                completions=[] if malformed else [str(i) for i in range(width)],
                completion_trainable=mask,
            )
        ]
    )
    state.worker._call_rollout_generation = Mock(return_value=results)
    state.worker.one_step_generation()
    state.worker.report_rollouts()
    state.local.synchronize_rollouts()


def prepare(state, ready=True):
    return state.status.colocated_preparation(
        state.replica.name,
        state.local.current_step,
        state.status.total_steps,
        ready,
        "policy-session",
        0,
    )


def ack(state):
    state.status.train_ack(
        state.replica.name,
        state.local.current_step,
        state.status.total_steps,
        False,
        state.local.train_report_data.get(state.local.current_step, {}),
        Mock(),
    )


@pytest.mark.parametrize("centralized", [False, True])
@pytest.mark.parametrize(
    "outcome", ["healthy", "rejection", "partial", "failure", "malformed"]
)
def test_terminal_drop_then_success_reaches_zero(monkeypatch, centralized, outcome):
    state = system(
        monkeypatch, centralized=centralized, width=4 if outcome == "partial" else 2
    )
    if outcome == "rejection":
        generate(state, mask=[False, False])
    elif outcome in {"failure", "malformed"}:
        generate(state, empty=outcome == "failure", malformed=outcome == "malformed")
    for _ in range(2):
        generate(
            state, mask=[True, True, False, False] if outcome == "partial" else None
        )
    assert state.local.policy.data_queue.qsize() == 4
    assert state.status.samples_on_the_fly == 4
    assert state.status.training_dispatches[1].colocated_fetched_rollouts == 4
    assert len(state.sent) == (
        0 if outcome == "healthy" else 2 if outcome == "partial" else 1
    )
    assert all(
        not report.payloads and set(report.metrics) == {"discarded_samples"}
        for report in state.sent
    )
    assert prepare(state) == "train"
    for _ in range(4):
        state.local.policy.data_queue.get_nowait()
    ack(state)
    assert state.status.training_finished()
    assert state.status.training_dispatches[1].settled
    assert not state.status.dispatched_rollouts_by_step
    assert state.status.samples_on_the_fly == 0
    assert state.status.rollout_buffer.empty() and state.local.policy.data_queue.empty()
    assert state.worker._prompt_queue.empty() and state.pending.empty()
    assert not state.local._unreported_rollouts
    # Receipt replay remains harmless after preparation and the final ACK.
    for request in state.sent[-1:]:
        assert post(request) == {"message": "Colocated discard settled"}
    ack(state)
    assert state.status.samples_on_the_fly == 0


def test_lost_ack_from_nonzero_rank_settles_once(monkeypatch):
    state = system(monkeypatch, rank=1, lost_reply=True)
    generate(state, mask=[False, False])
    assert len(state.sent) == 2
    assert state.sent[0] == state.sent[1]
    assert state.sent[0].src_global_rank == 1
    assert state.status.samples_on_the_fly == 0
    assert state.status.training_dispatches[1].colocated_fetched_rollouts == 0
    assert not state.worker.api_client._report_failed


@pytest.mark.parametrize("centralized", [False, True])
def test_valid_queue_surplus_remains_charged(monkeypatch, centralized):
    state = system(monkeypatch, centralized=centralized)
    generate(state, mask=[False, False])
    for _ in range(3):
        generate(state)
    assert state.status.samples_on_the_fly == 6
    assert prepare(state) == "train"
    for _ in range(4):
        state.local.policy.data_queue.get_nowait()
    ack(state)
    assert state.local.policy.data_queue.qsize() == state.status.samples_on_the_fly == 2


def test_exhaustion_does_not_settle_acknowledged_drops_twice(monkeypatch):
    state = system(monkeypatch)
    generate(state, empty=True)
    generate(state)
    assert state.status.samples_on_the_fly == 2
    with patch.object(TrainingCompleteCommand, "trigger"):
        assert prepare(state, False) == "stop"
        assert prepare(state, False) == "stop"
    assert state.status.samples_on_the_fly == 0
    assert state.status.training_dispatches[1].cancelled
    assert post(state.sent[0]) == {"message": "Colocated discard settled"}
    assert state.status.samples_on_the_fly == 0


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1", 3])
def test_invalid_count_is_rejected_before_mutation(monkeypatch, count):
    state = system(monkeypatch)
    generate(state)
    report = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="rollout-session",
        report_sequence=0,
        payloads=[],
        metrics={"discarded_samples": count},
    )
    assert post(report).status_code == 409
    assert state.status.samples_on_the_fly == 2
    assert state.status.training_dispatches[1].colocated_fetched_rollouts == 2
    assert state.status.terminal_error is None


def consume_batch(state):
    assert prepare(state) == "train"
    values = [state.local.policy.data_queue.get_nowait() for _ in range(4)]
    ack(state)
    return values


@pytest.mark.parametrize("centralized", [False, True])
def test_stale_surplus_is_settled_before_the_next_on_policy_refill(
    monkeypatch, centralized
):
    state = system(monkeypatch, centralized=centralized, steps=2, on_policy=True)
    generate(state, empty=True)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    dispatch = state.status.training_dispatches[2]
    assert state.local.policy.data_queue.qsize() == 2
    assert dispatch.colocated_carried_rollouts == state.status.samples_on_the_fly == 2
    state.worker.current_weight_version = state.local.rollout.current_weight_version = 1
    state.local.advance_iteration()
    assert state.local.policy.data_queue.empty()
    assert dispatch.colocated_carried_rollouts == state.status.samples_on_the_fly == 0
    assert state.sent[-1].metrics == {"discarded_queued_samples": 2}
    assert post(state.sent[-1]) == {"message": "Colocated discard settled"}
    assert state.status.samples_on_the_fly == 0
    for _ in range(2):
        generate(state)
    consume_batch(state)
    assert state.status.training_finished()
    assert state.status.samples_on_the_fly == 0


@pytest.mark.parametrize("centralized", [False, True])
def test_off_policy_surplus_is_reused_in_its_existing_order(monkeypatch, centralized):
    state = system(monkeypatch, centralized=centralized, steps=2)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    retained = list(state.local.policy.data_queue.queue)
    state.local.advance_iteration()
    assert all(
        current is previous
        for current, previous in zip(state.local.policy.data_queue.queue, retained)
    )
    assert state.status.training_dispatches[2].colocated_carried_rollouts == 2
    generate(state)
    assert all(
        current is previous
        for current, previous in zip(consume_batch(state)[:2], retained)
    )
    assert state.status.samples_on_the_fly == 0
    assert not state.sent


@pytest.mark.parametrize("centralized", [False, True])
def test_exhaustion_retires_known_carry_without_a_second_queue_debit(
    monkeypatch, centralized
):
    state = system(monkeypatch, centralized=centralized, steps=2)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    state.local.advance_iteration()
    state.remote.data_fetcher.get_batched_prompt = Mock(return_value=([], True))
    assert asyncio.run(state.remote._get_batched_prompt_impl(1)) == ([], True)
    with patch.object(TrainingCompleteCommand, "trigger"):
        assert prepare(state, False) == "stop"
    assert state.status.current_step == 1
    assert state.status.samples_on_the_fly == 0
    assert state.status.training_dispatches[2].rollout_count == 2
    # agree_prepared_batch(stop) clears already-cancelled local inputs first.
    for _ in range(2):
        state.local.policy.data_queue.get_nowait()
    state.local.finalize_queued_rollouts()
    assert not state.sent


@pytest.mark.parametrize("centralized", [False, True])
def test_clean_exit_retires_unused_queue_once(monkeypatch, centralized):
    state = system(monkeypatch, centralized=centralized)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    gather = Mock(side_effect=lambda value, **kwargs: [value])
    monkeypatch.setattr(
        "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu", gather
    )
    assert state.status.samples_on_the_fly == 2
    state.local.finalize_queued_rollouts()
    state.local.finalize_queued_rollouts()
    assert state.local.policy.data_queue.empty()
    assert not state.local._unreported_rollouts
    assert state.status.samples_on_the_fly == 0
    assert len(state.sent) == 1
    assert state.sent[0].metrics == {"discarded_queued_samples": 2}
    gather.assert_called_once_with(None)


@pytest.mark.parametrize("rank", [0, 1])
def test_lost_queue_discard_ack_replays_the_same_rank_receipt(monkeypatch, rank):
    state = system(monkeypatch, rank=rank, lost_reply=True)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    state.local.finalize_queued_rollouts()
    assert state.status.samples_on_the_fly == 0
    assert len(state.sent) == 2 and state.sent[0] == state.sent[1]
    assert state.sent[0].src_global_rank == rank
    assert not state.client._report_failed
    assert not state.local._unreported_rollouts


def test_queue_discard_cannot_retire_a_fresh_preparing_reservation(monkeypatch):
    state = system(monkeypatch)
    generate(state)
    request = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="rollout-session",
        report_sequence=0,
        payloads=[],
        metrics={"discarded_queued_samples": 1},
    )
    assert post(request).status_code == 409
    assert state.status.samples_on_the_fly == 2
    assert state.status.training_dispatches[1].colocated_fetched_rollouts == 2


def test_queue_discard_cannot_retire_a_sealed_optimizer_batch(monkeypatch):
    state = system(monkeypatch)
    for _ in range(3):
        generate(state)
    assert prepare(state) == "train"
    request = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="rollout-session",
        report_sequence=0,
        payloads=[],
        metrics={"discarded_queued_samples": 3},
    )
    assert post(request).status_code == 409
    assert state.status.samples_on_the_fly == 6
    assert state.status.dispatched_rollouts_by_step[1] == 4


def test_only_unbound_surplus_can_retire_while_an_optimizer_ack_is_pending(monkeypatch):
    state = system(monkeypatch)
    for _ in range(3):
        generate(state)
    assert prepare(state) == "train"
    for _ in range(4):
        state.local.policy.data_queue.get_nowait()
    state.local.finalize_queued_rollouts()
    assert state.status.samples_on_the_fly == 4
    assert state.status.dispatched_rollouts_by_step[1] == 4
    ack(state)
    assert state.status.samples_on_the_fly == 0


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1", 3])
def test_invalid_terminal_queue_count_has_no_side_effect(monkeypatch, count):
    state = system(monkeypatch)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    request = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="rollout-session",
        report_sequence=0,
        payloads=[],
        metrics={"discarded_queued_samples": count},
    )
    assert post(request).status_code == 409
    assert state.status.samples_on_the_fly == 2
    assert state.status.terminal_error is None


def test_failed_queue_cleanup_cannot_be_retried_as_empty_success(monkeypatch):
    state = system(monkeypatch)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    report = Mock(side_effect=RuntimeError("lost queue receipt"))
    state.client.post_queued_rollout_discard = report
    with pytest.raises(RuntimeError, match="lost queue receipt"):
        state.local.finalize_queued_rollouts()
    assert state.local.policy.data_queue.empty()
    assert state.status.samples_on_the_fly == 2
    with pytest.raises(RuntimeError, match="lost queue receipt"):
        state.local.finalize_queued_rollouts()
    report.assert_called_once_with(2)


def test_empty_rank_waits_for_peer_queue_report_before_clean_exit(monkeypatch):
    state = system(monkeypatch)
    gather = Mock(return_value=[None, "peer queue report failed"])
    monkeypatch.setattr(
        "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu", gather
    )
    with pytest.raises(RuntimeError, match="peer queue report failed"):
        state.local.finalize_queued_rollouts()
    with pytest.raises(RuntimeError, match="peer queue report failed"):
        state.local.finalize_queued_rollouts()
    gather.assert_called_once_with(None)
    assert not state.sent


def test_failed_cleanup_agreement_cannot_be_reentered(monkeypatch):
    state = system(monkeypatch)
    gather = Mock(side_effect=RuntimeError("missing peer"))
    monkeypatch.setattr(
        "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu", gather
    )
    with pytest.raises(RuntimeError, match="missing peer"):
        state.local.finalize_queued_rollouts()
    with pytest.raises(RuntimeError, match="agreement failed"):
        state.local.finalize_queued_rollouts()
    assert gather.call_count == 1


@pytest.mark.parametrize("centralized", [False, True])
@pytest.mark.parametrize("fresh", [0, 1, 2])
def test_exhaustion_settles_carry_and_fresh_input_but_not_an_acked_drop_twice(
    monkeypatch, centralized, fresh
):
    state = system(monkeypatch, centralized=centralized, width=1, steps=2)
    for _ in range(5):
        generate(state)
    consume_batch(state)
    state.local.advance_iteration()
    assert state.status.training_dispatches[2].colocated_carried_rollouts == 1
    generate(state, empty=True)
    receipt = state.sent[-1]
    for _ in range(fresh):
        generate(state)
    state.remote.data_fetcher.get_batched_prompt = Mock(return_value=([], True))
    asyncio.run(state.remote._get_batched_prompt_impl(1))
    assert state.status.training_dispatches[2].colocated_fetched_rollouts == fresh
    assert state.local.policy.data_queue.qsize() == 1 + fresh < 4
    with patch.object(TrainingCompleteCommand, "trigger"):
        assert prepare(state, False) == "stop"
    assert state.status.samples_on_the_fly == 0
    assert state.status.training_dispatches[2].rollout_count == 1 + fresh
    assert post(receipt) == {"message": "Colocated discard settled"}
    assert state.status.samples_on_the_fly == 0


@pytest.mark.parametrize("payload", [False, True])
def test_queued_discard_cannot_hide_accepted_or_fresh_outcomes(monkeypatch, payload):
    state = system(monkeypatch)
    generate(state)
    request = RolloutRequest(
        src_replica_name="source",
        payloads=[RLPayload(completions=["a"], rewards=[1.0], advantages=[0.0])]
        if payload
        else [],
        metrics={"discarded_queued_samples": 1},
    )
    if not payload:
        request.metrics["discarded_samples"] = 1
    with pytest.raises(ValueError, match="cannot contain fresh outcomes"):
        state.client.post_rollout_completion(request)
    assert not state.sent
    assert state.status.samples_on_the_fly == state.local.policy.data_queue.qsize() == 2


def test_unreachable_queue_report_closes_the_source_and_clean_exit(monkeypatch):
    state = system(monkeypatch)
    for _ in range(3):
        generate(state)
    consume_batch(state)
    monkeypatch.setattr(
        requests, "post", Mock(side_effect=requests.ConnectionError("unreachable"))
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        state.local.finalize_queued_rollouts()
    assert state.client._report_failed
    assert state.status.samples_on_the_fly == 2
    with pytest.raises(RuntimeError, match="cannot continue"):
        state.client.post_rollout_completion(
            RolloutRequest(src_replica_name="source", payloads=[])
        )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        state.local.finalize_queued_rollouts()


def test_fresh_discard_cannot_retire_a_prepared_batch(monkeypatch):
    state = system(monkeypatch)
    generate(state)
    generate(state)
    assert prepare(state) == "train"
    report = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="rollout-session",
        report_sequence=0,
        payloads=[],
        metrics={"discarded_samples": 1},
    )
    assert post(report).status_code == 409
    assert state.status.samples_on_the_fly == 4
    assert state.status.terminal_error is None


def test_discard_cannot_release_reservations_outside_current_fetch(monkeypatch):
    state = system(monkeypatch)
    generate(state)
    state.status.samples_on_the_fly += 10
    report = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="rollout-session",
        report_sequence=0,
        payloads=[],
        metrics={"discarded_samples": 3},
    )
    assert post(report).status_code == 409
    assert state.status.samples_on_the_fly == 12
    assert state.status.training_dispatches[1].colocated_fetched_rollouts == 2


def test_explicit_retry_does_not_enqueue_selected_payload_twice(monkeypatch):
    state = system(monkeypatch, width=4)
    local_put = Mock(wraps=state.local.put_rollouts)
    state.local.put_rollouts = local_put
    generate(state, mask=[True, True, False, False])
    request = local_put.call_args.args[0]
    assert state.local.policy.data_queue.qsize() == 2
    assert state.client.post_rollout_completion(request)
    assert local_put.call_count == 1
    assert state.local.policy.data_queue.qsize() == state.status.samples_on_the_fly == 2
    assert state.sent[0] == state.sent[1]


def test_concurrent_explicit_replay_has_one_local_delivery(monkeypatch):
    state = system(monkeypatch, width=4)
    asyncio.run(state.remote._get_batched_prompt_impl(1))
    request = RolloutRequest(
        src_replica_name="source",
        payloads=[
            RLPayload(
                prompt_idx=0,
                completions=["a", "b"],
                rewards=[0.0, 1.0],
                advantages=[-1.0, 1.0],
            )
        ],
        metrics={"discarded_samples": 2},
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(state.client.post_rollout_completion, [request, request])
        )
    assert results == [True, True]
    assert len(state.sent) == 2 and state.sent[0] == state.sent[1]
    assert state.local.policy.data_queue.qsize() == state.status.samples_on_the_fly == 2


@pytest.mark.parametrize("failure", ["unreachable", "retired", "local"])
def test_failed_delivery_prevents_later_healthy_admission(monkeypatch, failure):
    state = system(monkeypatch)
    if failure == "unreachable":
        monkeypatch.setattr(
            requests, "post", Mock(side_effect=requests.ConnectionError("unreachable"))
        )
    elif failure == "retired":
        state.remote.rollout_status_manager.rollout_replicas.pop("source")
    else:
        state.local.put_rollouts = Mock(
            side_effect=RuntimeError("uncertain local delivery")
        )
    with pytest.raises(RuntimeError):
        generate(state, mask=[False, False])
    with pytest.raises(RuntimeError, match="cannot continue"):
        state.client.post_rollout_completion(
            RolloutRequest(src_replica_name="source", payloads=[])
        )
    assert state.local.policy.data_queue.empty()


@pytest.mark.parametrize("change", ["payload", "identity", "metadata", "source"])
def test_non_accounting_or_unfenced_report_has_no_side_effect(monkeypatch, change):
    state = system(monkeypatch)
    generate(state)
    report = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="rollout-session",
        report_sequence=0,
        payloads=[],
        metrics={"discarded_samples": 1},
    )
    if change == "payload":
        report.payloads = [
            RLPayload(completions=["a"], rewards=[1.0], advantages=[0.0])
        ]
    elif change == "identity":
        report.completion_identities = []
    elif change == "metadata":
        report.metrics["filtered_positive"] = 1
    else:
        report.report_session_id = "old-incarnation"
    assert post(report).status_code == 409
    assert state.status.samples_on_the_fly == 2
    assert state.status.terminal_error is None
