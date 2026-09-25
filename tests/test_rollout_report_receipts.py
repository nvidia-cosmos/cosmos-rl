# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from cosmos_rl.dispatcher import run_web_panel as web
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import RLPayload, TrainingCompletionIdentity
from cosmos_rl.dispatcher.protocol import RegisterRequest, Role, RolloutRequest
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.status import RolloutStatusManager
from cosmos_rl.utils import constant
import test_terminal_drain_protocol as fixture


def setup(monkeypatch):
    policies, _ = fixture.TestTerminalMatrix._manager(0)
    policies.config.rollout = SimpleNamespace(include_stop_str_in_output=False)
    policies.config.train.sync_weight_interval = 1
    policy = policies.config.train.train_policy
    policy.type, policy.variant, policy.on_policy = "grpo", "grpo", False
    policy.allowed_outdated_steps, policy.rollout_as_token_ids = 10, True
    policies.samples_on_the_fly = 10
    policies.try_trigger_data_fetch_and_training = Mock()
    workers = RolloutStatusManager()
    replica = Replica(
        "source",
        Role.ROLLOUT,
        [fixture._rollout_atom("source", 0, 0), fixture._rollout_atom("source", 1, 1)],
    )
    workers.rollout_replicas = {"source": replica}
    atom = next(atom for atom in replica.atoms.values() if atom.global_rank == 0)
    atom.report_session_id = "session"
    controller = object.__new__(Controller)
    controller.config = policies.config
    controller.policy_status_manager, controller.rollout_status_manager = (
        policies,
        workers,
    )
    controller.completion_admission = None
    controller.life_cycle_lock = asyncio.Lock()
    controller.stat_n_samples = controller.stat_completion_tokens_count = 0
    controller.begin_time = None
    monkeypatch.setattr(web, "controller", controller)
    request = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        report_session_id="session",
        report_sequence=0,
        payloads=[
            RLPayload(
                prompt_idx=4,
                completions=["reply"],
                rewards=[1.0],
                advantages=[0.0],
                completion_token_ids=[[[3]]],
                training_work_id=replica.producer_reservations.issue(0, 1),
                training_completion_slots=[0],
            )
        ],
    )
    return controller, atom, request


def post(request):
    # Real HTTP reconstructs a request for each retry; extraction may populate
    # optional payload fields without changing the caller's original wire body.
    return asyncio.run(
        web.put_rollout_group(RolloutRequest.model_validate(request.model_dump()))
    )


def test_lost_reply_does_not_repeat_legacy_admission(monkeypatch):
    controller, atom, request = setup(monkeypatch)
    assert post(request) == post(request) == {"message": "Rollout put"}
    policies = controller.policy_status_manager
    assert policies.rollout_buffer.qsize() == controller.stat_n_samples == 1
    assert policies.samples_on_the_fly == 10
    assert atom.rollout_report_receipt.sequence == 0


def test_lost_dapo_report_reply_does_not_repeat_filtered_settlement(monkeypatch):
    controller, _, request = setup(monkeypatch)
    controller.config.train.train_policy.variant = "dapo"
    request.metrics = {"sampled": 2, "filtered_positive": 1}
    rejected = controller.rollout_status_manager["source"].producer_reservations.issue(
        0, 1
    )
    request.training_rejections = [TrainingCompletionIdentity(work_id=rejected, slot=0)]
    for _ in range(2):
        assert post(request) == {"message": "Rollout put"}
    policies = controller.policy_status_manager
    assert policies.samples_on_the_fly == 9
    assert policies.remain_samples_num == 19
    assert policies.filter_records["sampled"] == 2
    assert controller.stat_n_samples == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("report_session_id", "other"),
        ("src_global_rank", 999),
        ("report_sequence", 2),
        ("report_sequence", None),
    ],
)
def test_invalid_source_or_sequence_rejected_before_mutation(monkeypatch, field, value):
    controller, atom, request = setup(monkeypatch)
    setattr(request, field, value)
    assert post(request).status_code == 409
    assert controller.stat_n_samples == 0
    assert atom.rollout_report_receipt.sequence == -1


def test_changed_retry_is_rejected_without_repeating_side_effects(monkeypatch):
    controller, _, request = setup(monkeypatch)
    assert post(request) == {"message": "Rollout put"}
    request.payloads[0].completions = ["different"]
    assert post(request).status_code == 409
    assert controller.stat_n_samples == 1


def test_end_receipt_retries_but_cannot_accept_later_work_from_that_rank(monkeypatch):
    controller, atom, request = setup(monkeypatch)
    request.payloads, request.is_end = [], True
    expected = {"message": "Rollout end signal received"}
    assert post(request) == post(request) == expected
    assert atom.rollout_reports_ended
    assert not controller.rollout_status_manager["source"].status.ended
    request.report_sequence, request.is_end = 1, False
    assert post(request).status_code == 409
    assert controller.stat_n_samples == 0


def test_retry_waits_for_in_progress_original_then_reuses_its_receipt(monkeypatch):
    controller, _, request = setup(monkeypatch)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        original = web._apply_rollout_group

        async def paused(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(web, "_apply_rollout_group", paused)
        wire = request.model_dump()
        first = asyncio.create_task(
            web.put_rollout_group(RolloutRequest.model_validate(wire))
        )
        await asyncio.wait_for(entered.wait(), 1)
        duplicate = asyncio.create_task(
            web.put_rollout_group(RolloutRequest.model_validate(wire))
        )
        await asyncio.sleep(0)
        assert not duplicate.done()
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, duplicate), 1)
        assert results == [{"message": "Rollout put"}] * 2

    asyncio.run(run())
    assert controller.stat_n_samples == 1


def test_retired_source_cannot_re_admit_even_its_last_report(monkeypatch):
    controller, _, request = setup(monkeypatch)
    assert post(request) == {"message": "Rollout put"}
    controller.rollout_status_manager.rollout_replicas.pop("source")
    assert post(request).status_code == 410
    assert controller.stat_n_samples == 1


@pytest.mark.parametrize("replace", [False, True])
def test_departure_while_waiting_for_lifecycle_lock_cannot_admit(monkeypatch, replace):
    controller, _, request = setup(monkeypatch)

    async def run():
        async with controller.life_cycle_lock:
            pending = asyncio.create_task(web.put_rollout_group(request))
            await asyncio.sleep(0)
            assert not pending.done()
            controller.rollout_status_manager.rollout_replicas.pop("source")
            if replace:
                new_atom = fixture._rollout_atom("source", 0, 0)
                new_atom.report_session_id = "new-session"
                controller.rollout_status_manager.rollout_replicas["source"] = Replica(
                    "source", Role.ROLLOUT, [new_atom]
                )
        assert (await asyncio.wait_for(pending, 1)).status_code == 410

    asyncio.run(run())
    assert controller.stat_n_samples == 0
    assert controller.policy_status_manager.terminal_error is None


def test_old_source_receipt_cannot_enter_replacement_with_same_name(monkeypatch):
    controller, _, request = setup(monkeypatch)
    assert post(request) == {"message": "Rollout put"}
    new_atom = fixture._rollout_atom("source", 0, 0)
    new_atom.report_session_id = "new-session"
    controller.rollout_status_manager.rollout_replicas["source"] = Replica(
        "source", Role.ROLLOUT, [new_atom]
    )
    assert post(request).status_code == 409
    assert controller.stat_n_samples == 1
    assert new_atom.rollout_report_receipt.sequence == -1


def test_nonreporting_rank_rejected_even_with_valid_session(monkeypatch):
    controller, atom, request = setup(monkeypatch)
    atom.rollout_reporter = False
    assert post(request).status_code == 409
    assert controller.stat_n_samples == 0


def test_registered_wrapper_reporter_seals_whole_replica_end(monkeypatch):
    controller, atom, request = setup(monkeypatch)
    for peer in controller.rollout_status_manager["source"].atoms.values():
        peer.rollout_reporter = peer is atom
    controller.policy_status_manager.on_rollout_is_end = Mock()
    request.payloads, request.is_end = [], True
    request.stays_command_participant = True
    assert post(request) == post(request) == {"message": "Rollout end signal received"}
    assert controller.rollout_status_manager["source"].status.ended
    controller.policy_status_manager.on_rollout_is_end.assert_called_once()


def test_malformed_shape_is_cached_rejection_without_poisoning_accounting(monkeypatch):
    controller, atom, request = setup(monkeypatch)
    request.payloads[0].advantages = []
    assert post(request).status_code == post(request).status_code == 409
    assert controller.stat_n_samples == 0
    assert not atom.rollout_report_receipt.failed
    assert controller.policy_status_manager.terminal_error is None


def test_uncertain_mutation_poisoned_and_observed_by_controller_monitor(monkeypatch):
    controller, atom, request = setup(monkeypatch)

    async def partial(*args, **kwargs):
        controller.stat_n_samples += 1
        raise RuntimeError("injected partial settlement")

    monkeypatch.setattr(web, "_apply_rollout_group", partial)
    with pytest.raises(RuntimeError, match="partial"):
        post(request)
    assert atom.rollout_report_receipt.failed
    assert controller.policy_status_manager.terminal_error is not None
    assert post(request).status_code == 503
    assert controller.stat_n_samples == 1


def http_response(status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({"ok": True}).encode()
    return response


def test_client_lost_reply_and_explicit_retry_keep_identity(monkeypatch):
    client = APIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    client._registered_replica_name, client._registered_global_rank = "source", 0
    client.max_retries = 2
    send = Mock(
        side_effect=[
            requests.ConnectionError("lost reply"),
            http_response(),
            http_response(),
        ]
    )
    monkeypatch.setattr(requests, "post", send)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", lambda _: None)
    request = RolloutRequest(src_replica_name="source", payloads=[])
    assert client.post_rollout_completion(request)
    assert client.post_rollout_completion(request)
    assert client._report_sequence == 1
    assert all(
        call.kwargs == send.call_args_list[0].kwargs for call in send.call_args_list
    )
    assert send.call_args.kwargs["timeout"] == constant.COSMOS_CONTROL_HTTP_TIMEOUT
    assert send.call_args.kwargs["json"]["src_global_rank"] == 0


def test_client_rejection_is_not_silent_report_loss(monkeypatch):
    client = APIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    client._registered_replica_name, client._registered_global_rank = "source", 0
    send = Mock(return_value=http_response(409))
    monkeypatch.setattr(requests, "post", send)
    request = RolloutRequest(src_replica_name="source", payloads=[])
    with pytest.raises(RuntimeError, match="bounded HTTP"):
        client.post_rollout_completion(request)
    with pytest.raises(RuntimeError, match="cannot continue"):
        client.post_rollout_completion(request)
    assert send.call_count == 1 and client._report_sequence == 0


def test_client_unreachable_report_has_three_bounded_attempts(monkeypatch):
    client = APIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    client._registered_replica_name, client._registered_global_rank = "source", 0
    send = Mock(side_effect=requests.Timeout("unreachable"))
    monkeypatch.setattr(requests, "post", send)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="bounded HTTP"):
        client.post_rollout_completion(
            RolloutRequest(src_replica_name="source", payloads=[])
        )
    assert send.call_count == 3
    assert all(
        call.kwargs["timeout"] == constant.COSMOS_CONTROL_HTTP_TIMEOUT
        for call in send.call_args_list
    )
    assert client._report_failed and client._report_sequence == 0


@pytest.mark.parametrize("role", [Role.ROLLOUT, Role.POLICY])
def test_registration_retries_keep_session_and_changed_incarnation_rejected(
    monkeypatch, role
):
    client = APIClient(role, remote_ips=["localhost"], remote_port=12345)
    send = Mock(return_value=http_response())
    monkeypatch.setattr(requests, "post", send)
    arguments = (
        "source",
        role,
        ["pp", "dp_shard", "cp", "tp"],
        [0] * 4,
        [1] * 4,
        0,
        "127.0.0.1",
        "fixture",
    )
    client.register(*arguments)
    client.register(*arguments)
    assert send.call_args_list[0].kwargs == send.call_args_list[1].kwargs
    wire = RegisterRequest.model_validate(send.call_args.kwargs["json"])
    atom = Atom.from_register_request(wire)
    assert atom.report_session_id == client._report_session_id
    replica = Replica("source", role, [atom])
    assert not replica.arrive(Atom.from_register_request(wire))
    wire.report_session_id = "another-process"
    with pytest.raises(ValueError, match="identity"):
        replica.arrive(Atom.from_register_request(wire))


def test_rollout_registration_rejects_unfenced_workers_before_mesh_changes():
    controller = object.__new__(Controller)
    controller.life_cycle_lock = asyncio.Lock()
    controller.policy_status_manager = SimpleNamespace(stop_reason=None)
    controller.rollout_status_manager = Mock()
    controller.config = object()
    atom = Atom(0, "127.0.0.1", "fixture", None, [0] * 4, [1] * 4, "source")
    with pytest.raises(ValueError, match="receipt-aware"):
        asyncio.run(controller.register(atom, Role.ROLLOUT))
    controller.rollout_status_manager.register.assert_not_called()
    atom.report_session_id = "session"
    asyncio.run(controller.register(atom, Role.ROLLOUT))
    controller.rollout_status_manager.register.assert_called_once()


def test_trt_wrapper_handoff_keeps_one_registered_report_stream(monkeypatch):
    from test_trt_validation_delivery import lifecycle_method

    send = Mock(return_value=http_response())
    monkeypatch.setattr(requests, "post", send)
    executor = APIClient(Role.ROLLOUT, remote_ips=["localhost"], remote_port=12345)
    executor.register(
        "source",
        Role.ROLLOUT,
        ["pp", "dp_shard", "cp", "tp"],
        [0] * 4,
        [1] * 4,
        0,
        "127.0.0.1",
        "fixture",
        rollout_reporter=True,
    )
    source = executor.delegate_rollout_reporting()
    with pytest.raises(ValueError, match="unused"):
        executor.delegate_rollout_reporting()
    with pytest.raises(RuntimeError, match="cannot continue"):
        executor.post_rollout_completion(
            RolloutRequest(src_replica_name="source", payloads=[])
        )
    wrapper = SimpleNamespace(
        api_client=APIClient(Role.ROLLOUT, remote_ips=["localhost"], remote_port=12345)
    )
    lifecycle_method("bind_report_source")(wrapper, source)
    assert wrapper._is_registered and wrapper.replica_name == "source"
    request = RolloutRequest(
        src_replica_name=wrapper.replica_name, is_end=True, payloads=[]
    )
    assert wrapper.api_client.post_rollout_completion(request)
    assert send.call_count == 2  # One registration, one wrapper report.
    assert (
        send.call_args.kwargs["json"]["report_session_id"]
        == source["report_session_id"]
    )
    assert send.call_args.kwargs["json"]["src_global_rank"] == 0
    with pytest.raises(ValueError, match="already bound"):
        wrapper.api_client.adopt_rollout_reporting(source)


@pytest.mark.parametrize(
    "source",
    [
        "old-rankless-protocol",
        {},
        {
            "replica_name": "source",
            "global_rank": 0,
            "report_session_id": "session",
            "controller_execution_id": "old-controller",
        },
    ],
)
def test_wrapper_rejects_unfenced_or_stale_handoff(source):
    client = APIClient(Role.ROLLOUT, remote_ips=["localhost"], remote_port=12345)
    with pytest.raises(ValueError, match="Invalid"):
        client.adopt_rollout_reporting(source)
    assert client._registered_replica_name is None
