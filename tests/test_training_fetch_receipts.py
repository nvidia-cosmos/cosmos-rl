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
from cosmos_rl.colocated.api_client import ColocatedAPIClient
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.replica import Replica
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils import constant
from test_rollout_report_receipts import setup as report_setup


def setup(monkeypatch):
    controller, atom, _ = report_setup(monkeypatch)
    from cosmos_rl.dispatcher.reservations import ProducerReservations

    controller.rollout_status_manager[
        "source"
    ].producer_reservations = ProducerReservations()
    config = controller.config
    config.rollout.n_generation = 2
    policy = config.train.train_policy
    policy.on_policy, policy.allowed_outdated_steps = True, 0
    policy.outdated_rollout_fetch_batch_size = 0
    policy.max_inflight_steps, policy.max_retry_for_on_policy = 1, 0
    controller.policy_status_manager.samples_on_the_fly = 0
    controller.weight_version_to_prompt_num = {}
    controller.weight_version_to_replacement_prompt_num = {}
    controller._soft_throttle_engaged_since = None
    controller._soft_throttle_last_log_ts = 0
    values = iter([7, 7, 2])
    controller.data_fetcher = SimpleNamespace(
        get_batched_prompt=Mock(
            side_effect=lambda n, *args, **kwargs: (
                [RLPayload(prompt_idx=next(values)) for _ in range(n)],
                False,
            )
        )
    )
    identity = dict(
        src_replica_name="source",
        src_global_rank=0,
        fetch_session_id="session",
        fetch_sequence=0,
    )
    return controller, atom, identity


def fetch(n, identity):
    return asyncio.run(web.get_batched_prompt(n, **identity))


def test_lost_reply_replays_prompts_without_sampler_or_reservation_mutation(
    monkeypatch,
):
    controller, atom, identity = setup(monkeypatch)
    first = fetch(1, identity)
    assert first == fetch(1, identity)
    assert [p["prompt_idx"] for p in first["payloads_list"]] == [7]
    controller.data_fetcher.get_batched_prompt.assert_called_once()
    assert controller.policy_status_manager.samples_on_the_fly == 2
    assert controller.weight_version_to_prompt_num == {0: 1}
    first["payloads_list"][0]["prompt_idx"] = 999
    assert fetch(1, identity)["payloads_list"][0]["prompt_idx"] == 7
    assert atom.rollout_fetch_receipt.sequence == 0


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"fetch_session_id": "old-source"}, 409),
        ({"fetch_sequence": None}, 409),
        ({"fetch_sequence": 2}, 409),
        ({"src_global_rank": 99}, 409),
        ({"src_replica_name": "retired"}, 410),
    ],
)
def test_unfenced_fetch_does_not_advance_sampler(monkeypatch, changes, status):
    controller, _, identity = setup(monkeypatch)
    identity.update(changes)
    assert fetch(1, identity).status_code == status
    controller.data_fetcher.get_batched_prompt.assert_not_called()
    assert controller.policy_status_manager.samples_on_the_fly == 0


def test_changed_expired_and_replaced_sources_cannot_refetch(monkeypatch):
    controller, atom, identity = setup(monkeypatch)
    fetch(1, identity)
    assert fetch(2, identity).status_code == 409
    identity["fetch_sequence"] = 1
    assert fetch(1, identity) == {"payloads_list": [], "is_end": False}
    identity["fetch_sequence"] = 0
    assert fetch(1, identity).status_code == 409
    assert atom.rollout_fetch_receipt.sequence == 1
    fresh = type(atom)(
        0,
        "localhost",
        "fixture",
        None,
        [0] * 4,
        [1] * 4,
        "source",
        report_session_id="new-session",
    )
    controller.rollout_status_manager.rollout_replicas["source"] = Replica(
        "source", Role.ROLLOUT, [fresh]
    )
    identity["fetch_sequence"] = 1
    assert fetch(1, identity).status_code == 409
    controller.data_fetcher.get_batched_prompt.assert_called_once()


def test_sampler_failure_is_terminal_and_not_retried(monkeypatch):
    controller, atom, identity = setup(monkeypatch)
    controller.data_fetcher.get_batched_prompt.side_effect = RuntimeError(
        "advanced then failed"
    )
    with pytest.raises(RuntimeError, match="advanced"):
        fetch(1, identity)
    assert fetch(1, identity).status_code == 503
    controller.data_fetcher.get_batched_prompt.assert_called_once()
    assert atom.rollout_fetch_receipt.failed
    assert isinstance(controller.policy_status_manager.terminal_error, RuntimeError)


def test_ended_replica_gets_replayable_empty_terminal_fetch(monkeypatch):
    controller, _, identity = setup(monkeypatch)
    controller.rollout_status_manager["source"].status.ended = True
    assert (
        fetch(1, identity)
        == fetch(1, identity)
        == {"payloads_list": [], "is_end": True}
    )
    controller.data_fetcher.get_batched_prompt.assert_not_called()


def test_source_retired_while_fetch_waits_cannot_advance_sampler(monkeypatch):
    controller, _, identity = setup(monkeypatch)

    async def run():
        async with controller.life_cycle_lock:
            task = asyncio.create_task(web.get_batched_prompt(1, **identity))
            await asyncio.sleep(0)
            assert not task.done()
            controller.rollout_status_manager.rollout_replicas.pop("source")
        assert (await asyncio.wait_for(task, 1)).status_code == 410

    asyncio.run(run())
    controller.data_fetcher.get_batched_prompt.assert_not_called()


@pytest.mark.parametrize("client_type", [APIClient, ColocatedAPIClient])
def test_client_lost_reply_keeps_request_identity_and_original_reservation(
    monkeypatch, client_type
):
    controller, _, _ = setup(monkeypatch)
    client = client_type(Role.ROLLOUT, remote_ips=["localhost"], remote_port=12345)
    client._registered_replica_name, client._registered_global_rank = "source", 0
    client._report_session_id = "session"
    attempts = []

    def get(url, *, params, timeout):
        assert timeout == constant.COSMOS_CONTROL_HTTP_TIMEOUT
        attempts.append(dict(params))
        result = asyncio.run(web.get_batched_prompt(**params))
        if len(attempts) == 1:
            raise requests.ConnectionError("reply lost after sampler advance")
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(result).encode()
        return response

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", lambda _: None)
    payloads, end = client.get_next_prompt(1)
    assert not end and payloads[0]["prompt_idx"] == 7
    assert attempts[0] == attempts[1] and client._fetch_sequence == 1
    assert controller.policy_status_manager.samples_on_the_fly == 2
    controller.data_fetcher.get_batched_prompt.assert_called_once()


def test_client_exhausted_attempts_are_terminal_not_empty_work(monkeypatch):
    client = APIClient(Role.ROLLOUT, remote_ips=["localhost"], remote_port=12345)
    client._registered_replica_name, client._registered_global_rank = "source", 0
    send = Mock(side_effect=requests.Timeout("unreachable"))
    monkeypatch.setattr(requests, "get", send)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="bounded HTTP"):
        client.get_next_prompt(1)
    with pytest.raises(RuntimeError, match="cannot continue"):
        client.get_next_prompt(1)
    assert send.call_count == 3 and client._fetch_sequence == 0
    assert all(
        call.kwargs["timeout"] == constant.COSMOS_CONTROL_HTTP_TIMEOUT
        for call in send.call_args_list
    )


def test_rank_zero_training_fetch_error_uses_existing_prompt_broadcast(monkeypatch):
    import threading
    from queue import Queue
    from cosmos_rl.rollout.worker import rollout_control

    forwarded = []
    monkeypatch.setattr(
        rollout_control.dist_utils,
        "broadcast_object_cpu",
        lambda value: forwarded.append(value) or value,
    )
    worker = SimpleNamespace(
        global_rank=0,
        _prompt_fetch_lock=threading.Lock(),
        api_client=SimpleNamespace(
            get_next_prompt=Mock(side_effect=RuntimeError("fetch rejected"))
        ),
        parallel_dims=SimpleNamespace(mesh={"dp": SimpleNamespace(size=lambda: 1)}),
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="Training prompt fetch failed"):
        rollout_control.DisaggregatedRolloutControlWorker.request_new_prompts(
            worker, 1, Queue()
        )
    assert forwarded == [((None, False), "RuntimeError: fetch rejected")]
    worker.global_rank = 1
    monkeypatch.setattr(
        rollout_control.dist_utils, "broadcast_object_cpu", lambda value: forwarded[0]
    )
    with pytest.raises(RuntimeError, match="Training prompt fetch failed"):
        rollout_control.DisaggregatedRolloutControlWorker.request_new_prompts(
            worker, 1, Queue()
        )
    worker.api_client.get_next_prompt.assert_called_once()


def test_background_fetch_failure_is_not_clean_exhaustion(monkeypatch):
    from test_rollout_prefetch_loop_integration import _make_worker_for_prefetch

    worker = _make_worker_for_prefetch(api_responses=[], submit_setup_recorder=[])
    worker.api_client = APIClient(
        "ROLLOUT", remote_ips=["localhost"], remote_port=12345
    )
    worker.api_client._registered_replica_name = "source"
    worker.api_client._registered_global_rank = 0
    send = Mock(side_effect=requests.Timeout("unreachable"))
    monkeypatch.setattr(requests, "get", send)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="source cannot continue"):
        worker._prefetch_loop()
    assert send.call_count == 3
    assert not worker.state.prompt_fetch_end()
    with pytest.raises(RuntimeError, match="source cannot continue"):
        worker._raise_training_fetch_error()


@pytest.mark.parametrize("already_stopped", [False, True])
def test_consumer_observes_failed_fetch_without_an_end_report(
    monkeypatch, already_stopped
):
    import threading
    from cosmos_rl.rollout.worker import rollout_control

    worker = object.__new__(rollout_control.DisaggregatedRolloutControlWorker)
    worker.api_client = SimpleNamespace(training_fetch_failed=True)
    worker.config = SimpleNamespace(rollout=SimpleNamespace(prefetch_rollout=False))
    worker.shutdown_signal = threading.Event()
    if already_stopped:
        worker.shutdown_signal.set()
    worker.consume_command = Mock(
        side_effect=AssertionError("must observe fetch failure first")
    )
    monkeypatch.setattr(
        rollout_control,
        "get_async_r2r_sync_mode",
        lambda _: rollout_control.AsyncR2RSyncMode.DISABLED,
    )
    with pytest.raises(RuntimeError, match="Training prompt fetch failed"):
        worker._main_loop_impl()
    worker.consume_command.assert_not_called()


def test_delegating_report_source_also_fences_executor_fetches():
    client = APIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    client._registered_replica_name, client._registered_global_rank = "source", 0
    source = client.delegate_rollout_reporting()
    with pytest.raises(RuntimeError, match="cannot continue"):
        client.get_next_prompt(1)
    wrapper = APIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    wrapper.adopt_rollout_reporting(source)
    assert not wrapper.training_fetch_failed
