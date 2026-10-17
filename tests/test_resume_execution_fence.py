# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.protocol import RolloutRequest


def test_setup_replaces_saved_execution_id_and_disables_legacy_fence(monkeypatch):
    from cosmos_rl.dispatcher.controller import Controller, ParallelizedShardMapper

    # Stop after runtime config initialization, before resources are allocated.
    monkeypatch.setattr(
        ParallelizedShardMapper,
        "get_instance",
        Mock(side_effect=RuntimeError("stop before allocation")),
    )
    config = SimpleNamespace(
        controller_execution_id="saved-attempt",
        train=SimpleNamespace(train_policy=SimpleNamespace(type="grpo")),
        rollout=SimpleNamespace(completion_admission=False),
    )
    identities = []
    for adapter in (object(), object(), None):
        controller = Controller.__new__(Controller)
        controller.config = None
        with pytest.raises(RuntimeError, match="stop before allocation"):
            controller.setup(
                config,
                redis_port=0,
                redis_logfile_path="unused",
                resume_adapter=adapter,
            )
        identities.append(config.controller_execution_id)
    assert identities[0] != "saved-attempt"
    assert identities[1] != identities[0]
    assert identities[2] is None


@pytest.mark.parametrize("token", [None, "previous-execution"])
@pytest.mark.parametrize("is_end", [False, True])
def test_old_reports_rejected_before_any_state_mutation(monkeypatch, token, is_end):
    from cosmos_rl.dispatcher import run_web_panel as panel

    controller = Mock(config=SimpleNamespace(controller_execution_id="new-execution"))
    monkeypatch.setattr(panel, "controller", controller)
    extract = Mock(side_effect=AssertionError("must not extract stale payloads"))
    monkeypatch.setattr(panel, "extract_rollouts", extract)
    result = asyncio.run(
        panel.put_rollout_group(
            RolloutRequest(
                controller_execution_id=token,
                src_replica_name="old",
                payloads=[],
                metrics={"discarded_samples": 9},
                is_end=is_end,
            )
        )
    )
    assert result.status_code == 410
    assert controller.mock_calls == []
    extract.assert_not_called()


@pytest.mark.parametrize("token", [None, "new-execution"])
def test_disabled_fence_and_current_execution_keep_normal_path(monkeypatch, token):
    from cosmos_rl.dispatcher import run_web_panel as panel
    from rollout_receipt_fixture import install_report_source

    controller = Mock(config=SimpleNamespace(controller_execution_id=token))
    controller.rollout_status_manager.rollout_end.return_value = False
    rollout_end = controller.rollout_status_manager.rollout_end
    monkeypatch.setattr(panel, "controller", controller)
    request = install_report_source(
        controller,
        RolloutRequest(
            controller_execution_id=token,
            src_replica_name="worker",
            payloads=[],
            is_end=True,
        ),
    )
    result = asyncio.run(panel.put_rollout_group(request))
    assert result == {"message": "Rollout end signal received"}
    rollout_end.assert_called_once()


def test_client_sends_pinned_attempt_and_does_not_retry_stale_result(monkeypatch):
    client = APIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    client._registered_replica_name, client._registered_global_rank = "worker", 0
    client.controller_execution_id = "old-execution"
    client.get_alternative_urls = Mock(return_value=["http://controller/rollouts"])
    client.max_retries = 3
    response = requests.Response()
    response.status_code = 410
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    sleep = Mock(side_effect=AssertionError("must not retry stale reports"))
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", sleep)
    request = RolloutRequest(
        controller_execution_id="spoofed", src_replica_name="worker", payloads=[]
    )
    assert not client.post_rollout_completion(request)
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["controller_execution_id"] == "old-execution"
    assert request.controller_execution_id == "spoofed"  # no caller mutation
    sleep.assert_not_called()


@pytest.mark.parametrize("colocated", [False, True])
@pytest.mark.parametrize("role", ["POLICY", "ROLLOUT"])
@pytest.mark.parametrize("execution_id", [None, "resumed-attempt"])
def test_worker_init_comm_propagates_execution_fence(
    monkeypatch, colocated, role, execution_id
):
    from cosmos_rl.comm import base
    from cosmos_rl.colocated.api_client import ColocatedAPIClient

    monkeypatch.setenv("COSMOS_CONTROLLER_HOST", "localhost:12345")
    monkeypatch.setattr(base.dist_utils, "broadcast_object_cpu", lambda value: value)
    worker = SimpleNamespace(
        role=role,
        global_rank=0,
        config=SimpleNamespace(
            controller_execution_id=execution_id,
            train=SimpleNamespace(train_policy=SimpleNamespace(type="grpo")),
        ),
        init_redis=Mock(),
        register_to_controller=Mock(),
    )
    if colocated:
        worker.colocated = True
    base.CommMixin.init_comm(worker)
    assert isinstance(worker.api_client, ColocatedAPIClient) is colocated
    assert worker.api_client.controller_execution_id == execution_id
    worker.init_redis.assert_called_once()
    worker.register_to_controller.assert_called_once()


def test_colocated_worker_fetch_uses_resumed_controller_fence(monkeypatch):
    import json
    from cosmos_rl.comm import base
    from cosmos_rl.dispatcher import run_web_panel as panel
    from test_training_fetch_receipts import setup

    controller, _, _ = setup(monkeypatch)
    controller.config.controller_execution_id = "resumed-attempt"
    controller.config.mode = "colocated"
    monkeypatch.setenv("COSMOS_CONTROLLER_HOST", "localhost:12345")
    monkeypatch.setattr(base.dist_utils, "broadcast_object_cpu", lambda value: value)
    worker = SimpleNamespace(
        colocated=True,
        role="ROLLOUT",
        global_rank=0,
        config=controller.config,
        init_redis=Mock(),
        register_to_controller=Mock(),
    )
    base.CommMixin.init_comm(worker)
    client = worker.api_client
    client._registered_replica_name, client._registered_global_rank = "source", 0
    client._report_session_id = "session"
    outcomes = []

    def get(url, *, params, timeout):
        result = asyncio.run(panel.get_batched_prompt(**params))
        response = requests.Response()
        if hasattr(result, "status_code"):
            response.status_code, response._content = result.status_code, result.body
        else:
            response.status_code, response._content = 200, json.dumps(result).encode()
        outcomes.append((params["controller_execution_id"], response.status_code))
        return response

    monkeypatch.setattr(requests, "get", get)
    payloads, ended = client.get_next_prompt(1)
    assert not ended and payloads[0]["prompt_idx"] == 7
    assert outcomes == [("resumed-attempt", 200)]
    controller.data_fetcher.get_batched_prompt.assert_called_once()
