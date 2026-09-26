# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils import constant
from cosmos_rl.utils.network_util import make_request_with_retry


def response(status):
    result = requests.Response()
    result.status_code = status
    result.url = "http://controller/report"
    return result


def test_rollout_nccl_error_is_acknowledged_without_reaping_other_replicas():
    controller = Controller.__new__(Controller)
    controller.policy_status_manager = {}
    rollout = SimpleNamespace(status="healthy", weight_version=7)
    controller.rollout_status_manager = {"rollout": rollout}
    asyncio.run(controller.set_replica_ncclerror("rollout", "injected"))
    assert controller.rollout_status_manager == {"rollout": rollout}
    assert rollout.status == "healthy" and rollout.weight_version == 7


@pytest.mark.parametrize(
    "failure", [requests.Timeout("hung server"), response(500), response(404)]
)
def test_error_reporting_is_one_bounded_best_effort_request(monkeypatch, failure):
    post = Mock(
        side_effect=failure if isinstance(failure, Exception) else None,
        return_value=failure,
    )
    monkeypatch.setattr("cosmos_rl.dispatcher.api.client.requests.post", post)
    client = APIClient(Role.ROLLOUT, ["127.0.0.1", "127.0.0.2"], 8123)
    client.post_nccl_comm_error("rollout", RuntimeError("injected"))
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == constant.COSMOS_CONTROL_HTTP_TIMEOUT


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_deterministic_http_errors_are_not_retried(monkeypatch, status):
    sleep = Mock()
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", sleep)
    request = Mock(return_value=response(status))
    with pytest.raises(requests.HTTPError):
        make_request_with_retry(request, ["http://controller"], max_retries=60)
    assert request.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
def test_transient_http_errors_retain_existing_retry_behavior(monkeypatch, status):
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", lambda _: None)
    request = Mock(side_effect=[response(status), response(200)])
    assert (
        make_request_with_retry(
            request, ["http://controller"], max_retries=2
        ).status_code
        == 200
    )
    assert request.call_count == 2
