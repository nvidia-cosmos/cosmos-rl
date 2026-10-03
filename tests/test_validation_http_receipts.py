# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest
import requests

from cosmos_rl.colocated.api_client import ColocatedAPIClient
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.command import RolloutToRolloutBroadcastCommand
from cosmos_rl.rollout.validation import ValidationSession, validation_round_for_command
from cosmos_rl.utils import constant


def response(status=200, body=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body or {}).encode()
    return result


@pytest.mark.parametrize("client_type", [APIClient, ColocatedAPIClient])
@pytest.mark.parametrize("operation", ["fetch", "report"])
def test_network_retry_preserves_request_identity(monkeypatch, client_type, operation):
    client = client_type("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    client.max_retries = 2
    session = ValidationSession(client, "round", 2, "a", 0)
    call = Mock(
        side_effect=[
            requests.ConnectionError("lost reply"),
            response(body={"payloads_list": [], "is_end": True}),
        ]
    )
    monkeypatch.setattr(requests, "get" if operation == "fetch" else "post", call)
    if operation == "fetch":
        assert session.fetch(2) == ([], True)
        assert session.fetch_sequence == 1
        field = "params"
    else:
        session.report([], is_end=True)
        assert session.report_sequence == 1
        field = "json"
    assert call.call_count == 2
    assert call.call_args_list[0].kwargs == call.call_args_list[1].kwargs
    assert call.call_args.kwargs[field]["validation_round_id"] == "round"
    assert call.call_args.kwargs["timeout"] == constant.COSMOS_CONTROL_HTTP_TIMEOUT


@pytest.mark.parametrize("operation", ["fetch", "report"])
def test_rejected_receipt_is_terminal_without_sequence_advance(monkeypatch, operation):
    client = APIClient("ROLLOUT", remote_ips=["localhost"], remote_port=12345)
    session = ValidationSession(client, "round", 2, "a", 0)
    call = Mock(return_value=response(409))
    monkeypatch.setattr(requests, "get" if operation == "fetch" else "post", call)
    with pytest.raises(RuntimeError):
        if operation == "fetch":
            session.fetch(2)
        else:
            session.report([], is_end=True)
    assert call.call_count == 1
    assert session.fetch_sequence == session.report_sequence == 0


def test_unfenced_controller_is_not_silently_treated_as_completed_validation():
    command = RolloutToRolloutBroadcastCommand("a", ["a"], 2, 10, False)
    with pytest.raises(ValueError, match="matching controller/worker"):
        validation_round_for_command(command)
    command.validation_protocol_version = 1
    assert validation_round_for_command(command) is None
    command.validation_round_id = "round"
    assert validation_round_for_command(command) == "round"
