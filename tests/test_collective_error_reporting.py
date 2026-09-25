# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reporting a collective failure cannot enter the operational retry chain."""

from unittest.mock import Mock, patch

import pytest
import requests

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils import constant


@pytest.mark.parametrize(
    "failure", [None, requests.Timeout("offline"), requests.HTTPError("500")]
)
def test_collective_error_reporting_makes_one_timed_attempt(failure):
    client = APIClient(role="policy", remote_ips=["127.0.0.1"], remote_port=8000)
    response = Mock()
    if isinstance(failure, requests.HTTPError):
        response.raise_for_status.side_effect = failure
    with (
        patch(
            "cosmos_rl.dispatcher.api.client.requests.post", return_value=response
        ) as post,
        patch("cosmos_rl.dispatcher.api.client.make_request_with_retry") as retry,
    ):
        if isinstance(failure, requests.Timeout):
            post.side_effect = failure
        if failure is None:
            client.post_nccl_comm_error("policy-0", RuntimeError("native failed"))
        else:
            with pytest.raises(RuntimeError, match="Failed to report") as caught:
                client.post_nccl_comm_error("policy-0", RuntimeError("native failed"))
            assert caught.value.__cause__ is failure
        post.assert_called_once()
        assert post.call_args.kwargs["timeout"] == constant.COSMOS_CONTROL_HTTP_TIMEOUT
        retry.assert_not_called()
