# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shutdown unregister must not retry a controller that has already exited."""

from unittest.mock import patch

import pytest
import requests

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils import constant
from cosmos_rl.utils.api_suffix import COSMOS_API_UNREGISTER_SUFFIX


@pytest.mark.parametrize(
    "error", [None, requests.ConnectionError("controller gone"), requests.Timeout()]
)
def test_unregister_is_one_bounded_best_effort_request(error):
    client = APIClient("ROLLOUT", ["localhost", "127.0.0.1"], 8000)
    with (
        patch(
            "cosmos_rl.dispatcher.api.client.requests.post", side_effect=error
        ) as post,
        patch("cosmos_rl.dispatcher.api.client.make_request_with_retry") as retry,
    ):
        client.unregister("replica")
    post.assert_called_once_with(
        f"http://localhost:8000{COSMOS_API_UNREGISTER_SUFFIX}",
        json={"replica_name": "replica"},
        timeout=constant.COSMOS_CONTROL_HTTP_TIMEOUT,
    )
    retry.assert_not_called()
    assert client.max_retries == constant.COSMOS_HTTP_RETRY_CONFIG.max_retries
