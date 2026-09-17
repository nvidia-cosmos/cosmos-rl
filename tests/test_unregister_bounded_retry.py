# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""``APIClient.unregister`` against a controller that has already finalized.

A replica that unregisters after the controller shut down only ever sees
connection-refused.  The deep ``COSMOS_HTTP_RETRY_CONFIG`` chain (60 attempts,
exponential backoff to 60 s) turned that into a ~50 min stall per replica while
the job kept its allocation.  Unregister is best-effort (the controller reaps a
silent replica by heartbeat timeout), so it must give up after a few attempts.
"""

from unittest.mock import patch

import requests

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils import constant


def test_unregister_gives_up_after_a_few_refused_attempts():
    client = APIClient(Role.ROLLOUT, remote_ips=["127.0.0.1"], remote_port=1)

    with (
        patch(
            "cosmos_rl.dispatcher.api.client.requests.post",
            side_effect=requests.ConnectionError("connection refused"),
        ) as post,
        patch("cosmos_rl.utils.network_util.time.sleep") as sleep,
    ):
        client.unregister("rollout-0")

    assert post.call_count == constant.COSMOS_HTTP_UNREGISTER_MAX_RETRY
    # Every retry waits ``(1 + jitter) * delay`` with the 1 s initial delay:
    # seconds in total, not the tens of minutes of the deep chain.
    assert sleep.call_count == constant.COSMOS_HTTP_UNREGISTER_MAX_RETRY - 1
    assert all(call.args[0] < 3.0 for call in sleep.call_args_list)
