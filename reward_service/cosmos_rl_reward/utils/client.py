# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from functools import partial
from cosmos_rl_reward.utils.logging import logger
from cosmos_rl_reward.utils.util import make_request_with_retry_sync
from cosmos_rl_reward.utils.protocol import (
    COSMOS_RL_REWARD_REDIS_PUSH_API_SUFFIX,
    COSMOS_RL_REWARD_REDIS_PULL_API_SUFFIX,
)
import json
import requests


class KVClientManager:
    """
    A client manager for interacting with a key-value store like Redis service over HTTP.
    It supports pushing and pulling scores associated with a unique identifier (UUID).
    """

    def __init__(self, url: str, token: str = None):
        self.url = url
        self.token = token
        self.session = requests.Session()

    def push_scores(self, uuid: str, scores):
        try:
            type = scores.get("type", None)
            response = make_request_with_retry_sync(
                partial(
                    self.session.post,
                    data={"uuid": uuid, "scores": json.dumps(scores), "type": type}
                    if type is not None
                    else {"uuid": uuid, "scores": json.dumps(scores)},
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=(5, 30),
                ),
                urls=[self.url + COSMOS_RL_REWARD_REDIS_PUSH_API_SUFFIX],
                max_retries=20,
                max_delay=5.0,
            )
            if response.status_code != 200:
                logger.error(f"Failed to push scores for UUID {uuid}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error pushing scores for UUID {uuid}: {e}")
            return False
        return True

    def pull_scores(self, uuid: str, type: str = None):
        try:
            response = make_request_with_retry_sync(
                partial(
                    self.session.post,
                    data={"uuid": uuid, "type": type}
                    if type is not None
                    else {"uuid": uuid},
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=(5, 30),
                ),
                urls=[self.url + COSMOS_RL_REWARD_REDIS_PULL_API_SUFFIX],
                max_retries=20,
                max_delay=5.0,
            )
            if response.status_code != 200:
                logger.error(f"Failed to pull scores for UUID {uuid}: {response.text}")
                return None
            response = response.json()
            return response
        except Exception as e:
            logger.error(f"Error pulling scores for UUID {uuid}: {e}")
            return None
