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


from cosmos_rl_reward.utils.redis import RedisAsyncHandler
from cosmos_rl_reward.utils.kv_store import SimpleKeyValueStore
from cosmos_rl_reward.utils.logging import logger
from concurrent.futures import ProcessPoolExecutor
import asyncio
import os


class KeyValueHandler:
    """
    A handler for managing key-value pairs using either Redis or a simple in-memory store.
    This class supports asynchronous operations and process pools for concurrent access.
    It provides methods to set and get scores associated with unique identifiers (UUIDs).
    It records the calculated reward results in the key-value store for later retrieval.
    """

    def __init__(self, redis_port: int = None):
        self.redis_port = int(redis_port or os.getenv("REDIS_PORT", None))
        self.push_pool = None
        self.pull_pool = None
        self.init_redis()

    def init_process_pools(self, max_workers: int = 16):
        self.push_pool = ProcessPoolExecutor(
            max_workers=min(max(1, os.cpu_count() // 2), max_workers)
        )
        self.pull_pool = ProcessPoolExecutor(
            max_workers=min(max(1, os.cpu_count() // 2), max_workers)
        )

    def init_redis(self):
        if self.redis_port is None or self.redis_port <= 0:
            logger.info("Using SimpleKeyValueStore as Redis is not configured.")
            self.controller = SimpleKeyValueStore()
        else:
            logger.info(f"Starting Redis client on port {self.redis_port}")
            self.controller = RedisAsyncHandler(ips=["0.0.0.0"], port=self.redis_port)

    def init_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def set_score(self, uuid: str, scores: str, type: str):
        logger.info(f"Setting scores for UUID {uuid}: {scores}, type: {type}")
        await self.controller.set_key_value(
            f"{uuid}_{type}",  # Store scores with UUID as key
            scores,
        )

    async def get_score(self, uuid: str, type: str):
        if type is None:
            type = "dance_grpo"  # Default type if not provided
        logger.info(f"Pulling scores for UUID {uuid}, type: {type}")
        value = await self.controller.get_key_value(
            f"{uuid}_{type}"  # Retrieve scores using UUID as key
        )
        return value

    @classmethod
    def get_instance(cls, redis_port: int = None):
        if not hasattr(cls, "_instance"):
            cls._instance = cls(redis_port)
        return cls._instance

    @staticmethod
    def set_score_sync(uuid: str, scores: str, type: str, redis_port: int):
        controller = KeyValueHandler.get_instance(redis_port)
        controller.init_loop()  # Ensure the event loop is initialized
        asyncio.get_event_loop().run_until_complete(
            controller.set_score(uuid, scores, type)
        )
        controller.loop.close()  # Close the loop after operation

    @staticmethod
    def get_score_sync(uuid: str, type: str, redis_port: int):
        controller = KeyValueHandler.get_instance(redis_port)
        controller.init_loop()  # Ensure the event loop is initialized
        value = asyncio.get_event_loop().run_until_complete(
            controller.get_score(uuid, type)
        )
        controller.loop.close()  # Close the loop after operation
        return value
