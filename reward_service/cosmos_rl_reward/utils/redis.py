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

import redis.asyncio as redis
from cosmos_rl.utils.redis_stream import RedisOpType
from cosmos_rl.utils.constant import (
    COSMOS_HTTP_RETRY_CONFIG,
    COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
)
from typing import List
from functools import partial
from cosmos_rl_reward.utils.logging import logger
import subprocess
import sys
import uuid
import tempfile
import cosmos_rl.utils.network_util as network_util
from cosmos_rl_reward.utils.util import make_request_with_retry


class RedisAsyncHandler:
    def __init__(self, ips: List[str], port: int):
        """
        Initialize the RedisAsyncHandler.

        Args:
            ips (List[str]): The alternative IP addresses of the Redis server.
            port (int): The port of the Redis server.
            stream_name (str): The name of the Redis stream to interact with.
        """
        self.ips = ips
        self.port = port
        self.redis_clients = []
        for ip in ips:
            self.redis_clients.append(
                redis.Redis(host=ip, port=self.port, db=0, decode_responses=False)
            )
        # asyncio.run(self.ping())

    async def set_key_value(self, key: str, value: str):
        """
        Set a key-value pair in Redis asynchronously.

        Args:
            key (str): The key to set.
            value (str): The value to associate with the key.
        """
        # Add message to stream
        try:
            await make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.SET,
                    key,
                    value,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
            logger.info(f"[RedisAsyncHandler] Set key {key} in Redis with value {value}")
        except Exception as e:
            logger.error(f"[RedisAsyncHandler] Failed to write to Redis stream {key}: {e}")

    async def get_key_value(self, key: str):
        """
        Get a value by key from Redis asynchronously.
        Args:
            key (str): The key to retrieve.
        Returns:
            str: The value associated with the key.
        """

        try:
            value = await make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.GET,
                    key,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
            return value
        except Exception as e:
            logger.error(f"[RedisAsyncHandler] Failed to read from Redis key {key}: {e}")
            return None

    async def remove_key(self, key: str):
        """
        Remove a key from Redis.

        Args:
            key (str): The key to remove.
        """
        try:
            deleted_count = await make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.DELETE,
                    key,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
            return deleted_count
        except Exception as e:
            logger.error(f"[RedisAsyncHandler] Failed to delete key {key} from Redis: {e}")
            return 0

    def requests_for_alternative_clients(self, op: RedisOpType, *args, **kwargs):
        """
        Make requests to alternative clients based on the operation type.

        Args:
            op (RedisOpType): The operation type (XADD or XREAD or PING).
            *args: Positional arguments for the request.
            **kwargs: Keyword arguments for the request.

        Returns:
            list: A list of Callable objects for the requests.
        """
        calls = []
        if op == RedisOpType.XADD:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xadd,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.XREAD:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xread,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.PING:
            for redis_client in self.redis_clients:
                calls.append(redis_client.ping)
        elif op == RedisOpType.SET:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.set,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.GET:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.get,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.DELETE:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.delete,
                        *args,
                        **kwargs,
                    )
                )
        else:
            raise ValueError(f"Unsupported operation type: {op}")

        def make_timed_call(call):
            async def timed_call(*args, **kwargs):
                logger.info(f"[RedisAsyncHandler] Executing Redis call: {call.args}")
                res = await call(*args, **kwargs)
                logger.info(f"[RedisAsyncHandler] Executed Redis call: {call.args}")
                return res

            return timed_call

        return [make_timed_call(call) for call in calls]

    async def ping(self):
        """
        Ping the Redis server to check connectivity.
        """
        # Wait for redis to be ready
        try:
            await make_request_with_retry(
                self.requests_for_alternative_clients(RedisOpType.PING),
                response_parser=None,
                max_retries=COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
            )
        except Exception as e:
            logger.error(f"[RedisAsyncHandler] Failed to ping Redis when init Redis: {e}")
            raise e


def start_redis_server(redis_port: int, redis_logfile_path: str = "/tmp/redis.log"):
    """
    Start a Redis server on the specified port with a temporary configuration file.
    """

    if redis_port is None or redis_port <= 0:
        logger.info("[RedisAsyncHandler] Redis port is not set or invalid, skipping Redis server start.")
        return -1
    redis_free_port = network_util.find_available_port(redis_port)

    random_db_file_name = f"cosmos_rl_{str(uuid.uuid4())}.rdb"
    config_file_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".redis_config.conf"
    )
    custom_config = """
maxmemory 5G
maxmemory-policy allkeys-lfu
"""
    redis_cfg_path = network_util.write_redis_config(
        redis_free_port,
        redis_logfile_path,
        file_path=config_file_path.name,
        custom_config=custom_config,
    )
    redis_server_cmd = (
        f'redis-server {redis_cfg_path} --dbfilename {random_db_file_name} --save ""'
    )

    redis_server_proc = subprocess.Popen(
        redis_server_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr
    )

    # Check if the redis server started successfully
    redis_server_proc.wait()
    ret_code = redis_server_proc.returncode

    if ret_code is not None and ret_code != 0:
        raise RuntimeError(
            f"Failed to start redis server with command: {redis_server_cmd} with return code {ret_code}"
        )
    else:
        logger.info(
            f"[RedisAsyncHandler] Redis server started on port {redis_free_port} with command {redis_server_cmd}"
        )
    return redis_free_port
