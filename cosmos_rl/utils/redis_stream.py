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

import time
from cosmos_rl.utils import constant
import redis
from datetime import datetime
from cosmos_rl.utils.constant import (
    RedisStreamConstant,
    COSMOS_HTTP_RETRY_CONFIG,
    COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
)
from typing import List, Dict
from cosmos_rl.utils.network_util import make_request_with_retry
from functools import partial
from cosmos_rl.utils.logging import logger
import enum
import msgpack
import uuid


class RedisOpType(enum.Enum):
    XADD = "add"
    XREAD = "read"
    PING = "ping"
    SET = "set"
    GET = "get"
    DELETE = "delete"
    GETDEL = "getdel"
    XGROUP_CREATE = "xgroup_create"
    XREADGROUP = "xreadgroup"
    XACK = "xack"
    XDEL = "xdel"


class RedisStreamHandler:
    def __init__(self, ips: List[str], port: int):
        """
        Initialize the RedisStreamHandler.

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
        self.latest_id_command = "0-0"
        self.latest_id_rollout = "0-0"
        # Teacher request related
        self.latest_id_teacher_request = "0-0"
        self.teacher_request_group = "teacher_request_group"
        self.teacher_request_stream = "teacher_request_stream"
        self.ping()

    def set_key_value(self, key: str, value: str) -> bool:
        # Add message to stream
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.SET,
                    key,
                    value,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
            return True
        except Exception as e:
            logger.error(f"[Redis] Failed to write to Redis stream {key}: {e}")
            return False

    def get_key_value(self, key: str, op: RedisOpType = RedisOpType.GET) -> str:
        try:
            value = make_request_with_retry(
                self.requests_for_alternative_clients(
                    op,
                    key,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
            return value
        except Exception as e:
            logger.info(f"[Redis] Failed to read from Redis key {key}: {e}")
            return None

    def remove_key(self, key: str):
        """
        Remove a key from Redis.

        Args:
            key (str): The key to remove.
        """
        try:
            deleted_count = make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.DELETE,
                    key,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
            return deleted_count
        except Exception as e:
            logger.error(f"[Redis] Failed to delete key {key} from Redis: {e}")
            return 0

    def publish_command(self, data, stream_name: str):
        """
        Write data to the Redis stream.

        Args:
            data : The packed command to write to the stream.

        Returns:
            str: The ID of the added stream entry.
        """
        message = {"command": data, "timestamp": datetime.now().isoformat()}
        # Add message to stream
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XADD,
                    stream_name + "_command",
                    message,
                    maxlen=RedisStreamConstant.STREAM_MAXLEN,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Redis] Failed to write to Redis stream {stream_name}_command: {e}"
            )
            raise e

    def subscribe_command(self, stream_name: str) -> List[Dict]:
        """
        Read data from the Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to read from.

        Returns:
            list: A list of stream entries.
        """
        try:
            messages = make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XREAD,
                    {stream_name + "_command": self.latest_id_command},
                    count=RedisStreamConstant.CMD_FETCH_SIZE,
                    block=RedisStreamConstant.CMD_READING_TIMEOUT_MS,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Redis] Failed to read from Redis stream {stream_name}_command: {e}"
            )
            raise e
        commands = []
        if messages:
            for _, message_list in messages:
                for message_id, message_data in message_list:
                    commands.append(message_data[b"command"])
                    self.latest_id_command = message_id
        return commands

    def publish_rollout(self, data, stream_name: str):
        """
        Write data to the Redis stream.

        Args:
            data : The packed rollout to write to the stream.

        Returns:
            str: The ID of the added stream entry.
        """
        message = {"rollout": data, "timestamp": datetime.now().isoformat()}
        # Add message to stream
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XADD,
                    stream_name + "_rollout",
                    message,
                    maxlen=RedisStreamConstant.STREAM_MAXLEN,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Redis] Failed to write to Redis stream {stream_name}_rollout: {e}"
            )
            raise e

    def subscribe_rollout(self, stream_name: str, count: int = -1) -> List[Dict]:
        """
        Read data from the Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to read from.
            count (int): The number of messages to read.

        Returns:
            list: A list of stream entries.
        """
        try:
            messages = make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XREAD,
                    {stream_name + "_rollout": self.latest_id_rollout},
                    count=RedisStreamConstant.ROLLOUT_FETCH_SIZE
                    if count <= 0
                    else count,
                    block=RedisStreamConstant.ROLLOUT_READING_TIMEOUT_MS,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Redis] Failed to read from Redis stream {stream_name}_rollout: {e}"
            )
        rollouts = []
        if messages:
            for _, message_list in messages:
                for message_id, message_data in message_list:
                    rollouts.append(message_data[b"rollout"])
                    self.latest_id_rollout = message_id
        return rollouts

    def create_teacher_request_group(self):
        if hasattr(self, "teacher_request_group_created"):
            return
        # Create teacher request group
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XGROUP_CREATE,
                    self.teacher_request_stream,
                    self.teacher_request_group,
                    id=self.latest_id_teacher_request,
                    mkstream=True,
                ),
                response_parser=None,
                exception_parser=lambda e: "BUSYGROUP"
                in str(
                    e
                ),  # If the group is already created, it will raise a BUSYGROUP error.
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
            self.teacher_request_group_created = True
        except Exception as e:
            logger.error(
                f"[Redis] Failed to write to Redis stream teacher_request: {e}"
            )
            raise e

    def publish_teacher_request(self, data: Dict, replica_name: str) -> List[str]:
        """
        Write data to the Redis stream.

        Args:
            data (Dict): The teacher request to write to the stream.
            stream_name (str): The name of the Redis stream to write to.

        Returns:
            List[str]: The UUIDs of the teacher result.
        """
        if "teacher_result_uuid" in data:
            uuid_values = data["teacher_result_uuid"]
            data.update({"replica_name": replica_name})
        else:
            uuid_values = []
            for _ in data["completion_token_ids"]:
                uuid_value = str(uuid.uuid4())
                uuid_values.append(uuid_value)
            data.update(
                {"teacher_result_uuid": uuid_values, "replica_name": replica_name}
            )
        message = {
            "teacher_request": msgpack.packb(data),
            "timestamp": datetime.now().isoformat(),
        }
        self.create_teacher_request_group()
        # Add message to stream
        logger.debug(
            f"[Redis] Publishing teacher request to Redis stream {self.teacher_request_group}: token lengths {[len(tokens) for tokens in data['completion_token_ids']]} for uuids {uuid_values}"
        )
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XADD,
                    self.teacher_request_stream,
                    message,
                    maxlen=RedisStreamConstant.STREAM_MAXLEN,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Redis] Failed to write to Redis stream teacher_request: {e}"
            )
            # return if failed to write to Redis stream for fault tolerance
            # raise e
        logger.debug(
            f"[Redis] Published teacher request to Redis stream {self.teacher_request_stream}: {uuid_values}"
        )
        return uuid_values

    def subscribe_teacher_request(
        self, replica_name: str, count: int = -1
    ) -> List[Dict]:
        """
        Read data from the Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to read from.
            count (int): The number of messages to read.

        Returns:
            list: A list of stream entries.
        """
        self.create_teacher_request_group()
        try:
            messages = make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XREADGROUP,
                    self.teacher_request_group,
                    replica_name,
                    {self.teacher_request_stream: ">"},
                    count=RedisStreamConstant.TEACHER_REQUEST_FETCH_SIZE
                    if count <= 0
                    else count,
                    block=RedisStreamConstant.TEACHER_REQUEST_READING_TIMEOUT_MS,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            messages = []  # return empty list if failed to read from Redis stream
            logger.error(
                f"[Redis] Failed to read from Redis stream teacher_request: {e}"
            )
        teacher_requests = []
        if messages:
            for _, message_list in messages:
                for message_id, message_data in message_list:
                    teacher_request = msgpack.unpackb(message_data[b"teacher_request"])
                    try:
                        # messages = make_request_with_retry(
                        #     self.requests_for_alternative_clients(
                        #         RedisOpType.XACK,
                        #         self.teacher_request_stream,
                        #         self.teacher_request_group,
                        #         message_id,
                        #     ),
                        #     response_parser=None,
                        #     max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
                        # )
                        messages = make_request_with_retry(
                            self.requests_for_alternative_clients(
                                RedisOpType.XDEL,
                                self.teacher_request_stream,
                                message_id,
                            ),
                            response_parser=None,
                            max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
                        )
                    except Exception as e:
                        logger.error(
                            f"[Redis] Failed to acknowledge message {message_id} from Redis stream teacher_request: {e}"
                        )
                    teacher_requests.append(teacher_request)
        return teacher_requests

    def set_teacher_result(
        self,
        uuid_value: str,
        data: Dict,
        replica_name: str,
        timeout: float = constant.COSMOS_TEACHER_RESULT_SET_TIMEOUT,
    ) -> bool:
        """
        Write teacher result to Redis.

        Args:
            data (Dict): The teacher result to write to the stream.
            stream_name (str): The name of the Redis stream to write to.
            replica_name (str): The name of the replica to write to.
        Returns:
            bool: True if successful, False otherwise.
        """
        data.update(
            {
                "timestamp": datetime.now().isoformat(),
                "replica_name": replica_name,
            }
        )
        start_time = time.time()
        while time.time() - start_time < float(timeout):
            if self.set_key_value(uuid_value, msgpack.packb(data)):
                return True
            time.sleep(constant.COSMOS_TEACHER_RESULT_RETRY_TIMEOUT_INTERVAL)
        return False

    def get_teacher_result(
        self,
        uuid_value: str,
        timeout: float = constant.COSMOS_TEACHER_RESULT_GET_TIMEOUT,
    ) -> Dict:
        """
        Get teacher result from Redis.

        Args:
            uuid_value (str): The UUID of the teacher result to get.

        Returns:
            Dict: The teacher result.
        """
        start_time = time.time()
        while time.time() - start_time < float(timeout):
            value = self.get_key_value(uuid_value, op=RedisOpType.GETDEL)
            if value is not None:
                break
            time.sleep(constant.COSMOS_TEACHER_RESULT_RETRY_TIMEOUT_INTERVAL)
        if value is None:
            logger.error(
                f"[Redis] Failed to get teacher result from Redis key {uuid_value}"
            )
            return None
        return msgpack.unpackb(value)

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
        elif op == RedisOpType.XGROUP_CREATE:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xgroup_create,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.XREADGROUP:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xreadgroup,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.XACK:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xack,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.XDEL:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xdel,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.GETDEL:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.getdel,
                        *args,
                        **kwargs,
                    )
                )
        else:
            raise ValueError(f"Unsupported operation type: {op}")
        return calls

    def ping(self):
        # Wait for redis to be ready
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(RedisOpType.PING),
                response_parser=None,
                max_retries=COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
            )
        except Exception as e:
            logger.error(f"[Redis] Failed to ping Redis when init Redis: {e}")
            raise e
