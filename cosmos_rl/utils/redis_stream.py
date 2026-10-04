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

from cosmos_rl.utils import constant, teacher_channel
import redis
from datetime import datetime
from cosmos_rl.utils.constant import (
    RedisStreamConstant,
    COSMOS_HTTP_RETRY_CONFIG,
    COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
    COSMOS_HTTP_STREAM_POLL_MAX_RETRY,
)
from typing import List, Dict
from cosmos_rl.utils.network_util import make_request_with_retry
from functools import partial
from cosmos_rl.utils.logging import logger
import enum
import msgpack


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

    @staticmethod
    def _is_polling_read_miss(exc: Exception) -> bool:
        """Return True for expected streaming poll misses.

        Redis stream reads use blocking XREAD as a poll primitive. A socket
        timeout means no item arrived before the block timeout; connection
        refused can also appear while worker polling threads unwind after the
        controller has stopped its embedded Redis.
        """
        if isinstance(exc, redis.exceptions.TimeoutError):
            return True
        if isinstance(exc, redis.exceptions.ConnectionError):
            message = str(exc).lower()
            return "connection refused" in message or "error 111" in message
        return False

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
        messages = None
        try:
            messages = make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XREAD,
                    {stream_name + "_command": self.latest_id_command},
                    count=RedisStreamConstant.CMD_FETCH_SIZE,
                    block=RedisStreamConstant.CMD_READING_TIMEOUT_MS,
                ),
                response_parser=None,
                # Polling read: the caller loops on shutdown_signal, so fail fast
                # instead of running the deep retry storm that hangs teardown.
                max_retries=COSMOS_HTTP_STREAM_POLL_MAX_RETRY,
            )
        except Exception as e:
            if self._is_polling_read_miss(e):
                logger.debug(
                    f"[Redis] Poll read returned no command from {stream_name}_command: {e}"
                )
                return []
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
        messages = None
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
                # Polling read: the caller loops on shutdown_signal, so fail fast
                # instead of running the deep retry storm that hangs teardown.
                max_retries=COSMOS_HTTP_STREAM_POLL_MAX_RETRY,
            )
        except Exception as e:
            if self._is_polling_read_miss(e):
                logger.debug(
                    f"[Redis] Poll read returned no rollout from {stream_name}_rollout: {e}"
                )
                return []
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

    def publish_teacher_request(
        self, data: Dict, replica_name: str, *, stop_event=None
    ) -> List[str]:
        """Publish once per completion identity; never report a failed write as success."""
        return teacher_channel.publish(
            self,
            data,
            replica_name,
            constant.COSMOS_TEACHER_RESULT_SET_TIMEOUT,
            stop_event,
        )

    def subscribe_teacher_request(
        self, replica_name: str, count: int = -1
    ) -> List[Dict]:
        """Retain stream bodies and pending entries until result publication."""
        deadline = teacher_channel.deadline_after(1.0)

        def read(client):
            try:
                client.xgroup_create(
                    self.teacher_request_stream,
                    self.teacher_request_group,
                    id="0",
                    mkstream=True,
                )
            except redis.ResponseError as error:
                if "BUSYGROUP" not in str(error):
                    raise
            return client.xreadgroup(
                self.teacher_request_group,
                replica_name,
                {self.teacher_request_stream: ">"},
                count=RedisStreamConstant.TEACHER_REQUEST_FETCH_SIZE
                if count <= 0
                else count,
                block=100,
            )

        try:
            messages = teacher_channel.bounded_call(self.redis_clients, deadline, read)
        except teacher_channel.TeacherDeadline:
            return []
        requests = []
        for _, entries in messages or []:
            for _, fields in entries:
                request = msgpack.unpackb(fields[b"teacher_request"])
                if request.get("is_end"):
                    teacher_channel.complete(
                        self, request["_teacher_control_id"], b"", 1.0, control=True
                    )
                requests.append(request)
        return requests

    def set_teacher_result(
        self,
        uuid_value: str,
        data: Dict,
        replica_name: str,
        timeout: float = constant.COSMOS_TEACHER_RESULT_SET_TIMEOUT,
    ) -> bool:
        """Publish a result before acknowledging/removing its pending request."""
        packed = msgpack.packb(dict(data, replica_name=replica_name))
        try:
            return teacher_channel.complete(self, uuid_value, packed, timeout)
        except teacher_channel.TeacherDeadline:
            return False

    def get_teacher_result(
        self,
        uuid_value: str,
        timeout: float = constant.COSMOS_TEACHER_RESULT_GET_TIMEOUT,
        *,
        stop_event=None,
    ) -> bytes:
        """Replay-safe read; caller acknowledges only after owning the bytes."""
        return teacher_channel.read_result(self, uuid_value, timeout, stop_event)

    def acknowledge_teacher_result(self, uuid_value: str) -> None:
        teacher_channel.acknowledge_result(self, uuid_value)

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
