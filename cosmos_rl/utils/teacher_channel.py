# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Replay-safe, bounded teacher delivery; not durable or elastic recovery.

Requests remain pending until their result is published. Completed identities
and unclaimed results expire after one day; publication has a shorter, sealed
deadline. Redis must retain its run identity and use noeviction. A dead teacher
is not transparently replaced: trainers can skip a timed-out update together.
"""

import hashlib
import math
import threading
import time
import uuid

import msgpack
import redis
from redis.backoff import NoBackoff
from redis.retry import Retry


RETENTION_SECONDS = 86400
# Bound even an OS/DNS operation that ignores socket timeouts. Timed-out calls
# retain their private client until they finish; no caller closes an active IO.
_IO_SLOTS = threading.BoundedSemaphore(8)


class TeacherDeadline(TimeoutError):
    pass


def deadline_after(timeout):
    if not math.isfinite(timeout) or timeout < 0 or timeout > 3600:
        raise ValueError("Teacher timeout must be finite and in [0, 3600] seconds")
    return time.monotonic() + timeout


def bounded_call(clients, deadline, operation, stop=None):
    """One absolute deadline covers IO, admission, retries and backoff."""
    if not clients:
        raise ValueError("Teacher IO requires at least one Redis client")
    last_error = None
    attempt = 0
    while time.monotonic() < deadline and not (stop and stop.is_set()):
        remaining = deadline - time.monotonic()
        if not _IO_SLOTS.acquire(timeout=max(0, min(0.05, remaining))):
            continue
        done = threading.Event()
        result = []
        source = clients[attempt % len(clients)]
        attempt += 1

        def perform(source=source, done=done, result=result):
            pool = None
            try:
                options = dict(source.connection_pool.connection_kwargs)
                # Do not copy a maintenance handler bound to another pool or
                # let automatic reconnect restore that pool's unbounded timeout.
                for name in (
                    "maint_notifications_pool_handler",
                    "maint_notifications_config",
                    "orig_host_address",
                    "orig_socket_timeout",
                    "orig_socket_connect_timeout",
                ):
                    options.pop(name, None)
                options.update(
                    socket_timeout=0.5, retry=Retry(NoBackoff(), 0), protocol=2
                )
                if (
                    source.connection_pool.connection_class
                    is not redis.UnixDomainSocketConnection
                ):
                    options["socket_connect_timeout"] = 0.5
                pool = redis.ConnectionPool(
                    connection_class=source.connection_pool.connection_class,
                    **options,
                )
                with redis.Redis(connection_pool=pool) as client:
                    result.append((True, operation(client)))
            except Exception as error:
                result.append((False, error))
            finally:
                try:
                    if pool is not None:
                        pool.disconnect()
                finally:
                    _IO_SLOTS.release()
                    done.set()

        thread = threading.Thread(target=perform, name="teacher-redis-io", daemon=True)
        try:
            thread.start()
        except BaseException:
            _IO_SLOTS.release()
            raise
        while not done.wait(max(0, min(0.05, deadline - time.monotonic()))):
            if time.monotonic() >= deadline or (stop and stop.is_set()):
                raise TeacherDeadline("Teacher IO deadline expired or stopped")
        success, value = result[0]
        if time.monotonic() >= deadline or (stop and stop.is_set()):
            raise TeacherDeadline("Teacher IO completed after deadline or stop")
        if success:
            return value
        if not isinstance(value, (redis.ConnectionError, redis.TimeoutError)):
            raise value
        last_error = value
        remaining = max(0, min(0.05, deadline - time.monotonic()))
        if stop is not None:
            stop.wait(remaining)
        else:
            time.sleep(remaining)
    raise TeacherDeadline("Teacher IO deadline expired or stopped") from last_error


_PUBLISH = """
local now = redis.call('TIME')
if tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000) > tonumber(ARGV[1]) then
    return redis.error_reply('teacher publication expired')
end
local info = redis.call('INFO', 'server')
if not string.find(info, 'run_id:' .. ARGV[2], 1, true) then
    return redis.error_reply('teacher Redis incarnation changed')
end
local existing = redis.call('HGET', KEYS[2], 'digest')
if existing then
    if existing ~= ARGV[3] then return redis.error_reply('teacher identity reused') end
    if redis.call('HGET', KEYS[2], 'state') == 'publishing' then
        return redis.error_reply('teacher publication completion uncertain')
    end
    return redis.call('HGET', KEYS[2], 'entry')
end
if redis.call('XLEN', KEYS[1]) >= tonumber(ARGV[5]) then
    return redis.error_reply('teacher request capacity exhausted')
end
local group = redis.pcall('XGROUP', 'CREATE', KEYS[1], ARGV[6], '0', 'MKSTREAM')
if type(group) == 'table' and group.err and not string.find(group.err, 'BUSYGROUP') then
    return group
end
-- Redis scripts serialize execution but do not roll back runtime errors.
-- Poison before mutation; a partial write must not be retried as new work.
redis.call('HSET', KEYS[2], 'digest', ARGV[3], 'state', 'publishing')
redis.call('EXPIRE', KEYS[2], ARGV[7])
local entry = redis.call('XADD', KEYS[1], '*', 'teacher_request', ARGV[4])
redis.call('PERSIST', KEYS[2])
redis.call('HSET', KEYS[2], 'digest', ARGV[3], 'entry', entry, 'state', 'pending')
return entry
"""

_COMPLETE = """
local state = redis.call('HGET', KEYS[2], 'state')
if not state then return redis.error_reply('unknown teacher request') end
if state == 'complete' then return 1 end
if state ~= 'pending' then return redis.error_reply('teacher completion uncertain') end
local entry = redis.call('HGET', KEYS[2], 'entry')
redis.call('HSET', KEYS[2], 'state', 'completing')
if ARGV[3] ~= 'control' then
    redis.call('SET', KEYS[3], ARGV[1], 'EX', ARGV[2])
end
-- Once XDEL frees stream capacity, a partial completion must not leave an
-- immortal poison marker outside that capacity bound. Expire it first.
redis.call('EXPIRE', KEYS[2], ARGV[2])
redis.call('XACK', KEYS[1], ARGV[4], entry)
redis.call('XDEL', KEYS[1], entry)
redis.call('HSET', KEYS[2], 'state', 'complete')
redis.call('EXPIRE', KEYS[2], ARGV[2])
return 1
"""


def marker(identity):
    return f"cosmos:teacher:request:{identity}"


def result_key(identity):
    return f"cosmos:teacher:result:{identity}"


def publish(handler, data, replica_name, timeout, stop=None):
    deadline = deadline_after(timeout)
    policy = bounded_call(
        handler.redis_clients,
        deadline,
        lambda client: client.config_get("maxmemory-policy")["maxmemory-policy"],
        stop,
    )
    if policy != "noeviction":
        raise redis.ResponseError("teacher channel requires noeviction")
    if "_teacher_deadline_ms" not in data:
        data["_teacher_deadline_ms"] = int((time.time() + timeout) * 1000)
    if "_teacher_run_id" not in data:
        data["_teacher_run_id"] = bounded_call(
            handler.redis_clients,
            deadline,
            lambda client: client.info("server")["run_id"],
            stop,
        )
    identities = data.setdefault(
        "teacher_result_uuid", [str(uuid.uuid4()) for _ in data["completion_token_ids"]]
    )
    if len(identities) != len(data["completion_token_ids"]) or len(
        set(identities)
    ) != len(identities):
        raise ValueError("Teacher completion identities must be unique and aligned")
    payload = {
        key: value for key, value in data.items() if not key.startswith("_teacher_")
    }
    payload["replica_name"] = replica_name
    requests = []
    for identity, tokens in zip(identities, data["completion_token_ids"]):
        if not isinstance(identity, str) or not identity:
            raise ValueError("Teacher identity must be a nonempty string")
        requests.append(
            (
                identity,
                dict(
                    payload,
                    teacher_result_uuid=[identity],
                    completion_token_ids=[tokens],
                ),
            )
        )
    if data.get("is_end"):
        if identities:
            raise ValueError("Teacher end marker must not resubmit completed requests")
        identity = data.setdefault("_teacher_control_id", str(uuid.uuid4()))
        requests.append((identity, dict(payload, _teacher_control_id=identity)))
    for identity, request in requests:
        packed = msgpack.packb(request)
        digest = hashlib.sha256(packed).hexdigest()
        arguments = (
            _PUBLISH,
            2,
            handler.teacher_request_stream,
            marker(identity),
            data["_teacher_deadline_ms"],
            data["_teacher_run_id"],
            digest,
            packed,
            10000,
            handler.teacher_request_group,
            RETENTION_SECONDS,
        )
        bounded_call(
            handler.redis_clients,
            deadline,
            lambda client, arguments=arguments: client.eval(*arguments),
            stop,
        )
    return identities


def complete(handler, identity, packed, timeout, *, control=False):
    return bool(
        bounded_call(
            handler.redis_clients,
            deadline_after(timeout),
            lambda client: client.eval(
                _COMPLETE,
                3,
                handler.teacher_request_stream,
                marker(identity),
                result_key(identity),
                packed,
                RETENTION_SECONDS,
                "control" if control else "result",
                handler.teacher_request_group,
            ),
        )
    )


def read_result(handler, identity, timeout, stop=None):
    deadline = deadline_after(timeout)
    try:
        while True:
            value = bounded_call(
                handler.redis_clients,
                deadline,
                lambda client: client.get(result_key(identity)),
                stop,
            )
            if value is not None:
                return value
            remaining = max(0, min(0.05, deadline - time.monotonic()))
            if stop is not None:
                stop.wait(remaining)
            else:
                time.sleep(remaining)
    except TeacherDeadline:
        return None


def acknowledge_result(handler, identity, timeout=1.0):
    # Only after the consumer owns the bytes. Duplicate completion does not
    # resurrect an acknowledged result, and deletion itself is replay-safe.
    bounded_call(
        handler.redis_clients,
        deadline_after(timeout),
        lambda client: client.delete(result_key(identity)),
    )
