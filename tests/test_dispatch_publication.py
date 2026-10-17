# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import redis

from cosmos_rl.utils.redis_publication import PublicationPlan
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.dispatcher.publication import ControllerPublisher


@pytest.fixture
def client(tmp_path):
    path = str(tmp_path / "redis.sock")
    server = subprocess.Popen(
        [
            "redis-server",
            "--port",
            "0",
            "--unixsocket",
            path,
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
    )
    connection = redis.Redis(unix_socket_path=path, socket_timeout=1, protocol=2)
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                connection.ping()
                break
            except redis.ConnectionError:
                assert time.monotonic() < deadline and server.poll() is None
                time.sleep(0.01)
        yield connection
    finally:
        connection.close()
        server.terminate()
        server.wait(timeout=5)


def plan(client):
    return PublicationPlan.create(
        [
            ("one_rollout", "rollout", b"first"),
            ("one_rollout", "rollout", b"second"),
            ("one_command", "command", b"fetch-two"),
            ("two_command", "command", b"fetch-local"),
        ]
    ).for_server(client.info("server")["run_id"])


def test_lost_reply_retries_one_logical_dispatch_without_duplicate_entries(client):
    operation = plan(client)

    class LostReply:
        def eval(self, *args):
            client.eval(*args)
            raise redis.ConnectionError("reply lost after commit")

    with pytest.raises(redis.ConnectionError):
        operation.publish(LostReply(), maxlen=100)
    ids = operation.publish(client, maxlen=100)
    assert operation.publish(client, maxlen=100) == ids
    assert client.xlen("one_rollout") == 2
    assert client.xlen("one_command") == client.xlen("two_command") == 1
    assert [data[b"rollout"] for _, data in client.xrange("one_rollout")] == [
        b"first",
        b"second",
    ]


def test_changed_plan_cannot_reuse_publication_identity(client):
    operation = plan(client)
    operation.publish(client, maxlen=100)
    altered = replace(operation, entries=(("one_command", "command", b"changed"),))
    with pytest.raises(redis.ResponseError, match="different content"):
        altered.publish(client, maxlen=100)
    assert client.xlen("one_command") == 1


def test_partial_script_failure_cannot_duplicate_already_published_prefix(client):
    operation = plan(client)
    # Inject the state left by a Redis script command failing after reservation
    # and its first append; Redis does not roll those earlier writes back.
    client.set(operation.key, operation.digest + ":pending", ex=60)
    client.xadd("one_rollout", {"rollout": b"first"})
    with pytest.raises(redis.ResponseError, match="incomplete"):
        operation.publish(client, maxlen=100)
    assert client.xlen("one_rollout") == 1 and not client.exists("one_command")


def test_expired_marker_eviction_does_not_authorize_replay(client):
    operation = replace(plan(client), deadline=time.time() - 1)
    with pytest.raises(redis.ResponseError, match="expired"):
        operation.publish(client, maxlen=100)
    assert not client.exists("one_rollout")


def test_different_redis_incarnation_cannot_replay_committed_plan(client):
    operation = plan(client)
    operation.publish(client, maxlen=100)
    restarted = replace(operation, server_id="0" * 40)
    with pytest.raises(redis.ResponseError, match="restarted"):
        restarted.publish(client, maxlen=100)
    assert client.xlen("one_rollout") == 2


def test_evicting_redis_is_rejected_before_publication(client):
    client.config_set("maxmemory-policy", "allkeys-lfu")
    handler = RedisStreamHandler.__new__(RedisStreamHandler)
    handler.redis_clients = [client]
    handler._publication_lock = threading.Lock()
    handler._publication_failure = None
    handler._publication_server_id = None
    with pytest.raises(ValueError, match="noeviction"):
        handler.publish_command(b"payload", "replica")
    assert not client.exists("replica_command")


def test_invalid_later_stream_is_rejected_before_any_publication(client):
    operation = plan(client)
    client.set("two_command", b"wrong-type")
    with pytest.raises(redis.ResponseError, match="not a stream"):
        operation.publish(client, maxlen=100)
    assert not client.exists(operation.key) and not client.exists("one_rollout")


def test_distinct_plans_preserve_stream_order(client):
    first, second = plan(client), plan(client)
    first.publish(client, maxlen=100)
    second.publish(client, maxlen=100)
    first.publish(client, maxlen=100)
    assert client.xlen("one_rollout") == 4
    assert client.xlen("one_command") == 2


def test_concurrent_retries_have_one_commit(client):
    operation = plan(client)
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: operation.publish(client, maxlen=100), range(32)))
    assert all(result == ids[0] for result in ids)
    assert client.xlen("one_rollout") == 2 and client.xlen("one_command") == 1


def test_dispatch_cannot_trim_its_own_unconsumed_payload(client):
    with pytest.raises(ValueError, match="retention"):
        plan(client).publish(client, maxlen=1)
    assert not client.exists("one_rollout")


@pytest.mark.parametrize("kind", ["command", "rollout"])
def test_stream_handler_retries_the_same_publication_plan(client, kind):
    class LostReply:
        calls = 0

        def eval(self, *args):
            self.calls += 1
            result = client.eval(*args)
            if self.calls == 1:
                raise redis.ConnectionError("committed reply was lost")
            return result

    transport = LostReply()
    handler = RedisStreamHandler.__new__(RedisStreamHandler)
    handler.redis_clients = [transport]
    handler._publication_lock = threading.Lock()
    handler._publication_failure = None
    handler._publication_server_id = client.info("server")["run_id"]
    getattr(handler, "publish_" + kind)(b"payload", "replica")
    assert transport.calls == 2
    assert client.xlen("replica_" + kind) == 1


def test_terminal_plan_failure_blocks_new_publications(client):
    handler = RedisStreamHandler.__new__(RedisStreamHandler)
    handler.redis_clients = [client]
    handler._publication_lock = threading.Lock()
    handler._publication_failure = None
    handler._publication_server_id = client.info("server")["run_id"]
    operation = replace(plan(client), deadline=time.time() - 1)
    with pytest.raises(TimeoutError):
        handler.publish_plan(operation)
    with pytest.raises(RuntimeError, match="terminally failed"):
        handler.publish_command(b"must-not-continue", "replica")
    assert not client.exists("replica_command")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_publication_lifetime_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        PublicationPlan.create([("stream", "command", b"data")], timeout_s=timeout)


def test_outbox_keeps_http_loop_responsive_and_preserves_publication_order():
    entered, release = threading.Event(), threading.Event()
    published = []
    failure = Mock()

    def send(operation):
        entered.set()
        assert release.wait(3), "test did not release its writer"
        published.append(operation.entries[0][2])
        return ["1-0"]

    async def scenario():
        publisher = ControllerPublisher(
            SimpleNamespace(publish_plan=send), on_failure=failure
        )
        try:
            first = publisher.publish_command(b"first", "replica")
            second = publisher.publish_command(b"second", "replica")
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert entered.is_set()
            # Heartbeat/timer work can run while the Redis writer is parked.
            tick = asyncio.Event()
            asyncio.get_running_loop().call_soon(tick.set)
            await asyncio.wait_for(tick.wait(), 1)
            assert not first.done() and not second.done() and published == []
        finally:
            release.set()
            await publisher.close()
        assert published == [b"first", b"second"]
        failure.assert_not_called()

    asyncio.run(scenario())


def test_outbox_failure_is_terminal_and_observed_without_polling():
    error = redis.ResponseError("injected ambiguous partial publication")
    failed = threading.Event()
    observed = []

    def fail(operation):
        raise error

    def on_failure(exception):
        observed.append(exception)
        failed.set()

    async def scenario():
        publisher = ControllerPublisher(
            SimpleNamespace(publish_plan=fail), on_failure=on_failure
        )
        try:
            future = publisher.publish_command(b"first", "replica")
            with pytest.raises(redis.ResponseError):
                await asyncio.wrap_future(future)
            assert failed.wait(1)
            with pytest.raises(RuntimeError, match="terminally failed"):
                publisher.publish_command(b"must-not-publish", "replica")
        finally:
            with pytest.raises((RuntimeError, redis.ResponseError)):
                await publisher.close()
        assert observed == [error]

    asyncio.run(scenario())
