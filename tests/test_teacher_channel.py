# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import ast
from pathlib import Path
import threading
import time

import msgpack
import pytest
import redis
import cosmos_rl

from cosmos_rl.utils import teacher_channel as channel
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.utils.teacher_results import TeacherResultInbox


@pytest.fixture
def client(tmp_path):
    socket = str(tmp_path / "teacher.sock")
    process = subprocess.Popen(
        [
            "redis-server",
            "--port",
            "0",
            "--unixsocket",
            socket,
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
    )
    connection = redis.Redis(unix_socket_path=socket, socket_timeout=1, protocol=2)
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                connection.ping()
                break
            except redis.ConnectionError:
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
        yield connection
    finally:
        connection.close()
        process.terminate()
        process.wait(timeout=5)


def handler(client):
    result = RedisStreamHandler.__new__(RedisStreamHandler)
    result.redis_clients = [client]
    result.teacher_request_stream = "teacher_request_stream"
    result.teacher_request_group = "teacher_request_group"
    return result


def request():
    return {"prompt_idx": 0, "completion_token_ids": [[[1], [2]], [[3], [4]]]}


def test_pending_body_survives_teacher_death_and_completion_retires_pel(client):
    transport = handler(client)
    identities = transport.publish_teacher_request(request(), "rollout")
    fetched = transport.subscribe_teacher_request("teacher", count=2)
    assert [item["teacher_result_uuid"][0] for item in fetched] == identities
    assert client.xlen(transport.teacher_request_stream) == 2
    assert (
        client.xpending(
            transport.teacher_request_stream, transport.teacher_request_group
        )["pending"]
        == 2
    )
    # A process death here cannot delete either body. No automatic teacher
    # replacement is promised; missing targets settle at the trainer boundary.
    for identity in identities:
        assert transport.set_teacher_result(
            identity, {"teacher_logprobs": [[-1.0]]}, "teacher", timeout=1
        )
        result = transport.get_teacher_result(identity, timeout=1)
        assert msgpack.unpackb(result)["teacher_logprobs"] == [[-1.0]]
        assert transport.get_teacher_result(identity, timeout=1) == result
        transport.acknowledge_teacher_result(identity)
        assert transport.set_teacher_result(
            identity, {"teacher_logprobs": [[-2.0]]}, "teacher", timeout=1
        )
        assert client.get(channel.result_key(identity)) is None
    assert client.xlen(transport.teacher_request_stream) == 0
    assert (
        client.xpending(
            transport.teacher_request_stream, transport.teacher_request_group
        )["pending"]
        == 0
    )


def test_publication_lost_reply_does_not_duplicate_request(client, monkeypatch):
    original = redis.Redis.eval
    lost = []

    def uncertain(self, script, *args):
        result = original(self, script, *args)
        if script == channel._PUBLISH and not lost:
            lost.append(True)
            raise redis.ConnectionError("committed publication reply lost")
        return result

    monkeypatch.setattr(redis.Redis, "eval", uncertain)
    transport = handler(client)
    data = request()
    identities = transport.publish_teacher_request(data, "rollout")
    assert transport.publish_teacher_request(data, "rollout") == identities
    assert lost and client.xlen(transport.teacher_request_stream) == 2


def test_failed_publication_does_not_return_identities(client, monkeypatch):
    def rejected(*args):
        raise redis.ResponseError("publication rejected")

    monkeypatch.setattr(redis.Redis, "eval", rejected)
    with pytest.raises(redis.ResponseError, match="rejected"):
        handler(client).publish_teacher_request(request(), "rollout")


def test_identity_mutation_and_restart_rejected(client):
    transport = handler(client)
    data = request()
    transport.publish_teacher_request(data, "rollout")
    data["completion_token_ids"][0][0][0] = 999
    with pytest.raises(redis.ResponseError, match="identity reused"):
        transport.publish_teacher_request(data, "rollout")
    data["_teacher_run_id"] = "different-incarnation"
    with pytest.raises(redis.ResponseError, match="incarnation changed"):
        transport.publish_teacher_request(data, "rollout")


def test_nondestructive_result_read_survives_lost_reply(client, monkeypatch):
    transport = handler(client)
    identity = transport.publish_teacher_request(request(), "rollout")[0]
    transport.set_teacher_result(
        identity, {"teacher_logprobs": [[-1.0]]}, "teacher", timeout=1
    )
    original = redis.Redis.get
    lost = []

    def uncertain(self, key):
        result = original(self, key)
        if key == channel.result_key(identity) and not lost:
            lost.append(True)
            raise redis.ConnectionError("result reply lost")
        return result

    monkeypatch.setattr(redis.Redis, "get", uncertain)
    assert transport.get_teacher_result(identity, timeout=1) is not None
    assert lost and client.get(channel.result_key(identity)) is not None


def test_end_marker_is_acknowledged_without_training_result(client):
    transport = handler(client)
    transport.publish_teacher_request(
        {"is_end": True, "completion_token_ids": []}, "controller"
    )
    assert transport.subscribe_teacher_request("teacher")[0]["is_end"]
    assert client.xlen(transport.teacher_request_stream) == 0
    assert (
        client.xpending(
            transport.teacher_request_stream, transport.teacher_request_group
        )["pending"]
        == 0
    )


def test_request_capacity_never_trims_unconsumed_work(client):
    transport = handler(client)
    for _ in range(10000):
        client.xadd(transport.teacher_request_stream, {"existing": b"retained"})
    first = client.xrange(transport.teacher_request_stream, count=1)
    with pytest.raises(redis.ResponseError, match="capacity exhausted"):
        transport.publish_teacher_request(request(), "rollout")
    assert client.xlen(transport.teacher_request_stream) == 10000
    assert client.xrange(transport.teacher_request_stream, count=1) == first


def test_expired_publication_cannot_recreate_completed_work(client):
    data = request()
    data["_teacher_deadline_ms"] = 1
    with pytest.raises(redis.ResponseError, match="expired"):
        handler(client).publish_teacher_request(data, "rollout")


def test_partial_publication_is_poisoned_not_replayed(client):
    transport = handler(client)
    data = request()
    identities = transport.publish_teacher_request(data, "rollout")
    client.hset(channel.marker(identities[0]), "state", "publishing")
    with pytest.raises(redis.ResponseError, match="completion uncertain"):
        transport.publish_teacher_request(data, "rollout")
    assert client.xlen(transport.teacher_request_stream) == 2


def test_deadline_bounds_an_unresponsive_io_operation(client):
    release = threading.Event()
    entered = threading.Event()

    def blocked(_):
        entered.set()
        release.wait()

    start = time.monotonic()
    try:
        with pytest.raises(channel.TeacherDeadline):
            channel.bounded_call([client], channel.deadline_after(0.1), blocked)
        assert entered.is_set() and time.monotonic() - start < 0.5
    finally:
        release.set()


def test_missing_read_and_shutdown_have_bounded_wait(client):
    transport = handler(client)
    start = time.monotonic()
    assert transport.get_teacher_result("missing", timeout=0.1) is None
    assert time.monotonic() - start < 0.5
    stop = threading.Event()
    stop.set()
    start = time.monotonic()
    assert transport.get_teacher_result("missing", timeout=30, stop_event=stop) is None
    assert time.monotonic() - start < 0.1


def test_inbox_uses_one_batch_deadline_and_rejects_late_completion():
    inbox = TeacherResultInbox()
    for identity in ("a", "b", "c"):
        inbox.admit(identity)
    inbox.complete("b", b"healthy")
    start = time.monotonic()
    assert inbox.wait(["a", "b", "c"], 0.05) == [None, b"healthy", None]
    assert time.monotonic() - start < 0.1
    inbox.retire(["a", "b", "c"])
    assert not inbox.complete("a", b"too late")
    assert not inbox.pending("a")


def test_completion_and_acknowledgement_replies_can_be_lost(client, monkeypatch):
    transport = handler(client)
    identity = transport.publish_teacher_request(request(), "rollout")[0]
    transport.subscribe_teacher_request("teacher", count=2)
    original_eval = redis.Redis.eval
    original_delete = redis.Redis.delete
    lost = set()

    def uncertain_eval(self, script, *args):
        result = original_eval(self, script, *args)
        if script == channel._COMPLETE and "complete" not in lost:
            lost.add("complete")
            raise redis.ConnectionError("completed result reply lost")
        return result

    def uncertain_delete(self, *args):
        result = original_delete(self, *args)
        if "ack" not in lost:
            lost.add("ack")
            raise redis.ConnectionError("result acknowledgement reply lost")
        return result

    monkeypatch.setattr(redis.Redis, "eval", uncertain_eval)
    monkeypatch.setattr(redis.Redis, "delete", uncertain_delete)
    assert transport.set_teacher_result(
        identity, {"teacher_logprobs": [[-1.0]]}, "teacher", timeout=1
    )
    owned = transport.get_teacher_result(identity, timeout=1)
    transport.acknowledge_teacher_result(identity)
    assert msgpack.unpackb(owned)["teacher_logprobs"] == [[-1.0]]
    assert lost == {"complete", "ack"}
    assert client.get(channel.result_key(identity)) is None
    assert (
        client.xpending(
            transport.teacher_request_stream, transport.teacher_request_group
        )["pending"]
        == 1
    )


def test_partial_result_publication_cannot_be_replayed(client):
    transport = handler(client)
    identity = transport.publish_teacher_request(request(), "rollout")[0]
    client.hset(channel.marker(identity), "state", "completing")
    with pytest.raises(redis.ResponseError, match="completion uncertain"):
        transport.set_teacher_result(identity, {}, "teacher", timeout=1)
    assert client.get(channel.result_key(identity)) is None


def test_partial_completion_after_stream_retirement_has_bounded_marker(
    client, monkeypatch
):
    transport = handler(client)
    identity = transport.publish_teacher_request(request(), "rollout")[0]
    transport.subscribe_teacher_request("teacher", count=1)
    original = channel._COMPLETE
    monkeypatch.setattr(
        channel,
        "_COMPLETE",
        original.replace(
            "redis.call('XDEL', KEYS[1], entry)",
            "redis.call('XDEL', KEYS[1], entry)\nredis.call('INJECTED_INVALID_COMMAND')",
        ),
    )
    with pytest.raises(redis.ResponseError):
        transport.set_teacher_result(identity, {"value": 1}, "teacher", timeout=1)
    assert client.xlen(transport.teacher_request_stream) == 1  # other completion
    assert (
        client.xpending(
            transport.teacher_request_stream, transport.teacher_request_group
        )["pending"]
        == 0
    )
    assert client.hget(channel.marker(identity), "state") == b"completing"
    assert 0 < client.ttl(channel.marker(identity)) <= channel.RETENTION_SECONDS
    monkeypatch.setattr(channel, "_COMPLETE", original)
    with pytest.raises(redis.ResponseError, match="completion uncertain"):
        transport.set_teacher_result(identity, {"value": 2}, "teacher", timeout=1)
    assert msgpack.unpackb(client.get(channel.result_key(identity))) == {
        "value": 1,
        "replica_name": "teacher",
    }


def test_empty_client_configuration_does_not_consume_io_admission():
    for _ in range(10):
        with pytest.raises(ValueError, match="at least one Redis client"):
            channel.bounded_call([], channel.deadline_after(0.1), lambda _: None)
    acquired = []
    try:
        for _ in range(8):
            assert channel._IO_SLOTS.acquire(blocking=False)
            acquired.append(True)
    finally:
        for _ in acquired:
            channel._IO_SLOTS.release()


def test_deadline_also_bounds_repeated_connection_failures(client, monkeypatch):
    calls = []

    def failed(self, key):
        calls.append(key)
        raise redis.ConnectionError("unavailable")

    monkeypatch.setattr(redis.Redis, "get", failed)
    start = time.monotonic()
    assert handler(client).get_teacher_result("missing", timeout=0.12) is None
    assert 1 <= len(calls) < 5 and time.monotonic() - start < 0.5


def test_eviction_policy_is_rejected_before_publication(client):
    client.config_set("maxmemory-policy", "allkeys-lru")
    with pytest.raises(redis.ResponseError, match="noeviction"):
        handler(client).publish_teacher_request(request(), "rollout")
    assert client.xlen("teacher_request_stream") == 0


def test_actual_partial_script_write_leaves_bounded_poison_marker(client, monkeypatch):
    original = redis.Redis.eval
    injected = []

    def partial(self, script, *args):
        if script == channel._PUBLISH and not injected:
            injected.append(True)
            script = script.replace(
                "redis.call('XADD', KEYS[1], '*', 'teacher_request', ARGV[4])",
                "redis.call('COSMOS_INJECTED_UNKNOWN_COMMAND')",
            )
        return original(self, script, *args)

    monkeypatch.setattr(redis.Redis, "eval", partial)
    data = request()
    transport = handler(client)
    with pytest.raises(redis.ResponseError):
        transport.publish_teacher_request(data, "rollout")
    identity = data["teacher_result_uuid"][0]
    assert injected and client.hget(channel.marker(identity), "state") == b"publishing"
    assert 0 < client.ttl(channel.marker(identity)) <= channel.RETENTION_SECONDS
    assert client.xlen(transport.teacher_request_stream) == 0
    with pytest.raises(redis.ResponseError, match="completion uncertain"):
        transport.publish_teacher_request(data, "rollout")


def test_controller_owned_redis_uses_the_required_retention_policy():
    path = Path(cosmos_rl.__file__).parent / "dispatcher/controller.py"
    tree = ast.parse(path.read_text())
    config = next(
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "custom_config"
            for target in node.targets
        )
    )
    assert "maxmemory-policy noeviction" in config
    assert "allkeys" not in config
