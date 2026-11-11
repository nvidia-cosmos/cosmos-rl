# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import ast
import asyncio
import logging
from pathlib import Path
from queue import Queue
import threading
import time
from types import SimpleNamespace

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


def test_teacher_pending_byte_capacity_is_atomic_and_released_once(client):
    transport = handler(client)
    data = {"prompt_idx": 0, "completion_token_ids": [[[1], [2]]]}
    identities = transport.publish_teacher_request(data, "rollout")
    budget_key = transport.teacher_request_stream + ":bytes"
    occupied = int(client.get(budget_key))
    assert occupied > 0
    transport.teacher_max_pending_bytes = occupied
    with pytest.raises(redis.ResponseError, match="byte capacity"):
        transport.publish_teacher_request(
            {"prompt_idx": 1, "completion_token_ids": [[[3]]]}, "rollout"
        )
    assert client.xlen(transport.teacher_request_stream) == 1
    assert int(client.get(budget_key)) == occupied
    assert transport.set_teacher_result(identities[0], {"logprobs": [0.0]}, "teacher")
    assert int(client.get(budget_key)) == 0
    assert transport.set_teacher_result(identities[0], {"logprobs": [0.0]}, "teacher")
    assert int(client.get(budget_key)) == 0


def test_teacher_oversized_request_is_not_published(client):
    transport = handler(client)
    transport.teacher_max_request_bytes = 1
    with pytest.raises(ValueError, match="request exceeds"):
        transport.publish_teacher_request(request(), "rollout")
    assert client.xlen(transport.teacher_request_stream) == 0


def test_teacher_oversized_result_keeps_request_ownership(client):
    transport = handler(client)
    identities = transport.publish_teacher_request(request(), "rollout")
    before = int(client.get(transport.teacher_request_stream + ":bytes"))
    transport.teacher_max_result_bytes = 1
    with pytest.raises(ValueError, match="result exceeds"):
        transport.set_teacher_result(identities[0], {"logprobs": [0.0]}, "teacher")
    assert client.xlen(transport.teacher_request_stream) == 2
    assert int(client.get(transport.teacher_request_stream + ":bytes")) == before


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


@pytest.mark.parametrize("order", [("policy", "rollout"), ("rollout", "policy")])
@pytest.mark.parametrize("teachers", [1, 2])
def test_controller_departure_stops_all_teachers_without_final_weight_sync(
    client, order, teachers
):
    from cosmos_rl.dispatcher.controller import Controller
    from cosmos_rl.reference.worker.teacher_worker import TeacherWorker

    class Manager(dict):
        def unregister(self, name, *args):
            self.pop(name)

    transport = handler(client)
    controller = Controller.__new__(Controller)
    controller.policy_status_manager = Manager(policy=object())
    controller.rollout_status_manager = Manager(rollout=object())
    controller.teacher_result_manager = {f"teacher-{i}" for i in range(teachers)}
    controller.life_cycle_lock = asyncio.Lock()
    controller.redis_controller = transport

    async def complete():
        await controller.unregister(order[0])
        assert client.xlen(transport.teacher_request_stream) == 0
        await controller.unregister(order[1])
        assert client.xlen(transport.teacher_request_stream) == 1
        for i in range(teachers):
            worker = SimpleNamespace(
                global_rank=0,
                replica_name=f"teacher-{i}",
                shutdown_signal=threading.Event(),
                redis_controller=transport,
                engine=SimpleNamespace(batch_size=2),
                data_queue=Queue(),
            )
            await TeacherWorker.fetch_rollouts(worker)
            assert worker.data_queue.get_nowait() == {"is_end": True}
            assert worker.data_queue.empty()
            await controller.unregister(worker.replica_name)
        assert not controller.teacher_result_manager

    asyncio.run(complete())
    assert client.xlen(transport.teacher_request_stream) == 0
    assert int(client.get(transport.teacher_request_stream + ":bytes")) == 0
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


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_waiting_update_reclaims_prefetch_space_and_preserves_future_reply():
    inbox = TeacherResultInbox(max_bytes=8)
    inbox.admit("future")
    inbox.admit("current")
    assert inbox.complete("future", b"future!!")
    # A full prefetch cache must not admit unbounded replies or repeatedly read
    # the same large body; the active update can reclaim its local space.
    assert not inbox.complete("current", b"current!")
    assert inbox.next_pending() is None
    result = []
    thread = threading.Thread(target=lambda: result.extend(inbox.wait(["current"], 1)))
    thread.start()
    try:
        wait_until(lambda: bool(inbox._waiting))
        assert inbox.next_pending() == "current"
        assert inbox.complete("current", b"current!")
    finally:
        thread.join(2)
    assert result == [b"current!"] and inbox._bytes == 8
    assert inbox.pending("future")
    inbox.retire(["current"])
    assert inbox.complete("future", b"future!!")
    assert inbox.wait(["future"], 0) == [b"future!!"]
    inbox.retire(["future"])
    assert inbox._bytes == 0


def test_oversized_update_exits_wait_without_exceeding_inbox_budget():
    inbox = TeacherResultInbox(max_bytes=8)
    for identity in ("a", "b"):
        inbox.admit(identity)
    assert inbox.complete("a", b"12345")
    result = []
    thread = threading.Thread(target=lambda: result.extend(inbox.wait(["a", "b"], 2)))
    thread.start()
    try:
        wait_until(lambda: bool(inbox._waiting))
        assert not inbox.complete("b", b"67890")
    finally:
        thread.join(1)
    assert not thread.is_alive()
    assert result == [None, None] and inbox._bytes == 5
    inbox.retire(["a", "b"])
    assert not inbox.complete("b", b"late")
    assert inbox._bytes == 0


def test_retirement_retry_preserves_the_rest_of_the_window():
    inbox = TeacherResultInbox(max_bytes=8)
    for identity in ("a", "b", "c"):
        inbox.admit(identity)
    inbox.retire(["a", "b", "c"])
    with pytest.raises(ConnectionError):
        inbox.release_retired(lambda _: (_ for _ in ()).throw(ConnectionError()))
    released = []
    inbox.release_retired(released.append)
    assert sorted(released) == ["a", "b", "c"]
    inbox.release_retired(released.append)
    assert len(released) == 3


def teacher_worker_loop():
    # Exercise the actual worker method without constructing its GPU model.
    path = Path(cosmos_rl.__file__).parent / "policy/worker/rl_worker.py"
    method = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "teacher_interact_loop"
    )
    namespace = {"logger": logging.getLogger("teacher-test")}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[method.name]


def test_worker_missing_first_target_does_not_block_a_healthy_later_reply(client):
    transport = handler(client)
    identities = transport.publish_teacher_request(request(), "rollout")
    assert transport.set_teacher_result(identities[1], {"value": "healthy"}, "teacher")
    inbox = TeacherResultInbox(max_bytes=1024)
    for identity in identities:
        inbox.admit(identity)
    stopped = threading.Event()
    worker = SimpleNamespace(
        trainer=SimpleNamespace(teacher_results=inbox),
        redis_controller=transport,
        shutdown_signal=stopped,
    )
    thread = threading.Thread(target=teacher_worker_loop(), args=(worker,))
    thread.start()
    try:
        bodies = inbox.wait(identities, 0.5)
        assert bodies[0] is None
        assert msgpack.unpackb(bodies[1])["value"] == "healthy"
        inbox.retire(identities)
        wait_until(lambda: client.xlen(transport.teacher_request_stream) == 0)
        assert transport.set_teacher_result(identities[0], {"value": "late"}, "teacher")
        assert client.get(channel.result_key(identities[0])) is None
    finally:
        stopped.set()
        thread.join(2)
    assert not thread.is_alive()


def test_worker_keeps_channel_capacity_until_consumption_and_retries_cleanup(client):
    transport = handler(client)
    value = {"teacher_logprobs": [[-1.0]]}
    size = len(msgpack.packb(dict(value, replica_name="teacher")))
    transport.teacher_max_completed_bytes = size
    identities = transport.publish_teacher_request(request(), "rollout")
    assert transport.set_teacher_result(identities[0], value, "teacher")
    inbox = TeacherResultInbox(max_bytes=size)
    for identity in identities:
        inbox.admit(identity)
    stopped = threading.Event()
    worker = SimpleNamespace(
        trainer=SimpleNamespace(teacher_results=inbox),
        redis_controller=transport,
        shutdown_signal=stopped,
    )
    acknowledge = transport.acknowledge_teacher_result
    attempts = []

    def lost_ack(identity, **kwargs):
        acknowledge(identity, **kwargs)
        attempts.append(identity)
        if len(attempts) == 1:
            raise redis.ConnectionError("acknowledgement reply lost")

    transport.acknowledge_teacher_result = lost_ack
    thread = threading.Thread(target=teacher_worker_loop(), args=(worker,))
    thread.start()
    try:
        body = inbox.wait([identities[0]], 2)[0]
        assert body is not None and inbox._bytes == size
        assert not attempts
        assert client.get(channel.result_key(identities[0])) == body
        with pytest.raises(redis.ResponseError, match="completed result capacity"):
            transport.set_teacher_result(identities[1], value, "teacher")
        inbox.retire([identities[0]])
        wait_until(lambda: len(attempts) >= 2)
        assert attempts[:2] == [identities[0], identities[0]]
        assert transport.set_teacher_result(identities[1], value, "teacher")
        assert inbox.wait([identities[1]], 2)[0] is not None
        inbox.retire([identities[1]])
        wait_until(
            lambda: client.hlen(transport.teacher_request_stream + ":completed_sizes")
            == 0
        )
        assert inbox._bytes == 0
        assert transport.set_teacher_result(identities[0], value, "teacher")
        assert client.get(channel.result_key(identities[0])) is None
    finally:
        stopped.set()
        thread.join(2)
    assert not thread.is_alive()


def test_completion_and_acknowledgement_replies_can_be_lost(client, monkeypatch):
    transport = handler(client)
    identity = transport.publish_teacher_request(request(), "rollout")[0]
    transport.subscribe_teacher_request("teacher", count=2)
    original_eval = redis.Redis.eval
    lost = set()

    def uncertain_eval(self, script, *args):
        result = original_eval(self, script, *args)
        if script == channel._COMPLETE and "complete" not in lost:
            lost.add("complete")
            raise redis.ConnectionError("completed result reply lost")
        if script == channel._ACK and "ack" not in lost:
            lost.add("ack")
            raise redis.ConnectionError("result acknowledgement reply lost")
        return result

    monkeypatch.setattr(redis.Redis, "eval", uncertain_eval)
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


def test_completed_result_budget_retains_ownership_until_ack(client):
    transport = handler(client)
    transport.teacher_max_completed_bytes = 80
    identities = transport.publish_teacher_request(request(), "rollout")
    payload = {"value": "x" * 30}
    assert transport.set_teacher_result(identities[0], payload, "teacher")
    retained = client.hgetall(transport.teacher_request_stream + ":completed_sizes")
    assert 0 < sum(map(int, retained.values())) <= 80
    with pytest.raises(redis.ResponseError, match="completed result capacity"):
        transport.set_teacher_result(identities[1], payload, "teacher")
    assert client.xlen(transport.teacher_request_stream) == 1
    assert transport.get_teacher_result(identities[0], timeout=1) is not None
    transport.acknowledge_teacher_result(identities[0])
    transport.acknowledge_teacher_result(identities[0])
    assert transport.set_teacher_result(identities[1], payload, "teacher")
    assert transport.set_teacher_result(identities[0], payload, "teacher")
    assert client.get(channel.result_key(identities[0])) is None
    assert client.hlen(transport.teacher_request_stream + ":completed_sizes") == 1


def test_completed_result_expiry_reclaims_capacity(client, monkeypatch):
    transport = handler(client)
    transport.teacher_max_completed_bytes = 80
    identities = transport.publish_teacher_request(request(), "rollout")
    payload = {"value": "x" * 30}
    assert transport.set_teacher_result(identities[0], payload, "teacher")
    # Force the existing expiry index into the past; pruning atomically removes
    # the body too, so capacity never undercharges a still-retained result.
    client.zadd(
        transport.teacher_request_stream + ":completed_expiry", {identities[0]: 0}
    )
    assert transport.set_teacher_result(identities[1], payload, "teacher")
    assert client.get(channel.result_key(identities[0])) is None
    assert client.hlen(transport.teacher_request_stream + ":completed_sizes") == 1


def test_abandoned_pending_request_is_cancelled_without_late_resurrection(client):
    transport = handler(client)
    identity = transport.publish_teacher_request(
        {"completion_token_ids": [[[1]]]}, "rollout"
    )[0]
    transport.subscribe_teacher_request("teacher", count=1)
    transport.acknowledge_teacher_result(identity)
    assert client.xlen(transport.teacher_request_stream) == 0
    assert int(client.get(transport.teacher_request_stream + ":bytes")) == 0
    assert transport.set_teacher_result(identity, {"value": "late"}, "teacher")
    assert client.get(channel.result_key(identity)) is None
