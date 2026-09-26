# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic accepted-operation regressions against an owned real Redis server."""

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import tempfile
import time
import threading

import pytest
import redis
import torch

from cosmos_rl.utils.payload_transport.nccl.buffer_registry import SendBufferRegistry
from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
from cosmos_rl.utils.payload_transport.nccl.rendezvous import (
    NcclRendezvous,
    TransferStatus,
)
from cosmos_rl.utils.transport_failure import TransportDeadline, TransportUnusableError


@pytest.fixture(scope="module")
def redis_client():
    with tempfile.TemporaryDirectory(prefix="cosmos-rv-") as directory:
        socket_path = directory + "/redis.sock"
        process = subprocess.Popen(
            [
                "redis-server",
                "--port",
                "0",
                "--unixsocket",
                socket_path,
                "--save",
                "",
                "--appendonly",
                "no",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        client = redis.Redis(
            unix_socket_path=socket_path,
            decode_responses=True,
            socket_timeout=2,
            protocol=2,
        )
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    client.ping()
                    break
                except redis.ConnectionError:
                    assert time.monotonic() < deadline, "Redis failed to start"
                    time.sleep(0.01)
            yield client
        finally:
            client.close()
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("first", [TransferStatus.ACCEPTED, TransferStatus.CANCELLED])
def test_one_atomic_winner_no_accept_after_cancel(redis_client, first):
    rv = NcclRendezvous(redis_client, "atomic")
    key = "atomic:" + first.value
    redis_client.set(key, "requested", ex=60)
    with ThreadPoolExecutor(8) as pool:
        answers = list(
            pool.map(lambda _: rv.respond(resp_key=key, status=first), range(32))
        )
    assert sum(answers) == 1
    loser = (
        TransferStatus.CANCELLED
        if first is TransferStatus.ACCEPTED
        else TransferStatus.ACCEPTED
    )
    assert not rv.respond(resp_key=key, status=loser)
    assert redis_client.ttl(key) > 0
    if first is TransferStatus.ACCEPTED:
        assert rv.respond(resp_key=key, status=TransferStatus.COMPLETE)
        assert not rv.respond(resp_key=key, status=TransferStatus.FAILED)


def test_lost_publish_reply_resolves_same_operation(redis_client):
    channel = "lost-publish"
    seen = []
    pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(channel)
    rv = NcclRendezvous(redis_client, "lost", uid_fn=lambda: [1, 2, 3])

    def serve():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = pubsub.get_message(timeout=0.1)
            if message:
                request = json.loads(message["data"])
                seen.append(request)
                assert rv.respond(
                    resp_key=request["resp_key"], status=TransferStatus.ACCEPTED
                )
                return
        raise AssertionError("Request was not published")

    class LostReply:
        def __getattr__(self, name):
            return getattr(redis_client, name)

        def publish(self, *args):
            redis_client.publish(*args)
            raise redis.ConnectionError("response lost after commit")

    receiver = NcclRendezvous(LostReply(), "lost", uid_fn=lambda: [1, 2, 3])
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(serve)
        try:
            result = receiver.initiate(
                transfer_id="payload",
                sender_replica="r",
                sender_rank=0,
                receiver_replica="p",
                receiver_rank=0,
                request_channel=channel,
                need_uid=True,
                timeout=3,
            )
            future.result(5)
        finally:
            pubsub.close()
    assert result.accepted
    assert len(seen) == 1
    assert result.response_key == seen[0]["resp_key"]
    assert seen[0]["uid_chars"] == [1, 2, 3]


def test_accepted_request_captures_uid_and_duplicate_does_not_lease_twice(redis_client):
    producer = NCCLRolloutMixin()
    producer._nccl_sender_rank = 0
    producer._nccl_registry = SendBufferRegistry(capacity=4)
    producer._nccl_comm_cache = CommCache(
        build_fn=lambda *args: 7, abort_fn=lambda _: None
    )
    producer._nccl_rendezvous = NcclRendezvous(redis_client, "lease")
    producer._nccl_registry.register("payload", torch.zeros(16, dtype=torch.uint8))
    pending = []
    producer._enqueue_send = lambda pair, item: pending.append(item)
    key = "lease:operation"
    redis_client.set(key, "requested", ex=60)
    request = {
        "transfer_id": "payload",
        "resp_key": key,
        "receiver_rank": 0,
        "receiver_replica": "policy",
        "uid_key": "expired-key",
        "uid_chars": [7, 8],
        "req_deadline": time.time() + 30,
    }
    try:
        producer._handle_request(request)
        producer._handle_request(request)
        entry = producer._nccl_registry.get("payload")
        assert entry.inflight == 1
        assert len(pending) == 1
        assert pending[0].uid_key == (7, 8)
        assert redis_client.get("expired-key") is None
    finally:
        for item in pending:
            item.operation.close()
            producer._nccl_registry.abandon_inflight(item.entry)


def test_each_invocation_has_a_new_operation_identity(redis_client):
    rv = NcclRendezvous(redis_client, "identity")
    results = [
        rv.initiate(
            transfer_id="payload",
            sender_replica="r",
            sender_rank=0,
            receiver_replica="p",
            receiver_rank=0,
            request_channel="no-sender",
            need_uid=False,
            timeout=0,
        )
        for _ in range(2)
    ]
    assert results[0].response_key != results[1].response_key
    assert all(
        item.status is TransferStatus.CANCELLED and not item.late_accept
        for item in results
    )


def test_peer_observer_fails_without_native_caller_progress_and_restarts(redis_client):
    rv = NcclRendezvous(redis_client, "observer")
    for generation in range(2):
        key = f"observer:{generation}"
        redis_client.set(key, "accepted", ex=60)
        failed = threading.Event()
        messages = []

        def fatal(message):
            messages.append(message)
            failed.set()

        operation = TransportDeadline(30, "blocked native init", fatal=fatal)
        rv.watch_operation(key, operation)
        with rv._watch_lock:
            observer = rv._watcher
        assert rv.respond(resp_key=key, status=TransferStatus.FAILED)
        assert failed.wait(3), "Observer depended on native caller progress"
        assert messages == [
            "blocked native init: peer reported accepted transfer failure"
        ]
        with pytest.raises(TransportUnusableError):
            operation.close()
        observer.join(3)
        assert not observer.is_alive()
        with rv._watch_lock:
            assert rv._watched == {} and rv._watcher is None
