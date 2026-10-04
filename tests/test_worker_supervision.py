# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
import logging
from pathlib import Path
from queue import Queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import List

import msgpack
import pytest
import redis
import cosmos_rl
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.utils import redis_stream
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.utils.worker_threads import (
    OwnedWorkerThreads,
    start_worker_thread,
    stop_worker_threads,
)


def test_healthy_close_joins_all_users_before_resource_retirement():
    stop = threading.Event()
    retired = []
    owner = OwnedWorkerThreads(stop)
    threads = [
        owner.start(f"reader-{i}", lambda: (stop.wait(), retired.append(True)))
        for i in range(3)
    ]
    assert all(isinstance(thread, threading.Thread) for thread in threads)
    owner.close(1)
    assert len(retired) == 3 and all(not thread.is_alive() for thread in threads)
    owner.close(0)
    with pytest.raises(RuntimeError, match="admission"):
        owner.start("late", lambda: None)


def test_first_background_error_is_latched_and_closes_admission():
    failures = []
    stop = threading.Event()
    owner = OwnedWorkerThreads(stop, fatal=failures.append)

    def fail():
        raise ValueError("malformed command")

    thread = owner.start("decoder", fail)
    thread.join(1)
    assert stop.is_set() and len(failures) == 1 and "malformed command" in failures[0]
    with pytest.raises(RuntimeError, match="background"):
        owner.close(1)


def test_timed_out_join_does_not_authorize_resource_retirement():
    release = threading.Event()
    failures = []
    owner = OwnedWorkerThreads(threading.Event(), fatal=failures.append)
    thread = owner.start("blocked", release.wait)
    try:
        with pytest.raises(RuntimeError, match="background"):
            owner.close(0.01)
        assert thread.is_alive() and "still owns" in failures[0]
    finally:
        release.set()
        thread.join(1)


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_close_budget_rejected(timeout):
    with pytest.raises(ValueError):
        OwnedWorkerThreads(threading.Event()).close(timeout)


@pytest.mark.parametrize("mode", ["producer", "join"])
def test_terminal_path_exits_even_if_main_consumer_is_blocked(mode):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import threading
from queue import Queue
from cosmos_rl.utils.worker_threads import OwnedWorkerThreads
owner = OwnedWorkerThreads(threading.Event())
if MODE == 'producer':
    def fail():
        raise ValueError('injected producer failure')
    owner.start('producer', fail)
else:
    owner.start('blocked', threading.Event().wait)
    owner.close(0.02)
Queue().get()
""".replace("MODE", repr(mode)),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1 and "[Worker FATAL]" in result.stderr
    assert (
        "injected producer failure" if mode == "producer" else "still owns"
    ) in result.stderr


def method(relative, name, extra=None):
    path = Path(cosmos_rl.__file__).parent / relative
    tree = ast.parse(path.read_text())
    definition = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    definition.decorator_list = []
    namespace = dict(
        asyncio=asyncio,
        List=List,
        Rollout=Rollout,
        msgpack=msgpack,
        logger=logging.getLogger("test"),
        start_worker_thread=start_worker_thread,
        GRPOTrainer=object,
        MultiReplicaSFTPolicyWorker=object,
        time=time,
    )
    namespace.update(extra or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("malformed", [False, True])
def test_actual_fetch_method_never_swallows_a_consumed_decode_failure(malformed):
    stop = threading.Event()
    batches = [
        [
            msgpack.packb({"prompt_idx": 1}),
            msgpack.packb({"prompt_idx": [] if malformed else 2}),
        ]
    ]

    def subscribe(_):
        if batches:
            return batches.pop()
        stop.set()
        return []

    worker = SimpleNamespace(
        global_rank=0,
        shutdown_signal=stop,
        replica_name="r",
        redis_controller=SimpleNamespace(subscribe_rollout=subscribe),
        data_queue=Queue(),
        teacher_prefetch_queue=Queue(),
    )
    fetch = method("policy/worker/rl_worker.py", "fetch_rollouts")
    if malformed:
        with pytest.raises(ValueError):
            asyncio.run(fetch(worker))
        assert worker.data_queue.empty()
    else:
        asyncio.run(fetch(worker))
        assert worker.data_queue.qsize() == 2


@pytest.mark.parametrize(
    "poll", ["subscribe_command", "subscribe_rollout", "subscribe_teacher_request"]
)
def test_permanent_redis_error_reaches_producer(monkeypatch, poll):
    handler = RedisStreamHandler.__new__(RedisStreamHandler)
    handler.latest_id_command = handler.latest_id_rollout = "0-0"
    handler.teacher_request_group_created = True
    handler.teacher_request_stream = "teacher_request_stream"
    handler.teacher_request_group = "teacher_request_group"
    handler.redis_clients = []

    def invalid(*args, **kwargs):
        raise redis.ResponseError("WRONGTYPE")

    monkeypatch.setattr(redis_stream, "make_request_with_retry", invalid)
    if poll == "subscribe_teacher_request":
        monkeypatch.setattr(redis_stream.teacher_channel, "bounded_call", invalid)
    with pytest.raises(redis.ResponseError):
        getattr(handler, poll)("r")


def test_actual_grpo_startup_retains_three_thread_handles():
    class ReachedMain(Exception):
        pass

    stop = threading.Event()

    async def poll():
        stop.wait()

    def reached_main():
        raise ReachedMain()

    worker = SimpleNamespace(
        shutdown_signal=stop,
        fetch_command=poll,
        fetch_rollouts=poll,
        teacher_interact_loop=stop.wait,
        global_rank=0,
        parallel_dims=SimpleNamespace(pp_cp_tp_coord=(0, 1)),
        config=SimpleNamespace(distillation=SimpleNamespace(enable=True)),
        broadcast_command=reached_main,
    )
    try:
        with pytest.raises(ReachedMain):
            method("policy/worker/rl_worker.py", "main_loop")(worker)
        handles = [
            worker.fetch_command_thread,
            worker.fetch_rollouts_thread,
            worker.teacher_interact_thread,
        ]
        assert all(isinstance(handle, threading.Thread) for handle in handles)
    finally:
        stop_worker_threads(worker)
    assert all(not handle.is_alive() for handle in handles)


@pytest.mark.parametrize("kind", ["sft", "teacher"])
def test_other_actual_startup_paths_retain_join_handles(kind):
    class ReachedMain(Exception):
        pass

    stop = threading.Event()

    async def poll():
        stop.wait()

    def reached_main():
        raise ReachedMain()

    worker = SimpleNamespace(
        shutdown_signal=stop,
        fetch_command=poll,
        fetch_rollouts=poll,
        global_rank=0,
        profiler=SimpleNamespace(start=reached_main),
        engine=SimpleNamespace(model_load_from_hf=lambda: None),
        end_event=threading.Event(),
        dispatch_rollouts=reached_main,
    )
    relative = (
        "policy/worker/multi_replica_sft_worker.py"
        if kind == "sft"
        else "reference/worker/teacher_worker.py"
    )
    try:
        with pytest.raises(ReachedMain):
            method(relative, "main_loop")(worker)
        handle = getattr(
            worker, f"fetch_{'command' if kind == 'sft' else 'rollouts'}_thread"
        )
        assert isinstance(handle, threading.Thread)
    finally:
        stop_worker_threads(worker)
    assert not handle.is_alive()


def test_teacher_fetch_observes_shutdown_without_end_of_stream():
    stop = threading.Event()
    stop.set()
    worker = SimpleNamespace(global_rank=0, shutdown_signal=stop)
    # No Redis or engine access is allowed after shutdown is signaled.
    asyncio.run(method("reference/worker/teacher_worker.py", "fetch_rollouts")(worker))


def test_failed_thread_start_does_not_leave_unjoinable_handle(monkeypatch):
    owner = OwnedWorkerThreads(threading.Event())

    def fail(_):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="cannot start"):
        owner.start("failed-start", lambda: None)
    owner.close(0)


def test_start_racing_close_has_no_unowned_thread():
    for _ in range(20):
        owner = OwnedWorkerThreads(threading.Event())
        barrier = threading.Barrier(2)
        admitted = []

        def start():
            barrier.wait()
            try:
                admitted.append(owner.start("producer", owner.stop_event.wait))
            except RuntimeError as error:
                assert "admission" in str(error)

        starter = threading.Thread(target=start)
        starter.start()
        barrier.wait()
        owner.close(1)
        starter.join(1)
        assert not starter.is_alive()
        assert all(not thread.is_alive() for thread in admitted)


@pytest.fixture
def redis_client(tmp_path):
    socket = str(tmp_path / "worker.sock")
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
    client = redis.Redis(unix_socket_path=socket, socket_timeout=1, protocol=2)
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                client.ping()
                break
            except redis.ConnectionError:
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
        yield client
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize("malformed", [False, True])
def test_real_redis_consumed_batch_is_delivered_or_explicitly_fails(
    redis_client, malformed
):
    handler = RedisStreamHandler.__new__(RedisStreamHandler)
    handler.redis_clients = [redis_client]
    handler.latest_id_rollout = "0-0"
    redis_client.xadd("r_rollout", {"rollout": msgpack.packb({"prompt_idx": 1})})
    last = redis_client.xadd(
        "r_rollout", {"rollout": msgpack.packb({"prompt_idx": [] if malformed else 2})}
    )
    stop = threading.Event()

    def subscribe(name):
        batch = handler.subscribe_rollout(name)
        stop.set()
        return batch

    worker = SimpleNamespace(
        global_rank=0,
        shutdown_signal=stop,
        replica_name="r",
        redis_controller=SimpleNamespace(subscribe_rollout=subscribe),
        data_queue=Queue(),
        teacher_prefetch_queue=Queue(),
    )
    failures = []
    owner = OwnedWorkerThreads(stop, fatal=failures.append)
    fetch = method("policy/worker/rl_worker.py", "fetch_rollouts")
    thread = owner.start("rollout-ingestion", lambda: asyncio.run(fetch(worker)))
    thread.join(5)
    assert not thread.is_alive() and handler.latest_id_rollout == last
    if malformed:
        assert len(failures) == 1 and "validation" in failures[0].lower()
        with pytest.raises(RuntimeError, match="background"):
            owner.close(1)
    else:
        owner.close(1)
        assert failures == [] and worker.data_queue.qsize() == 2


def test_real_stalled_redis_read_returns_to_the_worker_stop_check(
    redis_client, monkeypatch
):
    original = redis.Redis
    socket_path = redis_client.connection_pool.connection_kwargs["path"]
    created = []

    def client_factory(**options):
        assert options["socket_connect_timeout"] == 2
        client = original(
            unix_socket_path=socket_path,
            socket_timeout=options["socket_timeout"],
            retry=options["retry"],
            protocol=options["protocol"],
        )
        created.append(client)
        return client

    for name in (
        "CMD_READING_TIMEOUT_MS",
        "ROLLOUT_READING_TIMEOUT_MS",
        "TEACHER_REQUEST_READING_TIMEOUT_MS",
    ):
        monkeypatch.setattr(redis_stream.RedisStreamConstant, name, 20)
    monkeypatch.setattr(redis, "Redis", client_factory)
    handler = RedisStreamHandler(["test"], 0)
    redis_client.execute_command("CLIENT", "PAUSE", 3000, "ALL")
    stop = threading.Event()
    entered = threading.Event()

    def subscribe(name):
        entered.set()
        return handler.subscribe_rollout(name)

    worker = SimpleNamespace(
        global_rank=0,
        shutdown_signal=stop,
        replica_name="r",
        redis_controller=SimpleNamespace(subscribe_rollout=subscribe),
        data_queue=Queue(),
        teacher_prefetch_queue=Queue(),
    )
    owner = OwnedWorkerThreads(stop)
    fetch = method("policy/worker/rl_worker.py", "fetch_rollouts")
    thread = owner.start("stalled-redis", lambda: asyncio.run(fetch(worker)))
    try:
        assert entered.wait(1)
        start = time.monotonic()
        owner.close(2)
        assert not thread.is_alive() and time.monotonic() - start < 2
    finally:
        for client in created:
            client.close()
