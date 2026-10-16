# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Admission ownership and actual live-weight handler fencing contracts."""

import asyncio
import ast
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cosmos_rl
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    RolloutTask,
    RolloutTaskScheduler,
)
from cosmos_rl.rollout.worker import rollout_control as control


def scheduler_for(generate=None):
    return RolloutTaskScheduler(
        SimpleNamespace(
            rollout_generation=generate, is_engine_initialized=lambda: False
        ),
        object(),
        check_interval=0.001,
    )


def test_pause_acknowledges_dequeue_before_yielding():
    dequeued, release_get, requested, paused, release_pause, finished = (
        threading.Event() for _ in range(6)
    )
    errors = []

    class InterleavedQueue(Queue):
        def get_nowait(self):
            dequeued.set()
            assert release_get.wait(3)
            return super().get_nowait()

    async def generate(**kwargs):
        while not finished.is_set():
            await asyncio.sleep(0.001)
        return [RolloutResult(completions=["ok"])]

    scheduler = scheduler_for(generate)
    scheduler.task_queue = InterleavedQueue()
    scheduler.put_rollout(RolloutTask(0, RLPayload(prompt_idx=0)))

    def pause():
        try:
            requested.set()
            with scheduler.paused(timeout=2):
                paused.set()
                assert release_pause.wait(3)
        except BaseException as error:
            errors.append(error)

    pauser = threading.Thread(target=pause)
    scheduler.start(lambda _: None, wait_initialized=True)
    try:
        assert dequeued.wait(2)
        pauser.start()
        assert requested.wait(2)
        assert not paused.wait(0.05), "pause missed a dequeued task"
        release_get.set()
        assert not paused.wait(0.05), "pause missed active generation"
        finished.set()
        assert paused.wait(2)
    finally:
        release_get.set()
        finished.set()
        release_pause.set()
        if pauser.ident is not None:
            pauser.join(3)
        scheduler.stop()
    assert not errors and not pauser.is_alive()


def test_nested_draining_pause_waits_even_inside_nondraining_owner():
    entered, release, finished = (threading.Event() for _ in range(3))

    async def generate(**kwargs):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.001)
        finished.set()
        return [RolloutResult(completions=["ok"])]

    scheduler = scheduler_for(generate)
    scheduler.start(lambda _: None, wait_initialized=True)
    scheduler.put_rollout(RolloutTask(0, RLPayload(prompt_idx=0)))
    releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
    try:
        assert entered.wait(2)
        with scheduler.paused(wait_for_active_tasks=False):
            releaser.start()
            with scheduler.paused(timeout=2):
                assert finished.is_set(), "nested pause did not drain"
            assert scheduler.is_paused()
        assert not scheduler.is_paused()
    finally:
        release.set()
        if releaser.ident is not None:
            releaser.join(2)
        scheduler.stop()


def test_overlapping_pause_owners_cannot_resume_each_other():
    scheduler = scheduler_for()
    scheduler.start(lambda _: None, wait_initialized=True)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    exited = [threading.Event(), threading.Event()]
    errors = []

    def pause(index):
        try:
            with scheduler.paused(timeout=2):
                entered[index].set()
                assert release[index].wait(3)
        except BaseException as error:
            errors.append(error)
        finally:
            exited[index].set()

    threads = [threading.Thread(target=pause, args=(i,)) for i in range(2)]
    try:
        threads[0].start()
        assert entered[0].wait(2)
        threads[1].start()
        assert entered[1].wait(2)
        release[0].set()
        assert exited[0].wait(2)
        assert scheduler.is_paused(), "another owner still holds a pause"
        release[1].set()
        assert exited[1].wait(2)
        assert not scheduler.is_paused()
    finally:
        for event in release:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(3)
        scheduler.stop()
    assert not errors


@pytest.mark.parametrize("manual_first", [False, True])
def test_manual_and_context_owners_are_independent(manual_first):
    scheduler = scheduler_for()
    scheduler._running.set()
    if manual_first:
        scheduler.pause()
    with pytest.raises(ValueError):
        with scheduler.paused():
            scheduler.resume()
            assert scheduler.is_paused()
            scheduler.pause()
            raise ValueError("body failed")
    assert scheduler.is_paused()
    scheduler.resume()
    assert not scheduler.is_paused()


def test_timeout_releases_only_its_own_pause():
    scheduler = scheduler_for()
    scheduler._running.set()
    scheduler._active_tasks.add(object())
    with scheduler.paused(wait_for_active_tasks=False):
        with pytest.raises(TimeoutError):
            with scheduler.paused(timeout=0):
                pytest.fail("pause yielded with active tasks")
        assert scheduler.is_paused()
    assert not scheduler.is_paused()


def test_draining_pause_cannot_block_its_own_event_loop():
    async def run():
        scheduler = scheduler_for()
        scheduler._running.set()
        scheduler._loop = asyncio.get_running_loop()
        task = asyncio.create_task(asyncio.sleep(0))
        scheduler._active_tasks.add(task)
        try:
            with pytest.raises(RuntimeError, match="own event loop"):
                with scheduler.paused():
                    pytest.fail("deadlocked event loop")
            assert not scheduler.is_paused()
        finally:
            await task

    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "fence", "write", "completion", "timeout"])
def test_live_weight_barrier_orders_reads_writes_and_poison(monkeypatch, failure):
    worker = object.__new__(control.DisaggregatedRolloutControlWorker)
    worker.scheduler = scheduler_for()
    worker.scheduler._running.set()
    worker.inference_stream = object()
    calls = []

    def fence(timeout):
        assert worker.scheduler.is_paused()
        calls.append("fence")
        if failure == "fence":
            raise RuntimeError("injected")

    def record(stream):
        assert stream is worker.inference_stream and worker.scheduler.is_paused()
        calls.append("record")

    def query():
        assert worker.scheduler.is_paused()
        calls.append("query")
        if failure == "completion":
            raise RuntimeError("injected")
        return failure != "timeout"

    worker.rollout = SimpleNamespace(synchronize_generation=fence)
    monkeypatch.setattr(
        control.torch.cuda, "Event", lambda: SimpleNamespace(record=record, query=query)
    )
    monkeypatch.setattr(control.constant, "COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT", 0)

    def update():
        with worker._paused_async_live_weights():
            assert calls == ["fence"] and worker.scheduler.is_paused()
            calls.append("write")
            if failure == "write":
                raise RuntimeError("injected")

    if failure:
        with pytest.raises((RuntimeError, TimeoutError)):
            update()
        assert worker.scheduler.is_paused() and worker._async_weight_write_failed
        with pytest.raises(RuntimeError, match="unusable"):
            update()
    else:
        update()
        assert calls == ["fence", "write", "record", "query"]
        assert not worker.scheduler.is_paused()


def test_sync_engine_does_not_use_async_fences(monkeypatch):
    worker = object.__new__(control.DisaggregatedRolloutControlWorker)
    worker.scheduler = None
    worker.rollout = SimpleNamespace(synchronize_generation=Mock())
    event = Mock(side_effect=AssertionError("unexpected GPU fence"))
    monkeypatch.setattr(control.torch.cuda, "Event", event)
    with worker._paused_async_live_weights():
        pass
    worker.rollout.synchronize_generation.assert_not_called()
    event.assert_not_called()


def test_native_p2r_harness_binds_production_live_weight_context():
    # Load the actual lightweight class without importing the GPU/vLLM launcher
    # or constructing its model. The full eight-GPU suite exercises real copies.
    source = Path(__file__).with_name("launch_test_worker.py")
    definition = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "TestRollout"
    )
    namespace = {
        "DisaggregatedRolloutControlWorker": control.DisaggregatedRolloutControlWorker,
        "PolicyToRolloutUnicastCommand": control.PolicyToRolloutUnicastCommand,
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    worker = object.__new__(namespace["TestRollout"])
    worker.replica_name = "rollout-0"
    worker.inference_stream = object()
    worker.config = SimpleNamespace(rollout=SimpleNamespace(async_r2r_sync="disabled"))
    worker._execute_p2r_recv = Mock()
    command = SimpleNamespace(dst_replica_name=worker.replica_name)
    control.DisaggregatedRolloutControlWorker.policy_to_rollout_unicast(worker, command)
    worker._execute_p2r_recv.assert_called_once_with(command, worker.inference_stream)
    assert (
        worker._paused_async_live_weights.__func__
        is control.DisaggregatedRolloutControlWorker._paused_async_live_weights
    )


@pytest.mark.parametrize("route", ["p2r", "r2r_source", "r2r_destination"])
def test_actual_weight_handlers_enter_barrier_after_initialization(route):
    worker = object.__new__(control.DisaggregatedRolloutControlWorker)
    worker.replica_name = "rollout-0"
    worker.inference_stream = object()
    worker.config = SimpleNamespace(rollout=SimpleNamespace(async_r2r_sync="disabled"))
    calls = []
    worker.lazy_initialize_rollout_engine = Mock(
        side_effect=lambda *a, **k: calls.append("init")
    )

    @contextmanager
    def barrier():
        calls.append("pause")
        yield
        calls.append("resume")

    worker._paused_async_live_weights = barrier
    worker._execute_p2r_recv = Mock(side_effect=lambda *a: calls.append("write"))
    worker._execute_rollout_broadcast = Mock(
        side_effect=lambda *a: calls.append("write")
    )
    command = SimpleNamespace(
        src_replica_name="other" if route == "r2r_destination" else worker.replica_name,
        dst_replica_name=worker.replica_name,
    )
    if route == "p2r":
        worker.policy_to_rollout_unicast(command)
    else:
        worker.broadcast_to_all_rollout_replica(command)
    assert calls == ([] if route == "r2r_source" else ["init"]) + [
        "pause",
        "write",
        "resume",
    ]


def backend_fence_methods():
    source = (
        Path(cosmos_rl.__file__).parent / "rollout/vllm_rollout/vllm_rollout_async.py"
    )
    tree = ast.parse(source.read_text())
    methods = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for method in cls.body:
                if (
                    isinstance(method, ast.FunctionDef)
                    and method.name == "synchronize_generation"
                ):
                    namespace = {"asyncio": asyncio, "torch": control.torch}
                    exec(
                        compile(
                            ast.Module(body=[method], type_ignores=[]),
                            str(source),
                            "exec",
                        ),
                        namespace,
                    )
                    methods[cls.name] = namespace[method.name]
    return methods


def test_backend_worker_fences_its_actual_cuda_context(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(control.torch.cuda, "synchronize", sync)
    backend_fence_methods()["VLLMColocateWorkerExtension"](object())
    sync.assert_called_once_with()


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_backend_rpc_is_acknowledged_or_retained_not_cancelled(outcome):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    release = threading.Event()
    calls = []

    async def rpc(name):
        calls.append(name)
        if outcome == "failure":
            raise RuntimeError("injected backend failure")
        if outcome == "timeout":
            while not release.is_set():
                await asyncio.sleep(0.001)
        return [None]

    engine = SimpleNamespace(
        _engine_initialized=SimpleNamespace(is_set=lambda: True),
        _engine_event_loop=loop,
        rollout_engine=SimpleNamespace(collective_rpc=rpc),
    )
    try:
        invoke = backend_fence_methods()["vLLMRolloutAsync"]
        if outcome == "success":
            invoke(engine, timeout=2)
            assert engine._generation_fence_future is None
        else:
            with pytest.raises(RuntimeError if outcome == "failure" else TimeoutError):
                invoke(engine, timeout=0.02)
            assert not engine._generation_fence_future.cancelled()
            if outcome == "timeout":
                assert not engine._generation_fence_future.done()
                release.set()
                engine._generation_fence_future.result(timeout=2)
        assert calls == ["synchronize_generation"]
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)
        loop.close()
