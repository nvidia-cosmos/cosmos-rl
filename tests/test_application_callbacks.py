# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Callbacks bind to execution context and telemetry cannot hold lifecycle work."""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time
from unittest.mock import Mock

import pytest

from cosmos_rl.utils.application_callbacks import (
    CallbackContext,
    CallbackFactory,
    CallbackRegistration,
    CallbackRegistry,
)


def test_required_callbacks_return_values_and_preserve_original_error():
    error = RuntimeError("required failure")
    registry = CallbackRegistry(
        {"value": lambda value: value + 1, "failure": Mock(side_effect=error)},
        CallbackContext(role="rollout", global_rank=3),
    )
    assert registry.invoke("value", 4) == 5
    with pytest.raises(RuntimeError) as raised:
        registry.invoke("failure")
    assert raised.value is error
    registry.close()
    with pytest.raises(RuntimeError, match="closed"):
        registry.invoke("value", 1)


def test_factory_creates_once_per_execution_worker_with_explicit_context():
    contexts, handlers = [], []
    barrier = threading.Barrier(2)

    def create(context):
        contexts.append(context)
        handler = Mock(return_value=context.worker_id)
        handlers.append(handler)
        return handler

    registry = CallbackRegistry(
        {"read": CallbackFactory(create)},
        CallbackContext(
            role="reward", replica_name="source", global_rank=2, is_validation=True
        ),
    )

    def execute():
        barrier.wait(timeout=2)
        return registry.invoke("read"), registry.invoke("read")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: execute(), range(2)))
    assert len(contexts) == 2
    assert {context.process_id for context in contexts} == {os.getpid()}
    assert all(
        context.role == "reward"
        and context.global_rank == 2
        and context.replica_name == "source"
        and context.is_validation
        for context in contexts
    )
    assert all(first == second for first, second in results)
    assert len({first for first, _ in results}) == 2
    assert [handler.call_count for handler in handlers] == [2, 2]
    registry.close()


def test_new_threads_never_reuse_an_old_worker_factory_context():
    contexts = []

    def create(context):
        contexts.append(context)
        return lambda: context.worker_id

    registry = CallbackRegistry(
        {"read": CallbackFactory(create)}, CallbackContext(role="reward")
    )
    identities = []
    for _ in range(8):
        with ThreadPoolExecutor(max_workers=1) as pool:
            identities.append(pool.submit(registry.invoke, "read").result())
    assert len(contexts) == len(set(identities)) == 8
    registry.close()


_FORK_REGISTRY = None
_FORK_EVIDENCE_PATH = None


class _ProcessHandler:
    def __init__(self, context):
        self.context = context
        with open(_FORK_EVIDENCE_PATH, "a") as evidence:
            evidence.write(f"created {os.getpid()} {context.process_id}\n")

    def __call__(self):
        return self.context.process_id

    def close(self):
        with open(_FORK_EVIDENCE_PATH, "a") as evidence:
            evidence.write(f"closed {os.getpid()} {self.context.process_id}\n")


def _process_invoke():
    return _FORK_REGISTRY.invoke("read")


def test_factory_binds_and_closes_in_process_worker_after_parent_initialization(
    tmp_path,
):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    global _FORK_REGISTRY, _FORK_EVIDENCE_PATH
    context = multiprocessing.get_context("fork")
    _FORK_EVIDENCE_PATH = tmp_path / "factory_lifetime.txt"
    _FORK_REGISTRY = CallbackRegistry(
        {"read": CallbackFactory(_ProcessHandler)}, CallbackContext(role="reward")
    )
    assert _process_invoke() == os.getpid()
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        child_pid = executor.submit(_process_invoke).result(timeout=5)
        assert child_pid != os.getpid()
    _FORK_REGISTRY.close()
    evidence = [
        (kind, int(pid), int(context_pid))
        for kind, pid, context_pid in (
            line.split() for line in _FORK_EVIDENCE_PATH.read_text().splitlines()
        )
    ]
    assert set(evidence) == {
        (kind, pid, pid)
        for kind in ("created", "closed")
        for pid in (os.getpid(), child_pid)
    }


def test_best_effort_never_accepts_borrowed_or_required_work():
    registry = CallbackRegistry(
        {"observe": CallbackRegistration(lambda *_: None, policy="best_effort")},
        CallbackContext(role="controller"),
    )
    with pytest.raises(ValueError, match="must be required"):
        registry.invoke("observe", {"borrowed": "payload"})
    with pytest.raises(ValueError, match="must be required"):
        registry.invoke("observe", {}, borrowed=False, required=True)
    registry.close()


@pytest.mark.parametrize("factory", [False, True])
def test_required_callbacks_and_factories_reject_unfinished_async_readers(factory):
    async def asynchronous(*args):
        return None

    registry = CallbackRegistry(
        {"read": CallbackFactory(asynchronous) if factory else asynchronous},
        CallbackContext(role="reward"),
    )
    with pytest.raises(TypeError, match="complete synchronously"):
        registry.invoke("read")
    registry.close()


def test_best_effort_factory_and_handler_do_not_block_caller_and_full_queue_drops(
    caplog,
):
    factory_entered, release_factory = threading.Event(), threading.Event()
    handled = []

    def create(context):
        assert context.worker_id.startswith("ApplicationCallback:observe")
        factory_entered.set()
        assert release_factory.wait(2)
        return lambda event: handled.append(event)

    registry = CallbackRegistry(
        {
            "observe": CallbackRegistration(
                CallbackFactory(create), policy="best_effort", queue_capacity=1
            )
        },
        CallbackContext(role="controller"),
    )
    started = time.monotonic()
    registry.invoke("observe", {"event": 1}, borrowed=False)
    assert time.monotonic() - started < 0.5
    try:
        assert factory_entered.wait(1)
        registry.invoke("observe", {"event": 2}, borrowed=False)
        assert "queue full" in caplog.text
    finally:
        release_factory.set()
        registry.close(timeout_s=2)
    assert handled == [{"event": 1}]


def test_best_effort_handler_exception_drops_event_and_next_event_still_runs(caplog):
    observed = threading.Event()

    def observe(event):
        if event == 1:
            raise ValueError("diagnostics failure")
        observed.set()

    registry = CallbackRegistry(
        {"observe": CallbackRegistration(observe, policy="best_effort")},
        CallbackContext(role="controller"),
    )
    registry.invoke("observe", 1, borrowed=False)
    registry.invoke("observe", 2, borrowed=False)
    assert observed.wait(1)
    registry.close()
    assert "event dropped" in caplog.text


def test_factory_close_finishes_all_handlers_and_keeps_required_failure_sticky():
    original = RuntimeError("close failed")
    first = Mock()
    first.close.side_effect = original
    second = Mock()
    registry = CallbackRegistry(
        {
            "first": CallbackFactory(lambda _: first),
            "second": CallbackFactory(lambda _: second),
        },
        CallbackContext(role="reward"),
    )
    registry.invoke("first")
    registry.invoke("second")
    for _ in range(2):
        with pytest.raises(RuntimeError) as error:
            registry.close()
        assert error.value is original
    first.close.assert_called_once()
    second.close.assert_called_once()


def test_best_effort_close_timeout_is_bounded_for_detached_telemetry(caplog):
    entered, release = threading.Event(), threading.Event()

    def observe(_):
        entered.set()
        assert release.wait(2)

    registry = CallbackRegistry(
        {"observe": CallbackRegistration(observe, policy="best_effort")},
        CallbackContext(role="controller"),
    )
    registry.invoke("observe", {"small": "event"}, borrowed=False)
    try:
        assert entered.wait(1)
        started = time.monotonic()
        registry.close(timeout_s=0.01)
        assert time.monotonic() - started < 0.5
        assert "did not drain" in caplog.text
    finally:
        release.set()
