# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Accepted producer sends retain their slot until native completion."""

import asyncio
from types import SimpleNamespace
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from cosmos_rl.utils.payload_transport.ucxx import operation, ucxx_buffer
from cosmos_rl.utils.transport_failure import TransportUnusableError


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_producer_send_budget_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        ucxx_buffer.UCXXBufferConfig(send_timeout=timeout)


@pytest.mark.parametrize("phase", ["status", "payload"])
@pytest.mark.parametrize("outcome", ["healthy", "cancel", "timeout", "error"])
def test_slot_owned_until_accepted_send_completes(monkeypatch, outcome, phase):
    failures = []
    monkeypatch.setattr(operation, "fail_transport", failures.append)
    monkeypatch.setattr(operation, "_TERMINAL_OPERATIONS", [])

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        native_tasks, cancellations = [], []
        owner = object.__new__(ucxx_buffer.UCXXBuffer)
        owner.config = SimpleNamespace(
            send_timeout=0.05 if outcome == "timeout" else 2.0
        )
        owner._shutdown_flag = threading.Event()
        owner._endpoints_lock = threading.Lock()
        owner._active_endpoints = []
        owner._thread_metrics_lock = threading.Lock()
        owner._thread_metrics = {}
        raw = np.arange(8, dtype=np.uint8)
        owner._buffer = SimpleNamespace(
            schema=[object()],
            read_raw=Mock(return_value=raw),
            mark_consumed=Mock(),
            release_reading=Mock(),
            close=Mock(),
        )

        class Endpoint:
            _ep = SimpleNamespace(raise_on_error=Mock())

            async def close(self):
                pass

            async def recv(self, array):
                array[0] = 0

            async def send(self, array):
                if (array.size == 1) != (phase == "status"):
                    return
                native_tasks.append(asyncio.current_task())
                entered.set()
                if outcome == "error":
                    owner._shutdown_flag.set()
                    raise RuntimeError("injected after native send issue")
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellations.append(True)
                    raise
                owner._shutdown_flag.set()

        endpoint = Endpoint()
        future = asyncio.create_task(owner._handle_connection(endpoint))
        await entered.wait()
        if outcome == "healthy":
            release.set()
        elif outcome == "cancel":
            future.cancel()
        elif outcome == "error":
            owner._shutdown_flag.set()
        try:
            if outcome == "healthy":
                await asyncio.wait_for(future, 1)
                owner._buffer.mark_consumed.assert_called_once_with(0)
                assert not failures
            else:
                with pytest.raises(TransportUnusableError):
                    await asyncio.wait_for(future, 0.5)
                assert failures
                assert not cancellations
                owner._buffer.mark_consumed.assert_not_called()
                assert any(
                    raw is item
                    for retained in operation._TERMINAL_OPERATIONS
                    for item in retained.owners
                )
            owner._buffer.release_reading.assert_not_called()
        finally:
            owner._shutdown_flag.set()
            release.set()
            await asyncio.gather(future, *native_tasks, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("published", [False, True])
def test_unexpected_loop_failure_retains_published_listener(monkeypatch, published):
    failures, retained = [], []
    monkeypatch.setattr(operation, "fail_transport", failures.append)
    monkeypatch.setattr(operation, "_TERMINAL_OPERATIONS", retained)
    owner = object.__new__(ucxx_buffer.UCXXBuffer)
    owner._thread_metrics_lock = threading.Lock()
    owner._thread_metrics = {}
    owner._listeners = [None]
    owner._server_loops = [None]
    owner._handler_tasks_per_thread = [[]]
    owner._buffer = Mock()

    async def broken_main(*args):
        if published:
            owner._listeners[0] = object()
        raise RuntimeError("unexpected server-loop failure")

    owner._async_server_main = broken_main
    try:
        if published:
            with pytest.raises(TransportUnusableError):
                owner._run_server_loop(0, 12345)
            assert failures and owner._server_failure
            assert not owner._server_loops[0].is_closed()
            assert any(owner is item for op in retained for item in op.owners)
        else:
            owner._run_server_loop(0, 12345)
            assert not failures and owner._server_loops[0] is None
    finally:
        if owner._server_loops[0] is not None:
            owner._server_loops[0].close()


def test_idle_header_is_one_retained_request_until_native_retirement(monkeypatch):
    failures = []
    monkeypatch.setattr(operation, "fail_transport", failures.append)
    monkeypatch.setattr(operation, "_TERMINAL_OPERATIONS", [])

    async def run():
        entered, closed = asyncio.Event(), asyncio.Event()
        receives, cancelled = [], []
        owner = object.__new__(ucxx_buffer.UCXXBuffer)
        owner.config = SimpleNamespace(send_timeout=1)
        owner._shutdown_flag = threading.Event()
        owner._endpoints_lock = threading.Lock()
        owner._active_endpoints = []
        owner._buffer = Mock()
        owner._HANDLER_RECV_TIMEOUT = 0.01
        owner._HANDLER_MAX_IDLE_CYCLES = 100

        class Endpoint:
            _ep = SimpleNamespace(raise_on_error=Mock())

            async def recv(self, array):
                receives.append(array)
                entered.set()
                try:
                    await closed.wait()
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise
                raise RuntimeError("completed native cancellation after close")

            async def close(self):
                closed.set()

        endpoint = Endpoint()
        task = asyncio.create_task(owner._handle_connection(endpoint))
        await entered.wait()
        await asyncio.sleep(0.08)  # Several old per-receive cancellation cycles.
        assert len(receives) == 1 and not cancelled
        owner._shutdown_flag.set()
        await asyncio.wait_for(task, 1)
        assert closed.is_set() and not failures and not owner._active_endpoints
        assert not cancelled
        owner._buffer.read_raw.assert_not_called()
        owner._buffer.mark_consumed.assert_not_called()

    asyncio.run(run())
