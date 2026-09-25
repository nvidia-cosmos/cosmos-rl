# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native lifetime probes independent of optional UCXX/CUDA installation."""

import asyncio
import contextlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cosmos_rl.utils.payload_transport.ucxx import operation as module
from cosmos_rl.utils.payload_transport.ucxx import ucxx_buffer
from cosmos_rl.utils.payload_transport.ucxx.strategy import UCXXTransportStrategy
from cosmos_rl.utils.trajectory import TensorSpec
from cosmos_rl.utils.transport_failure import TransportUnusableError


@pytest.fixture
def terminal(monkeypatch):
    failures = []
    monkeypatch.setattr(module, "fail_transport", failures.append)
    retained = []
    monkeypatch.setattr(module, "_TERMINAL_OPERATIONS", retained)
    return failures, retained


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ucxx_buffer, "_CONTEXT_OWNERS", {})
    monkeypatch.setattr(ucxx_buffer, "_CONTEXT_FAILURE", None)
    monkeypatch.setattr(ucxx_buffer, "UCXX_AVAILABLE", True)
    monkeypatch.setattr(ucxx_buffer, "ucxx", SimpleNamespace(init=lambda: None))
    result = ucxx_buffer.UCXXClient()
    # Exercise the real pool; make only cudaHostAlloc independent of CUDA.
    empty = torch.empty

    def pageable(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", pageable)
    return result


class Endpoint:
    def __init__(self, *, status=0, fail=None, release=None):
        self._ep = SimpleNamespace(raise_on_error=Mock())
        self.status = status
        self.fail = fail
        self.release = release
        self.entered = asyncio.Event()
        self.cancelled = False
        self.closed = False
        self.calls = 0
        self.target = None

    async def send(self, value):
        pass

    async def recv(self, value):
        self.calls += 1
        if self.calls % 2:
            value[0] = self.status
            return
        self.target = value
        self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
            if self.fail:
                raise self.fail
            value[:] = 7
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def close(self):
        self.closed = True


def install_endpoint(monkeypatch, endpoint):
    created = []

    async def create(host, port):
        created.append(port)
        return endpoint

    monkeypatch.setattr(ucxx_buffer.ucxx, "create_endpoint", create, raising=False)
    return created


def schema():
    return [TensorSpec((4,), np.uint8, "x")]


def test_healthy_completion_reuses_endpoint_and_buffer(client, monkeypatch, terminal):
    async def run():
        endpoint = Endpoint()
        created = install_endpoint(monkeypatch, endpoint)
        data = await client.read("host", 1, 2, schema())
        assert (data["x"] == 7).all()
        backing = data["_pinned_buf"]
        client.return_pinned(backing)
        second = await client.read("host", 1, 3, schema())
        assert second["_pinned_buf"] is backing
        assert created == [1] and not endpoint.closed
        assert not client._operations
        await client.close()
        assert endpoint.closed
        with pytest.raises(RuntimeError, match="closing"):
            await client.read("host", 1, 4, schema())

    asyncio.run(run())
    assert terminal == ([], [])


@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_pending_read_never_cancelled_recycled_closed_or_retried(
    client, monkeypatch, terminal, mode
):
    async def run():
        release = asyncio.Event()
        endpoint = Endpoint(release=release)
        created = install_endpoint(monkeypatch, endpoint)
        future = asyncio.create_task(
            client.read("host", 1, 2, schema(), timeout=0.05, ports=[1, 2])
        )
        await endpoint.entered.wait()
        if mode == "cancel":
            future.cancel()
        with pytest.raises(TransportUnusableError):
            await future
        assert created == [1]
        assert not endpoint.cancelled and not endpoint.closed
        assert not client._pool and not client._pinned_pool
        failures, retained = terminal
        assert len(failures) == len(retained) == 1
        owner = retained[0]
        assert any(value is endpoint.target for value in owner.owners)
        assert any(value is endpoint for value in owner.owners)
        with pytest.raises(TransportUnusableError):
            await client.read("host", 1, 3, schema())
        with pytest.raises(TransportUnusableError):
            client.return_pinned(torch.empty(4))
        with pytest.raises(TransportUnusableError):
            await client.close()
        # Late native completion does not clear the terminal latch or ownership.
        release.set()
        tasks = [value for value in owner.owners if isinstance(value, asyncio.Task)]
        await asyncio.gather(*tasks)
        assert (endpoint.target == 7).all()
        assert owner.failure and owner.owners
        assert not endpoint.closed

    asyncio.run(run())


def test_clean_stale_slot_is_recoverable(client, monkeypatch, terminal):
    async def run():
        endpoint = Endpoint(status=1)
        created = install_endpoint(monkeypatch, endpoint)
        with pytest.raises(ucxx_buffer.StaleSlotError):
            await client.read("host", 1, 2, schema(), ports=[1, 2])
        assert created == [1] and not endpoint.closed
        assert not client._failure and not client._operations
        await client.close()

    asyncio.run(run())
    assert terminal == ([], [])


def test_completed_connection_failure_can_rotate(client, monkeypatch, terminal):
    async def run():
        error = type("UCXXConnectionResetError", (Exception,), {})
        created = []
        endpoint = Endpoint()

        async def create(host, port):
            created.append(port)
            if port == 1:
                raise error("connection rejected before native buffer issue")
            return endpoint

        monkeypatch.setattr(ucxx_buffer.ucxx, "create_endpoint", create, raising=False)
        assert (await client.read("host", 1, 2, schema(), ports=[1, 2]))["x"].size == 4
        assert created == [1, 2]
        await client.close()

    asyncio.run(run())
    assert terminal == ([], [])


def test_nested_library_timeout_is_not_completion_proof(client, monkeypatch, terminal):
    async def run():
        endpoint = Endpoint(fail=TimeoutError("nested waiter timed out"))
        created = install_endpoint(monkeypatch, endpoint)
        with pytest.raises(TransportUnusableError):
            await client.read("host", 1, 2, schema(), ports=[1, 2])
        assert created == [1] and not endpoint.closed

    asyncio.run(run())
    assert len(terminal[0]) == 1


def test_failed_batch_retains_other_concurrent_read_tasks(terminal):
    async def run():
        release, entered = asyncio.Event(), asyncio.Event()
        cancelled = []

        async def read(host, port, slot, schema, **kwargs):
            if slot == 1:
                entered.set()
                try:
                    await release.wait()
                    return {"x": np.zeros(1, np.uint8)}
                except asyncio.CancelledError:
                    cancelled.append(slot)
                    raise
            await entered.wait()
            raise TransportUnusableError("another peer lost native completion")

        strategy = UCXXTransportStrategy()
        strategy._device = "cpu"
        strategy._read_timeout = 1
        strategy._client = SimpleNamespace(read=read)
        tasks = [
            (slot, {"_worker_ip": "host", "_ucxx_port": 1, "_slot": slot})
            for slot in (1, 2)
        ]
        with pytest.raises(TransportUnusableError):
            await strategy._fetch_all(tasks)
        assert not cancelled and strategy._failure
        owned_tasks = next(
            value for value in terminal[1][0].owners if isinstance(value, list)
        )
        assert any(not task.done() for task in owned_tasks)
        release.set()
        await asyncio.gather(*owned_tasks, return_exceptions=True)
        assert not cancelled

    asyncio.run(run())


def test_close_does_not_cancel_active_read(client, monkeypatch, terminal):
    async def run():
        release = asyncio.Event()
        endpoint = Endpoint(release=release)
        install_endpoint(monkeypatch, endpoint)
        task = asyncio.create_task(client.read("host", 1, 2, schema()))
        await endpoint.entered.wait()
        with pytest.raises(RuntimeError, match="active native"):
            await client.close()
        assert not endpoint.cancelled and not endpoint.closed
        release.set()
        await task
        await client.close()
        assert endpoint.closed

    asyncio.run(run())
    assert terminal == ([], [])


def test_watchdog_runs_while_event_loop_is_blocked(terminal):
    owner = object()
    operation = module.UCXXOperation(0.03, "blocked loop", owners=(owner,))
    time.sleep(0.08)  # blocks asyncio/consumer polling, not the watchdog thread
    assert terminal[0]
    assert operation is terminal[1][0] and operation.owners == [owner]
    with pytest.raises(TransportUnusableError):
        operation.complete()


def test_delayed_timer_cannot_accept_late_success(monkeypatch, terminal):
    monkeypatch.setattr(threading.Timer, "start", lambda self: None)
    operation = module.UCXXOperation(1, "late native result", owners=(object(),))
    operation.deadline.deadline = time.monotonic() - 1
    with pytest.raises(TransportUnusableError):
        operation.complete()
    assert terminal[0] and operation.owners


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_budget_rejected_before_allocation(client, timeout):
    client._acquire_pinned = Mock(side_effect=AssertionError("must not allocate"))
    with pytest.raises(ValueError, match="finite and positive"):
        asyncio.run(client.read("host", 1, 2, schema(), timeout=timeout))
    client._acquire_pinned.assert_not_called()


@pytest.mark.parametrize(
    "specs",
    [
        [TensorSpec((-1,), np.uint8, "x")],
        [TensorSpec((4,), object, "x")],
        [TensorSpec((4,), np.uint8, "_pinned_buf")],
        schema() + schema(),
    ],
)
def test_invalid_schema_rejected_before_allocation(client, specs):
    client._acquire_pinned = Mock(side_effect=AssertionError("must not allocate"))
    with pytest.raises(ValueError, match="schema"):
        asyncio.run(client.read("host", 1, 2, specs))
    client._acquire_pinned.assert_not_called()


@pytest.fixture
def device_copy(monkeypatch):
    """Real tensor/view math; fake only device issue/completion for CPU probes."""
    consumer = UCXXTransportStrategy()
    consumer._client = SimpleNamespace(return_pinned=Mock())
    consumer._device = "cuda:0"
    consumer._read_timeout = 0.05
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    original_to = torch.Tensor.to

    def to(tensor, device, **kwargs):
        if torch.device(device).type == "cuda":
            return tensor.clone()
        return original_to(tensor, device, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    event = Mock()
    event.query.return_value = True
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    pinned = torch.arange(4, dtype=torch.uint8)
    payload = {"x": pinned.numpy(), "_pinned_buf": pinned}
    return consumer, payload, event


def test_device_completion_precedes_buffer_recycle(device_copy, terminal):
    consumer, payload, event = device_copy
    consumer._client.return_pinned.side_effect = lambda buf: event.query.assert_called()
    data = consumer._copy_to_device(payload)
    consumer._client.return_pinned.assert_called_once_with(payload["_pinned_buf"])
    assert torch.equal(data["x"], torch.arange(4, dtype=torch.uint8))
    payload["_pinned_buf"].fill_(9)
    assert data["x"].tolist() == [0, 1, 2, 3]
    assert terminal == ([], [])


@pytest.mark.parametrize("failure", ["copy", "record", "query", "pending"])
def test_device_failure_never_falls_back_or_recycles(
    device_copy, terminal, monkeypatch, failure
):
    consumer, payload, event = device_copy
    if failure == "copy":
        monkeypatch.setattr(
            torch.Tensor, "to", Mock(side_effect=RuntimeError("issued"))
        )
    elif failure == "pending":
        event.query.return_value = False
    else:
        getattr(event, failure).side_effect = RuntimeError("issued")
    with pytest.raises(TransportUnusableError):
        consumer._copy_to_device(payload)
    consumer._client.return_pinned.assert_not_called()
    assert consumer._failure
    assert any(value is payload for value in terminal[1][0].owners)
    assert any(value is payload["_pinned_buf"] for value in terminal[1][0].owners)
    with pytest.raises(TransportUnusableError):
        consumer.sync_fetch({})


def test_malformed_array_rejected_before_any_device_work(device_copy, terminal):
    consumer, payload, event = device_copy
    payload["x"] = np.zeros(4, dtype=object)
    with pytest.raises(ValueError, match="Unsupported"):
        consumer._copy_to_device(payload)
    event.record.assert_not_called()
    assert not consumer._failure and terminal == ([], [])


def test_default_watchdog_exits_without_native_cleanup():
    program = """
import time
from cosmos_rl.utils.payload_transport.ucxx.operation import UCXXOperation
operation = UCXXOperation(0.05, 'UCXX blocked native canary', owners=(bytearray(8),))
time.sleep(2)
raise AssertionError('watchdog did not terminate')
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, timeout=30
    )
    assert result.returncode == 86, result.stderr.decode()
    assert b"UCXX blocked native canary" in result.stderr
    assert b"exiting without native cleanup" in result.stderr


def test_allocation_failure_is_recoverable_only_after_proven_drain(
    device_copy, terminal, monkeypatch
):
    consumer, payload, event = device_copy
    copy = Mock(side_effect=torch.OutOfMemoryError("allocation rejected"))
    monkeypatch.setattr(torch.Tensor, "to", copy)
    with pytest.raises(ValueError, match="after safe drain"):
        consumer._copy_to_device(payload)
    copy.assert_called_once()
    event.record.assert_called_once()
    event.query.assert_called()
    consumer._client.return_pinned.assert_called_once()
    assert not consumer._failure and terminal == ([], [])
