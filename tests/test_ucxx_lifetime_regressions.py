# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-entrypoint negative controls executable on the pre-fix revision."""

import asyncio
import contextlib
import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cosmos_rl.utils.payload_transport.ucxx import ucxx_buffer
from cosmos_rl.utils.payload_transport.ucxx.strategy import UCXXTransportStrategy
from cosmos_rl.utils.trajectory import TensorSpec
from cosmos_rl.utils.transport_failure import TransportUnusableError


@pytest.fixture(autouse=True)
def observe_terminal_without_exiting(monkeypatch):
    monkeypatch.setattr(ucxx_buffer, "_CONTEXT_OWNERS", {}, raising=False)
    monkeypatch.setattr(ucxx_buffer, "_CONTEXT_FAILURE", None, raising=False)
    name = "cosmos_rl.utils.payload_transport.ucxx.operation"
    if importlib.util.find_spec(name) is not None:
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "fail_transport", Mock())
        monkeypatch.setattr(module, "_TERMINAL_OPERATIONS", [])


@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_uncertain_read_is_terminal_without_cancelling_native_waiter(monkeypatch, mode):
    async def run():
        release, entered = asyncio.Event(), asyncio.Event()
        native_tasks, cancelled, closed, attempts = [], [], [], []

        class Endpoint:
            async def send(self, array):
                pass

            async def recv(self, array):
                if array.size == 1:
                    array[0] = 0
                    return
                native_tasks.append(asyncio.current_task())
                entered.set()
                try:
                    await release.wait()
                    array[:] = 3
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise

            async def close(self):
                closed.append(True)

        async def create(host, port):
            attempts.append(port)
            return Endpoint()

        monkeypatch.setattr(ucxx_buffer, "UCXX_AVAILABLE", True)
        monkeypatch.setattr(
            ucxx_buffer,
            "ucxx",
            SimpleNamespace(init=lambda: None, create_endpoint=create),
        )
        client = ucxx_buffer.UCXXClient()
        client._acquire_pinned = lambda size: torch.empty(size, dtype=torch.uint8)
        future = asyncio.create_task(
            client.read(
                "host",
                1,
                2,
                [TensorSpec((4,), np.uint8, "x")],
                timeout=0.04,
                ports=[1, 2],
            )
        )
        await entered.wait()
        if mode == "cancel":
            future.cancel()
        try:
            with pytest.raises(TransportUnusableError):
                await future
            assert not cancelled and not closed
            assert attempts == [1]
        finally:
            release.set()
            await asyncio.gather(*native_tasks, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("prefetch", [False, True])
def test_copy_error_does_not_fall_back_and_recycle_live_storage(monkeypatch, prefetch):
    backing = torch.arange(4, dtype=torch.uint8)
    payload = {"x": backing.numpy(), "_pinned_buf": backing}

    async def read(*args, **kwargs):
        return dict(payload)

    client = SimpleNamespace(read=read, return_pinned=Mock())
    strategy = UCXXTransportStrategy()
    strategy._client = client
    strategy._device = "cuda:0"
    strategy._read_timeout = 2
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: object())
    calls = []

    def partial_copy(value, *args, **kwargs):
        calls.append(value)
        if len(calls) == 1:
            raise RuntimeError("copy failed after native issue")
        return value.clone()

    monkeypatch.setattr(torch.Tensor, "to", partial_copy)
    metadata = {"_ucxx": True, "_worker_ip": "host", "_ucxx_port": 1, "_slot": 2}
    with pytest.raises(TransportUnusableError):
        if prefetch:
            strategy.fetch_batch([(0, metadata)])
        else:
            strategy.sync_fetch(metadata)
    assert len(calls) == 1
    client.return_pinned.assert_not_called()
