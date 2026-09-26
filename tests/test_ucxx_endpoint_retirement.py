# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A Python close return cannot hide the retained native endpoint's error."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.utils.payload_transport.ucxx import operation, ucxx_buffer
from cosmos_rl.utils.transport_failure import TransportUnusableError


@pytest.mark.parametrize("outcome", ["healthy", "pending", "cancel", "error"])
def test_pool_close_observes_native_request_before_releasing_endpoint(
    monkeypatch, outcome
):
    failures, retained = [], []
    monkeypatch.setattr(operation, "fail_transport", failures.append)
    monkeypatch.setattr(operation, "_TERMINAL_OPERATIONS", retained)
    original = operation.UCXXOperation
    monkeypatch.setattr(
        ucxx_buffer,
        "UCXXOperation",
        lambda timeout, *args, **kwargs: original(0.08, *args, **kwargs),
    )

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        cancelled = []
        requests = []

        async def native_wait():
            requests.append(asyncio.current_task())
            entered.set()
            if outcome == "error":
                return  # Python close silently returns despite native timeout.
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        native = SimpleNamespace(
            raise_on_error=Mock(
                side_effect=RuntimeError("native close timed out")
                if outcome == "error"
                else None
            )
        )

        class Endpoint:
            _ep = native
            closed = False

            async def close(self):
                await native_wait()
                self.closed = True  # Deliberately provides no native fence.

        endpoint = Endpoint()
        client = ucxx_buffer.UCXXClient.__new__(ucxx_buffer.UCXXClient)
        client._failure = None
        client._closing = False
        client._operations = set()
        client._pool = {("host", 1): deque([endpoint])}
        task = asyncio.create_task(client.close())
        if outcome == "healthy":
            release.set()
        elif outcome == "cancel":
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
        try:
            if outcome == "healthy":
                await task
                assert entered.is_set() and endpoint.closed and not client._pool
                assert not failures
            else:
                with pytest.raises(TransportUnusableError):
                    await task
                assert failures and not cancelled
                assert endpoint.closed == (outcome == "error")
                assert list(client._pool[("host", 1)]) == [endpoint]
                assert any(native is item for op in retained for item in op.owners)
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)

    asyncio.run(run())
