# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real local UCXX endpoints: two producers and an idle pooled client share a worker."""

import argparse
import asyncio
import os
from pathlib import Path
import socket
import threading
import uuid

import numpy as np

from cosmos_rl.utils.payload_transport.ucxx import ucxx_buffer as module
from cosmos_rl.utils.trajectory import TensorSpec
import cosmos_rl


def server(*, elements=4, threads=2, send_timeout=30, start=True):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    buffer = module.UCXXBuffer(
        module.UCXXBufferConfig(
            buffer_name=f"context-canary-{uuid.uuid4().hex}",
            max_entries=2,
            entry_size_bytes=elements * 4 + 64,
            schema=[TensorSpec((elements,), np.float32, "x")],
            port=port,
            n_server_threads=threads,
            send_timeout=send_timeout,
        )
    )
    buffer._local_ip = "127.0.0.1"
    if start:
        buffer.start_server()
    return buffer


async def main():
    first, second = server(), server()
    client = module.UCXXClient()
    expected = np.arange(4, dtype=np.float32)

    async def fetch(producer):
        for port in producer.ports:
            slot = producer.write({"x": expected})
            result = await client.read("127.0.0.1", port, slot, producer.config.schema)
            np.testing.assert_array_equal(result["x"], expected)
            client.return_pinned(result["_pinned_buf"])

    await fetch(first)
    await fetch(second)
    await asyncio.to_thread(first.stop_server, timeout=10)
    first.close()
    first.unlink()
    assert len(module._CONTEXT_OWNERS) == 2
    await fetch(second)
    await client.close()
    assert len(module._CONTEXT_OWNERS) == 1
    await asyncio.to_thread(second.stop_server, timeout=10)
    second.close()
    second.unlink()
    assert not module._CONTEXT_OWNERS
    assert module.ucxx.core._ctx is None
    print(
        "UCXX_SHARED_CONTEXT_NATIVE_PASS reads=6 retired_producers=2 loops=4",
        flush=True,
    )


async def producer_send(case):
    expected = np.arange(1 << 20, dtype=np.float32)
    producer = server(elements=expected.size, threads=1, send_timeout=1)
    peer_owner = object()
    module._acquire_ucxx_context(peer_owner)
    endpoint = await module.ucxx.create_endpoint("127.0.0.1", producer.port)
    slot = producer.write({"x": expected})
    await endpoint.send(np.array([slot], dtype=np.int64))
    status = np.empty(1, dtype=np.uint8)
    await endpoint.recv(status)
    assert status[0] == 0
    print(f"UCXX_PRODUCER_NATIVE_ISSUED case={case} accepted=True", flush=True)
    if case == "producer-timeout":
        # Real rendezvous send cannot complete until its peer posts the payload
        # receive. Keep the endpoint alive, but deliberately never post it.
        await asyncio.sleep(10)
        raise AssertionError("producer native send escaped its terminal deadline")
    received = np.empty(expected.nbytes, dtype=np.uint8)
    await endpoint.recv(received)
    np.testing.assert_array_equal(received.view(np.float32), expected)
    retirement = module.UCXXOperation(5, "canary peer close", owners=(endpoint,))
    await module._close_endpoint_owned(endpoint, retirement)
    retirement.complete()
    module._release_ucxx_context(peer_owner)
    await asyncio.to_thread(producer.stop_server, timeout=5)
    producer.close()
    producer.unlink()
    print("UCXX_PRODUCER_NATIVE_PASS exact=True", flush=True)


async def partial_start():
    producer = server(start=False)
    original = module.ucxx.create_listener

    def create(*args, **kwargs):
        if threading.current_thread().name == f"UCXXServer-{producer._base_port + 1}":
            raise RuntimeError("injected partial listener startup")
        return original(*args, **kwargs)

    module.ucxx.create_listener = create
    try:
        try:
            producer.start_server(timeout=0.3)
            raise AssertionError("partial startup must remain observable")
        except RuntimeError as error:
            assert "failed to start" in str(error)
    finally:
        module.ucxx.create_listener = original
    await asyncio.to_thread(producer.stop_server, timeout=5)
    assert not module._CONTEXT_OWNERS and module.ucxx.core._ctx is None
    # A proven partial-start drain permits a fresh context; it is not a failed
    # accepted native operation or an elastic recovery claim.
    producer.start_server()
    client = module.UCXXClient()
    expected = np.arange(4, dtype=np.float32)
    slot = producer.write({"x": expected})
    result = await client.read("127.0.0.1", producer.port, slot, producer.config.schema)
    np.testing.assert_array_equal(result["x"], expected)
    await client.close()
    await asyncio.to_thread(producer.stop_server, timeout=5)
    producer.close()
    producer.unlink()
    print("UCXX_PARTIAL_START_NATIVE_PASS restart_exact=True", flush=True)


if __name__ == "__main__":
    if "EXPECTED_PACKAGE_ROOT" in os.environ:
        assert Path(cosmos_rl.__file__).resolve().parent == Path(
            os.environ["EXPECTED_PACKAGE_ROOT"]
        )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("shared", "producer-healthy", "producer-timeout", "partial-start"),
        default="shared",
    )
    args = parser.parse_args()
    if args.case == "shared":
        asyncio.run(main())
    elif args.case == "partial-start":
        asyncio.run(partial_start())
    else:
        asyncio.run(producer_send(args.case))
