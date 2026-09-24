# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank native UCXX baseline, independent of Cosmos transport classes.

Run with torchrun on two nodes to distinguish runtime connectivity failures
from Cosmos transport lifecycle failures. Uses CPU bytes over native UCXX.
"""

import asyncio
import json
import os
from datetime import timedelta

import numpy as np
import torch.distributed as dist
import ucxx


async def exchange():
    ucxx.init()
    rank = dist.get_rank()
    listener = None
    done = asyncio.Event()

    async def serve(endpoint):
        try:
            data = np.empty(16, dtype=np.int64)
            await asyncio.wait_for(endpoint.recv(data), timeout=15)
            np.testing.assert_array_equal(data, np.arange(16, dtype=np.int64))
            await asyncio.wait_for(endpoint.send(data + 1), timeout=15)
            print(json.dumps({"rank": rank, "native_ucxx_exchange": True}), flush=True)
        finally:
            await endpoint.close()
            done.set()

    address = [None]
    if rank == 0:
        listener = ucxx.create_listener(serve, port=0)
        address[0] = (os.environ["MASTER_ADDR"], listener.port)
    dist.broadcast_object_list(address, src=0)
    try:
        if rank == 0:
            await asyncio.wait_for(done.wait(), timeout=30)
        else:
            endpoint = await asyncio.wait_for(
                ucxx.create_endpoint(*address[0]), timeout=15
            )
            try:
                await asyncio.wait_for(
                    endpoint.send(np.arange(16, dtype=np.int64)), timeout=15
                )
                result = np.empty(16, dtype=np.int64)
                await asyncio.wait_for(endpoint.recv(result), timeout=15)
                np.testing.assert_array_equal(result, np.arange(16, dtype=np.int64) + 1)
                print(
                    json.dumps({"rank": rank, "native_ucxx_exchange": True}), flush=True
                )
            finally:
                await endpoint.close()
    finally:
        if listener is not None:
            listener.close()


if __name__ == "__main__":
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        print(
            json.dumps(
                {
                    "ucxx_version": ucxx.__version__,
                    "ucx_version": ucxx.get_ucx_version(),
                }
            ),
            flush=True,
        )
        asyncio.run(exchange())
    finally:
        ucxx.reset()
        dist.destroy_process_group()
