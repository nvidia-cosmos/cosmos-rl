# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real Torch collectives with controlled MPI rank metadata, not a TRT engine.

Run four CPU processes or two GPU processes on each of two nodes. The MPI initializer runs
against a reassigned two-rank communicator while Torch must retain global ranks
0..3. CPU mode uses Gloo; GPU mode retains production's NCCL/Gloo backend pair.
This does not validate MPI launch, dynamic-port discovery or TRT multi-node support.
"""

import argparse
from datetime import timedelta
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

import cosmos_rl


class Comm:
    def __init__(self, rank, size, local_rank, local_size):
        self.rank, self.size = rank, size
        self.local_rank, self.local_size = local_rank, local_size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Split_type(self, **kwargs):
        return Comm(self.local_rank, self.local_size, self.local_rank, self.local_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--expected-package", required=True)
    args = parser.parse_args()
    assert Path(cosmos_rl.__file__).resolve().parent == Path(args.expected_package)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 4
    local_rank = rank % 2
    cuda_rank = int(os.environ["LOCAL_RANK"])
    if not args.cpu:
        assert int(os.environ["LOCAL_WORLD_SIZE"]) == 2 and cuda_rank == local_rank
    global_comm = Comm(rank, world, cuda_rank, int(os.environ["LOCAL_WORLD_SIZE"]))
    local_comm = Comm(local_rank, 2, cuda_rank, global_comm.local_size)
    mpi = ModuleType("mpi4py")
    mpi.MPI = SimpleNamespace(COMM_WORLD=global_comm)
    trt = ModuleType("tensorrt_llm")
    trt.__path__ = []
    utils = ModuleType("tensorrt_llm._utils")
    utils.mpi_broadcast = lambda value, root: value
    env = {
        "COSMOS_WORLD_SIZE": str(world),
        "COSMOS_LOCAL_WORLD_SIZE": "2",
        "COSMOS_RDZV_ENDPOINT": f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    }
    original_init = dist.init_process_group
    original_set_device = torch.cuda.set_device
    devices = []

    def set_device(value):
        devices.append(value)
        assert value == local_rank
        if not args.cpu:
            original_set_device(value)

    def initialize(backend, **kwargs):
        assert kwargs["rank"] == rank and kwargs["world_size"] == world, kwargs
        return original_init(
            "gloo" if args.cpu else backend,
            timeout=timedelta(seconds=60),
            **kwargs,
        )

    try:
        with (
            patch.dict(
                sys.modules,
                {"mpi4py": mpi, "tensorrt_llm": trt, "tensorrt_llm._utils": utils},
            ),
            patch.dict(os.environ, env),
            patch.object(torch.cuda, "set_device", set_device),
            patch.object(dist, "init_process_group", initialize),
        ):
            helper = importlib.import_module("cosmos_rl.utils.mpi_distributed")
            helper.set_mpi_comm(local_comm)
            helper.init_distributed_with_MPI()
            assert devices == [local_rank]
            assert dist.get_rank() == rank and dist.get_world_size() == world
            seen = [None] * world
            dist.all_gather_object(seen, (dist.get_rank(), int(os.environ["RANK"])))
            assert seen == [(i, i) for i in range(world)], seen
            for step in range(1, 6):
                for device in ["cpu"] if args.cpu else ["cpu", f"cuda:{cuda_rank}"]:
                    value = torch.full((16,), float((rank + 1) * step), device=device)
                    dist.all_reduce(value)
                    torch.testing.assert_close(
                        value, torch.full_like(value, 10.0 * step)
                    )
            dist.barrier()
            print(
                f"MPI_GLOBAL_RANK_PASS rank={rank} local={local_rank} cpu={args.cpu} parity=True",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
