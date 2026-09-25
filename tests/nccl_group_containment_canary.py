# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU managed-group parity and terminal fault injection; run with Python."""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import threading

import torch
import torch.distributed as dist

from cosmos_rl.utils import pynccl


def worker(case):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size in (1, 2)
    uid = [pynccl.create_nccl_uid() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = pynccl.create_nccl_comm(uid[0], rank, world_size, timeout_ms=10_000)
    stream = torch.cuda.Stream()
    held = []
    for step in range(2):
        with torch.cuda.stream(stream):
            held = [
                torch.full((4096,), step + 1.0 if rank == 0 else -1.0, device="cuda")
                for _ in range(3)
            ]
            if step and case == "start-error" and rank == 0:
                original = pynccl._nccl.ncclGroupStart

                def start():
                    original()
                    raise ValueError("injected after native GroupStart")

                pynccl._nccl.ncclGroupStart = start
            with pynccl.nccl_group(comm, timeout_ms=1500 if step else 10_000):
                for tensor in held:
                    pynccl.nccl_broadcast(tensor, 0, comm, stream=stream)
                if step and rank == 0 and case != "healthy":
                    print(f"GROUP_INJECT case={case}", flush=True)
                    if case == "body-error":
                        raise ValueError("injected group body")
                    if case == "abort":
                        timer = threading.Timer(0.1, pynccl.nccl_abort, args=(comm,))
                        timer.daemon = True
                        timer.start()
                    threading.Event().wait(20)
                    raise AssertionError("Group deadline did not terminate the process")
        stream.synchronize()
        for tensor in held:
            torch.testing.assert_close(tensor, torch.full_like(tensor, step + 1.0))
        print(f"GROUP_PARITY_PASS rank={rank} step={step} tensors=3", flush=True)
        dist.barrier()
    pynccl.nccl_abort(comm)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("healthy", "body-error", "start-error", "deadline", "abort"),
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    assert hasattr(pynccl, "nccl_group"), (
        f"Managed group API absent from {pynccl.__file__}"
    )
    print(f"GROUP_PACKAGE path={pynccl.__file__}", flush=True)
    if args.worker:
        worker(args.case)
        return
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.ranks}",
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            args.case,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    for rank in range(args.ranks):
        assert f"GROUP_PARITY_PASS rank={rank} step=0 tensors=3" in output
    if args.case == "healthy":
        assert result.returncode == 0, result.returncode
        for rank in range(args.ranks):
            assert f"GROUP_PARITY_PASS rank={rank} step=1 tensors=3" in output
    else:
        assert result.returncode != 0 and "[Transport FATAL]" in output
        assert "GROUP_PARITY_PASS rank=0 step=1" not in output
        cause = {
            "body-error": "ValueError: injected group body",
            "start-error": "ValueError: injected after native GroupStart",
            "deadline": "accepted operation deadline expired",
            "abort": "abort requested inside active NCCL group",
        }[args.case]
        assert cause in output, output
    print(f"NCCL_GROUP_GATE_PASS case={args.case} recovery=False", flush=True)


if __name__ == "__main__":
    main()
