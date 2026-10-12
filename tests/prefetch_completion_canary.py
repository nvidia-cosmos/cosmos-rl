# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Starve the timer callback; completion itself must enforce the deadline.

Run under torchrun on one or two nodes. Each supervisor requires its worker's
actual exit status. This checks prefetch completion/ownership, not peer transport
or scheduler failure propagation. CUDA arms retain a real pending device copy.
"""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist


def worker(args):
    from cosmos_rl.utils.payload_transport import prefetch_mixin as module

    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if device.type == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
        payload = torch.arange(32, device=device, dtype=torch.float32)
        stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize(device)
    else:
        payload = torch.arange(32, dtype=torch.float32)
        stream = None
    clock = [100.0]
    module.time = SimpleNamespace(monotonic=lambda: clock[0])

    class PausedTimer:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def cancel(self):
            pass

    module.threading.Timer = PausedTimer

    class Packer(module.PrefetchDataPackerMixin):
        def _filter_prefetch_tasks(self, rollouts):
            return [(0, "ref")]

        def _fetch_batch(self, tasks):
            if stream is not None:
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    torch.cuda._sleep(200_000_000)
                    result = payload.clone()
                    event = torch.cuda.Event()
                    event.record(stream)
                    if args.case == "late":
                        assert not event.query(), "pending-copy injection did not start"
                        # Test fixture owns the stream/event as well as the result.
                        self.pending_copy = (stream, event, result)
                        print("PREFETCH_PENDING_COPY injected=True", flush=True)
                    else:
                        stream.synchronize()
            else:
                result = payload.clone()
            clock[0] = 106.0 if args.case == "late" else 101.0
            print(
                f"PREFETCH_COMPLETION case={args.case} timer_callback=False", flush=True
            )
            return {"ref": {"x": result}}

    packer = Packer()
    packer._transport_strategy = SimpleNamespace(
        before_join=lambda: None, on_prefetch_complete=lambda *a: None
    )
    packer._setup_prefetch(prefetch_timeout=5)
    if args.prepared:
        future = packer.start_prepared_prefetch(
            ["ref"], lambda: packer._preparation_local.cache
        )
        value = future.result(timeout=10)
        assert future._prefetch_handoff.wait(5)
    else:
        packer.start_prefetch(["ref"])
        packer.wait_prefetch()
        value = packer._prefetch_cache
    if args.case == "late":
        print("UNSAFE_LATE_RESULT", flush=True)
        raise AssertionError("late completion published success")
    torch.testing.assert_close(value["ref"]["x"], payload)
    clock[0] = 200.0
    assert packer._prefetch_failure is None
    packer.shutdown_prefetch(join_timeout=5)
    print("PREFETCH_HEALTHY_PASS", flush=True)


def main(args):
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    try:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            args.case,
        ]
        if args.cpu:
            command.append("--cpu")
        if args.prepared:
            command.append("--prepared")
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        print(result.stdout, result.stderr, flush=True)
        expected = 86 if args.case == "late" else 0
        assert result.returncode == expected, result
        if args.case == "late":
            assert "[Transport FATAL] prefetch batch" in result.stderr
            assert "UNSAFE_LATE_RESULT" not in result.stdout
            if not args.cpu:
                assert "PREFETCH_PENDING_COPY injected=True" in result.stdout
        else:
            assert "PREFETCH_HEALTHY_PASS" in result.stdout
        dist.barrier()
        print(
            f"PREFETCH_COMPLETION_PASS rank={rank} case={args.case} prepared={args.prepared}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("healthy", "late"), required=True)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
