# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank real-UCXX consumer ownership gate, on one or two nodes.

torchrun --standalone --nproc-per-node=2 tests/ucxx_operation_canary.py \
    --case healthy [--prefetch]

Cases: healthy, stale (safe rejection followed by success), warm-stall (accepted
payload never arrives). Supervisors observe the receiver's own fatal exit before
cleaning up its peer. This does not claim launcher/scheduler failure propagation
or validate production producer/global-context shutdown.
"""

import argparse
import asyncio
from datetime import timedelta
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist


def free_port():
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def worker(args):
    import cosmos_rl
    import ucxx
    from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
    from cosmos_rl.utils.payload_transport.ucxx.strategy import compose_ucxx_transport

    expected_root = os.environ.get("EXPECTED_PACKAGE_ROOT")
    if expected_root:
        assert (
            Path(cosmos_rl.__file__).resolve().parent == Path(expected_root).resolve()
        )
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    assert torch.cuda.is_available() and dist.get_world_size() == 2

    async def serve():
        ucxx.init()
        done = asyncio.Event()
        errors = []

        async def connection(endpoint):
            try:
                while True:
                    slot = np.empty(1, np.int64)
                    await endpoint.recv(slot)
                    step = int(slot[0])
                    assert 1 <= step <= 3
                    if step == 2 and args.case == "stale":
                        await endpoint.send(np.array([1], np.uint8))
                        continue
                    await endpoint.send(np.array([0], np.uint8))
                    if step == 2 and args.case == "warm-stall":
                        print("UCXX_INJECT accepted_payload_stalled=True", flush=True)
                        await asyncio.Event().wait()
                    payload = np.arange(4096, dtype=np.float32) + step
                    await endpoint.send(payload.view(np.uint8))
                    print(f"UCXX_SENT step={step}", flush=True)
                    if step == 3:
                        break
            except BaseException as error:
                errors.append(error)
            finally:
                await endpoint.close()
                done.set()

        listener = ucxx.create_listener(connection, port=0)
        address = [(os.environ["MASTER_ADDR"], listener.port)]
        dist.broadcast_object_list(address, 0)
        await done.wait()
        listener.close()
        assert not errors, errors

    if rank == 0:
        asyncio.run(serve())
        # The fixture has one endpoint, and all its requests finished before
        # this point. This is not Cosmos' process-global shutdown implementation.
        ucxx.core._ctx.worker.stop_progress_thread()
        ucxx.reset()
    else:
        address = [None]
        dist.broadcast_object_list(address, 0)
        host, port = address[0]

        class Base:
            def get_policy_input(self, sample=None, rollout_output=None, *a, **kw):
                return rollout_output

        class Packer(PrefetchDataPackerMixin, Base):
            pass

        packer = Packer()
        compose_ucxx_transport(
            packer, device=device, read_timeout=5, prefetch_timeout=30
        )
        prior = []
        for step in range(1, 4):
            metadata = {
                "_ucxx": True,
                "_worker_ip": host,
                "_ucxx_port": port,
                "_slot": step,
                "_buffer_handle": {
                    "schema": [{"name": "x", "shape": [4096], "dtype": "float32"}]
                },
            }
            if args.prefetch:
                packer.start_prefetch([metadata])
                if args.case == "warm-stall" and step == 2:
                    # Native watchdog must fire even without wait_prefetch().
                    time.sleep(45)
                packer.wait_prefetch()
            result = packer.get_policy_input(rollout_output=metadata)
            if args.case == "warm-stall" and step == 2:
                print("UNSAFE_CONTINUATION", flush=True)
                raise AssertionError("uncertain native read became an ordinary result")
            if args.case == "stale" and step == 2:
                assert result is None
                print("UCXX_STALE_PASS recoverable=True", flush=True)
            else:
                torch.testing.assert_close(
                    result["x"],
                    torch.arange(4096, device=device, dtype=torch.float32) + step,
                )
                prior.append((step, result["x"]))
                # Previous decoded tensors remain independent when pinned
                # storage is recycled for subsequent reads.
                for old_step, old_value in prior:
                    torch.testing.assert_close(
                        old_value,
                        torch.arange(4096, device=device, dtype=torch.float32)
                        + old_step,
                    )
                print(f"UCXX_PAYLOAD_PASS step={step} exact=True", flush=True)
            packer.release_prefetch()
        packer.close_transport(timeout=10)
        if ucxx.core._ctx is not None:
            ucxx.core._ctx.worker.stop_progress_thread()
            ucxx.reset()
    dist.barrier()
    print(f"UCXX_WORKER_PASS rank={rank} case={args.case}", flush=True)
    dist.destroy_process_group()


def supervise(args, *, worker_script=None, validate=None):
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    port = [free_port() if rank == 0 else None]
    dist.broadcast_object_list(port, 0)
    env = dict(os.environ, MASTER_PORT=str(port[0]))
    env.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    command = [
        sys.executable,
        str(Path(worker_script or __file__).resolve()),
        "--worker",
        "--case",
        args.case,
    ]
    if args.prefetch:
        command.append("--prefetch")
    if getattr(args, "cpu", False):
        command.append("--cpu")
    child = None
    try:
        with tempfile.TemporaryFile(mode="w+t") as output:
            child = subprocess.Popen(
                command, env=env, stdout=output, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + 120
            failed_at = None
            while True:
                states = [None, None]
                dist.all_gather_object(states, child.poll())
                if all(code is not None for code in states):
                    break
                if any(code is not None and code != 0 for code in states):
                    failed_at = failed_at or time.monotonic()
                if failed_at is not None and time.monotonic() - failed_at > 3:
                    if child.poll() is None:
                        child.terminate()
                    break
                assert time.monotonic() < deadline, (
                    "UCXX canary did not reach an outcome"
                )
                time.sleep(0.1)
            child.wait(timeout=5)
            output.seek(0)
            reports = [None, None]
            dist.all_gather_object(reports, (child.returncode, output.read()))
        if rank == 0:
            print("\n".join(report[1] for report in reports), flush=True)
        if validate is not None:
            validate(reports, args)
        elif args.case == "warm-stall":
            assert "UCXX_INJECT" in reports[0][1]
            assert reports[1][0] == 86, reports
            assert "[Transport FATAL] UCXX read" in reports[1][1]
            assert "UCXX_PAYLOAD_PASS step=1 exact=True" in reports[1][1]
            assert "UNSAFE_CONTINUATION" not in reports[1][1]
        else:
            assert all(code == 0 for code, _ in reports), reports
            assert "UCXX_PAYLOAD_PASS step=3 exact=True" in reports[1][1]
            if args.case == "stale":
                assert "UCXX_STALE_PASS recoverable=True" in reports[1][1]
        print(
            f"UCXX_OPERATION_PASS rank={rank} case={args.case} prefetch={args.prefetch}",
            flush=True,
        )
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", required=True, choices=("healthy", "stale", "warm-stall")
    )
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    worker(args) if args.worker else supervise(args)
