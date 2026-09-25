# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual producer + GPU consumer on two ranks/nodes, with bounded fault supervision.

torchrun --standalone --nproc-per-node=2 tests/ucxx_producer_canary.py --case healthy
Also supports --prefetch and --case producer-timeout. Supervisors observe the
producer's own exit 86 before stopping its idle peer; not launcher propagation.
"""

import argparse
import asyncio
from datetime import timedelta
import os
from pathlib import Path
import socket

import numpy as np
import torch
import torch.distributed as dist

from ucxx_context_canary import server
from ucxx_operation_canary import supervise


def worker(args):
    import cosmos_rl
    from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
    from cosmos_rl.utils.payload_transport.ucxx import ucxx_buffer as module
    from cosmos_rl.utils.payload_transport.ucxx.strategy import compose_ucxx_transport

    expected_root = os.environ.get("EXPECTED_PACKAGE_ROOT")
    if expected_root:
        assert (
            Path(cosmos_rl.__file__).resolve().parent == Path(expected_root).resolve()
        )
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    host = socket.gethostbyname(os.environ["MASTER_ADDR"])
    if host.startswith("127."):
        host = "127.0.0.1"
    assert dist.get_world_size() == 2
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu:
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)

    class Base:
        def get_policy_input(self, sample=None, rollout_output=None, *a, **kw):
            return rollout_output

    class Packer(PrefetchDataPackerMixin, Base):
        pass

    async def stall_receive(item):
        module.ucxx.init()
        endpoint = await module.ucxx.create_endpoint(
            item["_worker_ip"], item["_ucxx_port"]
        )
        await endpoint.send(np.array([item["_slot"]], dtype=np.int64))
        status = np.empty(1, dtype=np.uint8)
        await endpoint.recv(status)
        assert status[0] == 0
        print(
            "UCXX_PRODUCER_FAULT_REACHED accepted=True payload_receive=False",
            flush=True,
        )
        await asyncio.sleep(60)
        raise AssertionError("producer deadline failed")

    def run():
        producer = (
            server(elements=1 << 20, threads=2, send_timeout=3) if rank == 0 else None
        )
        packer = None
        if rank == 1 and args.case == "healthy":
            packer = Packer()
            compose_ucxx_transport(
                packer, device=device, read_timeout=10, prefetch_timeout=20
            )
        for step in range(1, 4):
            expected = np.arange(1 << 20, dtype=np.float32) + step
            metadata = [None]
            if rank == 0:
                slot = producer.write({"x": expected})
                metadata[0] = {
                    "_ucxx": True,
                    "_worker_ip": host,
                    "_ucxx_port": producer.port,
                    "_ports": producer.ports,
                    "_slot": slot,
                    "_buffer_handle": producer.get_handle(),
                }
            dist.broadcast_object_list(metadata, src=0)
            if rank == 1:
                item = metadata[0]
                if args.case == "producer-timeout":
                    asyncio.run(stall_receive(item))
                if args.prefetch:
                    packer.start_prefetch([item])
                    packer.wait_prefetch()
                result = packer.get_policy_input(rollout_output=item)
                torch.testing.assert_close(
                    result["x"], torch.from_numpy(expected).to(device)
                )
                packer.release_prefetch()
                print(f"UCXX_PRODUCER_PAYLOAD_PASS step={step} exact=True", flush=True)
            dist.barrier()
        if rank == 1:
            packer.close_transport(timeout=10)
        dist.barrier()
        if rank == 0:
            producer.stop_server(timeout=10)
            producer.close()
            producer.unlink()
        assert not module._CONTEXT_OWNERS and module.ucxx.core._ctx is None

    run()
    dist.barrier()
    print(f"UCXX_PRODUCER_WORKER_PASS rank={rank}", flush=True)
    dist.destroy_process_group()


def validate(reports, args):
    if args.case == "producer-timeout":
        assert (
            "UCXX_PRODUCER_FAULT_REACHED accepted=True payload_receive=False"
            in reports[1][1]
        ), reports
        assert reports[0][0] == 86, reports
        assert "[Transport FATAL] UCXX producer slot" in reports[0][1], reports
    else:
        assert all(code == 0 for code, _ in reports), reports
        assert "UCXX_PRODUCER_PAYLOAD_PASS step=3 exact=True" in reports[1][1]
        assert "UCXX_PRODUCER_WORKER_PASS rank=0" in reports[0][1]
        assert "UCXX_PRODUCER_WORKER_PASS rank=1" in reports[1][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", required=True, choices=("healthy", "producer-timeout")
    )
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        supervise(args, worker_script=__file__, validate=validate)
