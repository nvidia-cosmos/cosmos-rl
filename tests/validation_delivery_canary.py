# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two real HTTP producers with CPU/CUDA work and committed-but-lost replies.

Run under torchrun with two ranks. This exercises real validation routes,
sampler, client retries and rank-owned receipts, not a full model/simulator.
"""

import argparse
from datetime import timedelta
import os
from pathlib import Path
from queue import Queue
import socket
import threading
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import cosmos_rl
from cosmos_rl.dispatcher import run_web_panel
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.protocol import Role, ValidationReportRequest
from cosmos_rl.rollout.validation import ValidationSession
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_VALIDATION_REPORT_SUFFIX,
)
from test_validation_delivery_contract import bind_controller, manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("healthy", "empty", "lost-fetch", "lost-report", "fetch-rejected"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-package-root", type=Path, required=True)
    args = parser.parse_args()
    assert (
        Path(cosmos_rl.__file__).resolve().parent
        == args.expected_package_root.resolve()
    )
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl" if args.device == "cuda" else "gloo",
        timeout=timedelta(seconds=90),
    )
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    indices = [] if args.case == "empty" else [7, 7, 2, 3, 4]
    server = thread = sock = None
    connection = [None]
    injected = set()
    if rank == 0:
        import pytest

        instance, round_id, _ = manager(indices, extra_replicas=("b",))
        patcher = pytest.MonkeyPatch()
        bind_controller(patcher, instance)
        app = FastAPI()

        @app.get(COSMOS_API_NEXT_PROMPT_SUFFIX)
        async def get(
            n: int,
            validation_step: int,
            validation_round_id: str,
            src_replica_name: str,
            fetch_sequence: int,
            rank_in_mesh: int | None = None,
        ):
            result = await run_web_panel.get_batched_prompt(
                n,
                validation_step,
                rank_in_mesh,
                validation_round_id,
                src_replica_name,
                fetch_sequence,
            )
            key = ("fetch", src_replica_name, fetch_sequence)
            if args.case == "fetch-rejected":
                assert result.status_code == 409
                assert b"Stale validation round" in result.body
                injected.add("stale-fetch")
                print("INJECTED rejected-stale-fetch", flush=True)
            if (
                args.case == "lost-fetch"
                and fetch_sequence == 0
                and key not in injected
            ):
                assert isinstance(result, dict)
                injected.add(key)
                print(f"INJECTED committed-fetch rank={src_replica_name}", flush=True)
                return JSONResponse(status_code=503, content={"error": "lost_reply"})
            return result

        @app.post(COSMOS_API_VALIDATION_REPORT_SUFFIX)
        async def post(request: ValidationReportRequest):
            result = await run_web_panel.validation_report(request)
            key = ("report", request.src_replica_name, request.report_sequence)
            if (
                args.case == "lost-report"
                and request.report_sequence == 0
                and key not in injected
            ):
                assert isinstance(result, dict)
                injected.add(key)
                print(
                    f"INJECTED committed-report rank={request.src_replica_name}",
                    flush=True,
                )
                return JSONResponse(status_code=503, content={"error": "lost_reply"})
            return result

        sock = socket.socket()
        sock.bind(("0.0.0.0", 0))
        sock.listen()
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        connection[0] = (os.environ["MASTER_ADDR"], sock.getsockname()[1], round_id)
    dist.broadcast_object_list(connection)
    address, port, round_id = connection[0]
    try:
        client = APIClient(Role.ROLLOUT, remote_ips=[address], remote_port=port)
        client.max_retries = 2
        if args.case == "fetch-rejected":
            from cosmos_rl.rollout.worker.rollout_control import (
                DisaggregatedRolloutControlWorker,
            )

            worker = SimpleNamespace(
                global_rank=rank,
                _prompt_fetch_lock=threading.Lock(),
                _validation_session=ValidationSession(client, "stale", 2, "a", 0),
                parallel_dims=SimpleNamespace(
                    mesh={"dp": SimpleNamespace(size=lambda: 1)}
                ),
                config=SimpleNamespace(
                    train=SimpleNamespace(
                        train_policy=SimpleNamespace(
                            data_dispatch_as_rank_in_mesh=False
                        )
                    )
                ),
            )
            try:
                DisaggregatedRolloutControlWorker.request_new_prompts(
                    worker,
                    2,
                    Queue(),
                    validation_step=2,
                )
            except RuntimeError as error:
                assert "Validation prompt fetch failed" in str(error)
            else:
                raise AssertionError(
                    "A rejected validation fetch was silently accepted"
                )
            dist.barrier()
            if rank == 0:
                assert injected == {"stale-fetch"}
                assert not instance.validation_round._pending
                assert not instance.validation_round.complete
            print(
                f"VALIDATION_DELIVERY_PASS rank={rank} case={args.case} device={args.device} rejected=True",
                flush=True,
            )
            dist.barrier()
            return
        session = ValidationSession(client, round_id, 2, "a" if rank == 0 else "b", 0)
        counts = torch.zeros(2, device=args.device, dtype=torch.float64)
        while True:
            raw, end = session.fetch(2)
            payloads = [RLPayload.model_validate(value) for value in raw]
            for payload in payloads:
                value = (
                    torch.tensor(float(payload.prompt_idx), device=args.device) * 2 + 1
                )
                payload.rewards, payload.advantages = [value.item()], [0.0]
                counts += counts.new_tensor([1, value.item()])
            if payloads:
                session.report(payloads)
            if end:
                break
        session.report([], is_end=True)
        dist.all_reduce(counts)
        assert counts.tolist() == [len(indices), 2 * sum(indices) + len(indices)]
        dist.barrier()
        if rank == 0:
            assert instance.validation_round.complete
            assert instance.validation_round.reported_prompts == len(indices)
            assert not instance.val_report_data and not instance.data_fetcher.val_iters
            instance.try_trigger_data_fetch_and_training.assert_called_once()
            assert len(injected) == (2 if args.case.startswith("lost-") else 0)
        print(
            f"VALIDATION_DELIVERY_PASS rank={rank} case={args.case} device={args.device} prompts={len(indices)}",
            flush=True,
        )
        dist.barrier()
    finally:
        if server is not None:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()
            sock.close()
            patcher.undo()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
