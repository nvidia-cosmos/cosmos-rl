# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank real HTTP training-fetch retries and distributed fetch rejection."""

import argparse
from datetime import timedelta
import os
from pathlib import Path
from queue import Queue
import socket
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pytest
import torch
import torch.distributed as dist
import uvicorn

import cosmos_rl
from cosmos_rl.dispatcher import run_web_panel as web
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils.api_suffix import COSMOS_API_NEXT_PROMPT_SUFFIX
from test_training_fetch_receipts import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("healthy", "lost-fetch", "sampler-failure", "fetch-rejected"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-package-root", type=Path, required=True)
    args = parser.parse_args()
    assert (
        Path(cosmos_rl.__file__).resolve().parent
        == args.expected_package_root.resolve()
    )
    if args.device == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl" if args.device == "cuda" else "gloo",
        timeout=timedelta(seconds=90),
    )
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    server = thread = sock = patcher = None
    connection, seen, injected = [None], [], []
    try:
        if rank == 0:
            patcher = pytest.MonkeyPatch()
            controller, atom, _ = setup(patcher)
            if args.case == "sampler-failure":
                original_fetch = controller.data_fetcher.get_batched_prompt.side_effect

                def failed(*args, **kwargs):
                    consumed, _ = original_fetch(*args, **kwargs)
                    assert [payload.prompt_idx for payload in consumed] == [7]
                    injected.append("sampler-failure")
                    print("INJECTED sampler-failure after-advance=True", flush=True)
                    raise RuntimeError("sampler advanced then failed")

                controller.data_fetcher.get_batched_prompt.side_effect = failed
            app = FastAPI()

            @app.get(COSMOS_API_NEXT_PROMPT_SUFFIX)
            async def fetch(
                n: int,
                src_replica_name: str,
                src_global_rank: int,
                fetch_session_id: str,
                fetch_sequence: int,
                rank_in_mesh: int | None = None,
                controller_execution_id: str | None = None,
            ):
                seen.append((fetch_session_id, fetch_sequence))
                try:
                    result = await web.get_batched_prompt(
                        n,
                        src_replica_name=src_replica_name,
                        src_global_rank=src_global_rank,
                        fetch_session_id=fetch_session_id,
                        fetch_sequence=fetch_sequence,
                        rank_in_mesh=rank_in_mesh,
                        controller_execution_id=controller_execution_id,
                    )
                except RuntimeError:
                    assert args.case == "sampler-failure"
                    return JSONResponse(
                        status_code=500, content={"error": "sampler-failure"}
                    )
                if args.case == "lost-fetch" and not injected:
                    assert len(result["payloads_list"]) == 1
                    injected.append("lost-fetch")
                    print("INJECTED lost-fetch reserved=True", flush=True)
                    return JSONResponse(
                        status_code=503, content={"error": "lost-reply"}
                    )
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
            connection[0] = (os.environ["MASTER_ADDR"], sock.getsockname()[1])
        dist.broadcast_object_list(connection)
        host, port = connection[0]
        client = APIClient("ROLLOUT", remote_ips=[host], remote_port=port)
        client._registered_replica_name, client._registered_global_rank = "source", 0
        client._report_session_id = (
            "old-session" if args.case == "fetch-rejected" else "session"
        )
        if args.case == "fetch-rejected":
            from cosmos_rl.rollout.worker.rollout_control import (
                DisaggregatedRolloutControlWorker,
            )

            worker = SimpleNamespace(
                global_rank=rank,
                api_client=client,
                _prompt_fetch_lock=threading.Lock(),
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
            with pytest.raises(RuntimeError, match="Training prompt fetch failed"):
                DisaggregatedRolloutControlWorker.request_new_prompts(
                    worker, 1, Queue()
                )
        elif rank == 1:
            if args.case == "sampler-failure":
                with pytest.raises(RuntimeError, match="bounded HTTP"):
                    client.get_next_prompt(1)
                assert client._fetch_failed
            else:
                payloads, end = client.get_next_prompt(1)
                assert not end and [p["prompt_idx"] for p in payloads] == [7]
                assert (
                    payloads[0]["weight_version"] == 0 and client._fetch_sequence == 1
                )
                proof = torch.tensor(payloads[0]["prompt_idx"], device=args.device) * 2
                assert proof.item() == 14
        proof = torch.tensor(1.0, device=args.device)
        dist.all_reduce(proof)
        assert proof.item() == 2.0
        if rank == 0:
            expected = {
                "healthy": 1,
                "lost-fetch": 2,
                "sampler-failure": 3,
                "fetch-rejected": 1,
            }[args.case]
            session = "old-session" if args.case == "fetch-rejected" else "session"
            assert seen == [(session, 0)] * expected
            assert controller.data_fetcher.get_batched_prompt.call_count == int(
                args.case != "fetch-rejected"
            )
            assert controller.policy_status_manager.samples_on_the_fly == (
                2 if args.case in ("healthy", "lost-fetch") else 0
            )
            assert atom.rollout_fetch_receipt.failed == (args.case == "sampler-failure")
        print(
            f"TRAINING_FETCH_PASS rank={rank} case={args.case} device={args.device}",
            flush=True,
        )
        dist.barrier()
    finally:
        if server is not None:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()
        if sock is not None:
            sock.close()
        if patcher is not None:
            patcher.undo()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
