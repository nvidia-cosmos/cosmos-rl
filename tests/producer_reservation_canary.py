# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP fetch/report/departure, Redis dispatch and tiny CPU/CUDA optimizer.

Two ranks support one or two nodes. Dataset/model values and membership setup
are controlled; trainer ACKs travel over Gloo fixture coordination. This is not
native rollout-engine recovery or proof that departed payload storage survives.
"""

import argparse
import asyncio
from datetime import timedelta
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import msgpack
import pytest
import redis
import requests
import torch
import torch.distributed as dist
import uvicorn

import cosmos_rl
from cosmos_rl.dispatcher import run_web_panel as web
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.command import Command
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.dispatcher.protocol import Role, RolloutRequest
from cosmos_rl.dispatcher.publication import ControllerPublisher
from cosmos_rl.dispatcher.replica import Replica
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_ROLLOUT_SUFFIX,
)
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from dispatch_commit_canary import worker
from test_producer_reservations import allow_another_prompt, completed_request
from test_terminal_drain_protocol import _rollout_atom
from test_training_fetch_receipts import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("healthy", "departure", "lost-fetch", "lost-report", "strict-refill"),
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
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    server = thread = sock = redis_process = publisher = patcher = reader = None
    address, injected = [None], []
    try:
        if rank == 0:
            patcher = pytest.MonkeyPatch()
            controller, _, _ = setup(patcher)
            manager = controller.policy_status_manager
            manager.total_steps, manager.remain_samples_num = 3, 6
            manager.should_weight_sync_after_train_ack = lambda *_: False
            manager.set_discard_refill_hook(
                controller.register_discarded_samples_for_refill
            )
            controller.rollout_status_manager.trigger_rebuild_mesh = lambda *_: None
            survivor = _rollout_atom("survivor", 0, 0)
            survivor.report_session_id = "survivor-session"
            controller.rollout_status_manager.rollout_replicas["survivor"] = Replica(
                "survivor", Role.ROLLOUT, [survivor]
            )
            if args.case != "strict-refill":
                allow_another_prompt(controller)
            with socket.socket() as probe:
                probe.bind(("0.0.0.0", 0))
                redis_port = probe.getsockname()[1]
            redis_process = subprocess.Popen(
                [
                    "redis-server",
                    "--bind",
                    "0.0.0.0",
                    "--protected-mode",
                    "no",
                    "--port",
                    str(redis_port),
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                ],
                stdout=subprocess.DEVNULL,
            )
            raw = redis.Redis(host="127.0.0.1", port=redis_port, socket_timeout=2)
            deadline = time.monotonic() + 10
            while True:
                try:
                    raw.ping()
                    break
                except redis.ConnectionError:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            failures = []
            publisher = ControllerPublisher(
                RedisStreamHandler(["127.0.0.1"], redis_port),
                on_failure=failures.append,
            )
            manager.redis_handler = publisher
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
                result = await web.get_batched_prompt(
                    n,
                    rank_in_mesh=rank_in_mesh,
                    controller_execution_id=controller_execution_id,
                    src_replica_name=src_replica_name,
                    src_global_rank=src_global_rank,
                    fetch_session_id=fetch_session_id,
                    fetch_sequence=fetch_sequence,
                )
                if args.case == "lost-fetch" and not injected:
                    assert manager.samples_on_the_fly == 4
                    injected.append("fetch")
                    print("INJECTED lost-fetch reserved=4", flush=True)
                    return JSONResponse(
                        status_code=503, content={"error": "lost-fetch-reply"}
                    )
                return result

            @app.post(COSMOS_API_ROLLOUT_SUFFIX)
            async def report(request: RolloutRequest):
                result = await web.put_rollout_group(request)
                if args.case == "lost-report" and not injected:
                    assert manager.rollout_buffer.qsize() == 2
                    injected.append("report")
                    print("INJECTED lost-report accepted=2", flush=True)
                    return JSONResponse(
                        status_code=503, content={"error": "lost-report-reply"}
                    )
                return result

            @app.post("/fixture/depart")
            async def depart():
                await controller.unregister("source")
                return {"outstanding": manager.samples_on_the_fly}

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
            address[0] = (os.environ["MASTER_ADDR"], sock.getsockname()[1], redis_port)
        dist.broadcast_object_list(address)
        host, port, redis_port = address[0]
        reader = RedisStreamHandler([host], redis_port)
        if rank == 1:
            client = APIClient(Role.ROLLOUT, remote_ips=[host], remote_port=port)
            client._registered_replica_name, client._registered_global_rank = (
                "source",
                0,
            )
            client._report_session_id = "session"
            payloads, end = client.get_next_prompt(
                1 if args.case == "strict-refill" else 2
            )
            assert not end and len(payloads) == (
                1 if args.case == "strict-refill" else 2
            )
            assert all(item["training_completion_slots"] == [0, 1] for item in payloads)
            if args.case == "strict-refill":
                response = requests.post(
                    f"http://{host}:{port}/fixture/depart", timeout=5
                )
                assert response.status_code == 200 and response.json() == {
                    "outstanding": 0
                }
                client = APIClient(Role.ROLLOUT, remote_ips=[host], remote_port=port)
                client._registered_replica_name, client._registered_global_rank = (
                    "survivor",
                    0,
                )
                client._report_session_id = "survivor-session"
                payloads, end = client.get_next_prompt(1)
                assert (
                    len(payloads) == 1
                    and payloads[0]["weight_version"] == 0
                    and not end
                )
            count = 2 if args.case == "healthy" else 1
            for payload in payloads[:count]:
                payload = RLPayload.model_validate(payload)
                payload.weight_version = (
                    0  # Actual adopted model, not requested work version.
                )
                request = completed_request(payload, 2)
                request.src_replica_name = client._registered_replica_name
                request.report_session_id = request.report_sequence = None
                reward = (torch.tensor(2.0, device=args.device) * 3).item()
                request.payloads[0].rewards = [reward, reward + 1]
                assert client.post_rollout_completion(request)
                assert client.post_rollout_completion(
                    request
                )  # Lost reply/replayed report.
            if args.case not in ("healthy", "strict-refill"):
                response = requests.post(
                    f"http://{host}:{port}/fixture/depart", timeout=5
                )
                assert response.status_code == 200 and response.json() == {
                    "outstanding": 2
                }
            policy, acknowledgements = worker(torch.device(args.device), reader)
        dist.barrier()
        steps = 2 if args.case == "healthy" else 1
        if rank == 0:
            assert (
                manager.samples_on_the_fly
                == manager.rollout_buffer.qsize()
                == 2 * steps
            )
            assert controller.data_fetcher.get_batched_prompt.call_count == (
                2 if args.case == "strict-refill" else 1
            )
            assert len(injected) == int(args.case in ("lost-fetch", "lost-report"))
        for step in range(1, steps + 1):
            if rank == 0:
                PolicyStatusManager.try_trigger_data_fetch_and_training(manager)
                asyncio.run(publisher.flush())
                assert not failures
                assert not manager.training_dispatches[step].settled
            dist.barrier()
            acknowledgement = None
            if rank == 1:
                commands, payloads = (
                    reader.subscribe_command("policy-0"),
                    reader.subscribe_rollout("policy-0"),
                )
                assert len(commands) == 1 and len(payloads) == 2
                for payload in payloads:
                    policy.data_queue.put(
                        Rollout.model_validate(msgpack.unpackb(payload))
                    )
                assert not policy.execute_data_fetch(Command.depack(commands[0]))
                assert policy.trainer.updates == policy.trainer.scheduler_calls == step
                acknowledgement = acknowledgements[-1]
            gathered = [None, None] if rank == 0 else None
            dist.gather_object(acknowledgement, gathered, dst=0)
            if rank == 0:
                for _ in range(2):
                    manager.train_ack(*gathered[1], controller.rollout_status_manager)
                assert manager.training_dispatches[step].settled
                assert manager.samples_on_the_fly == 2 * (steps - step)
        dist.barrier()
        print(
            f"PRODUCER_RESERVATION_PASS rank={rank} case={args.case} updates={steps} device={args.device}",
            flush=True,
        )
    finally:
        if server is not None:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()
            sock.close()
        if publisher is not None:
            asyncio.run(publisher.close())
        if redis_process is not None:
            redis_process.terminate()
            redis_process.wait(timeout=5)
        if patcher is not None:
            patcher.undo()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
