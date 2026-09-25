# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP report receipts with controlled CPU/CUDA result production.

Two torchrun ranks host a real controller admission path and a remote producer.
This checks exactly-once admission, not distributed optimizer atomicity.
"""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import socket
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pytest
import torch
import torch.distributed as dist
import uvicorn

import cosmos_rl
from cosmos_rl.dispatcher import run_web_panel as web
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.utils.api_suffix import COSMOS_API_ROLLOUT_SUFFIX
from test_rollout_report_receipts import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "healthy",
            "lost-reply",
            "changed-report",
            "partial-settlement",
            "retired-source",
        ),
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
        assert torch.cuda.is_available()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl" if args.device == "cuda" else "gloo",
        timeout=timedelta(seconds=90),
    )
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    server = thread = sock = patcher = None
    connection = [None]
    seen = []
    injected = []
    try:
        if rank == 0:
            patcher = pytest.MonkeyPatch()
            controller, atom, reserved_report = setup(patcher)
            if args.case == "partial-settlement":

                async def partial(*arguments, **kwargs):
                    controller.stat_n_samples += 1
                    injected.append("partial-settlement")
                    print("INJECTED partial-settlement after-mutation=True", flush=True)
                    raise RuntimeError("injected partial settlement")

                patcher.setattr(web, "_apply_rollout_group", partial)
            app = FastAPI()

            @app.post(COSMOS_API_ROLLOUT_SUFFIX)
            async def post(request: RolloutRequest):
                seen.append((request.report_session_id, request.report_sequence))
                if args.case == "retired-source" and not injected:
                    controller.rollout_status_manager.rollout_replicas.pop("source")
                    injected.append("retired-source")
                    print("INJECTED retired-source before-admission=True", flush=True)
                try:
                    result = await web.put_rollout_group(request)
                except RuntimeError:
                    assert args.case == "partial-settlement"
                    return JSONResponse(
                        status_code=500, content={"error": "partial-settlement"}
                    )
                if args.case == "lost-reply" and not injected:
                    assert result == {"message": "Rollout put"}
                    injected.append("lost-reply")
                    print("INJECTED lost-reply committed=True", flush=True)
                    return JSONResponse(
                        status_code=503, content={"error": "lost-reply"}
                    )
                if args.case == "changed-report" and isinstance(result, JSONResponse):
                    assert result.status_code == 409
                    injected.append("changed-report")
                    print("INJECTED changed-report rejected=True", flush=True)
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
            connection[0] = (
                os.environ["MASTER_ADDR"],
                sock.getsockname()[1],
                reserved_report.payloads[0].training_work_id,
            )
        dist.broadcast_object_list(connection)
        if rank == 1:
            host, port, work_id = connection[0]
            client = APIClient("ROLLOUT", remote_ips=[host], remote_port=port)
            client._registered_replica_name, client._registered_global_rank = (
                "source",
                0,
            )
            client._report_session_id = "session"
            reward = (
                torch.tensor(7.0, device=args.device, dtype=torch.float64) * 2 + 1
            ).item()
            request = RolloutRequest(
                src_replica_name="source",
                payloads=[
                    RLPayload(
                        prompt_idx=7,
                        training_work_id=work_id,
                        training_completion_slots=[0],
                        completions=["reply"],
                        rewards=[reward],
                        advantages=[0.0],
                        completion_token_ids=[[[3]]],
                    )
                ],
            )
            if args.case == "partial-settlement":
                with pytest.raises(RuntimeError, match="bounded HTTP"):
                    client.post_rollout_completion(request)
                assert client._report_failed
            elif args.case == "retired-source":
                assert not client.post_rollout_completion(request)
            else:
                assert client.post_rollout_completion(request)
                if args.case == "changed-report":
                    request.payloads[0].completions = ["changed"]
                    with pytest.raises(RuntimeError, match="bounded HTTP"):
                        client.post_rollout_completion(request)
                else:
                    assert client.post_rollout_completion(request)
                assert client._report_sequence == 1
        proof = torch.tensor(1.0, device=args.device)
        dist.all_reduce(proof)
        assert proof.item() == 2.0
        if rank == 0:
            expected_calls = {
                "healthy": 2,
                "lost-reply": 3,
                "changed-report": 2,
                "partial-settlement": 3,
                "retired-source": 1,
            }[args.case]
            assert seen == [("session", 0)] * expected_calls
            assert len(injected) == (args.case != "healthy")
            if args.case == "partial-settlement":
                assert atom.rollout_report_receipt.failed
                assert controller.policy_status_manager.terminal_error is not None
                assert controller.stat_n_samples == 1
                assert controller.policy_status_manager.rollout_buffer.empty()
            elif args.case == "retired-source":
                assert controller.stat_n_samples == 0
            else:
                assert controller.stat_n_samples == 1
                assert controller.policy_status_manager.rollout_buffer.qsize() == 1
                assert (
                    controller.policy_status_manager.rollout_buffer.get().reward == 15.0
                )
                assert controller.policy_status_manager.samples_on_the_fly == 10
        dist.barrier()
        print(
            f"ROLLOUT_REPORT_PASS rank={rank} case={args.case} device={args.device}",
            flush=True,
        )
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
