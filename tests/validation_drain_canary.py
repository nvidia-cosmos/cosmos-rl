# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real dispatch/optimizer/validation receipts across a two-rank drain.

Redis carries production policy commands and payloads. Gloo carries fixture
control and substitutes for HTTP receipts and weight transfer. CUDA runs the
tiny optimizer and validation arithmetic, not a full model or rollout backend.
"""

import argparse
import asyncio
from datetime import timedelta
import os
from pathlib import Path
import socket
import subprocess
import time

import msgpack
import pytest
import redis
import torch
import torch.distributed as dist

import cosmos_rl
from cosmos_rl.dispatcher.command import Command, TrainingCompleteCommand
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.dispatcher.protocol import ValidationReportRequest
from cosmos_rl.dispatcher.publication import ControllerPublisher
from cosmos_rl.dispatcher.run_web_panel import extract_rollouts
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from dispatch_commit_canary import worker
from test_validation_drain import setup


CASES = {
    "zero": (0, 10),
    "early": (3, 10),
    "periodic": (4, 1),
    "initial": (3, 10),
    "nominal": (6, 10),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
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
    device = torch.device(args.device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    accepted, frequency = CASES[args.case]
    expected_steps = accepted // 2
    server = publisher = patcher = None
    address = [None]
    try:
        if rank == 0:
            with socket.socket() as probe:
                probe.bind(("0.0.0.0", 0))
                port = probe.getsockname()[1]
            server = subprocess.Popen(
                [
                    "redis-server",
                    "--bind",
                    "0.0.0.0",
                    "--protected-mode",
                    "no",
                    "--port",
                    str(port),
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                ],
                stdout=subprocess.DEVNULL,
            )
            address[0] = (os.environ["MASTER_ADDR"], port)
        dist.broadcast_object_list(address)
        host, port = address[0]
        raw = redis.Redis(host=host, port=port, socket_timeout=2)
        deadline = time.monotonic() + 10
        while True:
            try:
                raw.ping()
                break
            except redis.ConnectionError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        reader = RedisStreamHandler([host], port)
        if rank == 0:
            patcher = pytest.MonkeyPatch()
            manager, rollouts, replica, p2r, r2r, _ = setup(
                patcher, accepted, freq=frequency
            )
            # Use the production completion publication too, not the unit fixture mock.
            patcher.undo()
            patcher.setattr(
                "cosmos_rl.dispatcher.command.PolicyToRolloutUnicastCommand.trigger",
                p2r,
            )
            patcher.setattr(
                "cosmos_rl.dispatcher.command.RolloutToRolloutBroadcastCommand.trigger",
                r2r,
            )
            failures = []
            publisher = ControllerPublisher(
                RedisStreamHandler([host], port),
                on_failure=lambda error: failures.append(error),
            )
            manager.redis_handler = publisher
            manager.total_steps = 3
            manager.data_fetcher.val_dataloader = [
                ([index], [RLPayload(prompt_idx=index, prompt=str(index))])
                for index in (7, 7, 2)
            ]
            for rollout in manager.rollout_buffer.queue:
                rollout.reward = float(rollout.prompt_idx % 3)
            if args.case == "initial":
                manager.config.validation.val_before_train = True
                manager.prepare_validation_round(0, 3, [replica])
            manager.on_rollout_is_end(rollouts)
            rounds = []
        else:
            policy, acknowledgements = worker(device, reader)
        while True:
            message = [None]
            if rank == 0:
                asyncio.run(publisher.flush())
                assert not failures
                if manager.dispatched_rollouts_by_step:
                    message[0] = ("train", manager.current_step)
                elif manager.data_fetcher.activated_val_iter is not None:
                    round_ = manager.validation_round
                    assert round_ is not None
                    batch = manager.fetch_validation_prompts(
                        8, round_.step, None, round_.round_id, "a", 0
                    )
                    assert batch.is_end and len(batch.payloads) == 3
                    message[0] = (
                        "validate",
                        round_.step,
                        round_.round_id,
                        [payload.model_dump() for payload in batch.payloads],
                    )
                elif (
                    manager.completion_step is not None
                    and not manager.terminal_complete
                ):
                    message[0] = ("complete", manager.current_step)
                else:
                    assert manager.terminal_complete and manager.validation_drained
                    message[0] = ("done",)
            dist.broadcast_object_list(message)
            action = message[0][0]
            if action == "done":
                break
            receipt = None
            if rank == 1:
                if action in ("train", "complete"):
                    commands = reader.subscribe_command("policy-0")
                    assert len(commands) == 1
                    instruction = Command.depack(commands[0])
                    if action == "train":
                        data = reader.subscribe_rollout("policy-0")
                        assert len(data) == 2
                        for payload in data:
                            policy.data_queue.put(
                                Rollout.model_validate(msgpack.unpackb(payload))
                            )
                        assert policy.execute_data_fetch(instruction) == (
                            message[0][1] == 3
                        )
                    else:
                        assert isinstance(instruction, TrainingCompleteCommand)
                        assert instruction.final_step == expected_steps
                        assert instruction.checkpoint_total_steps == 3
                        assert policy.execute_training_complete(instruction)
                    receipt = acknowledgements[-1]
                else:
                    _, step, round_id, payloads = message[0]
                    assert policy.trainer.updates == step
                    values = [RLPayload.model_validate(payload) for payload in payloads]
                    for payload in values:
                        prediction = (
                            torch.tensor(float(payload.prompt_idx), device=device)
                            * policy.trainer.weight.detach()
                        )
                        payload.rewards, payload.advantages = [prediction.item()], [0.0]
                    receipt = ValidationReportRequest(
                        src_replica_name="a",
                        src_global_rank=0,
                        validation_step=step,
                        validation_round_id=round_id,
                        report_sequence=0,
                        payloads=values,
                    ).model_dump()
                assert policy.trainer.updates == policy.trainer.scheduler_calls
            gathered = [None, None] if rank == 0 else None
            dist.gather_object(receipt, gathered, dst=0)
            if rank == 0:
                if action in ("train", "complete"):
                    manager.train_ack(*gathered[1], rollouts)
                    manager.train_ack(*gathered[1], rollouts)
                else:
                    request = ValidationReportRequest.model_validate(gathered[1])
                    rounds.append(request.validation_step)
                    for terminal in (False, True):
                        if terminal:
                            (
                                request.payloads,
                                request.is_end,
                                request.report_sequence,
                            ) = [], True, 1
                        results = extract_rollouts(
                            request.payloads, True, is_validation=True
                        )
                        for _ in range(2):
                            manager.validation_report_validation_results(
                                request.validation_step,
                                results,
                                rollouts,
                                request=request,
                            )
                assert manager.total_steps == manager.training_horizon() == 3
        if rank == 0:
            assert manager.current_step == expected_steps
            assert not manager.samples_on_the_fly and not manager.val_report_data
            assert (
                rounds
                == {
                    "zero": [0],
                    "early": [1],
                    "periodic": [1, 2],
                    "initial": [0, 1],
                    "nominal": [3],
                }[args.case]
            )
            assert (
                manager.completion_step is None
                if args.case == "nominal"
                else manager.completion_step == expected_steps + 1
            )
        else:
            assert (
                policy.trainer.updates
                == policy.trainer.scheduler_calls
                == expected_steps
            )
        dist.barrier()
        print(
            f"VALIDATION_DRAIN_PASS rank={rank} case={args.case} updates={expected_steps} horizon=3 device={device}",
            flush=True,
        )
    finally:
        if publisher is not None:
            asyncio.run(publisher.close())
        if patcher is not None:
            patcher.undo()
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
