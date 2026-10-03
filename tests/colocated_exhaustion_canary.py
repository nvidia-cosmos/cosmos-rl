# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded real HTTP + two replicas/two ranks control-flow canary.

Runs the production colocated loop, preparation API and queue synchronization.
Generation and the trainer are test doubles; the latter performs an actual
gradient all-reduce and optimizer update. This is not a model-quality test.
"""

import argparse
import asyncio
from datetime import timedelta
import json
import multiprocessing as mp
import os
from queue import Queue
import socket
import threading
import time
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.colocated.rl_worker import ColocatedRLControlWorker
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.command import (
    Command,
    DataFetchCommand,
    TrainingCompleteCommand,
)
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils.api_suffix import COSMOS_API_COLOCATED_PREPARATION_SUFFIX


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(
    index, backend, case, centralized, http_port, group_ports, terminal, results
):
    rank, replica = index % 2, index // 2
    name = f"policy-{replica}"
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(index))
    device = torch.device("cpu" if backend == "gloo" else f"cuda:{index}")
    if backend == "nccl":
        torch.cuda.set_device(index)
    # Separate process groups reflect separate policy replicas; the HTTP
    # preparation barrier is responsible for agreement between replicas.
    dist.init_process_group(
        "gloo" if backend == "gloo" else "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{group_ports[replica]}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    client = APIClient(Role.POLICY, ["127.0.0.1"], http_port)
    client._report_session_id = name
    client._registered_global_rank = rank
    data_queue = Queue()
    controller = object.__new__(ColocatedController)
    controller.config = NS(
        train=NS(
            train_batch_per_replica=16,
            train_policy=NS(uncentralized_training=not centralized),
        ),
        rollout=NS(n_generation=8),
    )
    controller._unreported_rollouts = []
    controller.policy = NS(
        global_rank=rank,
        world_size=2,
        replica_name=name,
        api_client=client,
        data_queue=data_queue,
    )
    mesh = NS(
        size=lambda: 2, get_group=lambda: dist.group.WORLD, get_local_rank=lambda: rank
    )
    controller.rollout = NS(parallel_dims=NS(mesh={"dp": mesh}, cp_coord=(0, 1)))
    controller.init_data_fetch_command = NS(global_step=1, total_steps=1)
    state = NS(calls=0, updates=0, stopped=False, shutdown=False)
    weight = torch.nn.Parameter(torch.tensor([1.0], device=device))
    optimizer = torch.optim.SGD([weight], lr=0.1)

    def generate():
        state.calls += 1
        count = 8
        if replica == 1 and rank == 1 and case in {"exhausted", "refill"}:
            count = 4
        if case == "empty":
            count = 0
        if centralized:
            controller._unreported_rollouts.append([1] * count)
        else:
            for _ in range(count):
                data_queue.put(1)
        return case != "refill" or state.calls == 2, count // 8

    def consume(command, **kwargs):
        if command is DataFetchCommand:
            assert not kwargs.get("no_exec", False), "fake training ACK path"
            optimizer.zero_grad()
            weight.square().sum().backward()
            dist.all_reduce(weight.grad)
            weight.grad.div_(2)
            optimizer.step()
            state.updates += 1

    def finish(command):
        assert isinstance(command, TrainingCompleteCommand)
        assert command.final_step == 0 and command.checkpoint_total_steps == 1
        state.stopped = state.shutdown = True

    def get_terminal():
        deadline = time.monotonic() + 30
        while name not in terminal:
            if time.monotonic() >= deadline:
                raise TimeoutError("No terminal command")
            time.sleep(0.01)
        return Command.depack(terminal[name])

    controller.init_commands = lambda: True
    controller.prepare_iteration = lambda: True
    controller.advance_iteration = lambda: None
    controller.policy_consume_one_step_commands_util_data_fetch = get_terminal
    controller.finish_requested_stop = finish
    controller.rollout_completed_for_data_fetch_n_training = lambda pending: None

    def end_iteration():
        state.shutdown = True
        return False

    controller.rollout_consume_one_step_commands_util_r2r = end_iteration
    loop = NS(
        config=controller.config,
        controller=controller,
        policy=NS(consume_command=consume),
        rollout=NS(
            consume_command=lambda *args: None,
            parallel_dims=controller.rollout.parallel_dims,
            rollout_for_one_minor_step=generate,
            report_rollouts=lambda **kwargs: None,
            shutdown_signal=NS(is_set=lambda: state.shutdown),
        ),
    )
    try:
        ColocatedRLControlWorker.main_loop(loop)
        expected = 0 if case in {"exhausted", "empty"} else 1
        # Centralized gathering can pool the 8+4 tail, but still cannot make 16.
        assert state.updates == expected
        assert state.stopped == (expected == 0)
        assert abs(weight.item() - (1.0 if expected == 0 else 0.8)) < 1e-6
        if expected == 0:
            assert data_queue.empty()
        results.put(
            {
                "rank": index,
                "updates": state.updates,
                "calls": state.calls,
                "stopped": state.stopped,
                "weight": weight.item(),
            }
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument(
        "--case", choices=["healthy", "exhausted", "refill", "empty"], required=True
    )
    parser.add_argument("--centralized", action="store_true")
    args = parser.parse_args()
    if args.backend == "nccl":
        assert torch.cuda.is_available() and torch.cuda.device_count() >= 4
    from cosmos_rl.dispatcher import run_web_panel
    from test_colocated_exhaustion import dispatched

    status, _, _ = dispatched()
    status.total_steps = 1
    status.training_dispatches[1].total_steps = 1
    status.config.train.train_batch_per_replica = 16
    status.training_dispatches[1].rollout_count = 32
    status.dispatched_rollouts_by_step[1] = 32
    status.samples_on_the_fly = 32
    run_web_panel.controller = NS(
        life_cycle_lock=asyncio.Lock(), policy_status_manager=status
    )
    app = FastAPI()
    app.post(COSMOS_API_COLOCATED_PREPARATION_SUFFIX)(
        run_web_panel.colocated_preparation
    )
    http_port, group_ports = port(), [port(), port()]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=http_port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    ctx = mp.get_context("spawn")
    with ctx.Manager() as shared:
        terminal, results = shared.dict(), ctx.Queue()
        status.redis_handler.publish_command.side_effect = (
            lambda command, name: terminal.__setitem__(name, command)
        )
        thread.start()
        deadline = time.monotonic() + 15
        while not server.started:
            if time.monotonic() >= deadline:
                raise TimeoutError("HTTP server failed to start")
            time.sleep(0.01)
        processes = [
            ctx.Process(
                target=worker,
                args=(
                    i,
                    args.backend,
                    args.case,
                    args.centralized,
                    http_port,
                    group_ports,
                    terminal,
                    results,
                ),
            )
            for i in range(4)
        ]
        try:
            for process in processes:
                process.start()
            deadline = time.monotonic() + 90
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            assert all(p.exitcode == 0 for p in processes), [
                p.exitcode for p in processes
            ]
            reports = sorted(
                [results.get(timeout=5) for _ in processes],
                key=lambda report: report["rank"],
            )
            stopped = args.case in {"exhausted", "empty"}
            assert status.current_step == (0 if stopped else 1)
            assert bool(status.stop_reason) == stopped
            assert bool(status.dispatched_rollouts_by_step) != stopped
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "case": args.case,
                        "centralized": args.centralized,
                        "reports": reports,
                    }
                )
            )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            server.should_exit = True
            thread.join(5)


if __name__ == "__main__":
    main()
