# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two torchrun ranks: actual controller dispatch -> Redis -> worker/trainer.

Rank zero owns the controller/Redis fixture; rank one executes the production
RLPolicyWorker data-fetch entrypoint with a tiny real optimizer. The Gloo group
only moves fixture setup and HTTP-ACK substitutes; it is not a training group.
Supports one or two nodes. Partial-publication failure replaces only the fatal
exit callback so the parent can assert its outcome instead of exiting the test.
"""

import argparse
import asyncio
from datetime import timedelta
import os
from queue import Queue
import socket
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import redis
import torch
import torch.distributed as dist

from cosmos_rl.dispatcher.command import Command, TrainingCompleteCommand
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.publication import ControllerPublisher
from cosmos_rl.dispatcher.status import JobPhase, PolicyStatus, RolloutStatusManager
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.utils.redis_stream import RedisStreamHandler
import test_terminal_drain_protocol as terminal_fixture


class FaultClient:
    def __init__(self, client, case):
        self.client, self.case = client, case
        self.seen = set()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def eval(self, script, nkeys, *args):
        if self.case == "partial":
            # Fail after only the first rollout, before its DataFetch command.
            script = script.replace(
                "'timestamp', ARGV[4])",
                "'timestamp', ARGV[4])\n    if i == 2 then return redis.error_reply('injected partial publication') end",
            )
        result = self.client.eval(script, nkeys, *args)
        if self.case == "lost-reply" and args[0] not in self.seen:
            self.seen.add(args[0])
            raise redis.ConnectionError("injected lost reply after successful commit")
        return result


class TinyTrainer:
    def __init__(self, device):
        self.weight = torch.nn.Parameter(
            torch.ones((), device=device, dtype=torch.float64)
        )
        self.optimizer = torch.optim.SGD([self.weight], lr=0.01, momentum=0.9)
        self.reference_weight = 1.0
        self.reference_momentum = 0.0
        self.updates = 0
        self.scheduler_calls = 0

    def update_lr_schedulers(self, total_steps):
        assert total_steps == 3
        self.scheduler_calls += 1

    def step_training(self, *, rollouts, current_step, **kwargs):
        assert current_step == self.updates + 1 and len(rollouts) == 2
        x = torch.tensor(
            [r.prompt_idx + 1 for r in rollouts],
            device=self.weight.device,
            dtype=torch.float64,
        )
        y = torch.tensor(
            [r.reward for r in rollouts], device=self.weight.device, dtype=torch.float64
        )
        self.optimizer.zero_grad()
        loss = ((x * self.weight - y) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        gradient = sum(
            2
            * (r.prompt_idx + 1)
            * (self.reference_weight * (r.prompt_idx + 1) - r.reward)
            for r in rollouts
        ) / len(rollouts)
        self.reference_momentum = 0.9 * self.reference_momentum + gradient
        self.reference_weight -= 0.01 * self.reference_momentum
        torch.testing.assert_close(
            self.weight,
            torch.full_like(self.weight, self.reference_weight),
            rtol=1e-12,
            atol=1e-12,
        )
        self.updates += 1
        return {"train/loss_avg": loss.item(), "train_step": current_step}


def worker(device, reader):
    instance = RLPolicyWorker.__new__(RLPolicyWorker)
    instance.replica_name = "policy-0"
    instance.global_rank = 0  # rank inside the one-rank policy replica
    instance.world_size = instance.dp_world_size = 1
    instance.parallel_dims = SimpleNamespace(
        pp_enabled=False, get_rank_in_dim=lambda *_: 0
    )
    instance.config = SimpleNamespace(
        train=SimpleNamespace(
            local_dataset=False,
            train_policy=SimpleNamespace(uncentralized_training=False),
        )
    )
    instance.data_queue = Queue()
    instance.signal_handler = None
    instance.inter_policy_nccl = None
    instance.is_master_replica = True
    instance.profiler = SimpleNamespace(step=Mock(), check_finished=lambda: False)
    instance.prepare_teacher_uuids_for_prefetch = lambda dp_id, count: dp_id
    instance.trainer = TinyTrainer(device)
    acknowledgements = []
    instance.api_client = SimpleNamespace(
        post_policy_train_ack=lambda *args: acknowledgements.append(args)
    )
    return instance, acknowledgements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "healthy",
            "lost-reply",
            "partial",
            "departure",
            "pending-ack",
            "stale-surplus",
        ),
        required=True,
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    local_rank = int(os.environ["LOCAL_RANK"])
    if not args.cpu:
        assert torch.cuda.is_available(), "Required GPU canary cannot skip"
        torch.cuda.set_device(local_rank)
    device = torch.device("cpu" if args.cpu else f"cuda:{local_rank}")
    server = publisher = reader = None
    try:
        address = [None]
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
        dist.broadcast_object_list(address, src=0)
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
            writer = RedisStreamHandler([host], port)
            writer.redis_clients = [FaultClient(raw, args.case)]
            failures = []
            publisher = ControllerPublisher(writer, on_failure=failures.append)
            manager, _ = terminal_fixture.TestTerminalMatrix._manager(0)
            manager.redis_handler = publisher
            manager.total_steps = 3
            manager.remain_samples_num = 6
            manager.config.train.train_policy.type = "grpo"
            manager.config.train.train_policy.on_policy = False
            manager.should_weight_sync_after_train_ack = lambda *_: False
            rollout_status = SimpleNamespace(replica_scaling_log=[])
            if args.case == "stale-surplus":
                manager.config.train.train_policy.allowed_outdated_steps = 0
                # Two previously admitted surplus entries become unusable
                # after update one, as can happen when a cohort shrinks.
                manager.remain_samples_num += 2
        else:
            policy, acknowledgements = worker(device, reader)
        steps = 1 if args.case == "departure" else 3
        for step in range(1, steps + 1):
            outcome = [None]
            if rank == 0:
                for index in range(2):
                    manager.rollout_buffer.put(
                        Rollout(
                            prompt_idx=2 * (step - 1) + index,
                            reward=float(step + index),
                            weight_version=step - 1,
                        )
                    )
                manager.samples_on_the_fly += 2
                if args.case == "stale-surplus" and step == 1:
                    for index in range(2):
                        manager.rollout_buffer.put(
                            Rollout(prompt_idx=100 + index, weight_version=0)
                        )
                    manager.samples_on_the_fly += 2
                manager.try_trigger_data_fetch_and_training()
                if args.case == "pending-ack":
                    # The actual worker still has its issued command. A
                    # transient READY flag must not publish a second update.
                    manager.status["policy-0"] = PolicyStatus.READY
                    probes = [
                        Rollout(prompt_idx=200 + i, weight_version=step)
                        for i in range(2)
                    ]
                    for rollout in probes:
                        manager.rollout_buffer.put(rollout)
                    manager.samples_on_the_fly += 2
                    manager.try_trigger_data_fetch_and_training()
                    assert manager.current_step == step
                    assert manager.dispatched_rollouts_by_step == {step: 2}
                    assert [
                        manager.rollout_buffer.get_nowait() for _ in range(2)
                    ] == probes
                    manager.samples_on_the_fly -= 2
                    print(f"DISPATCH_PENDING_ACK_BLOCKED step={step}", flush=True)
                if args.case == "partial":
                    try:
                        asyncio.run(publisher.flush())
                        raise AssertionError("partial publication was accepted")
                    except redis.ResponseError as error:
                        assert "injected partial" in str(error)
                    assert failures and raw.xlen("policy-0_command") == 0
                    assert raw.xlen("policy-0_rollout") == 1
                    outcome[0] = False
                else:
                    asyncio.run(publisher.flush())
                    assert not failures
                    assert raw.xlen("policy-0_command") == step
                    assert raw.xlen("policy-0_rollout") == 2 * step
                    outcome[0] = True
            dist.broadcast_object_list(outcome, src=0)
            if not outcome[0]:
                if rank == 1:
                    assert policy.trainer.updates == 0
                break
            receipt = None
            if rank == 1:
                commands = reader.subscribe_command("policy-0")
                rollouts = reader.subscribe_rollout("policy-0")
                assert len(commands) == 1 and len(rollouts) == 2
                for payload in rollouts:
                    policy.data_queue.put(
                        Rollout.model_validate(msgpack.unpackb(payload))
                    )
                stop = policy.execute_data_fetch(Command.depack(commands[0]))
                assert stop == (step == 3)
                receipt = acknowledgements[-1]
                assert policy.trainer.updates == policy.trainer.scheduler_calls == step
            gathered = [None, None] if rank == 0 else None
            dist.gather_object(receipt, gathered, dst=0)
            if rank == 0:
                if args.case == "departure":
                    rollout_status = RolloutStatusManager()
                    rollout_status.rollout_replicas = {
                        "departing": terminal_fixture._registered_rollout("departing"),
                        "ended": terminal_fixture._registered_rollout(
                            "ended", ended=True
                        ),
                    }
                    # This fixture exercises dispatch/ACK ownership, not mesh
                    # reconstruction or recovery of lost producer payloads.
                    for replica in rollout_status.rollout_replicas.values():
                        replica.in_mesh = False
                    rollout_status.unregister("departing", manager)
                    assert manager.job_phase == JobPhase.DRAINING
                    assert manager.completion_step is None
                    assert manager.dispatched_rollouts_by_step == {1: 2}
                manager.train_ack(*gathered[1], rollout_status)
                manager.train_ack(*gathered[1], rollout_status)  # lost ACK reply
                assert manager.samples_on_the_fly == 0
                assert manager.current_step == step
                assert manager.training_dispatches[step].settled
                if args.case == "stale-surplus" and step == 1:
                    assert manager.rollout_buffer.empty()
                    assert manager.filter_records["outdated"] == 2
                    assert manager.remain_samples_num == 4
                    print(
                        "DISPATCH_STALE_SURPLUS_REJECTED count=2 extra_updates=0",
                        flush=True,
                    )
        if args.case == "departure":
            if rank == 0:
                asyncio.run(publisher.flush())
                assert manager.completion_step == 2
                assert not manager.terminal_complete
            dist.barrier()
            receipt = None
            if rank == 1:
                commands = reader.subscribe_command("policy-0")
                assert len(commands) == 1
                complete = Command.depack(commands[0])
                assert isinstance(complete, TrainingCompleteCommand)
                assert complete.final_step == 1
                assert complete.checkpoint_total_steps == 3
                assert policy.execute_training_complete(complete)
                assert policy.trainer.updates == policy.trainer.scheduler_calls == 1
                receipt = acknowledgements[-1]
            gathered = [None, None] if rank == 0 else None
            dist.gather_object(receipt, gathered, dst=0)
            if rank == 0:
                manager.train_ack(*gathered[1], rollout_status)
                manager.train_ack(*gathered[1], rollout_status)
                assert manager.terminal_complete
                assert manager.current_step == 1
                assert manager.training_horizon() == manager.total_steps == 3
                assert manager.samples_on_the_fly == 0
        dist.barrier()
        print(
            f"DISPATCH_COMMIT_PASS rank={rank} case={args.case} updates={0 if args.case == 'partial' else steps} device={device}",
            flush=True,
        )
    finally:
        if publisher is not None:
            try:
                asyncio.run(publisher.close())
            except (redis.ResponseError, RuntimeError):
                if args.case != "partial":
                    raise
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
