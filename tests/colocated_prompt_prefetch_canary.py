# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native two-rank colocated scheduling canary with shared-model SGD updates.

Uses the real prompt fetch/scatter, generation mixin, RL driver and local
controller queues. Hook delays model independent preparation/generation work;
benchmark rates demonstrate scheduling overlap, not application throughput.
"""

import argparse
from datetime import timedelta
import json
import os
from queue import Queue
import statistics
import threading
import time
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.colocated.rl_worker import ColocatedRLControlWorker
from cosmos_rl.dispatcher.command import DataFetchCommand
from cosmos_rl.rollout import State
from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)


class Backend(RolloutGenerationMixin):
    def __init__(self, config, weight, *, delay, reject_first):
        self.config = config
        self.weight = weight
        self.delay = delay
        self.reject_first = reject_first
        self.prepared = []
        self.generated = []
        self._engine_initialized = True
        self.setup_generation()

    def _prepare_sample(self, payload, **kwargs):
        version = self.weight.detach().item()
        time.sleep(self.delay)
        assert self.weight.detach().item() == version, "weights changed during setup"
        self.prepared.append(payload.prompt_idx)
        return payload.prompt_idx, version

    def _generate(self, batch, **kwargs):
        assert all(version == self.weight.detach().item() for _, version in batch)
        time.sleep(self.delay)
        # Real device work using the same parameter later updated by SGD.
        result = self.weight.square().sum().detach().item()
        assert result > 0
        return batch

    def _postprocess(self, raw, payloads, **kwargs):
        self.generated.extend(index for index, _ in raw)
        if self.reject_first and payloads[0].prompt_idx < dist.get_world_size():
            return []
        return [NS(weight_version=payload.weight_version) for payload in payloads]


def run_case(device, *, prefetch, depth, centralized, reject_first, delay, batches=4):
    rank, world = dist.get_rank(), dist.get_world_size()
    target = batches * world
    config = NS(
        rollout=NS(
            prefetch_rollout=prefetch, prefetch_queue_maxsize=depth, n_generation=1
        ),
        train=NS(
            train_batch_per_replica=target,
            local_dataset=False,
            train_policy=NS(
                on_policy=True,
                uncentralized_training=not centralized,
                data_dispatch_as_rank_in_mesh=False,
            ),
        ),
    )
    weight = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    backend = Backend(config, weight, delay=delay, reject_first=reject_first)
    rollout = object.__new__(ColocatedRolloutControlWorker)
    rollout.config = config
    rollout.global_rank = rank
    rollout.rank_in_rollout_repicas = 0
    rollout.current_weight_version = 0
    rollout.batch_size = 1
    rollout.rollout = backend
    rollout.data_packer = rollout.data_fetcher = None
    rollout._prompt_fetch_lock = threading.Lock()
    rollout._prompt_queue = Queue(maxsize=depth)
    rollout.state = State()
    rollout.shutdown_signal = threading.Event()
    mesh = NS(
        size=lambda: world,
        get_group=lambda: dist.group.WORLD,
        get_local_rank=lambda: rank,
    )
    rollout.parallel_dims = NS(world_size=world, mesh={"dp": mesh}, cp_coord=(0, 1))
    fetched = 0
    total = 2 * target + (world if reject_first else 0)

    def fetch(n, **kwargs):
        nonlocal fetched
        assert threading.current_thread() is threading.main_thread()
        assert fetched < total, "fetched into another update or past EOF"
        batch = [
            dict(prompt_idx=i, prompt=str(i), weight_version=0)
            for i in range(fetched, min(fetched + n, total))
        ]
        fetched += len(batch)
        return batch, fetched == total

    rollout.api_client = NS(
        get_next_prompt=fetch, post_queued_rollout_discard=lambda n: True
    )
    controller = object.__new__(ColocatedController)
    controller.config = config
    controller.current_step = 0
    controller.train_report_data = {}
    controller._unreported_rollouts = []
    controller.rollout = rollout
    policy = controller.policy = NS(
        global_rank=rank, world_size=world, data_queue=Queue()
    )

    def generate():
        batch = rollout._prompt_queue.get_nowait()
        results = backend.rollout_generation(batch)
        for result in results:
            result.weight_version = rollout.current_weight_version
        if centralized:
            controller._unreported_rollouts.append(results)
        else:
            for result in results:
                policy.data_queue.put_nowait(result)
        return batch, results

    rollout.one_step_generation = generate
    rollout.consume_command = lambda command: None
    rollout.report_rollouts = lambda **kwargs: None
    controller.init_commands = lambda: True
    controller.prepare_iteration = lambda: controller.current_step < 2

    def agree(ready):
        assert rollout._prompt_queue.empty(), "raw prompts crossed the weight boundary"
        assert not backend._setup_futures and not backend._setup_payloads
        assert len(backend.prepared) == len(backend.generated)
        return ready

    controller.agree_prepared_batch = agree
    controller.rollout_completed_for_data_fetch_n_training = lambda pending: None

    def consume(command):
        if command is not DataFetchCommand:
            return
        count = (
            target if centralized and rank == 0 else (batches if not centralized else 0)
        )
        samples = [policy.data_queue.get_nowait() for _ in range(count)]
        assert all(
            sample.weight_version == controller.current_step - 1 for sample in samples
        )
        optimizer.zero_grad()
        weight.square().sum().backward()
        dist.all_reduce(weight.grad)
        weight.grad.div_(world)
        optimizer.step()

    policy.consume_command = consume

    def sync():
        rollout.current_weight_version = controller.current_step
        if controller.current_step == 2:
            rollout.shutdown_signal.set()
        return True

    controller.rollout_consume_one_step_commands_util_r2r = sync
    driver = NS(config=config, controller=controller, rollout=rollout, policy=policy)
    dist.barrier()
    started = time.perf_counter()
    try:
        ColocatedRLControlWorker.main_loop(driver)
        dist.barrier()
        elapsed = time.perf_counter() - started
        torch.testing.assert_close(weight.detach(), torch.full_like(weight, 0.64))
        assert controller.current_step == 2
        assert not backend._setup_futures and rollout._prompt_queue.empty()
        assert backend.prepared == backend.generated
        if rank == 0:
            assert fetched == total
            print(
                json.dumps(
                    dict(
                        event="COLOCATED_PREFETCH_PASS",
                        prefetch=prefetch,
                        depth=depth,
                        centralized=centralized,
                        reject_first=reject_first,
                        updates=2,
                        prompts=fetched,
                        seconds=elapsed,
                        updates_per_second=2 / elapsed,
                    )
                ),
                flush=True,
            )
        return elapsed
    finally:
        backend.shutdown_generation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo" if args.cpu else "cpu:gloo,cuda:nccl", timeout=timedelta(seconds=90)
    )
    assert dist.get_world_size() == 2
    try:
        for prefetch in (False, True):
            for centralized in (False, True):
                for reject_first in (False, True):
                    run_case(
                        device,
                        prefetch=prefetch,
                        depth=2,
                        centralized=centralized,
                        reject_first=reject_first,
                        delay=0.001,
                    )
        if args.benchmark:
            times = {False: [], True: []}
            for repeat in range(4):
                for prefetch in (False, True) if repeat % 2 == 0 else (True, False):
                    seconds = run_case(
                        device,
                        prefetch=prefetch,
                        depth=2,
                        centralized=True,
                        reject_first=False,
                        delay=0.02,
                        batches=32,
                    )
                    if repeat:
                        times[prefetch].append(seconds)
            if dist.get_rank() == 0:
                baseline = statistics.median(times[False])
                enabled = statistics.median(times[True])
                print(
                    json.dumps(
                        dict(
                            event="COLOCATED_PREFETCH_SYNTHETIC_BENCHMARK",
                            device=str(device),
                            seconds=times,
                            baseline_median=baseline,
                            prefetch_median=enabled,
                            speedup=baseline / enabled,
                            note="20ms synthetic prepare/generate hooks; not application throughput",
                        )
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
