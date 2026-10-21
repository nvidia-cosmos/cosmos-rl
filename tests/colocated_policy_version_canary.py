# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two real optimizer updates through colocated version/queue boundaries.

Exercises actual controller methods and NCCL/Gloo collectives with generated
sample fixtures, not a simulator, model checkpoint restore, or controller server.
"""

import argparse
from datetime import timedelta
import os
from queue import Queue
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.dispatcher.command import DataFetchCommand


def run_case(rank, device, start, on_policy, centralized):
    world = dist.get_world_size()
    batch_per_rank = 2
    controller = object.__new__(ColocatedController)
    controller.config = NS(
        train=NS(
            train_policy=NS(on_policy=on_policy, uncentralized_training=not centralized)
        )
    )
    controller.current_step, controller.total_steps = 0, 100
    controller.train_report_data = {}
    controller._unreported_rollouts = []
    controller.policy = NS(global_rank=rank, world_size=world, data_queue=Queue())
    mesh = NS(get_group=lambda: dist.group.WORLD, get_local_rank=lambda: rank)
    discarded = []
    controller.rollout = NS(
        parallel_dims=NS(mesh={"dp": mesh}, cp_coord=(0, 1)),
        current_weight_version=start,
        api_client=NS(
            post_queued_rollout_discard=lambda count: discarded.append(count) or True
        ),
    )
    command = object.__new__(DataFetchCommand)
    command.global_step, command.total_steps = start + 1, start + 2
    controller.policy_consume_one_step_commands_util_data_fetch = lambda: command
    initial = []
    controller.rollout_consume_one_step_commands_util_r2r = (
        lambda **kwargs: initial.append(
            (controller.current_step, controller.total_steps, kwargs)
        )
        or True
    )
    assert controller.init_commands()
    assert initial == [(start, start + 2, {"initial": True})]
    weight = torch.nn.Parameter(torch.tensor([1.0], device=device))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    generated = 0
    for update in range(2):
        controller.advance_iteration()
        pending = controller.pending_policy_samples_all_replicas()
        if pending < batch_per_rank * world:
            # Unequal surplus on owning ranks; every rank still refills together.
            samples = [
                NS(weight_version=start + update, origin=rank, ordinal=i)
                for i in range(4 + 2 * rank)
            ]
            generated += 1
            if centralized:
                controller._unreported_rollouts.append(samples)
            else:
                for sample in samples:
                    controller.policy.data_queue.put_nowait(sample)
            controller.synchronize_rollouts()
        assert (
            controller.pending_policy_samples_all_replicas() >= batch_per_rank * world
        )
        consumed = []
        if not centralized or rank == 0:
            count = batch_per_rank * (world if centralized else 1)
            consumed = [controller.policy.data_queue.get_nowait() for _ in range(count)]
            assert all(
                sample.weight_version == start + (update if on_policy else 0)
                for sample in consumed
            )
        optimizer.zero_grad()
        weight.square().sum().backward()
        dist.all_reduce(weight.grad)
        weight.grad.div_(world)
        optimizer.step()
        torch.testing.assert_close(
            weight.detach(), torch.tensor([0.8 ** (update + 1)], device=device)
        )
        assert controller.current_step == start + update + 1
        # Model completed weight sync before the next on-policy admission.
        if on_policy:
            controller.rollout.current_weight_version = controller.current_step
    assert generated == (2 if on_policy else 1)
    assert controller.total_steps == start + 2
    controller.finalize_queued_rollouts()
    assert controller.policy.data_queue.empty()
    if centralized:
        expected_discarded = ([6, 6] if on_policy else [2]) if rank == 0 else []
    else:
        expected_discarded = [2 + 2 * rank] * 2 if on_policy else ([2] if rank else [])
    assert discarded == expected_discarded, (rank, discarded, expected_discarded)
    print(
        f"COLOCATED_POLICY_VERSION_PASS rank={rank} start={start} "
        f"on_policy={on_policy} centralized={centralized} updates=2 generated={generated} discarded={discarded}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu:
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo" if args.cpu else "cpu:gloo,cuda:nccl",
        timeout=timedelta(seconds=60),
    )
    assert dist.get_world_size() == 2
    try:
        for start in (0, 8):
            for on_policy in (False, True):
                for centralized in (False, True):
                    run_case(dist.get_rank(), device, start, on_policy, centralized)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
