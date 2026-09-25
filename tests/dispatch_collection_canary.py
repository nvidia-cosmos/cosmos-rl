# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native DP queue scatter: reject invalid counts, preserve every valid sample."""

import argparse
from datetime import timedelta
import os
from queue import Queue
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_rl.policy.worker.base import PolicyWorkerBase
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.utils.parallelism import ParallelDims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.cpu:
        assert torch.cuda.is_available()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "gloo" if args.cpu else "nccl", timeout=timedelta(seconds=60)
    )
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size > 1
    dims = ParallelDims(
        dp_shard=1, dp_replicate=size, cp=1, tp=1, pp=1, world_size=size
    )
    for decentralized in (False, True):
        config = SimpleNamespace(
            policy=SimpleNamespace(parallelism=SimpleNamespace(dp_shard_size=1)),
            train=SimpleNamespace(
                train_batch_per_replica=size + 1,
                local_dataset=False,
                train_policy=SimpleNamespace(
                    type="grpo",
                    mini_batch=1,
                    trainer_type=None,
                    uncentralized_training=decentralized,
                ),
            ),
        )
        startup = SimpleNamespace(config=config, parallel_dims=dims)
        try:
            PolicyWorkerBase.check_config(startup)
        except (AssertionError, ValueError) as error:
            assert "divisible" in str(error)
        else:
            raise AssertionError("unsupported full-DP count reached initialization")
        queue = Queue()
        owned = (
            list(range(rank, 2 * size, size))
            if decentralized
            else (list(range(2 * size)) if rank == 0 else [])
        )
        for index in owned:
            queue.put(SimpleNamespace(prompt_idx=index, teacher_result_uuid=None))
        worker = SimpleNamespace(
            config=config,
            trainer=SimpleNamespace(),
            global_rank=rank,
            dp_world_size=size,
            world_size=size,
            parallel_dims=SimpleNamespace(get_rank_in_dim=lambda dim, r: r),
            replica_batch_for_this_step=size + 1,
            data_queue=queue,
            prepare_teacher_uuids_for_prefetch=lambda *args: 0,
        )
        try:
            RLPolicyWorker.dispatch_rollouts(worker)
        except ValueError as error:
            assert "refusing to silently round" in str(error)
        else:
            raise AssertionError("invalid collection was rounded down")
        assert queue.qsize() == len(owned)
        dist.barrier()
        config.train.train_batch_per_replica = worker.replica_batch_for_this_step = (
            2 * size
        )
        PolicyWorkerBase.check_config(startup)
        received = RLPolicyWorker.dispatch_rollouts(worker)
        assert queue.empty() and len(received) == 2
        gathered = [None] * size
        dist.all_gather_object(gathered, [r.prompt_idx for r in received])
        assert sorted(index for batch in gathered for index in batch) == list(
            range(2 * size)
        )
        print(
            f"DISPATCH_COLLECTION_PASS rank={rank} decentralized={decentralized} "
            f"global_samples={2 * size} invalid_untouched=True",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
