# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""torchrun --standalone --nproc-per-node=2 tests/weight_adoption_canary.py"""

import argparse
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker import weight_sync as ws
from cosmos_rl.utils.pynccl import create_nccl_comm, create_nccl_uid, nccl_abort
from test_weight_adoption import make_worker


def initialize_transport():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "This canary requires two global ranks"
    uid = [create_nccl_uid() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = create_nccl_comm(uid[0], rank, 2, timeout_ms=60_000)
    return rank, torch.device("cuda", local_rank), comm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    rank, device, comm = initialize_transport()
    worker, model = make_worker(device)
    worker.replica_name = f"rollout-{rank}"
    worker.replica_name_to_rank = {"rollout-0": 0, "rollout-1": 1}
    worker.rank_in_rollout_repicas = rank
    worker.global_commnicator_idex = comm
    worker.config.rollout.r2r_sync_pack_tensors = args.packed
    worker.config.rollout.r2r_sync_bucket_size_bytes = 2048
    worker.config.rollout.async_r2r_sync = "inference"
    worker.non_trainable_params_received = False
    outputs = []

    def forward(observation):
        assert torch.cuda.current_stream() == worker.inference_stream
        return torch.nn.functional.linear(observation, model.embedding)

    worker.rollout._servicer = SimpleNamespace(policy_fn=forward)

    def generate(**kwargs):
        output = worker.rollout._servicer.policy_fn(
            torch.ones(1, 8, device=worker.device)
        )
        outputs.append((kwargs["current_weight_version"], output))
        return [RolloutResult(completions=[output])]

    worker.rollout.rollout_generation = generate
    # Replace only the Redis phase barrier with this launch's Gloo group. Native
    # grouped/packed NCCL, writer ownership and rollout forward entrypoints run.
    original_barrier = ws.r2r_barrier
    ws.r2r_barrier = lambda *a, **kw: (dist.barrier() or True)
    wst = worker._weight_sync_thread
    try:
        with torch.no_grad():
            for version in range(1, 13):
                if rank == 0:
                    wst._execute_p2r(SimpleNamespace(weight_step=version))
                wst._execute_r2r(
                    SimpleNamespace(
                        weight_step=version,
                        src_replica_name="rollout-0",
                        dst_replica_names=["rollout-0", "rollout-1"],
                        total_steps=20,
                        replica_should_stop=lambda: False,
                    )
                )
                assert worker.non_trainable_params_received
                assert worker.current_weight_version == version - 1
                # Keep adoption/forward queued while the next native transfer
                # starts. The next writer must wait for the pending buffer read.
                with torch.cuda.stream(worker.inference_stream):
                    torch.cuda._sleep(50_000_000)
                result = worker._call_rollout_generation(
                    payloads=[], is_validation=False
                )
                assert result[0].weight_version == version
            torch.cuda.synchronize()
            for version, output in outputs:
                torch.testing.assert_close(output, torch.full_like(output, 8 * version))
            assert worker.current_weight_version == 12
            torch.testing.assert_close(model.head, torch.full_like(model.head, 12))
            assert len(outputs) == 12
            print(
                f"WEIGHT_ADOPTION_PASS rank={rank} packed={args.packed} versions=12 forward_parity=True",
                flush=True,
            )
        dist.barrier()
    finally:
        ws.r2r_barrier = original_barrier
        nccl_abort(comm)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
