# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank controller-admission/optimizer canary, not a transport benchmark.

Uses real lazy prompt fetching, fenced HTTP handlers, producer reservations and
policy queues. Fixtures disable automatic command publication so the resulting
balanced batches can drive a small real collective optimizer update. Generation
and model execution are deliberately synthetic; NCCL/UCXX payload transfer is
not qualified by this test.
"""

import argparse
import datetime
import os

import pytest
import torch
import torch.distributed as dist

from test_rank_dispatch_refill import exercise_partial_rank_refill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.backend == "nccl"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=datetime.timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 2
        reports = [None]
        if rank == 0:
            with pytest.MonkeyPatch.context() as patch:
                batches = exercise_partial_rank_refill(patch)
            reports[0] = [
                [float(rollout.reward) for rollout in batch] for batch in batches
            ]
        dist.broadcast_object_list(reports, src=0, device=device)
        targets = torch.tensor(reports[0][rank], device=device)
        assert targets.numel() == 2
        weight = torch.nn.Parameter(torch.zeros((), device=device))
        optimizer = torch.optim.SGD([weight], lr=0.1)
        loss = ((weight - targets) ** 2).mean()
        loss.backward()
        dist.all_reduce(weight.grad)
        weight.grad.div_(dist.get_world_size())
        optimizer.step()
        torch.testing.assert_close(weight.detach(), torch.tensor(0.2, device=device))
        dist.barrier()
        print(
            f"RANK_DISPATCH_CANARY_PASS rank={rank} backend={args.backend} samples=2 update=1",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
