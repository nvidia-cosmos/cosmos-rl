# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable real-rank agreement/final-save-failure canary (no model downloads).

Run from the source root:
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/generic_stop_rank_canary.py
Uses CPU/Gloo. This tests the production worker stop helpers and controller
boundary, not model training, HTTP transport, or full role lifecycle.
"""

import asyncio
from datetime import timedelta
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_rl.dispatcher.step_boundary import StepBoundary
from cosmos_rl.policy.worker.stop import training_boundary, final_checkpoint


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = int(os.environ["RANK"])
    loop = asyncio.new_event_loop()
    barrier = StepBoundary()
    acknowledgements = []
    reason = None

    def boundary(name, step, *, checkpoint_complete=False):
        assert rank == 0, "Only the replica leader may contact the controller"
        if checkpoint_complete:
            acknowledgements.append(step)
            return {"complete": barrier.complete(name, step)}
        return loop.run_until_complete(
            barrier.arrive(name, step, {name}, lambda: reason)
        )

    worker = SimpleNamespace(
        global_rank=rank,
        replica_name="policy",
        train_step=0,
        api_client=SimpleNamespace(training_boundary=boundary),
        trainer=SimpleNamespace(),
    )
    assert not training_boundary(worker, 0)
    reason = "budget"
    # Retrying an already granted step cannot revoke permission on one rank.
    assert not training_boundary(worker, 0)
    update = torch.tensor([rank + 1.0])
    dist.all_reduce(update)
    assert update.item() == 3
    worker.train_step = 1
    assert training_boundary(worker, 1)
    assert worker.requested_stop_reason == "budget"

    def fail_on_one_rank():
        if rank == 1:
            raise RuntimeError("injected final-save failure")

    try:
        final_checkpoint(worker, fail_on_one_rank)
    except RuntimeError:
        pass
    else:
        raise AssertionError("A rank-local save failure must fail every rank")
    assert acknowledgements == []

    flushed = []
    worker.trainer.ckpt_manager = SimpleNamespace(finalize=lambda: flushed.append(True))
    final_checkpoint(worker, lambda: None)
    assert flushed == [True]
    assert acknowledgements == ([1] if rank == 0 else [])
    dist.barrier()
    if rank == 0:
        print(
            "PASS: real-rank stop agreement, failed-save suppression, final-save ACK",
            flush=True,
        )
    loop.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
