# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe-boundary stop integration for worker-owned training loops."""

import torch
import torch.distributed as dist
from cosmos_rl.utils import distributed as dist_util


def training_boundary(worker, completed_step):
    """All ranks receive the same decision; no rank starts a divergent update."""
    decision = None
    if worker.global_rank == 0:
        try:
            decision = worker.api_client.training_boundary(
                worker.replica_name, completed_step
            )
        except Exception as error:
            decision = {"error": str(error)}
    decision = dist_util.broadcast_object_cpu(decision, src=0)
    if "error" in decision:
        raise RuntimeError(decision["error"])
    worker.requested_stop_reason = decision["reason"]
    return decision["stop"]


def final_checkpoint(worker, save):
    """No successful completion acknowledgement before save/futures agreement."""
    error = None
    try:
        save()
        manager = getattr(worker.trainer, "ckpt_manager", None)
        if manager is not None:
            manager.finalize()
    except Exception as failure:
        error = failure
    success = dist_util.all_reduce_tensor_object_cpu(
        torch.tensor([error is None], dtype=torch.int32), op=dist.ReduceOp.MIN
    ).item()
    if not success:
        if error is not None:
            raise error
        raise RuntimeError("Final checkpoint failed on another rank")
    decision = None
    if worker.global_rank == 0:
        try:
            worker.api_client.training_boundary(
                worker.replica_name, worker.train_step, checkpoint_complete=True
            )
            decision = {"ok": True}
        except Exception as failure:
            decision = {"error": str(failure)}
    decision = dist_util.broadcast_object_cpu(decision, src=0)
    if "error" in decision:
        raise RuntimeError(decision["error"])
