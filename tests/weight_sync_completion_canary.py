# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A warm native R2R peer stalls while a weight write owns its buffer."""

from datetime import timedelta
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from cosmos_rl.rollout.worker import weight_sync as ws
from cosmos_rl.utils import pynccl
from test_weight_adoption import make_worker


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ["COSMOS_NCCL_TIMEOUT_MS"] = "60000"
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    device = torch.device("cuda", local_rank)
    worker, _ = make_worker(device)
    worker.replica_name = f"rollout-{rank}"
    worker.replica_name_to_rank = {"rollout-0": 0, "rollout-1": 1}
    worker.rank_in_rollout_repicas = rank
    worker.config.rollout.r2r_sync_pack_tensors = True
    worker._buffer_version = 1
    worker._buffer_weight_version = 1
    uid = [pynccl.create_nccl_uid() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    worker.global_commnicator_idex = pynccl.create_nccl_comm(uid[0], rank, 2)
    wst = worker._weight_sync_thread
    wst._stream.wait_event(worker._buffer_adopt_event)

    # Establish the real communicator and kernels before arming the short fault
    # deadline. Packing gives a standalone native broadcast without GroupEnd.
    ws.do_nccl_broadcast_grouped(worker, "rollout-0", wst._stream)
    wst._stream.synchronize()
    dist.barrier()
    if rank == 1:
        print("R2R_PEER_STALLED_AFTER_READINESS", flush=True)
        time.sleep(30)
        os._exit(3)

    os.environ["COSMOS_NCCL_TIMEOUT_MS"] = "1500"
    command = SimpleNamespace(
        weight_step=1,
        total_steps=2,
        src_replica_name="rollout-0",
        dst_replica_names=["rollout-0", "rollout-1"],
        replica_should_stop=lambda: False,
    )
    # The peer passed readiness and then stalled before its next native call.
    with patch.object(ws, "r2r_barrier", return_value=True):
        wst._execute_r2r(command)
    print("UNEXPECTED_WEIGHT_COMPLETION", flush=True)
    torch.cuda.current_stream().wait_event(wst._last_event)
    torch.cuda.synchronize()
    raise AssertionError("An unmatched broadcast cannot complete")


if __name__ == "__main__":
    main()
