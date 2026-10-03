# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank actual cached-UID P2P receive and delayed copy-back reuse canary.

Run with torchrun (two ranks, one GPU each), on one or two nodes. The P2P
manager's readiness exchange is replaced by Gloo; communicator creation,
send/receive and worker copy-back are native production paths. This is not a
full controller or colocated IPC test.
"""

from datetime import timedelta
import argparse
from contextlib import nullcontext
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_rl.collective.collective import P2RCollectiveManager
from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils.pynccl import create_nccl_uid, nccl_abort, nccl_group
from test_p2r_copyback_lifetime import exercise_copyback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    role = Role.POLICY if rank == 0 else Role.ROLLOUT
    manager = P2RCollectiveManager(
        "policy" if rank == 0 else "rollout",
        SimpleNamespace(world_size=1 if rank == 0 else 2),
        SimpleNamespace(mode="colocated_separated"),
        SimpleNamespace(),
        role,
    )
    manager.global_rank = rank
    # The paired non-IPC edge is policy rank 0 -> rollout rank 1.
    command = PolicyToRolloutUnicastCommand("policy", "rollout", 1, 2)
    key = manager.generate_mesh_key(command, 0, 1, is_p2p=True)
    uid = [create_nccl_uid() if rank == 0 else None]
    dist.broadcast_object_list(uid, 0)
    manager.unique_ids_cache[key] = uid[0]
    manager._wait_transfer_ready = lambda *args, **kwargs: None
    try:
        manager._setup_p2p_communicators(command)
        first_handle = manager.nccl_comm_cache[key]
        for step in range(2):
            manager._setup_p2p_communicators(command)
            assert manager.nccl_comm_cache[key] == first_handle
            if rank == 0:
                values = [
                    torch.full((1 << 20,), value, dtype=torch.float16, device="cuda")
                    for value in (1, 2)
                ]
                with nccl_group(first_handle) if args.grouped else nullcontext():
                    for tensor in values:
                        manager.send("policy_rollout", tensor, 1)
                torch.cuda.synchronize()
            else:
                exercise_copyback(
                    native_recv=lambda tensor: manager.recv(
                        "policy_rollout", tensor, 0
                    ),
                    rollout_rank=1,
                    receive_scope=(
                        (lambda: nccl_group(first_handle))
                        if args.grouped
                        else nullcontext
                    ),
                )
            dist.barrier()
            print(
                f"P2R_COPYBACK_PASS rank={rank} step={step} cached_uid=True delayed_copy=True grouped={args.grouped}",
                flush=True,
            )
    finally:
        for handle in manager.nccl_comm_cache.values():
            nccl_abort(handle)
        manager.nccl_comm_cache.clear()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
