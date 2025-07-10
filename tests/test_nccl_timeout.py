# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.distributed as dist
import subprocess

# For testing, set the timeout to 60000ms
os.environ["COSMOS_NCCL_TIMEOUT_MS"] = "60000"
from cosmos_rl.utils.distributed import broadcast_object_cpu, nccl_timeout_watchdog
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_allreduce,
    nccl_abort,
)


def routine(N: int, device: torch.device, rank: int, world_size: int):
    nccl_uid = broadcast_object_cpu(create_nccl_uid() if rank == 0 else None)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        comm = create_nccl_comm(nccl_uid, rank, world_size)
        try:
            with nccl_timeout_watchdog(wait_stream=True):
                send_buffer = torch.arange(N, dtype=torch.float32, device=device)
                recv_buffer = torch.zeros(N, dtype=torch.float32, device=device)

                if rank != 3:
                    # Simulate a failed allreduce on rank 3
                    try:
                        nccl_allreduce(send_buffer, recv_buffer, 0, comm)
                        print(f"[RANK {rank}] arrived here")
                    except Exception as e:
                        print(f"[RANK {rank}] error in nccl_allreduce: {e}")
                else:
                    nccl_abort(comm)
        except Exception as e:
            print(f"[RANK {rank}] error in nccl_timeout_watchdog: {e}")

        from cosmos_rl.utils import pynccl as _pynccl

        print("-" * 80)
        print(
            f"[rank {rank}] comm_store keys after exit watchdog: {list(_pynccl._COMM_REGISTRY._store.keys())}"
        )
        print(f"[rank {rank}] task queue size: {_pynccl._task_q.qsize()}")
        print("-" * 80)

        if rank < 2:
            # Test that the new group still works after the previous nccl failure
            print(f"[rank {rank}] creating 2nd comm")
            world_size = 2
            group_ranks = [0, 1]
            new_group = dist.new_group(ranks=group_ranks, backend="cuda:nccl,cpu:gloo")

            nccl_uid = broadcast_object_cpu(
                create_nccl_uid() if rank == 0 else None, group=new_group
            )
            comm = create_nccl_comm(nccl_uid, rank, world_size)
            send_tensor = torch.arange(N, dtype=torch.float32, device=device)
            recv_tensor = torch.zeros(N, dtype=torch.float32, device=device)
            nccl_allreduce(send_tensor, recv_tensor, 0, comm)
            torch.testing.assert_close(
                recv_tensor,
                torch.arange(N, dtype=torch.float32, device=device) * world_size,
            )
            nccl_abort(comm)

        print(f"[rank {rank}] finished the test")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ###
    # Global timeout: 60s
    # 1. We initialize comm and launch allreduce on all ranks except rank 3
    # 2. Rank 3 will sleep for 100s
    #     By design, the allreduce could trigger a cpu waiting timeout on all ranks except rank 3
    # 3. Rank 0 and 1 will continue to run and complete the allreduce
    routine(N=8, device=device, rank=rank, world_size=world_size)

    print(f"[RANK {rank}] Running 2nd routine\n")
    from importlib.metadata import version

    nccl_v = version("nvidia-nccl-cu12")
    if nccl_v >= "2.26.2":
        routine(N=512 * 4096 * 512, device=device, rank=rank, world_size=world_size)

    # Global barrier to ensure all ranks finished routines
    dist.barrier()

    # Wait background worker to drain queue and communicator store
    from cosmos_rl.utils import pynccl as _pynccl

    def _wait_for_cleanup(timeout_s: float = 10.0, interval: float = 0.02) -> bool:
        import time

        start = time.time()
        while time.time() - start < timeout_s:
            if _pynccl._task_q.qsize() == 0 and len(_pynccl._COMM_REGISTRY._store) == 0:
                return True
            time.sleep(interval)
        return False

    clean_ok = _wait_for_cleanup()
    if dist.get_rank() == 0:
        print("=" * 80)
        pending_tasks = list(_pynccl._task_q.queue)
        print(f"[test] remaining task count: {len(pending_tasks)}")
        if pending_tasks:
            print("[test] pending tasks:")
            for t in pending_tasks:
                print("   ", t)

        print(
            f"[test] remaining comm_store keys: {list(_pynccl._COMM_REGISTRY._store.keys())}"
        )

        if clean_ok:
            print(
                "[test] cleanup succeeded – task queue and comm store are empty. All tests passed."
            )
        else:
            raise AssertionError(
                "[test] cleanup timed out – background worker did not drain queues in time."
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    if os.environ.get("RECURSIVE_ENTRYPOINT") is None:
        n_gpu = torch.cuda.device_count()
        command = [
            "torchrun",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(n_gpu),
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "localhost:12345",
            os.path.abspath(__file__),
        ]
        env = os.environ.copy()
        env["RECURSIVE_ENTRYPOINT"] = "1"
        subprocess.run(command, env=env)
    else:
        main()
