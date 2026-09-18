# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise suppressed cleanup alongside real two-rank background transfers.

PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
    tests/cuda_cache_cleanup_canary.py --backend nccl

Use --backend gloo for CPU protocol smoke coverage. Use a finite outer timeout.
This validates the prevention policy, not reproduction of an allocator deadlock
or a full training workload. Direct raw flushing is deliberately not attempted
during communication; a test sentinel would fail if the helper reached it.
"""

import argparse
from datetime import timedelta
import os
import threading
from unittest.mock import patch

import torch
import torch.distributed as dist

from cosmos_rl.utils.cuda_cache import empty_cuda_cache, suppress_cuda_cache_cleanup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = (
        torch.device("cuda", local_rank)
        if args.backend == "nccl"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    assert empty_cuda_cache(), "Fresh process must retain pre-transport cleanup"
    suppress_cuda_cache_cleanup()
    dist.init_process_group(args.backend, timeout=timedelta(seconds=45))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    failures = []
    completed = []

    def transfer():
        try:
            if device.type == "cuda":
                torch.cuda.set_device(device)
            for step in range(20):
                expected = torch.full(
                    (262144,), step + 1, dtype=torch.float32, device=device
                )
                payload = expected if rank == 0 else torch.empty_like(expected)
                work = (
                    dist.isend(payload, dst=1)
                    if rank == 0
                    else dist.irecv(payload, src=0)
                )
                work.wait()
                assert torch.equal(payload, expected), (
                    f"Payload mismatch at step {step}"
                )
                completed.append(step)
        except BaseException as error:
            failures.append(error)

    with patch.object(
        torch.cuda, "empty_cache", side_effect=AssertionError("unsafe raw flush")
    ) as raw:
        thread = threading.Thread(target=transfer, daemon=True)
        thread.start()
        attempts = 0
        while thread.is_alive() or attempts < 100:
            assert not empty_cuda_cache()
            attempts += 1
            thread.join(0.001)
        if failures:
            raise failures[0]
        assert completed == list(range(20))
        raw.assert_not_called()
        dist.barrier()
        dist.destroy_process_group()
        assert not empty_cuda_cache(), "Teardown must not reenable flushing"
    print(
        f"PASS rank={rank} backend={args.backend} transfers=20 skipped_cleanup={attempts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
