# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU byte round trip: torchrun --standalone --nproc-per-node=2 this_file."""

import os
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from cosmos_rl.utils.payload_transport.pack import pack_trajectory_into
from cosmos_rl.utils.payload_transport.nccl.strategy import _unpack
from cosmos_rl.utils.tensor_packing import (
    pack_tensors_into_buffer,
    unpack_tensors_from_buffer,
)
from cosmos_rl.utils.trajectory import (
    TensorSpec,
    build_trajectory_schema,
    schema_layout,
)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        schema = build_trajectory_schema(dict(max_steps=4, obs_dim=3, action_dim=1))
        schema += [
            TensorSpec((), np.float64, "scalar"),
            TensorSpec((1,), np.int64, "mode"),
        ]
        offsets, size = schema_layout(schema)
        payload = torch.empty(size, dtype=torch.uint8, device=device)
        for iteration in range(3):
            if rank == 0:
                pack_trajectory_into(
                    payload,
                    {
                        "observations": torch.full(
                            (2, 3), iteration + 1.0, dtype=torch.bfloat16, device=device
                        ),
                        "scalar": torch.tensor(
                            iteration + 0.5, dtype=torch.float64, device=device
                        ),
                        "mode": torch.arange(8, device=device).reshape(1, 8)[:, 3],
                    },
                    schema,
                    offsets,
                    2,
                    device,
                )
            dist.broadcast(payload, src=0)
            restored = _unpack(payload, schema, device)
            torch.testing.assert_close(
                restored["observations"],
                torch.full((2, 3), iteration + 1.0, device=device),
            )
            assert restored["scalar"].item() == iteration + 0.5
            assert restored["mode"].item() == 3
            assert torch.count_nonzero(restored["actions"]).item() == 0

            # Exercise both directions using real singleton-strided destinations.
            backing = torch.full((1, 8), -1.0, device=device)
            destination = backing[:, 5]
            scalar = torch.tensor(0.0, device=device)
            wire = torch.empty(8, dtype=torch.uint8, device=device)
            if rank == 0:
                source = torch.arange(8.0, device=device).reshape(1, 8)[:, 2]
                pack_tensors_into_buffer(
                    [source, torch.tensor(7.0, device=device)], wire
                )
            dist.broadcast(wire, src=0)
            unpack_tensors_from_buffer(wire, [destination, scalar])
            assert destination.item() == 2 and scalar.item() == 7
            assert torch.all(backing[:, :5] == -1) and torch.all(backing[:, 6:] == -1)
        dist.barrier()
        print(f"PAYLOAD_NORMALIZATION_CANARY_PASS rank={rank} iterations=3", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
