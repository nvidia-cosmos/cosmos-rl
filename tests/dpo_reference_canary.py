# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real FSDP reference swap, gradients, and sharded checkpoint continuation.

torchrun --standalone --nproc-per-node=2 tests/dpo_reference_canary.py
One rank also works as a local GPU smoke. Uses no downloaded models or datasets.
"""

import argparse
from datetime import timedelta
import os
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from test_dpo_reference_policy import batches, make_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-free", action="store_true")
    parser.add_argument("--keep-unsharded", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    os.environ["COSMOS_ALIGNMENT_DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("dp",))
    reference = not args.reference_free

    def fresh():
        trainer = make_trainer(False)
        trainer.use_reference_policy = reference
        fully_shard(
            trainer.model, mesh=mesh, reshard_after_forward=not args.keep_unsharded
        )
        if reference:
            trainer._capture_reference()
        return trainer

    trainer = fresh()
    oracle = make_trainer(False)
    oracle.use_reference_policy = reference
    if reference:
        oracle._capture_reference()
    with torch.no_grad():
        for model in (trainer.model, oracle.model):
            for parameter in model.parameters():
                parameter.add_(0.03)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.02)
    oracle_optimizer = torch.optim.AdamW(oracle.model.parameters(), lr=0.02)

    def rank_batch(which):
        pair = batches(trainer.device)
        for batch in pair:
            batch["input_ids"] = (batch["input_ids"] + which) % 5
        return pair

    local_chosen, local_rejected = rank_batch(rank)
    all_pairs = [rank_batch(peer) for peer in range(size)]
    global_chosen, global_rejected = [
        {
            key: torch.cat([pair[index][key] for pair in all_pairs])
            for key in local_chosen
        }
        for index in (0, 1)
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="dpo-reference-canary-") as directory:
            for step in range(3):
                for key, tensor in trainer.model.state_dict().items():
                    full = (
                        tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                    )
                    torch.testing.assert_close(
                        full,
                        oracle.model.state_dict()[key],
                        rtol=1e-9,
                        atol=1e-10,
                        msg=f"policy before forward: {key}",
                    )
                for key, tensor in trainer.reference_state_dict.items():
                    full = (
                        tensor.to(trainer.device).full_tensor()
                        if isinstance(tensor, DTensor)
                        else tensor.to(trainer.device)
                    )
                    torch.testing.assert_close(
                        full,
                        oracle.reference_state_dict[key].to(trainer.device),
                        rtol=0,
                        atol=0,
                        msg=f"reference before forward: {key}",
                    )
                loss = trainer._dpo_forward_and_loss(local_chosen, local_rejected)
                expected = oracle._dpo_forward_and_loss(global_chosen, global_rejected)
                mean_loss = loss.detach().clone()
                dist.all_reduce(mean_loss)
                mean_loss /= size
                torch.testing.assert_close(
                    mean_loss, expected.detach(), rtol=1e-10, atol=1e-10
                )
                loss.backward()
                expected.backward()
                optimizer.step()
                oracle_optimizer.step()
                optimizer.zero_grad()
                oracle_optimizer.zero_grad()
                for key, tensor in trainer.model.state_dict().items():
                    full = (
                        tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                    )
                    torch.testing.assert_close(
                        full, oracle.model.state_dict()[key], rtol=1e-9, atol=1e-10
                    )
                for key, tensor in trainer.reference_state_dict.items():
                    full = (
                        tensor.to(trainer.device).full_tensor()
                        if isinstance(tensor, DTensor)
                        else tensor.to(trainer.device)
                    )
                    torch.testing.assert_close(
                        full,
                        oracle.reference_state_dict[key].to(trainer.device),
                        rtol=0,
                        atol=0,
                    )
                if step == 0:
                    path = os.path.join(directory, "rank.pth")
                    torch.save(
                        {
                            "model": trainer.model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "dpo_reference_policy": reference,
                            "dpo_reference_state": trainer.reference_state_dict
                            if reference
                            else None,
                        },
                        path,
                    )
                    checkpoint = torch.load(
                        path, map_location="cpu", weights_only=False
                    )
                    trainer = fresh()
                    trainer.model.load_state_dict(checkpoint.pop("model"))
                    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.8)
                    optimizer.load_state_dict(checkpoint.pop("optimizer"))
                    trainer._restore_reference(checkpoint)
                print(
                    f"DPO_REFERENCE_STEP rank={rank} step={step} reference={reference} parity=True",
                    flush=True,
                )
        print(
            f"DPO_REFERENCE_PASS rank={rank} reference={reference} fsdp=True resumed=True",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
