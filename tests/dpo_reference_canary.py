# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real FSDP reference swap, gradients, and sharded checkpoint continuation.

torchrun --standalone --nproc-per-node=2 tests/dpo_reference_canary.py
One rank also works as a local GPU smoke. Uses no downloaded models or datasets.
"""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace

import cosmos_rl
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from test_dpo_reference_policy import batches, make_trainer
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.checkpoint import CheckpointMananger


def check_controller_agreement(manager, worker_info):
    """Use the production HTTP client/route with independent checkpoint reads."""
    import uvicorn
    from fastapi import FastAPI
    from cosmos_rl.dispatcher import run_web_panel as panel
    from cosmos_rl.dispatcher.api.client import APIClient
    from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher

    fetcher = object.__new__(ControllerDataFetcher)
    fetcher.ckpt_extra_info = manager.load_extra_info_from_checkpoint()
    panel.controller = SimpleNamespace(data_fetcher=fetcher)
    app = FastAPI()
    app.post("/resume")(panel.resume_info)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("Resume HTTP fixture did not start")
            time.sleep(0.01)
        client = object.__new__(APIClient)
        client.max_retries = 1
        client.get_alternative_urls = lambda suffix: [
            f"http://127.0.0.1:{listener.getsockname()[1]}/resume"
        ]
        client.post_resume_info(worker_info)
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "Resume HTTP fixture did not shut down"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-free", action="store_true")
    parser.add_argument("--keep-unsharded", action="store_true")
    parser.add_argument("--expected-package-root", type=Path)
    args = parser.parse_args()
    if args.expected_package_root is not None:
        assert (
            Path(cosmos_rl.__file__).resolve().parent
            == args.expected_package_root.resolve()
        )
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
                    # Each process owns a shard's artifact here; FSDP still runs
                    # across all ranks. This does not claim cohort discovery.
                    config = Config.from_dict(
                        {
                            "train": {
                                "output_dir": directory,
                                "timestamp": "run",
                                "ckpt": {
                                    "enable_checkpoint": True,
                                    "save_mode": "sync",
                                    "export_safetensors": False,
                                },
                            }
                        }
                    )
                    manager = CheckpointMananger(config)
                    trainer.config = config
                    trainer.ckpt_manager = manager
                    trainer.parallel_dims = SimpleNamespace(
                        dp_replicate_coord=(0, 1), dp_replicate_enabled=False
                    )
                    trainer.optimizers = optimizer
                    trainer.lr_schedulers = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=1
                    )
                    trainer.checkpointing(total_steps=3, train_step=1, save_freq=1)
                    config.train.resume = os.path.join(
                        manager.ckpt_output_dir, "step_1", "policy"
                    )
                    trainer = fresh()
                    trainer.config = config
                    trainer.parallel_dims = SimpleNamespace(
                        dp_replicate_coord=(0, 1), dp_replicate_enabled=False
                    )
                    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.8)
                    trainer.optimizers = optimizer
                    trainer.lr_schedulers = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=1
                    )
                    trainer.model_resume_from_checkpoint = (
                        lambda: manager.load_checkpoint(
                            trainer.model, optimizer, trainer.lr_schedulers, "unused"
                        )[0]
                    )
                    total, saved_step, metadata = trainer.load_model()
                    assert (total, saved_step) == (3, 1)
                    check_controller_agreement(manager, metadata)
                    print(
                        f"DPO_HTTP_AGREEMENT_PASS rank={rank} reference={reference}",
                        flush=True,
                    )
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
