# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process GPU checkpoint continuation through production LR rebuilds.

torchrun --standalone --nproc-per-node=2 tests/scheduler_resume_gpu_canary.py save ROOT
torchrun --standalone --nproc-per-node=2 tests/scheduler_resume_gpu_canary.py resume ROOT
"""

import argparse
import copy
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import cosmos_rl

from cosmos_rl.policy.config import Config
from cosmos_rl.policy.trainer.optm import OptimizersContainer, build_lr_schedulers
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.policy.worker.multi_replica_sft_worker import MultiReplicaSFTPolicyWorker
from cosmos_rl.utils.checkpoint import CheckpointMananger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("save", "resume"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--cpu", action="store_true", help="local Gloo harness check")
    parser.add_argument("--reference-reset", action="store_true")
    parser.add_argument(
        "--expected-package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cosmos_rl",
        help="Pin source-checkout or installed-wheel imports.",
    )
    args = parser.parse_args()
    assert (
        Path(cosmos_rl.__file__).resolve().parent
        == args.expected_package_root.resolve()
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not args.cpu:
        assert torch.cuda.is_available()
        torch.cuda.set_device(local_rank)
    device = torch.device("cpu" if args.cpu else f"cuda:{local_rank}")
    dist.init_process_group("gloo" if args.cpu else "nccl")
    config = Config.from_dict(
        {
            "train": {
                "output_dir": str(args.root / f"rank-{rank}"),
                "timestamp": "scheduler-continuity",
                "optm_warmup_steps": 4,
                "optm_decay_type": "linear",
                "optm_min_lr_factor": 0.0,
                "optm_warmup_start_factor": 0.0,
                "train_policy": {
                    "kl_beta": 0.1 if args.reference_reset else 0.0,
                    "reference_reset_interval": 3 if args.reference_reset else None,
                    "reset_optimizer_with_reference": True,
                },
                "ckpt": {
                    "enable_checkpoint": True,
                    "save_mode": "sync",
                    "upload_s3": False,
                    "export_safetensors": False,
                },
            }
        }
    )
    # Each rank owns its own test checkpoint directory. This validates native
    # gradient agreement and local resume state, not topology-changing restore.
    manager = CheckpointMananger(config)
    torch.manual_seed(31)
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    optimizer = OptimizersContainer(
        torch.optim.SGD, [model], [{"lr": 0.1, "momentum": 0.9}]
    )
    scheduler = build_lr_schedulers(optimizer, config, 20)
    expected_path = args.root / f"expected-{rank}.pt"
    current_step = 0 if args.phase == "save" else 3
    ref_trainer = GRPOTrainer.__new__(GRPOTrainer)
    ref_trainer.model, ref_trainer.config = model, config
    ref_trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    ref_trainer.optimizers, ref_trainer.lr_schedulers = optimizer, scheduler
    ref_trainer.ckpt_manager = manager
    ref_trainer.reference_state_dict = {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in model.state_dict().items()
    }
    ref_trainer.reference_reset_step = 0

    def rebuild_optimizer():
        nonlocal optimizer
        optimizer = OptimizersContainer(
            torch.optim.SGD, [model], [{"lr": 0.1, "momentum": 0.9}]
        )
        ref_trainer.optimizers = optimizer

    ref_trainer.build_optimizers = rebuild_optimizer

    def update():
        nonlocal current_step
        current_step += 1
        lrs = [group["lr"] for opt in optimizer for group in opt.param_groups]
        optimizer.zero_grad()
        value = torch.tensor([[float(rank + 1)]], device=model.weight.device)
        loss = model(value).square().sum()
        if args.reference_reset:
            # An anchored term makes resetting/reloading the reference affect
            # the next update, not just the checkpoint's metadata comparison.
            loss = (
                loss
                + 0.1
                * (model.weight - ref_trainer.reference_state_dict["weight"].to(device))
                .square()
                .sum()
            )
        loss.backward()
        dist.all_reduce(model.weight.grad)
        model.weight.grad.div_(dist.get_world_size())
        optimizer.step()
        if args.reference_reset:
            ref_trainer._finish_training_batch(
                current_step, 20, 20 - current_step, save=False
            )
        else:
            scheduler.step()
        result = {
            "lr_before": lrs,
            "model": manager.offload_state_dict_cpu(model.state_dict()),
            "optimizer": manager.offload_state_dict_cpu(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
        }
        if args.reference_reset:
            result["reference"] = copy.deepcopy(ref_trainer.reference_state_dict)
            result["reset_step"] = ref_trainer.reference_reset_step
        return result

    try:
        if args.phase == "save":
            for _ in range(3):
                update()
            if args.reference_reset:
                ref_trainer.save_checkpoint(3, 20, 17, is_final=False)
                checkpoint = os.path.join(manager.ckpt_output_dir, "step_3", "policy")
            else:
                checkpoint = manager.save_checkpoint(
                    model, optimizer, scheduler, step=3, total_steps=20
                )
            manager.finalize()
            torch.save(
                {"checkpoint": checkpoint, "updates": [update(), update()]},
                expected_path,
            )
            print(f"SCHEDULER_CHECKPOINT_PASS rank={rank} saved_step=3", flush=True)
        else:
            expected = torch.load(expected_path, map_location="cpu", weights_only=False)
            # Use the checkpoint API's returned leaf path, including its policy
            # subdirectory, rather than guessing the directory layout.
            config.train.resume = expected["checkpoint"]
            info, scheduler = manager.load_checkpoint(
                model,
                optimizer,
                partial(build_lr_schedulers, optimizer, config),
                model_name_or_path="unused-test-model",
            )
            assert info["step"] == 3 and info["total_steps"] == 20
            if args.reference_reset:
                ref_trainer._restore_checkpoint_reference(info)
            trainer = SimpleNamespace(
                optimizers=optimizer,
                lr_schedulers=scheduler,
                lr_schedulers_updated=False,
                build_lr_schedulers=lambda steps: build_lr_schedulers(
                    optimizer, config, steps
                ),
            )
            GRPOTrainer.update_lr_schedulers(trainer, 20)
            scheduler = trainer.lr_schedulers
            worker = object.__new__(MultiReplicaSFTPolicyWorker)
            worker.trainer = trainer
            worker.config = config
            worker.total_steps = None
            worker.loaded_total_steps = 20
            worker._prepare_lr_schedulers(20)
            assert trainer.lr_schedulers is scheduler
            ref_trainer.lr_schedulers = scheduler
            for step, reference in zip((4, 5), expected["updates"], strict=True):
                actual = update()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert actual["lr_before"][0] > 0
                print(
                    f"SCHEDULER_CONTINUITY_PASS rank={rank} resumed_update={step} exact_parity=True",
                    flush=True,
                )
                if args.reference_reset:
                    print(
                        f"REFERENCE_CONTINUITY_PASS rank={rank} resumed_update={step} reset_step=3 exact_parity=True",
                        flush=True,
                    )
    finally:
        manager.finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
