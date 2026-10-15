# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint artifacts: newest healthy restore or terminal selected failure.

torchrun --standalone --nproc-per-node=2 tests/resume_selection_canary.py ROOT
"""

import argparse
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_rl.policy.config import Config
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer
from cosmos_rl.utils.checkpoint import CheckpointMananger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cpu" if args.cpu else f"cuda:{local_rank}")
    if not args.cpu:
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    dist.init_process_group("gloo" if args.cpu else "nccl")
    try:
        for case in (
            "healthy",
            "corrupt-optimizer",
            "corrupt-metadata",
            "missing-metadata",
        ):
            # Rank-local artifact roots isolate this selection test from topology
            # migration. Training updates still use a real gradient collective.
            root = args.root / case / f"rank-{rank}"
            assert not root.exists(), "Use a fresh evidence directory"
            config = Config.from_dict(
                {
                    "train": {
                        "output_dir": str(root / "saved"),
                        "timestamp": "saved",
                        "resume": False,
                        "ckpt": {
                            "enable_checkpoint": True,
                            "save_mode": "sync",
                            "max_keep": 10,
                            "upload_s3": False,
                            "export_safetensors": False,
                        },
                    }
                }
            )
            manager = CheckpointMananger(config)
            torch.manual_seed(51)
            model = torch.nn.Linear(2, 1).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

            def update():
                optimizer.zero_grad()
                model(
                    torch.full((1, 2), float(rank + 1), device=device)
                ).square().sum().backward()
                for param in model.parameters():
                    dist.all_reduce(param.grad)
                    param.grad.div_(dist.get_world_size())
                optimizer.step()
                scheduler.step()

            for step in (1, 2):
                update()
                selected = Path(
                    manager.save_checkpoint(model, optimizer, scheduler, step, 20)
                )
            manager.finalize()
            update()
            expected = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            if case == "corrupt-optimizer":
                (selected / "optimizer_rank_0.pth").write_bytes(
                    b"corrupt committed artifact"
                )
            elif case == "corrupt-metadata":
                (selected / "extra_info_rank_0.pth").write_bytes(
                    b"corrupt committed artifact"
                )
            elif case == "missing-metadata":
                (selected / "extra_info_rank_0.pth").unlink()
            config.train.resume = True
            hf_calls = []

            def fresh_weights():
                hf_calls.append(True)
                raise AssertionError("Selected restore fell back to fresh weights")

            def restore():
                info, _ = manager.load_checkpoint(model, optimizer, scheduler, "unused")
                return info

            trainer = SimpleNamespace(
                config=config,
                model_resume_from_checkpoint=restore,
                model_load_from_hf=fresh_weights,
                _restore_checkpoint_reference=lambda info: None,
                set_model_train=model.train,
                map_w_from_policy_to_rollout=object(),
            )
            if case == "healthy":
                info = GRPOTrainer.weight_resume(trainer)
                assert info["step"] == 2
                assert Path(manager.selected_checkpoint_path) == selected.resolve()
                update()
                torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
            else:
                try:
                    GRPOTrainer.weight_resume(trainer)
                except (pickle.UnpicklingError, FileNotFoundError) as error:
                    assert not hf_calls
                    if case == "missing-metadata":
                        assert type(error) is FileNotFoundError
                    else:
                        assert isinstance(error, pickle.UnpicklingError)
                    assert Path(manager.selected_checkpoint_path) == selected.resolve()
                else:
                    raise AssertionError("Selected corrupt checkpoint silently resumed")
            assert not hf_calls
            # Harness synchronization only; this adds no production recovery or
            # agreement collective and does not test launcher propagation.
            dist.barrier()
            print(
                f"RESUME_SELECTION_PASS rank={rank} case={case} fresh_fallback=False",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
