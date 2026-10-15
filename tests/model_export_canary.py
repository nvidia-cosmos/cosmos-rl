# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank pipeline export: real shards, complete index, collective write error.

torchrun --standalone --nproc-per-node=2 tests/model_export_canary.py \
    --device cuda --case healthy --output /tmp/export-case
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
import cosmos_rl
from cosmos_rl.policy.trainer.llm_trainer.llm_trainer import LLMTrainer
from cosmos_rl.utils.model_export import finish_model_export, ModelExportThread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument(
        "--case",
        choices=["healthy", "shard-failure", "previous-failure"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cosmos_rl",
        help="Pin the imported package for source-checkout or installed-wheel runs.",
    )
    args = parser.parse_args()
    assert (
        Path(cosmos_rl.__file__).resolve().parent
        == args.expected_package_root.resolve()
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cpu")
    if args.device == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=60)
    )
    assert dist.get_world_size() == 2
    model = torch.nn.Linear(2, 1, bias=False, device=device)
    model.weight.data.fill_(rank + 3)
    model.weight_mapper = SimpleNamespace(
        policy_map_local_key_to_hf_key=lambda name: f"stage_{rank}.{name}",
        policy_map_local_key_for_export_tensor=lambda name, tensor: [(name, tensor)],
    )
    trainer = SimpleNamespace(
        model_parts=[model],
        global_rank=rank,
        upload_thread=None,
        data_packer=None,
        hf_config=SimpleNamespace(save_pretrained=lambda path: None),
        parallel_dims=SimpleNamespace(
            dp_replicate_coord=(0, 1),
            dp_replicate_enabled=False,
            dp_shard_coord=(0, 1),
            cp_coord=(0, 1),
            tp_coord=(0, 1),
            pp_coord=(rank, 2),
            pp_enabled=True,
            pp=2,
            mesh={"pp": SimpleNamespace(get_group=lambda: dist.group.WORLD)},
        ),
        config=SimpleNamespace(
            policy=SimpleNamespace(
                lora=None, model_name_or_path=str(args.output), model_revision=None
            ),
            train=SimpleNamespace(
                ckpt=SimpleNamespace(upload_hf=False, upload_s3=False)
            ),
        ),
    )
    index_path = args.output / "export/model.safetensors.index.json"
    if args.case == "previous-failure" and rank == 0:

        def fail_previous():
            raise OSError("injected previous export failure")

        trainer.upload_thread = ModelExportThread(target=fail_previous)
        trainer.upload_thread.start()

    def write(values, path):
        assert not index_path.exists(), "manifest published before all stage writes"
        if rank == 1:
            time.sleep(0.2)
            assert not index_path.exists(), "delayed stage was not awaited"
            if args.case == "shard-failure":
                raise OSError("injected stage-1 shard failure")
        save_file(values, path)

    failure = None
    try:
        with (
            patch("cosmos_rl.policy.trainer.llm_trainer.llm_trainer.save_file", write),
            patch(
                "cosmos_rl.policy.trainer.llm_trainer.llm_trainer.GenerationConfig.from_pretrained",
                side_effect=FileNotFoundError("no generation config"),
            ),
            patch(
                "cosmos_rl.utils.util.resolve_model_path",
                lambda *a, **k: str(args.output),
            ),
        ):
            try:
                LLMTrainer.export_safetensors(trainer, str(args.output), "export")
                model.weight.data.add_(100)
                finish_model_export(trainer)
            except RuntimeError as error:
                failure = str(error)
        dist.barrier()
        if args.case == "previous-failure":
            assert failure is not None and "Previous pipeline export failed" in failure
            assert not index_path.exists()
            assert not list(args.output.glob("*.incomplete-*/*.safetensors"))
        elif args.case == "shard-failure":
            assert failure is not None and "injected stage-1 shard failure" in failure
            assert not index_path.exists()
        else:
            assert failure is None, failure
            index = json.loads(index_path.read_text())
            assert set(index["weight_map"]) == {"stage_0.weight", "stage_1.weight"}
            assert index["metadata"]["total_size"] == 16
            for stage in range(2):
                name = f"stage_{stage}.weight"
                saved = load_file(str(index_path.parent / index["weight_map"][name]))
                torch.testing.assert_close(saved[name], torch.full((1, 2), stage + 3.0))
        print(f"rank={rank} case={args.case} pipeline_export=PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
