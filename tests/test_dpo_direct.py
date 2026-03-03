# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Toy test for Nemotron VLM DPO training - runs directly without controller.
# Tests: DPODataset, HFVLMDataPacker (dpo_process_sample, dpo_collate_fn), DPOTrainer
#
# Usage (single GPU):
#   cd nemotron_vl
#   export USE_QWEN_VL_PROCESS=1
#   export USE_SIGLIP2_PROCESS=1
#   torchrun --nproc_per_node=1 test_dpo_direct.py
#
# Usage (multi GPU, e.g. 8 GPUs for TP=8):
#   torchrun --nproc_per_node=8 test_dpo_direct.py --tp_size 8
# Usage /workspace/ruipul/vlm_data/MMPR-v1.2
#   torchrun --nproc_per_node=8 tests/test_dpo_direct.py --tp_size 8 --num_steps 2 --dataset_path /workspace/ruipul/vlm_data/MMPR-v1.2
#
# For tp_size > 1, dataloader_broadcast=True ensures rank 0 generates batches
# and broadcasts to other ranks (avoids empty batch on non-zero ranks).

import os
import sys
import argparse

# Must set before any cosmos_rl imports
os.environ.setdefault("USE_QWEN_VL_PROCESS", "1")
os.environ.setdefault("USE_SIGLIP2_PROCESS", "1")
os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"

# Add nemotron_vl to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

import torch
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.worker import SFTPolicyWorker
from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed
from cosmos_rl.utils import util
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.utils.logging import logger
from transformers import AutoConfig
import uuid

# Import nemotron-specific modules for monkey patching
import sys

print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_vl"))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_vl")
)
from nemotron_parallelize import parallelize
from weight_converter import convert_weight_from_hf
from launcher_dpo import (
    policy_map_local_key_for_export_tensor,
    step_hook,
    get_dataset as get_dpo_dataset,
)


def patched_parallelize_fn(self):
    return parallelize, self


def build_toy_config(
    model_path: str,
    dataset_path: str,
    tp_size: int = 1,
    dp_shard_size: int = 1,
    dp_replicate_size: int = 1,
    num_train_steps: int = 4,
    train_layers: Optional[list] = None,
) -> CosmosConfig:
    """Build minimal config for toy DPO test."""
    full_config = {
        "redis": "12800",
        "train": {
            "resume": False,
            "epoch": 1,
            "output_dir": "./outputs/test_dpo_direct",
            "epsilon": 1e-6,
            "optm_name": "AdamW",
            "optm_lr": 1e-4,
            "optm_impl": "fused",
            "optm_weight_decay": 0.01,
            "optm_betas": [0.9, 0.999],
            "optm_warmup_steps": 0.03,
            "optm_decay_type": "cosine",
            "optm_grad_norm_clip": 1.0,
            "async_tp_enabled": False,
            "compile": False,
            "param_dtype": "bfloat16",
            "fsdp_reduce_dtype": "float32",
            "master_dtype": "float32",
            "fsdp_offload": False,
            "fsdp_reshard_after_forward": "default",
            "train_batch_per_replica": 2,
            "sync_weight_interval": 1,
            "enable_validation": False,
            "validation_step": 30,
            "validation_batch_per_replica": 2,
            "max_num_steps": num_train_steps,
            "train_policy": {
                "type": "sft",
                "trainer_type": "dpo",
                "dataset": {"name": dataset_path, "subset": "", "split": "train"},
                "conversation_column_name": "",
                "mini_batch": 2,
                "dataloader_shuffle": True,
                "dataloader_num_workers": 0,
                "dataloader_prefetch_factor": 1,
                "dataloader_broadcast": True,  # For tp>1: rank 0 generates batch and broadcasts
            },
            "ckpt": {
                "enable_checkpoint": False,
                "save_freq": 10000,
                "save_mode": "async",
            },
        },
        "policy": {
            "model_name_or_path": model_path,
            "model_safetensor_path": "/workspace/ruipul/cosmos-rl-private/nemotron_vl/outputs/siglip2_official_nemotron_vl_stage1/20260224071826/safetensors/step_12105",
            "model_max_length": 8192,
            "model_gradient_checkpointing": True,
            "parallelism": {
                "n_init_replicas": 1,
                "tp_size": tp_size,
                "cp_size": 1,
                "dp_shard_size": dp_shard_size,
                "pp_size": 1,
                "dp_replicate_size": dp_replicate_size,
                "cp_rotate_method": "allgather",
            },
        },
        "logging": {
            "logger": ["console"],
            "project_name": "nemotron_vl_test",
            "experiment_name": "test_dpo_direct",
        },
        "validation": {"enable": False},
        "custom": {
            "enable_moe_load_balancing_training": False,
            "train_layers": train_layers,  # None=train all (required when tp>1); ["PatchMerger"] for fast single-GPU test
            "include_video": True,
            "dpo_beta": 0.1,
            # MPO paper: sigmoid (preference) + bco_pair (quality) + sft
            "dpo_loss_type": ["sigmoid", "bco_pair"],
            "dpo_loss_weights": [0.8, 0.2],
        },
    }
    return CosmosConfig.from_dict(full_config)


def run_dpo_direct(
    model_path: str,
    dataset_path: str,
    tp_size: int = 1,
    num_train_steps: int = 4,
):
    """Run DPO training directly without controller - tests DPODataset, HFVLMDataPacker, DPOTrainer."""

    # PatchMerger-only + tp>1 causes "mixed torch.Tensor and DTensor" in norm.backward.
    # Nemotron requires tp>1 -> must use train_layers=None when tp>1.
    # train_layers = None if tp_size > 1 else ["PatchMerger"]
    train_layers = ["PatchMerger"]

    # 1. Apply nemotron monkey patches (same as launcher_dpo.py)
    import cosmos_rl

    cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(
        patched_parallelize_fn
    )
    cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
    cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
    cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor

    # 2. Mock CommMixin.init_comm to skip controller registration
    from cosmos_rl.comm.base import CommMixin
    from cosmos_rl.dispatcher.api.client import APIClient

    def dummy_init_comm(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        logger.info(f"[Test] model type {hf_config.model_type}")

    CommMixin.init_comm = dummy_init_comm

    # 3. Build config and parallel_dims
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dp_replicate_size = max(1, world_size // tp_size)
    config = build_toy_config(
        model_path=model_path,
        dataset_path=dataset_path,
        tp_size=tp_size,
        dp_shard_size=1,
        dp_replicate_size=dp_replicate_size,
        num_train_steps=num_train_steps,
        train_layers=train_layers,
    )

    init_distributed()
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    parallel_dims.build_mesh(device_type="cuda")

    # 4. Create SFTPolicyWorker with DPO dataset and HFVLMDataPacker
    data_packer = HFVLMDataPacker()
    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        dataset=get_dpo_dataset,
        data_packer=data_packer,
        val_data_packer=data_packer,
    )

    # 5. Run training loop
    losses = []
    for step, global_batch in enumerate(
        sft_worker.get_batch_from_dataloader(sft_worker.train_data_loader)
    ):
        if step >= num_train_steps:
            break
        data_arrival_event = torch.cuda.Event(enable_timing=True)
        data_arrival_event.record()
        report_data = sft_worker.trainer.step_training(
            global_batch=global_batch,
            total_steps=sft_worker.total_steps,
            train_step=sft_worker.train_step,
            save_freq=sft_worker._save_freq,
            data_arrival_event=data_arrival_event,
        )
        sft_worker.train_step += 1
        if report_data and util.is_master_rank(parallel_dims, sft_worker.global_rank):
            losses.append(report_data.get("train/loss_avg", 0))
            logger.info(
                f"Step {sft_worker.train_step}/{sft_worker.total_steps}, "
                f"Loss: {report_data.get('train/loss_avg', 0):.5f}"
            )

    global_rank = sft_worker.global_rank
    destroy_distributed()

    if global_rank == 0:
        assert (
            len(losses) == num_train_steps
        ), f"Expected {num_train_steps} steps, got {len(losses)}"
        logger.info(f"[Test] DPO direct run completed. Losses: {losses}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct DPO test for Nemotron VLM (no controller)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/nemotron-vlm/NVIDIA-Nemotron-3-Nano-SIGLIP2-officaial-30B-A3B-BF16",
        help="Path to Nemotron model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/ruipul/projects/vlm_data_curation/data_clean/debug_samples",
        help="Path to MMPR-style DPO dataset (with meta.json)",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size (default 1 for single GPU)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="Number of training steps to run",
    )
    args = parser.parse_args()

    run_dpo_direct(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        tp_size=args.tp_size,
        num_train_steps=args.num_steps,
    )


if __name__ == "__main__":
    main()
