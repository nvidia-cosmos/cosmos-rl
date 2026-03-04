# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Toy test for Nemotron VLM SFT training - runs directly without controller.
# Tests: dataset, data_packer (HFVLMDataPacker), SFTTrainer
#
# 换 data_packer: 1) import 你的 packer; 2) 在 main() 里 data_packer = YourPacker()
#    当前 CustomDataset 返回 messages，需兼容 sft_process_sample(messages) 的 packer
# 换 dataset: 传入 factory 函数 get_dataset(config) -> Dataset，或传 None 用 config 从 HF/磁盘加载
# 换 trainer: 设 trainer_type="sft"|"pi_sft" 等。注意 grpo 需 RLPolicyWorker+rollout，本测试用 SFTPolicyWorker
#
# Usage (single GPU):
#   cd nemotron_vl
#   export USE_QWEN_VL_PROCESS=1
#   export USE_SIGLIP2_PROCESS=1
#   torchrun --nproc_per_node=1 test_sft_direct.py
#
# Usage (multi GPU, e.g. 2 GPUs for TP=2):
#   torchrun --nproc_per_node=2 test_sft_direct.py --tp_size 2

import os
import sys
import argparse

# Must set before any cosmos_rl imports
os.environ.setdefault("USE_QWEN_VL_PROCESS", "1")
os.environ.setdefault("USE_SIGLIP2_PROCESS", "1")
os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"

# Add nemotron_vl to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
import sys

print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_vl"))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_vl")
)
# Import nemotron-specific modules for monkey patching
from nemotron_parallelize import parallelize
from weight_converter import convert_weight_from_hf
from launcher import (
    policy_map_local_key_for_export_tensor,
    step_hook,
    get_dataset as get_launcher_dataset,
)


def get_my_dataset(config: CosmosConfig):
    """自定义 dataset 示例。替换 dataset=get_my_dataset 即可使用。"""
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def setup(self, config, *args, **kwargs):
            # 从 config.train.train_policy.dataset.name 或 HF 等加载
            # self.data = [...]  每条为 messages: [{"role":"user","content":[...]}, ...]
            self.data = []

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[
                idx
            ]  # messages 格式，需与 data_packer.sft_process_sample 兼容

    return MyDataset()


def patched_parallelize_fn(self):
    return parallelize, self


def build_toy_config(
    model_path: str,
    tp_size: int = 1,
    dp_shard_size: int = 1,
    dp_replicate_size: int = 1,
    num_train_steps: int = 4,
    trainer_type: str = "sft",
) -> CosmosConfig:
    """Build minimal config for toy SFT test."""
    full_config = {
        "redis": "12800",
        "train": {
            "resume": False,
            "epoch": 1,
            "output_dir": "./outputs/test_sft_direct",
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
                "trainer_type": trainer_type,  # 换 trainer: "sft"|"grpo"|自定义注册的 type
                "dataset": {
                    "name": "/workspace/haoyuan/vlm_data/stage1",
                    "subset": "",
                    "split": "train",
                },
                "conversation_column_name": "conversation",  # CustomDataset returns messages directly
                "mini_batch": 2,
                "dataloader_shuffle": True,
                "dataloader_num_workers": 0,
                "dataloader_prefetch_factor": 1,
            },
            "ckpt": {
                "enable_checkpoint": False,
                "save_freq": 10000,
                "save_mode": "async",
            },
        },
        "policy": {
            "model_name_or_path": model_path,
            "model_safetensor_path": model_path,
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
            "experiment_name": "test_sft_direct",
        },
        "validation": {"enable": False},
        "custom": {
            "enable_moe_load_balancing_training": False,
            "train_layers": [
                "PatchMerger"
            ],  # Use "PatchMerger" (not "FSDPPatchMerger") when not using FSDP
            "include_video": True,  # Same as working training config
            "single_image_max_num_patches": 1960,
            "single_frame_max_num_patches": 196,
        },
    }
    return CosmosConfig.from_dict(full_config)


def run_sft_direct(
    model_path: str,
    tp_size: int = 1,
    num_train_steps: int = 4,
    data_packer=None,
    dataset=None,
    trainer_type: str = "sft",
):
    """Run SFT training directly without controller - tests dataset, data_packer, trainer."""

    # 1. Apply nemotron monkey patches (same as launcher.py)
    import cosmos_rl

    cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(
        patched_parallelize_fn
    )
    cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
    cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
    cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor

    # 2. Mock CommMixin.init_comm to skip controller registration (like launch_test_worker)
    from cosmos_rl.comm.base import CommMixin
    from cosmos_rl.dispatcher.api.client import APIClient

    def dummy_init_comm(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        # Use fake endpoint - we never register, so no connection is made
        self.api_client = APIClient(self.role, ["0.0.0.0"], 8000)
        hf_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path, trust_remote_code=True
        )
        logger.info(f"[Test] model type {hf_config.model_type}")
        # data_packer will be set by init_data_packer in build_runner

    CommMixin.init_comm = dummy_init_comm

    # 3. Build config and parallel_dims
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dp_replicate_size = max(1, world_size // tp_size)
    config = build_toy_config(
        model_path=model_path,
        tp_size=tp_size,
        dp_shard_size=1,
        dp_replicate_size=dp_replicate_size,
        num_train_steps=num_train_steps,
        trainer_type=trainer_type,
    )

    init_distributed()
    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    parallel_dims.build_mesh(device_type="cuda")

    # 4. Dataset: 传入 (config)->Dataset 的 factory，或 None 则从 config.dataset.name 加载 HF/磁盘数据
    # 5. Create SFTPolicyWorker
    # 换 dataset: 传 get_my_dataset 等，需实现 setup(config) 和 __getitem__-> messages (list[dict])
    # 换 data_packer: 传入 HFVLMDataPacker() 等，需兼容 dataset 返回格式
    dp = data_packer if data_packer is not None else HFVLMDataPacker()
    ds = dataset if dataset is not None else get_launcher_dataset
    sft_worker = SFTPolicyWorker(
        config=config,
        parallel_dims=parallel_dims,
        dataset=ds,
        data_packer=dp,
        val_data_packer=dp,
    )

    # 6. Run training loop for a few steps
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
        logger.info(f"[Test] SFT direct run completed. Losses: {losses}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct SFT test for Nemotron VLM (no controller)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/nemotron-vlm/NVIDIA-Nemotron-3-Nano-SIGLIP2-officaial-30B-A3B-BF16",
        help="Path to Nemotron model",
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
    parser.add_argument(
        "--trainer_type",
        type=str,
        default="sft",
        help="Trainer type: sft, dpo, pi_sft, or custom @TrainerRegistry.register type",
    )
    args = parser.parse_args()

    # 换 data_packer / dataset / trainer_type: 修改此处
    data_packer = HFVLMDataPacker()
    dataset = get_launcher_dataset
    run_sft_direct(
        model_path=args.model_path,
        tp_size=args.tp_size,
        num_train_steps=args.num_steps,
        data_packer=data_packer,
        dataset=dataset,
        trainer_type=args.trainer_type,
    )


if __name__ == "__main__":
    main()
