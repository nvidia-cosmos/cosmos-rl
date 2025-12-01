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
import copy
from datetime import timedelta

# Set the environment variable to use HF rotary implementation
os.environ["COSMOS_USE_HF_IMPL"] = "1"
import torch
import unittest
from PIL import Image
from functools import partial
from contextlib import contextmanager
from qwen_vl_utils import process_vision_info

from cosmos_rl.policy.model import ModelRegistry
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import async_safe_ce
from transformers import (
    AutoConfig,
    AutoProcessor,
)


IGNORE_INDEX = -100


def test_hf_model_forward(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
        return logits[:, -1, :]


def test_cosmos_hf_model(model, inputs):
    with torch.no_grad():
        logits = model(**inputs)
        return logits[:, -1, :]


@contextmanager
def cosmos_default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def pp_loss_fn(config, parallel_dims):
    loss_scaling_factor = 1.0
    if parallel_dims.dp_shard_enabled:
        dp_group = parallel_dims.mesh["dp_shard"].get_group()
    else:
        dp_group = None

    if parallel_dims.cp_enabled:
        cp_group = parallel_dims.mesh["cp"].get_group()
    else:
        cp_group = None

    return partial(
        async_safe_ce,
        loss_scaling_factor=loss_scaling_factor,
        dp_group=dp_group,
        cp_group=cp_group,
        ignore_index=IGNORE_INDEX,
    )


def init_cosmos_rl_model(config, is_train=True, device="cuda"):
    if config.policy.parallelism.dp_replicate_size == -1:
        # recompute it
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        config.policy.parallelism.dp_replicate_size = (
            world_size // config.policy.parallelism.dp_shard_size
        )
        print(
            f"Recomputed dp_replicate_size: {config.policy.parallelism.dp_replicate_size} given world_size: {world_size} and dp_shard_size: {config.policy.parallelism.dp_shard_size}"
        )

    model = ModelRegistry.build_model(config)

    # if config.policy.model_gradient_checkpointing:
    #     apply_ac(model, config.policy.model_name_or_path)
    # init parallel_dims
    parallel_dims: ParallelDims = ParallelDims.from_config(
        parallesim_config=config.policy.parallelism
    )
    parallel_dims.build_mesh(device_type=device)

    print(f"parallel_dims: {parallel_dims}")

    parallelize_fn, _ = model.parallelize_fn
    if is_train:
        loss_fn = pp_loss_fn(config, parallel_dims)
    else:
        loss_fn = None
    pp_scheduler, pp_scheduler_val = parallelize_fn(
        model, parallel_dims, config, pp_loss_fn=loss_fn
    )
    assert pp_scheduler is None, "pp_scheduler should be None"
    assert pp_scheduler_val is None, "pp_scheduler_val should be None"
    if not config.train.fsdp_offload:
        model._apply(
            lambda t: torch.empty_like(t, device="cuda")
            if t.device.type == "meta"
            else t.to("cuda"),
            recurse=True,
        )
    model.post_to_empty_hook(config)

    torch.cuda.empty_cache()
    model.load_hf_weights(
        config.policy.model_name_or_path,
        parallel_dims,
        device,
    )
    return [model], pp_scheduler, parallel_dims, loss_fn


class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


# ================================
# create config
# ================================
config_dict = {
    "policy": {
        "model_name_or_path": None,
        "model_max_length": 1024,
        "parallelism": {
            "tp_size": 2,
            "cp_size": 1,
            "ep_size": 1,
            "dp_shard_size": 1,
            "dp_replicate_size": 1,
            "pp_size": 1,
            "world_size": 2,
            "pp_dynamic_shape": False,
        },
        "lora": None,
        "enable_liger_kernel": False,
        "trainable_map": None,
        "model_gradient_checkpointing": True,
    },
    "train": {
        "fsdp_offload": False,
        "compile": False,
        "master_dtype": "bfloat16",
        "param_dtype": "bfloat16",
        "fsdp_reduce_dtype": "float32",
        "fsdp_reshard_after_forward": "default",
        "param_torch_dtype": torch.bfloat16,
        "train_policy": {"mini_batch": 1},
        "fp8": {"enable_fp8": False},
        "async_tp_enabled": False,
        # "sequence_packing": True,
    },
    "rollout": {
        "multi_turn_config": {
            "enable": False,
            "custom_chat_template_path": None,
        },
    },
}


class TestHFModelTP(unittest.TestCase):
    def test_tp_post_to_empty_hook(self):
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="cuda:nccl,cpu:gloo",
                    timeout=timedelta(seconds=300),  # 5 minutes timeout
                )
                torch.cuda.set_device(torch.distributed.get_rank())

        max_position_embeddings = 1024
        config_dict["policy"]["model_max_length"] = max_position_embeddings

        # Load cosmos config
        cosmos_config = Config(config_dict)

        for model_id in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "microsoft/phi-4",
        ]:
            # Load hf config
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config.max_position_embeddings = max_position_embeddings
            cosmos_config.policy.model_name_or_path = model_id
            # Remove the model type from the model registry, so that the model will run in the hfmodel path.
            if ModelRegistry.check_model_type_supported(config.model_type):
                ModelRegistry._MODEL_REGISTRY.pop(config.model_type)

            for dtype in [torch.bfloat16, torch.float32]:
                # Avoid oom for microsoft/phi-4 with float32.
                if dtype == torch.float32 and model_id == "microsoft/phi-4":
                    continue
                cosmos_config.train.master_dtype = {
                    torch.bfloat16: "bfloat16",
                    torch.float32: "float32",
                }[dtype]
                cosmos_config.train.param_dtype = {
                    torch.bfloat16: "bfloat16",
                    torch.float32: "float32",
                }[dtype]
                config.torch_dtype = dtype

                cosmos_model_list, _, _, _ = init_cosmos_rl_model(
                    cosmos_config, device="cuda"
                )
                cosmos_hf_model = cosmos_model_list[0]
                cosmos_named_buffers = {
                    k: v.clone() for k, v in cosmos_hf_model.model.named_buffers()
                }
                model_class = cosmos_hf_model.model_class
                del cosmos_hf_model
                torch.cuda.empty_cache()

                # Load hf model
                hf_model = model_class.from_pretrained(
                    model_id, trust_remote_code=True, config=config
                ).to("cuda", dtype=dtype)
                hf_named_buffers = {k: v.clone() for k, v in hf_model.named_buffers()}
                del hf_model
                torch.cuda.empty_cache()
                if torch.distributed.get_rank() == 0:
                    for name, cosmos_hf_buffer in cosmos_named_buffers.items():
                        assert (
                            name in hf_named_buffers
                        ), f"Buffer {name} not found in hf model"
                        hf_buffer = hf_named_buffers[name]
                        assert (
                            cosmos_hf_buffer.shape == hf_buffer.shape
                        ), f"Shape mismatch: {cosmos_hf_buffer.shape} != {hf_buffer.shape} for {name}"
                        assert (
                            cosmos_hf_buffer.dtype == hf_buffer.dtype
                        ), f"Dtype mismatch: {cosmos_hf_buffer.dtype} != {hf_buffer.dtype} for {name}"
                        assert torch.equal(
                            cosmos_hf_buffer, hf_buffer
                        ), f"Buffer {name} is not equal to the one in hf model"

                    print(f"{model_id} with {dtype=} post_to_empty_hook test passed.")
                del hf_named_buffers
                del cosmos_named_buffers
                torch.cuda.empty_cache()

    def test_tp_forward(self):
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="cuda:nccl,cpu:gloo",
                    timeout=timedelta(seconds=300),  # 5 minutes timeout
                )
                torch.cuda.set_device(torch.distributed.get_rank())

        max_position_embeddings = 4096
        config_dict["policy"]["model_max_length"] = max_position_embeddings

        # Load cosmos config
        cosmos_config = Config(config_dict)

        for model_id in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "microsoft/phi-4",
        ]:
            # Load hf config
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config.max_position_embeddings = max_position_embeddings

            cosmos_config.policy.model_name_or_path = model_id
            # Remove the model type from the model registry, so that the model will run in the hfmodel path.
            if ModelRegistry.check_model_type_supported(config.model_type):
                ModelRegistry._MODEL_REGISTRY.pop(config.model_type)

            dtype = torch.bfloat16
            cosmos_config.train.master_dtype = "bfloat16"
            cosmos_config.train.param_dtype = "bfloat16"
            config.torch_dtype = dtype

            # Load cosmos model
            cosmos_model_list, _, _, _ = init_cosmos_rl_model(
                cosmos_config, device="cuda"
            )
            cosmos_hf_model = cosmos_model_list[0]

            # Load hf model
            hf_model = cosmos_hf_model.model_class.from_pretrained(
                model_id, trust_remote_code=True, config=config
            ).to("cuda", dtype=dtype)

            # Load processor
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )
            error_occurred = False
            try:
                if cosmos_hf_model.is_vlm:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    image = Image.open(
                        os.path.join(current_dir, "data", "test_hf_model.jpg")
                    )
                    messages = [
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image,
                                    },
                                    {"type": "text", "text": "describe the image"},
                                ],
                            }
                        ]
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False)
                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                        text=text,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a pirate chatbot who always responds in pirate speak!",
                        },
                        {"role": "user", "content": "Who are you?"},
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False)
                    inputs = processor(
                        text=text,
                        return_tensors="pt",
                    ).to("cuda")

                hf_forward_logits = test_hf_model_forward(
                    hf_model, copy.deepcopy(inputs)
                )
                cosmos_forward_logits = test_cosmos_hf_model(
                    cosmos_hf_model, copy.deepcopy(inputs)
                )

                if torch.distributed.get_rank() == 0:
                    max_index_hf = hf_forward_logits.argmax(dim=-1)
                    max_index_cosmos_rl = cosmos_forward_logits.argmax(dim=-1)
                    max_logit_hf = hf_forward_logits.max(dim=-1).values
                    max_logit_cosmos_rl = cosmos_forward_logits.max(dim=-1).values
                    print(
                        f"max_index_hf: {max_index_hf} | max_index_cosmos_rl: {max_index_cosmos_rl} | max_logit_hf: {max_logit_hf} | max_logit_cosmos_rl: {max_logit_cosmos_rl}"
                    )
                    assert max_index_hf == max_index_cosmos_rl
                    assert (max_logit_hf - max_logit_cosmos_rl).abs() < 0.5
                    print(f"{model_id} forward test passed.")

                del cosmos_hf_model
                del hf_model
                del cosmos_forward_logits
                del hf_forward_logits
                torch.cuda.empty_cache()
            except Exception as e:
                error_occurred = True
                print(
                    f"Rank {torch.distributed.get_rank()} - {model_id} forward test failed."
                )
                print(f"Rank {torch.distributed.get_rank()} - Error: {e}")

            # Synchronize error state across all ranks to avoid hanging
            error_tensor = torch.tensor([1.0 if error_occurred else 0.0], device="cuda")
            torch.distributed.all_reduce(
                error_tensor, op=torch.distributed.ReduceOp.MAX
            )

            if error_tensor.item() > 0:
                if torch.distributed.get_rank() == 0:
                    print("Test failed on at least one rank. Exiting...")
                exit(-1)


# torchrun --nproc_per_node=2 test_hf_models_tp.py
if __name__ == "__main__":
    unittest.main()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
