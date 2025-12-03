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

# Set the environment variable to use HF rotary implementation
os.environ["COSMOS_USE_HF_IMPL"] = "1"
import torch
import unittest
import numpy as np
from functools import partial

from cosmos_rl.policy.model import ModelRegistry
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import async_safe_ce
from cosmos_rl.dispatcher.data.packer.qwen3_vl_data_packer import (
    Qwen3_VL_DataPacker,
)
from transformers import (
    AutoProcessor,
    Qwen3VLMoeForConditionalGeneration,
)


IGNORE_INDEX = -100


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


def create_assistant_tokens_mask(tokens, processor):  # Qwen2 based model
    if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
        mask = torch.stack(
            [
                create_assistant_tokens_mask(tokens[i], processor)
                for i in range(tokens.shape[0])
            ]
        )
        assert mask.shape == tokens.shape
        return mask
    np_tokens = (
        tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)
    )
    assert np_tokens.ndim == 1
    # Constants defining bos, eos and fixed offsets.
    BOS_TOKEN = "<|im_start|>"
    EOS_TOKEN = "<|im_end|>"
    ROLE = "assistant"
    # Offsets: skip the bos + "assistant\n" (always 3 tokens) and include the eos (+1) for supervision
    START_OFFSET = 3
    END_OFFSET = 1

    # Retrieve token IDs for the markers and the role.
    bos_token_id = processor.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    eos_token_id = processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    role_id = processor.tokenizer.convert_tokens_to_ids(ROLE)

    # Locate all positions where the start and end markers appear.
    start_indices = np.where(np_tokens == bos_token_id)[0]
    end_indices = np.where(np_tokens == eos_token_id)[0]

    # Initialize the mask with False values.
    masks = np.zeros_like(np_tokens, dtype=bool)
    assert len(start_indices) == len(
        end_indices
    ), "remember to set add_generation_prompt=False in apply_chat_template"
    # For each pair of bos/eos, check if the role is 'assistant'
    # and apply the mask accordingly.
    for start, end in zip(start_indices, end_indices):
        if np_tokens[start + 1] == role_id:
            print(
                f"start: {start}, end: {end}, role_id: {role_id}, np_tokens[start + 1]: {np_tokens[start + 1]}"
            )
            # Mask tokens from after the assistant header (start+3) to include the end marker (end+1)
            masks[start + START_OFFSET : end + END_OFFSET] = True
    assert masks.shape == np_tokens.shape
    print(f"masks: {masks}")
    if isinstance(tokens, torch.Tensor):
        return torch.from_numpy(masks)
    else:
        return masks.tolist()


def create_debug_input(model_name_or_path):
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "min_pixels": 1024 * 28 * 28,
                    "max_pixels": 1024 * 28 * 28,
                },
                {"type": "text", "text": "Describe the image caption."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The image captures a serene and joyful moment on a sandy beach at sunset. The sky is a soft gradient of warm colors, transitioning from a pale blue at the horizon to a deeper, golden hue near the horizon line. The ocean waves gently roll onto the shore, creating a tranquil and picturesque backdrop.\nIn the foreground, a person is sitting on the sand, facing a light-colored dog, likely a Labrador Retriever. The person is dressed in a plaid shirt and dark pants, with their legs crossed and one hand resting on their knee. The dog is wearing a harness with a colorful collar and is sitting on its haunches, facing the person. The dog's tail is wagging, and it appears to be engaged in a playful interaction with the person, possibly giving a high-five or pawing at their hand.",
                },
            ],
        },
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    for k, v in inputs.items():
        print(f"k: {k}, type: {type(v)}")
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda")

    labels = inputs["input_ids"].clone()
    token_mask = create_assistant_tokens_mask(inputs["input_ids"], processor)
    labels[~token_mask] = IGNORE_INDEX
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "labels": labels,
    }


class TestCosmosHfPrecision(unittest.TestCase):
    def test_cosmos_hf_precision(self):
        model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        # ================================
        # create config
        # ================================
        config_dict = {
            "policy": {
                "model_name_or_path": model_name,
                "model_max_length": 40960,
                "parallelism": {
                    "tp_size": 4,
                    "cp_size": 1,
                    "ep_size": 1,
                    "dp_shard_size": -1,
                    "dp_replicate_size": 1,
                    "pp_size": 1,
                    "world_size": int(os.environ.get("WORLD_SIZE", 1)),
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
                "master_dtype": "float32",
                "param_dtype": "bfloat16",
                "fsdp_reduce_dtype": "float32",
                "fsdp_reshard_after_forward": "default",
                "param_torch_dtype": torch.bfloat16,
                "train_policy": {"mini_batch": 1},
                "fp8": {"enable_fp8": False},
                "async_tp_enabled": False,
                "force_use_hf": False,
                # "sequence_packing": True,
            },
            "rollout": {
                "multi_turn_config": {
                    "enable": False,
                    "custom_chat_template_path": None,
                },
            },
        }
        config = Config(config_dict)

        # ================================
        # create data packer
        # ================================
        data_packer = Qwen3_VL_DataPacker()
        data_packer.setup(config=config)

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="cuda:nccl,cpu:gloo")
                torch.cuda.set_device(torch.distributed.get_rank())

        # ================================
        # create data batch
        # ================================
        data_batch = create_debug_input(model_name_or_path=model_name)
        # fill in position_ids
        position_ids, _ = data_packer._get_rope_index(
            input_ids=data_batch["input_ids"],
            image_grid_thw=data_batch.get("image_grid_thw", None),
            video_grid_thw=data_batch.get("video_grid_thw", None),
            attention_mask=data_batch.get("attention_mask", None),
        )
        if position_ids is not None:
            data_batch["position_ids"] = position_ids.permute(1, 0, 2).contiguous()
            print(f"position_ids: {data_batch['position_ids'].shape}")

        for k, v in data_batch.items():
            data_batch[k] = v.to("cuda")

        # compute logits for hf model
        hf_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        hf_model.eval()
        with torch.no_grad():
            logits_hf = hf_model(
                data_batch["input_ids"],
                pixel_values=data_batch["pixel_values"],
                attention_mask=data_batch["attention_mask"],
                image_grid_thw=data_batch["image_grid_thw"],
                use_cache=False,
                return_dict=True,
            ).logits
            print(f"before logits_hf: {logits_hf.shape}")
        del hf_model
        torch.cuda.empty_cache()

        ce_loss_hf = torch.nn.functional.cross_entropy(
            input=logits_hf[:, :-1].flatten(0, 1),
            target=data_batch["labels"][:, 1:].flatten(0, 1),
            ignore_index=-100,
            reduction="none",
        )
        ce_loss_hf_mean = ce_loss_hf.mean()
        logits_hf = logits_hf[:, -1, :]

        cosmos_model_list, _, _, _ = init_cosmos_rl_model(config, device="cuda")
        cosmos_model = cosmos_model_list[0]
        cosmos_model.eval()
        with torch.no_grad():
            logits_cosmos_rl = cosmos_model(
                data_batch["input_ids"],
                pixel_values=data_batch["pixel_values"],
                attention_mask=data_batch["attention_mask"],
                image_grid_thw=data_batch["image_grid_thw"],
                position_ids=data_batch.get("position_ids", None),
            )
        del cosmos_model
        torch.cuda.empty_cache()

        ce_loss = torch.nn.functional.cross_entropy(
            input=logits_cosmos_rl[:, :-1].flatten(0, 1),
            target=data_batch["labels"][:, 1:].flatten(0, 1),
            ignore_index=-100,
            reduction="none",
        )
        cs_loss_mean = ce_loss.mean()
        logits_cosmos_rl = logits_cosmos_rl[:, -1, :]

        if torch.distributed.get_rank() == 0:
            max_index_hf = logits_hf.argmax(dim=-1)
            max_index_cosmos_rl = logits_cosmos_rl.argmax(dim=-1)
            diff_loss_mean = (ce_loss_hf_mean - cs_loss_mean).abs()
            diff_logits = (logits_cosmos_rl - logits_hf).abs()
            print(
                f"ce_loss_hf_mean: {ce_loss_hf_mean} | cs_loss_mean: {cs_loss_mean} | diff_loss_mean: {diff_loss_mean}"
            )
            print(
                f"logits_hf: {logits_hf} | logits_cosmos_rl: {logits_cosmos_rl} | diff_logits: {diff_logits}"
            )
            print(
                f"max_index_hf: {max_index_hf} | max_index_cosmos_rl: {max_index_cosmos_rl}"
            )
            assert diff_loss_mean < 1e-3
            assert max_index_hf == max_index_cosmos_rl


# torchrun --nproc_per_node=4 test_qwen3_vl_moe.py
if __name__ == "__main__":
    unittest.main()
