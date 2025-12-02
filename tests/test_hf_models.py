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
import torch
import unittest
from contextlib import contextmanager
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.model.hf_models import HFModel
from cosmos_rl.policy.config import Config as CosmosConfig, ParallelismConfig
from accelerate import init_on_device


@contextmanager
def cosmos_default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def test_hf_model_generate(model, inputs):
    generation_kwargs = dict(
        max_new_tokens=1,
        output_logits=True,
        return_dict_in_generate=True,
    )
    logits = model.generate(**inputs, **generation_kwargs).logits[0]
    return logits


def test_hf_model_forward(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
        return logits[:, -1, :]


def test_cosmos_hf_model(model, inputs):
    with torch.no_grad():
        logits = model(**inputs)
        return logits[:, -1, :]


class TestHFModel(unittest.TestCase):
    def test_post_to_empty_hook(self):
        for model_id in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "llava-hf/llava-1.5-7b-hf",
            "Qwen/Qwen3-VL-4B-Instruct",
            # "google/gemma-3-12b-it",              # Need access to the repo
            # "mistralai/Mistral-7B-Instruct-v0.3", # Need access to the repo
            "microsoft/phi-4",
            # "nvidia/NVIDIA-Nemotron-Nano-9B-v2",          # Need to install causal_conv1d, mamba_ssm
            # "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", # Need to install causal_conv1d, mamba_ssm, timm
        ]:
            for dtype in [torch.bfloat16, torch.float32]:
                # To avoid out-of-memory issues, bypass float32 precision for models which have more than 10B parameters
                if (
                    model_id
                    in [
                        "google/gemma-3-12b-it",
                        "microsoft/phi-4",
                        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
                    ]
                    and dtype == torch.float32
                ):
                    continue
                max_position_embeddings = 1024
                # Load config
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                config.max_position_embeddings = max_position_embeddings
                config.torch_dtype = dtype
                # Load cosmos hf model
                cosmos_hf_model = None
                with init_on_device("meta", include_buffers=False):
                    with cosmos_default_dtype(dtype):
                        cosmos_hf_model = HFModel.from_pretrained(
                            config,
                            model_id,
                            max_position_embeddings=max_position_embeddings,
                        )
                cosmos_hf_model._apply(
                    lambda t: torch.empty_like(t, device="cuda")
                    if t.device.type == "meta"
                    else t.to("cuda"),
                    recurse=True,
                )
                cosmos_hf_model.post_to_empty_hook(CosmosConfig())
                parallel_dims = ParallelDims.from_config(ParallelismConfig(tp_size=1))
                cosmos_hf_model.load_hf_weights(
                    model_id, parallel_dims, "cuda", revision=None
                )
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
                hf_named_buffers = {k: v for k, v in hf_model.named_buffers()}
                del hf_model
                torch.cuda.empty_cache()

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

    def test_forward(self):
        for model_id in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "llava-hf/llava-1.5-7b-hf",
            "Qwen/Qwen3-VL-4B-Instruct",
            # "google/gemma-3-12b-it",              # Need access to the repo
            # "mistralai/Mistral-7B-Instruct-v0.3", # Need access to the repo
            "microsoft/phi-4",
            # "nvidia/NVIDIA-Nemotron-Nano-9B-v2",          # Need to install causal_conv1d, mamba_ssm
            # "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", # Need to install causal_conv1d, mamba_ssm, timm
        ]:
            dtype = torch.bfloat16
            max_position_embeddings = 4096
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config.max_position_embeddings = max_position_embeddings
            config._attn_implementation = "eager"
            config.torch_dtype = dtype
            cosmos_hf_model = None
            with init_on_device("meta", include_buffers=False):
                with cosmos_default_dtype(dtype):
                    cosmos_hf_model = HFModel.from_pretrained(
                        config,
                        model_id,
                        max_position_embeddings=max_position_embeddings,
                    )
            cosmos_hf_model._apply(
                lambda t: torch.empty_like(t, device="cuda")
                if t.device.type == "meta"
                else t.to("cuda"),
                recurse=True,
            )
            cosmos_hf_model.post_to_empty_hook(CosmosConfig())
            parallel_dims = ParallelDims.from_config(ParallelismConfig(tp_size=1))
            cosmos_hf_model.load_hf_weights(
                model_id, parallel_dims, "cuda", revision=None
            )
            cosmos_hf_model.eval()

            hf_model = (
                cosmos_hf_model.model_class.from_pretrained(
                    model_id, trust_remote_code=True, config=config
                )
                .to("cuda", dtype=dtype)
                .eval()
            )

            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )
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
                    kwargs = {
                        "padding": True,
                        "return_tensors": "pt",
                    }
                    if model_id == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16":
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        text = tokenizer.apply_chat_template(messages, tokenize=False)
                        image_inputs = [image]
                        video_inputs = None
                        # Nemotron-Nano do not have pad_token_id, so we need to set padding to False
                        kwargs["padding"] = False
                    else:
                        text = processor.apply_chat_template(messages, tokenize=False)
                        image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                        text=text, images=image_inputs, videos=video_inputs, **kwargs
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

                if model_id == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16":
                    from cosmos_rl.policy.model.hf_models.patch import (
                        post_hf_models_patch,
                    )

                    post_hf_models_patch(config, hf_model)
                    # num_patches is not needed for forward/generate
                    inputs.pop("num_patches")

                hf_generate_logits = test_hf_model_generate(
                    hf_model, copy.deepcopy(inputs)
                )
                hf_forward_logits = test_hf_model_forward(
                    hf_model, copy.deepcopy(inputs)
                )
                cosmos_hf_logits = test_cosmos_hf_model(
                    cosmos_hf_model, copy.deepcopy(inputs)
                )
                assert torch.equal(
                    hf_generate_logits, hf_forward_logits
                ), f"{hf_generate_logits} != {hf_forward_logits}"
                assert torch.equal(
                    hf_generate_logits, cosmos_hf_logits
                ), f"{hf_generate_logits} != {cosmos_hf_logits}"

                print(f"{model_id} forward test passed.")
                del cosmos_hf_model
                del hf_model
                del cosmos_hf_logits
                del hf_generate_logits
                del hf_forward_logits
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{model_id} forward test failed.")
                print(f"{e}")
                exit(-1)


if __name__ == "__main__":
    unittest.main()
