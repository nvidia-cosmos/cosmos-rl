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

from cosmos_rl.policy.model.qwen2_5_vl import Qwen2_5_VLConditionalModel
from transformers import Qwen2_5_VLForConditionalGeneration
from cosmos_rl.policy.config import ParallelismConfig
from cosmos_rl.utils.parallelism import ParallelDims
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import torch
import unittest


class TestCosmosHfPrecision(unittest.TestCase):
    def test_cosmos_hf_precision(self):
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        device = torch.device("cuda")
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)

        cosmos_model = (
            Qwen2_5_VLConditionalModel.from_pretrained(hf_model.config, model_name)
            .to(device)
            .to(torch.bfloat16)
        )

        cosmos_model.load_hf_weights(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            parallel_dims=ParallelDims.from_config(
                ParallelismConfig(
                    tp_size=1,
                    pp_size=1,
                    dp_replicate_size=1,
                    dp_shard_size=1,
                )
            ),
            device=device,
        )

        cosmos_model.model.rotary_emb.inv_freq = (
            cosmos_model.model.rotary_emb.inv_freq.to(torch.float32)
        )
        cosmos_model.model.rotary_emb.reset_inv_freq()
        cosmos_model.visual.rotary_pos_emb.inv_freq = (
            cosmos_model.visual.rotary_pos_emb.inv_freq.to(torch.float32)
        )
        cosmos_model.visual.rotary_pos_emb.reset_inv_freq()

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        "max_pixels": 89100 * 8,
                    },
                    {
                        "type": "text",
                        "text": "This is a demo image, please look carefully and describe what is happening?",
                    },
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        image_grid_thw = inputs.image_grid_thw

        cosmos_model.post_to_empty_hook(None)

        with torch.no_grad():
            cosmos_result = cosmos_model.visual.forward(
                inputs.pixel_values.unsqueeze(0).to(torch.bfloat16).to(device),
                grid_thw=image_grid_thw.to(device),
            )
            hf_result = hf_model.visual(
                inputs.pixel_values.to(torch.bfloat16).to(device),
                inputs.image_grid_thw.to(device),
            )

            self.assertTrue(torch.sum((cosmos_result - hf_result).abs()) < 1e-6)


if __name__ == "__main__":
    unittest.main()
