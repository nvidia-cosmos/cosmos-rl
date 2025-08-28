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


import torch
import unittest
from transformers import AutoConfig
from contextlib import contextmanager

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.model.hf_models import HFModel
from cosmos_rl.policy.config import Config as CosmosConfig, ParallelismConfig


@contextmanager
def cosmos_default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


class TestHFModel(unittest.TestCase):
    def test_post_to_empty_hook(self):
        for model_id in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "google/gemma-3-12b-it",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/phi-4",
        ]:
            for dtype in [torch.bfloat16, torch.float32]:
                max_position_embeddings = 1024
                # Load config
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                config.max_position_embeddings = max_position_embeddings
                # Load cosmos hf model
                cosmos_hf_model = None
                with torch.device("meta"):
                    with cosmos_default_dtype(dtype):
                        cosmos_hf_model = HFModel.from_pretrained(
                            config,
                            model_id,
                            max_position_embeddings=max_position_embeddings,
                        )
                cosmos_hf_model.to_empty(device="cuda")
                cosmos_hf_model.post_to_empty_hook(CosmosConfig())
                parallel_dims = ParallelDims.from_config(ParallelismConfig(tp_size=1))
                cosmos_hf_model.load_hf_weights(
                    model_id, parallel_dims, "cuda", revision=None
                )

                # Load hf model
                hf_model = cosmos_hf_model.model_class.from_pretrained(
                    model_id, trust_remote_code=True, torch_dtype=dtype, config=config
                ).to("cuda")
                hf_named_buffers = {k: v for k, v in hf_model.named_buffers()}

                for name, cosmos_hf_buffer in cosmos_hf_model.model.named_buffers():
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

                del cosmos_hf_model
                del hf_model
                del hf_named_buffers

    def test_forward(self):
        pass


if __name__ == "__main__":
    unittest.main()
