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


import unittest
from packaging import version
import transformers
from transformers import AutoConfig

import cosmos_rl.utils.util as util
from cosmos_rl.policy.model.hf_models import HFModel
from cosmos_rl.policy.model.qwen2_5_vl import Qwen2_5_VLConditionalModel
from cosmos_rl.policy.model.qwen3_vl_moe import Qwen3VLMoeModel
from cosmos_rl.dispatcher.data.packer import BaseDataPacker, DecoderOnlyLLMDataPacker
from cosmos_rl.dispatcher.data.packer.deepseek_data_packer import DeepSeek_DataPacker
from cosmos_rl.dispatcher.data.packer.qwen2_5_vlm_data_packer import (
    Qwen2_5_VLM_DataPacker,
)
from cosmos_rl.dispatcher.data.packer.qwen3_vl_data_packer import Qwen3_VL_DataPacker
from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker
from cosmos_rl.policy.config import Config, PolicyConfig, TrainingConfig


class TestDataPacker(unittest.TestCase):
    def test_deep_seek_data_packer(self):
        MAX_LEN = 5
        config = Config(
            policy=PolicyConfig(model_max_length=MAX_LEN), train=TrainingConfig()
        )
        data_packer = DeepSeek_DataPacker()
        data_packer.setup(config)

        TEST_SAMPLES = [
            {
                "token_ids": [10, 9, 8],
                "label_ids": [7, 6, 5],
            },
            {
                "token_ids": [10, 9, 8, 7, 6, 5, 4],
                "label_ids": [3, 2, 1, 0, 11, 12],
            },
        ]
        output = data_packer.sft_collate_fn(TEST_SAMPLES, 2, -100)
        assert output["input_ids"].shape == (len(TEST_SAMPLES), MAX_LEN)
        assert output["label_ids"].shape == (len(TEST_SAMPLES), MAX_LEN)

    def test_all_data_packers(self):
        for model_id in [
            "microsoft/phi-4",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        ]:
            if (
                version.parse(transformers.__version__) < version.parse("4.57.0")
                and "qwen3-vl" in model_id.lower()
            ):
                continue
            hf_config = util.retry(AutoConfig.from_pretrained)(
                model_id, trust_remote_code=True
            )
            is_vlm = getattr(hf_config, "vision_config", None) is not None

            try:
                data_packer = BaseDataPacker.get_default_data_packer(
                    hf_config.model_type
                )
            except ValueError:
                data_packer = (
                    DecoderOnlyLLMDataPacker() if not is_vlm else HFVLMDataPacker()
                )

            config = Config(
                policy=PolicyConfig(model_max_length=4096, model_name_or_path=model_id),
                train=TrainingConfig(),
            )
            data_packer.setup(config)

            if model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
                assert isinstance(data_packer, Qwen2_5_VLM_DataPacker)
                assert (
                    hf_config.model_type
                    in Qwen2_5_VLConditionalModel.supported_model_types()
                )
            elif model_id == "Qwen/Qwen3-VL-8B-Instruct":
                assert isinstance(data_packer, HFVLMDataPacker)
                # HFModel use hfmodel as the default model type, so qwen3-vl should not be in the supported_model_types
                assert hf_config.model_type not in HFModel.supported_model_types()
            elif model_id == "Qwen/Qwen3-VL-30B-A3B-Instruct":
                # Qwen3-VL-Moe uses a custom implementation and relies on Qwen3_VL_DataPacker as its specific data packer
                assert isinstance(data_packer, Qwen3_VL_DataPacker)
                assert hf_config.model_type in Qwen3VLMoeModel.supported_model_types()
            elif model_id == "microsoft/phi-4":
                assert isinstance(data_packer, DecoderOnlyLLMDataPacker)

            if model_id in [
                "Qwen/Qwen3-VL-8B-Instruct",
                "Qwen/Qwen3-VL-30B-A3B-Instruct",
            ]:
                print(f"model_id: {model_id}")
                MAX_PIXELS = 128 * 1024
                FPS1 = 2
                FPS2 = 1
                content1 = [
                    {
                        "type": "video",
                        "video": "tests/data/test_data_packer.mp4",
                        "max_pixels": MAX_PIXELS,
                        "fps": FPS1,
                    },
                    {
                        "type": "text",
                        "text": "describe this video.",
                    },
                ]
                content2 = [
                    {
                        "type": "video",
                        "video": "tests/data/test_data_packer.mp4",
                        "max_pixels": MAX_PIXELS,
                        "fps": FPS2,
                    },
                    {
                        "type": "text",
                        "text": "please describe this video.",
                    },
                ]
                sample1 = [{"role": "user", "content": content1}]
                sample2 = [{"role": "user", "content": content2}]
                return_dict1 = data_packer.sft_process_sample(sample1)
                return_dict2 = data_packer.sft_process_sample(sample2)
                pixel_value_video_shape0 = return_dict1["pixel_values_videos"].shape[0]
                pixel_value_video_shape1 = return_dict2["pixel_values_videos"].shape[0]
                computed_max_len = max(
                    len(return_dict1["input_ids"]), len(return_dict2["input_ids"])
                )
                output = data_packer.sft_collate_fn(
                    [return_dict1, return_dict2], computed_max_len, -100
                )
                assert output["input_ids"].shape == (2, computed_max_len)
                assert output["label_ids"].shape == (2, computed_max_len)
                assert (
                    pixel_value_video_shape0 == FPS1 // FPS2 * pixel_value_video_shape1
                )


if __name__ == "__main__":
    unittest.main()
