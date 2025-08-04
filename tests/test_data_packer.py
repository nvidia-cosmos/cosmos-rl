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
from unittest.mock import Mock

from cosmos_rl.dispatcher.data.packer.deepseek_data_packer import DeepSeek_DataPacker
from cosmos_rl.policy.config import Config, PolicyConfig, TrainingConfig


class TestDataPacker(unittest.TestCase):
    def test_deep_seek_data_packer(self):
        MAX_LEN = 5
        config = Config(
            policy=PolicyConfig(model_max_length=MAX_LEN), train=TrainingConfig()
        )
        tokenizer = Mock()
        data_packer = DeepSeek_DataPacker()
        data_packer.setup(config, tokenizer)

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
        output = data_packer.sft_collate_fn(TEST_SAMPLES, 2, -100, -100)
        assert output["input_ids"].shape == (len(TEST_SAMPLES), MAX_LEN)
        assert output["label_ids"].shape == (len(TEST_SAMPLES), MAX_LEN)


if __name__ == "__main__":
    unittest.main()
