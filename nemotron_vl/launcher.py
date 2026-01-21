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

import argparse
import toml
import json

import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nemotron_parallelize import parallelize
from weight_converter import convert_weight_from_hf

from torch.utils.data import Dataset

import cosmos_rl.utils.util as util
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher
from cosmos_rl.policy.config import Config
import cosmos_rl.policy.model.hf_models as hf_models 
from cosmos_rl.policy.config import Config as CosmosConfig

def modify_messages(messages, max_pixels = None):
    for message in messages:
        if isinstance(message['content'], str):
            message['content'] = [{'type': 'text', 'text': message['content']}]
        for content in message['content']:
            if content['type'] in ['image', 'video']:
                content[content['type']] = content[content['type']].replace('workspace', 'data')
                if max_pixels is not None:
                    content['max_pixels'] = max_pixels
    return messages

class CustomDataset(Dataset):
    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.data_list = []
        data_path = config.train.train_policy.dataset.name
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data_list.append(json.loads(line)['messages'])
        self.max_pixels = config.policy.model_max_length * 0.9 * ((16 * 2) ** 2)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.data_list[idx]
        sample = modify_messages(sample, self.max_pixels)
        return sample

if __name__ == "__main__":
    def patched_parallelize_fn(self):
        # whatever you want to return
        return parallelize, self

    def get_dataset(config: CosmosConfig):
        return CustomDataset()

    hf_models.HFModel.parallelize_fn = property(patched_parallelize_fn)
    hf_models.convert_weight_from_hf = convert_weight_from_hf

    launch_dispatcher(
        dataset=get_dataset,
    )

