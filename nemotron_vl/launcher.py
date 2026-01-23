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

import json
import os, sys
import torch, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nemotron_parallelize import parallelize
from weight_converter import convert_weight_from_hf
from torch.utils.data import Dataset
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher
import cosmos_rl.policy.model.hf_models as hf_models
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger

def modify_messages(messages, max_pixels = None):
    for message in messages:
        if isinstance(message['content'], str):
            message['content'] = [{'type': 'text', 'text': message['content']}]
        for content in message['content']:
            if content['type'] in ['image', 'video']:
                if max_pixels is not None:
                    content['max_pixels'] = max_pixels
    return messages

class CustomDataset(Dataset):
    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.data_list = []
        data_path = config.train.train_policy.dataset.name
        jsonl_files = sorted(
            [f for f in os.listdir(data_path) if f.endswith(".jsonl")]
        )
        for file_name in jsonl_files:
            if 'webvid' in file_name:
                continue
            with open(os.path.join(data_path, file_name)) as f:
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

    
class CustomDatasetForText(Dataset):
    def setup(self, config: CosmosConfig, *args, **kwargs):
        from datasets import load_dataset
        self.ds = load_dataset("open-thoughts/OpenThoughts3-1.2M")['train'].shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> list[dict]:
        messages: list[dict] = self.ds[idx]['conversations']
        from_to_role = {
            "human": "user",
            "gpt": "assistant",
            "system": "system",
        }
        # rename key from `from` to `role`, and `value` to `content`
        messages = [{'role': from_to_role[turn['from']], 'content': turn['value']} for turn in messages]
        return messages

@torch.no_grad()
def policy_map_local_key_for_export_tensor(self, name, expert_weight: torch.Tensor):

    # Only For Nemotron-3-Nano Base LLM Model naming convention
    # Leave the prefix "model." for Nemotron-3-Nano Vision-Language Model naming convention
    if name.startswith("model.backbone."):
        name = name[len("model."):]

    if match := re.search(
        r"backbone\.layers\.(\d+)\.mixer\.experts\.(gate_and_up_projs|down_projs)",
        name,
    ):
        def yield_weight(n_experts, expert_weight, w_name, layer_id):
            for expert_id in range(n_experts):
                single_expert_weight = expert_weight[expert_id].contiguous()
                yield (
                    f"backbone.layers.{layer_id}.mixer.experts.{expert_id}.{w_name}.weight",
                    single_expert_weight,
                )
        layer_id = int(match.group(1))
        w_name = match.group(2)
        n_experts = expert_weight.shape[0]
        if w_name == "gate_and_up_projs":
            # gate_and_up_projs is `up_proj` in nemotron
            # shape: [experts, ffn_dim, hidden_dim]
            yield from yield_weight(
                n_experts, expert_weight, "up_proj", layer_id
            )
        else:
            yield from yield_weight(n_experts, expert_weight, "down_proj", layer_id)
    elif match := re.search(
        r"model\.language_model\.layers\.(\d+)\.mixer\.experts\.(gate_and_up_projs|down_projs)",
        name,
    ):
        def yield_weight(n_experts, expert_weight, w_name, layer_id):
            for expert_id in range(n_experts):
                single_expert_weight = expert_weight[expert_id].contiguous()
                yield (
                    f"model.language_model.layers.{layer_id}.mixer.experts.{expert_id}.{w_name}.weight",
                    single_expert_weight,
                )
        layer_id = int(match.group(1))
        w_name = match.group(2)
        n_experts = expert_weight.shape[0]
        if w_name == "gate_and_up_projs":
            # gate_and_up_projs is `up_proj` in nemotron
            # shape: [experts, ffn_dim, hidden_dim]
            yield from yield_weight(
                n_experts, expert_weight, "up_proj", layer_id
            )
        else:
            yield from yield_weight(n_experts, expert_weight, "down_proj", layer_id)
    else:
        yield name, expert_weight

def patched_parallelize_fn(self):
    # whatever you want to return
    return parallelize, self

# MoE: Aux-free load balancing update bias after each step update.
def step_hook(self):
    enable_moe_load_balancing_training = self.cosmos_config.custom.get("enable_moe_load_balancing_training", True)
    if enable_moe_load_balancing_training:
        for _, module in self.language_model.named_modules():
            if 'NemotronHBlock' in type(module).__name__ and module.block_type == "moe":
                module.mixer.gate.update_bias()
    elif not hasattr(self, "_warn_moe_load_balancing_training_once"):
        self._warn_moe_load_balancing_training_once = True
        print("WARNING: MoE load balancing training is disabled. Please set enable_moe_load_balancing_training to True in the config['custom'] to enable it.")

def get_dataset(config: CosmosConfig):
    return CustomDataset()

if __name__ == "__main__":
    # 1. 参数requires_grad配置
    # 2. 训练长度设置
    # 3. MoE的router是否需要训练load-balancing

    # Override the parallelize_fn to support EP
    hf_models.HFModel.parallelize_fn = property(patched_parallelize_fn)
    # Override the convert_weight_from_hf to support EP weight sharding during initialization
    hf_models.convert_weight_from_hf = convert_weight_from_hf
    # Override the step_hook to enable aux-free load balancing update bias after each step update.
    hf_models.HFModel.step_hook = step_hook
    # Map the weight name from custom DeepEP convention back to HF convention for safetensor saving.
    HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor
    
    launch_dispatcher(
        dataset=get_dataset,
    )

