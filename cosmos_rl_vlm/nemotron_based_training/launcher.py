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
from typing import Optional
import os
import sys

# Enable EP mesh to be represented by TP mesh, and also treat EP as a sub-group of Data Parallelism.
os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"

import torch
import re
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nemotron_parallelize import parallelize
from weight_converter import convert_weight_from_hf
from torch.utils.data import Dataset
from cosmos_rl.policy.config import Config as CosmosConfig

try:
    import wandb
except ImportError:
    wandb = None
########################################################
# Auxiliary helper functions for MoE load balancing tracking.
########################################################


def _to_cpu_np(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _balance_metrics(counts_1d: np.ndarray):
    counts = counts_1d.astype(np.float32)
    total = float(counts.sum() + 1e-9)
    p = counts / total
    entropy = float(-(p * np.log(p + 1e-9)).sum())
    max_fraction = float(p.max())
    std = float(counts.std())
    return entropy, max_fraction, std, total


def report_moe_load_to_wandb(
    step: int, loads_by_layer: dict, prefix="moe", log_every=10
):
    if len(loads_by_layer) == 0:
        return None
    if (step % log_every) != 0:
        return None

    layer_names = sorted(loads_by_layer.keys())
    layer_loads = [_to_cpu_np(loads_by_layer[name]).reshape(-1) for name in layer_names]

    if wandb is not None:
        table = wandb.Table(
            columns=["step", "layer", "layer_name", "expert", "tokens", "frac"]
        )
    else:
        table = None
    entropies, max_fracs, stds, totals = [], [], [], []
    for li, (lname, counts) in enumerate(zip(layer_names, layer_loads)):
        entropy, max_fraction, std, total = _balance_metrics(counts)
        entropies.append(entropy)
        max_fracs.append(max_fraction)
        stds.append(std)
        totals.append(total)

        if table is not None:
            frac = counts / (counts.sum() + 1e-9)
            for ei in range(counts.shape[0]):
                table.add_data(
                    int(step),
                    int(li),
                    lname,
                    int(ei),
                    float(counts[ei]),
                    float(frac[ei]),
                )

    return {
        f"{prefix}/layer_expert_table": table,
        f"{prefix}/entropy_mean": float(np.mean(entropies)),
        f"{prefix}/entropy_min": float(np.min(entropies)),
        f"{prefix}/max_fraction_mean": float(np.mean(max_fracs)),
        f"{prefix}/max_fraction_max": float(np.max(max_fracs)),
        f"{prefix}/tokens_per_layer_mean": float(np.mean(totals)),
        f"{prefix}/tokens_per_layer_std_mean": float(np.mean(stds)),
    }


########################################################


def modify_messages(messages, max_pixels=None):
    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for content in message["content"]:
            if content["type"] == "image":
                if max_pixels is not None:
                    content["max_pixels"] = max_pixels
            elif content["type"] == "video":
                content["fps"] = 1
                if max_pixels is not None:
                    content["total_pixels"] = max_pixels
    return messages


class CustomDataset(Dataset):
    """
    Custom dataset for Nemotron-3-Nano Vision-Language Model alignment.
    Assume the dataset are in JSONL format stored in `config.train.train_policy.dataset.name`, and each line is a JSON object with 'messages' key.
    """

    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.data_list = []
        data_path = config.train.train_policy.dataset.name
        jsonl_files = sorted([f for f in os.listdir(data_path) if f.endswith(".jsonl")])
        for file_name in jsonl_files:
            if not config.custom.get("include_video", False) and "webvid" in file_name:
                continue
            with open(os.path.join(data_path, file_name)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data_list.append(json.loads(line)["messages"])
        self.max_pixels = config.policy.model_max_length * 0.9 * ((16 * 2) ** 2)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.data_list[idx]
        sample = modify_messages(sample, self.max_pixels)
        return sample


@torch.no_grad()
def policy_map_local_key_for_export_tensor(self, name, expert_weight: torch.Tensor):
    # Only For Nemotron-3-Nano Base LLM Model naming convention
    # Leave the prefix "model." for Nemotron-3-Nano Vision-Language Model naming convention
    if name.startswith("model.backbone."):
        name = name[len("model.") :]

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
            yield from yield_weight(n_experts, expert_weight, "up_proj", layer_id)
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
            yield from yield_weight(n_experts, expert_weight, "up_proj", layer_id)
        else:
            yield from yield_weight(n_experts, expert_weight, "down_proj", layer_id)
    else:
        yield name, expert_weight


def patched_parallelize_fn(self):
    # whatever you want to return
    return parallelize, self


# MoE: Aux-free load balancing update bias after each step update.
def step_hook(self, step: int) -> Optional[dict]:
    if not hasattr(self, "_stateful_expert_load_per_layer"):
        self._stateful_expert_load_per_layer = {}

    enable_moe_load_balancing_training = self.cosmos_config.custom.get(
        "enable_moe_load_balancing_training", True
    )
    report_every = self.cosmos_config.custom.get("n_step_per_workload_report", 10)

    if enable_moe_load_balancing_training:
        for name, module in self.language_model.named_modules():
            if "NemotronHBlock" in type(module).__name__ and module.block_type == "moe":
                local_expert_load = module.mixer.gate.update_bias()
                with torch.no_grad():
                    if name not in self._stateful_expert_load_per_layer:
                        self._stateful_expert_load_per_layer[name] = local_expert_load
                    else:
                        self._stateful_expert_load_per_layer[name] += local_expert_load
    elif not hasattr(self, "_warn_moe_load_balancing_training_once"):
        self._warn_moe_load_balancing_training_once = True
        print(
            "WARNING: MoE load balancing training is disabled. Please set enable_moe_load_balancing_training to True in the config['custom'] to enable it."
        )

    report_data = None
    if step % report_every == 0:
        report_data = report_moe_load_to_wandb(
            step, self._stateful_expert_load_per_layer, prefix="moe"
        )
        # reset window accumulation
        self._stateful_expert_load_per_layer = {}
    return report_data


def get_dataset(config: CosmosConfig):
    return CustomDataset()


if __name__ == "__main__":
    # Do some monkey patching to support Nemotron-3-Nano Vision-Language Model parallelization.
    import cosmos_rl

    # Override the parallelize_fn to support EP parallelization.
    cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(
        patched_parallelize_fn
    )
    # Override the convert_weight_from_hf to support EP weight sharding during initialization
    cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
    # Override the step_hook to enable aux-free load balancing update bias after each step update.
    cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
    # Map the weight name from custom DeepEP convention back to HF convention for safetensor saving.
    cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor

    # Launch the worker
    cosmos_rl.launcher.worker_entry.main(
        # Uncomment this if you want to use a custom dataset
        dataset=get_dataset,
    )
