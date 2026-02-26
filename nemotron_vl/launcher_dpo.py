# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in written, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DPO Launcher for Nemotron-3-Nano VLM with MMPR-v1.2 preference dataset.
Dataset: /workspace/ruipul/vlm_data/MMPR-v1.2 (chosen/rejected pairs from meta.json)
"""

import json
import os
import sys

os.environ["USE_QWEN_VL_PROCESS"] = "1"
os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"

import torch
import re

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
# MoE load balancing (same as launcher.py)
########################################################


def _to_cpu_np(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    import numpy as np
    return np.asarray(x, dtype=np.float32)


def _balance_metrics(counts_1d):
    import numpy as np
    counts = counts_1d.astype(np.float32)
    total = float(counts.sum() + 1e-9)
    p = counts / total
    entropy = float(-(p * np.log(p + 1e-9)).sum())
    max_fraction = float(p.max())
    std = float(counts.std())
    return entropy, max_fraction, std, total


def report_moe_load_to_wandb(step, loads_by_layer, prefix="moe", log_every=10):
    if len(loads_by_layer) == 0:
        return None
    if (step % log_every) != 0:
        return None

    import numpy as np
    layer_names = sorted(loads_by_layer.keys())
    layer_loads = [_to_cpu_np(loads_by_layer[name]).reshape(-1) for name in layer_names]

    if wandb is not None:
        table = wandb.Table(columns=["step", "layer", "layer_name", "expert", "tokens", "frac"])
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
                table.add_data(int(step), int(li), lname, int(ei), float(counts[ei]), float(frac[ei]))

    return {
        f"{prefix}/layer_expert_table": table,
        f"{prefix}/entropy_mean": float(np.mean(entropies)),
        f"{prefix}/entropy_min": float(np.min(entropies)),
        f"{prefix}/max_fraction_mean": float(np.mean(max_fracs)),
        f"{prefix}/max_fraction_max": float(np.max(max_fracs)),
        f"{prefix}/tokens_per_layer_mean": float(np.mean(totals)),
        f"{prefix}/tokens_per_layer_std_mean": float(np.mean(stds)),
    }


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
                content["max_frames"] = 30
                if max_pixels is not None:
                    content["total_pixels"] = max_pixels
    return messages


class DPODataset(Dataset):
    """
    MMPR-v1.2 DPO dataset. Loads preference pairs from meta.json + annotations.

    Each sample returned by __getitem__ must have the following keys:
        - chosen: list[dict]  # Conversation messages (Qwen/Nemotron format) for the preferred response
        - rejected: list[dict]  # Conversation messages for the dispreferred response

    Each message: {"role": "user"|"assistant", "content": ...}
    User content for VLM: [{"type": "image", "image": path}, {"type": "text", "text": "..."}]
    """

    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.data_list = []
        data_path = config.train.train_policy.dataset.name
        meta_path = os.path.join(data_path, "meta.json")
        self.max_pixels = 32*32*2048

        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"MMPR meta.json not found at {meta_path}. "
                "Ensure dataset path is /workspace/ruipul/vlm_data/MMPR-v1.2"
            )

        with open(meta_path) as f:
            meta = json.load(f)

        base_dir = os.path.dirname(data_path)

        for ds_name, ds_info in meta.items():
            root = ds_info.get("root", "")
            ann = ds_info.get("annotation", "")
            if not ann:
                continue

            ann_path = os.path.join(base_dir, ann) if not os.path.isabs(ann) else ann
            root_path = os.path.join(base_dir, root) if not os.path.isabs(root) else root

            if not os.path.isfile(ann_path):
                continue

            repeat_raw = ds_info.get("repeat_time", 1) or 1
            if repeat_raw <= 0:
                continue

            with open(ann_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            if 0 < repeat_raw < 1:
                import random
                n_take = max(1, int(len(lines) * repeat_raw))
                lines = random.Random(42).sample(lines, min(n_take, len(lines)))
                repeat = 1
            else:
                repeat = max(1, int(repeat_raw))

            for _ in range(repeat):
                for line in lines:
                    try:
                        item = json.loads(line)
                        if "chosen" not in item or "rejected" not in item:
                            continue
                        self.data_list.append({
                            "root": root_path,
                            "item": item,
                        })
                    except Exception:
                        continue

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        entry = self.data_list[idx]
        root = entry["root"]
        item = entry["item"]

        question = item.get("question", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        image_path = None
        if "image" in item and item["image"]:
            img_ref = item["image"]
            if not isinstance(img_ref, list):
                img_ref = [img_ref]

            if img_ref:
                image_path = [os.path.join(root, image_path) for image_path in img_ref]
                for img in image_path:
                    if not os.path.isfile(img):
                        raise FileNotFoundError(f"Image not found: {img}")

        def build_messages(response: str) -> list:
            if image_path:
                contents = [{"type": "image", "image": image_path_item} for image_path_item in image_path]
            else:
                contents = []
            if question:
                contents.append({"type": "text", "text": question})
            if not contents:
                contents = [{"type": "text", "text": ""}]
            messages = [
                {"role": "user", "content": contents},
                {"role": "assistant", "content": response},
            ]
            return modify_messages(messages, self.max_pixels)

        chosen_messages = build_messages(chosen)
        rejected_messages = build_messages(rejected)

        return {"chosen": chosen_messages, "rejected": rejected_messages}


@torch.no_grad()
def policy_map_local_key_for_export_tensor(self, name, expert_weight: torch.Tensor):
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
            yield from yield_weight(n_experts, expert_weight, "up_proj", layer_id)
        else:
            yield from yield_weight(n_experts, expert_weight, "down_proj", layer_id)
    else:
        yield name, expert_weight


def patched_parallelize_fn(self):
    return parallelize, self


def step_hook(self, step):
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
            "WARNING: MoE load balancing training is disabled. "
            "Set enable_moe_load_balancing_training=True in config['custom'] to enable."
        )

    report_data = None
    if step % report_every == 0:
        report_data = report_moe_load_to_wandb(
            step, self._stateful_expert_load_per_layer, prefix="moe"
        )
        self._stateful_expert_load_per_layer = {}
    return report_data


def get_dataset(config: CosmosConfig):
    return DPODataset()


if __name__ == "__main__":
    import cosmos_rl

    cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(patched_parallelize_fn)
    cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
    cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
    cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = (
        policy_map_local_key_for_export_tensor
    )

    from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_dataset,
        data_packer=HFVLMDataPacker(),
    )
