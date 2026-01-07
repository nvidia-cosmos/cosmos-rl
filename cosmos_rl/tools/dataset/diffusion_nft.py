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
import json
import os
import toml
from typing import Any, List, Union

import torch
from torch.utils.data import Dataset

from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.config import DatasetConfig
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}


class DiffusionNFTDataPacker(BaseDataPacker):
    def __init__(self):
        super().__init__()

    def get_rollout_input(
        self,
        completions: List[Any] = None,
        completed_conversations: List[Any] = None,
        logprobs: List[Any] = None,
        token_ids: List[Any] = None,
        payload: RLPayload = None,
        n_generation: int = 8,
    ):
        assert payload is not None, "Payload cannot be None."
        prompts = [payload.prompt["prompt"]] * n_generation
        metadatas = [payload.prompt["metadata"]] * n_generation
        return prompts, metadatas

    def get_policy_input(
        self,
        sample: List[Rollout],
        rollout_output: Union[str, List[int]] = None,
        n_ignore_prefix_tokens: int = 0,
    ):
        # Batching the list of rollouts into a single input for the policy
        # Only extra_info is needed for diffusion NFT
        for s in sample:
            s.extra_info["advantages"] = torch.tensor(s.advantage).unsqueeze(0)
        inputs_list = [rollout.extra_info for rollout in sample]
        collated_samples = {
            k: (
                torch.cat([s[k] for s in inputs_list], dim=0)
                if not isinstance(inputs_list[0][k], dict)
                else {
                    sk: torch.cat([s[k][sk] for s in inputs_list], dim=0)
                    for sk in inputs_list[0][k]
                }
            )
            for k in inputs_list[0].keys()
        }
        return collated_samples


def get_dataset(dataset_config: DatasetConfig) -> Dataset:
    assert dataset_config.name in [
        "pickscore",
        "ocr",
        "geneval",
    ], f"Unknown dataset name: {dataset_config.name}"
    prompt_fn = "geneval" if dataset_config.name == "geneval" else "general_ocr"
    if prompt_fn == "general_ocr":
        dataset = TextPromptDataset(dataset_config.name, split=dataset_config.split)
    elif prompt_fn == "geneval":
        dataset = GenevalPromptDataset(dataset_config.name, split=dataset_config.split)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_config.name}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = CosmosConfig.from_dict(config)

    train_dataset = get_dataset(config.train.train_policy.dataset)
    val_dataset = get_dataset(config.validation.dataset)
    rollout_batch_size = (
        config.train.train_policy.dataloader_batch_size or config.rollout.batch_size
    )

    launch_worker(
        dataset=train_dataset,
        val_dataset=val_dataset,
        data_packer=DiffusionNFTDataPacker(),
        val_data_packer=DiffusionNFTDataPacker(),
    )
