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
import toml, copy
from cosmos_rl.utils.logging import logger

from torch.utils.data import Dataset
from datasets import concatenate_datasets
import cosmos_rl.utils.util as util
from typing import List, Dict, Union
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import DecoderOnlyLLMDataPacker


# This dataset is used for SFT with raw text input, which is used for models like Mistral
# It converts the conversation list to string format for models requiring raw text input
# This handles cases like Mistral where conversation dicts need string conversion
# to avoid role alternation errors
class SFTRawTextDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def setup(
        self,
        config: Config,
    ):
        self.config = config.train.train_policy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve raw item from dataset
        raw_item = self.dataset[idx]["conversation"]
        # raw_item = [{'role': 'user', 'content': 'what is 2.0909090923 x 0.897987987'}]
        # ["conversation"]
        # # Convert conversation list to string format
        # if isinstance(raw_item, list):
        #     raw_item = "\n".join(
        #         [f"{turn['role']}: {turn['content']}" for turn in raw_item]
        #     )
        raw_item = [
            {"role": turn["role"], "content": turn["content"]} for turn in raw_item if turn["role"] in ["user", "assistant"]
        ]
        if len(raw_item) == 0:
            # logger.warning(f"Empty conversation at index {idx}, returning default prompt")
            raw_item = [{"role": "user", "content": "Hello!"}]
        
        return raw_item
    
class DecoderOnlyLLMSFTRawTextDataPacker(DecoderOnlyLLMDataPacker):
    def sft_process_sample(self, sample: Union[str, List[Dict[str, str]]]) -> List[int]:
        """
        Process the sample into the format required by the SFT model.
        Accepts either raw text or conversation format.
        """
        # 1. if item is a string, then assume it is a raw text
        if isinstance(sample, str):
            token_ids = self.tokenizer(sample, add_special_tokens=False).input_ids
            label_ids = token_ids.copy()
        # 2. if item is a list, then assume it is in conversation format:
        else:
            original_sample = copy.deepcopy(sample)

            try:
                if (
                    self.tokenizer.pad_token is None
                    or self.tokenizer.pad_token_id is None
                ):
                    raise ValueError("pad_token and pad_token_id should be set")

                assistant_contents = []
                pad_token = self.tokenizer.pad_token
                pad_token_id = self.tokenizer.pad_token_id
                eos_token_id = self.tokenizer.eos_token_id
                pad_run_length = 10


                # logger.info(f"Processing sample: {sample}")
                for x in sample:
                    if x["role"] == "assistant":
                        assistant_contents.append(x["content"])
                        x["content"] = ""

                token_ids = self.tokenizer.apply_chat_template(
                    sample,
                    return_dict=True,
                    add_generation_prompt=False,
                )["input_ids"]
                
                for x in sample:
                    if x["role"] == "assistant":
                        x["content"] = assistant_contents.pop(0)
                
                full_ids = self.tokenizer.apply_chat_template(
                    sample,
                    return_dict=True,
                    add_generation_prompt=False,
                )["input_ids"]
                label_ids = [-100] * len(token_ids) + full_ids[len(token_ids):]
                token_ids = full_ids
                


                # for assistant_content in assistant_contents:
                #     replaced, token_ids, label_ids = self._replace_assistant_content(
                #         token_ids,
                #         label_ids,
                #         pad_token_id=pad_token_id,
                #         eos_token_id=eos_token_id,
                #         replacement_ids=self.tokenizer.encode(
                #             assistant_content, add_special_tokens=False
                #         ),
                #         pad_run_length=pad_run_length,
                #     )
                #     if not replaced:
                #         raise ValueError("No assistant content to replace")
                if len(token_ids) != len(label_ids):
                    raise ValueError(
                        f"token_ids and label_ids should have the same length, but got {len(token_ids)} and {len(label_ids)}"
                    )
            except Exception as e:
                logger.info(f"Processing sample: {sample}, error: {e}")
                raise
                # Fall back to the non-assistant-masking
                token_ids = self.tokenizer.apply_chat_template(
                    original_sample,
                    return_assistant_tokens_mask=False,
                    return_dict=True,
                    add_generation_prompt=False,
                )["input_ids"]
                label_ids = token_ids.copy()
        assert isinstance(token_ids, list), "token_ids should be a list"
        assert isinstance(token_ids[0], int), "Each item in token_ids should be an int"
        return {
            "token_ids": token_ids,
            "label_ids": label_ids,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)
    # Download HF dataset only on launcher worker
    dataset = util.load_data_from_disk_or_hf(
        config.train.train_policy.dataset.name,
        config.train.train_policy.dataset.subset,
        config.train.train_policy.dataset.revision or None,
    )
    dataset_list = []
    for split_name in config.train.train_policy.dataset.split:
        print(
            f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
        )
        dataset_list.append(dataset[split_name])
    train_dataset = concatenate_datasets(dataset_list)
    launch_dispatcher(
        dataset=SFTRawTextDataset(dataset=train_dataset),
        data_packer=DecoderOnlyLLMSFTRawTextDataPacker(),
    )