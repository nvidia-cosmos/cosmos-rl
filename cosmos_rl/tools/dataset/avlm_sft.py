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

import copy
import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from typing import List, Any, Dict, Optional
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.util import retry
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
import argparse
import toml


IGNORE_LABEL_ID = -100


class AVLMDataPacker(DataPacker):
    """
    Data protocol & processing logic for AVLM SFT.
    """

    Payload = List[Dict[str, Any]]

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        super().setup(config, tokenizer, *args, **kwargs)
        self.hf_processor = retry(AutoProcessor.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )

        hf_config = retry(AutoConfig.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )

        image_token_id = getattr(hf_config, "image_token_id", None) or getattr(
            hf_config.vision_config, "image_token_id", None
        )
        if image_token_id is None:
            image_token_id = getattr(hf_config, "image_token_index", None) or getattr(
                hf_config.vision_config, "image_token_index", None
            )
        assert image_token_id is not None, f"Cannot find image token id in {hf_config=}"
        self.image_token_id = image_token_id
        self.image_token = getattr(self.hf_processor, "image_token", None)

        video_token_id = getattr(hf_config, "video_token_id", None) or getattr(
            hf_config.vision_config, "video_token_id", None
        )
        if video_token_id is None:
            video_token_id = getattr(hf_config, "video_token_index", None) or getattr(
                hf_config.vision_config, "video_token_index", None
            )
        if video_token_id is None:
            self.video_token = None
            self.video_token_id = None
        else:
            self.video_token = self.tokenizer.decode([video_token_id])
            self.video_token_id = video_token_id
        self.vision_ids = [self.image_token_id, self.video_token_id]
        self.hf_config = hf_config

    def get_rollout_input(self, sample: Payload) -> Any:
        return sample

    def _replace_assistant_content(
        self,
        token_ids: List[int],
        label_ids: List[int],
        pad_token_id: int,
        eos_token_id: int,
        replacement_ids: List[int],
        pad_run_length: int = 10,
    ) -> List[int]:
        """
        Find the first run of exactly `pad_run_length` pad_token_id's in token_ids,
        replace that run with replacement_ids, and return the new list.
        If no such run is found, returns the original list unchanged.
        """
        n = len(token_ids)
        target_run = [pad_token_id] * pad_run_length

        # find the start index of the first matching run
        for i in range(n - pad_run_length + 1):
            if token_ids[i : i + pad_run_length] == target_run:
                # splice in the replacement
                if (
                    len(token_ids) > i + pad_run_length
                    and token_ids[i + pad_run_length] == eos_token_id
                ):
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + [eos_token_id]
                        + label_ids[i + pad_run_length + 1 :]
                    )
                else:
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + label_ids[i + pad_run_length :]
                    )
                return (
                    True,
                    token_ids[:i] + replacement_ids + token_ids[i + pad_run_length :],
                    label_ids,
                )
        # no match found
        return False, token_ids, label_ids

    def _process_single_sample(
        self,
        conversation: "AVLMDataPacker.Payload",
        add_generation_prompt: bool,
    ) -> Dict[str, Any]:
        try:
            # Replace all the assistant content with consecutive `pad_token` * 10
            pad_token = self.tokenizer.pad_token
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_run_length = 10
            assistant_content = []
            assert (
                "messages" in conversation
            ), f"messages not in conversation: {conversation}"
            assert (
                "images" in conversation
            ), f"images not in conversation: {conversation}"
            messages = conversation["messages"]
            image_inputs = conversation["images"]
            for message in messages:
                if message["role"] == "assistant":
                    content = message["content"]
                    new_content = copy.deepcopy(content)
                    if isinstance(new_content, str):
                        assistant_content.append(new_content)
                        new_content = [
                            {"text": pad_token * pad_run_length, "type": "text"}
                        ]
                    elif isinstance(new_content, dict):
                        assert "text" in new_content, f"text not in content: {content}"
                        assistant_content.append(new_content["text"])
                        new_content["text"] = pad_token * pad_run_length
                    elif isinstance(content, list):
                        for i, item in enumerate(content):
                            if isinstance(item, dict):
                                assert "text" in item, f"text not in content: {item}"
                                assistant_content.append(item["text"])
                                new_content[i]["text"] = pad_token * pad_run_length
                            else:
                                raise ValueError(
                                    f"Unsupported content type: {type(item)}"
                                )
                    else:
                        raise ValueError(f"Unsupported content type: {type(content)}")
                    message["content"] = new_content
                elif message["role"] == "user":
                    content = message["content"]
                    new_content = copy.deepcopy(content)
                    if isinstance(content, list):
                        for i, item in enumerate(content):
                            if isinstance(item, dict):
                                if "type" in item and item["type"] == "image":
                                    new_content[i]["image"] = image_inputs[0]
                                # elif "video" in item:
                                #     new_content[i]["video"] = video_inputs
                                else:
                                    continue
                            else:
                                raise ValueError(
                                    f"Unsupported content type: {type(item)}"
                                )
                    message["content"] = new_content

            text = self.hf_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

            kwarg = {
                "return_tensors": "pt",
                "images": image_inputs,
            }

            image_inputs, video_inputs = self.hf_processor.process_vision_info(messages)
            kwarg["images"] = image_inputs
            kwarg["videos"] = video_inputs
            inputs = self.hf_processor(
                text=[text],
                **kwarg,
            )

            input_ids = inputs["input_ids"][0].tolist()
            label_ids = [IGNORE_LABEL_ID] * len(input_ids)

            for assistant_content in assistant_content:
                replacement_ids = self.tokenizer.encode(
                    assistant_content, add_special_tokens=False
                )

                replaced, input_ids, label_ids = self._replace_assistant_content(
                    input_ids,
                    label_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    replacement_ids=replacement_ids,
                    pad_run_length=pad_run_length,
                )
                if not replaced:
                    raise ValueError("No assistant content to replace")
                if len(input_ids) != len(label_ids):
                    raise ValueError(
                        f"input_ids and label_ids should have the same length, but got {len(input_ids)} and {len(label_ids)}"
                    )
        except Exception as e:
            print(f"Error processing sample: {e}, please fix to ensure SFT works")
            raise e

        result_dict = {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }

        result_dict["pixel_values"] = (
            inputs["pixel_values"] if "pixel_values" in inputs else None
        )
        result_dict["image_sizes"] = (
            inputs["image_sizes"] if "image_sizes" in inputs else None
        )
        result_dict["batch_num_images"] = (
            inputs["batch_num_images"] if "batch_num_images" in inputs else None
        )

        return result_dict

    def _collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        pixel_values = [x["pixel_values"] for x in processed_samples]
        image_sizes = [x["image_sizes"] for x in processed_samples]
        batch_num_images = [x["batch_num_images"] for x in processed_samples]

        if all([x is not None for x in pixel_values]):
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            assert all([x is None for x in pixel_values]), "pixel_values should be None"
            pixel_values = None

        if all([x is not None for x in image_sizes]):
            image_sizes = torch.cat(image_sizes, dim=0)
        else:
            assert all([x is None for x in image_sizes]), "image_sizes should be None"
            image_sizes = None

        if all([x is not None for x in batch_num_images]):
            batch_num_images = torch.cat(batch_num_images)
        else:
            assert all(
                [x is None for x in batch_num_images]
            ), "batch_num_images should be None"
            batch_num_images = None

        batch = {}
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values[0]

        if image_sizes is not None:
            batch["image_sizes"] = image_sizes

        if batch_num_images is not None:
            batch["batch_num_images"] = batch_num_images

        # Pad the input_ids, logprob_masks
        batch["input_ids"] = torch.tensor(
            [
                x["input_ids"][:computed_max_len]
                + [self.tokenizer.pad_token_id]
                * (max(0, computed_max_len - len(x["input_ids"])))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        if "label_ids" in processed_samples[0]:
            batch["label_ids"] = torch.tensor(
                [
                    x["label_ids"][:computed_max_len]
                    + [IGNORE_LABEL_ID]
                    * (max(0, computed_max_len - len(x["label_ids"])))
                    for x in processed_samples
                ],
                dtype=torch.long,
            )
        batch["logprob_masks"] = torch.tensor(
            [
                x["logprob_masks"][:computed_max_len]
                + [0] * (max(0, computed_max_len - len(x["logprob_masks"])))
                for x in processed_samples
            ],
            dtype=torch.bool,
        )

        assert len(batch["input_ids"]) == len(
            batch["logprob_masks"]
        ), "The length of input_ids, logprob_masks should be the same"

        return batch

    def get_policy_input(
        self,
        sample: "AVLMDataPacker.Payload",
        rollout_output: Optional[str] = None,
        n_ignore_prefix_tokens: int = 0,
        add_generation_prompt: bool = True,
    ) -> Any:
        # assert all(
        #     isinstance(x, dict) and "role" in x and "content" in x for x in sample
        # ), "All samples should be in conversation format, but got: {}".format(sample)
        x = self._process_single_sample(
            sample,
            add_generation_prompt=add_generation_prompt,
        )

        return_dict = {}

        return_dict["pixel_values"] = x["pixel_values"] if "pixel_values" in x else None
        return_dict["image_sizes"] = x["image_sizes"] if "image_sizes" in x else None
        return_dict["batch_num_images"] = (
            x["batch_num_images"] if "batch_num_images" in x else None
        )

        # Common fields
        input_ids = x["input_ids"]

        return_dict["input_ids"] = input_ids

        return_dict["logprob_masks"] = (
            [0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (-n_ignore_prefix_tokens)
            + [0]
        )

        return_dict["label_ids"] = x["label_ids"]
        return return_dict

    def policy_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        return max([len(x["input_ids"]) for x in processed_samples])

    def policy_collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        for x in processed_samples:
            if "label_ids" in x:
                del x["label_ids"]
        return self._collate_fn(processed_samples, computed_max_len)

    def sft_process_sample(self, sample: "AVLMDataPacker.Payload") -> Dict[str, Any]:
        """
        Accepts either raw text or conversation format.
        """
        return self.get_policy_input(sample, add_generation_prompt=False)

    def sft_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        return max([len(x["input_ids"]) for x in processed_samples])

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        # Reuse the RL collate minibatch function
        model_inputs: Dict[str, Any] = self._collate_fn(
            processed_samples, computed_max_len
        )
        del model_inputs["logprob_masks"]
        # Mask the loss on vision padding tokens
        if self.vision_ids is not None:
            assert isinstance(self.vision_ids, list)
            for vision_id in self.vision_ids:
                if vision_id is not None:
                    model_inputs["label_ids"][
                        model_inputs["label_ids"] == vision_id
                    ] = ignore_label_id

        return model_inputs


class AVLMDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer

        if config.train.train_policy.dataset.split:
            if isinstance(config.train.train_policy.dataset.split, list):
                dataset_list = []
                for split_name in config.train.train_policy.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.train.train_policy.dataset.split, str)
                self.dataset = self.dataset[config.train.train_policy.dataset.split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        # The dataset is a list of {messages, images} pair
        conversations = self.dataset[idx]
        # In case your model does not support image in RGBA mode, convert them to RGB mode
        if "images" in conversations:
            images = conversations["images"]
            new_images = []
            for image in images:
                if image.mode == "RGB":
                    new_images.append(image)
                else:
                    new_images.append(image.convert("RGB"))
            conversations["images"] = new_images
        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)

    def get_dataset(config: CosmosConfig) -> Dataset:
        dataset = load_dataset(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
        )
        return AVLMDataset(dataset)

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        return AVLMDataset()

    # It is best practice to pass the dataset as a factory function
    launch_worker(
        dataset=get_dataset,
        data_packer=AVLMDataPacker(),
        val_data_packer=AVLMDataPacker(),
    )
