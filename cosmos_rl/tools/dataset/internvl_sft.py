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
import toml
import torch
import argparse
from PIL import Image
from datasets import load_dataset
from typing import List, Any, Dict, Optional, Union
from torch.utils.data import Dataset, ConcatDataset
from cosmos_rl.utils.util import retry
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from transformers import AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IGNORE_LABEL_ID = -100

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, pil_image=None, input_size=448, max_num=12):
    image = (
        pil_image if pil_image is not None else Image.open(image_file).convert("RGB")
    )
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL_DataPacker(DataPacker):
    """
    Data protocol & processing logic for the InternVL for SFT and RL training.
    """

    Payload = List[Dict[str, Any]]

    class RLPolicyInput:
        input_ids: List[int]
        logprob_masks: List[int]

        def __init__(self, input_ids: List[int], logprob_masks: List[int]):
            self.input_ids = input_ids
            self.logprob_masks = logprob_masks

    def setup(self, config: CosmosConfig, *args, **kwargs):
        super().setup(config, *args, **kwargs)
        hf_config = retry(AutoConfig.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )
        self.image_token_id = 151671
        self.vision_ids = [self.image_token_id]
        self.hf_config = hf_config

    def get_rollout_input(self, sample: Payload) -> Any:
        # Here we need to convert the conversation format to the format required by vllm
        # TODO(huik): support RL
        return {}

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
        conversation: "InternVL_DataPacker.Payload",
        add_generation_prompt: bool,
    ) -> Dict[str, Any]:
        try:
            # Replace all the assistant content with consecutive `pad_token` * 10
            pad_token = self.tokenizer.pad_token
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_run_length = 10
            assistant_contents = []
            images = conversation["images"][0]
            messages = conversation["messages"]
            result = []
            for item in messages:
                role = item["role"]
                contents = item["content"]
                has_image = role == "user" and any(
                    c["type"] == "image" for c in contents
                )
                texts = []
                for c in contents:
                    if c["type"] == "text" and c["text"] is not None:
                        if role == "assistant":
                            assistant_contents.append(c["text"])
                            text = pad_token * 10
                        else:
                            text = c["text"]
                            if has_image:
                                text = "<image>\n" + text
                        texts.append(text)
                role_text = "\n".join(texts)
                result.append(f"{role}: {role_text}")
            prompt = "\n".join(result)

            # assistant_contents = ["This image"]
            # prompt = f"User: <image>\ndescribe the image.\nAssistant: {pad_token * 10}"

            pixel_values = load_image(None, pil_image=images)
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )

            IMG_START_TOKEN = "<img>"
            IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
            IMG_END_TOKEN = "</img>"
            num_image_token = 256
            for num_patches in num_patches_list:
                image_tokens = (
                    IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * num_image_token * num_patches
                    + IMG_END_TOKEN
                )
                prompt = prompt.replace("<image>", image_tokens, 1)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs["pixel_values"] = pixel_values

            input_ids = inputs["input_ids"][0].tolist()
            label_ids = [IGNORE_LABEL_ID] * len(input_ids)

            for assistant_content in assistant_contents:
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
            "input_ids": copy.deepcopy(input_ids),
            "label_ids": copy.deepcopy(label_ids),
        }

        if "pixel_values" in inputs:
            result_dict["pixel_values"] = inputs["pixel_values"]

        image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long)
        result_dict["image_flags"] = image_flags
        return result_dict

    def _collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        pixel_values = [x["pixel_values"] for x in processed_samples]
        image_flags = [x["image_flags"] for x in processed_samples]

        if all([x is not None for x in pixel_values]):
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            # TODO(jiaxin): handle the case when there is mixed input: some with image, some without image
            assert all([x is None for x in pixel_values]), "pixel_values should be None"
            pixel_values = None

        if all([x is not None for x in image_flags]):
            image_flags = torch.cat(image_flags, dim=0)
        else:
            assert all([x is None for x in image_flags]), "image_flags should be None"
            image_flags = None

        batch = {}
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        if image_flags is not None:
            batch["image_flags"] = image_flags

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
        sample: "InternVL_DataPacker.Payload",
        rollout_output: Optional[Union[str, List[int]]] = None,
        n_ignore_prefix_tokens: int = 0,
        add_generation_prompt: bool = True,
    ) -> Any:
        x = self._process_single_sample(
            sample, add_generation_prompt=add_generation_prompt
        )

        return_dict = {}
        if "pixel_values" in x:
            return_dict["pixel_values"] = x["pixel_values"]
        else:
            return_dict["pixel_values"] = None

        if "image_flags" in x:
            return_dict["image_flags"] = x["image_flags"]
        else:
            return_dict["image_flags"] = None

        # Common fields
        input_ids = x["input_ids"]
        completion_ids = []
        if rollout_output:
            rollout_as_token_ids = isinstance(rollout_output, list) and all(
                isinstance(i, int) for i in rollout_output
            )
            if rollout_as_token_ids:
                completion_ids = rollout_output
            else:
                completion_ids = self.tokenizer(rollout_output).input_ids
            return_dict["input_ids"] = input_ids + completion_ids
        else:
            return_dict["input_ids"] = input_ids

        return_dict["logprob_masks"] = (
            [0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0]
        )

        # TODO(jiaxin): this is special for SFT, will be removed in ``policy_collate_fn``
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

    def sft_process_sample(
        self, sample: "InternVL_DataPacker.Payload"
    ) -> Dict[str, Any]:
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


class InternVLSFTDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def setup(self, config: CosmosConfig, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config

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
        # idx = 0
        conversations = self.dataset[idx]
        # Just for reference
        # In case your model does not support image in RGBA mode, convert them to RGB mode
        # if "images" in conversations:
        #     images = conversations["images"]
        #     new_images = []
        #     for image in images:
        #         if image.mode == "RGB":
        #             new_images.append(image)
        #         else:
        #             new_images.append(image.convert("RGB"))
        #     conversations["images"] = new_images
        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = CosmosConfig.from_dict(config)

    def get_dataset(config: CosmosConfig) -> Dataset:
        dataset = load_dataset(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
        )
        return InternVLSFTDataset(dataset)

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
        data_packer=InternVL_DataPacker(),
        val_data_packer=InternVL_DataPacker(),
    )
