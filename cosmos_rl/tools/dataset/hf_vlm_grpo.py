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

from typing import Any, List, Dict
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from transformers import AutoTokenizer
from cosmos_rl.dispatcher.data.packer import DataPacker, HFVLMDataPacker
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
import argparse
import toml
import base64
import io
from PIL import Image as PILImage


class HFVLMGRPODataset(Dataset):
    def setup(self, config: CosmosConfig, tokenizer: AutoTokenizer, *args, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
        )
        # Change the system prompt to the one you want
        self.system_prompt = "You are a mathematics expert. For each math problem presented together with an image, thoroughly interpret the visual and textual information and provide a detailed, step-by-step solution, ensuring that your reasoning is clear and precise at every stage."
        self.prompt_column_name = config.train.train_policy.prompt_column_name
        self.response_column_name = config.train.train_policy.response_column_name
        # Below is the image/video column name for the dataset lmms-lab/multimodal-open-r1-8k-verified
        # You can change it to the column name of your dataset
        self.image_column_name = "image"
        self.video_column_name = ""
        self.has_image = self.image_column_name != ""
        self.has_video = self.video_column_name != ""
        self.reward_function = config.train.train_policy.reward_function
        self.has_boxed_math_reward = "boxed_math" in self.reward_function
        self.has_format_reward = "format" in self.reward_function
        assert (
            self.has_image or self.has_video
        ), "At least one of image or video column name must be provided"
        logger.info(f"[HFVLMGRPODataset] system_prompt: {self.system_prompt}")

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

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """
        Return a tuple of (prompt, reference answer)
        """
        payload = self.dataset[idx]
        prompt = payload[self.prompt_column_name]
        user_prompt = prompt
        system_prompt = self.system_prompt
        if self.has_format_reward:
            user_prompt += "\nPlease answer the question in the following format: <think> your reasoning process </think> <answer> your final answer </answer>."
        if self.has_boxed_math_reward:
            user_prompt += "\nPlease ensure your final answer is put in \\boxed{}."

        user_conv = [
            {
                "type": "text",
                "text": user_prompt,
            },
        ]
        # TODO: add video support
        if self.has_image:
            images = payload[self.image_column_name]
            img_obj = None
            pil_img = None
            if isinstance(images, list):
                assert len(images) == 1, "Only one image is supported"
                img_obj = images[0]
            else:
                img_obj = images
            assert isinstance(
                img_obj, PILImage.Image
            ), "image_obj is not PIL.Image.Image"
            pil_img = img_obj
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            multi_modal_content = {
                "type": "image",
                "image": img_b64,
            }
            user_conv.insert(0, multi_modal_content)

        conversations = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_conv,
            },
        ]

        return conversations

    def get_reference_answer(self, idx: int) -> str:
        response = self.dataset[idx][self.response_column_name]
        if self.has_boxed_math_reward and "boxed" not in response:
            response = "$\\boxed{" + response + "}$"
        return response


class HFVLMGRPOValDataset(HFVLMGRPODataset):
    """
    This is a validation dataset for Cosmos GRPO, which is used to evaluate the performance of the model.
    It should be used in the launcher to evaluate the model during training.
    """

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        if not config.train.enable_validation:
            logger.warning(
                "Validation is not enabled in the config. Skipping setup for CosmosGRPOValDataset."
            )
            return

        self.config = config
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            config.validation.dataset.name, config.validation.dataset.subset
        )
        # self.prompt_column_name = config.train.train_policy.prompt_column_name
        # self.response_column_name = config.train.train_policy.response_column_name
        self.image_column_name = config.train.train_policy.image_column_name
        self.video_column_name = config.train.train_policy.video_column_name
        self.has_image = self.image_column_name != ""
        self.has_video = self.video_column_name != ""
        assert (
            self.has_image or self.has_video
        ), "At least one of image or video column name must be provided"

        if config.validation.dataset.split:
            if isinstance(config.validation.dataset.split, list):
                dataset_list = []
                for split_name in config.validation.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.validation.dataset.split, str)
                self.dataset = self.dataset[config.validation.dataset.split]


class DemoDataPacker(DataPacker):
    """
    This is a demo data packer that wraps the underlying data packer of the selected model.
    This is meaningless for this example, but useful for explaining:
        - how dataset data is processed and collated into a mini-batch for rollout engine;
        - how rollout output is processed and collated into a mini-batch for policy model;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check source code of Qwen2_5_VLM_DataPacker to see how it's implemented
        self.underlying_data_packer = HFVLMDataPacker()

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        This method is optional and get called by launcher after being mounted
        `config`: config;
        `tokenizer`: tokenizer;
        """
        super().setup(config, tokenizer, *args, **kwargs)
        self.underlying_data_packer.setup(config, tokenizer, *args, **kwargs)

    def get_rollout_input(self, item: Any) -> Any:
        """
        Convert dataset item into what rollout engine (e.g. vllm) expects
        """
        return self.underlying_data_packer.get_rollout_input(item)

    def rollout_collate_fn(self, items: List[Any]) -> Any:
        """
        Collate the rollout inputs into a mini-batch for rollout engine
        """
        return self.underlying_data_packer.rollout_collate_fn(items)

    def get_policy_input(
        self, item: Any, rollout_output: str, n_ignore_prefix_tokens: int = 0
    ) -> Any:
        """
        Process samples & rollout output before collating them into a mini-batch
        """
        return self.underlying_data_packer.get_policy_input(
            item, rollout_output, n_ignore_prefix_tokens
        )

    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute the maximum sequence length of the mini-batch
        """
        return self.underlying_data_packer.policy_compute_max_len(processed_samples)

    def policy_collate_fn(
        self, processed_samples: List[Any], computed_max_len: int
    ) -> Dict[str, Any]:
        """
        Collate the mini-batch into the kwargs required by the policy model
        """
        return self.underlying_data_packer.policy_collate_fn(
            processed_samples, computed_max_len
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)

    # It is best practice to pass the dataset and val_dataset as factory functions
    # so that the dataset and val_dataset can be loaded on demand. (Not all workers need them)
    def get_dataset(config: CosmosConfig) -> Dataset:
        return HFVLMGRPODataset()

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        return HFVLMGRPOValDataset() if config.train.enable_validation else None

    launch_worker(
        dataset=get_dataset,
        data_packer=DemoDataPacker(),
        val_dataset=get_val_dataset,
        val_data_packer=DemoDataPacker(),
    )
