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

from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import AutoTokenizer
import argparse
import toml

FPS = 1
MAX_PIXELS = 81920


class HFVLMSFTDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer
        self.conversation_column_name = (
            config.train.train_policy.conversation_column_name
        )
        self.image_column_name = config.train.train_policy.image_column_name
        self.video_column_name = config.train.train_policy.video_column_name
        self.has_image = self.image_column_name != ""
        self.has_video = self.video_column_name != ""
        # self.image_dir = None
        # self.image_files_paths = None
        # self.video_dir = None
        # self.video_files_paths = None
        assert (
            self.has_image or self.has_video
        ), "At least one of image or video column name must be provided"

        if config.train.train_policy.dataset.split:
            if isinstance(config.train.train_policy.dataset.split, list):
                dataset_list = []
                for split_name in config.train.train_policy.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.train.train_policy.dataset.split, str)
                self.dataset = self.dataset[config.train.train_policy.dataset.split]

        # get multi-modal files paths
        # cosmos_cache_dir = os.environ.get(
        #     "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
        # )
        # if self.has_image:
        #     self.image_dir = os.path.join(
        #         cosmos_cache_dir,
        #         "datasets",
        #         basename_from_modelpath(config.train.train_policy.dataset.name),
        #         config.train.train_policy.dataset.subset,
        #         self.image_column_name,
        #     )
        #     if not os.path.exists(self.image_dir):
        #         raise FileNotFoundError(
        #             f"Dataset directory {self.image_dir} does not exist. Please check the dataset path."
        #         )
        #     image_files_paths = {}
        #     for root, dirs, files in os.walk(self.image_dir):
        #         for file in files:
        #             if file.endswith((".jpg", ".jpeg", ".png")):  # Common image extensions
        #                 image_files_paths[file] = os.path.join(root, file)
        #     self.image_files_paths = image_files_paths

        # if self.has_video:
        #     self.video_dir = os.path.join(
        #         cosmos_cache_dir,
        #         "datasets",
        #         basename_from_modelpath(config.train.train_policy.dataset.name),
        #         config.train.train_policy.dataset.subset,
        #         self.video_column_name,
        #     )
        #     if not os.path.exists(self.video_dir):
        #         raise FileNotFoundError(
        #             f"Dataset directory {self.video_dir} does not exist. Please check the dataset path."
        #         )
        #     video_files_paths = {}
        #     for root, dirs, files in os.walk(self.video_dir):
        #         for file in files:
        #             if file.endswith((".mp4", ".avi", ".mov")):  # Common video extensions
        #                 video_files_paths[file] = os.path.join(root, file)
        #     self.video_files_paths = video_files_paths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Return a tuple of (prompt, reference answer)
        """
        conversations = self.dataset[idx]
        # conversations = copy.deepcopy(payload[self.conversation_column_name])

        # for conv in conversations:
        #     if conv["role"] == "user":
        #         assert isinstance(conv["content"], str), "User message must be string"
        #         # Rewrite to support image/video tokens
        #         image_info = None
        #         content = [
        #             {
        #                 "type": "text",
        #                 "text": conv["content"],
        #             },
        #         ]
        #         if self.has_image:
        #             image_info = {
        #                 "type": "image",
        #                 "image": payload[self.image_column_name],
        #             }
        #             content.insert(0, image_info)

        #         conv["content"] = content

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
        return HFVLMSFTDataset(dataset)

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
    )
