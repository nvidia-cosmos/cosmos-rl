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

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from cosmos_rl.policy.config import (
    Config,
)
from cosmos_rl.dispatcher.run_web_panel import main as launch_dispatcher
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from datasets import load_dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os


# This dataset is used for SFT with raw text input, which is used for models like Mistral
# It converts the conversation list to string format for models requiring raw text input
# This handles cases like Mistral where conversation dicts need string conversion
# to avoid role alternation errors
class SFTRawTextDataset(Dataset):
    def __init__(self):
        # --- dataset (streaming + shuffle) ---
        base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
        num_shards = 46
        urls = [base_url.format(i=i) for i in range(num_shards)]

        dataset = load_dataset(
            "webdataset",
            data_files={"train": urls},
            split="train",
            streaming=True,
        )

        # Shuffle via buffer (DataLoader's shuffle doesn't apply to Iterable-style streams)
        dataset = dataset.shuffle(buffer_size=10000, seed=os.getenv("RANK"))

        # --- collate helper: turn PIL -> torch.Tensor, keep metadata as-is ---
        to_tensor = ToTensor()

        def collate_pil(batch):
            def convert(x):
                if isinstance(x, Image.Image):
                    x = x.convert("RGB")
                    # resize to 384x384
                    x = x.resize((384, 384))
                    return to_tensor(x)  # [C,H,W], float32 in [0,1]
                return x

            # Items are usually dicts from WebDataset (e.g., {"jpg": PIL.Image, "txt": "...", ...})
            if isinstance(batch[0], dict):
                out = {}
                for k in batch[0].keys():
                    vals = [convert(b[k]) for b in batch if k in b]
                    # Try to stack; if it fails (e.g., variable-length strings), keep as list
                    try:
                        out[k] = default_collate(vals)
                    except Exception:
                        out[k] = vals
                return out
            else:
                return default_collate([convert(b) for b in batch])

        self.collate_pil = collate_pil
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_pil,
        )
        self.iterator = iter(self.loader)

    def setup(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
    ):
        self.config = config.train.train_policy
        self.tokenizer = tokenizer
        self.column_name = self.config.conversation_column_name
        self.cache = None

    def __len__(self):
        return 649_000

    def __getitem__(self, idx):
        # get random item from loader
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


if __name__ == "__main__":

    def create_dataset(config):
        return SFTRawTextDataset()

    launch_dispatcher(
        dataset=create_dataset,
    )
    # dataset = SFTRawTextDataset()
    # next_item = dataset[0]
    # print(f"next_item: {next_item}")
