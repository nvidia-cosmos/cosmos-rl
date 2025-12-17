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


from typing import Optional, Any, List, Dict, Union
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.utils.logging import logger
from torchvision import transforms as T
import torch
import json
from decord import VideoReader, cpu
import numpy as np
import os
from PIL import Image
from cosmos_rl.tools.dataset.wfm.local_datasets.dataset_utils import (
    ResizePreprocess,
    ToTensorVideo,
)
import cosmos_rl.utils.util as util
import glob

class LocalDiffusersDataset(Dataset):
    def setup(self, config: CosmosConfig, *args, **kwargs):
        """
        This method is optional and get called by launcher after being mounted
        `config`: config;
        """
        self.config = config
        # TODO (yy): set by config
        self.is_video = True
        self.sequence_length = 41
        self.dataset_dir = self.config.train.train_policy.dataset.local_dir
        prompt_files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
        if self.is_video:
            self.suffix = 'mp4'
        else:
            self.suffix = 'jpg'
        self.visual_preprocess = self.visual_preprocess(1024)
        self.all_file_names = [os.path.basename(f).split('.')[0] for f in prompt_files]

    def __len__(self):
        return len(self.all_file_names)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        data = {}
        meta_file = os.path.join(self.dataset_dir, self.all_file_names[idx] + '.json')
        visual_path = meta_file.replace('json', self.suffix)
        if self.is_video:
            video, fps = self._get_frames(visual_path)
            video = video.permute(
                1, 0, 2, 3
            )  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Meta info
            meta_info = json.load(open(meta_file))
            data["visual"] = video
            data["is_video"] = True

            data.update(meta_info)
        else:
            image = self._get_image(visual_path)
            data["visual"] = image
            meta_info = json.load(open(meta_file))
            data["is_video"] = False

            data.update(meta_info)

        return data
    
    def visual_preprocess(self, resolution):
        if self.is_video:
            transforms = [
                    ToTensorVideo(),  # TCHW
                    ResizePreprocess([480, 832]),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
        else:
            transforms = [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize(resolution),  # Image.BICUBIC
                T.CenterCrop(resolution),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        return T.Compose(transforms)
    
    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx) if max_start_idx > 0 else 0
        end_frame = start_frame + self.sequence_length if max_start_idx > 0 else self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps
    
    def _get_frames(self, file_path):
        frames, fps = self._load_video(file_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.visual_preprocess(frames)
        return frames, fps

    def _get_image(self, file_path):
        image = Image.open(file_path)
        image = self.visual_preprocess(image)
        return image

class LocalDiffusersValDataset(Dataset):
    """
    Diffusers validation only need a couple of prompt and sampling parameters
    """

    def setup(self, config: CosmosConfig, *args, **kwargs):
        if not config.validation.enable:
            logger.warning(
                "Validation is not enabled in the config. Skipping setup for MathValDataset."
            )
            return
        local_path = config.validation.dataset.local_dir
        self.is_video = config.policy.diffusers_config.is_video
        self.prompts = json.load(open(local_path))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        data = {
            'height': 480,
            'width': 832,
            'guidance_scale': 4.5,
            'inference_step': 20,
            'prompt': self.prompts[idx]['prompt']
        }
        if self.is_video:
            data.update({'frames': 41})

        return data

class DiffusersPacker(DataPacker):
    """
    This is a demo data packer that wraps the underlying data packer of the selected model.
    This is meaningless for this example, but useful for explaining:
        - how dataset data is processed and collated into a mini-batch for rollout engine;
        - how rollout output is processed and collated into a mini-batch for policy model;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, config: CosmosConfig, *args, **kwargs):
        """
        This method is optional and get called by launcher after being mounted
        `config`: config;
        """
        super().setup(config, *args, **kwargs)

    def get_rollout_input(self, item: Any) -> Any:
        """
        Convert dataset item into what rollout engine (e.g. vllm) expects
        """
        pass

    def rollout_collate_fn(self, items: List[Any]) -> Any:
        """
        Collate the rollout inputs into a mini-batch for rollout engine
        """
        pass

    def get_policy_input(
        self,
        item: Any,
        rollout_output: Union[str, List[int]],
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """
        Process samples & rollout output before collating them into a mini-batch
        """
        pass

    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute the maximum sequence length of the mini-batch
        """
        pass

    def policy_collate_fn(
        self, processed_samples: List[Any], computed_max_len: int
    ) -> Dict[str, Any]:
        """
        Collate the mini-batch into the kwargs required by the policy model
        """
        pass

    def sft_process_sample(self, sample):
        return sample

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        is_validation: int = False
    ) -> Dict[str, Any]:
        if not is_validation:
            batch_image = [sample["visual"] for sample in processed_samples]
            batch_prompt = [sample["prompt"] for sample in processed_samples]
            return {"visual": batch_image, "prompt": batch_prompt}
        else:
            return processed_samples

if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        dataset = LocalDiffusersDataset()
        return dataset

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        val_dataset = LocalDiffusersValDataset()
        return val_dataset

    launch_worker(
        dataset=get_dataset,
        val_dataset=get_val_dataset,
        # Optional: if not provided, the default data packer of the selected model will be used
        data_packer=DiffusersPacker(),
        val_data_packer=DiffusersPacker(),
    )
