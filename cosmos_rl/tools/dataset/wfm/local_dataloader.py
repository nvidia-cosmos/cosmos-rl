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

from cosmos_rl.policy.config.wfm import (
    LocalImageDataLoaderConfig,
    LocalVideoDataLoaderConfig,
)
from cosmos_rl.tools.dataset.wfm.local_datasets.dataset_utils import (
    get_sampler,
    get_generic_dataloader,
)
from cosmos_rl.tools.dataset.wfm.local_datasets.dataset_image import ImageDataset
from cosmos_rl.tools.dataset.wfm.local_datasets.dataset_video import VideoDataset


def get_image_dataloader(config: LocalImageDataLoaderConfig):
    image_dataset = ImageDataset(
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        offline_text_embedding=config.offline_text_embedding,
        text_encoder_type=config.text_encoder_type,
    )
    dataloader = get_generic_dataloader(
        dataset=image_dataset,
        sampler=get_sampler(dataset=image_dataset),
        batch_size=config.batch_size,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return dataloader


def get_video_dataloader(config: LocalVideoDataLoaderConfig):
    video_dataset = VideoDataset(
        dataset_dir=config.dataset_dir,
        num_frames=config.num_frames,
        video_size=config.video_size,
        offline_text_embedding=config.offline_text_embedding,
        text_encoder_type=config.text_encoder_type,
    )
    dataloader = get_generic_dataloader(
        dataset=video_dataset,
        sampler=get_sampler(dataset=video_dataset),
        batch_size=config.batch_size,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return dataloader
