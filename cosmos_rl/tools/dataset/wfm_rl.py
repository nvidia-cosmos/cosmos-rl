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

from typing import Iterable, Union
from cosmos_rl.policy.config.wfm import (
    JointDataLoaderConfig,
    LocalDataLoaderConfig,
)
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.tools.dataset.wfm.joint_dataloader import (
    IterativeJointDataLoader,
)
from cosmos_rl.tools.dataset.wfm.local_dataloader import (
    get_image_dataloader,
    get_video_dataloader,
)


def get_joint_dataloader(config: JointDataLoaderConfig):
    return IterativeJointDataLoader(
        dataloaders={
            "image_data": config.image_dataloader,
            "video_data": config.video_dataloader,
        }
    )


def get_local_dataloader(config: LocalDataLoaderConfig):
    # TODO(dinghaoy): support both image and video dataloader
    if config.image_dataloader is not None and config.image_dataloader.ratio > 0:
        return get_image_dataloader(config.image_dataloader)
    elif config.video_dataloader is not None and config.video_dataloader.ratio > 0:
        return get_video_dataloader(config.video_dataloader)
    else:
        raise ValueError(
            "Either image_dataloader or video_dataloader must be provided in LocalDataLoaderConfig."
        )


def get_dataloader(
    config: Union[JointDataLoaderConfig, LocalDataLoaderConfig],
) -> Iterable:
    if config.type == "local":
        return get_local_dataloader(config)
    elif config.type == "web":
        return get_joint_dataloader(config)
    else:
        raise ValueError(f"Unknown dataloader type: {config.type}")


if __name__ == "__main__":
    launch_worker(
        dataset=None,
        dataloader=get_dataloader,
    )
