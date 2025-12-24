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


from typing import Any, List, Dict, Union

import torch


from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.tools.dataset.wfm.data_sources.mock_data import (
    LambdaDataset,
    CombinedDictDataset,
)
import uuid


def get_dummy_visual_dataset(
    config: CosmosConfig,
    **kwargs,
):
    is_video = config.policy.diffusers_config.is_video
    height, width = config.policy.diffusers_config.inference_size
    train_frames = config.policy.diffusers_config.train_frames

    def visual_fn():
        if is_video:
            # Video tensor should be (C, T, H, W) and range from -1.0, 1.0
            return (
                2.0 * torch.rand(3, train_frames, height, width).to(torch.float32) - 1.0
            )
        else:
            # Image tensor should be (C, H, W) and range from -1.0 to 1.0
            return 2.0 * torch.rand(3, height, width)

    return CombinedDictDataset(
        **{
            "visual": LambdaDataset(visual_fn),
            "prompt": LambdaDataset(lambda: ""),
        }
    )


def get_dummy_validation_dataset(
    config: CosmosConfig,
    **kwargs,
):
    is_video = config.policy.diffusers_config.is_video
    height, width = config.policy.diffusers_config.inference_size
    infernece_frame = config.policy.diffusers_config.inference_frames
    inference_step = config.policy.diffusers_config.inference_step
    guidance_scale = config.policy.diffusers_config.guidance_scale

    dict_dataset = {
        "height": LambdaDataset(lambda: height, length=32),
        "width": LambdaDataset(lambda: width, length=32),
        "inference_step": LambdaDataset(lambda: inference_step, length=32),
        "guidance_scale": LambdaDataset(lambda: guidance_scale, length=32),
        "prompt": LambdaDataset(lambda: str(uuid.uuid4()), length=32),
    }
    if is_video:
        dict_dataset["frame"] = LambdaDataset(lambda: infernece_frame, length=32)
    return CombinedDictDataset(**dict_dataset)


class DiffusersPacker(BaseDataPacker):
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
        self, processed_samples: List[Dict[str, Any]], is_validation: int = False
    ) -> Dict[str, Any]:
        if not is_validation:
            batch_image = [sample["visual"] for sample in processed_samples]
            batch_prompt = [sample["prompt"] for sample in processed_samples]
            return {"visual": batch_image, "prompt": batch_prompt}
        else:
            return processed_samples


if __name__ == "__main__":
    launch_worker(
        dataset=get_dummy_visual_dataset,
        val_dataset=get_dummy_validation_dataset,
        # Optional: if not provided, the default data packer of the selected model will be used
        data_packer=DiffusersPacker(),
        val_data_packer=DiffusersPacker(),
    )
