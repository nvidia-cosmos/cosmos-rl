# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F

from cosmos_rl.tools.dataset.wfm.webdataset.augmentors.augmentor import (
    Augmentor,
)


class HorizontalFlip(Augmentor):
    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        args: Optional[dict] = None,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs horizontal flipping.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        flip_enabled = getattr(self.args, "enabled", True)
        if flip_enabled:
            p = getattr(self.args, "prob", 0.5)
            coin_flip = torch.rand(1).item() > p
            for key in self.input_keys:
                if coin_flip:
                    data_dict[key] = transforms_F.hflip(data_dict[key])

        return data_dict
