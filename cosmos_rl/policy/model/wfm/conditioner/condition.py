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

import torch
from torch.distributed import ProcessGroup

from abc import ABC
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from cosmos_rl.utils.wfm.utils import DataType
from cosmos_rl.utils.wfm.context_parallel import (
    broadcast,
    broadcast_split_tensor,
)


@dataclass(frozen=True)
class BaseCondition(ABC):
    """
    Attributes:
        _is_broadcasted: Flag indicating if parallel broadcast splitting
            has been performed. This is an internal implementation detail.
    """

    _is_broadcasted: bool = False

    def to_dict(self, skip_underscore: bool = True) -> Dict[str, Any]:
        """Converts the condition to a dictionary.

        Returns:
            Dictionary containing the condition's fields and values.
        """
        # return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not (f.name.startswith("_") and skip_underscore)
        }

    @property
    def is_broadcasted(self) -> bool:
        return self._is_broadcasted

    def broadcast(self, process_group: torch.distributed.ProcessGroup):
        """Broadcasts and splits the condition across the checkpoint parallelism group.
        For most condition, such asT2VCondition, we do not need split.

        Args:
            process_group: The process group for broadcast and split

        Returns:
            A new BaseCondition instance with the broadcasted and split condition.
        """
        if self.is_broadcasted:
            return self
        return broadcast_condition(self, process_group)


@dataclass(frozen=True)
class T2VCondition(BaseCondition):
    crossattn_emb: Optional[torch.Tensor] = None
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None

    def edit_data_type(self, data_type: DataType):
        """Edit the data type of the condition.

        Args:
            data_type: The new data type.

        Returns:
            A new T2VCondition instance with the new data type.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["data_type"] = data_type
        return type(self)(**kwargs)

    @property
    def is_video(self) -> bool:
        return self.data_type == DataType.VIDEO


@dataclass(frozen=True)
class Vid2VidCondition(T2VCondition):
    use_video_condition: bool = False
    # the following two attributes are used to set the video condition; during training, inference
    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
    ) -> "Vid2VidCondition":
        """
        Sets the video conditioning frames for video-to-video generation.

        This method creates a conditioning mask for the input video frames that determines
        which frames will be used as context frames for generating new frames. The method
        handles both image batches (T=1) and video batches (T>1) differently.

        Args:
            gt_frames: A tensor of ground truth frames with shape [B, C, T, H, W], where:
                B = batch size
                C = number of channels
                T = number of frames
                H = height
                W = width

            random_min_num_conditional_frames: Minimum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            random_max_num_conditional_frames: Maximum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            num_conditional_frames: Optional; If provided, all examples in the batch will use
                exactly this many frames for conditioning. If None, a random number of frames
                between random_min_num_conditional_frames and random_max_num_conditional_frames
                will be selected for each example in the batch.

        Returns:
            A new Vid2VidCondition object with the gt_frames and conditioning mask set.
            The conditioning mask (condition_video_input_mask_B_C_T_H_W) is a binary tensor
            of shape [B, 1, T, H, W] where 1 indicates frames used for conditioning and 0
            indicates frames to be generated.

        Notes:
            - For image batches (T=1), no conditioning frames are used (num_conditional_frames_B = 0).
            - For video batches:
                - If num_conditional_frames is provided, all examples use that fixed number of frames.
                - Otherwise, each example randomly uses between random_min_num_conditional_frames and
                random_max_num_conditional_frames frames.
            - The mask marks the first N frames as conditioning frames (set to 1) for each example.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        # condition_video_input_mask_B_C_T_H_W
        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )
        if T == 1:  # handle image batch
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:  # handle video batch
            if num_conditional_frames is not None:
                num_conditional_frames_B = (
                    torch.ones(B, dtype=torch.int32) * num_conditional_frames
                )
            else:
                num_conditional_frames_B = torch.randint(
                    random_min_num_conditional_frames,
                    random_max_num_conditional_frames + 1,
                    size=(B,),
                )
        for idx in range(B):
            condition_video_input_mask_B_C_T_H_W[
                idx, :, : num_conditional_frames_B[idx], :, :
            ] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = (
            condition_video_input_mask_B_C_T_H_W
        )
        return type(self)(**kwargs)

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Vid2VidCondition":
        _condition = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:
            # Do not use classifier free guidance on conditional frames.
            # YB found that it leads to worse results.
            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(
        self, process_group: torch.distributed.ProcessGroup
    ) -> "Vid2VidCondition":
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        new_condition = T2VCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(
                    gt_frames, seq_dim=2, process_group=process_group
                )
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W,
                    seq_dim=2,
                    process_group=process_group,
                )
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = (
            condition_video_input_mask_B_C_T_H_W
        )
        return type(self)(**kwargs)


def broadcast_condition(condition, process_group: Optional[ProcessGroup] = None):
    """
    Broadcast the condition from the minimum rank in the specified group(s).
    """
    if condition.is_broadcasted:
        return condition

    kwargs = condition.to_dict(skip_underscore=False)
    for key, value in kwargs.items():
        if value is not None:
            kwargs[key] = broadcast(value, process_group)
    kwargs["_is_broadcasted"] = True
    return type(condition)(**kwargs)
