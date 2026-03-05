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

"""
Video2World conditioner for Cosmos Policy.

Mutable (``frozen=False``) condition dataclasses and conditioner classes
ported from cosmos-policy.  The frozen counterparts live in ``condition.py``;
these mutable versions are needed because the policy model mutates condition
objects in-place during inference (proprio injection, mask manipulation, etc.).

Key differences from the frozen ``Vid2VidCondition`` / ``Vid2VidConditioner``:

* Condition dataclasses are **mutable** so ``gt_frames``,
  ``condition_video_input_mask_B_C_T_H_W``, etc. can be reassigned.
* ``Video2WorldCondition.set_video_condition`` adds
  ``conditional_frames_probs`` for weighted sampling of the number of
  conditioning frames.
* ``Video2WorldConditioner.get_condition_uncondition`` returns ``None`` for
  the uncondition when every embedder's dropout rate is 0, avoiding an
  unnecessary forward pass.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch

from cosmos_rl.utils.wfm.utils import DataType
from cosmos_rl.policy.model.wfm.conditioner import GeneralConditioner
from cosmos_rl.policy.model.wfm.conditioner.emb_models import TextAttr


# ---------------------------------------------------------------------------
# Mutable condition hierarchy (mirrors frozen T2VCondition / Vid2VidCondition)
# ---------------------------------------------------------------------------


@dataclass(frozen=False)
class MutableT2VCondition:
    """Mutable counterpart of the frozen :class:`T2VCondition` in ``condition.py``."""

    _is_broadcasted: bool = False
    crossattn_emb: Optional[torch.Tensor] = None
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None

    def to_dict(self, skip_underscore: bool = True) -> Dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not (f.name.startswith("_") and skip_underscore)
        }

    @property
    def is_broadcasted(self) -> bool:
        return self._is_broadcasted

    def edit_data_type(self, data_type: DataType) -> "MutableT2VCondition":
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["data_type"] = data_type
        return type(self)(**kwargs)

    @property
    def is_video(self) -> bool:
        return self.data_type == DataType.VIDEO


@dataclass(frozen=False)
class MutableVid2VidCondition(MutableT2VCondition):
    """Mutable counterpart of the frozen :class:`Vid2VidCondition` in ``condition.py``."""

    use_video_condition: bool = False
    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
    ) -> "MutableVid2VidCondition":
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        B, _, T, H, W = gt_frames.shape
        mask = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )

        if T == 1:
            ncf_B = torch.zeros(B, dtype=torch.int32)
        else:
            if num_conditional_frames is not None:
                if isinstance(num_conditional_frames, torch.Tensor):
                    ncf_B = (
                        torch.ones(B, dtype=torch.int32) * num_conditional_frames.cpu()
                    )
                else:
                    ncf_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
            else:
                ncf_B = torch.randint(
                    random_min_num_conditional_frames,
                    random_max_num_conditional_frames + 1,
                    size=(B,),
                )

        for idx in range(B):
            mask[idx, :, : ncf_B[idx], :, :] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = mask
        return type(self)(**kwargs)

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "MutableVid2VidCondition":
        cond = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:
            cond.use_video_condition = True
        return cond


# ---------------------------------------------------------------------------
# Video2World condition extensions
# ---------------------------------------------------------------------------


class Video2WorldCondition(MutableVid2VidCondition):
    """Extends :class:`MutableVid2VidCondition` with ``conditional_frames_probs``
    support in :meth:`set_video_condition`."""

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "Video2WorldCondition":
        if conditional_frames_probs is not None and num_conditional_frames is None:
            B = gt_frames.shape[0]
            options = list(conditional_frames_probs.keys())
            weights = list(conditional_frames_probs.values())
            sampled = random.choices(options, weights=weights, k=B)
            kwargs = self.to_dict(skip_underscore=False)
            kwargs["gt_frames"] = gt_frames
            _, _, T, H, W = gt_frames.shape
            mask = torch.zeros(
                B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
            )
            ncf_B = torch.tensor(sampled, dtype=torch.int32)
            for idx in range(B):
                mask[idx, :, : ncf_B[idx], :, :] += 1
            kwargs["condition_video_input_mask_B_C_T_H_W"] = mask
            return type(self)(**kwargs)

        return super().set_video_condition(
            gt_frames=gt_frames,
            random_min_num_conditional_frames=random_min_num_conditional_frames,
            random_max_num_conditional_frames=random_max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )


class Video2WorldConditionV2(Video2WorldCondition):
    """Variant that zeros out conditional frames when ``use_video_condition``
    is False (unconditional branch in CFG)."""

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "Video2WorldConditionV2":
        num_conditional_frames = (
            0 if not self.use_video_condition else num_conditional_frames
        )
        return super().set_video_condition(
            gt_frames=gt_frames,
            random_min_num_conditional_frames=random_min_num_conditional_frames,
            random_max_num_conditional_frames=random_max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=conditional_frames_probs,
        )

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Video2WorldConditionV2":
        del is_cfg_conditional
        return MutableVid2VidCondition.set_video_condition(
            self,
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )


# ---------------------------------------------------------------------------
# Conditioners
# ---------------------------------------------------------------------------


class Video2WorldConditioner(GeneralConditioner):
    """Conditioner that produces :class:`Video2WorldCondition` objects.

    Overrides ``get_condition_uncondition`` to skip the uncondition forward
    pass when every embedder has a dropout rate of 0 (always-conditional
    generation instead of CFG).
    """

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Video2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return Video2WorldCondition(**output)

    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        cond_rates, uncond_rates = {}, {}
        for name, embedder in self.embedders.items():
            cond_rates[name] = 0.0
            uncond_rates[name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        condition = self(data_batch, override_dropout_rate=cond_rates)
        if cond_rates == uncond_rates:
            return condition, None
        uncondition = self(data_batch, override_dropout_rate=uncond_rates)
        return condition, uncondition

    def get_condition_with_negative_prompt(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        cond_rates, uncond_rates = {}, {}
        for name, embedder in self.embedders.items():
            cond_rates[name] = 0.0
            if isinstance(embedder, TextAttr):
                uncond_rates[name] = 0.0
            else:
                uncond_rates[name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        data_neg = copy.deepcopy(data_batch)
        if "neg_t5_text_embeddings" in data_neg and isinstance(
            data_neg["neg_t5_text_embeddings"], torch.Tensor
        ):
            data_neg["t5_text_embeddings"] = data_neg["neg_t5_text_embeddings"]

        condition = self(data_batch, override_dropout_rate=cond_rates)
        uncondition = self(data_neg, override_dropout_rate=uncond_rates)
        return condition, uncondition


class Video2WorldConditionerV2(GeneralConditioner):
    """Conditioner that produces :class:`Video2WorldConditionV2` objects."""

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Video2WorldConditionV2:
        output = super()._forward(batch, override_dropout_rate)
        return Video2WorldConditionV2(**output)
