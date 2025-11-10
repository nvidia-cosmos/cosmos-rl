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
from transformers import AutoConfig
from typing import Any


def pre_hf_models_patch(hf_config: AutoConfig):
    if (
        hf_config.model_type == "internvl_chat"
        and hasattr(hf_config, "llm_config")
        and hf_config.llm_config.model_type == "gpt_oss"
    ):
        hf_config.vision_config.drop_path_rate = 0.0
        print("Set drop_path_rate to 0.0")
    elif hf_config.model_type == "NemotronH_Nano_VL_V2":
        # It's hardcoded for now
        hf_config.vision_config.num_hidden_layers = 32


def post_hf_models_patch(hf_config: AutoConfig, model: Any):
    if (
        hf_config.model_type == "internvl_chat"
        and hasattr(hf_config, "llm_config")
        and hf_config.llm_config.model_type == "gpt_oss"
    ):
        model.img_context_token_id = 200021
        print("Set img_context_token_id to 200021")
    elif hf_config.model_type == "NemotronH_Nano_VL_V2":

        def patch_forward(self, **kwargs) -> torch.LongTensor:
            pixel_values = kwargs.get("pixel_values", None)
            pixel_values_videos = kwargs.get("pixel_values_videos", None)
            input_ids = kwargs.get("input_ids", None)
            attention_mask = kwargs.get("attention_mask", None)
            assert self.img_context_token_id is not None
            if pixel_values is not None or pixel_values_videos is not None:
                image_vit_embeds, video_vit_embeds = None, None
                if pixel_values is not None:
                    pixel_values = pixel_values.to(
                        dtype=self.vision_model.config.torch_dtype
                    )
                    image_vit_embeds = self.extract_feature(pixel_values)
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.to(
                        dtype=self.vision_model.config.torch_dtype
                    )
                    video_vit_embeds = self.extract_feature(pixel_values_videos)
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
                B, N, C = inputs_embeds.shape
                inputs_embeds = inputs_embeds.reshape(B * N, C)
                input_ids_copy = input_ids.reshape(B * N)
                if image_vit_embeds is not None:
                    image_mask = input_ids_copy == self.img_context_token_id
                    assert image_mask.sum() != 0
                    inputs_embeds[image_mask] = image_vit_embeds.reshape(-1, C).to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                if video_vit_embeds is not None:
                    if B > 1:
                        raise NotImplementedError(
                            "Video is not supported for batch size > 1"
                        )
                    video_mask = input_ids_copy == self.video_context_token_id
                    assert video_mask.sum() != 0
                    inputs_embeds[video_mask] = video_vit_embeds.reshape(-1, C).to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                # if video_vit_embeds is not None and self.video_pruning_rate > 0:  # EVS
                #     h = w = int(video_vit_embeds.shape[1] ** 0.5)  # assumption here (and everywhere else) is that shape is square
                #     evs_mask = EfficientVideoSampling.compute_retention_mask(
                #         video_embeds=video_vit_embeds,
                #         thw=(video_vit_embeds.shape[0], h, w),
                #         spatial_merge_size=1,  # we already work on vision embeddings, so no downsampling to follow
                #         q=self.video_pruning_rate,
                #     )
                #     print(f"pruning rate: {self.video_pruning_rate}, EVS mask: {evs_mask.sum().item()} tokens retained out of {evs_mask.numel()} total video tokens ({evs_mask.sum().item() / evs_mask.numel() * 100:.2f}%)")

                #     retention_mask = torch.ones_like(input_ids_copy, dtype=torch.bool)
                #     retention_mask[video_mask] = evs_mask.view(-1)
                #     inputs_embeds = inputs_embeds[retention_mask].unsqueeze(0)  # adding batch=1
                #     if attention_mask is not None:
                #         attention_mask = attention_mask[:, retention_mask].contiguous()
                #     if input_ids is not None:
                #         input_ids = input_ids[:, retention_mask].contiguous()
                # else:
                inputs_embeds = inputs_embeds.reshape(B, N, C)
            else:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return outputs

        model.forward = patch_forward.__get__(model, type(model))
