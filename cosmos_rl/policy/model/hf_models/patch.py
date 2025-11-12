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


# Copied from https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/blob/main/evs.py
class EfficientVideoSampling:
    @staticmethod
    def compute_retention_mask(
        *,
        video_embeds: torch.FloatTensor,
        thw: torch.LongTensor,
        spatial_merge_size: int,
        q: float,
    ):
        """
        Computes the retention mask for video embeddings based on the grid dimensions.
        Args:
            video_embeds (`torch.FloatTensor` of shape `(T * H * W, hidden_size)`):
                The video embeddings to compute the retention mask for.
            thw (`torch.LongTensor` of shape `(3)`):
                The temporal, height and width of feature shape of each video in LLM.
            spatial_merge_size (`int`): The spatial merge size of the video embeddings.
                If embeddings will be downsampled *later*, this should be the downsampling factor.
            q: (`float`): Pruning rate factor, indicating number of tokens to prune (remove)
        Returns:
            `torch.Tensor`: The retention mask for the video embeddings (T * H * W).
                1 for tokens to keep, 0 for tokens to prune.
        """
        T, H, W = thw

        # video_embeds = einops.rearrange(
        #     video_embeds,
        #     "(T H W) C -> T H W C",
        #     T=T,
        #     H=H // spatial_merge_size,
        #     W=W // spatial_merge_size,
        # )
        # Use reshape instead of einops to avoid graph breaks
        video_embeds = video_embeds.reshape(
            T, H // spatial_merge_size, W // spatial_merge_size, video_embeds.size(-1)
        )

        # Core EVS
        similarity = torch.nn.functional.cosine_similarity(
            video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
        )
        dissimilarity = 1 - similarity

        # Always ensure we include all tokens from the first frame
        dissimilarity = torch.cat(
            [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
        )
        dissimilarity_flat = dissimilarity.view(-1)

        min_num_tokens = (H // spatial_merge_size) * (
            W // spatial_merge_size
        )  # a single frame
        evs_num_tokens = int(T * min_num_tokens * (1 - q))
        num_tokens_to_keep = max(min_num_tokens, evs_num_tokens)

        order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
        topk_indices = order[:num_tokens_to_keep]

        retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
        retention_mask[topk_indices] = True
        retention_mask = retention_mask.reshape(dissimilarity.size())

        mask = retention_mask.view(-1)  # "T H W -> (T H W)"
        return mask


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
        # Set video pruning rate for efficient inference
        # model.video_pruning_rate = 0.75
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
                if video_vit_embeds is not None and self.video_pruning_rate > 0:  # EVS
                    h = w = int(
                        video_vit_embeds.shape[1] ** 0.5
                    )  # assumption here (and everywhere else) is that shape is square
                    evs_mask = EfficientVideoSampling.compute_retention_mask(
                        video_embeds=video_vit_embeds,
                        thw=(video_vit_embeds.shape[0], h, w),
                        spatial_merge_size=1,  # we already work on vision embeddings, so no downsampling to follow
                        q=self.video_pruning_rate,
                    )
                    # print(f"pruning rate: {self.video_pruning_rate}, EVS mask: {evs_mask.sum().item()} tokens retained out of {evs_mask.numel()} total video tokens ({evs_mask.sum().item() / evs_mask.numel() * 100:.2f}%)")

                    retention_mask = torch.ones_like(input_ids_copy, dtype=torch.bool)
                    retention_mask[video_mask] = evs_mask.view(-1)
                    inputs_embeds = inputs_embeds[retention_mask].unsqueeze(
                        0
                    )  # adding batch=1
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, retention_mask].contiguous()
                    if input_ids is not None:
                        input_ids = input_ids[:, retention_mask].contiguous()
                else:
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
