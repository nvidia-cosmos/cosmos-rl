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
from typing import Any
from transformers import AutoConfig
from cosmos_rl.utils.logging import logger


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
        # Set video pruning rate to 0 for training
        hf_config.video_pruning_rate = 0.0


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
                    # if B > 1:
                    #     raise NotImplementedError(
                    #         "Video is not supported for batch size > 1"
                    #     )
                    video_mask = input_ids_copy == self.video_context_token_id
                    assert video_mask.sum() != 0
                    inputs_embeds[video_mask] = video_vit_embeds.reshape(-1, C).to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
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


# Get packed attention mask
def get_packed_attention_mask(lengths, device):
    # lengths: list of sequence lengths
    L = sum(lengths)
    mask = torch.zeros((L, L), dtype=torch.bool, device=device)
    offset = 0
    for length in lengths:
        mask[offset : offset + length, offset : offset + length] = torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=device)
        )
        offset += length
    return mask


# Get packed sliding attention mask
def get_packed_sliding_attention_mask(lengths, sliding_window, device):
    # lengths: list of sequence lengths
    L = sum(lengths)
    mask = torch.zeros((1, 1, L, L), dtype=torch.bool, device=device)
    cur = 0

    for L in lengths:
        start = cur
        end = cur + L
        for i in range(start, end):
            valid_start = max(start, i - sliding_window + 1)
            valid_end = i + 1
            mask[0, 0, i, valid_start:valid_end] = True
        cur = end
    return mask


def make_new_self_attn_forward(original_attn_forward):
    def self_attn_forward(self, hidden_states, *args, **kwargs):
        valid_input_len = kwargs.get("valid_input_len", None)
        if valid_input_len is not None:
            attention_mask_cache = kwargs.get("attention_mask_cache")
            if hasattr(self, "is_sliding") and self.is_sliding:
                assert (
                    hasattr(self, "sliding_window") and self.sliding_window is not None
                ), "sliding_window must be set for sliding attention"
                attention_mask = attention_mask_cache.get(
                    f"sliding_attention_mask_{self.sliding_window}", None
                )
                if attention_mask is None:
                    attention_mask = get_packed_sliding_attention_mask(
                        valid_input_len.tolist(),
                        self.sliding_window,
                        hidden_states.device,
                    )
                    attention_mask_cache[
                        f"sliding_attention_mask_{self.sliding_window}"
                    ] = attention_mask
            else:
                attention_mask = attention_mask_cache.get("full_attention_mask", None)
                if attention_mask is None:
                    attention_mask = get_packed_attention_mask(
                        valid_input_len.tolist(), hidden_states.device
                    )
                    attention_mask_cache["full_attention_mask"] = attention_mask
            kwargs["attention_mask"] = attention_mask

        return original_attn_forward(hidden_states, *args, **kwargs)

    return self_attn_forward


def sequence_packing_forward_patch(hf_config: AutoConfig, hfmodel):
    patch_success = False
    try:
        if hf_config.model_type in SEQUENCE_PACKING_FORWARD_PATCH_FUNCTIONS:
            SEQUENCE_PACKING_FORWARD_PATCH_FUNCTIONS[hf_config.model_type](
                hfmodel.model
            )
            patch_success = True
        else:
            if not hfmodel.is_vlm:
                SEQUENCE_PACKING_FORWARD_PATCH_FUNCTIONS["llm"](hfmodel.model)
                patch_success = True
            else:
                logger.warning(
                    f"Failed to patch sequence packing forward for {hf_config.model_type}, supported models: {SEQUENCE_PACKING_FORWARD_PATCH_FUNCTIONS.keys()}"
                )
    except Exception as e:
        logger.error(f"Failed to patch sequence packing forward: {e}")
    return patch_success


def sequence_packing_forward_qwen3_vl_patch(model):
    original_forward = model.language_model.forward

    def sequence_packing_forward_qwen3_vl_inner(*args, **kwargs):
        valid_input_len = kwargs.get("valid_input_len", None)
        if valid_input_len is not None:
            inputs_embeds = kwargs.get("inputs_embeds")
            visual_pos_masks = kwargs.get("visual_pos_masks", None)
            position_ids = kwargs.get("position_ids", None)

            batch_size = valid_input_len.shape[0]
            inputs_embeds_list = []
            visual_pos_masks_list = []
            position_ids_list = []
            for i in range(batch_size):
                valid_len = valid_input_len[i].item()
                cur_inputs_embeds = inputs_embeds[i : i + 1, :valid_len, :].clone()
                inputs_embeds_list.append(cur_inputs_embeds)

                if visual_pos_masks is not None:
                    cur_visual_mask = visual_pos_masks[i : i + 1, :valid_len].clone()
                    visual_pos_masks_list.append(cur_visual_mask)

                if position_ids is not None:
                    cur_position_ids = position_ids[:, i : i + 1, :valid_len].clone()
                    position_ids_list.append(cur_position_ids)

            kwargs["inputs_embeds"] = torch.cat(inputs_embeds_list, dim=1)
            if len(visual_pos_masks_list) > 0:
                kwargs["visual_pos_masks"] = torch.cat(visual_pos_masks_list, dim=1)

            if len(position_ids_list) > 0:
                kwargs["position_ids"] = torch.cat(position_ids_list, dim=2)
            # Clear attention mask cache
            kwargs["attention_mask_cache"] = {}

            del (
                inputs_embeds_list,
                visual_pos_masks_list,
                position_ids_list,
            )
        else:
            logger.warning(
                "valid_input_len is not provided, skip sequence packing forward"
            )
        # Call original forward
        result = original_forward(*args, **kwargs)
        return result

    # Replace the forward method
    model.language_model.forward = sequence_packing_forward_qwen3_vl_inner

    # Replace the self_attn.forward method
    for layer in model.language_model.layers:
        original_attn_forward = layer.self_attn.forward
        layer.self_attn.forward = make_new_self_attn_forward(
            original_attn_forward
        ).__get__(layer.self_attn, type(layer.self_attn))


def sequence_packing_forward_gemma3_vl_patch(model):
    original_forward = model.model.forward

    def sequence_packing_forward_gemma3_vl_inner(*args, **kwargs):
        valid_input_len = kwargs.get("valid_input_len", None)
        if valid_input_len is not None:
            input_ids = kwargs.get("input_ids")
            batch_size = valid_input_len.shape[0]

            input_ids_list = []
            cache_position_list = []
            for i in range(batch_size):
                valid_len = valid_input_len[i].item()
                cur_input_ids = input_ids[i : i + 1, :valid_len].clone()
                input_ids_list.append(cur_input_ids)
                cache_position_list.append(
                    torch.arange(0, valid_len, device=input_ids.device)
                )
            kwargs["input_ids"] = torch.cat(input_ids_list, dim=1)
            kwargs["cache_position"] = torch.cat(cache_position_list, dim=0)
            # Clear attention mask cache
            kwargs["attention_mask_cache"] = {}
        return original_forward(*args, **kwargs)

    model.model.forward = sequence_packing_forward_gemma3_vl_inner

    # Replace the self_attn.forward method
    for layer in model.language_model.layers:
        original_attn_forward = layer.self_attn.forward
        layer.self_attn.forward = make_new_self_attn_forward(
            original_attn_forward
        ).__get__(layer.self_attn, type(layer.self_attn))


def sequence_packing_forward_llm_patch(model):
    original_forward = model.model.forward

    def sequence_packing_forward_llm_inner(*args, **kwargs):
        valid_input_len = kwargs.get("valid_input_len", None)
        if valid_input_len is not None:
            input_ids = kwargs.get("input_ids")
            batch_size = valid_input_len.shape[0]

            input_ids_list = []
            cache_position_list = []
            for i in range(batch_size):
                valid_len = valid_input_len[i].item()
                cur_input_ids = input_ids[i : i + 1, :valid_len].clone()
                input_ids_list.append(cur_input_ids)
                cache_position_list.append(
                    torch.arange(0, valid_len, device=input_ids.device)
                )

            kwargs["input_ids"] = torch.cat(input_ids_list, dim=1)
            kwargs["cache_position"] = torch.cat(cache_position_list, dim=0)
            kwargs["position_ids"] = kwargs["cache_position"].clone().unsqueeze(0)
            # Clear attention mask cache
            kwargs["attention_mask_cache"] = {}
            del (
                input_ids_list,
                cache_position_list,
            )
        else:
            logger.warning(
                "valid_input_len is not provided, skip sequence packing forward"
            )
        # Call original forward
        result = original_forward(*args, **kwargs)
        return result

    # Replace the forward method
    model.model.forward = sequence_packing_forward_llm_inner

    # Replace the self_attn.forward method
    for layer in model.model.layers:
        original_attn_forward = layer.self_attn.forward
        layer.self_attn.forward = make_new_self_attn_forward(
            original_attn_forward
        ).__get__(layer.self_attn, type(layer.self_attn))


# In order to support sequence packing during forward passes, the forward method of the language model must be patched.
# The patching logic is model-dependent, with special handling required for Vision-Language Models (VLMs) and other architectures.
SEQUENCE_PACKING_FORWARD_PATCH_FUNCTIONS = {
    "qwen3_vl": sequence_packing_forward_qwen3_vl_patch,
    "gemma3": sequence_packing_forward_gemma3_vl_patch,
    "llm": sequence_packing_forward_llm_patch,
}
