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


def post_hf_models_patch(hf_config: AutoConfig, model: Any):
    if (
        hf_config.model_type == "internvl_chat"
        and hasattr(hf_config, "llm_config")
        and hf_config.llm_config.model_type == "gpt_oss"
    ):
        model.img_context_token_id = 200021
        print("Set img_context_token_id to 200021")


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


def make_new_self_attn_forward(original_attn_forward):
    def self_attn_forward(self, hidden_states, *args, **kwargs):
        valid_input_len = kwargs.get("valid_input_len", None)
        if valid_input_len is not None:
            attention_mask = get_packed_attention_mask(
                valid_input_len.tolist(), hidden_states.device
            )
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


def sequence_packing_forward_llm_patch(model):
    original_forward = model.model.forward

    def sequence_packing_forward_llm_inner(*args, **kwargs):
        valid_input_len = kwargs.get("valid_input_len", None)
        if valid_input_len is not None:
            input_ids = kwargs.get("input_ids", None)
            inputs_embeds = kwargs.get("inputs_embeds", None)
            position_ids = kwargs.get("position_ids", None)

            batch_size = valid_input_len.shape[0]
            input_ids_list = []
            inputs_embeds_list = []
            position_ids_list = []
            for i in range(batch_size):
                valid_len = valid_input_len[i].item()
                if input_ids is not None:
                    cur_input_ids = input_ids[i : i + 1, :valid_len].clone()
                    input_ids_list.append(cur_input_ids)
                if inputs_embeds is not None:
                    cur_inputs_embeds = inputs_embeds[i : i + 1, :valid_len, :].clone()
                    inputs_embeds_list.append(cur_inputs_embeds)
                if position_ids is not None:
                    cur_position_ids = position_ids[i : i + 1, :valid_len].clone()
                    position_ids_list.append(cur_position_ids)

            if len(input_ids_list) > 0:
                kwargs["input_ids"] = torch.cat(input_ids_list, dim=1)
            if len(inputs_embeds_list) > 0:
                kwargs["inputs_embeds"] = torch.cat(inputs_embeds_list, dim=1)
            if len(position_ids_list) > 0:
                kwargs["position_ids"] = torch.cat(position_ids_list, dim=1)

            del (
                input_ids_list,
                inputs_embeds_list,
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
    model.model.forward = sequence_packing_forward_llm_inner

    # Replace the self_attn.forward method
    for layer in model.model.layers:
        original_attn_forward = layer.self_attn.forward
        layer.self_attn.forward = make_new_self_attn_forward(
            original_attn_forward
        ).__get__(layer.self_attn, type(layer.self_attn))

    # Replace the forward method


# In order to support sequence packing during forward passes, the forward method of the language model must be patched.
# The patching logic is model-dependent, with special handling required for Vision-Language Models (VLMs) and other architectures.
SEQUENCE_PACKING_FORWARD_PATCH_FUNCTIONS = {
    "qwen3_vl": sequence_packing_forward_qwen3_vl_patch,
    "llm": sequence_packing_forward_llm_patch,
}
