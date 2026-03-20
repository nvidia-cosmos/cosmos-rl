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

import importlib
import threading

import torch
import transformers
from typing import Any, Optional
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.utils.import_utils import (
    is_torchdynamo_compiling,
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

from cosmos_rl.utils.logging import logger

_EXPECTED_TRANSFORMERS_VERSION = "4.57.6"


def pre_hf_models_patch(hf_config: AutoConfig):
    # Set attention implementation to flash_attention_2 by default
    if hasattr(hf_config, "_attn_implementation"):
        hf_config._attn_implementation = "flash_attention_2"

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
    elif hf_config.model_type in ["qwen3_5", "qwen3_5_moe"]:
        if transformers.__version__ < "5.4.0":
            # Qwen3.5 models can encounter illegal memory access errors when using Flash Attention with transformers versions earlier than 5.4.0.
            # This was resolved in transformers PR #44399, so for older versions we force the use of SDPA.
            hf_config._attn_implementation = "sdpa"
            logger.warning(
                "Qwen3.5/Qwen3.5-MoE models can encounter illegal memory access errors when using Flash Attention with transformers versions earlier than 5.4.0. "
                "So for older versions we force the use of SDPA."
            )


def post_hf_models_patch(hf_config: AutoConfig, model: Any):
    if (
        hf_config.model_type == "internvl_chat"
        and hasattr(hf_config, "llm_config")
        and hf_config.llm_config.model_type == "gpt_oss"
    ):
        model.img_context_token_id = 200021
        print("Set img_context_token_id to 200021")
    elif hf_config.model_type == "qwen3_vl":
        if hasattr(model, "model") and hasattr(
            getattr(model.model, "visual", None), "config"
        ):
            visual_forward_qwen3_vl_patch(model.model)
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
    attn_implementation = getattr(hfmodel.model.config, "_attn_implementation", "")
    if not attn_implementation.startswith("flash_attention"):
        hfmodel.model.set_attn_implementation("flash_attention_2")
        logger.warning(
            f"Model {hf_config.model_type} is not using flash attention by default for sequence packing, switched to flash_attention_2. "
            f"Original attn implementation: {attn_implementation}"
        )

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
    language_model = getattr(model, "language_model", None) or getattr(
        model.model, "language_model", None
    )
    if language_model is None:
        raise ValueError("language_model not found")
    original_forward = language_model.forward

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
    language_model.forward = sequence_packing_forward_qwen3_vl_inner

    # Replace the self_attn.forward method
    for layer in language_model.layers:
        original_attn_forward = layer.self_attn.forward
        layer.self_attn.forward = make_new_self_attn_forward(
            original_attn_forward
        ).__get__(layer.self_attn, type(layer.self_attn))


def sequence_packing_forward_qwen3_5_patch(model):
    if not is_causal_conv1d_available():
        raise ImportError(
            "Qwen3.5 sequence packing requires causal_conv1d. "
            "Install with: pip install causal-conv1d"
        )
    if not is_flash_linear_attention_available():
        raise ImportError(
            "Qwen3.5 sequence packing requires flash-linear-attention (fla>=0.2.2). "
            "Install with: pip install flash-linear-attention"
        )
    original_forward = model.model.language_model.forward

    def sequence_packing_forward_qwen3_5_inner(*args, **kwargs):
        valid_input_len = kwargs.get("valid_input_len")
        if valid_input_len is not None:
            kwargs["valid_input_len"] = valid_input_len
            inputs_embeds = kwargs.get("inputs_embeds")
            position_ids = kwargs.get("position_ids", None)
            batch_size = valid_input_len.shape[0]
            inputs_embeds_list = []
            position_ids_list = []
            for i in range(batch_size):
                valid_len = valid_input_len[i].item()
                cur_inputs_embeds = inputs_embeds[i : i + 1, :valid_len, :].clone()
                inputs_embeds_list.append(cur_inputs_embeds)
                if position_ids is not None:
                    cur_position_ids = position_ids[:, i : i + 1, :valid_len].clone()
                    position_ids_list.append(cur_position_ids)
            kwargs["inputs_embeds"] = torch.cat(inputs_embeds_list, dim=1)
            if len(position_ids_list) > 0:
                kwargs["position_ids"] = torch.cat(position_ids_list, dim=2)
            # Clear attention mask cache
            kwargs["attention_mask_cache"] = {}
            del (
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
    model.model.language_model.forward = sequence_packing_forward_qwen3_5_inner

    # Replace the self_attn.forward and linear_attn (sequence packing) for each layer
    for layer in model.model.language_model.layers:
        if layer.layer_type == "full_attention":
            original_attn_forward = layer.self_attn.forward
            layer.self_attn.forward = make_new_self_attn_forward(
                original_attn_forward
            ).__get__(layer.self_attn, type(layer.self_attn))
        elif layer.layer_type == "linear_attention":
            _patch_linear_attn_for_sequence_packing(layer)


# Thread-local storage for decoder kwargs during forward (avoids circular refs that break model.train())
_decoder_kwargs_tls = threading.local()


def _patch_linear_attn_for_sequence_packing(decoder_layer):
    """
    Patch Qwen3.5 linear_attn (GatedDeltaNet) so sequence packing works.
    Uses thread-local for kwargs to avoid circular references that cause RecursionError in model.train().
    """
    linear_attn = decoder_layer.linear_attn

    # (1) Decoder layer forward: set current kwargs in thread-local so linear_attn can read them
    _original_decoder_forward = decoder_layer.forward

    def _decoder_forward_with_kwargs(self, *args, **kwargs):
        prev = getattr(_decoder_kwargs_tls, "current", None)
        _decoder_kwargs_tls.current = kwargs
        try:
            return _original_decoder_forward(*args, **kwargs)
        finally:
            _decoder_kwargs_tls.current = prev

    decoder_layer.forward = _decoder_forward_with_kwargs.__get__(
        decoder_layer, type(decoder_layer)
    )

    # (2) linear_attn.forward: read kwargs from thread-local, build seq_idx/cu_seqlens, set on self
    _original_linear_forward = linear_attn.forward

    def _linear_attn_forward_with_packing(
        self, hidden_states, cache_params=None, attention_mask=None
    ):
        kwargs = getattr(_decoder_kwargs_tls, "current", {}) or {}
        valid_input_len = kwargs.get("valid_input_len", None)
        seq_idx = None
        cu_seqlens = None
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed = (
            cache_params is not None
            and getattr(cache_params, "has_previous_state", False)
            and seq_len == 1
        )
        if valid_input_len is not None and not use_precomputed and batch_size == 1:
            valid_list = valid_input_len.tolist()
            if sum(valid_list) == seq_len:
                segment_ids = torch.cat(
                    [
                        torch.full(
                            (L,), i, dtype=torch.int32, device=hidden_states.device
                        )
                        for i, L in enumerate(valid_list)
                    ],
                    dim=0,
                )
                seq_idx = segment_ids.unsqueeze(0)
                cu_seqlens = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
                        valid_input_len.cumsum(0),
                    ],
                    dim=0,
                )
        self._seq_pack_seq_idx = seq_idx
        self._seq_pack_cu_seqlens = cu_seqlens
        try:
            return _original_linear_forward(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
            )
        finally:
            del self._seq_pack_seq_idx
            del self._seq_pack_cu_seqlens

    linear_attn.forward = _linear_attn_forward_with_packing.__get__(
        linear_attn, type(linear_attn)
    )

    # (3) causal_conv1d_fn: inject seq_idx when set
    _original_causal_conv1d_fn = linear_attn.causal_conv1d_fn
    if _original_causal_conv1d_fn is not None:

        def _wrapped_causal_conv1d_fn(*args, **fn_kwargs):
            seq_idx = getattr(linear_attn, "_seq_pack_seq_idx", None)
            if seq_idx is not None:
                fn_kwargs["seq_idx"] = seq_idx
            return _original_causal_conv1d_fn(*args, **fn_kwargs)

        linear_attn.causal_conv1d_fn = _wrapped_causal_conv1d_fn

    # (4) chunk_gated_delta_rule: inject cu_seqlens when set
    _original_chunk_rule = linear_attn.chunk_gated_delta_rule

    def _wrapped_chunk_gated_delta_rule(*args, **fn_kwargs):
        cu_seqlens = getattr(linear_attn, "_seq_pack_cu_seqlens", None)
        if cu_seqlens is not None:
            fn_kwargs["cu_seqlens"] = cu_seqlens
        return _original_chunk_rule(*args, **fn_kwargs)

    linear_attn.chunk_gated_delta_rule = _wrapped_chunk_gated_delta_rule


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


def visual_forward_qwen3_vl_patch(model):
    """Monkey-patch a ``Qwen3VLModel`` **instance's** forward with two improvements:

    1. **Merged image+video ViT pass**: When a batch contains both images and
       videos, concatenate their pixels and run a single ``get_image_features``
       call instead of separate image/video calls.  (Follows the NemotronVL
       pattern in ``modeling_nemotron_vl_h.py:2135-2152``.)

    2. **Dummy visual forward for pure-text batches**: Under FSDP, every rank
       must call ``self.visual(...)`` each forward step so that collective
       all-gather operations stay in sync.  When a batch contains only text,
       a lightweight dummy image (16x16 zeros) is pushed through the full
       ViT -> merger -> deepstack pipeline, then outputs are sliced to ``[0:0]``
       so they carry ``grad_fn`` but contribute no features.

    Args:
        model: The ``Qwen3VLModel`` instance (i.e. ``model.model`` when
            the outer model is ``Qwen3VLForConditionalGeneration``).
    """
    if transformers.__version__ != _EXPECTED_TRANSFORMERS_VERSION:
        logger.warning(
            "visual_forward_qwen3_vl_patch was written for transformers==%s, "
            "but found transformers==%s. The patched forward may be incompatible "
            "with the installed version — verify Qwen3VLModel.forward signature and internals.",
            _EXPECTED_TRANSFORMERS_VERSION,
            transformers.__version__,
        )

    # Resolve the output dataclass from the actual runtime module
    model_module = importlib.import_module(type(model).__module__)
    Qwen3VLModelOutputWithPast = getattr(model_module, "Qwen3VLModelOutputWithPast")

    # Replaces Qwen3VLModel.forward from:
    #   transformers.models.qwen3_vl.modeling_qwen3_vl  (transformers v4.57.6)
    def visual_forward_qwen3_vl_inner(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        image_embeds = None
        video_embeds = None
        deepstack_image_embeds = None
        skip_visual = False

        # ---- merged visual forward (follows NemotronVL: modeling_nemotron_vl_h.py:2135-2152) ----
        if pixel_values is None and pixel_values_videos is None:
            skip_visual = True
        elif pixel_values is None:
            final_pixel_value = pixel_values_videos
            final_thw = video_grid_thw
            num_image = 0
        elif pixel_values_videos is None:
            final_pixel_value = pixel_values
            final_thw = image_grid_thw
            num_image = image_grid_thw.shape[0]
        else:
            final_pixel_value = torch.cat([pixel_values, pixel_values_videos], dim=0)
            final_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
            num_image = image_grid_thw.shape[0]

        if not skip_visual:
            # Qwen3VLModel.get_image_features: ViT → merger → torch.split per image/video
            # Returns (tuple_of_per_item_embeds, list_of_deepstack_layer_tensors)
            all_embeds, deepstack_image_embeds = self.get_image_features(
                final_pixel_value, final_thw
            )
            image_embeds = list(all_embeds[:num_image])
            video_embeds = list(all_embeds[num_image:])
        elif self.training:
            # ---- dummy visual forward for pure-text batches ----
            # Run a tiny dummy image through the full visual pipeline so that
            # FSDP all-gather operations stay synchronised across ranks.
            # Slice outputs to [0:0] so no dummy features leak into the LM,
            # while the empty tensors still carry grad_fn (SliceBackward)
            # keeping the ViT → merger → deepstack graph connected.
            dummy_h, dummy_w = 16, 16
            dummy_pixels = torch.zeros(
                dummy_h * dummy_w,
                self.visual.config.temporal_patch_size
                * self.visual.config.patch_size**2
                * 3,
                device=inputs_embeds.device,
                dtype=self.visual.dtype,
            )
            dummy_thw = torch.tensor(
                [[1, dummy_h, dummy_w]], device=inputs_embeds.device
            )
            image_embeds, deepstack_image_embeds = self.get_image_features(
                dummy_pixels, dummy_thw
            )
            image_embeds = [e[0:0] for e in image_embeds]
            deepstack_image_embeds = [e[0:0] for e in deepstack_image_embeds]

        # ---- scatter embeddings into inputs_embeds ----
        # Qwen3VLModel.get_placeholder_mask: finds image/video token positions in input_ids
        if image_embeds is not None and len(image_embeds) > 0:
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if video_embeds is not None and len(video_embeds) > 0:
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ---- aggregate visual_pos_masks / deepstack_visual_embeds ----
        # Consumed by Qwen3VLTextModel._deepstack_process which does:
        #   hidden_states[visual_pos_masks, :].clone() + visual_embeds
        # So deepstack_visual_embeds[i] must be in SEQUENCE order (matching
        # visual_pos_masks), not ViT batch order.
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            # deepstack_image_embeds from merged ViT is in concatenation order:
            # [image_tokens..., video_tokens...].  Reorder to sequence order
            # so _deepstack_process adds features to the correct positions.
            n_image_tok = image_mask.sum().item()
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for ds_embed in deepstack_image_embeds:
                img_ds = ds_embed[:n_image_tok]
                vid_ds = ds_embed[n_image_tok:]
                embed_joint = ds_embed.new_zeros(
                    visual_pos_masks.sum(), ds_embed.shape[-1]
                )
                embed_joint[image_mask_joint] = img_ds
                embed_joint[video_mask_joint] = vid_ds
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_image_embeds

        # ---- position ids (unchanged) ----
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor
                        / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Qwen3VLTextModel.forward — calls _deepstack_process at each deepstack layer
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    # Replace the forward method
    model.forward = visual_forward_qwen3_vl_inner.__get__(model, type(model))
    logger.info(
        "Patched %s instance forward with merged visual pass + pure-text dummy forward",
        type(model).__name__,
    )


# In order to support sequence packing during forward passes, the forward method of the language model must be patched.
# The patching logic is model-dependent, with special handling required for Vision-Language Models (VLMs) and other architectures.
SEQUENCE_PACKING_FORWARD_PATCH_FUNCTIONS = {
    "qwen3_vl": sequence_packing_forward_qwen3_vl_patch,
    "qwen3_5": sequence_packing_forward_qwen3_5_patch,
    "qwen3_5_moe": sequence_packing_forward_qwen3_5_patch,
    "gemma3": sequence_packing_forward_gemma3_vl_patch,
    "llm": sequence_packing_forward_llm_patch,
}
