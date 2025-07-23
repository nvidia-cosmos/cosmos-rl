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
from torch import nn
import inspect
from typing import Tuple, List, Optional, Callable
from transformers import AutoConfig
from cosmos_rl.utils.util import (
    sync_model_vocab,
    clear_weight_name,
    retry,
    safe_deep_getattr,
    load_model_class_by_config,
)
from cosmos_rl.utils.constant import COSMOS_HF_MODEL_TYPES
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.model.hf_models.weight_converter import convert_weight_from_hf
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from functools import cached_property


@ModelRegistry.register(HFModelWeightMapper)
class HFModel(BaseModel):
    """
    HFModel Module

    Args:
        hf_config : Model configuration arguments.
        model: model loaded from hf.

    Attributes:
        hf_config : Model configuration arguments.
        model: model loaded from hf.
    """

    @staticmethod
    def supported_model_types():
        return [COSMOS_HF_MODEL_TYPES]

    def __init__(self, hf_config, model, model_class, is_vlm=False):
        super().__init__(hf_config)
        self.hf_config = hf_config
        self.model = model
        self.model_class = model_class
        self.is_vlm = is_vlm

    @cached_property
    def model_forward_valid_kwargs(self):
        sig = inspect.signature(self.model.forward)
        return sig.parameters.keys()

    def _process_vision_embeddings(
        self, inputs_embeds, input_ids, pixel_values, grid_thw, pad_token_id
    ):
        """Helper function to process vision embeddings (images or videos)"""
        n_tokens = (input_ids == pad_token_id).sum().item()
        if n_tokens > 0:
            # TODO: check whether vision_model.forward has grid_thw as input
            # e.g. vision models like SiglipVisionModel do not have grid_thw as input
            vision_embeds = self.vision_model(pixel_values, grid_thw=grid_thw)
            assert (
                vision_embeds.shape[0] == n_tokens
            ), "vision_embeds.shape[0] must be equal to n_tokens"
            mask = input_ids == pad_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            vision_mask = mask_expanded.to(inputs_embeds.device)

            vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(vision_mask, vision_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_lengths_per_sample: Optional[torch.Tensor] = None,
        pixel_values_videos_lengths_per_sample: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        kwargs_filtered = {
            k: v for k, v in kwargs.items() if k in self.model_forward_valid_kwargs
        }
        if self.is_vlm:
            embed_tokens = getattr(self.language_model, "embed_tokens", None)
            assert embed_tokens is not None, "embed_tokens is not found"
            inputs_embeds = embed_tokens(input_ids)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            n_video_tokens = (
                0
                if self.video_token_id is None
                else (input_ids == self.video_token_id).sum().item()
            )
            if n_image_tokens > 0:
                assert (
                    image_grid_thw is not None
                ), "image_grid_thw must be provided if there are image tokens"
                total_image_lengths = pixel_values_lengths_per_sample.sum().item()
                unpadded_pixels = torch.zeros(
                    total_image_lengths,
                    pixel_values.shape[2],
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
                current_index = 0
                for i in range(pixel_values_lengths_per_sample.shape[0]):
                    image_length = pixel_values_lengths_per_sample[i].item()
                    unpadded_pixels[current_index : current_index + image_length] = (
                        pixel_values[i, :image_length]
                    )
                    current_index += image_length
                inputs_embeds = self._process_vision_embeddings(
                    inputs_embeds,
                    input_ids,
                    unpadded_pixels,
                    image_grid_thw,
                    self.image_token_id,
                )

            if n_video_tokens > 0:
                assert (
                    video_grid_thw is not None
                ), "video_grid_thw must be provided if there are video tokens"
                total_video_lengths = (
                    pixel_values_videos_lengths_per_sample.sum().item()
                )
                unpadded_pixels = torch.zeros(
                    total_video_lengths,
                    pixel_values_videos.shape[2],
                    device=pixel_values_videos.device,
                    dtype=pixel_values_videos.dtype,
                )
                current_index = 0
                for i in range(pixel_values_videos_lengths_per_sample.shape[0]):
                    video_length = pixel_values_videos_lengths_per_sample[i].item()
                    unpadded_pixels[current_index : current_index + video_length] = (
                        pixel_values_videos[i, :video_length]
                    )
                    current_index += video_length
                inputs_embeds = self._process_vision_embeddings(
                    inputs_embeds,
                    input_ids,
                    unpadded_pixels,
                    video_grid_thw,
                    self.video_token_id,
                )
            kwargs_filtered["inputs_embeds"] = inputs_embeds

        out = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            *args,
            **kwargs_filtered,
        )
        return out.logits

    @property
    def image_token_id(self):
        # image_token_id or image_token_index
        image_token_id = None
        if self.is_vlm:
            if hasattr(self.hf_config, "image_token_id"):
                # Qwen2_5_VL
                image_token_id = self.hf_config.image_token_id
            elif hasattr(self.hf_config, "image_token_index"):
                # Gemma3-it
                image_token_id = self.hf_config.image_token_index
            else:
                raise ValueError(f"Can not get image token id from {self.hf_config}")
        return image_token_id

    @property
    def video_token_id(self):
        return self.hf_config.video_token_id if self.is_vlm else None

    @property
    def lm_layers(self):
        lm_layers = None
        sub_lm_model = getattr(self.language_model, "model", None)
        if sub_lm_model is None:
            lm_layers = self.language_model.layers
        else:
            lm_layers = sub_lm_model.layers
        assert (
            lm_layers is not None
        ), f"Can not get lm layers from {self.language_model}"
        return lm_layers

    @property
    def vision_layers(self):
        vision_layers = None
        if self.vision_model is not None:
            vision_layers = getattr(self.vision_model, "blocks", None)
            if vision_layers is None:
                # Models like Gemma3-4b-it use SiglipVisionModel as vision_model
                vision_layers = safe_deep_getattr(
                    self.vision_model, "vision_model.encoder.layers", None
                )
            assert (
                vision_layers is not None
            ), f"Can not get vision layers from {self.vision_model}."
        return vision_layers

    @property
    def n_lm_layers(self):
        n_lm_layers = 0
        if hasattr(self.text_config, "num_hidden_layers"):
            n_lm_layers = self.text_config.num_hidden_layers
        else:
            logger.warning(f"Can not get num of llm layers from {self.text_config}.")
        return n_lm_layers

    @property
    def n_vision_layers(self):
        n_vision_layers = 0
        if hasattr(self.vision_config, "num_hidden_layers"):
            n_vision_layers = self.vision_config.num_hidden_layers
        # qwen2.5-vl
        elif hasattr(self.vision_config, "depth"):
            n_vision_layers = self.vision_config.depth
        else:
            logger.warning(
                f"Can not get num of vision model layers from {self.vision_config}."
            )
        return n_vision_layers

    @property
    def text_config(self):
        text_config = None
        if self.is_vlm:
            text_config = getattr(self.hf_config, "text_config", None)
            if text_config is None:
                text_config = self.hf_config
        else:
            text_config = self.hf_config
        return text_config

    @property
    def vision_config(self):
        return self.hf_config.vision_config if self.is_vlm else None

    @property
    def language_model(self):
        language_model = None
        if self.is_vlm:
            if hasattr(self.model, "language_model"):
                language_model = self.model.language_model
            elif hasattr(self.model, "model"):
                language_model = self.model.model
            else:
                logger.warning(f"Can not get language model from {self.model}.")
        else:
            language_model = self.model
        return language_model

    @property
    def vision_model(self):
        vision_model = None
        if self.is_vlm:
            # vision_tower or visual
            if hasattr(self.model, "vision_tower"):
                vision_model = self.model.vision_tower
            elif hasattr(self.model, "visual"):
                vision_model = self.model.visual
            else:
                raise ValueError(f"Can not get vision model from {self.model}")
        return vision_model

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        # reset buffer registered in __init__() function,
        # e.g. rotary_emb.inv_freq, embed_tokens.embed_scale
        # For hf compatibility
        language_model = (
            getattr(self.language_model, "model", None) or self.language_model
        )
        rotary_emb = getattr(language_model, "rotary_emb", None) or getattr(
            language_model, "rotary_pos_emb", None
        )
        current_device = torch.cuda.current_device()
        if rotary_emb is not None:
            rope_init_fn = getattr(rotary_emb, "rope_init_fn", None)
            if rope_init_fn is not None:
                inv_freq, rotary_emb.attention_scaling = rope_init_fn(
                    self.text_config, device=current_device
                )
                rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
            else:
                logger.warning(
                    "rotary_emb does not have rope_init_fn, cannot reset inv_freq."
                )
        # Models like Gemma have rotary_emb_local
        rotary_emb_local = getattr(language_model, "rotary_emb_local", None)
        if rotary_emb_local is not None:
            rope_init_fn = getattr(rotary_emb_local, "rope_init_fn", None)
            if rope_init_fn is not None:
                local_inv_freq, rotary_emb_local.attention_scaling = rope_init_fn(
                    self.text_config, device=current_device
                )
                rotary_emb_local.register_buffer(
                    "inv_freq", local_inv_freq, persistent=False
                )
            else:
                logger.warning(
                    "rotary_emb_local does not have rope_init_fn, cannot reset inv_freq."
                )

        if self.text_config.model_type in ["gemma3_text"]:
            embed_tokens = language_model.embed_tokens
            embed_scale = self.hf_config.hidden_size**0.5
            embed_tokens.register_buffer(
                "embed_scale", torch.tensor(embed_scale), persistent=False
            )

        vision_model = self.vision_model
        if vision_model is not None:
            rotary_emb = getattr(vision_model, "rotary_pos_emb", None) or getattr(
                vision_model, "rotary_emb", None
            )
            if rotary_emb is not None:
                rope_init_fn = getattr(rotary_emb, "rope_init_fn", None)
                if rope_init_fn is not None:
                    inv_freq, rotary_emb.attention_scaling = rope_init_fn(
                        self.vision_config, device=current_device
                    )
                    rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.hf_models.parallelize import parallelize

        return parallelize, self

    def apply_pipeline_split(self, pp_rank, pp_size):
        """
        Apply pipeline split to the model.
        This typically involves splitting the model into multiple stages,
        and moving each stage to a different device.
        """
        assert False, "Pipeline split is not supported for HFModel"

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_path (str): Path to the HuggingFace model.
            parallel_dims (ParallelDims): Parallel dimensions definition.
            info_inly (bool): Only collect the tensor infomation without actual data loading.
        """
        model_type = retry(AutoConfig.from_pretrained)(model_name_or_path).model_type
        model_with_weights = self.model_class.from_pretrained(model_name_or_path).to(
            device
        )

        state_dict = model_with_weights.state_dict()
        self_state_dict = self.model.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}
        all_tensor_names = self_state_dict.keys()
        lm_head_weight_key = "lm_head.weight"
        embed_tokens_weight_key = "model.embed_tokens.weight"
        reserved = {}

        for name, tensor in state_dict.items():
            if name == embed_tokens_weight_key:
                reserved[name] = tensor
            dest_name, shared_weight = convert_weight_from_hf(
                tensor, name, model_type, parallel_dims
            )

            target_tensor = self_state_dict[dest_name]
            is_dist_tensor = isinstance(target_tensor, torch.distributed.tensor.DTensor)
            local_view = target_tensor.to_local() if is_dist_tensor else target_tensor
            assert (
                local_view.shape == shared_weight.shape
            ), f"Shape mismatch: {local_view.shape} != {shared_weight.shape} for {dest_name}"
            with torch.no_grad():
                local_view.data.copy_(shared_weight)

        if (
            lm_head_weight_key not in all_tensor_names
            and embed_tokens_weight_key in all_tensor_names
        ):
            # tied with embed_tokens.weight
            name = lm_head_weight_key
            assert embed_tokens_weight_key in reserved
            tensor = reserved[embed_tokens_weight_key]
            dest_name, shared_weight = convert_weight_from_hf(
                tensor, name, model_type, parallel_dims
            )
            if dest_name in self_state_dict:
                target_tensor = self_state_dict[dest_name]
                is_dist_tensor = isinstance(
                    target_tensor, torch.distributed.tensor.DTensor
                )
                local_view = (
                    target_tensor.to_local() if is_dist_tensor else target_tensor
                )
                assert (
                    local_view.shape == shared_weight.shape
                ), f"Shape mismatch: {local_view.shape} != {shared_weight.shape} for {dest_name}"
                with torch.no_grad():
                    local_view.data.copy_(shared_weight)

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        position_ids = None
        inputs = kwargs["input_ids"]
        seq_dim_idx = 1 if not self.is_vlm else 2
        return position_ids, inputs, seq_dim_idx

    def separate_model_parts(self) -> List[nn.Module]:
        if self.is_vlm:
            # TODO: Add lm head
            return [self.language_model, self.vision_model]
        else:
            return [self.language_model]

    @cached_property
    def _get_nparams_and_flops_fn(
        self, is_vision_model: bool
    ) -> Callable[[int], tuple[int, int]]:
        if is_vision_model:
            nparams = sum(p.numel() for p in self.parameters())
            nparams_embedding = sum(
                sum(p.numel() for p in m.parameters())
                for m in self.children()
                if isinstance(m, nn.Embedding)
            )

            # Reasoning behind the factor of 12 for the self-attention part of the formula:
            # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
            # 2. the flash attention does 1 more matmul recomputation in the backward
            #    but recomputation should not be counted in calculating MFU           (+0)
            # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
            # 4. we follow the convention and do not account for sparsity in causal attention
            num_heads = 0
            if hasattr(self.vision_config, "num_attention_heads"):
                num_heads = self.vision_config.num_attention_heads
            elif hasattr(self.vision_config, "num_heads"):
                num_heads = self.vision_config.num_heads
            else:
                logger.warning(f"Can not get num of heads from {self.vision_config}.")
            layers, heads, head_dim = (
                self.n_vision_layers,
                num_heads,
                self.vision_config.hidden_size // num_heads,
            )
            return lambda seq_len: (
                nparams,
                6 * (nparams - nparams_embedding)
                + 12 * layers * heads * head_dim * seq_len,
            )
        else:
            nparams = sum(p.numel() for p in self.parameters())
            nparams_embedding = sum(
                sum(p.numel() for p in m.parameters())
                for m in self.children()
                if isinstance(m, nn.Embedding)
            )
            # Reasoning behind the factor of 12 for the self-attention part of the formula:
            # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
            # 2. the flash attention does 1 more matmul recomputation in the backward
            #    but recomputation should not be counted in calculating MFU           (+0)
            # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
            # 4. we follow the convention and do not account for sparsity in causal attention
            layers, heads, head_dim = (
                self.n_lm_layers,
                self.text_config.num_attention_heads,
                self.text_config.hidden_size // self.text_config.num_attention_heads,
            )
            return lambda seq_len: (
                nparams,
                6 * (nparams - nparams_embedding)
                + 12 * layers * heads * head_dim * seq_len,
            )

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        n_params = 0
        n_flops = 0
        if self.vision_model is not None:
            n_params, n_flops = self._get_nparams_and_flops_fn(is_vision_model=True)(
                seq_len
            )

        lm_n_params, lm_n_flops = self._get_nparams_and_flops_fn(is_vision_model=False)(
            seq_len
        )
        n_params += lm_n_params
        n_flops += lm_n_flops
        return n_params, n_flops

    @classmethod
    def from_model_args(cls, hf_config) -> "HFModel":
        """
        Initialize a HFModel model from a HFModelArgs object.

        Args:
            hf_config : hf model config.

        Returns:
            HFModel: HFModel model.

        """
        is_vlm = getattr(hf_config, "vision_config", None) is not None
        model_class = None
        try:
            model_class = load_model_class_by_config(hf_config)
            model = model_class(hf_config)
        except Exception as e:
            logger.error(f"Can not load {hf_config.model_type}")
            raise e
        return cls(hf_config, model, model_class, is_vlm=is_vlm)

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "HFModel":
        """
        Initialize a HFModel model from a pretrained model.

        Args:
            hf_config (AutoConfig): HuggingFace config.
            model_name_or_path (str): Model name or path to the pretrained model.
            max_position_embeddings (int): Maximum position embeddings.

        Returns:
            HFModel: HFModel model.

        """

        if max_position_embeddings is None:
            max_position_embeddings = hf_config.max_position_embeddings
        else:
            hf_config.max_position_embeddings = max_position_embeddings
        _ = sync_model_vocab(model_name_or_path)

        return cls.from_model_args(hf_config)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    @classmethod
    def fqn_filter_for_fp8(cls) -> List[str]:
        llm = [
            "lm_head",
        ]
        visual = [
            "visual",
            "vision_tower",
        ]  # Filter Linear in visual out, they will corrupt the FP8 Linear.
        return llm + visual

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        assert cp_size == 1, "cp is not supported for HFModel"
        assert tp_size == 1, "tp is not supported for HFModel"
