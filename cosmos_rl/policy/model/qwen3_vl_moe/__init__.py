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

import os
import re
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from cosmos_rl.utils.util import (
    resolve_model_path,
    IdentityLayer,
    clear_weight_name,
    sync_model_vocab,
)
from cosmos_rl.utils.ulysses import (
    slice_inputs_for_ulysses,
)
from cosmos_rl.utils.logging import logger
from safetensors import safe_open
from cosmos_rl.policy.model.qwen3_vl_moe.weight_converter import (
    convert_weight_from_hf,
)
from cosmos_rl.dispatcher.data.packer.qwen3_vl_data_packer import (
    Qwen3_VL_DataPacker,
)
from cosmos_rl.policy.model.qwen3_vl_moe.weight_mapper import Qwen3VLMoeWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import ModelRegistry, BaseModel
from cosmos_rl.utils.sequence_packing import pack_sequences_for_inputs
from cosmos_rl.policy.kernel.moe.moe import MoEArgs
from cosmos_rl.policy.model.qwen3_moe import (
    Qwen3MoEBlock,
    Qwen3MoeArgs,
    build_norm as qwen3_moe_build_norm,
)

from cosmos_rl.policy.model.vision_encoder.qwen3_vl_moe import (
    Qwen3VLMoe_Encoder_Args,
    Qwen3VLMoeVisionModel,
)


@dataclass
class Qwen3VLMoe_Args:
    lm_args: Qwen3MoeArgs
    encoder_args: Qwen3VLMoe_Encoder_Args
    hf_config: AutoConfig = None


class Qwen3VLMoeTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3MoeArgs, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_seq_len
        self.original_max_seq_len = config.max_seq_len
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[config.rope_type]
        self.mrope_section = config.hf_config.rope_scaling.get(
            "mrope_section", [24, 20, 20]
        )
        self.reset_inv_freq(device=device)

    def reset_inv_freq(self, device: torch.device = None):
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config.hf_config, device
        )
        if not hasattr(self, "inv_freq"):
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.inv_freq.to(torch.float32)
            with torch.no_grad():
                self.inv_freq.data.copy_(inv_freq)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VLMoe has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3MoE(nn.Module):
    def __init__(self, model_args: Qwen3MoeArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.rotary_emb = Qwen3VLMoeTextRotaryEmbedding(model_args)
        self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.moe_args = MoEArgs(
            norm_topk_prob=getattr(model_args.hf_config, "norm_topk_prob", False),
            n_routed_experts=model_args.n_experts,
            n_shared_experts=getattr(model_args.hf_config, "n_shared_experts", 0),
            n_activated_experts=model_args.hf_config.num_experts_per_tok,
            n_expert_groups=getattr(model_args.hf_config, "n_group", 0),
            n_limited_groups=getattr(model_args.hf_config, "topk_group", 0),
            train_gate=model_args.train_gate,
            enable_router_bias=False,
            gate_bias_update_factor=model_args.gate_bias_update_factor,
            aux_loss_coeff=model_args.aux_loss_coeff,
            score_func=getattr(model_args.hf_config, "scoring_func", "softmax"),
            route_scale=getattr(model_args.hf_config, "routed_scaling_factor", 1.0),
            dim=model_args.dim,
            moe_inter_dim=model_args.ffn_dim,
        )
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = Qwen3MoEBlock(
                layer_id, model_args, self.moe_args
            )

        self.norm = qwen3_moe_build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode="qwen3_moe",
        )

        if not model_args.hf_config.tie_word_embeddings:
            self.tie_embed_tokens = False
            self.lm_head = nn.Linear(
                model_args.dim,
                model_args.vocab_size,
                bias="lm_head" in model_args.biases,
            )
        else:
            self.tie_embed_tokens = True
        self.identity_layer = IdentityLayer()
        self.gather_layer = IdentityLayer()

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        interested_tokens: Optional[torch.BoolTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,  # Additional arguments for compatibility
    ):
        h = self.identity_layer(inputs_embeds)

        position_embeddings = self.rotary_emb(h, position_ids)

        cp_mesh = kwargs.get("cp_mesh", None)

        if "valid_input_len" in kwargs:
            valid_input_len = kwargs["valid_input_len"]
            updated_kwargs = pack_sequences_for_inputs(
                inputs_embeds,
                valid_input_len,
                list(position_embeddings),
                interested_tokens,
                inputs_seq_dim=1,
                inputs_batch_dim=0,
                position_ids_seq_dim=2,
                position_ids_batch_dim=1,
                interested_tokens_seq_dim=1,
                interested_tokens_batch_dim=0,
                padding_mask=kwargs.get("padding_mask", None),
                cp_mesh=cp_mesh,
            )
            position_embeddings = tuple(updated_kwargs.pop("position_ids"))
            interested_tokens = updated_kwargs.pop("interested_tokens")
            h = updated_kwargs.pop("inputs")
            h = self.identity_layer(h)
            kwargs.update(updated_kwargs)
        elif cp_mesh is not None:
            [inputs_embeds, interested_tokens] = slice_inputs_for_ulysses(
                [inputs_embeds, interested_tokens],
                cp_mesh,
                seq_dims=[1, 1],
            )
            position_embeddings = tuple(
                slice_inputs_for_ulysses(
                    list(position_embeddings),
                    cp_mesh,
                    seq_dims=[2] * len(position_embeddings),
                )
            )
            h = self.identity_layer(inputs_embeds)

        for layer_idx, layer in self.layers.items():
            if (
                hasattr(layer, "_gradient_checkpointing_enabled")
                and layer._gradient_checkpointing_enabled
            ):
                h = torch.utils.checkpoint.checkpoint(
                    layer,
                    h,
                    position_embeddings,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                h = layer(h, position_embeddings=position_embeddings, **kwargs)
            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and int(layer_idx) in range(
                len(deepstack_visual_embeds)
            ):
                h = self.gather_layer(h)
                h = self._deepstack_process(
                    h,
                    visual_pos_masks,
                    deepstack_visual_embeds[int(layer_idx)],
                )
                h = self.identity_layer(h)

        # Add `if` check just in case `pp` is enabled
        if self.norm is not None:
            if interested_tokens is not None:
                assert not isinstance(
                    h, torch.distributed.tensor.DTensor
                ), "interested_tokens must be a local tensor"
                h = h[interested_tokens]
            assert self.lm_head is not None, "lm_head must be provided in last stage"
            h = self.lm_head(self.norm(h))
        return h

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def current_device(self):
        return next(self.parameters()).device

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        if not self.moe_args.fake_balanced_gate:
            for layer in self.layers.values():
                layer.mlp.gate.weight.requires_grad_(False)

        # rotary.inv_freq could get deleted and not re-initialized
        # so we need to delete it manually
        current_device = torch.cuda.current_device()
        self.rotary_emb.to(current_device)
        self.rotary_emb.reset_inv_freq()

    @cached_property
    def _get_nparams_and_flops_fn(self) -> Callable[[int], tuple[int, int]]:
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
            len(self.layers),
            self.model_args.n_heads,
            self.model_args.dim // self.model_args.n_heads,
        )
        return lambda seq_len: (
            nparams,
            6 * (nparams - nparams_embedding)
            + 12 * layers * heads * head_dim * seq_len,
        )


@ModelRegistry.register(
    Qwen3VLMoeWeightMapper, default_data_packer_cls=Qwen3_VL_DataPacker
)
class Qwen3VLMoeModel(BaseModel):
    def __init__(self, config: Qwen3VLMoe_Args):
        super().__init__(config.hf_config)
        self.config = config
        self.hf_config = config.hf_config
        self.model = Qwen3MoE(config.lm_args)
        self.visual = Qwen3VLMoeVisionModel(config.encoder_args)
        self.vocab_size = config.lm_args.vocab_size

    def _process_vision_embeddings(
        self, inputs_embeds, input_ids, pixel_values, grid_thw, pad_token_id
    ):
        """Helper function to process vision embeddings (images or videos)"""
        n_tokens = (input_ids == pad_token_id).sum().item()
        deepstack_image_embeds = None
        mask = None
        if n_tokens > 0:
            vision_embeds, deepstack_image_embeds = self.visual(
                pixel_values, grid_thw=grid_thw
            )
            assert (
                vision_embeds.shape[0] == n_tokens
            ), f"vision_embeds.shape[0] must be equal to n_tokens, but got {vision_embeds.shape[0]} != {n_tokens}"
            mask = input_ids == pad_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            vision_mask = mask_expanded.to(inputs_embeds.device)

            vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(vision_mask, vision_embeds)
        return inputs_embeds, deepstack_image_embeds, mask

    def _get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VLMoe use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        image_token_id = self.hf_config.image_token_id
        video_token_id = self.hf_config.video_token_id
        vision_start_token_id = self.hf_config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if position_ids is None:
            position_ids, _ = self._get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=None,
            )
            position_ids = position_ids.permute(1, 0, 2).contiguous()

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if self.model.embed_tokens is not None:
            assert input_ids.dtype in [
                torch.int32,
                torch.int64,
            ], "input_ids must be of type int32 or int64"
            inputs_embeds = self.model.embed_tokens(input_ids)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            n_video_tokens = (input_ids == self.video_token_id).sum().item()
            deepstack_image_embeds = None
            deepstack_video_embeds = None
            image_mask = None
            video_mask = None

            # print(f"inputs_embeds: {inputs_embeds.shape}, input_ids: {input_ids.shape}, n_image_tokens: {n_image_tokens}")
            # get vision embeddings as tokens for next phase
            if n_image_tokens > 0:
                assert (
                    image_grid_thw is not None
                ), "image_grid_thw must be provided if there are image tokens"
                inputs_embeds, deepstack_image_embeds, image_mask = (
                    self._process_vision_embeddings(
                        inputs_embeds,
                        input_ids,
                        pixel_values,
                        image_grid_thw,
                        self.image_token_id,
                    )
                )

            if n_video_tokens > 0:
                assert (
                    video_grid_thw is not None
                ), "video_grid_thw must be provided if there are video tokens"
                inputs_embeds, deepstack_video_embeds, video_mask = (
                    self._process_vision_embeddings(
                        inputs_embeds,
                        input_ids,
                        pixel_values_videos,
                        video_grid_thw,
                        self.video_token_id,
                    )
                )

            if image_mask is not None and video_mask is not None:
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(
                    deepstack_image_embeds, deepstack_video_embeds
                ):
                    embed_joint = img_embed.new_zeros(
                        visual_pos_masks.sum(), img_embed.shape[-1]
                    ).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_mask is not None:
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_mask is not None:
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_video_embeds
        else:
            assert (
                input_ids.is_floating_point()
            ), "input of pipeline stage > 0 must be of floating point type"
            inputs_embeds = input_ids
        # For GRPO, we can pass in the logprob_masks to the model
        # to avoid computing the logits which are not needed for the model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids.permute(1, 0, 2).contiguous(),
            interested_tokens=kwargs.pop("interested_tokens", None),
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,  # Additional arguments for compatibility
        )
        return outputs

    @property
    def image_token_id(self):
        return self.hf_config.image_token_id

    @property
    def video_token_id(self):
        return self.hf_config.video_token_id

    @property
    def delay_cp_slice_inputs(self):
        return True

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq_dim_idx = 2
        position_ids = kwargs.get("position_ids", None)
        return (
            kwargs["position_ids"].permute(1, 0, 2).contiguous()
            if position_ids is not None
            else None,
            kwargs["input_ids"],
            seq_dim_idx,
        )

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        self.model.post_to_empty_hook(cosmos_config)
        if self.visual is not None:
            self.visual.rotary_pos_emb.to(torch.cuda.current_device())
            self.visual.rotary_pos_emb.reset_inv_freq()

    def apply_pipeline_split(self, pp_rank, pp_size):
        """
        Apply pipeline split to the model.
        This typically involves splitting the model into multiple stages,
        and moving each stage to a different device.
        """
        assert pp_size > 1
        is_first = pp_rank == 0
        is_last = pp_rank == pp_size - 1

        # Compute the layers belonging to this stage
        n_layers = len(self.model.layers)
        layers_per_stage = n_layers // pp_size

        if not is_first:
            self.model.embed_tokens = None
            self.visual = None
        if not is_last:
            self.model.lm_head = None
            self.model.norm = None

        local_layers = torch.nn.ModuleDict()
        for i in range(
            pp_rank * layers_per_stage,
            ((pp_rank + 1) * layers_per_stage) if not is_last else n_layers,
        ):
            local_layers[str(i)] = self.model.layers[str(i)]

        # Reset the layers for pipeline splitting
        self.model.layers = local_layers

    @classmethod
    def from_model_args(cls, model_args: Qwen3VLMoe_Args) -> "Qwen3VLMoeModel":
        """
        Initialize a Qwen3VLMoeModel model from a Qwen3VLMoe_Args object.

        Args:
            model_args (Qwen3VLMoe_Args): Model configuration arguments.

        Returns:
            Qwen3VLMoeModel: Qwen3VLMoeModel model.

        """
        return cls(model_args)

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_path (str): Path to the HuggingFace model.
            parallel_dims (ParallelDims): Parallel dimensions definition.
        """
        # Load all safetensors from `model_path`
        model_type = self.hf_config.model_type
        lm_type = self.hf_config.text_config.model_type
        model_path = resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        # Load LM weights
        lm_state_dict = self.model.state_dict()
        lm_state_dict = {clear_weight_name(k): v for k, v in lm_state_dict.items()}
        # print(f"lm_state_dict: {lm_state_dict.keys()}")
        # Rename dict to remove all `._orig_mod` in keys
        visual_state_dict = self.visual.state_dict()
        visual_state_dict = {
            clear_weight_name(k): v for k, v in visual_state_dict.items()
        }

        with torch.device(device):
            for f in safetensors_files:
                weights_of_ckpt = {}
                ckpt = safe_open(
                    os.path.join(model_path, f), framework="pt", device=str(device)
                )
                keys = ckpt.keys()
                for name in keys:
                    ckpt_tensor = ckpt.get_tensor(name)
                    weights_of_ckpt[name] = ckpt_tensor

                n_experts = self.config.lm_args.n_experts

                for name in weights_of_ckpt.keys():
                    tensor = weights_of_ckpt[name]
                    dest_name, shared_weight = convert_weight_from_hf(
                        tensor, name, model_type, lm_type, n_experts, parallel_dims
                    )
                    if dest_name is None:
                        # This is due to the expert parallelism grouping
                        continue

                    # For MoE LM
                    if match := re.search(  # noqa: F841
                        r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)",
                        dest_name,
                    ):
                        tp_ep_rank, tp_ep_size = parallel_dims.tp_coord
                        assert (
                            n_experts % tp_ep_size == 0
                        ), "n_experts must be divisible by tp_ep_size"

                        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                            dp_shard_rank = parallel_dims.mesh[
                                tuple(("dp_shard_cp",))
                            ].get_local_rank()
                            dp_shard_size = parallel_dims.mesh[
                                tuple(("dp_shard_cp",))
                            ].size()
                        else:
                            dp_shard_rank = 0
                            dp_shard_size = 1

                        n_expert_per_ep = n_experts // tp_ep_size

                        for expert_id in range(n_experts):
                            belongs_to_current_ep = (
                                tp_ep_rank * n_expert_per_ep
                                <= expert_id  # Expert index
                                < (tp_ep_rank + 1) * n_expert_per_ep
                            )

                            belongs_to_current_dp_shard = (
                                expert_id - tp_ep_rank * n_expert_per_ep
                            ) // (n_expert_per_ep // dp_shard_size) == dp_shard_rank

                            if belongs_to_current_ep and belongs_to_current_dp_shard:
                                expert_shard_weight = shared_weight[expert_id]
                                # Convert expert_id to local_expert_id
                                n_local_experts = (
                                    n_experts
                                    // parallel_dims.tp
                                    // (parallel_dims.dp_shard * parallel_dims.cp)
                                )
                                expert_id = expert_id % n_local_experts
                                tensor_to_copy = []
                                if "gate_up_proj" in dest_name:
                                    # gate_and_up_projs
                                    gate_proj_name = dest_name.replace(
                                        "gate_up_proj", "gate_and_up_projs"
                                    )
                                    target_gate_proj_tensor = lm_state_dict[
                                        gate_proj_name
                                    ]
                                    tensor_to_copy.append(
                                        (
                                            target_gate_proj_tensor,
                                            expert_shard_weight,
                                        )
                                    )
                                elif "down_proj" in dest_name:
                                    # layers.0.mlp.experts.down_projs
                                    down_proj_name = dest_name.replace(
                                        "down_proj", "down_projs"
                                    )
                                    target_down_proj_tensor = lm_state_dict[
                                        down_proj_name
                                    ]
                                    tensor_to_copy.append(
                                        (target_down_proj_tensor, expert_shard_weight)
                                    )

                                for target_tensor, expert_weight in tensor_to_copy:
                                    is_dist_tensor = isinstance(
                                        target_tensor, torch.distributed.tensor.DTensor
                                    )
                                    local_view = (
                                        target_tensor.to_local()
                                        if is_dist_tensor
                                        else target_tensor
                                    )

                                    local_view = local_view[expert_id]
                                    expert_weight = expert_weight.transpose(0, 1)

                                    assert (
                                        local_view.shape == expert_weight.shape
                                    ), f"Shape mismatch: {local_view.shape} != {expert_weight.shape} for {dest_name} with original shape {target_tensor.shape}"
                                    with torch.no_grad():
                                        local_view.data.copy_(expert_weight)
                            else:
                                continue
                        continue

                    if dest_name in lm_state_dict:
                        target_tensor = lm_state_dict[dest_name]
                    elif dest_name in visual_state_dict:
                        target_tensor = visual_state_dict[dest_name]
                    elif parallel_dims.pp_enabled:
                        # logger.warning(f"Skipping weight: {dest_name} because it's not in the model due to pipeline split")
                        continue
                    else:
                        raise ValueError(f"Unsupported weight: {dest_name}")

                    is_dist_tensor = isinstance(
                        target_tensor, torch.distributed.tensor.DTensor
                    )
                    local_view = (
                        target_tensor.to_local() if is_dist_tensor else target_tensor
                    )

                    assert (
                        local_view.shape == shared_weight.shape
                    ), f"Shape mismatch: {local_view.shape} != {shared_weight.shape} for {dest_name} with original shape {target_tensor.shape}"
                    with torch.no_grad():
                        local_view.data.copy_(shared_weight)

    def separate_model_parts(self) -> List[nn.Module]:
        return [self.model, self.visual]

    @property
    def parallelize_fn(self) -> Tuple[Callable, nn.Module]:
        from cosmos_rl.policy.model.qwen3_vl_moe.parallelize import parallelize

        return parallelize, self

    @staticmethod
    def supported_model_types():
        return ["qwen3_vl_moe"]

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "Qwen3VLMoeModel":
        """
        Initialize a Qwen3VLMoeModel model from a pretrained model.

        Args:
            model_name_or_path (str): Model name or path to the pretrained model.

        Returns:
            Qwen3VLMoeModel: Qwen3VLMoeModel model.

        """
        if hf_config.model_type not in cls.supported_model_types():
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        if max_position_embeddings is None:
            max_position_embeddings = hf_config.max_position_embeddings
        else:
            hf_config.max_position_embeddings = max_position_embeddings

        torch_dtype = hf_config.torch_dtype
        hf_config.text_config.torch_dtype = torch_dtype
        hf_config.vision_config.torch_dtype = torch_dtype
        vocab_size = sync_model_vocab(model_name_or_path)

        lm_config = hf_config.text_config
        # Qwen3MoE does not have any biases
        bias_list = []
        rope_scaling = {}
        if hasattr(lm_config, "rope_scaling"):
            rope_scaling = lm_config.rope_scaling or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        try:
            head_dim = lm_config.head_dim
        except Exception:
            head_dim = lm_config.hidden_size // lm_config.num_attention_heads
            logger.warning(f"head_dim not found in config, using {head_dim}")

        lm_args = Qwen3MoeArgs(
            dim=lm_config.hidden_size,
            ffn_dim=lm_config.moe_intermediate_size,
            n_layers=lm_config.num_hidden_layers,
            n_experts=lm_config.num_experts,
            n_heads=lm_config.num_attention_heads,
            n_kv_heads=lm_config.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_seq_len=max_position_embeddings,
            rope_theta=lm_config.rope_theta,
            q_k_norm_enabled=True,
            norm_type="rmsnorm",
            rope_type=rope_type,
            biases=bias_list,
            train_gate=True,
            gate_bias_update_factor=1.0,
            aux_loss_coeff=0.0,
            hf_config=lm_config,
        )

        vision_config = hf_config.vision_config
        encoder_args = Qwen3VLMoe_Encoder_Args(
            depth=vision_config.depth,
            hidden_size=vision_config.hidden_size,
            hidden_act=vision_config.hidden_act,
            intermediate_size=vision_config.intermediate_size,
            n_heads=vision_config.num_heads,
            in_channels=vision_config.in_channels,
            patch_size=vision_config.patch_size,
            spatial_merge_size=vision_config.spatial_merge_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            num_position_embeddings=vision_config.num_position_embeddings,
            out_hidden_size=vision_config.out_hidden_size,
            deepstack_visual_indexes=vision_config.deepstack_visual_indexes,
            norm_type="layernorm",
            layer_norm_eps=1e-6,
            hf_config=vision_config,
        )
        args = Qwen3VLMoe_Args(
            lm_args=lm_args,
            encoder_args=encoder_args,
            hf_config=hf_config,
        )
        return cls.from_model_args(args)

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        n_params = 0
        n_flops = 0
        if self.visual is not None:
            n_params, n_flops = self.visual._get_nparams_and_flops_fn()(seq_len)
        if self.model is not None:
            lm_n_params, lm_n_flops = self.model._get_nparams_and_flops_fn(seq_len)
            n_params += lm_n_params
            n_flops += lm_n_flops
        return n_params, n_flops

    @classmethod
    def fqn_filter_for_quantization(cls) -> List[str]:
        llm = [
            "lm_head",
        ]
        visual = [
            "visual",
        ]  # Filter Linear in visual out, they will corrupt the FP8/FP4 Linear.
        return llm + visual

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        visual_n_heads = self.config.encoder_args.n_heads
        llm_n_heads = self.config.lm_args.n_heads
        cp_compatible = (
            visual_n_heads % (cp_size * tp_size) == 0
            and llm_n_heads % (cp_size * tp_size) == 0
        )
        if not cp_compatible:
            raise ValueError(
                f"Model is not compatible with cp parallelism, model's visual_n_heads={visual_n_heads} or llm_n_heads={llm_n_heads} is not divisible by cp size({cp_size}) * tp_size({tp_size}) = {cp_size * tp_size}"
            )

    def check_tp_compatible(self, tp_size: int):
        visual_n_heads = self.config.encoder_args.n_heads
        llm_n_heads = self.config.lm_args.n_heads
        llm_n_kv_heads = self.config.lm_args.n_kv_heads
        llm_n_experts = self.config.lm_args.n_experts
        non_divisible_by_tp_size = (
            visual_n_heads % tp_size != 0
            or llm_n_heads % tp_size != 0
            or llm_n_kv_heads % tp_size != 0
            or llm_n_experts % tp_size != 0
        )
        if non_divisible_by_tp_size:
            raise ValueError(
                f"Model is not compatible with tp/ep parallelism, model's visual_n_heads={visual_n_heads} or llm_n_heads={llm_n_heads} or llm_n_kv_heads={llm_n_kv_heads} or llm_n_experts={llm_n_experts} is not satisified by tp size({tp_size})"
            )
