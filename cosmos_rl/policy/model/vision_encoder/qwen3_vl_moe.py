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
from typing import Callable, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import AutoConfig
from cosmos_rl.policy.kernel.modeling_utils import FlashAttnMeta


@dataclass
class Qwen3VLMoe_Encoder_Args:
    depth: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    n_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    num_position_embeddings: int
    out_hidden_size: int
    deepstack_visual_indexes: List[int]
    norm_type: str = "layernorm"
    layer_norm_eps: float = 1e-6
    hf_config: AutoConfig = None


class Qwen3VLMoeVisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLMoe_Encoder_Args, bias: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLMoeVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLMoeVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.reset_inv_freq()

    def reset_inv_freq(self, device: torch.device = None):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        inv_freq = inv_freq.to(device) if device is not None else inv_freq
        if not hasattr(self, "inv_freq"):
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.inv_freq.to(torch.float32)
            with torch.no_grad():
                self.inv_freq.data.copy_(inv_freq)

    def forward(self, seqlen: int, device: torch.device = None) -> torch.Tensor:
        if self.inv_freq.dtype != torch.float32:
            self.reset_inv_freq(device=device)
            assert (
                self.inv_freq.dtype == torch.float32
            ), "inv_freq dtype should be float32"
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLMoeVisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3VLMoe_Encoder_Args, use_postshuffle_norm=False
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


if os.environ.get("COSMOS_USE_HF_IMPL", "0").lower() in ["1", "true"]:

    def apply_rotary_pos_emb_vision(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q, k = q.float(), k.float()
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        q_embed = q_embed.to(orig_q_dtype)
        k_embed = k_embed.to(orig_k_dtype)
        return q_embed, k_embed
else:

    def apply_rotary_pos_emb_vision(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to the query and key tensors.
        """
        cos = cos.chunk(2, dim=-1)[0].contiguous()
        sin = sin.chunk(2, dim=-1)[0].contiguous()
        apply_rotary_emb = FlashAttnMeta().apply_rotary_emb
        q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
        k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
        return q_embed, k_embed


class Qwen3VLMoeVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attention_dropout = 0.0
        self.attn_func = FlashAttnMeta().flash_attn_varlen_func

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings

        q, k = apply_rotary_pos_emb_vision(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        input_dtype = q.dtype
        # flash_attn only support bfloat16/float16
        # Cast qkv to torch.bfloat16 if dtype for forward is torch.float32
        if input_dtype == torch.float32:
            target_dtype = torch.bfloat16
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)

        attn_output = self.attn_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        ).reshape(seq_length, -1)
        attn_output = self.proj(attn_output.to(input_dtype))
        return attn_output


class Qwen3VLMoeVisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLMoe_Encoder_Args) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLMoeVisionAttention(
            config.hidden_size, num_heads=config.n_heads
        )
        self.mlp = Qwen3VLMoeVisionMLP(config=config, bias=True)

    def forward(
        self, hidden_states, cu_seqlens, position_embeddings, max_seqlen
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLMoeVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLMoeVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.n_heads
        self.rotary_pos_emb = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = torch.nn.ModuleDict()
        for layer_id in range(config.depth):
            self.blocks[str(layer_id)] = Qwen3VLMoeVisionBlock(config)

        self.merger = Qwen3VLMoeVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLMoeVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(
                merge_size, device=device
            )  # intra-block row offsets
            intra_col = torch.arange(
                merge_size, device=device
            )  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(
            idx_list, dtype=torch.long, device=self.pos_embed.weight.device
        )
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=self.pos_embed.weight.device,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        with torch.no_grad():
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        deepstack_feature_lists = []
        for layer_num, blk in self.blocks.items():
            if (
                hasattr(blk, "_gradient_checkpointing_enabled")
                and blk._gradient_checkpointing_enabled
            ):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    blk,
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                    max_seqlen,
                    use_reentrant=False,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    max_seqlen=max_seqlen,
                )
            if int(layer_num) in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(int(layer_num))
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists

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
            len(self.blocks),
            self.config.n_heads,
            self.config.hidden_size // self.config.n_heads,
        )
        return lambda seq_len: (
            nparams,
            6 * (nparams - nparams_embedding)
            + 12 * layers * heads * head_dim * seq_len,
        )
