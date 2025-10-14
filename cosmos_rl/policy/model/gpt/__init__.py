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
import math
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
from transformers import AutoConfig
from cosmos_rl.utils.util import (
    resolve_model_path,
    IdentityLayer,
    clear_weight_name,
    sync_model_vocab,
    retry,
)
from safetensors import safe_open
from cosmos_rl.policy.model.base import ModelRegistry, BaseModel
from cosmos_rl.policy.model.gpt.weight_mapper import GPTWeightMapper
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.model.gpt.weight_converter import convert_weight_from_hf
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from functools import cached_property
import cosmos_rl.policy.kernel.modeling_utils as modeling_utils
from cosmos_rl.policy.kernel.norm import RMSNorm
import cosmos_rl.policy.kernel.rope as rope
from cosmos_rl.policy.kernel.fused import MLPActMulFunc
from naruto.q_former import QformerEncoder
from naruto.util import vae_encode_mode
from diffusers import AutoencoderKL
from cosmos_rl.policy.model.gpt.diffuse_head import DiffusionCondAttentionNet


def build_norm(
    norm_type: str, dim: int, eps: float, casting_mode: Optional[str] = None
):
    assert norm_type == "rmsnorm", f"Unknown norm_type: '{norm_type}'"
    return RMSNorm(dim, eps, casting_mode=casting_mode)


@dataclass
class GPTArgs:
    dim: int
    ffn_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int

    head_dim: int
    vocab_size: int
    max_seq_len: int
    biases: List[str] = field(default_factory=lambda: [])
    q_k_norm_enabled: bool = False
    norm_eps: float = 1e-6
    rope_theta: float = 10000
    norm_type: str = "rmsnorm"
    rope_type: str = "default"
    hf_config: AutoConfig = None


class RotaryEmbedding(nn.Module):
    def __init__(self, args: GPTArgs, device=None):
        super().__init__()
        self.args = args
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[args.rope_type]
        self.device = device
        self.config = args
        self.reset_inv_freq(device=device)

    def reset_inv_freq(self, device: torch.device = None):
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config.hf_config, self.device
        )
        if not hasattr(self, "inv_freq"):
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.inv_freq.to(torch.float32)
            with torch.no_grad():
                self.inv_freq.data.copy_(inv_freq)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq.dtype != torch.float32:
            self.reset_inv_freq(device=x.device)
            assert self.inv_freq.dtype == torch.float32

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (GPTArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        q_proj (Linear): Linear transformation for queries.
        k_proj (Linear): Linear transformation for keys.
        v_proj (Linear): Linear transformation for values.
        o_proj (Linear): Linear transformation for output.
    """

    def __init__(self, model_args: GPTArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.attn_func = modeling_utils.flash_attn_func
        self.attn_func_varlen = modeling_utils.flash_attn_varlen_func
        self.dim = model_args.dim
        # MM
        self.mm_q_proj = nn.Linear(
            model_args.dim,
            model_args.n_heads * self.head_dim,
            bias="mm_q_proj" in model_args.biases,
        )
        self.mm_k_proj = nn.Linear(
            model_args.dim,
            model_args.n_kv_heads * self.head_dim,
            bias="mm_k_proj" in model_args.biases,
        )
        self.mm_q_norm = (
            build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
                casting_mode=model_args.hf_config.model_type,
            )
            if model_args.q_k_norm_enabled
            else None
        )
        self.mm_k_norm = (
            build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
                casting_mode=model_args.hf_config.model_type,
            )
            if model_args.q_k_norm_enabled
            else None
        )
        self.mm_v_proj = nn.Linear(
            model_args.dim,
            model_args.n_kv_heads * self.head_dim,
            bias="mm_v_proj" in model_args.biases,
        )
        self.mm_o_proj = nn.Linear(
            model_args.n_heads * self.head_dim,
            model_args.dim,
            bias="mm_o_proj" in model_args.biases,
        )

        self.q_proj = nn.Linear(
            model_args.dim,
            model_args.n_heads * self.head_dim,
            bias="q_proj" in model_args.biases,
        )
        self.q_norm = (
            build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
                casting_mode=model_args.hf_config.model_type,
            )
            if model_args.q_k_norm_enabled
            else None
        )

        self.k_proj = nn.Linear(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias="k_proj" in model_args.biases,
        )
        self.k_norm = (
            build_norm(
                model_args.norm_type,
                dim=self.head_dim,
                eps=model_args.norm_eps,
                casting_mode=model_args.hf_config.model_type,
            )
            if model_args.q_k_norm_enabled
            else None
        )

        self.v_proj = nn.Linear(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias="v_proj" in model_args.biases,
        )
        self.o_proj = nn.Linear(
            model_args.n_heads * self.head_dim,
            model_args.dim,
            bias="o_proj" in model_args.biases,
        )
        self.rope_func = rope.RotaryPositionEmbedding()

        for module in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.q_norm,
            self.k_norm,
        ]:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor],
        vision_token_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            position_embeddings (torch.Tensor): Position embeddings.
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for variable-length sequences.
            max_seqlen (int, optional): Maximum sequence length for variable-length sequences.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape

        xq = torch.zeros(
            [bs, seqlen, self.head_dim * self.n_heads], device=x.device, dtype=x.dtype
        )
        xk = torch.zeros(
            [bs, seqlen, self.head_dim * self.n_kv_heads],
            device=x.device,
            dtype=x.dtype,
        )
        xv = torch.zeros(
            [bs, seqlen, self.head_dim * self.n_kv_heads],
            device=x.device,
            dtype=x.dtype,
        )

        # Separate the query, key, value for text and vision tokens
        # print(f"x[~vision_token_mask].view(-1, self.n_heads * self.head_dim)): {x[~vision_token_mask].view(-1, self.n_heads * self.head_dim).shape}")
        xq[~vision_token_mask] = self.q_norm(
            self.q_proj(x[~vision_token_mask].view(-1, self.dim)).view(
                -1, self.n_heads, self.head_dim
            )
        ).view(-1, self.head_dim * self.n_heads)
        xk[~vision_token_mask] = self.k_norm(
            self.k_proj(x[~vision_token_mask].view(-1, self.dim)).view(
                -1, self.n_kv_heads, self.head_dim
            )
        ).view(-1, self.head_dim * self.n_kv_heads)
        xv[~vision_token_mask] = self.v_proj(x[~vision_token_mask])

        xq[vision_token_mask] = self.mm_q_norm(
            self.mm_q_proj(x[vision_token_mask].view(-1, self.dim)).view(
                -1, self.n_heads, self.head_dim
            )
        ).view(-1, self.head_dim * self.n_heads)
        xk[vision_token_mask] = self.mm_k_norm(
            self.mm_k_proj(x[vision_token_mask].view(-1, self.dim)).view(
                -1, self.n_kv_heads, self.head_dim
            )
        ).view(-1, self.head_dim * self.n_kv_heads)
        xv[vision_token_mask] = self.mm_v_proj(x[vision_token_mask])

        # xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # if self.q_norm is not None:
        #     xq = self.q_norm(xq.view(bs, seqlen, -1, self.head_dim))
        # if self.k_norm is not None:
        #     xk = self.k_norm(xk.view(bs, seqlen, -1, self.head_dim))

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = self.rope_func(xq, xk, cos, sin, unsqueeze_dim=2)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                raise ValueError("Flash attention only supports float32 input")
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        if cu_seqlens is not None:
            assert (
                max_seqlen is not None
            ), "max_seqlen must be provided for variable-length sequences"
            xq = xq.view(seqlen, -1, self.head_dim)
            xk = xk.view(seqlen, -1, self.head_dim)
            xv = xv.view(seqlen, -1, self.head_dim)
            output = self.attn_func_varlen(
                xq,
                xk,
                xv,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=True,
            )
        else:
            output = self.attn_func(xq, xk, xv, causal=True)
        output = output.view(bs, seqlen, -1)

        o_output = torch.zeros([bs, seqlen, self.dim], device=x.device, dtype=x.dtype)
        o_output[~vision_token_mask] = self.o_proj(output[~vision_token_mask])
        o_output[vision_token_mask] = self.mm_o_proj(output[vision_token_mask])
        return o_output


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        model_args: GPTArgs,
    ):
        super().__init__()
        self.ffn_dim = hidden_dim
        self.dim = dim
        self.up_proj = nn.Linear(dim, hidden_dim, bias="up_proj" in model_args.biases)
        self.mm_up_proj = nn.Linear(
            dim, hidden_dim, bias="mm_up_proj" in model_args.biases
        )
        self.down_proj = nn.Linear(
            hidden_dim, dim, bias="down_proj" in model_args.biases
        )
        self.mm_down_proj = nn.Linear(
            hidden_dim, dim, bias="mm_down_proj" in model_args.biases
        )
        self.gate_proj = nn.Linear(
            dim, hidden_dim, bias="gate_proj" in model_args.biases
        )
        self.mm_gate_proj = nn.Linear(
            dim, hidden_dim, bias="mm_gate_proj" in model_args.biases
        )
        self.act_mul_func = MLPActMulFunc(nn.SiLU())

        for module in [self.up_proj, self.down_proj, self.gate_proj]:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, x, vision_token_mask: Optional[torch.Tensor] = None):
        bs, seqlen, _ = x.shape
        x_output = torch.zeros([bs, seqlen, self.dim], device=x.device, dtype=x.dtype)
        x_output[~vision_token_mask] = self.down_proj(
            self.act_mul_func(
                self.gate_proj(x[~vision_token_mask]),
                self.up_proj(x[~vision_token_mask]),
            )
        )
        x_output[vision_token_mask] = self.mm_down_proj(
            self.act_mul_func(
                self.mm_gate_proj(x[vision_token_mask]),
                self.mm_up_proj(x[vision_token_mask]),
            )
        )
        return x_output
        # return self.down_proj(self.act_mul_func(self.gate_proj(x), self.up_proj(x)))


def modulate(
    x: torch.Tensor,
    shift: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    dim: int = 1,
):
    """
    x: [..., D]
    shift, scale: shape broadcastable to x, expand along dim (sequence dim or batch dim depending on caller).
    - If shift/scale is None, treat as 0 / 0 respectively (i.e., identity).
    """
    if scale is None and shift is None:
        return x
    # We want: (1 + scale) * x + shift
    if scale is not None:
        x = x * (1 + scale.unsqueeze(dim))
    if shift is not None:
        x = x + shift.unsqueeze(dim)
    return x


class GPTBlock(nn.Module):
    """
    GPTBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (GPTArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: GPTArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.self_attn = Attention(model_args)
        self.mlp = FeedForward(
            dim=model_args.dim,
            hidden_dim=model_args.ffn_dim,
            model_args=model_args,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.input_layernorm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
        )
        self.post_attention_layernorm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
        )
        self.mm_input_layernorm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
        )
        self.mm_post_attention_layernorm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
        )
        self.mm_adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(model_args.dim, 2 * model_args.dim, bias=True)
        )
        self.t_dim = 64
        self.t_proj = nn.Linear(self.t_dim, model_args.dim, bias=False)
        # buffer of frequencies (shared across dtypes; cast at use time)
        half = 32
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half).float() / half)
        self.register_buffer("t_freqs", freqs, persistent=False)

        for module in [self.input_layernorm, self.post_attention_layernorm]:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        vision_token_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GPTBlock.

        Args:
            x (torch.Tensor): Input tensor.
            position_embeddings (torch.Tensor): Position embeddings.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        bs, seqlen, _ = x.shape
        x_input = torch.zeros([bs, seqlen, self.dim], device=x.device, dtype=x.dtype)
        x_input[~vision_token_mask] = self.input_layernorm(x[~vision_token_mask])

        mods = self.mm_adaln(
            self.time_embed(timestep, self.dim, x.device, x.dtype)
        ).chunk(2, dim=1)
        shift_msa, scale_msa = mods

        before_mod = self.mm_input_layernorm(x[vision_token_mask]).view(
            x.shape[0], -1, self.dim
        )
        after_mod = modulate(before_mod, shift_msa, scale_msa, dim=1)
        x_input[vision_token_mask] = after_mod.view(-1, self.dim)
        h = x + self.self_attn(
            x_input,
            position_embeddings,
            vision_token_mask,
            cu_seqlens=kwargs.get("cu_seqlens", None),
            max_seqlen=kwargs.get("max_seqlen", None),
        )
        h_post_attention = torch.zeros(
            [bs, seqlen, self.dim], device=x.device, dtype=x.dtype
        )
        h_post_attention[~vision_token_mask] = self.post_attention_layernorm(
            h[~vision_token_mask]
        )
        h_post_attention[vision_token_mask] = self.mm_post_attention_layernorm(
            h[vision_token_mask]
        )
        out = h + self.mlp(h_post_attention, vision_token_mask)
        return out

    def time_embed(self, t: torch.Tensor, dim: int, device, dtype):
        freqs = self.t_freqs.to(device=device, dtype=dtype)
        ang = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)  # [B, 64]
        return self.t_proj(emb)  # [B, hidden_size_img]


@ModelRegistry.register(GPTWeightMapper)
class GPT(BaseModel):
    """
    GPT Module

    Args:
        model_args (GPTArgs): Model configuration arguments.

    Attributes:
        model_args (GPTArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        embed_tokens (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of GPT blocks.
        norm (RMSNorm): Layer normalization for the model output.
        lm_head (ColumnParallelLinear): Linear layer for final output.
    """

    @staticmethod
    def supported_model_types():
        return ["llama", "qwen2", "qwen3"]

    def __init__(self, model_args: GPTArgs):
        super().__init__(model_args.hf_config)

        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.rotary_emb = RotaryEmbedding(model_args)

        self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = GPTBlock(layer_id, model_args)

        self.norm = build_norm(
            model_args.norm_type,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            casting_mode=model_args.hf_config.model_type,
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

        for module in [
            self.embed_tokens,
            self.lm_head if hasattr(self, "lm_head") else None,
            self.norm,
        ]:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)

        self.vae = (
            AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae"
            )
            .eval()
            .to(torch.float32)
        )
        ENCODER_PATCH_SIZE = 2
        COND_DIM = 1536
        N_LAYERS = 32
        # TODO: Change to 2048
        COND_LEN = self.COND_LEN = 2048
        VAE_SCALE_IN_SPATIAL = 8
        ENCODER_TIME_ADALN = True
        vae_latent_channels = self.vae.config.latent_channels
        IMG_SIZE = 384

        self.encoder = QformerEncoder(
            patch_size=ENCODER_PATCH_SIZE,
            hidden_size=COND_DIM,
            num_heads=4,
            depth=N_LAYERS,
            K=COND_LEN,
            query_dim=COND_DIM,
            query_heads=8,
            bidirectional=False,
            in_channels=vae_latent_channels,
            input_size=IMG_SIZE // VAE_SCALE_IN_SPATIAL,
            gradient_checkpointing=False,
            time_adaln=ENCODER_TIME_ADALN,
        )
        # Load checkpoint of encoder
        # TODO(cjx): Change to local path
        # encoder_checkpoint = torch.load("/lustre/fsw/portfolios/sw/users/jiaxinc/.cache//huggingface/hub/models--Jiaxincc--naruto-beta/snapshots/bbae08ec14251afb4fdc8fcfc6a27cf0b8f9b402/latest_xl.pt", map_location="cpu")
        # self.encoder.load_state_dict(encoder_checkpoint["modelA"])
        from huggingface_hub import hf_hub_download

        # Download checkpoint from Hugging Face Hub at a given revision
        checkpoint_path = hf_hub_download(
            repo_id="Jiaxincc/naruto-beta",  # HF repo
            filename="latest_xl.pt",  # file inside repo
            revision="10489cf9848d87667455c6fcb46303a11a2c4e1d",  # commit hash / tag / branch
        )
        # Load encoder checkpoint
        encoder_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.encoder.load_state_dict(encoder_checkpoint["modelA"])
        logger.info(f"Loaded encoder checkpoint from {checkpoint_path}")
        self.mm_proj = nn.Linear(
            COND_DIM,
            model_args.dim,
        )
        self.COND_DIM = COND_DIM
        self.mm_transformer = DiffusionCondAttentionNet(
            d_in=COND_DIM,
            d_cond=model_args.dim,
            d_model=256,
            n_heads=4,
            d_ff=512,
            num_layers=3,
            x_tokens=32,
            cond_tokens=32,
            use_timestep=True,
        )
        self.mm_head = nn.Sequential(
            nn.Linear(model_args.dim, model_args.dim),
            nn.SiLU(),
            nn.Linear(model_args.dim, COND_DIM),
        )

        for param in self.vae.parameters():
            param.requires_grad_(False)
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.counter = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        imgs: torch.Tensor = None,  # [B, C, H, W]
        interested_tokens: Optional[torch.BoolTensor] = None,
        inference_mode: bool = False,
        *args,
        **kwargs,
    ):
        vision_start_id = 151652
        vision_end_id = 151653
        image_token_id = 151655
        # TODO(cjx)
        SCALE = 2.0
        # TODO: Remove this line
        # input_ids[:, -128:] = image_token_id
        # logger.info(f"imgs: {imgs.shape}, range: {imgs.min()}, {imgs.max()}")
        img_z1 = vae_encode_mode(self.vae, imgs)
        # [B, L, D]
        encoded_img = self.encoder(img_z1) * SCALE
        if inference_mode:
            with torch.no_grad():
                vision_token_includes_start_end = (
                    (input_ids == vision_start_id)
                    | (input_ids == vision_end_id)
                    | (input_ids == image_token_id)
                )
                inputs_embeds = self.embed_tokens(input_ids)
                vision_token_mask = input_ids == image_token_id
                position_embeddings = self.rotary_emb(
                    inputs_embeds, position_ids.to(dtype=torch.long)
                )
                noise = torch.randn(
                    [input_ids.shape[0], self.COND_LEN, self.COND_DIM],
                    device=input_ids.device,
                )
                begin_t = 0.3
                noise = begin_t * encoded_img + (1 - begin_t) * noise
                torch.save(noise, f"origin_mix_noise_{os.environ['RANK']}.pt")
                torch.save(
                    encoded_img, f"origin_mix_encoded_img_{os.environ['RANK']}.pt"
                )
                for timestep in torch.linspace(begin_t, 1, 100):
                    t_tensor = torch.tensor([timestep], device=input_ids.device).view(
                        input_ids.shape[0], 1, 1
                    )
                    inputs_embeds[vision_token_mask] = self.mm_proj(noise).view(
                        -1, self.model_args.dim
                    )
                    h = self.identity_layer(inputs_embeds)
                    for layer in self.layers.values():
                        h = layer(
                            h,
                            position_embeddings,
                            vision_token_includes_start_end,
                            t_tensor.view(-1),
                        )
                    velocity = self.mm_head(h[vision_token_mask]).view(noise.shape)
                    noise = noise + velocity * 0.01
                # save to localfile
                torch.save(noise, f"noise_{os.environ['RANK']}.pt")
        else:
            vision_token_includes_start_end = (
                (input_ids == vision_start_id)
                | (input_ids == vision_end_id)
                | (input_ids == image_token_id)
            )
            vision_token_mask = input_ids == image_token_id

            if self.embed_tokens is not None:
                inputs_embeds = self.embed_tokens(input_ids)
                # Do not remove this line
                # This is a trick for TP with torch.compile
                h = self.identity_layer(inputs_embeds)
            else:
                inputs_embeds = input_ids
                h = inputs_embeds

            # Tuple of (cos, sin)
            # cos: [B, seq_len, head_dim]
            position_embeddings = self.rotary_emb(h, position_ids.to(dtype=torch.long))

            # TODO: Process the vision tokens
            # Causal Encoder(image) -> image tokens
            # [N_IMAGES, L, D] -> [N_IMAGES, L, Hidden_dim]

            # Fake data
            # imgs = torch.randn(input_ids.shape[0], 3, 384, 384).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

            timestep = torch.rand([input_ids.shape[0], 1, 1], device=input_ids.device)
            noise = torch.randn_like(encoded_img)
            target = encoded_img - noise
            inputs_embeds[vision_token_mask] = (
                self.mm_proj(encoded_img * timestep + noise * (1 - timestep))
                .view(input_ids.shape[0], -1, self.model_args.dim)
                .view(-1, self.model_args.dim)
            )

            for layer in self.layers.values():
                h = torch.utils.checkpoint.checkpoint(
                    layer,
                    h,
                    position_embeddings,
                    vision_token_includes_start_end,
                    timestep.view(-1),
                    **kwargs,
                    use_reentrant=False,
                )

            loss = torch.nn.functional.mse_loss(
                self.mm_head(h[vision_token_mask]).view(target.shape),
                target,
            )
            return None, loss

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        # rotary.inv_freq could get deleted and not re-initialized
        # so we need to delete it manually
        self.rotary_emb.to(torch.cuda.current_device())
        self.rotary_emb.reset_inv_freq()

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.gpt.parallelize import parallelize

        return parallelize, self

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
        n_layers = len(self.layers)
        layers_per_stage = n_layers // pp_size

        if not is_first:
            self.embed_tokens = None
        if not is_last:
            self.lm_head = None
            self.norm = None

        local_layers = torch.nn.ModuleDict()
        for i in range(
            pp_rank * layers_per_stage,
            ((pp_rank + 1) * layers_per_stage) if not is_last else n_layers,
        ):
            local_layers[str(i)] = self.layers[str(i)]

        # Reset the layers for pipeline splitting
        self.layers = local_layers

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
            info_inly (bool): Only collect the tensor infomation without actual data loading.
        """
        # Load all safetensors from `model_path`
        model_type = retry(AutoConfig.from_pretrained)(model_name_or_path).model_type
        model_path = resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        self_state_dict = self.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}
        lm_head_weight_key = "lm_head.weight"
        embed_tokens_weight_key = "model.embed_tokens.weight"
        weights_of_ckpt_names = set()
        reserved = {}
        for f in safetensors_files:
            weights_of_ckpt = {}
            ckpt = safe_open(
                os.path.join(model_path, f), framework="pt", device=str(device)
            )
            keys = ckpt.keys()
            for name in keys:
                ckpt_tensor = ckpt.get_tensor(name)
                weights_of_ckpt[name] = ckpt_tensor
                weights_of_ckpt_names.add(name)
                if name == embed_tokens_weight_key:
                    reserved[name] = ckpt_tensor

            for name in weights_of_ckpt.keys():
                tensor = weights_of_ckpt[name]
                dest_name, shared_weight = convert_weight_from_hf(
                    tensor, name, model_type, parallel_dims
                )
                if dest_name not in self_state_dict and parallel_dims.pp_enabled:
                    # logger.info(f"Weight `{dest_name}` is discarded, maybe due to pipeline parallelism. Skipping this weight checking")
                    continue

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

        if (
            lm_head_weight_key not in weights_of_ckpt_names
            and embed_tokens_weight_key in weights_of_ckpt_names
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
        seq_dim_idx = 1
        inputs = kwargs["input_ids"]
        position_ids = (
            torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
            .unsqueeze(0)
            .expand_as(inputs)
        )
        return position_ids, inputs, seq_dim_idx

    def separate_model_parts(self) -> List[nn.Module]:
        return [self]

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

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        return self._get_nparams_and_flops_fn(seq_len)

    @classmethod
    def from_model_args(cls, model_args: GPTArgs) -> "GPT":
        """
        Initialize a GPT model from a GPTArgs object.

        Args:
            model_args (GPTArgs): Model configuration arguments.

        Returns:
            GPT: GPT model.

        """
        return cls(model_args)

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "GPT":
        """
        Initialize a GPT model from a pretrained model.

        Args:
            model_name_or_path (str): Model name or path to the pretrained model.

        Returns:
            GPT: GPT model.

        """
        if hf_config.model_type not in cls.supported_model_types():
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        if max_position_embeddings is None:
            max_position_embeddings = hf_config.max_position_embeddings
        else:
            hf_config.max_position_embeddings = max_position_embeddings

        vocab_size = sync_model_vocab(model_name_or_path)

        rope_scaling = {}
        if hasattr(hf_config, "rope_scaling"):
            rope_scaling = hf_config.rope_scaling or {}
        rope_type = rope_scaling.get("rope_type", "default")

        bias_list = {
            "qwen2": ["q_proj", "k_proj", "v_proj"],
            "llama": [],
            "qwen3": [],
        }[hf_config.model_type]

        try:
            head_dim = hf_config.head_dim
        except Exception:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            logger.warning(f"head_dim not found in config, using {head_dim}")

        model = cls.from_model_args(
            GPTArgs(
                dim=hf_config.hidden_size,
                ffn_dim=hf_config.intermediate_size,
                n_layers=hf_config.num_hidden_layers,
                n_heads=hf_config.num_attention_heads,
                n_kv_heads=hf_config.num_key_value_heads,
                head_dim=head_dim,
                vocab_size=vocab_size,
                max_seq_len=max_position_embeddings,
                rope_theta=hf_config.rope_theta,
                q_k_norm_enabled=hf_config.model_type == "qwen3",
                norm_type="rmsnorm",
                rope_type=rope_type,
                biases=bias_list,
                hf_config=hf_config,
            )
        )
        return model

    @classmethod
    def fqn_filter_for_fp8(cls) -> List[str]:
        return ["lm_head"]

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        if not (self.model_args.n_heads % (cp_size * tp_size) == 0):
            raise ValueError(
                f"Model is not compatible with cp parallelism, model's head number={self.model_args.n_heads} is not divisible by cp size({cp_size}) * tp_size({tp_size}) = {cp_size * tp_size}"
            )
