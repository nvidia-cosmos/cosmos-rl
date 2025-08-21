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

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformer_engine.pytorch.attention import DotProductAttention
    from transformer_engine.pytorch.module.rmsnorm import RMSNorm as _RMSNorm
except ImportError:
    print("transformer_engine.pytorch is not available. DeepSeek model will not work.")

    class _RMSNorm:
        pass


from cosmos_rl.policy.kernel.moe.moe import MoE


@dataclass
class DeepseekConfig:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bfloat16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 1
    max_seq_len: int = 4096 * 4
    dtype: Literal["bfloat16", "fp8"] = "bfloat16"
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    train_gate: bool = True
    gate_bias_update_factor: float = 0.01
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5
    enable_deepep: bool = False
    fake_balanced_gate: bool = False
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    # aux loss
    aux_loss_coeff: float = 0.0


class RMSNorm(_RMSNorm):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(normalized_shape=dim, eps=eps)


def precompute_base_freqs(args: DeepseekConfig) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepseekConfig): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, args: DeepseekConfig):
        super().__init__()
        self.args = args

    def _reset_base_freqs(self, device: torch.device):
        with torch.device(device):
            freqs = precompute_base_freqs(self.args).float()
        assert not freqs.is_meta, "Cannot call precompute_freqs on meta device"
        self.register_buffer("freqs", freqs, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "freqs"):
            self._reset_base_freqs(position_ids.device)

        freqs = torch.matmul(
            position_ids.unsqueeze(-1).float(), self.freqs.unsqueeze(0)
        )
        freqs_cis = torch.polar(
            torch.ones_like(freqs, dtype=torch.float32), freqs.float()
        )
        return freqs_cis


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(self, args: DeepseekConfig):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.q_a_proj = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_module = DotProductAttention(
            num_attention_heads=args.n_heads,
            kv_channels=(
                args.qk_nope_head_dim + args.qk_rope_head_dim,
                args.v_head_dim,
            ),
            attn_mask_type="causal",
            qkv_format="bshd",
            softmax_scale=self.softmax_scale,
        )
        self.attn_func = self.attn_module.__call__

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, local_seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, local_seq_len, _ = x.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(bsz, local_seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)

        kv = self.kv_b_proj(kv)
        kv = kv.view(bsz, -1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2).expand([bsz, -1, self.n_heads, self.qk_rope_head_dim])
        k = torch.cat([k_nope, k_pe], dim=-1)

        x = self.attn_func(q, k, v)

        x = self.o_proj(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        input_layernorm (nn.Module): Layer normalization for attention.
        post_attention_layernorm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: DeepseekConfig):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (DeepseekConfig): Model arguments containing block parameters.
        """
        super().__init__()
        self.self_attn = MLA(args)
        if layer_id < args.n_dense_layers:
            self.mlp = MLP(args.dim, args.inter_dim)
        else:
            self.mlp = MoE(args)
        self.input_layernorm = RMSNorm(args.dim)
        self.post_attention_layernorm = RMSNorm(args.dim)
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            padding_mask (torch.Tensor): Boolean tensor indicating padding positions.

        Returns:
            torch.Tensor: Output tensor after block computation.
            torch.Tensor | None: Auxiliary loss for load balancing (if applicable).
        """

        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
        )
        x = x + attn_out

        mlp_out, aux_loss = self._mlp(
            x=self.post_attention_layernorm(x),
            padding_mask=padding_mask,
        )
        x = x + mlp_out

        return x, aux_loss

    def _mlp(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Applies the MLP layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            padding_mask (torch.Tensor): Boolean tensor indicating padding positions.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
            torch.Tensor | None: Auxiliary loss for load balancing (if applicable).
        """
        if isinstance(self.mlp, MLP):
            return self.mlp(x), None
        else:
            assert isinstance(self.mlp, MoE)
            return self.mlp(x, padding_mask)


class TransformerModel(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        embed_tokens (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleDict): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: DeepseekConfig):
        """
        Initializes the Transformer model.

        Args:
            args (DeepseekConfig): Model arguments containing transformer parameters.
        """
        # Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(args.n_layers):
            self.layers[str(layer_id)] = Block(layer_id, args)
        self.norm = RMSNorm(args.dim)
        self.rotary_emb = RotaryEmbedding(args)

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the Transformer model.

        Important assumption: All devices on the same CP rank are fed the same inputs.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, local_seq_len).
            position_ids (torch.Tensor): Starting position in the sequence for rotary embeddings. Defaults to 0.
            padding_mask (torch.Tensor | None): Boolean tensor which indicates which tokens are padding.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
            torch.Tensor | None: Auxiliary loss for load balancing (if applicable).
        """
        freqs_cis = self.rotary_emb(position_ids)

        h = self.embed_tokens(tokens) if self.embed_tokens else tokens

        # Apply the transformer layers.
        aux_losses = []
        for layer in self.layers.values():
            h, aux_loss = layer(
                x=h,
                freqs_cis=freqs_cis,
                padding_mask=padding_mask,
            )
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        final_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None

        h = self.norm(h) if self.norm else h
        return h, final_aux_loss


class Transformer(nn.Module):
    def __init__(self, args: DeepseekConfig):
        super().__init__()
        self.model = TransformerModel(args)
        self.lm_head = nn.Linear(
            args.dim, args.vocab_size, dtype=torch.get_default_dtype(), bias=False
        )

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the Transformer model.

        Important assumption: All devices on the same CP rank are fed the same inputs.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, local_seq_len).
            position_ids (torch.Tensor): Input tensor containing the indices of the tokens.
                The shape is (batch_size, local_seq_len).
            padding_mask (torch.Tensor): Boolean tensor which indicates which tokens are padding.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
            torch.Tensor | None: Auxiliary loss for load balancing (if applicable).
        """
        h, aux_loss = self.model(
            tokens=tokens,
            position_ids=position_ids,
            padding_mask=padding_mask,
        )
        logits = self.lm_head(h) if self.lm_head else h
        return logits, aux_loss
