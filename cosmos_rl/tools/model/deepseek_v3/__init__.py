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

import re
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
from transformers import AutoConfig
import torch.distributed._symmetric_memory as symm_mem
from cosmos_rl.utils.util import (
    resolve_model_path,
    IdentityLayer,
    clear_weight_name,
    sync_model_vocab,
    retry,
)
from cosmos_rl.utils.logging import logger
from .weight_converter import (
    convert_weight_from_hf,
)
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.kernel.symm_mem_recipes import OnDeviceAllToAllV
from cosmos_rl.policy.kernel.moe.indices import generate_permute_indices
from cosmos_rl.policy.kernel.moe.grouped_gemm import group_gemm_imp
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import BaseModel
from transformers.activations import ACT2FN
from functools import cached_property


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def build_norm(norm_type: str, dim: int, eps: float):
    assert norm_type == "rmsnorm", f"Unknown norm_type: '{norm_type}'"
    return DeepseekV3RMSNorm(dim, eps)


@dataclass
class DeepseekV3MoeArgs:
    dim: int
    intermediate_size: int
    moe_intermediate_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    q_lora_rank: int
    kv_lora_rank: int
    v_head_dim: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    n_routed_experts: int
    n_shared_experts: int
    routed_scaling_factor: float
    scoring_func: str
    seq_aux: bool
    topk_method: str
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    moe_layer_freq: int
    biases: List[str] = field(default_factory=lambda: [])
    attention_bias: bool = False
    norm_topk_prob: bool = True
    rope_scaling: dict = None
    norm_eps: float = 1e-5
    rope_theta: float = 50000
    norm_type: str = "rmsnorm"
    rope_type: str = "default"
    hidden_act: str = "silu"
    hf_config: AutoConfig = None


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.max_seq_len_cached = None

    def reset_rotary_cache(self, device=None):
        if hasattr(self, "inv_freq"):
            return

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3MoeArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.max_position_embeddings = config.max_seq_len
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.attn_func = None

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            # scaling_factor = self.config.rope_scaling["factor"]
            # if scaling_type == "linear":
            #     self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
            #         self.qk_rope_head_dim,
            #         max_position_embeddings=self.max_position_embeddings,
            #         scaling_factor=scaling_factor,
            #         base=self.rope_theta,
            #     )
            # elif scaling_type == "dynamic":
            #     self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
            #         self.qk_rope_head_dim,
            #         max_position_embeddings=self.max_position_embeddings,
            #         scaling_factor=scaling_factor,
            #         base=self.rope_theta,
            #     )
            # elif scaling_type == "yarn":
            #     kwargs = {
            #         key: self.config.rope_scaling[key]
            #         for key in [
            #             "original_max_position_embeddings",
            #             "beta_fast",
            #             "beta_slow",
            #             "mscale",
            #             "mscale_all_dim",
            #         ]
            #         if key in self.config.rope_scaling
            #     }
            #     self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            #         self.qk_rope_head_dim,
            #         max_position_embeddings=self.max_position_embeddings,
            #         scaling_factor=scaling_factor,
            #         base=self.rope_theta,
            #         **kwargs,
            #     )
            # else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.view(bsz, q_len, -1, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, -1, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        self.rotary_emb.reset_rotary_cache(device=value_states.device)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        k_pe_expanded = k_pe.expand(-1, k_nope.shape[1], -1, -1)
        key_states = torch.cat([k_nope, k_pe_expanded], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class FakeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV3MLPWithEP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None, experts_num=1):
        super().__init__()
        self.config = config
        self.experts_num = experts_num
        self.hidden_size = config.dim if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        # Transpose the MLP for group gemm
        self.gate_proj = FakeLinear(
            self.hidden_size, self.intermediate_size, self.experts_num
        )
        self.up_proj = FakeLinear(
            self.hidden_size, self.intermediate_size, self.experts_num
        )
        self.down_proj = FakeLinear(
            self.intermediate_size, self.hidden_size, self.experts_num
        )
        self.act_fn = ACT2FN[config.hidden_act]


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.hf_config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            # assert not self.training
            scores_for_choice = scores.view(
                bsz * seq_len, -1
            ) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(
                ~score_mask.bool(), 0.0
            )  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = (
            topk_weight * self.routed_scaling_factor
        )  # must multiply the scaling factor

        return topk_idx, topk_weight


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    token_send_buf: Optional[torch.Tensor] = None
    token_gather_buf: Optional[torch.Tensor] = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.hf_config.num_experts_per_tok
        self.local_experts = config.n_routed_experts
        self.total_experts = config.n_routed_experts
        self.local_to_dtensor = IdentityLayer()
        self.reshard_helper_layer = IdentityLayer()
        self.gate = MoEGate(config)
        self.ep_size = 1
        self.experts_per_rank = config.n_routed_experts
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for i in range(config.n_routed_experts)
            ]
        )
        self.experts_ep = DeepseekV3MLPWithEP(
            config,
            intermediate_size=config.moe_intermediate_size,
            experts_num=config.n_routed_experts,
        )
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                config=config, intermediate_size=intermediate_size
            )
        self.group_gemm_imp = group_gemm_imp()

    def sort_tokens(self, x, topk_ids, topk_weights):
        # This part sorts the token indices so that tokens routed to the same expert reside consecutively.
        # An implication is that tokens to the same "expert group" (i.e., device) are also consecutive.
        # Since this is an "aritificial" index creation (final outcome being
        # `idxs`), we don't need gradients here.

        with torch.no_grad():
            # [seq_len, n_routed_experts]
            expert_counts = topk_ids.new_zeros((topk_ids.shape[0], self.total_experts))
            # Fill 1 to the selected experts
            expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = expert_counts.sum(dim=0)
            # Token indices for each expert
            token_indices = topk_ids.view(-1).argsort()

        sorted_tokens = x[token_indices // topk_ids.shape[1]]
        # assert sorted_tokens.shape == sorted_tokens_shape

        return (sorted_tokens, token_indices, tokens_per_expert)

    def get_send_buf(self):
        # [Why detach?] During a first forward-backward step, the buffer would
        # be included in a computational graph. In a second step, autograd will
        # return an error saying "Trying to backward through the graph a second
        # time (or directly access saved tensors more than once)". This is
        # because the buffer is still in the graph, and autograd is trying to
        # backward through the graph a second time. To avoid this, we detach the
        # buffer from the graph. `detach()` returns a new tensor, which shares
        # the same storage with the original one.
        self.token_send_buf.grad = None
        return self.token_send_buf.detach()

    def get_gather_buf(self):
        # See [Why detach?] in `get_send_buf`
        self.token_gather_buf.grad = None
        return self.token_gather_buf.detach()

    def moe_infer_ep(self, x, topk_ids, topk_weight):
        """
        x: [batch * local_seq_len, dim]
        topk_ids: [batch * local_seq_len, topk]
        topk_weight: [batch * local_seq_len, topk]

        sorted_tokens: [batch * local_seq_len * topk, dim]
        token_indices: [batch * local_seq_len * topk]
        tokens_per_expert: [n_experts]
        """
        (
            sorted_tokens,
            token_indices,
            tokens_per_expert,
        ) = self.sort_tokens(x, topk_ids, topk_weight)
        # keep the seqlen dimension for later use without holding onto the sorted tokens
        seqlen_sorted_tokens = sorted_tokens.shape[0]

        # Sum the tokens over local experts, then we get tokens per EP rank,
        # which is the input splits
        with torch.no_grad():
            # tokens_per_expert: [n_experts, 1]
            # tokens_per_expert_group: [n_experts, 1]
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )

            # For TP/EP mode, the input is sequencely parallelized
            # So each EP group will have distinct, but the same number of tokens
            # After this collective, tokens_per_expert_group is still of shape [n_experts, 1]

            # Let's say we are on EP rank 0:
            # recv: [(e0, e1, e2 ...), (e0, e1, e2 ...), ...], totally `n_experts` elements
            #        ----------------: tokens from EP group 0 to EP group 0
            #                          ----------------: tokens from EP group 1 to EP group 0
            #                          ...
            # So we can just concat
            dist.all_to_all_single(
                tokens_per_expert_group,
                tokens_per_expert,
                group=self.ep_group,
                async_op=False,
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        # Move input to the `token_send_buf` symm mem
        token_send_buf = self.get_send_buf()
        token_send_buf[: token_indices.shape[0]].copy_(sorted_tokens)
        # Note: `out=` avoids copy, but it is not differentiable
        # torch.index_select(x, 0, idxs // topk_ids.shape[1], out=token_send_buf[: idxs.shape[0]])

        # Reference:
        #   1. [TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/deepseek_v3/symm_mem_recipes/triton_on_device_all_to_all_v.py)
        #   2. [Symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes)
        token_gather_buf, output_splits = OnDeviceAllToAllV.apply(
            token_send_buf,
            input_splits,
            self.ep_group,
        )

        # We need to permute the received tokens so that tokens for the same expert are contiguous.
        # This part prepares a 1D tensor `permuted_indices` for such permutation.
        # This part doesn't need gradient.
        with torch.no_grad():
            ALIGN_SIZE_M = 128
            permuted_indices, m_sizes, m_offsets = generate_permute_indices(
                tokens_per_expert_group,
                self.local_experts,
                self.ep_size,
                token_gather_buf.shape[0],
                ALIGN_SIZE_M,
            )
        # Permute the received tokens so that tokens for the same expert are contiguous.
        contig_tokens = token_gather_buf[permuted_indices]

        # group gemm - handle all three group gemms (up, gate, down for all experts)
        # print(f"m_sizes: {m_sizes}, m_offsets: {m_offsets}")
        hidden_outputs = self.group_gemm_imp(
            contig_tokens,
            m_sizes,
            m_offsets,
            self.experts_ep.gate_proj.weight.to_local(),
            self.experts_ep.up_proj.weight.to_local(),
            self.experts_ep.down_proj.weight.to_local(),
            self.experts_ep.act_fn,
        )

        # Prepare buffer for tokens processed by experts
        processed_tokens = self.get_gather_buf()

        # Move into Symmetric Memory for the return shuffle
        processed_tokens[permuted_indices] = hidden_outputs

        # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
        # The input/output splits are just a reverse of the previous shuffle.
        token_return_buf, _ = OnDeviceAllToAllV.apply(
            processed_tokens,
            output_splits,
            self.ep_group,
        )

        returned_tokens = token_return_buf[:seqlen_sorted_tokens]
        output_tokens = torch.empty_like(returned_tokens)
        output_tokens[token_indices] = returned_tokens

        final_out = (
            output_tokens.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(returned_tokens.dtype)
        )

        return final_out

    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.config.n_routed_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.ep_size > 1:
            y = self.moe_infer_ep(hidden_states, topk_idx, topk_weight).view(
                *orig_shape
            )
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        # y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return self.reshard_helper_layer(y)


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3MoeArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = config.dim
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)
        self.use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )

        self.mlp = DeepseekV3MoE(config) if self.use_moe else DeepseekV3MLP(config)

        self.input_layernorm = DeepseekV3RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            config.dim, eps=config.norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        # Self Attention
        h = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
        )
        # Fully Connected
        output = self.mlp(self.post_attention_layernorm(h))
        output = h + output
        return output


class DeepseekV3MoEModel(BaseModel):
    """
    DeepseekV3MoEModel Module

    Args:
        model_args (DeepseekV3MoeArgs): Model configuration arguments.

    Attributes:
        model_args (DeepseekV3MoeArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        embed_tokens (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of DeepseekV3MoE blocks.
        norm (RMSNorm): Layer normalization for the model output.
        lm_head (ColumnParallelLinear): Linear layer for final output.
    """

    @staticmethod
    def supported_model_types():
        return ["deepseek_v3"]

    def __init__(self, model_args: DeepseekV3MoeArgs):
        super().__init__(model_args.hf_config)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(model_args, layer_idx)
                for layer_idx in range(model_args.n_layers)
            ]
        )
        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
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

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_ids)
            # Do not remove this line
            # This is a trick for TP with torch.compile
            hidden_states = self.identity_layer(hidden_states)
        else:
            hidden_states = input_ids

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
            )

        # Add `if` check just in case `pp` is enabled
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
            if not self.tie_embed_tokens:
                output = self.lm_head(hidden_states)
            else:
                is_w_dist_tensor = isinstance(
                    self.embed_tokens.weight, torch.distributed.tensor.DTensor
                )
                embed_tokens_weight = (
                    self.embed_tokens.weight.full_tensor()
                    if is_w_dist_tensor
                    else self.embed_tokens.weight
                )
                is_a_dist_tensor = isinstance(
                    hidden_states, torch.distributed.tensor.DTensor
                )
                hidden_states = (
                    hidden_states.full_tensor() if is_a_dist_tensor else hidden_states
                )
                output = hidden_states @ embed_tokens_weight.t()
            return output
        else:
            return hidden_states

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        for layer in self.layers:
            if layer.use_moe:
                layer.mlp.gate.weight.requires_grad_(False)

        # rotary.inv_freq could get deleted and not re-initialized
        # so we need to delete it manually
        # self.rotary_emb.reset_inv_freq()
        # Basically, max_seq_len * 2 is enough for all-to-all-v communication.
        overflow = 2

        # TODO(cjx): max_seq_len * mini_batch is a better choice
        MAX_BATCH_MUL_SEQ_LEN = (
            self.model_args.max_seq_len
            * cosmos_config.train.train_batch_per_replica
            * self.model_args.hf_config.num_experts_per_tok
        )

        OnDeviceAllToAllV.max_output_len = MAX_BATCH_MUL_SEQ_LEN * overflow
        # Init MoE kernel related buffers
        if DeepseekV3MoE.token_send_buf is None:
            dtype = self.model_args.hf_config.torch_dtype

            # Input buffer for DP-to-EP shuffle
            DeepseekV3MoE.token_send_buf = symm_mem.empty(
                MAX_BATCH_MUL_SEQ_LEN,
                self.model_args.dim,  # hidden dim
                dtype=dtype,
                device=self.current_device(),
            )
            DeepseekV3MoE.token_send_buf.zero_()
            # Input buffer for EP-to-DP shuffle
            DeepseekV3MoE.token_gather_buf = symm_mem.empty(
                MAX_BATCH_MUL_SEQ_LEN * overflow,
                self.model_args.dim,  # hidden dim
                dtype=dtype,
                device=self.current_device(),
            )
            DeepseekV3MoE.token_gather_buf.zero_()

    @property
    def parallelize_fn(self):
        from .parallelize import parallelize

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
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_path (str): Path to the HuggingFace model.
            parallel_dims (ParallelDims): Parallel dimensions definition.
            device (torch.device): Device to load the weights.
        """
        # Load all safetensors from `model_path`
        model_type = retry(AutoConfig.from_pretrained)(model_name_or_path).model_type
        model_path = resolve_model_path(model_name_or_path)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        self_state_dict = self.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}

        lm_head_weight_key = "lm_head.weight"
        embed_tokens_weight_key = "model.embed_tokens.weight"
        weights_of_ckpt_names = set()
        reserved = {}
        _, tp_ep_size = parallel_dims.tp_coord

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
                    tensor,
                    name,
                    model_type,
                    parallel_dims,
                    n_experts=self.model_args.n_routed_experts,
                )

                if dest_name is None:
                    # This is due to the expert parallelism grouping
                    continue

                expert_id = None
                if match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)",
                    dest_name,
                ):
                    expert_id = int(match.group(2))
                    if tp_ep_size > 1:
                        dest_name = dest_name.replace(
                            f"experts.{expert_id}.", "experts_ep."
                        )

                    # Convert expert_id to local_expert_id
                    n_local_experts = (
                        self.model_args.n_routed_experts
                        // parallel_dims.tp
                        // parallel_dims.dp_shard
                    )

                    expert_id = expert_id % n_local_experts

                if dest_name not in self_state_dict and parallel_dims.pp_enabled:
                    logger.info(
                        f"Weight `{dest_name}` is discarded, maybe due to pipeline parallelism or expert parallelism grouping. Skipping this weight checking"
                    )
                    continue

                target_tensor = self_state_dict[dest_name]
                if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                    target_tensor = target_tensor.to_local()
                # Write to the correct expert of the target tensor
                if expert_id is not None and tp_ep_size > 1:
                    target_tensor = target_tensor[expert_id]

                assert (
                    target_tensor.shape == shared_weight.shape
                ), f"Shape mismatch: {target_tensor.shape} != {shared_weight.shape} for {dest_name}"
                with torch.no_grad():
                    target_tensor.data.copy_(shared_weight)

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

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, int]:
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
    def from_model_args(cls, model_args: DeepseekV3MoeArgs) -> "DeepseekV3MoEModel":
        """
        Initialize a DeepseekV3MoE model from a DeepseekV3MoeArgs object.

        Args:
            model_args (DeepseekV3MoeArgs): Model configuration arguments.

        Returns:
            DeepseekV3MoE: DeepseekV3MoE model.

        """
        return cls(model_args)

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "DeepseekV3MoEModel":
        """
        Initialize a DeepseekV3MoE model from a pretrained model.

        Args:
            model_name_or_path (str): Model name or path to the pretrained model.

        Returns:
            DeepseekV3MoE: DeepseekV3MoE model.

        """
        try:
            if hf_config.model_type not in cls.supported_model_types():
                raise ValueError(f"Unsupported model type: {hf_config.model_type}")

            if max_position_embeddings is None:
                max_position_embeddings = hf_config.max_position_embeddings
            else:
                hf_config.max_position_embeddings = max_position_embeddings

            vocab_size = sync_model_vocab(model_name_or_path)
            rope_scaling = None
            if (
                hasattr(hf_config, "rope_scaling")
                and hf_config.rope_scaling is not None
            ):
                rope_scaling = hf_config.rope_scaling or {}
            rope_type = "default"
            if rope_scaling:
                rope_type = rope_scaling.get("rope_type", "default")

            # DeepseekV3MoE does not have any biases
            bias_list = []
            try:
                head_dim = hf_config.head_dim
            except Exception:
                head_dim = hf_config.hidden_size // hf_config.num_attention_heads
                logger.warning(f"head_dim not found in config, using {head_dim}")

            model = cls.from_model_args(
                DeepseekV3MoeArgs(
                    dim=hf_config.hidden_size,
                    intermediate_size=hf_config.intermediate_size,
                    moe_intermediate_size=hf_config.moe_intermediate_size,
                    n_layers=hf_config.num_hidden_layers,
                    n_heads=hf_config.num_attention_heads,
                    n_kv_heads=hf_config.num_key_value_heads,
                    head_dim=head_dim,
                    vocab_size=vocab_size,
                    max_seq_len=max_position_embeddings,
                    q_lora_rank=hf_config.q_lora_rank,
                    kv_lora_rank=hf_config.kv_lora_rank,
                    v_head_dim=hf_config.v_head_dim,
                    qk_nope_head_dim=hf_config.qk_nope_head_dim,
                    qk_rope_head_dim=hf_config.qk_rope_head_dim,
                    n_routed_experts=hf_config.n_routed_experts,
                    n_shared_experts=hf_config.n_shared_experts,
                    routed_scaling_factor=hf_config.routed_scaling_factor,
                    scoring_func=hf_config.scoring_func,
                    seq_aux=hf_config.seq_aux,
                    topk_method=hf_config.topk_method,
                    n_group=hf_config.n_group,
                    topk_group=hf_config.topk_group,
                    first_k_dense_replace=hf_config.first_k_dense_replace,
                    moe_layer_freq=hf_config.moe_layer_freq,
                    rope_theta=hf_config.rope_theta,
                    rope_type=rope_type,
                    rope_scaling=rope_scaling,
                    biases=bias_list,
                    attention_bias=hf_config.attention_bias,
                    norm_topk_prob=hf_config.norm_topk_prob,
                    hf_config=hf_config,
                )
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        return model

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        raise NotImplementedError(
            "Context Parallel is not supported for DeepseekV3MoEModel now."
        )
        # if not (self.model_args.n_heads % (cp_size * tp_size) == 0):
        #     raise ValueError(
        #         f"Model is not compatible with cp parallelism, model's head number={self.model_args.n_heads} is not divisible by cp size({cp_size}) * tp_size({tp_size}) = {cp_size * tp_size}"
        #     )
