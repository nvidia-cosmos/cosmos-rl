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
import torch
from torch import nn
import torch.nn.functional as F
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
from flash_attn import flash_attn_func


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
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
    return RMSNorm(dim, eps)


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
        self.attn_func = flash_attn_func

        self.q_proj = nn.Linear(
            model_args.dim,
            model_args.n_heads * self.head_dim,
            bias="q_proj" in model_args.biases,
        )
        self.q_norm = (
            build_norm(model_args.norm_type, dim=self.head_dim, eps=model_args.norm_eps)
            if model_args.q_k_norm_enabled
            else None
        )

        self.k_proj = nn.Linear(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias="k_proj" in model_args.biases,
        )
        self.k_norm = (
            build_norm(model_args.norm_type, dim=self.head_dim, eps=model_args.norm_eps)
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

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            position_embeddings (torch.Tensor): Position embeddings.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.q_norm is not None:
            xq = self.q_norm(xq.view(bs, seqlen, -1, self.head_dim))
        if self.k_norm is not None:
            xk = self.k_norm(xk.view(bs, seqlen, -1, self.head_dim))

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        output = self.attn_func(xq, xk, xv, causal=True)
        output = output.view(bs, seqlen, -1)
        return self.o_proj(output)


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
        self.up_proj = nn.Linear(dim, hidden_dim, bias="up_proj" in model_args.biases)
        self.down_proj = nn.Linear(
            hidden_dim, dim, bias="down_proj" in model_args.biases
        )
        self.gate_proj = nn.Linear(
            dim, hidden_dim, bias="gate_proj" in model_args.biases
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.post_attention_layernorm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
    ):
        """
        Perform a forward pass through the GPTBlock.

        Args:
            x (torch.Tensor): Input tensor.
            position_embeddings (torch.Tensor): Position embeddings.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.self_attn(self.input_layernorm(x), position_embeddings)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


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
            h = self.embed_tokens(input_ids)
            # Do not remove this line
            # This is a trick for TP with torch.compile
            h = self.identity_layer(h)
        else:
            h = input_ids

        position_embeddings = self.rotary_emb(h, position_ids.to(dtype=torch.long))
        for layer in self.layers.values():
            h = layer(h, position_embeddings=position_embeddings)

        # Add `if` check just in case `pp` is enabled
        if self.norm is not None:
            h = self.norm(h)
            if not self.tie_embed_tokens:
                output = self.lm_head(h)
            else:
                is_w_dist_tensor = isinstance(
                    self.embed_tokens.weight, torch.distributed.tensor.DTensor
                )
                embed_tokens_weight = (
                    self.embed_tokens.weight.full_tensor()
                    if is_w_dist_tensor
                    else self.embed_tokens.weight
                )
                is_a_dist_tensor = isinstance(h, torch.distributed.tensor.DTensor)
                h = h.full_tensor() if is_a_dist_tensor else h
                output = h @ embed_tokens_weight.t()
            return output
        else:
            return h

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
