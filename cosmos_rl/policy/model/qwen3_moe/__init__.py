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
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.model.qwen3_moe.weight_converter import (
    convert_weight_from_hf,
)
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.model.qwen3_moe.weight_mapper import Qwen3MoeWeightMapper
from cosmos_rl.utils.multi_rank_weight_loader import MultiRankWeightLoader
from cosmos_rl.policy.kernel.moe.moe import MoE, MoEArgs
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import ModelRegistry, BaseModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from functools import cached_property
from cosmos_rl.policy.kernel.modeling_utils import FlashAttnMeta
from cosmos_rl.policy.kernel.norm import RMSNorm
import cosmos_rl.policy.kernel.rope as rope
from cosmos_rl.utils.sequence_packing import pack_sequences_for_inputs


def build_norm(
    norm_type: str, dim: int, eps: float, casting_mode: Optional[str] = None
):
    assert norm_type == "rmsnorm", f"Unknown norm_type: '{norm_type}'"
    return RMSNorm(dim, eps, casting_mode=casting_mode)


@dataclass
class Qwen3MoeArgs:
    dim: int
    ffn_dim: int
    n_layers: int
    n_experts: int
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
    train_gate: bool = True
    gate_bias_update_factor: float = 0.0
    aux_loss_coeff: float = 0.0
    hf_config: AutoConfig = None


class RotaryEmbedding(nn.Module):
    def __init__(self, args: Qwen3MoeArgs, device=None):
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
        model_args (Qwen3MoeArgs): Model configuration arguments.

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

    def __init__(self, model_args: Qwen3MoeArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim

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
        flash_meta = FlashAttnMeta()
        self.attn_func = flash_meta.flash_attn_func
        self.attn_func_varlen = flash_meta.flash_attn_varlen_func

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor, shape: [bs, seqlen, head_num * head_dim].
            position_embeddings (torch.Tensor): Position embeddings.
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths.
            max_seqlen (int, optional): Maximum sequence length.

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
        xq, xk = self.rope_func(xq, xk, cos, sin, unsqueeze_dim=2)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            target_dtype = torch.bfloat16
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        if cu_seqlens is not None:
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
        return self.o_proj(output)


class MoEGate(nn.Module):
    def __init__(
        self,
        num_routed_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        dim: int,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        # topk selection algorithm
        self.norm_topk_prob = norm_topk_prob
        self.num_routed_experts = num_routed_experts
        self.weight = nn.Parameter(torch.empty((self.num_routed_experts, dim)))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_idx, topk_weight


class Qwen3MoEBlock(nn.Module):
    """
    Qwen3MoEBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (Qwen3MoeArgs): Model configuration arguments.

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

    def __init__(self, layer_id: int, model_args: Qwen3MoeArgs, moe_args: MoEArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.self_attn = Attention(model_args)
        self.mlp = MoE(moe_args)

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
        self.moe_args = moe_args

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ):
        """
        Perform a forward pass through the Qwen3MoEBlock.

        Args:
            x (torch.Tensor): Input tensor.
            position_embeddings (torch.Tensor): Position embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.self_attn(
            self.input_layernorm(x),
            position_embeddings,
            cu_seqlens=kwargs.get("cu_seqlens", None),
            max_seqlen=kwargs.get("max_seqlen", None),
        )

        out = self.mlp(self.post_attention_layernorm(h))[0]

        out = h + out
        return out


@ModelRegistry.register(Qwen3MoeWeightMapper)
class Qwen3MoE(BaseModel):
    """
    Qwen3MoE Module

    Args:
        model_args (Qwen3MoeArgs): Model configuration arguments.

    Attributes:
        model_args (Qwen3MoeArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        embed_tokens (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Qwen3Moe blocks.
        norm (RMSNorm): Layer normalization for the model output.
        lm_head (ColumnParallelLinear): Linear layer for final output.
    """

    @staticmethod
    def supported_model_types():
        return ["qwen3_moe"]

    def __init__(self, model_args: Qwen3MoeArgs):
        super().__init__(model_args.hf_config)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.rotary_emb = RotaryEmbedding(model_args)

        self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.moe_args = MoEArgs(
            n_routed_experts=model_args.n_experts,
            n_shared_experts=getattr(model_args.hf_config, "n_shared_experts", 0),
            n_activated_experts=model_args.hf_config.num_experts_per_tok,
            n_expert_groups=getattr(model_args.hf_config, "n_group", 0),
            n_limited_groups=getattr(model_args.hf_config, "topk_group", 0),
            norm_topk_prob=getattr(model_args.hf_config, "norm_topk_prob", False),
            train_gate=model_args.train_gate,
            gate_bias_update_factor=model_args.gate_bias_update_factor,
            aux_loss_coeff=model_args.aux_loss_coeff,
            score_func=getattr(model_args.hf_config, "scoring_func", "softmax"),
            route_scale=getattr(model_args.hf_config, "routed_scaling_factor", 1.0),
            enable_router_bias=False,
            dim=model_args.dim,
            moe_inter_dim=model_args.ffn_dim,
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = Qwen3MoEBlock(
                layer_id, model_args, self.moe_args
            )

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

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        interested_tokens: Optional[torch.BoolTensor] = None,
        *args,
        **kwargs,
    ):
        if self.embed_tokens is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Do not remove this line
            # This is a trick for TP with torch.compile
            h = self.identity_layer(inputs_embeds)
        else:
            inputs_embeds = input_ids
            h = input_ids

        position_embeddings = self.rotary_emb(h, position_ids.to(dtype=torch.long))

        if "valid_input_len" in kwargs:
            valid_input_len = kwargs["valid_input_len"]
            updated_kwargs = pack_sequences_for_inputs(
                inputs_embeds,
                valid_input_len,
                list(position_embeddings),
                interested_tokens,
                inputs_seq_dim=1,
                inputs_batch_dim=0,
                position_ids_seq_dim=1,
                position_ids_batch_dim=0,
                interested_tokens_seq_dim=1,
                interested_tokens_batch_dim=0,
                padding_mask=kwargs.get("padding_mask", None),
                cp_mesh=kwargs.get("cp_mesh", None),
            )
            position_embeddings = tuple(updated_kwargs.pop("position_ids"))
            interested_tokens = updated_kwargs.pop("interested_tokens")
            h = updated_kwargs.pop("inputs")
            h = self.identity_layer(h)
            kwargs.update(updated_kwargs)

        for layer in self.layers.values():
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

        # Add `if` check just in case `pp` is enabled
        if self.norm is not None:
            if interested_tokens is not None:
                assert not isinstance(
                    h, torch.distributed.tensor.DTensor
                ), "interested_tokens must be a local tensor"
                h = h[interested_tokens]
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
                # Since call dtensor.full_tensor here,
                # full_tensor's dtype will equal to shard's dtype which will not be controlled by mp_policy
                # for run torch.mm on input's dtype
                with torch.autocast(device="cuda", dtype=h.dtype):
                    output = h @ embed_tokens_weight.t()
            return output
        else:
            return h

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        if not self.moe_args.fake_balanced_gate:
            for layer in self.layers.values():
                layer.mlp.gate.weight.requires_grad_(False)

        # rotary.inv_freq could get deleted and not re-initialized
        # so we need to delete it manually
        self.rotary_emb.to(torch.cuda.current_device())
        self.rotary_emb.reset_inv_freq()

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.qwen3_moe.parallelize import parallelize

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
        # Initialize multi-rank weight loader
        loader = MultiRankWeightLoader(parallel_dims)

        # Load all safetensors from `model_path`
        model_type = retry(AutoConfig.from_pretrained)(model_name_or_path).model_type
        model_path = resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = sorted(
            [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        )

        self_state_dict = self.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}
        new_state_dict = self_state_dict.copy()
        for k, v in self_state_dict.items():
            if "mlp.experts.gate_and_up_projs" in k or "mlp.experts.down_projs" in k:
                new_state_dict[k.replace("projs", "proj.weight")] = v
                del new_state_dict[k]
        self_state_dict = new_state_dict
        lm_head_weight_key = "lm_head.weight"
        embed_tokens_weight_key = "model.embed_tokens.weight"

        # Step 1: Load files in parallel
        rank_tensors, rank_tensor_metadata, weights_of_ckpt_names = (
            loader.load_files_parallel(model_path, device, safetensors_files)
        )

        # Step 2: Gather tensor names and build mapping
        all_tensor_names, tensor_to_rank_map = (
            loader.gather_tensor_names_and_build_mapping(
                weights_of_ckpt_names, rank_tensors
            )
        )

        # Step 3: Process each tensor
        reserved = {}
        for name, tensor in loader.iterate_tensors(
            all_tensor_names,
            tensor_to_rank_map,
            rank_tensors,
            rank_tensor_metadata,
            device,
        ):
            # Save embed_tokens tensor for weight tying if needed
            if name == embed_tokens_weight_key:
                reserved[name] = tensor.clone()

            dest_name, shared_weight = convert_weight_from_hf(
                tensor,
                name,
                model_type,
                parallel_dims,
                n_experts=self.model_args.n_experts,
            )

            if dest_name is None:
                # This is due to the expert parallelism grouping
                continue

            expert_id = None
            if match := re.search(  # noqa: F841
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)",
                dest_name,
            ):
                # remove `experts.$ID.` from dest_name
                expert_id = int(match.group(2))
                dest_name = dest_name.replace(f"experts.{expert_id}.", "experts.")
                # Convert expert_id to local_expert_id
                n_local_experts = (
                    self.model_args.n_experts
                    // parallel_dims.tp
                    // (parallel_dims.dp_shard * parallel_dims.cp)
                )

                expert_id = expert_id % n_local_experts

            if dest_name not in self_state_dict and parallel_dims.pp_enabled:
                logger.info(
                    f"Weight `{dest_name}` is discarded, maybe due to pipeline parallelism or expert parallelism grouping. Skipping this weight checking"
                )
                continue
            slice_range = None
            if "gate_proj" in dest_name:
                dest_name = dest_name.replace("gate_proj", "gate_and_up_proj")
                slice_range = slice(0, self.model_args.ffn_dim)
            elif "up_proj" in dest_name:
                dest_name = dest_name.replace("up_proj", "gate_and_up_proj")
                slice_range = slice(self.model_args.ffn_dim, None)

            target_tensor = self_state_dict[dest_name]
            if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                target_tensor = target_tensor.to_local()
            # Write to the correct expert of the target tensor
            if expert_id is not None:
                target_tensor = target_tensor[expert_id]
            if slice_range is not None:
                assert (
                    target_tensor.shape[0] == 2 * self.model_args.ffn_dim
                ), f"Shape mismatch: {target_tensor.shape} != {2 * self.model_args.ffn_dim} for {dest_name}"
                target_tensor = target_tensor[slice_range]
            assert (
                target_tensor.shape == shared_weight.shape
            ), f"Shape mismatch: {target_tensor.shape} != {shared_weight.shape} for {dest_name}"
            with torch.no_grad():
                target_tensor.data.copy_(shared_weight)

        # Handle weight tying: lm_head shares weights with embed_tokens
        if (
            lm_head_weight_key not in all_tensor_names
            and embed_tokens_weight_key in all_tensor_names
        ):
            # tied with embed_tokens.weight
            name = lm_head_weight_key
            # All ranks should have embed_tokens_weight_key tensor from Step 3
            assert embed_tokens_weight_key in reserved, (
                f"embed_tokens_weight_key {embed_tokens_weight_key} not found in reserved. "
                f"This should have been saved during Step 3 processing."
            )
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
    def from_model_args(cls, model_args: Qwen3MoeArgs) -> "Qwen3MoE":
        """
        Initialize a Qwen3Moe model from a Qwen3MoeArgs object.

        Args:
            model_args (Qwen3MoeArgs): Model configuration arguments.

        Returns:
            Qwen3MoE: Qwen3MoE model.

        """
        return cls(model_args)

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "Qwen3MoE":
        """
        Initialize a Qwen3MoE model from a pretrained model.

        Args:
            model_name_or_path (str): Model name or path to the pretrained model.

        Returns:
            Qwen3MoE: Qwen3MoE model.

        """
        try:
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

            # Qwen3MoE does not have any biases
            bias_list = []
            try:
                head_dim = hf_config.head_dim
            except Exception:
                head_dim = hf_config.hidden_size // hf_config.num_attention_heads
                logger.warning(f"head_dim not found in config, using {head_dim}")

            model = cls.from_model_args(
                Qwen3MoeArgs(
                    dim=hf_config.hidden_size,
                    ffn_dim=hf_config.moe_intermediate_size,
                    n_layers=hf_config.num_hidden_layers,
                    n_experts=hf_config.num_experts,
                    n_heads=hf_config.num_attention_heads,
                    n_kv_heads=hf_config.num_key_value_heads,
                    head_dim=head_dim,
                    vocab_size=vocab_size,
                    max_seq_len=max_position_embeddings,
                    rope_theta=hf_config.rope_theta,
                    q_k_norm_enabled=hf_config.model_type == "qwen3_moe",
                    norm_type="rmsnorm",
                    rope_type=rope_type,
                    biases=bias_list,
                    hf_config=hf_config,
                )
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        return model

    @classmethod
    def fqn_filter_for_quantization(cls) -> List[str]:
        return ["lm_head"]

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        if not (self.model_args.n_heads % (cp_size * tp_size) == 0):
            raise ValueError(
                f"Model is not compatible with cp parallelism, model's head number={self.model_args.n_heads} is not divisible by cp size({cp_size}) * tp_size({tp_size}) = {cp_size * tp_size}"
            )

    def check_tp_compatible(self, tp_size: int):
        non_divisible_by_tp_size = (
            self.model_args.n_heads % tp_size != 0
            or self.model_args.n_kv_heads % tp_size != 0
            or self.model_args.n_experts % tp_size != 0
        )
        if non_divisible_by_tp_size:
            raise ValueError(
                f"Model is not compatible with tp/ep parallelism, model's head number={self.model_args.n_heads} or kv head number={self.model_args.n_kv_heads} or expert number={self.model_args.n_experts} is not satisified by tp size({tp_size})"
            )
