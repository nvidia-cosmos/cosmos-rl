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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import AutoConfig
from cosmos_rl.utils.util import (
    resolve_model_path,
    IdentityLayer,
    clear_weight_name,
    sync_model_vocab,
    retry,
)
from safetensors import safe_open
from cosmos_rl.policy.model.qwen2_5_vl.weight_converter import (
    convert_weight_from_hf,
)
from cosmos_rl.dispatcher.data.packer.qwen2_5_vlm_data_packer import (
    Qwen2_5_VLM_DataPacker,
)
from cosmos_rl.policy.model.qwen2_5_vl.weight_mapper import QwenVL25WeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import ModelRegistry, BaseModel
from functools import cached_property
import cosmos_rl.policy.kernel.modeling_utils as modeling_utils
from cosmos_rl.policy.kernel.norm import RMSNorm
from cosmos_rl.policy.kernel.fused import MLPActMulFunc
from cosmos_rl.policy.model.vision_encoder.qwen2_5_vl import (
    Qwen2_5_VL_Encoder_Args,
    Qwen2_5_VisionTransformerPretrainedModel,
    rotate_half,
)


@dataclass
class Qwen2_5_VL_LM_Args:
    mrope_section: List[int]
    dim: int
    ffn_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int
    biases: List[str] = field(default_factory=lambda: [])
    norm_eps: float = 1e-6
    rope_theta: float = 10000
    norm_type: str = "rmsnorm"
    rope_type: str = "default"
    hidden_act: str = "silu"
    hf_config: AutoConfig = None


@dataclass
class Qwen2_5_VL_Args:
    lm_args: Qwen2_5_VL_LM_Args
    encoder_args: Qwen2_5_VL_Encoder_Args
    hf_config: AutoConfig = None


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VL_LM_Args, device=None):
        super().__init__()
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[config.rope_type]
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

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq.dtype != torch.float32:
            self.reset_inv_freq(device=x.device)
            assert (
                self.inv_freq.dtype == torch.float32
            ), "inv_freq dtype should be float32"
        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2_5_VL_LM_Args):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim
        self.intermediate_size = config.ffn_dim
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_mul_func = MLPActMulFunc(ACT2FN[config.hidden_act])

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_mul_func(self.gate_proj(x), self.up_proj(x))
        )
        return down_proj


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=2):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
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
    assert isinstance(mrope_section, list), "mrope_section must be a list"
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5_VLAttention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (Qwen2_5_VL_LM_Args): Model configuration arguments.

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

    def __init__(self, model_args: Qwen2_5_VL_LM_Args):
        super().__init__()
        self.config = model_args
        self.mrope_section = model_args.mrope_section
        assert (
            len(self.mrope_section) == 3
        ), "mrope_section must be a list of 3 integers"

        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.attn_func = modeling_utils.flash_attn_func

        self.q_proj = nn.Linear(
            model_args.dim,
            model_args.n_heads * self.head_dim,
            bias="q_proj" in model_args.biases,
        )
        self.k_proj = nn.Linear(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias="k_proj" in model_args.biases,
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

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_multimodal_rotary_pos_emb(xq, xk, cos, sin, self.mrope_section)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                raise ValueError("Flash attention only supports float32 input")
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        output = self.attn_func(xq, xk, xv, causal=True)
        output = output.view(bs, seqlen, -1)
        return self.o_proj(output)


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VL_LM_Args, layer_idx: int):
        super().__init__()
        self.hidden_size = config.dim

        self.self_attn = Qwen2_5_VLAttention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(
            config.dim, config.norm_eps, casting_mode=config.hf_config.model_type
        )
        self.post_attention_layernorm = RMSNorm(
            config.dim, config.norm_eps, casting_mode=config.hf_config.model_type
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        # Self Attention
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return residual + hidden_states


class Qwen2_5_VLModel(nn.Module):
    def __init__(self, config: Qwen2_5_VL_LM_Args):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = Qwen2_5_VLDecoderLayer(config, layer_id)
        self.norm = RMSNorm(
            config.dim, config.norm_eps, casting_mode=config.hf_config.model_type
        )
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.identity_layer = IdentityLayer()

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        interested_tokens: Optional[torch.BoolTensor] = None,
    ):
        h = self.identity_layer(inputs_embeds)

        position_embeddings = self.rotary_emb(h, position_ids)

        for layer in self.layers.values():
            if (
                hasattr(layer, "_gradient_checkpointing_enabled")
                and layer._gradient_checkpointing_enabled
            ):
                h = torch.utils.checkpoint.checkpoint(
                    layer,
                    h,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                h = layer(h, position_embeddings=position_embeddings)

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
            self.config.n_heads,
            self.config.dim // self.config.n_heads,
        )
        return lambda seq_len: (
            nparams,
            6 * (nparams - nparams_embedding)
            + 12 * layers * heads * head_dim * seq_len,
        )


@ModelRegistry.register(
    QwenVL25WeightMapper, default_data_packer_cls=Qwen2_5_VLM_DataPacker
)
class Qwen2_5_VLConditionalModel(BaseModel):
    def __init__(self, config: Qwen2_5_VL_LM_Args):
        super().__init__(config.hf_config)
        self.config = config
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(config.encoder_args)
        self.model = Qwen2_5_VLModel(config.lm_args)
        self.vocab_size = config.lm_args.vocab_size

    def _process_vision_embeddings(
        self, inputs_embeds, input_ids, pixel_values, grid_thw, pad_token_id
    ):
        """Helper function to process vision embeddings (images or videos)"""
        n_tokens = (input_ids == pad_token_id).sum().item()
        if n_tokens > 0:
            vision_embeds = self.visual(pixel_values, grid_thw=grid_thw)
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
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.model.embed_tokens is not None:
            assert input_ids.dtype in [
                torch.int32,
                torch.int64,
            ], "input_ids must be of type int32 or int64"
            inputs_embeds = self.model.embed_tokens(input_ids)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            n_video_tokens = (input_ids == self.video_token_id).sum().item()

            # print(f"inputs_embeds: {inputs_embeds.shape}, input_ids: {input_ids.shape}, n_image_tokens: {n_image_tokens}, n_video_tokens: {n_video_tokens}")
            # get vision embeddings as tokens for next phase
            if n_image_tokens > 0:
                assert (
                    image_grid_thw is not None
                ), "image_grid_thw must be provided if there are image tokens"
                inputs_embeds = self._process_vision_embeddings(
                    inputs_embeds,
                    input_ids,
                    pixel_values,
                    image_grid_thw,
                    self.image_token_id,
                )

            if n_video_tokens > 0:
                assert (
                    video_grid_thw is not None
                ), "video_grid_thw must be provided if there are video tokens"
                inputs_embeds = self._process_vision_embeddings(
                    inputs_embeds,
                    input_ids,
                    pixel_values_videos,
                    video_grid_thw,
                    self.video_token_id,
                )
        else:
            assert (
                input_ids.is_floating_point()
            ), "input of pipeline stage > 0 must be of floating point type"
            inputs_embeds = input_ids

        # For GRPO, we can pass in the logprob_masks to the model
        # to avoid computing the logits which are not needed for the model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            # Permute back to [3, batch_size, seq_len] for Pipeline Parallelism micro batch
            position_ids=position_ids.permute(1, 0, 2).contiguous(),
            interested_tokens=kwargs.get("interested_tokens", None),
        )
        return outputs

    @property
    def image_token_id(self):
        return self.config.hf_config.image_token_id

    @property
    def video_token_id(self):
        return self.config.hf_config.video_token_id

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq_dim_idx = 2
        assert "position_ids" in kwargs, "position_ids must be provided"
        return (
            kwargs["position_ids"].permute(1, 0, 2).contiguous(),
            kwargs["input_ids"],
            seq_dim_idx,
        )

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        self.model.rotary_emb.to(torch.cuda.current_device())
        self.model.rotary_emb.reset_inv_freq()
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
    def from_model_args(
        cls, model_args: Qwen2_5_VL_Args
    ) -> "Qwen2_5_VLConditionalModel":
        """
        Initialize a GPT model from a GPTArgs object.

        Args:
            model_args (GPTArgs): Model configuration arguments.

        Returns:
            Qwen2_5_VLConditionalModel: Qwen2_5_VLConditionalModel model.

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
        model_type = retry(AutoConfig.from_pretrained)(model_name_or_path).model_type
        model_path = resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        # Load LM weights
        # model.safetensors.index.json
        lm_state_dict = self.model.state_dict()
        lm_state_dict = {clear_weight_name(k): v for k, v in lm_state_dict.items()}
        # Rename dict to remove all `._orig_mod` in keys
        if self.visual is not None:
            visual_state_dict = self.visual.state_dict()
            visual_state_dict = {
                clear_weight_name(k): v for k, v in visual_state_dict.items()
            }
        else:
            visual_state_dict = {}

        with torch.device(self.current_device()):
            for f in safetensors_files:
                weights_of_ckpt = {}
                ckpt = safe_open(
                    os.path.join(model_path, f), framework="pt", device=str(device)
                )
                keys = ckpt.keys()
                for name in keys:
                    ckpt_tensor = ckpt.get_tensor(name)
                    weights_of_ckpt[name] = ckpt_tensor

                for name in weights_of_ckpt.keys():
                    tensor = weights_of_ckpt[name]
                    dest_name, shared_weight = convert_weight_from_hf(
                        tensor, name, model_type, parallel_dims
                    )
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
        from cosmos_rl.policy.model.qwen2_5_vl.parallelize import parallelize

        return parallelize, self

    @staticmethod
    def supported_model_types():
        return ["qwen2_5_vl"]

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "Qwen2_5_VLConditionalModel":
        """
        Initialize a GPT model from a pretrained model.

        Args:
            model_name_or_path (str): Model name or path to the pretrained model.

        Returns:
            Qwen2_5_VLConditionalModel: Qwen2_5_VLConditionalModel model.

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
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))

        bias_list = ["q_proj", "k_proj", "v_proj"]

        lm_args = Qwen2_5_VL_LM_Args(
            mrope_section=hf_config.rope_scaling["mrope_section"],
            dim=hf_config.hidden_size,
            ffn_dim=hf_config.intermediate_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=hf_config.num_key_value_heads,
            vocab_size=vocab_size,
            max_seq_len=max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            norm_type="rmsnorm",
            hidden_act=hf_config.hidden_act,
            norm_eps=hf_config.rms_norm_eps,
            rope_type=rope_type,
            biases=bias_list,
            hf_config=hf_config,
        )

        encoder_args = Qwen2_5_VL_Encoder_Args(
            depth=hf_config.vision_config.depth,
            hidden_size=hf_config.vision_config.hidden_size,
            hidden_act=hf_config.vision_config.hidden_act,
            intermediate_size=hf_config.vision_config.intermediate_size,
            n_heads=hf_config.vision_config.num_heads,
            in_channels=hf_config.vision_config.in_chans,
            patch_size=hf_config.vision_config.patch_size,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
            temporal_patch_size=hf_config.vision_config.temporal_patch_size,
            tokens_per_second=hf_config.vision_config.tokens_per_second,
            window_size=hf_config.vision_config.window_size,
            fullatt_block_indexes=hf_config.vision_config.fullatt_block_indexes,
            out_hidden_size=hf_config.vision_config.out_hidden_size,
            norm_type="rmsnorm",
            norm_eps=hf_config.rms_norm_eps,
            hf_config=hf_config.vision_config,
        )
        args = Qwen2_5_VL_Args(
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
    def fqn_filter_for_fp8(cls) -> List[str]:
        llm = [
            "lm_head",
        ]
        visual = [
            "visual",
        ]  # Filter Linear in visual out, they will corrupt the FP8 Linear.
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
