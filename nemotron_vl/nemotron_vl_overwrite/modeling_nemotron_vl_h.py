# coding=utf-8
# Copyright 2024 HuggingFace Inc. team.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch NemotronH model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache  # we need __iter__ and __len__ of pkv
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    is_torchdynamo_compiling
)
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_mamba_2_ssm_available,    
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from .configuration_nemotron_h import NemotronHConfig, NemotronVLConfig

logger = logging.get_logger(__name__)


# Copied from transformers.models.mamba.modeling_mamba2.modeling_mamba2.py with MAMBA2->NEMOTRONH,Mamba2->NemotronH
# For Mamba2 components Mamba2->NemotronHMamba2
if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

try:
    #from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

is_fast_path_available = all(
    (
        selective_state_update,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        causal_conv1d_fn,
        causal_conv1d_update,
    )
)


_CHECKPOINT_FOR_DOC = "nvidia/Nemotron-H-56B-Base-8K"
_CONFIG_FOR_DOC = "NemotronHConfig"

###################################################################################
#   Helper methods for segment sum computation
###################################################################################

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states

###################################################################################
#   All layers used by language model
###################################################################################

# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py
class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        super().__init__()
        self.dtype = dtype
        self.hybrid_override_pattern = config.hybrid_override_pattern
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        ssm_state_size = config.ssm_state_size
        conv_kernel_size = config.conv_kernel
        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if self.hybrid_override_pattern[i] == "M":
                # Mamba layer
                self.conv_states += [
                    torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [
                    torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
                ]
            else:
                # Attention or MLP layer
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    # Copied from modeling_mamba2.py
    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()

class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    # jan28b version
    def forward(self, hidden_states, gate=None):
        return rmsnorm_fn(x=hidden_states,
                          weight=self.weight,
                          bias=None, # No bias
                          z=gate,
                          eps=self.variance_epsilon,
                          group_size=self.group_size,
                          norm_before_gate=False
        )

class NemotronHMamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.mamba_num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon

        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon, group_size=self.intermediate_size // self.n_groups)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        # Single step calculations via cache
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            _, _, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            # 2. Convolution sequence transformation
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            # 3. SSM transformation
            A = -torch.exp(self.A_log.float())  # (nheads,)
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)

            # 4. Final linear projection
            out = self.out_proj(hidden_states)[:, None, ...]

        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            if self.training and cache_params is None:
                out = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )

            else:
                _, _, gate, hidden_states_B_C, dt = projected_states.split(
                    [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                )

                # 2. Convolution sequence transformation
                # Init cache
                if cache_params is not None:
                    hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                    conv_states = nn.functional.pad(
                        hidden_states_B_C_transposed,
                        (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                    )
                    cache_params.update_conv_state(
                        layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True
                    )

                if self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                    )
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)
                hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )

                # 3. SSM transformation
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    dt,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

                # Init cache
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

                scan_output = scan_output.view(batch_size, seq_len, -1)

                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)

                # 4. Final linear projection
                out = self.out_proj(scan_output)
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Optional[HybridMambaAttentionDynamicCache]=None, cache_position:Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size-self.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        )

        # 2. Convolution sequence transformation
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False)

            # We need to guarantee that anything regarding the cache is on the same device
            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Init cache
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_B_C_transposed, (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0)
                )
                cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True)

            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            # We need to guarantee that anything regarding the cache is on the same device
            cache_device = cache_params.ssm_states.device

            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            # State calculation
            cache_params.update_ssm_state(
                layer_idx=self.layer_idx,
                new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # Contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

            # Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # 2. Compute the state for each intra-chunk
            # (right term of low-rank factorization of off-diagonal blocks; B terms)
            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
            # (middle term of factorization of off-diag blocks; A terms)
            if cache_params is not None and cache_position is not None and cache_position[0] > 0:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # 4. Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])

            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            # Init cache
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

class NemotronHRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        NemotronHRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Weights are in float32
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

class NemotronHBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # M: Mamba2, *: Attention, -: MLP
        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NEMOTRONH_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid layer pattern {config.hybrid_override_pattern[layer_idx]}")

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        moe_aux_loss = None
        with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
            # * Use torch.cuda.stream() to avoid NaN issues when using multiple GPUs
            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            if self.block_type == "mamba":
                hidden_states = self.mixer(
                    hidden_states, cache_params=cache_params, cache_position=cache_position
                )
            elif self.block_type == "attention":
                hidden_states = self.mixer(
                    hidden_states, cache_position=cache_position
                )
                hidden_states = hidden_states[0]
            elif self.block_type in ["mlp", "moe"]:
                moe_results = self.mixer(
                    hidden_states
                )
                if isinstance(moe_results, tuple):
                    hidden_states = moe_results[0]
                    moe_aux_loss = moe_results[1]
                else:
                    hidden_states = moe_results
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")

            hidden_states = residual + hidden_states
            return hidden_states, moe_aux_loss

# Copied from transformers.models.nemotron.modeling_nemotron Nemotron->NemotronH
class NemotronHMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

class NemotronHMOE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(config, intermediate_size=config.moe_intermediate_size, layer_idx=layer_idx)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size, layer_idx=layer_idx
        )

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
            else:
                # Local empty expert: no-op compute that still marks params as used.
                expert_dtype = expert.down_proj.weight.dtype
                dummy_out = expert(torch.zeros_like(hidden_states[0]).unsqueeze(0).to(expert_dtype))
                final_hidden_states = final_hidden_states + dummy_out

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

class NemotronHTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NemotronHAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: NemotronHConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_embeddings: Tuple[torch.Tensor, torch.Tensor], #TODO
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        #attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# Adapted from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->Jamba
#class JambaFlashAttention2(JambaAttention):
class NemotronHFlashAttention2(NemotronHAttention):
    """
    Jamba flash attention module. This module inherits from `JambaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self.config, "sliding_window", None),
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        #attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Adapted from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->Jamba
#class JambaSdpaAttention(JambaAttention):
class NemotronHSdpaAttention(NemotronHAttention):
    """
    Jamba attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `JambaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from NemotronHAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "NemotronHModel is using NemotronHSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if self.is_causal and causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


NEMOTRONH_ATTENTION_CLASSES = {
    "eager": NemotronHAttention,
    "flash_attention_2": NemotronHFlashAttention2,
    "sdpa": NemotronHSdpaAttention,
}

###################################################################################
#   All layers used by visual model
###################################################################################

class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        #attention_interface: Callable = eager_attention_forward
        self.config._attn_implementation = "flash_attention_2"
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

    def init_weights(self):
        # init weight with he_normal
        # init bias with zero
        # init layernorm with standard gaussian distribution
        nn.init.kaiming_uniform_(self.linear_fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_fc1.bias)
        nn.init.zeros_(self.linear_fc2.bias)
        self.norm.reset_parameters()


# Copy Qwen-VL's visual model from qwen3-vl's repo
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

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

class NemotronVLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = NemotronVLConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["NemotronHBlock", "Qwen3VLVisionBlock"]
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _is_stateful = True
    _supports_sdpa = True

###################################################################################
#   Visual Model (Migrating from Qwen3-VL)
###################################################################################

class Qwen3VLVisionModel(NemotronVLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.using_deepstack = len(self.deepstack_visual_indexes) > 0
        if self.using_deepstack:
            self.deepstack_merger_list = nn.ModuleList(
                [
                    Qwen3VLVisionPatchMerger(
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
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

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
        device = grid_thw.device

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

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
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

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if self.using_deepstack:
                if layer_num in self.deepstack_visual_indexes:
                    deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                        hidden_states
                    )
                    deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists

@dataclass
# Copied from transformers.models.mamba.modeling_mamba2.Mamba2Output with MAMBA2->NemotronH,Mamba2->NemotronH
class NemotronHOutput(ModelOutput):
    """
    Class for the NemotronH model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`HybridMambaAttentionDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[HybridMambaAttentionDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    aux_loss: Optional[torch.FloatTensor] = None


@dataclass
# Copied from transformers.models.mamba2.modeling_mamba2.MambaCausalLMOutput with Mamba2->NemotronH
class NemotronVLCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`HybridMambaAttentionDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[HybridMambaAttentionDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class NemotronHModel(NemotronVLPreTrainedModel):
    def __init__(self, config):
        config._attn_implementation = "sdpa"
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NemotronHBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # From zamba_modeling.py
        if use_cache and cache_params is None:
            logger.warning_once(
                "NemotronH requires an initialized `NemotronHHybridDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )

        hidden_states = inputs_embeds

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # Until HERE
        aux_loss = None
        for layer_idx, mixer_block in enumerate(self.layers):
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            if mixer_block.block_type == "mamba":
                layer_mask = mamba_mask
            elif mixer_block.block_type == "attention":
                layer_mask = causal_mask
            elif mixer_block.block_type in ["mlp", "moe"]:
                layer_mask = None
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, local_aux_loss = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, layer_mask
                )
            else:
                hidden_states, local_aux_loss = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )
            if local_aux_loss is not None:
                aux_loss = (aux_loss + local_aux_loss) if aux_loss is not None else local_aux_loss


            # TODO: Store attentions
            # if output_attentions:
            #     if layer_outputs[1] is not None:
            #         # append attentions only of attention layers. Mamba layers return `None` as the attention weights
            #         all_self_attns += (layer_outputs[1],)

            # TODO (Check): should it happen before the forward pass?
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return NemotronHOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            aux_loss=aux_loss,
        )

    # Copied from transformers.models.jamba.modeling_jamba.JambaModel._update_causal_mask
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask


class NemotronVLModel(NemotronVLPreTrainedModel):
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: NemotronVLConfig
    _no_split_modules = ["NemotronHBlock", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = NemotronHModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
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
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
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

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
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
    
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, NemotronHOutput]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        nemotron_h_outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return nemotron_h_outputs # NemotronHOutput

class NemotronVLForConditionCausalLM(NemotronVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config: NemotronVLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = NemotronVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder
    
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        

        # Copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`
        empty_past_kv = past_key_values is None

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if not empty_past_kv:
            if (
                inputs_embeds is not None  # Exception 1
                or cache_position[-1] >= input_ids.shape[1]  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )
        model_inputs["past_key_values"] = past_key_values
        
        # remove position_ids, since position_ids is calculated within model
        model_inputs["position_ids"] = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()} 

        # if cache_position is larger than 0 (which mean in decode phase), 
        # remove pixel_values (embedding is already done in prefill phase)
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds=None,
        use_cache=False,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, NemotronVLCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        nemotron_h_outputs = self.model(
            input_ids,
            attention_mask = attention_mask,
            position_ids=position_ids,
            past_key_value=cache_params,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        hidden_states = nemotron_h_outputs[0]
        # TODO: Check zamba_modeling.py: https://github.com/huggingface/transformers/blob/d7188ba600e36d3fd191b12e19f1b3bb81a8404f/src/transformers/models/zamba/modeling_zamba.py#L1284C1-L1286C2
        #logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + nemotron_h_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return NemotronVLCausalLMOutput(
            loss=loss,
            aux_loss=nemotron_h_outputs.aux_loss,
            logits=logits,
            cache_params=nemotron_h_outputs.cache_params,
            hidden_states=nemotron_h_outputs.hidden_states,
            attentions=nemotron_h_outputs.attentions,
        )
