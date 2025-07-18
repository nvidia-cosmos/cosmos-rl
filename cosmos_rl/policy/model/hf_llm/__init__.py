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
from typing import Tuple, List, Optional, Callable
from transformers import AutoConfig, AutoModelForCausalLM
from cosmos_rl.utils.util import (
    sync_model_vocab,
    clear_weight_name,
    retry,
)
from cosmos_rl.utils.constant import COSMOS_HF_MODEL_TYPES
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.model.hf_llm.weight_converter import convert_weight_from_hf
from cosmos_rl.policy.model.hf_llm.weight_mapper import HFLLMWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from functools import cached_property


@ModelRegistry.register(HFLLMWeightMapper)
class HFLLMModel(BaseModel):
    """
    HFLLM Module

    Args:
        hf_config : Model configuration arguments.
        model: AutoModelForCausalLM model.

    Attributes:
        hf_config : Model configuration arguments.
        model: AutoModelForCausalLM model.
        layers: List of AutoModelForCausalLM blocks.
        src_model_type: Model type.
    """

    @staticmethod
    def supported_model_types():
        return [COSMOS_HF_MODEL_TYPES]

    def __init__(self, hf_config, model):
        super().__init__(hf_config)
        self.hf_config = hf_config
        self.model = model
        self.layers = model.model.layers
        self.src_model_type = hf_config.model_type

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        out = self.model(
            input_ids,
            position_ids=position_ids,
            is_causal=True,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            *args,
            **kwargs,
        )
        return out.logits

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        # reset buffer registered in __init__() function,
        # e.g. rotary_emb.inv_freq, embed_tokens.embed_scale
        model = self.model.model
        rotary_emb = getattr(model, "rotary_emb", None)
        if rotary_emb is not None:
            rope_init_fn = getattr(rotary_emb, "rope_init_fn", None)
            if rope_init_fn is not None:
                inv_freq, rotary_emb.attention_scaling = rope_init_fn(
                    self.hf_config, None
                )
                rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
            else:
                logger.warning(
                    "rotary_emb does not have rope_init_fn, cannot reset inv_freq."
                )
        # Models like Gemma have rotary_emb_local
        rotary_emb_local = getattr(model, "rotary_emb_local", None)
        if rotary_emb_local is not None:
            rope_init_fn = getattr(rotary_emb_local, "rope_init_fn", None)
            if rope_init_fn is not None:
                local_inv_freq, rotary_emb_local.attention_scaling = rope_init_fn(
                    self.hf_config, None
                )
                rotary_emb_local.register_buffer(
                    "inv_freq", local_inv_freq, persistent=False
                )
            else:
                logger.warning(
                    "rotary_emb_local does not have rope_init_fn, cannot reset inv_freq."
                )

        if self.src_model_type in ["gemma3_text"]:
            embed_tokens = model.embed_tokens
            embed_scale = self.hf_config.hidden_size**0.5
            embed_tokens.register_buffer(
                "embed_scale", torch.tensor(embed_scale), persistent=False
            )

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.hf_llm.parallelize import parallelize

        return parallelize, self

    def apply_pipeline_split(self, pp_rank, pp_size):
        """
        Apply pipeline split to the model.
        This typically involves splitting the model into multiple stages,
        and moving each stage to a different device.
        """
        assert False, "Pipeline split is not supported for HFLLMModel"

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
        model_with_weights = AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        ).to(device)
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
        seq_dim_idx = 1
        inputs = kwargs["input_ids"]
        position_ids = (
            torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
            .unsqueeze(0)
            .expand_as(inputs)
        )
        return position_ids, inputs, seq_dim_idx

    def separate_model_parts(self) -> List[nn.Module]:
        return [self.model]

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
            self.hf_config.num_attention_heads,
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )
        return lambda seq_len: (
            nparams,
            6 * (nparams - nparams_embedding)
            + 12 * layers * heads * head_dim * seq_len,
        )

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        return self._get_nparams_and_flops_fn(seq_len)

    @classmethod
    def from_model_args(cls, hf_config) -> "HFLLMModel":
        """
        Initialize a HFLLM model from a HFLLMArgs object.

        Args:
            hf_config : hf model config.

        Returns:
            HFLLMModel: HFLLM model.

        """
        model = AutoModelForCausalLM.from_config(hf_config)
        return cls(hf_config, model)

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "HFLLMModel":
        """
        Initialize a HFLLM model from a pretrained model.

        Args:
            hf_config (AutoConfig): HuggingFace config.
            model_name_or_path (str): Model name or path to the pretrained model.
            max_position_embeddings (int): Maximum position embeddings.

        Returns:
            HFLLMModel: HFLLM model.

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
        return ["lm_head"]

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        if not (self.hf_config.num_attention_heads % (cp_size * tp_size) == 0):
            raise ValueError(
                f"Model is not compatible with cp parallelism, model's head number={self.hf_config.num_attention_heads} is not divisible by cp size({cp_size}) * tp_size({tp_size}) = {cp_size * tp_size}"
            )
