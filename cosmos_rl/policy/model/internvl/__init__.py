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
from typing import List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn
from transformers import AutoConfig
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
from cosmos_rl.policy.model.internvl.weight_converter import (
    convert_weight_from_hf,
)
from cosmos_rl.policy.model.internvl.weight_mapper import InternVLWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import ModelRegistry, BaseModel
from functools import cached_property
from cosmos_rl.utils.sequence_packing import pack_sequences_for_inputs
from cosmos_rl.policy.model.gpt import GPTArgs
from cosmos_rl.policy.kernel.moe.moe import MoEArgs
from cosmos_rl.policy.model.qwen3_moe import (
    Qwen3MoEBlock,
    Qwen3MoeArgs,
    RotaryEmbedding as Qwen3MoERotaryEmbedding,
    build_norm as qwen3_moe_build_norm,
)

from cosmos_rl.policy.model.vision_encoder.internvl import (
    InternVL_Encoder_Args,
    InternVisionModel,
)
# from cosmos_rl.dispatcher.data.packer.internvl_data_packer import (
#     InternVL_DataPacker,
# )

InternVL_LM_Args = Union[Qwen3MoeArgs, GPTArgs]


@dataclass
class InternVL_Args:
    lm_args: InternVL_LM_Args
    encoder_args: InternVL_Encoder_Args
    hf_config: AutoConfig = None


class Qwen3MoE(nn.Module):
    def __init__(self, model_args: InternVL_LM_Args):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.rotary_emb = Qwen3MoERotaryEmbedding(model_args)

        self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.moe_args = MoEArgs(
            norm_topk_prob=getattr(model_args.hf_config, "norm_topk_prob", False),
            n_routed_experts=model_args.n_experts,
            n_shared_experts=getattr(model_args.hf_config, "n_shared_experts", 0),
            n_activated_experts=model_args.hf_config.num_experts_per_tok,
            n_expert_groups=getattr(model_args.hf_config, "n_group", 0),
            n_limited_groups=getattr(model_args.hf_config, "topk_group", 0),
            train_gate=model_args.train_gate,
            gate_bias_update_factor=model_args.gate_bias_update_factor,
            aux_loss_coeff=model_args.aux_loss_coeff,
            enable_router_bias=False,
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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        interested_tokens: Optional[torch.BoolTensor] = None,
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
                position_ids_seq_dim=1,
                position_ids_batch_dim=0,
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
                    seq_dims=[1] * len(position_embeddings),
                )
            )
            h = self.identity_layer(inputs_embeds)

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
            assert self.lm_head is not None, "lm_head must be provided in last stage"
            h = self.lm_head(self.norm(h))
        return h

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
    InternVLWeightMapper  # , default_data_packer_cls=InternVL_DataPacker
)
class InternVLChatModel(BaseModel):
    def __init__(self, config: InternVL_Args):
        super().__init__(config.hf_config)
        self.config = config
        self.hf_config = config.hf_config
        self.lm_arch = self.hf_config.llm_config.architectures[0]
        self.model = self.get_language_model()
        self.visual = InternVisionModel(config.encoder_args)
        self.vocab_size = config.lm_args.vocab_size

        image_size = (
            self.hf_config.force_image_size or self.hf_config.vision_config.image_size
        )
        patch_size = self.hf_config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = self.hf_config.select_layer
        self.template = self.hf_config.template
        self.ps_version = self.hf_config.ps_version
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (self.hf_config.downsample_ratio**2)
        )
        self.downsample_ratio = self.hf_config.downsample_ratio
        vit_hidden_size = self.hf_config.vision_config.hidden_size
        llm_hidden_size = self.hf_config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def get_language_model(self):
        if architecture := self.lm_arch == "Qwen3MoeForCausalLM":
            return Qwen3MoE(self.config.lm_args)
        else:
            raise NotImplementedError(
                f"{architecture} is not implemented for InternVLChatModel."
            )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            logger.warning(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        vit_embeds = self.visual(
            pixel_values=pixel_values, select_layer=self.select_layer
        )
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _process_vision_embeddings(
        self, input_embeds, input_ids, pixel_values, pad_token_id
    ):
        """Helper function to process vision embeddings (images or videos)"""

        vision_embeds = self.extract_feature(pixel_values)
        batch, seq_len, feature_dim = input_embeds.shape
        input_embeds = input_embeds.reshape(batch * seq_len, feature_dim)
        input_ids = input_ids.reshape(batch * seq_len)
        selected = input_ids == self.image_token_id
        n_image_tokens_in_input_ids = selected.sum()
        n_image_tokens_in_vision_embeds = vision_embeds.reshape(-1, feature_dim).shape[
            0
        ]
        assert (
            n_image_tokens_in_input_ids == n_image_tokens_in_vision_embeds
        ), f"{n_image_tokens_in_input_ids} != {n_image_tokens_in_vision_embeds}"
        input_embeds = input_embeds.clone()
        input_embeds[selected] = (
            vision_embeds.reshape(-1, feature_dim)
            .clone()
            .to(input_embeds.device, input_embeds.dtype)
        )
        input_embeds = input_embeds.reshape(batch, seq_len, feature_dim)

        return input_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
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
            # print(f"inputs_embeds: {inputs_embeds.shape}, input_ids: {input_ids.shape}, n_image_tokens: {n_image_tokens}")
            # get vision embeddings as tokens for next phase
            if n_image_tokens > 0:
                inputs_embeds = self._process_vision_embeddings(
                    inputs_embeds,
                    input_ids,
                    pixel_values,
                    self.image_token_id,
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
            position_ids=position_ids,
            interested_tokens=kwargs.pop("interested_tokens", None),
            **kwargs,  # Additional arguments for compatibility
        )
        return outputs

    @property
    def image_token_id(self):
        # TODO: fix this
        # <IMG_CONTEXT> is a special token for image context
        return 151671

    @property
    def video_token_id(self):
        # TODO: fix this
        # <IMG_CONTEXT> is a special token for video context
        return 151671

    @property
    def delay_cp_slice_inputs(self):
        return True

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.lm_arch == "Qwen3MoeForCausalLM":
            seq_dim_idx = 1
            inputs = kwargs["input_ids"]
            position_ids = (
                torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
                .unsqueeze(0)
                .expand_as(inputs)
            )
            return position_ids, inputs, seq_dim_idx
        else:
            raise NotImplementedError(
                f"{self.lm_arch} is not implemented for get_position_ids"
            )

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        self.model.post_to_empty_hook(cosmos_config)

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
    def from_model_args(cls, model_args: InternVL_Args) -> "InternVLChatModel":
        """
        Initialize a GPT model from a GPTArgs object.

        Args:
            model_args (GPTArgs): Model configuration arguments.

        Returns:
            InternVLChatModel: InternVLChatModel model.

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
        lm_type = self.hf_config.llm_config.model_type
        model_path = resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        # Load LM weights
        lm_state_dict = self.model.state_dict()
        lm_state_dict = {clear_weight_name(k): v for k, v in lm_state_dict.items()}
        new_state_dict = lm_state_dict.copy()
        for k, v in lm_state_dict.items():
            if "mlp.experts.gate_and_up_projs" in k or "mlp.experts.down_projs" in k:
                new_state_dict[k.replace("projs", "proj.weight")] = v
                del new_state_dict[k]
        lm_state_dict = new_state_dict
        # print(f"lm_state_dict: {lm_state_dict.keys()}")
        # Rename dict to remove all `._orig_mod` in keys
        if self.visual is not None:
            visual_state_dict = self.visual.state_dict()
            visual_state_dict = {
                clear_weight_name(k): v for k, v in visual_state_dict.items()
            }
        else:
            visual_state_dict = {}

        multi_modal_projector_state_dict = self.mlp1.state_dict()
        multi_modal_projector_state_dict = {
            clear_weight_name(k): v for k, v in multi_modal_projector_state_dict.items()
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

                n_experts = (
                    self.config.lm_args.n_experts if lm_type == "qwen3_moe" else 0
                )

                for name in weights_of_ckpt.keys():
                    tensor = weights_of_ckpt[name]
                    dest_name, shared_weight = convert_weight_from_hf(
                        tensor, name, model_type, lm_type, n_experts, parallel_dims
                    )
                    if dest_name is None:
                        # This is due to the expert parallelism grouping
                        continue
                    # For MoE LM
                    expert_id = None
                    if match := re.search(  # noqa: F841
                        r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|gate_proj|down_proj)\.(weight|bias)",
                        dest_name,
                    ):
                        # remove `experts.$ID.` from dest_name
                        expert_id = int(match.group(2))
                        dest_name = dest_name.replace(
                            f"experts.{expert_id}.", "experts."
                        )
                        # Convert expert_id to local_expert_id
                        n_local_experts = (
                            n_experts
                            // parallel_dims.tp
                            // (parallel_dims.dp_shard * parallel_dims.cp)
                        )
                        expert_id = expert_id % n_local_experts

                    slice_range = None
                    if "gate_proj" in dest_name:
                        dest_name = dest_name.replace("gate_proj", "gate_and_up_proj")
                        slice_range = slice(0, self.model.model_args.ffn_dim)
                    elif "up_proj" in dest_name:
                        dest_name = dest_name.replace("up_proj", "gate_and_up_proj")
                        slice_range = slice(self.model.model_args.ffn_dim, None)

                    if dest_name in lm_state_dict:
                        target_tensor = lm_state_dict[dest_name]
                    elif dest_name in visual_state_dict:
                        target_tensor = visual_state_dict[dest_name]
                    elif dest_name in multi_modal_projector_state_dict:
                        target_tensor = multi_modal_projector_state_dict[dest_name]
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

                    # Write to the correct expert of the target tensor
                    if expert_id is not None:
                        local_view = local_view[expert_id]
                    if slice_range is not None:
                        assert (
                            local_view.shape[0] == 2 * self.model.model_args.ffn_dim
                        ), f"Shape mismatch: {local_view.shape} != {2 * self.model.model_args.ffn_dim} for {dest_name}"
                        local_view = local_view[slice_range]

                    assert (
                        local_view.shape == shared_weight.shape
                    ), f"Shape mismatch: {local_view.shape} != {shared_weight.shape} for {dest_name} with original shape {target_tensor.shape}"
                    with torch.no_grad():
                        local_view.data.copy_(shared_weight)

    def separate_model_parts(self) -> List[nn.Module]:
        return [self.model, self.visual, self.mlp1]

    @property
    def parallelize_fn(self) -> Tuple[Callable, nn.Module]:
        from cosmos_rl.policy.model.internvl.parallelize import parallelize

        return parallelize, self

    @staticmethod
    def supported_model_types():
        return ["internvl", "internvl_chat"]

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "InternVLChatModel":
        """
        Initialize a GPT model from a pretrained model.

        Args:
            model_name_or_path (str): Model name or path to the pretrained model.

        Returns:
            InternVLChatModel: InternVLChatModel model.

        """
        if hf_config.model_type not in cls.supported_model_types():
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        if max_position_embeddings is None:
            max_position_embeddings = hf_config.max_position_embeddings
        else:
            hf_config.max_position_embeddings = max_position_embeddings

        torch_dtype = hf_config.torch_dtype
        hf_config.llm_config.torch_dtype = torch_dtype
        hf_config.vision_config.torch_dtype = torch_dtype
        vocab_size = sync_model_vocab(model_name_or_path)

        lm_args = None
        if (
            architecture := hf_config.llm_config.architectures[0]
        ) == "Qwen3MoeForCausalLM":
            lm_config = hf_config.llm_config
            # Qwen3MoE does not have any biases
            bias_list = []
            rope_scaling = {}
            if hasattr(lm_config, "rope_scaling"):
                rope_scaling = lm_config.rope_scaling or {}
            rope_type = rope_scaling.get(
                "rope_type", rope_scaling.get("type", "default")
            )
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
        else:
            raise NotImplementedError(
                f"{architecture} is not implemented for InternVLModel."
            )

        vision_config = hf_config.vision_config
        encoder_args = InternVL_Encoder_Args(
            num_channels=vision_config.num_channels,
            patch_size=vision_config.patch_size,
            image_size=vision_config.image_size,
            qkv_bias=vision_config.qkv_bias,
            hidden_size=vision_config.hidden_size,
            num_attention_heads=vision_config.num_attention_heads,
            intermediate_size=vision_config.intermediate_size,
            qk_normalization=vision_config.qk_normalization,
            num_hidden_layers=vision_config.num_hidden_layers,
            hidden_act=vision_config.hidden_act,
            norm_type=vision_config.norm_type,
            layer_norm_eps=vision_config.layer_norm_eps,
            hf_config=vision_config,
        )
        args = InternVL_Args(
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
        visual_n_heads = self.config.encoder_args.num_attention_heads
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
        visual_n_heads = self.config.encoder_args.num_attention_heads
        llm_n_heads = self.config.lm_args.n_heads
        llm_n_kv_heads = self.config.lm_args.n_kv_heads
        non_divisible_by_tp_size = (
            visual_n_heads % tp_size != 0
            or llm_n_heads % tp_size != 0
            or llm_n_kv_heads % tp_size != 0
        )
        if non_divisible_by_tp_size:
            raise ValueError(
                f"Model is not compatible with tp parallelism, model's visual_n_heads={visual_n_heads} or llm_n_heads={llm_n_heads} or llm_n_kv_heads={llm_n_kv_heads} is not satisified by tp size({tp_size})"
            )
