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
from functools import cached_property
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open
from torch.nn.modules.module import _IncompatibleKeys
from transformers import AutoConfig

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")


from cosmos_rl.dispatcher.data.packer.deepseek_data_packer import DeepSeek_DataPacker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.kernel.moe import moe
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.deepseek_v3 import deepseekv3_mapped
from cosmos_rl.policy.model.deepseek_v3.checkpoint_planner import RenameLoadPlanner
from cosmos_rl.policy.model.deepseek_v3.weight_mapper import (
    DeepseekV3MoEWeightMapper,
    convert_weight_from_hf,
    weight_dequant,
)
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.util import clear_weight_name, resolve_model_path, retry

DCP_CHECKPOINT_PATH_PREFIX = "/root/.cache"
DCP_CHECKPOINT_PATH_SUFFIX = "dcp"


@ModelRegistry.register(
    DeepseekV3MoEWeightMapper, default_data_packer_cls=DeepSeek_DataPacker
)
class DeepseekV3MoEModel(BaseModel):
    @staticmethod
    def supported_model_types():
        return ["deepseek_v3"]

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
        if hf_config.model_type not in cls.supported_model_types():
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        assert (
            "deepseek-v3" in model_name_or_path.lower()
        ), f"Unsupported model {model_name_or_path}"

        if hf_config.num_hidden_layers == 4:
            deepseek_config = deepseekv3_mapped.DeepseekConfig(n_layers=4)
        elif hf_config.num_hidden_layers == 61:
            deepseek_config = deepseekv3_mapped.DeepseekConfig()
        else:
            raise ValueError(
                f"Only 4 or 61 layer models supported at the moment. Got hf_config.llm_config.num_hidden_layers={hf_config.llm_config.num_hidden_layers}"
            )

        model = DeepseekV3MoEModel(deepseek_config, hf_config=hf_config)

        logger.info("Initializing the model")
        with torch.no_grad():
            model.init_weights()
        logger.info("Model initialized")
        return model

    def __init__(
        self,
        model_config: deepseekv3_mapped.DeepseekConfig,
        hf_config: AutoConfig,
    ):
        super().__init__(hf_config=hf_config)
        self.config = model_config

        orig_precision = torch.get_default_dtype()
        precision = getattr(torch, model_config.dtype)
        torch.set_default_dtype(precision)
        logger.info(f"Setting torch default dtype from {orig_precision} to {precision}")

        self.build_model(model_config)

        torch.set_default_dtype(
            orig_precision
        )  # Reset the default dtype to the original value
        logger.info(f"Reset torch default dtype to {orig_precision}")

    def build_model(self, model_config: deepseekv3_mapped.DeepseekConfig):
        # Create reasoning model
        self.model = deepseekv3_mapped.Transformer(args=model_config)

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        self.apply(_init_weights)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        initial_aux_loss: Optional[torch.Tensor] = None,
        **data_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logger.debug(
            f"[model] input ids shape: {input_ids.shape}, position_ids: {position_ids.shape}"
        )

        logits, aux_loss = self.model(
            tokens=input_ids,
            position_ids=position_ids,
            padding_mask=data_batch.get("padding_mask", None),
        )

        if self.config.aux_loss_coeff > 0:
            if initial_aux_loss is not None and aux_loss is not None:
                final_aux_loss = initial_aux_loss + aux_loss
                return logits, final_aux_loss
            elif initial_aux_loss is not None:
                final_aux_loss = initial_aux_loss
                return logits, final_aux_loss
            else:
                return logits
        else:
            assert (
                initial_aux_loss is None
            ), "initial_aux_loss must be None when aux_loss_coeff = 0"
            assert aux_loss is None, "aux_loss must be None when aux_loss_coeff = 0"
            return logits

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.deepseek_v3.parallelize import parallelize_model

        return parallelize_model, self

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        return

    def separate_model_parts(self) -> List[nn.Module]:
        return [self]

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq_dim_idx = 1
        inputs = kwargs["input_ids"]
        position_ids = (
            torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
            .unsqueeze(0)
            .expand_as(inputs)
        )
        return position_ids, inputs, seq_dim_idx

    def apply_pipeline_split(self, pp_rank, pp_size):
        raise NotImplementedError

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
            self.config.n_layers,
            self.config.n_heads,
            self.config.dim // self.config.n_heads,
        )
        return lambda seq_len: (
            nparams,
            6 * (nparams - nparams_embedding)
            + 12 * layers * heads * head_dim * seq_len,
        )

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        return self._get_nparams_and_flops_fn(seq_len)

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        dcp_checkpoint_path = os.path.join(
            DCP_CHECKPOINT_PATH_PREFIX,
            model_name_or_path.split("/")[-1].lower(),
            DCP_CHECKPOINT_PATH_SUFFIX,
        )
        # if it is a huggingface model and no checkpoint exists, we need to load the weights from the safetensors files
        if len(model_name_or_path.split("/")) == 2 and (
            not os.path.exists(dcp_checkpoint_path)
            or len(os.listdir(dcp_checkpoint_path)) == 0
        ):
            # The checkpoint loading assumes bf16 checkpoints. The default deepseekv3 checkpoint is in fp8. We needs to convert the checkpoints from fp8 to bf16 first.
            # Can follow the instructions here: https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/deepseek.md
            # Load all safetensors from `model_path`
            model_type = retry(AutoConfig.from_pretrained)(
                model_name_or_path, trust_remote_code=True
            ).model_type
            model_path = resolve_model_path(model_name_or_path, revision)
            safetensors_files = [
                f for f in os.listdir(model_path) if f.endswith(".safetensors")
            ]

            self_state_dict = self.state_dict()
            self_state_dict = {
                clear_weight_name(k): v for k, v in self_state_dict.items()
            }

            lm_head_weight_key = "model.lm_head.weight"
            embed_tokens_weight_key = "model.model.embed_tokens.weight"
            weights_of_ckpt_names = set()
            reserved = {}
            scale_inv_paths = {}

            for f in safetensors_files:
                ckpt = retry(safe_open)(
                    os.path.join(model_path, f), framework="pt", device=str(device)
                )
                keys = ckpt.keys()
                for name in keys:
                    if name.endswith("weight_scale_inv"):
                        scale_inv_paths[name] = os.path.join(model_path, f)

            for f in safetensors_files:
                logger.info(f"Loading safetensors: {f}")
                weights_of_ckpt = {}
                ckpt = retry(safe_open)(
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
                    if name.endswith("weight_scale_inv") or "layers.61" in name:
                        # Skip since this weight is used for dequantization
                        continue

                    if (
                        "down_proj" in name
                        or "up_proj" in name
                        or "gate_proj" in name
                        or "self_attn.kv_a_proj_with_mqa" in name
                        or "self_attn.kv_b_proj" in name
                        or "self_attn.o_proj" in name
                        or "self_attn.q_a_proj" in name
                        or "self_attn.q_b_proj" in name
                    ) and "weight" in name:
                        inv_name = name + "_scale_inv"
                        inv_tensor = retry(safe_open)(
                            scale_inv_paths[inv_name],
                            framework="pt",
                            device=str(device),
                        ).get_tensor(inv_name)
                        tensor = weight_dequant(tensor, inv_tensor)

                    dest_name, shared_weight, expert_id = convert_weight_from_hf(
                        tensor,
                        name,
                        model_type,
                        parallel_dims,
                        n_experts=self.config.n_routed_experts,
                    )

                    if dest_name is None:
                        # This is due to the expert parallelism grouping
                        continue

                    if dest_name not in self_state_dict and parallel_dims.pp_enabled:
                        logger.info(
                            f"Weight `{dest_name}` is discarded, maybe due to pipeline parallelism or expert parallelism grouping. Skipping this weight checking"
                        )
                        continue

                    target_tensor = self_state_dict[dest_name]
                    if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                        target_tensor = target_tensor.to_local()
                    # Write to the correct expert of the target tensor
                    if expert_id is not None:
                        # Convert expert_id to local_expert_id
                        n_local_experts = (
                            self.config.n_routed_experts
                            // parallel_dims.tp
                            // parallel_dims.dp_shard
                        )

                        expert_id = expert_id % n_local_experts
                        target_tensor = target_tensor[expert_id]

                    assert (
                        target_tensor.shape == shared_weight.shape
                    ), f"Shape mismatch: {target_tensor.shape} != {shared_weight.shape} for {dest_name}"
                    with torch.no_grad():
                        target_tensor.data.copy_(shared_weight)
                torch.distributed.barrier()
                logger.info(f"Loaded safetensors: {f} successfully.")

            if (
                lm_head_weight_key not in weights_of_ckpt_names
                and embed_tokens_weight_key in weights_of_ckpt_names
            ):
                # tied with embed_tokens.weight
                name = lm_head_weight_key
                assert embed_tokens_weight_key in reserved
                tensor = reserved[embed_tokens_weight_key]
                dest_name, shared_weight = convert_weight_from_hf(
                    tensor,
                    name,
                    model_type,
                    parallel_dims,
                    n_experts=self.config.n_routed_experts,
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

            logger.info(f"Dumping the tensors to DCP folder {dcp_checkpoint_path}")
            os.makedirs(dcp_checkpoint_path, exist_ok=True)
            fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
                dcp_checkpoint_path
            )
            torch.distributed.checkpoint.save(
                state_dict=self_state_dict,
                storage_writer=fs_storage_writer,
            )
        else:
            logger.info("Loading from distributed checkpoints...")
            model_name_or_path = model_name_or_path.rstrip("/")
            if model_name_or_path.endswith("_hf"):
                model_name_or_path_dcp = model_name_or_path[:-3]
                logger.info(
                    f"Found model path with _hf prefix ({model_name_or_path}. Looking for non-hf checkpoint at: {model_name_or_path_dcp}"
                )
                model_name_or_path = model_name_or_path_dcp
            elif len(model_name_or_path.split("/")) == 2:
                model_name_or_path = dcp_checkpoint_path

            # Transformer engine adds this extra state which we dont have in the saved checkpoint.
            mapped_state_dict = {
                k: v
                for k, v in self.state_dict().items()
                if not k.endswith("_extra_state")
            }

            logger.info("Creating storage reader...")
            fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(
                dcp_checkpoint_path
            )
            logger.info("Loading checkpoint ...")
            torch.distributed.checkpoint.load(
                state_dict=mapped_state_dict,
                storage_reader=fs_storage_reader,
                planner=RenameLoadPlanner(allow_partial_load=False),
            )
            logger.info("Refreshing model.")
            self.load_state_dict(self.state_dict())
            logger.info("Checkpoint loaded.")

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ):
        """
        Ignore the missing keys with substrings matching `substring_to_ignore` (e.g., "_extra_state" keys imposed by
        TransformerEngine for FP8).
        """
        actual_missing_keys, unexpected_keys = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        if strict:
            if len(actual_missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(
                    f"Missing keys: {actual_missing_keys}\n\nUnexpected keys: {unexpected_keys}"
                )
        return _IncompatibleKeys(actual_missing_keys, unexpected_keys)

    @cached_property
    def weight_sync_transforms(
        self,
    ) -> List[Tuple[str, Union[torch.Tensor, Callable]]]:
        # 1. get all parameters, but not buffers
        transforms = {}

        for local_name, param in self.named_parameters():
            hf_name = self.weight_mapper.policy_map_local_key_to_hf_key(
                clear_weight_name(local_name)
            )

            is_dist_tensor = isinstance(param, torch.distributed.tensor.DTensor)
            transform_or_view = param.to_local() if is_dist_tensor else param

            assert (
                hf_name not in transforms
            ), f"Duplicate key found in transforms: {hf_name}"
            transforms[hf_name] = transform_or_view

        return sorted(transforms.items())


def _init_weights(module):
    std = 0.02

    def to_local(tensor):
        if isinstance(tensor, DTensor):
            return tensor.to_local()
        else:
            return tensor

    if isinstance(module, torch.nn.Linear):
        to_local(module.weight).normal_(mean=0.0, std=std)
        if module.bias is not None:
            to_local(module.bias).zero_()
    elif isinstance(module, torch.nn.Embedding):
        to_local(module.weight).normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            to_local(module.weight)[module.padding_idx].zero_()
    elif isinstance(module, moe.Gate):
        to_local(module.weight).normal_(mean=0.0, std=std)
