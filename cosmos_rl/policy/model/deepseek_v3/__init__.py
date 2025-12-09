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
import torch.distributed as dist
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
from cosmos_rl.policy.model.deepseek_v3.weight_mapper import (
    DeepseekV3MoEWeightMapper,
    convert_weight_from_hf,
    weight_dequant,
)
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.util import clear_weight_name, resolve_model_path, retry

DCP_CHECKPOINT_PATH_PREFIX = os.environ.get(
    "DCP_CHECKPOINT_PATH_PREFIX", "/root/.cache"
)
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
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # If the model path is a local path, do not dequant the weights
        dequant_weights = False if len(model_name_or_path.split("/")) > 2 else True

        def _broadcast_data(data, src_rank):
            container = [data]
            dist.broadcast_object_list(container, src=src_rank)
            return container[0]

        # 1. Setup Phase (Rank 0 resolves structure)
        model_type = None
        safetensors_files = []
        model_path = ""

        if rank == 0:
            model_type = retry(AutoConfig.from_pretrained)(
                model_name_or_path, trust_remote_code=True
            ).model_type
            model_path = resolve_model_path(model_name_or_path, revision)
            safetensors_files = [
                f for f in os.listdir(model_path) if f.endswith(".safetensors")
            ]
            safetensors_files.sort()  # Ensure deterministic order

        # Broadcast metadata
        setup_info = (model_type, safetensors_files, model_path) if rank == 0 else None
        model_type, safetensors_files, model_path = _broadcast_data(
            setup_info, src_rank=0
        )

        self_state_dict = self.state_dict()
        self_state_dict = {clear_weight_name(k): v for k, v in self_state_dict.items()}

        lm_head_weight_key = "model.lm_head.weight"
        embed_tokens_weight_key = "model.model.embed_tokens.weight"

        weights_of_ckpt_names = set()
        reserved = {}
        scale_inv_paths = {}

        # 2. Build Scale Map (Rank 0 scans headers quickly)
        if rank == 0:
            for f in safetensors_files:
                # CPU load for headers is fast/cheap
                ckpt = retry(safe_open)(
                    os.path.join(model_path, f), framework="pt", device="cpu"
                )
                keys = ckpt.keys()
                for name in keys:
                    if name.endswith("weight_scale_inv"):
                        scale_inv_paths[name] = os.path.join(model_path, f)

        scale_inv_paths = _broadcast_data(scale_inv_paths, src_rank=0)

        # 3. Parallel Batch Loading
        # Process files in chunks of size 'world_size'
        total_files = len(safetensors_files)
        for batch_start in range(0, total_files, world_size):
            batch_end = min(batch_start + world_size, total_files)
            batch_files = safetensors_files[batch_start:batch_end]

            # --- A. PARALLEL READ PHASE ---
            # Determine which file in this batch belongs to the current rank
            # rank 0 gets batch_files[0], rank 1 gets batch_files[1], etc.
            local_file_index = rank
            local_file_name = None
            local_file_data = {}  # Buffer to hold the full file in RAM

            if local_file_index < len(batch_files):
                local_file_name = batch_files[local_file_index]
                logger.info(f"Rank {rank} parallely reading: {local_file_name}")

                # Load the ENTIRE file into CPU RAM
                f_path = os.path.join(model_path, local_file_name)
                ckpt = retry(safe_open)(f_path, framework="pt", device=str(device))
                for k in ckpt.keys():
                    local_file_data[k] = ckpt.get_tensor(k)

            # Barrier ensures all ranks have finished reading their assigned file
            # before we start the network-heavy broadcast phase.
            torch.distributed.barrier()

            # --- B. BROADCAST & PROCESS PHASE ---
            # Iterate through the batch. Each rank takes a turn being the 'sender'
            for i, _ in enumerate(batch_files):
                owner_rank = (batch_start + i) % world_size
                is_owner = rank == owner_rank

                # Owner knows the keys (from their local_file_data buffer)
                current_file_keys = list(local_file_data.keys()) if is_owner else None
                file_keys = _broadcast_data(current_file_keys, src_rank=owner_rank)

                for name in file_keys:
                    tensor = None
                    skip_weight = False

                    # -- Owner Dequantizes locally --
                    if is_owner:
                        tensor = local_file_data[name]

                        if name.endswith("weight_scale_inv") or "layers.61" in name:
                            skip_weight = True
                        else:
                            if (
                                (
                                    "down_proj" in name
                                    or "up_proj" in name
                                    or "gate_proj" in name
                                    or "self_attn.kv_a_proj_with_mqa" in name
                                    or "self_attn.kv_b_proj" in name
                                    or "self_attn.o_proj" in name
                                    or "self_attn.q_a_proj" in name
                                    or "self_attn.q_b_proj" in name
                                )
                                and "weight" in name
                                and dequant_weights
                            ):
                                inv_name = name + "_scale_inv"

                                # Try to find scale in local buffer first (fastest)
                                if inv_name in local_file_data:
                                    inv_tensor = local_file_data[inv_name]
                                    tensor = weight_dequant(tensor, inv_tensor)
                                # Fallback: scale is in a different file (rare, but possible)
                                elif inv_name in scale_inv_paths:
                                    inv_path = scale_inv_paths[inv_name]
                                    inv_tensor = retry(safe_open)(
                                        inv_path, framework="pt", device=str(device)
                                    ).get_tensor(inv_name)
                                    tensor = weight_dequant(tensor, inv_tensor)

                    # -- Sync Skip --
                    skip_val = 1 if skip_weight else 0
                    skip_tensor = torch.tensor(
                        [skip_val], device=device, dtype=torch.int8
                    )
                    dist.broadcast(skip_tensor, src=owner_rank)
                    if skip_tensor.item() == 1:
                        continue

                    weights_of_ckpt_names.add(name)

                    # -- Broadcast Metadata --
                    if is_owner:
                        meta = (tensor.shape, tensor.dtype)
                    else:
                        meta = None
                    tensor_shape, tensor_dtype = _broadcast_data(
                        meta, src_rank=owner_rank
                    )

                    # -- Broadcast Data --
                    if not is_owner:
                        tensor = torch.empty(
                            tensor_shape, dtype=tensor_dtype, device=device
                        )

                    if is_owner:
                        tensor = tensor.to(device).contiguous()

                    dist.broadcast(tensor, src=owner_rank)

                    # -- Local Slicing/Sharding (All Ranks) --
                    if name == embed_tokens_weight_key:
                        reserved[name] = tensor.clone()

                    dest_name, shared_weight, expert_id = convert_weight_from_hf(
                        tensor,
                        name,
                        model_type,
                        parallel_dims,
                        n_experts=self.config.n_routed_experts,
                    )

                    if dest_name is None:
                        continue

                    if dest_name not in self_state_dict and parallel_dims.pp_enabled:
                        continue

                    slice_range = None
                    if ".experts.gate_proj" in dest_name:
                        dest_name = dest_name.replace("gate_projs", "gate_and_up_projs")
                        slice_range = slice(0, self.config.moe_inter_dim)
                    elif ".experts.up_proj" in dest_name:
                        dest_name = dest_name.replace("up_projs", "gate_and_up_projs")
                        slice_range = slice(self.config.moe_inter_dim, None)

                    target_tensor = self_state_dict[dest_name]
                    if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                        target_tensor = target_tensor.to_local()

                    if expert_id is not None:
                        n_local_experts = (
                            self.config.n_routed_experts
                            // parallel_dims.tp
                            // parallel_dims.dp_shard
                        )
                        expert_id = expert_id % n_local_experts
                        target_tensor = target_tensor[expert_id]

                    if slice_range is not None:
                        assert (
                            target_tensor.shape[0] == 2 * self.config.moe_inter_dim
                        ), f"Shape mismatch for {dest_name}"
                        target_tensor = target_tensor[slice_range]

                    assert (
                        target_tensor.shape == shared_weight.shape
                    ), f"Shape mismatch: {target_tensor.shape} != {shared_weight.shape} for {dest_name}"

                    with torch.no_grad():
                        target_tensor.data.copy_(shared_weight)

                    del tensor
                    del shared_weight

            # Cleanup buffer to free RAM for next batch
            del local_file_data
            torch.cuda.empty_cache()
            if rank == 0:
                logger.info(
                    f"Finished processing batch starting at index {batch_start}"
                )

        if (
            lm_head_weight_key not in weights_of_ckpt_names
            and embed_tokens_weight_key in weights_of_ckpt_names
        ):
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
                ), f"Shape mismatch for {dest_name}"
                with torch.no_grad():
                    local_view.data.copy_(shared_weight)

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

    @classmethod
    def fqn_filter_for_quantization(cls) -> List[str]:
        return ["lm_head"]

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        if not (self.config.n_heads % (cp_size * tp_size) == 0):
            raise ValueError(
                f"Model is not compatible with cp parallelism, model's head number={self.config.n_heads} is not divisible by cp size({cp_size}) * tp_size({tp_size}) = {cp_size * tp_size}"
            )

    def check_tp_compatible(self, tp_size: int):
        non_divisible_by_tp_size = self.config.n_heads % tp_size != 0
        if non_divisible_by_tp_size:
            raise ValueError(
                f"Model is not compatible with tp parallelism, model's head number={self.config.n_heads} is not satisified by tp size({tp_size})"
            )


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
