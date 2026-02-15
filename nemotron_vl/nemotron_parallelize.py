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

from typing import Callable, Optional
import os
import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.tensor.parallel import parallelize_module, ParallelStyle
from torch.distributed.tensor import Shard, distribute_module, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from cosmos_rl.policy.kernel.moe.moe import GroupedExpertsDeepEP
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.utils.parallelism import ParallelDims 
from cosmos_rl.policy.config import Config as CosmosConfig


class _ExpertParallel(ParallelStyle):
    """
    ExpertParallel class is used to shard the MoE parameters on the EP mesh.
    Dim `0` of each parameter is sharded since that is the expert dimension.
    """

    def _partition_fn(self, name, module, device_mesh):
        # shard on the expert dimension
        assert device_mesh.ndim == 1

        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, dist_param)

        if isinstance(module, GroupedExpertsDeepEP):
            module.init_token_dispatcher(ep_mesh=device_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )

def parallelize(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable] = None,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.mesh
    _, pp_size = parallel_dims.pp_coord

    assert (
        not parallel_dims.cp_enabled
    ), "Context parallelism is not supported for HFModel"
    assert pp_size == 1, "Pipeline parallelism is not supported for HFModel"

    if parallel_dims.tp_enabled:
        # Check deep_ep is installed
        try:
            import deep_ep
        except ImportError:
            raise ImportError("deep_ep is not installed. Please install it first to enable EP support.")

        # Swizzle the MoE module to use the internal MoE module
        # @dataclass
        # class MoEArgs:
        #     n_routed_experts: int
        #     n_shared_experts: int
        #     n_activated_experts: int
        #     n_expert_groups: int
        #     n_limited_groups: int
        #     train_gate: bool
        #     gate_bias_update_factor: float
        #     aux_loss_coeff: float
        #     score_func: str
        #     route_scale: float
        #     dim: int
        #     moe_inter_dim: int
        #     norm_topk_prob: bool = False
        #     fake_balanced_gate: bool = False
        #     enable_router_bias: bool = False
        #     enable_glu: bool = True
        #     act_fn: Optional[str] = None
        from cosmos_rl.policy.kernel.moe.moe import MoE, MoEArgs

        # TODO(jiaxinc): to be compatible with both pure text and vision-language models.
        language_model_config = model.hf_config.text_config if 'text_config' in model.hf_config else model.hf_config
        gate_training_enabled = config.custom.get("enable_moe_load_balancing_training", True)
        moe_args = MoEArgs(
            n_routed_experts=language_model_config.n_routed_experts,
            n_shared_experts=language_model_config.n_shared_experts,
            n_activated_experts=language_model_config.num_experts_per_tok,
            n_expert_groups=language_model_config.n_group,
            n_limited_groups=language_model_config.topk_group,
            train_gate=gate_training_enabled,
            gate_bias_update_factor=1e-3 if gate_training_enabled else 0.0,
            aux_loss_coeff=1e-4 if gate_training_enabled else 0.0,
            score_func="sigmoid", # TODO(jiaxinc): hardcoded to be consistent with the original implementation
            route_scale=language_model_config.routed_scaling_factor,
            dim=language_model_config.hidden_size,
            moe_inter_dim=language_model_config.moe_intermediate_size,
            shared_inter_dim=language_model_config.moe_shared_expert_intermediate_size,
            norm_topk_prob=language_model_config.norm_topk_prob,
            enable_router_bias=True,
            enable_glu=False,
            act_fn=language_model_config.mlp_hidden_act
        )

        cosmos_default_dtype = util.str2torch_dtype(
            config.train.master_dtype
            if config.train.master_dtype is not None
            else config.train.param_dtype
        )
        with util.cosmos_default_dtype(cosmos_default_dtype):
            for layer_id, transformer_block in enumerate(model.lm_layers):
                if transformer_block.block_type == "moe":
                    del transformer_block.mixer
                    with torch.device('meta'):
                        transformer_block.register_module("mixer", MoE(moe_args))
                    logger.info(f"Swizzled MoE module for layer {layer_id}")
                    parallelize_module(
                        module=transformer_block.mixer.experts,
                        device_mesh=world_mesh["tp"],
                        parallelize_plan=_ExpertParallel(),
                    )
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    # # Only allow Attention layer to be trained
    # for layer_id, transformer_block in enumerate(model.lm_layers):
    #     if transformer_block.block_type == "attention":
    #         for name, param in transformer_block.named_parameters():
    #             print(f"Setting {name} to train mode")
    #             param.requires_grad = True
    #         logger.info(f"Set Attention layer {layer_id} to train mode")
    # model.model.enable_input_require_grads()

    # apply_compile(model, fullgraph=True)

    # apply FSDP or HSDP
    if parallel_dims.dp_shard_enabled:
        dp_group_names = ["dp_shard"]
        if parallel_dims.cp_enabled:
            dp_group_names.append("cp")
        if parallel_dims.tp_enabled:
            fsdp_mesh_no_moe = world_mesh[tuple(dp_group_names + ["tp"])]._flatten(
                mesh_dim_name="fsdp_no_moe"
            )
            fsdp_mesh_no_moe_name = "fsdp_no_moe"
            fsdp_mesh_moe = world_mesh[tuple(dp_group_names)]._flatten(
                mesh_dim_name="fsdp_moe"
            )
            fsdp_mesh_moe_name = "fsdp_moe"
        else:
            fsdp_mesh_no_moe = fsdp_mesh_moe = world_mesh[tuple(("dp_shard_cp",))]
            fsdp_mesh_no_moe_name = fsdp_mesh_moe_name = "dp_shard_cp"

        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names_for_no_moe = ("dp_replicate", fsdp_mesh_no_moe_name)
            dp_mesh_dim_names_for_moe = ("dp_replicate", fsdp_mesh_moe_name)
        else:
            dp_mesh_dim_names_for_no_moe = (fsdp_mesh_no_moe_name,)
            dp_mesh_dim_names_for_moe = (fsdp_mesh_moe_name,)

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names_for_no_moe)],
            world_mesh[tuple(dp_mesh_dim_names_for_moe)],
            param_dtype=util.str2torch_dtype(config.train.param_dtype),
            reduce_dtype=util.str2torch_dtype(config.train.fsdp_reduce_dtype),
            pp_enabled=False,
            cpu_offload=config.train.fsdp_offload,
            reshard_after_forward_policy=config.train.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if config.train.fsdp_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=False,
            enable_compiled_autograd=False,
        )
    
    train_layers = config.custom.get("train_layers", None)
    if train_layers is not None:
        logger.info(f"All trainable layers: {train_layers}")
        for name, parameters in model.named_parameters():
            parameters.requires_grad = False

        for _, module in model.named_modules():
            # module might be warpped with FSDP, it will change class name to f"FSDP{original_module_name}"
            if type(module).__name__.replace('FSDP', '') in train_layers:
                for name, parameters in module.named_parameters():
                    # e_correction_bias always be not trainable
                    if 'e_correction_bias' not in name:
                        parameters.requires_grad = True
        
        # make lm_head trainable
        if 'lm_head' in train_layers:
            if model.lm_head is not None:
                for name, parameters in model.lm_head.named_parameters():
                    parameters.requires_grad = True

    return None, None

def apply_compile(model: nn.Module, fullgraph: bool = True):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.model.backbone.layers.named_children():
        if transformer_block.block_type == "moe":
            transformer_block = torch.compile(transformer_block, fullgraph=fullgraph)
            model.model.backbone.layers.register_module(layer_id, transformer_block)
            logger.info("Each TransformerBlock compiled with torch.compile")

def apply_fsdp(
    model: nn.Module,
    fsdp_mesh_no_moe: DeviceMesh,
    fsdp_mesh_moe: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config_no_moe = {"mesh": fsdp_mesh_no_moe, "mp_policy": mp_policy}
    fsdp_config_moe = {"mesh": fsdp_mesh_moe, "mp_policy": mp_policy}
    
    if cpu_offload:
        fsdp_config_no_moe["offload_policy"] = CPUOffloadPolicy()
        fsdp_config_moe["offload_policy"] = CPUOffloadPolicy()

    # Shard the vision model
    if model.vision_model is not None:
        logger.info("Applying FSDP to the visual model")
        for layer_id, transformer_block in enumerate(model.vision_layers):
            if reshard_after_forward_policy == "always":
                reshard_after_forward = True
            elif reshard_after_forward_policy == "never":
                reshard_after_forward = False
            elif reshard_after_forward_policy == "default":
                reshard_after_forward = int(layer_id) < model.n_vision_layers - 1
            else:
                raise ValueError(
                    f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
                )
            fully_shard(
                transformer_block,
                **fsdp_config_no_moe,
                reshard_after_forward=reshard_after_forward,
            )

    # Shard the multi-modal projector, 
    # If module is inside vision_model, apply fsdp before shard whole vision_model
    if model.multi_modal_projector is not None:
        fully_shard(
            model.multi_modal_projector,
            **fsdp_config_no_moe,
            reshard_after_forward=True,
        )
    
    fully_shard(
        model.vision_model,
        **fsdp_config_no_moe,
        reshard_after_forward=True,
    )

    # Shard the language model
    for layer_id, transformer_block in enumerate(model.lm_layers):
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            reshard_after_forward = int(layer_id) < model.n_lm_layers - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

        # Bottom-up: first shard the experts in EP group, then shard the transformer block in big-dp group
        if transformer_block.block_type == "moe":
            fully_shard(
                transformer_block.mixer.experts,
                **fsdp_config_moe,
                reshard_after_forward=reshard_after_forward,
            )

        fully_shard(
            transformer_block,
            **fsdp_config_no_moe,
            reshard_after_forward=reshard_after_forward,
        )

    embed_tokens = model.embed_tokens

    if embed_tokens is not None:
        logger.info("Applying FSDP to the language model embeddings")
        fully_shard(embed_tokens, **fsdp_config_no_moe, reshard_after_forward=True)
    fully_shard(model.model, **fsdp_config_no_moe, reshard_after_forward=True)

def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")

