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

"""
Distributed parallelization utilities for VLA models

This module provides utilities for applying tensor parallelism (TP), 
data parallelism (FSDP), and pipeline parallelism (PP) to VLA models
in the cosmos-rl framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Union
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel, 
    PrepareModuleInput,
)

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import str2torch_dtype


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    """Apply DDP to the model"""
    from torch.distributed._composable.replicate import replicate
    
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
    logger.info("Applied DDP to the model")


def get_vla_tp_parallelize_plan(
    model: nn.Module,
    parallel_dims: ParallelDims,
    **kwargs
) -> Dict[str, Union[ColwiseParallel, RowwiseParallel, PrepareModuleInput]]:
    """
    Create tensor parallelism plan for VLA models
    
    VLA models have a complex multimodal architecture:
    - Vision backbone (typically not parallelized)
    - Vision-language projector (typically not parallelized)  
    - Language model (parallelized like standard transformers)
    - Action head (can be parallelized)
    
    Args:
        model: VLA model instance
        parallel_dims: Parallelization dimensions
        
    Returns:
        Dictionary mapping module names to parallelization specs
    """
    
    tp_size = parallel_dims.tp
    if tp_size <= 1:
        return {}
    
    plan = {}
    
    # Language model tensor parallelism (similar to Llama/Vicuna)
    if hasattr(model, 'language_model'):
        # Attention layers
        plan.update({
            # Query, Key, Value projections - column parallel
            "language_model.model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "language_model.model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "language_model.model.layers.*.self_attn.v_proj": ColwiseParallel(),
            
            # Output projection - row parallel
            "language_model.model.layers.*.self_attn.o_proj": RowwiseParallel(),
            
            # MLP layers
            "language_model.model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "language_model.model.layers.*.mlp.up_proj": ColwiseParallel(), 
            "language_model.model.layers.*.mlp.down_proj": RowwiseParallel(),
        })
        
        # Embeddings
        plan.update({
            "language_model.model.embed_tokens": ColwiseParallel(),
            "language_model.lm_head": ColwiseParallel(),
        })
    
    # Action head parallelization (if present)
    if hasattr(model, 'action_head'):
        plan.update({
            "action_head": ColwiseParallel(),
        })
        
    # Vision backbone - typically kept replicated for stability
    # Projector - typically kept replicated
    
    logger.info(f"Created VLA TP plan with {len(plan)} parallelized modules")
    return plan


def apply_vla_tensor_parallelism(
    model: nn.Module,
    parallel_dims: ParallelDims,
    **kwargs
) -> nn.Module:
    """
    Apply tensor parallelism to VLA model
    
    Args:
        model: VLA model to parallelize
        parallel_dims: Parallelization configuration
        
    Returns:
        Parallelized model
    """
    
    if parallel_dims.tp <= 1:
        logger.info("TP size <= 1, skipping tensor parallelism")
        return model
        
    try:
        from torch.distributed.tensor.parallel import parallelize_module
        
        # Get parallelization plan
        tp_plan = get_vla_tp_parallelize_plan(model, parallel_dims, **kwargs)
        
        if not tp_plan:
            logger.warning("No TP plan generated for VLA model")
            return model
            
        # Apply tensor parallelism
        model = parallelize_module(
            model,
            parallel_dims.tp_mesh,
            parallelize_plan=tp_plan
        )
        
        logger.info(f"Applied tensor parallelism to VLA model with TP size {parallel_dims.tp}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to apply tensor parallelism to VLA model: {e}")
        raise


def apply_vla_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply FSDP to VLA model with selective sharding strategy:
    - Replicate vision backbone (small, ~400M params)
    - Shard language model (large, ~7B params)
    - Replicate projector and action head (tiny)
    
    This balances memory efficiency with communication overhead.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy
    from torch.distributed._composable.fsdp import fully_shard
    
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    
    # NOTE: We don't use CPUOffloadPolicy here for two reasons:
    # 1. It provides limited flexibility (can't separately control params/grads/optimizer)
    # 2. It stores BF16 on CPU which causes numerical errors
    # 
    # Instead, use manual offloading with cosmos_rl.utils.fsdp2_offload_utils:
    # - offload_params: Offload parameters (stores as FP32 on CPU for stability)
    # - offload_grads: Offload gradients separately
    # - offload_optimizer: Offload optimizer states (biggest memory savings)
    #
    # See examples/fsdp2_manual_offload_example.py for usage.
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    
    # VLAModel wraps the actual model in self.model
    actual_model = model.model if hasattr(model, 'model') else model

    if reshard_after_forward_policy == "always":
        reshard_after_forward = True
    elif reshard_after_forward_policy == "never":
        reshard_after_forward = False
    else: # default
        reshard_after_forward = True
    
    # Count layers for logging
    if hasattr(actual_model, 'language_model') and actual_model.language_model is not None:
        llm = actual_model.language_model
        if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
            n_layers = len(llm.model.layers)
            logger.debug(f"Language model has {n_layers} transformer layers (will be sharded by top-level FSDP)")
            for i in range(n_layers):
                fully_shard(llm.model.layers[i], **fsdp_config, reshard_after_forward=reshard_after_forward)
            fully_shard(llm.model.embed_tokens, **fsdp_config, reshard_after_forward=reshard_after_forward)
            fully_shard(llm.model.norm, **fsdp_config, reshard_after_forward=reshard_after_forward)
        fully_shard(llm.lm_head, **fsdp_config, reshard_after_forward=reshard_after_forward)
    
    fully_shard(actual_model.vision_backbone, **fsdp_config, reshard_after_forward=reshard_after_forward)
    fully_shard(actual_model.projector, **fsdp_config, reshard_after_forward=reshard_after_forward)


def parallelize_vla_model(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config,
    pp_loss_fn: Optional[Callable] = None,
):
    """
    Apply all forms of parallelism to VLA model
    
    Order of operations:
    1. Tensor Parallelism (within node)
    2. Data Parallelism (FSDP - selective sharding)
    3. DDP (if no FSDP)
    4. Pipeline Parallelism (future)
    
    VLA-specific strategy:
    - Language model (7B): FSDP sharded across GPUs
    - Vision backbone (400M): Replicated (no sharding)
    - Projector/Action head: Replicated (tiny)
    
    Args:
        model: VLA model to parallelize
        parallel_dims: Parallelization configuration
        config: Cosmos configuration object
        pp_loss_fn: Pipeline loss function (optional)
        
    Returns:
        Tuple of (pp_scheduler, pp_scheduler_val) for pipeline parallelism
    """
    # Get world mesh
    world_mesh = parallel_dims.mesh
    
    # Step 1: Apply Tensor Parallelism
    if parallel_dims.tp > 1:
        model = apply_vla_tensor_parallelism(model, parallel_dims)
    
    # Step 2: Apply Data Parallelism
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        # Apply FSDP or HSDP with selective sharding
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        
        apply_vla_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=str2torch_dtype(config.train.param_dtype),
            reduce_dtype=str2torch_dtype(config.train.fsdp_reduce_dtype),
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=config.train.fsdp_offload,
            reshard_after_forward_policy=config.train.fsdp_reshard_after_forward,
        )
            
    elif parallel_dims.dp_replicate_enabled:
        # Apply DDP (no sharding, full replication)
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        
        apply_ddp(
            model,
            world_mesh,
            enable_compile=config.train.compile,
            enable_compiled_autograd=config.train.compile,
        )
    
    # Step 3: Pipeline Parallelism (TODO: future implementation)
    if parallel_dims.pp > 1:
        logger.warning("Pipeline parallelism not yet implemented for VLA models")
        return None, None
    else:
        return None, None

