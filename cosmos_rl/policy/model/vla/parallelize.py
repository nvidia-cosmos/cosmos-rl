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
from typing import Dict, Any, Optional, Union, Callable
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel, 
    PrepareModuleInput,
    SequenceParallel,
)
from torch.distributed._tensor import Shard, Replicate

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger


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


def get_vla_fsdp_wrap_policy(
    model: nn.Module,
    parallel_dims: ParallelDims,
    **kwargs  
) -> Optional[Callable]:
    """
    Get FSDP wrap policy for VLA models
    
    VLA models benefit from wrapping at the transformer block level
    to balance memory usage and communication overhead.
    
    Args:
        model: VLA model instance  
        parallel_dims: Parallelization configuration
        
    Returns:
        FSDP wrap policy function or None
    """
    
    if parallel_dims.dp_shard <= 1:
        return None
        
    try:
        from torch.distributed.fsdp import ModuleWrapPolicy
        from functools import partial
        
        # Modules to wrap with FSDP
        wrap_modules = set()
        
        # Language model transformer blocks
        if hasattr(model, 'language_model'):
            # Try to find transformer blocks
            for name, module in model.named_modules():
                if ('layers' in name and 
                    ('TransformerBlock' in str(type(module)) or 
                     'Block' in str(type(module)) or
                     'Layer' in str(type(module)))):
                    wrap_modules.add(type(module))
                    
        # Vision backbone blocks (if large enough)
        if hasattr(model, 'vision_backbone'):
            for name, module in model.named_modules():
                if ('vision_backbone' in name and 
                    'Block' in str(type(module))):
                    wrap_modules.add(type(module))
        
        if wrap_modules:
            logger.info(f"FSDP wrap policy will wrap: {wrap_modules}")
            return ModuleWrapPolicy(wrap_modules)
        else:
            logger.warning("No suitable modules found for FSDP wrapping")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create FSDP wrap policy: {e}")
        return None


def apply_vla_data_parallelism(
    model: nn.Module,
    parallel_dims: ParallelDims,
    **kwargs
) -> nn.Module:
    """
    Apply FSDP data parallelism to VLA model
    
    Args:
        model: VLA model to parallelize
        parallel_dims: Parallelization configuration
        
    Returns:
        FSDP wrapped model
    """
    
    if parallel_dims.dp_shard <= 1:
        logger.info("DP size <= 1, skipping data parallelism") 
        return model
        
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        
        # Get wrap policy
        auto_wrap_policy = get_vla_fsdp_wrap_policy(model, parallel_dims, **kwargs)
        
        # Mixed precision policy
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
        
        # Apply FSDP
        model = FSDP(
            model,
            process_group=parallel_dims.dp_mesh,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
        )
        
        logger.info(f"Applied FSDP to VLA model with DP size {parallel_dims.dp_shard}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to apply FSDP to VLA model: {e}")
        raise


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
    2. Data Parallelism (across nodes)  
    3. Pipeline Parallelism (future)
    
    Args:
        model: VLA model to parallelize
        parallel_dims: Parallelization configuration
        config: Cosmos configuration object
        pp_loss_fn: Pipeline loss function (optional)
        
    Returns:
        Tuple of (pp_scheduler, pp_scheduler_val) for pipeline parallelism
    """
    
    logger.info("Starting VLA model parallelization")
    logger.info(f"Parallel dimensions: TP={parallel_dims.tp}, "
                f"DP={parallel_dims.dp_shard}, PP={parallel_dims.pp}")
    
    # Step 1: Apply Tensor Parallelism
    if parallel_dims.tp > 1:
        model = apply_vla_tensor_parallelism(model, parallel_dims)
    
    # Step 2: Apply Data Parallelism  
    if parallel_dims.dp_shard > 1:
        model = apply_vla_data_parallelism(model, parallel_dims)
    
    # Step 3: Pipeline Parallelism (TODO: future implementation)
    if parallel_dims.pp > 1:
        logger.warning("Pipeline parallelism not yet implemented for VLA models")
        # TODO: Implement pipeline parallelism for VLA models
        # For now, return None schedulers
        return None, None
    else:
        # No pipeline parallelism - return None schedulers
        return None, None

