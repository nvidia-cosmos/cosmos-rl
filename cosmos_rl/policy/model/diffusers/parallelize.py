from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import str2torch_dtype
from cosmos_rl.utils.parallelism import ParallelDims, pre_parallelize_sanity_check
from cosmos_rl.policy.config import Config as CosmosConfig

# TODO (yy): only support FSDP now, may support other parallelization policy if needed
# TODO (yy): Add EMA warpper
@pre_parallelize_sanity_check
def parallelize(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable] = None,
    ):
    world_mesh = parallel_dims.mesh
    _, pp_size = parallel_dims.pp_coord

    assert pp_size == 1, "Pipeline parallelism is not supported for DiffuserModel"
    assert not config.train.compile, "Compile is not supported for DiffuserModel"
    if parallel_dims.dp_shard_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model.transformer,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=str2torch_dtype(config.train.param_dtype),
            reduce_dtype=str2torch_dtype(config.train.fsdp_reduce_dtype),
            pp_enabled=False,
            cpu_offload=config.train.fsdp_offload,
            reshard_after_forward_policy=config.train.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")
        
    return None, None



def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    high_precision_mp_policy = MixedPrecisionPolicy(param_dtype=reduce_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # For diffusers, only shard transformer now
    fsdp_non_split_modules = getattr(model, "_no_split_modules", None)
    layerwise_high_precision = getattr(model, "_skip_layerwise_casting_patterns", None)

    high_precision_modules = []
    non_split_modules = []
    
    for module in model.modules():
        if module.__class__.__name__ in layerwise_high_precision:
            high_precision_modules.append(module)
        if module.__class__.__name__ in fsdp_non_split_modules:
            non_split_modules.append(module)

    for module in high_precision_modules:
        fsdp_config["mp_policy"] = high_precision_mp_policy
        fully_shard(module, **fsdp_config, reshard_after_forward=True)
    
    for module in non_split_modules:
        fsdp_config["mp_policy"] = mp_policy
        fully_shard(module, **fsdp_config, reshard_after_forward=True)

    fsdp_config["mp_policy"] = mp_policy
    fully_shard(model, **fsdp_config, reshard_after_forward=True)

    
    
