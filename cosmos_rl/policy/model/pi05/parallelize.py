# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Optional, Callable

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.replicate import replicate

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    """Apply DDP to the model."""
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
    logger.info("Applied DDP to PI05 model")


def parallelize(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config,
    pp_loss_fn: Optional[Callable] = None,
):
    """
    Apply DDP parallelism to PI05 model.

    Note: PI05 only supports DDP, not TP/PP/FSDP.
    """
    world_mesh = parallel_dims.mesh

    # Check unsupported parallelism
    if parallel_dims.tp > 1:
        raise ValueError("PI05 does not support tensor parallelism")
    if parallel_dims.pp > 1:
        raise ValueError("PI05 does not support pipeline parallelism")

    # Apply DDP
    if world_mesh.ndim > 1:
        raise RuntimeError("PI05 DDP only supports 1D parallelism")

    apply_ddp(
        model,
        world_mesh,
        enable_compile=config.train.compile,
        enable_compiled_autograd=config.train.compile,
    )

    return None, None
