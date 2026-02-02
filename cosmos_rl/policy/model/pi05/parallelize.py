# Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100, find_unused_parameters=True)
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
    # Check unsupported parallelism
    if parallel_dims.tp > 1:
        raise ValueError("PI05 does not support tensor parallelism")
    if parallel_dims.pp > 1:
        raise ValueError("PI05 does not support pipeline parallelism")
    if parallel_dims.cp > 1:
        raise ValueError("PI05 does not support context parallelism")
    if parallel_dims.dp_shard > 1:
        raise ValueError("PI05 does not support dp_shard (FSDP-style sharding)")

    # Apply DDP replicate, using the 1D dp_replicate submesh from Cosmos' N-D mesh.
    # Cosmos always includes a dp_shard dim in the global mesh (even if size=1), so
    # `parallel_dims.mesh` is often >1D. PI05 should replicate only across dp_replicate.
    if parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.mesh["dp_replicate"]
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=config.train.compile,
            enable_compiled_autograd=config.train.compile,
        )
    else:
        logger.info("PI05 running without DDP replicate (dp_replicate_size=1)")

    return None, None
