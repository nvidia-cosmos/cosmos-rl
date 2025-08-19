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
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import str2torch_dtype
from cosmos_rl.policy.config import Config as CosmosConfig
from typing import Callable, Optional


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
        not parallel_dims.tp_enabled
    ), "Tensor parallelism is not supported for HFModel"
    assert (
        not parallel_dims.cp_enabled
    ), "Context parallelism is not supported for HFModel"
    assert pp_size == 1, "Pipeline parallelism is not supported for HFModel"
    assert not config.train.compile, "Compile is not supported for HFModel"

    # apply FSDP or HSDP
    if parallel_dims.dp_shard_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
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

    return None, None


# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
}


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
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
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
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
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        fully_shard(
            model.vision_model,
            **fsdp_config,
            reshard_after_forward=True,
        )

    # Shard the multi-modal projector
    if model.multi_modal_projector is not None:
        fully_shard(
            model.multi_modal_projector,
            **fsdp_config,
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
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.embed_tokens is not None:
        logger.info("Applying FSDP to the language model embed_tokens")
        fully_shard(model.embed_tokens, **fsdp_config, reshard_after_forward=True)
    fully_shard(model.language_model, **fsdp_config, reshard_after_forward=True)
    fully_shard(model, **fsdp_config, reshard_after_forward=True)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
