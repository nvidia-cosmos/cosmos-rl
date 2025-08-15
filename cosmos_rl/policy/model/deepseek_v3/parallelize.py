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


import functools
import os
from typing import Callable, Optional

import torch
from torch import nn
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor import Shard, distribute_module, distribute_tensor
except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")

from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module

try:
    from torch.distributed.fsdp import fully_shard
except ImportError:
    print("torch.distributed.fsdp is not available. DeepSeek model will not work.")

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from transformer_engine.pytorch.attention import DotProductAttention

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.kernel.moe.moe import GroupedExpertsDeepEP, MoE
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.ulysses import swizzle_cp_forward, ulysses_attn_func


def _get_dp_mesh(
    world_mesh: DeviceMesh, parallel_dims: ParallelDims
) -> DeviceMesh | None:
    """
    Gets the HSDP or FSDP mesh based on the parallelism settings.
    If only FSDP is used, the model parameters are sharded on the FSDP mesh.
    If HSDP is used, the model parameters are both replicated and sharded
    across different dimensions of the mesh.
    """
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        if parallel_dims.dp_replicate_enabled:
            # HSDP: Replicate on `dp_replicate` dim and shard on `dp_shard_cp` dim.
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            # FSDP: Replicate on `dp_shard_cp` dim.
            dp_mesh_dim_names = ("dp_shard_cp",)
        return world_mesh[tuple(dp_mesh_dim_names)]
    else:
        assert not parallel_dims.dp_replicate_enabled
        return None


def _apply_cp(model: nn.Module, cp_mesh: DeviceMesh, parallel_dims: ParallelDims):
    """Apply Context Parallel to the model."""
    assert cp_mesh.size() > 1
    assert cp_mesh.ndim == 1

    if model.model is not None:
        _model = model.model

        for _, block in _model.model.layers.named_children():
            attn_module = block.self_attn.attn_module
            assert isinstance(
                attn_module, DotProductAttention
            ), "Context parallelism is only supported for DotProductAttention."
            assert attn_module.num_attention_heads % cp_mesh.size() == 0, (
                f"Number of attention heads {attn_module.num_attention_heads} must be divisible by "
                f"context parallel size {cp_mesh.size()}"
            )

            attn_module_with_cp = DotProductAttention(
                num_attention_heads=attn_module.num_attention_heads // cp_mesh.size(),
                kv_channels=(
                    attn_module.hidden_size_per_attention_head_k,
                    attn_module.hidden_size_per_attention_head_v,
                ),
                attn_mask_type=attn_module.attn_mask_type,
                qkv_format=attn_module.qkv_format,
                softmax_scale=block.self_attn.softmax_scale,
            )
            attn_func_with_cp = ulysses_attn_func(
                attn_module_with_cp.__call__,
                cp_mesh,
            )

            block.self_attn.attn_module = attn_module_with_cp
            block.self_attn.attn_func = attn_func_with_cp

        if _model.lm_head is not None:
            # Apply CP to the lm_head to get the logits for all tokens.
            swizzle_cp_forward(_model.lm_head, parallel_dims)


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


def _apply_ep(model: nn.Module, ep_mesh: DeviceMesh):
    """Applies EP to MoE module."""
    assert ep_mesh.size() > 1

    if model.model is not None:
        _model = model.model
        for _, block in _model.model.layers.named_children():
            if isinstance(block.mlp, MoE):
                parallelize_module(
                    module=block.mlp.experts,
                    device_mesh=ep_mesh,
                    parallelize_plan=_ExpertParallel(),
                )


def _apply_ac(model: nn.Module):
    """Apply activation checkpointing to the model."""
    if model.model is not None:
        _model = model.model
        for layer_id, block in _model.model.layers.named_children():
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
            _model.model.layers.register_module(layer_id, block)


def _apply_fsdp(
    model: nn.Module,
    meshes: dict[str, DeviceMesh],
    parallel_dims: ParallelDims,
):
    """Apply FSDP sharding to model layers using data parallel mesh."""
    default_dp_mesh = _get_dp_mesh(meshes["default"], parallel_dims)
    if default_dp_mesh is None:
        return

    pp_enabled = parallel_dims.pp_enabled

    fully_shard_default = functools.partial(
        fully_shard,
        mesh=default_dp_mesh,
        reshard_after_forward=not pp_enabled,
    )

    if model.model is not None:
        _model = model.model

        for _, block in _model.model.layers.named_children():
            if isinstance(block.mlp, MoE) and parallel_dims.dp_shard_with_ep_enabled:
                # Apply FSDP on dim=1 for grouped experts since we may have more
                # shards than experts (dim=0).
                assert "moe" in meshes
                fully_shard(
                    block.mlp.experts,
                    mesh=meshes["moe"]["dp_shard_with_ep"],
                    shard_placement_fn=lambda _: Shard(1),
                    reshard_after_forward=not pp_enabled,
                )

            # If FSDP is disabled for grouped experts because the parameters are already
            # fully sharded by PP and EP, then we need to explicitly remove the parameters
            # from FSDP for the transformer block.
            # If FSDP is enabled for grouped experts, the parameters are automatically
            # removed from the FSDP for the transformer block due to the rules of the
            # PyTorch FSDP implementation.
            ignored_params = None
            if (
                isinstance(block.mlp, MoE)
                and parallel_dims.ep_enabled
                and not parallel_dims.dp_shard_with_ep_enabled
            ):
                ignored_params = set(block.mlp.experts.parameters())

            fully_shard_default(block, ignored_params=ignored_params)

        fully_shard_default(_model)


def _get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module: torch.cuda
    return device_type, device_module


def _init_meshes(
    parallel_dims: ParallelDims,
) -> dict[str, DeviceMesh]:
    """
    Initialize the meshes for the model. There are generally two meshes, "default"
    for non-MoE modules and "moe" for MoE modules. Each mesh contains several
    dimensions for parallelization.

    Args:
        parallelism_config (TrainingConfig): The parallelism configuration for the model.

    Returns:
        meshes (dict[str, DeviceMesh]): The meshes for the model.
    """

    device_type, device_module = _get_device_info()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    meshes = parallel_dims.build_meshes_with_ep(device_type=device_type)
    return meshes


def parallelize_model(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable],
) -> nn.Module:
    """
    Parallelizes the DeepSeek model based on the provided meshes and parallel dimensions.

    Args:
        model (nn.Module): The DeepSeek model to parallelize. Note that this maybe
            a model part due to pipelining. We must check for the presence of an
            nn.Module before executing the relevant parallelization functions.

    Returns:
        nn.Module: The parallelized DeepSeek model.
    """
    meshes = _init_meshes(parallel_dims)
    del config
    del pp_loss_fn

    assert model.config.n_routed_experts % parallel_dims.ep == 0, (
        f"n_routed_experts {model.config.n_routed_experts} must be divisible by "
        f"expert_parallel_degree {parallel_dims.ep}"
    )
    assert parallel_dims.tp == 1, "Tensor parallelism not support for DeepSeek model"

    if parallel_dims.cp_enabled:
        _apply_cp(model, meshes["default"]["cp"], parallel_dims)

    if parallel_dims.ep_enabled:
        assert "moe" in meshes
        _apply_ep(model, meshes["moe"]["ep"])

    _apply_ac(model)

    _apply_fsdp(model, meshes, parallel_dims)

    return None, None
