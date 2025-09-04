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

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.pipelining.pipelining_utils import (
    generate_split_points,
    stage_ids_this_rank,
)
from torch.distributed import DeviceMesh


def pipeline_model(
    model: nn.Module,
    meshes: dict[str, DeviceMesh],
    parallel_dims: ParallelDims,
    device: torch.device,
) -> list[nn.Module]:
    """
    Divides the model into multiple model parts, one per virtual pipeline stage.
    There could be 1 or more virtual pipeline stages per PP rank. Invokes
    `model.parallize_fn` on each model part.

    Args:
        model (nn.Module): The original nn.Module before pipelining and
            parallelization.
        meshes (dict[str, DeviceMesh]): A mapping of mesh names to DeviceMesh objects.
            There are generally two meshes, "default" for non-MoE modules and "moe"
            for MoE modules. Each mesh contains several dimensions for parallelization.
        parallel_dims (ParallelDims): The parallel dimensions configuration.
        device (torch.device): The device for the current PP rank.

    Returns:
        model_parts (list[nn.Module]): The list of model parts assigned to this
            PP rank.
    """

    pp_mesh = meshes["default"]["pp"]
    assert pp_mesh.size() == parallel_dims.pp
    # parallelize_fn, _ = model.parallelize_fn

    model_parts = _pipeline_manual_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        parallel_dims=parallel_dims,
    )

    # # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
    # # optimizer, and checkpointing

    # config = None
    # pp_loss_fn = None

    # for i, m in enumerate(model_parts):

    #     # apply SPMD-style PT-D techniques
    #     m = parallelize_fn(m, parallel_dims, config, pp_loss_fn)
    #     model_parts[i] = m

    _setup_communication_channels(pp_mesh, device)

    return model_parts


def _setup_communication_channels(pp_mesh: DeviceMesh, device: torch.device) -> None:
    """
    This function sets up the channels for pipeline parallel communication.
    In the forward pass, PP rank `i' sends activations to PP rank
    `(i+1) mod (pp_size)`.
    In the backward pass, PP rank `i` sends activation gradients to PP rank
    `(i + pp_size - 1) mod (pp_size)`.

    Doing this setup is required because the `dist.batch_isend_irecv` function
    requires all ranks in a process group to participate in the first collective,
    otherwise the behaviour is undefined. With the regular PP communication,
    only two ranks ever participate in the same collective, so we require a
    separate setup mechanism to ensure that a process group is correctly setup
    for dist.batch_isend_irecv collectives.

    Args:
        pp_mesh (DeviceMesh): The device mesh used for pipeline parallel
            communication.
        device (torch.device): The device for the current PP rank.
    """
    logger.info(f"Setting up pipeline connectivity, pp_mesh: {pp_mesh}")
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    send_tensor = torch.arange(2, dtype=torch.bfloat16, device=device) + 2 * pp_rank

    def _setup_fwd():
        recv_tensor = torch.randn(2, dtype=torch.bfloat16, device=device)
        send_op = dist.P2POp(
            op=dist.isend,
            tensor=send_tensor,
            group=pp_mesh.get_group(),
            group_peer=(pp_rank + 1) % pp_size,
        )
        recv_op = dist.P2POp(
            op=dist.irecv,
            tensor=recv_tensor,
            group=pp_mesh.get_group(),
            group_peer=(pp_rank - 1 + pp_size) % pp_size,
        )
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()
        logger.info(f"Forward pass setup received {recv_tensor}")  # , rank0_only=False)

    def _setup_bwd():
        recv_tensor = torch.randn(2, dtype=torch.bfloat16, device=device)
        send_op = dist.P2POp(
            op=dist.isend,
            tensor=send_tensor,
            group=pp_mesh.get_group(),
            group_peer=(pp_rank - 1 + pp_size) % pp_size,
        )
        recv_op = dist.P2POp(
            op=dist.irecv,
            tensor=recv_tensor,
            group=pp_mesh.get_group(),
            group_peer=(pp_rank + 1) % pp_size,
        )
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()
        logger.info(
            f"Backward pass setup received {recv_tensor}"
        )  # , rank0_only=False)

    _setup_fwd()
    _setup_bwd()


def _pipeline_manual_split(
    whole_model: nn.Module, pp_mesh: DeviceMesh, parallel_dims: ParallelDims
) -> list[nn.Module]:
    """
    This API extracts one torch.nn.Module objects for the parts of the model
    configured to run inside this ranks.

    These objects are later wrapped within a PipelineStage object, and passed
    into a _PipelineSchedule.

    Args:
        whole_model (nn.Module): The original nn.Module before pipelining.
        pp_mesh (DeviceMesh): The device mesh used for pipeline parallel
            communication.

    Returns:
        model_parts (list[nn.Module]): The list of model parts assigned to this
            PP rank.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    parallelism_config = parallel_dims
    num_dense_layers = whole_model.config.n_dense_layers
    num_moe_layers = whole_model.config.n_layers - num_dense_layers
    schedule_str = parallelism_config.pp_schedule

    splits = generate_split_points(
        schedule_str=schedule_str,
        pp_size=pp_size,
        num_layers_per_stage=parallelism_config.pp_layers_per_stage,
        num_moe_layers=num_moe_layers,
        num_dense_layers=num_dense_layers,
    )

    num_stages = len(splits) - 1

    model_parts = []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, schedule_str):
        model_chunk = _build_stage(
            whole_model=whole_model,
            stage_idx=stage_idx,
            num_stages=num_stages,
            start_layer=splits[stage_idx],
            stop_layer=splits[stage_idx + 1],
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {splits[stage_idx]}, stop_layer {splits[stage_idx + 1]}",
            # rank0_only=False,
        )
        model_parts.append(model_chunk)
    return model_parts


def _build_stage(
    whole_model: nn.Module,
    stage_idx: int,
    num_stages: int,
    start_layer: str,
    stop_layer: str,
) -> nn.Module:
    """
    Builds the model part for a virtual pipeline stage by dropping nn.Module
    objects that are not assigned to this stage. Dropping is accomplished by
    setting the module reference to None. For ModuleDict objects, we can
    simply delete the dropped modules from the dictionary.

    The input modules (before the first transformer layer) are assigned to
    stage `0`, and the output modules (after the last transformer layer) are
    assigned to stage `num_stages - 1`. The transformer layers from
    `start_layer` (inclusive) to `stop_layer` (exclusive) are assigned to this
    stage.

    Return the nn.Module after dropping the modules that are not assigned to
    this stage.
    """
    model = copy.deepcopy(whole_model)
    if stage_idx != 0:
        model.vision_encoder = None
        model.mm_projector = None
        model.model.model.embed_tokens = None

    # Layers are kept in a contiguous region between start_layer (inclusive)
    # and stop_layer (exclusive).
    drop_layers = True
    for name in list(model.model.model.layers.keys()):
        if f"layers.{name}" == start_layer:
            drop_layers = False
        if f"layers.{name}" == stop_layer:
            drop_layers = True
        if drop_layers:
            del model.model.model.layers[name]

    if stage_idx != (num_stages - 1):
        model.model.model.norm = None
        model.model.lm_head = None

    return model
