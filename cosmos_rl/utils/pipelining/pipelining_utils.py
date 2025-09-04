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

import math
from typing import Callable

import torch
from cosmos_rl.policy.model.deepseek_v3.pipeline_parallelism.pipeline_schedules import (
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    _PipelineSchedule,
    get_schedule_class,
)
from cosmos_rl.policy.model.deepseek_v3.pipeline_parallelism.pipeline_stage import (
    PipelineStage,
)
from cosmos_rl.utils.logging import logger
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

__all__ = ["build_pipeline_schedule", "generate_split_points", "stage_ids_this_rank"]


def generate_split_points(
    schedule_str: str,
    pp_size: int,
    num_layers_per_stage: int | None,
    num_moe_layers: int,
    num_dense_layers: int,
) -> list[str]:
    """
    Generate a list of split points based on the input configs. In this function,
    the number of effective layers considered is the weighted summation of
    `num_moe_layers` and `num_dense_layers` where the weights are set to 1.0 for
    MoE layers and 0.5 for dense layers. We try to distribute all effective layers
    evenly onto the pipeline stages.

    The input modules (e.g., embedding tables, vision encoder) must always be
    assigned to stage 0 and the output modules (e.g., softmax) must always be
    assigned to stage `num_stages-1`.

    If schedule_str is a single-stage schedule, num_layers_per_stage must
    be set to None. We ensure that each pipeline rank has exactly 1 stage.
    If scheduler_str a multi-stage schedule, num_layers_per_stage must be
    set to a value >= 1.

    Args:
        schedule_str (str): The string of the schedule name.
        pp_size (int): Number of ranks assigned to the pipeline parallel dimension.
        num_layers_per_stage (int): The number of layers per (virtual) pipeline stage. Set this
            to >=1 only for multi-stage schedules. For single-stage schedules, the
            num_layers_per_stage is set internally to num_effective_layers // pp_size.
        num_moe_layers (int): The number of MoE layers in the model.
        num_dense_layers (int): The number of dense layers in the model.

    Returns:
        list[str]: A list of split point FQNs.
    """
    schedule_class = get_schedule_class(schedule_str)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    # If num_layers_per_stage is provided, we require a rigid fit of the effective layers
    if is_single_stage_schedule:
        assert (
            num_layers_per_stage is None
        ), f"Set num_layers_per_stage {num_layers_per_stage} to None for single stage schedules"
    else:
        assert (
            num_layers_per_stage is not None
        ), f"Set num_layers_per_stage {num_layers_per_stage} for multi stage schedules"

    weights = {
        "dense": 0.5,
        "moe": 1.0,
    }
    num_effective_layers = int(
        math.ceil(weights["dense"] * num_dense_layers + weights["moe"] * num_moe_layers)
    )

    if num_layers_per_stage is not None:
        # num_stages must be a multiple of pp_size.
        num_stages = (
            num_effective_layers + num_layers_per_stage - 1
        ) // num_layers_per_stage
        num_stages += (-num_stages) % pp_size
    else:
        num_layers_per_stage = (num_effective_layers + pp_size - 1) // pp_size
        num_stages = pp_size

    assert num_stages % pp_size == 0

    if num_stages > (num_dense_layers + num_moe_layers):
        raise ValueError(
            f"The number of stages {num_stages} is larger than the number of "
            f"layers (dense: {num_dense_layers}, moe {num_moe_layers}, effective {num_effective_layers}). "
            "This will cause at least one stage to have zero layers which is unsupported."
        )

    num_stages_per_rank = num_stages // pp_size
    if is_single_stage_schedule:
        assert (
            num_stages_per_rank == 1
        ), f"Number of stages per rank ({num_stages_per_rank}) must be 1 for single-stage schedules."
    else:
        assert (
            num_stages_per_rank >= 2
        ), f"Number of stages per rank ({num_stages_per_rank}) must be >= 2 for multi-stage schedules."

    layer_types = ["dense", "moe"]
    num_layers_left = {"dense": num_dense_layers, "moe": num_moe_layers}
    num_model_layers = num_dense_layers + num_moe_layers

    def _total_layers_left(input_dict: dict[str, int]):
        return sum([val for _, val in input_dict.items()])

    splits = []
    for stage_id in range(num_stages):
        total_layers_left = _total_layers_left(num_layers_left)
        splits.append("layers." + str(num_model_layers - total_layers_left))

        num_layers_left_this_stage = float(num_layers_per_stage)

        for layer_type in layer_types:
            if num_layers_left[layer_type] == 0:
                continue

            total_layers_left = _total_layers_left(num_layers_left)

            # If the number of layers left is smaller than the number of stages
            # to be filled, skip to the next stage. When this happens, at least
            # one layer should have been assigned to this stage, otherwise, this
            # is an internal error. We can only afford to assign one layer per
            # stage from here on to ensure that all stages will be filled.
            if total_layers_left < (num_stages - stage_id):
                break

            num_layers_this_stage = min(
                total_layers_left - num_stages + stage_id + 1,
                num_layers_left[layer_type],
                int(num_layers_left_this_stage / weights[layer_type]),
            )

            num_layers_left[layer_type] -= num_layers_this_stage
            num_layers_left_this_stage -= num_layers_this_stage * weights[layer_type]

    splits.append("layers." + str(num_model_layers))

    for layer_type in layer_types:
        assert num_layers_left[layer_type] == 0, (
            f"Layer type {layer_type} must have 0 layers left after placement, "
            f"found {num_layers_left[layer_type]}"
        )

    logger.info(
        f"GenerateSplitPoints with args, schedule: {schedule_str}, pp_size: {pp_size}, "
        f"num_model_layers: {num_model_layers}, "
        f"num_stages: {num_stages}, num_layers_per_stage: {num_layers_per_stage}, "
        f"num_effective_layers: {num_effective_layers}, "
    )

    logger.info(
        f"Here is the auto-generated split, which may be sub-optimal: {splits}."
    )
    return splits


def stage_ids_this_rank(
    pp_rank: int,
    pp_size: int,
    num_stages: int,
    schedule_str: str,
) -> tuple[int, ...]:
    """
    Computes the stage ids for the stages that will run on this pp rank
    for a looped schedule.
    """
    schedule_class = get_schedule_class(schedule_str)
    if schedule_class == ScheduleZBVZeroBubble:
        raise ValueError(f"Unsupported pipeline schedule {schedule_str}")

    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))


def build_pipeline_schedule(
    pp_mesh: DeviceMesh,
    batch_size: int,
    num_stages: int,
    schedule_str: str,
    microbatch_size: int,
    model_parts: list[nn.Module],
    device: torch.device,
    loss_fn: Callable[..., torch.Tensor] | None,
    has_backward: bool,
) -> _PipelineSchedule:
    """
    Builds a pipeline schedule for the given model configuration and pipeline
    parallel mesh. The train invokes `pp_schedule.step()` for executing a
    step instead of invoking the model's forward function. The loss and
    backward pass are all executed within `pp_schedule.step()`.

    Args:
        pp_mesh (DeviceMesh): The device mesh for pipeline parallelism.
        batch_size (int): The batch size for training summed across PP ranks.
        num_stages (int): The number of (virtual) pipeline stages.
        schedule_str (str): The string representation of the pipeline schedule.
        microbatch_size (int): The size of each microbatch. The batch is divided
            into smaller microbatches to reduce the number of pipeline bubbles.
        model_parts (list[nn.Module]): The list of model parts to be scheduled
            on this PP rank.
        device (torch.device): The device to use for the model parts.
        loss_fn (Callable): The loss function to use for training and validation.
            Set this to None for generation.
        has_backward (bool): Whether the pipeline schedule executes the backward
            pass as well.

    Returns:
        schedule (_PipelineSchedule): The pipeline schedule for the given stages.
    """
    schedule_class = get_schedule_class(schedule_str)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    stage_ids = stage_ids_this_rank(
        pp_rank=pp_mesh.get_local_rank(),
        pp_size=pp_mesh.size(),
        num_stages=num_stages,
        schedule_str=schedule_str,
    )

    stages = []
    for stage_idx, model_part in zip(stage_ids, model_parts):
        stage = PipelineStage(
            submodule=model_part,
            stage_index=stage_idx,
            num_stages=num_stages,
            device=device,
            group=pp_mesh.get_group(),
        )
        stages.append(stage)

    if is_single_stage_schedule and len(stages) > 1:
        raise ValueError(
            "Only one stage per rank is supported for single stage schedules "
            f"{schedule_str}, got {len(stages)} stages."
        )

    # Validate that the batch size is divisible by the microbatch_size
    # otherwise we'll hang or error during training.
    if batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by microbatch size {microbatch_size}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatch_size."
        )
    n_microbatches = batch_size // microbatch_size

    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = pp_mesh.size() * len(stages)
    if n_microbatches < num_total_stages and is_single_stage_schedule:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in significant pipeline bubbles."
        )

    schedule = schedule_class(
        stages[0] if is_single_stage_schedule else stages,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        has_backward=has_backward,
    )

    return schedule
