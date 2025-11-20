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

# Standard library imports
import math
import os
import time
import threading
from collections import defaultdict
from queue import Queue, Empty
from datetime import timedelta
from typing import Dict, Iterable, Optional, Union, Callable
from functools import partial
from dataclasses import dataclass

# Third party imports
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_module, Placement
from torch.distributed.tensor.parallel import ParallelStyle, ColwiseParallel

from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    Shard,
)
from torch.distributed.tensor.placement_types import UnevenShard
from torch.distributed.tensor._collective_utils import (
    fill_empty_tensor_to_shards,
    pad_tensor,
)

# Local imports
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils import constant, network_util
from cosmos_rl.dispatcher.command import Command, BuildMeshCommand
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils.pynccl import (
    get_nccl_timeout_ms,
    nccl_timeout_watchdog,
    create_nccl_comm,
    create_nccl_uid,
    nccl_abort,
    get_nccl_comm_nranks,
    nccl_broadcast,
    nccl_send,
    nccl_recv,
    nccl_allreduce,
)
from cosmos_rl.utils.constant import COSMOS_GLOO_TIMEOUT


def init_distributed(cpu_enabled: bool = True):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        return
    elif torch.distributed.is_initialized():
        return
    else:
        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            timeout=timedelta(seconds=COSMOS_GLOO_TIMEOUT),
        )


def destroy_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class UnevenShardLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, DTensor):
            weight = self.weight
            logger.info(
                f"UnevenShardLinear: Forwarding with weight shape {weight.shape} and input shape {weight.__create_chunk_list__()}"
            )
        else:
            logger.info(f"Expected weight to be a DTensor, but got {type(self.weight)}")
        
        logger.info(f"UnevenShardLinear: Forwarding with {input.shape} {isinstance(input, DTensor) and input.__create_chunk_list__() or 'N/A'}")
        # if isinstance(weight, DTensor) and weight.to_local().numel() == 0:
        #     logger.info(f"Input tensor is empty after to_local(), returning empty tensor")
        #     # return  torch.distributed.tensor.DTensor.from_local(
        #     #             torch.empty(
        #     #                 input.size(0), input.size(1), 0, device=input.device, dtype=input.dtype
        #     #             ),
        #     #             weight.device_mesh,
        #     #             [UnevenShard(2, weight.placements[0].min_chunk)],
        #     #             run_check=False,
        #     #         )
        #     time.sleep(120)
        #     return torch.empty(0, self.out_features, device=input.device, dtype=input.dtype)
        ret = torch.nn.functional.linear(input, weight, self.bias)
        logger.info(f"UnevenShardLinear: Forwarding with output shape {ret.shape} {isinstance(ret, DTensor) and ret.__create_chunk_list__() or 'N/A'}")
        return ret


# @dataclass(frozen=True)
# class UnevenShard(Shard):
#     min_chunk: int

#     def __eq__(self, other: object) -> bool:
#         if isinstance(other, UnevenShard):
#             return self.dim == other.dim and self.min_chunk == other.min_chunk
#         return False

#     def __hash__(self) -> int:
#         return hash((self.dim, self.min_chunk))

#     def __repr__(self) -> str:
#         """
#         machine readable representation of the UnevenShard placement
#         """
#         return f"(dim={self.dim}, min_chunk={self.min_chunk})"

#     def __str__(self) -> str:
#         """human readable representation of the UnevenShard placement"""
#         return f"_S({self.dim}, min_chunk={self.min_chunk})"

#     def _split_tensor(
#         self,
#         tensor: torch.Tensor,
#         num_chunks: int,
#         *,
#         with_padding: bool = True,
#         contiguous: bool = True,
#     ) -> tuple[list[torch.Tensor], list[int]]:
#         if tensor.shape[self.dim] % (self.min_chunk * num_chunks) == 0:
#             return super()._split_tensor(
#                 tensor,
#                 num_chunks,
#                 with_padding=with_padding,
#                 contiguous=contiguous,
#             )
#         assert (
#             self.dim <= tensor.ndim
#         ), f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
#         # chunk tensor over dimension `dim` into n slices
#         chunk_sizes = []
#         size = tensor.shape[self.dim]
#         for i in range(num_chunks):
#             chunk_size = min(self.min_chunk, size)
#             if chunk_size > 0:
#                 size -= chunk_size
#                 chunk_sizes.append(chunk_size)
#         while size > 0:
#             for i in range(num_chunks):
#                 chunk_size = min(self.min_chunk, size)
#                 if chunk_size > 0:
#                     size -= chunk_size
#                     chunk_sizes[i] += chunk_size

#         tensor_list = list(torch.split(tensor, chunk_sizes, dim=self.dim))

#         tensor_list = fill_empty_tensor_to_shards(
#             tensor_list, self.dim, num_chunks - len(tensor_list)
#         )

#         # compute the chunk size inline with ``torch.chunk`` to calculate padding
#         full_chunk_size = (tensor.size(self.dim) + num_chunks - 1) // num_chunks

#         shard_list: list[torch.Tensor] = []
#         pad_sizes: list[int] = []
#         for shard in tensor_list:
#             if with_padding:
#                 pad_size = full_chunk_size - shard.size(self.dim)
#                 shard = pad_tensor(shard, self.dim, pad_size)
#                 pad_sizes.append(pad_size)
#             if contiguous:
#                 shard = shard.contiguous()
#             shard_list.append(shard)
#         return shard_list, pad_sizes

#     def _local_shard_size_and_offset(
#         self,
#         curr_local_size: int,
#         num_chunks: int,
#         rank: int,
#     ) -> tuple[int, int]:
#         """
#         Given the size of the current local tensor (which may already be sharded on some dimensions),
#         computes the new local shard size and offset given the desired number of chunks
#         (num_chunks is generally equal to the size of the current sharding dim).

#         Note: new local shard offset is relative to the current sharded tensor, not the global tensor.
#         See `_utils.compute_local_shape_and_global_offset` for computing global offset.

#         Returns (new local shard size, offset)

#         """
#         # Compute the chunk size inline with ``torch.chunk``
#         chunk_sizes = []
#         size = curr_local_size
#         for i in range(num_chunks):
#             chunk_size = min(self.min_chunk, size)
#             if chunk_size > 0:
#                 size -= chunk_size
#                 chunk_sizes.append(chunk_size)
#         while size > 0:
#             for i in range(num_chunks):
#                 chunk_size = min(self.min_chunk, size)
#                 if chunk_size > 0:
#                     size -= chunk_size
#                     chunk_sizes[i] += chunk_size
        
#         for _ in range(num_chunks - len(chunk_sizes)):
#             chunk_sizes.append(0)

#         # logger.info(f"UnevenShard: local shard size {curr_local_size}, num_chunks {num_chunks}, chunk_sizes {chunk_sizes} {rank}")
#         return chunk_sizes[rank], sum(chunk_sizes[:rank])


class UnevenColwiseParallel(ColwiseParallel):
    def __init__(
        self,
        *,
        min_chunk: int,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=UnevenShard(-1, min_chunk),
            use_local_output=use_local_output,
        )
        self.min_chunk = min_chunk
        
    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        # back to local tensor
        # logger.info(
        #     f"Uneven: Forwarding with output shape {outputs.shape} {isinstance(outputs, DTensor) and outputs.__create_chunk_list__() or 'N/A'} {outputs._local_tensor.shape if isinstance(outputs, DTensor) else 'N/A'}"
        # )
        if outputs.placements != output_layouts:
            from cosmos_rl.utils.logging import logger
            logger.info(f"==== ColwiseParallel output layout {outputs.placements} is different from the desired output layout {output_layouts}, redistributing...")
            outputs = outputs.redistribute(placements=[Replicate()] * len(output_layouts), async_op=True).redistribute(placements=output_layouts, async_op=True)
        if not isinstance(outputs, DTensor):
            return outputs
        return outputs.to_local() if use_local_output else outputs

    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        for name, param in module.named_parameters():
            dist_param = torch.nn.Parameter(
                distribute_tensor(
                    param,
                    device_mesh,
                    [UnevenShard(0, self.min_chunk)],
                    src_data_rank=self.src_data_rank,
                )
            )
            logger.info(
                f"Registered uneven shard parameter {name} with original shape {dist_param.__create_chunk_list__()}"
            )
            module.register_parameter(name, dist_param)


def parse_modules_for_parallelism(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
) -> torch.nn.Module:
    upper_modules = {}
    for name, plan in parallelize_plan.items():
        if isinstance(plan, ColwiseParallel):
            upper_module_name = name.rsplit(".", 1)[0]
            if upper_module_name not in upper_modules:
                upper_modules[upper_module_name] = None

    for name, sub_module in module.named_modules():
        if name in upper_modules:
            upper_modules[name] = sub_module
        if parallelize_plan is not None:
            if isinstance(parallelize_plan, dict):
                if name in parallelize_plan and isinstance(
                    parallelize_plan[name], ColwiseParallel
                ):
                    upper_module_name = name.rsplit(".", 1)[0]
                    upper_module = upper_modules[upper_module_name]
                    if hasattr(upper_module, "head_dim"):
                        head_dim = upper_module.head_dim
                    else:
                        raise ValueError(
                            f"Expected upper module {upper_module_name} to have attribute 'head_dim'"
                        )
                    assert isinstance(
                        sub_module, torch.nn.Linear
                    ), f"Expected module {name} to be torch.nn.Linear"
                    head = sub_module.out_features // head_dim
                    if head < device_mesh.size():
                        updated_module = UnevenShardLinear(
                            sub_module.in_features,
                            sub_module.out_features,
                            bias=sub_module.bias is not None,
                        )
                        if "." in name:
                            uppper_path = name.rsplit(".", 1)[0]
                            leaf_name = name.rsplit(".", 1)[1]
                            uppper_module = module
                            for part in uppper_path.split("."):
                                uppper_module = getattr(uppper_module, part)
                        else:
                            uppper_module = module
                            leaf_name = name
                        # Replace the module
                        setattr(uppper_module, leaf_name, updated_module)
                        
                        def input_hook(module, input):
                            # logger.info(
                            #     f"UnevenShardLinear: Forwarding with weight shape {module.weight.shape} and input shape {input[0].shape} {isinstance(module.weight, DTensor) and module.weight.__create_chunk_list__() or 'N/A'}"
                            #     f"{isinstance(input[0], DTensor) and input[0].__create_chunk_list__() or 'N/A'}"
                            # )
                            if module.weight.numel() == 0:
                                return None
                            else:
                                return input
                        
                        def output_hook(module, input, output):
                            # logger.info(
                            #     f"UnevenShardLinear: Forwarding with weight shape {module.weight.shape} and output shape {output.shape} {isinstance(module.weight, DTensor) and module.weight.__create_chunk_list__() or 'N/A'}"
                            #     f"{isinstance(output, DTensor) and output.__create_chunk_list__() or 'N/A'}"
                            # )
                            return output
                            
                        sub_module.register_forward_pre_hook(input_hook)
                        sub_module.register_forward_hook(output_hook)
                        



def parse_modules_after_parallelism(
    model: torch.nn.Module,
    module: torch.nn.Module,
    module_name: str,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
) -> torch.nn.Module:
    upper_modules = {}
    for name, plan in parallelize_plan.items():
        if isinstance(plan, ColwiseParallel):
            upper_module_name = name.rsplit(".", 1)[0]
            if upper_module_name not in upper_modules:
                upper_modules[upper_module_name] = None

    for name, sub_module in module.named_modules():
        if name in upper_modules:
            upper_modules[name] = sub_module
        if parallelize_plan is not None:
            if isinstance(parallelize_plan, dict):
                if name in parallelize_plan and isinstance(
                    parallelize_plan[name], ColwiseParallel
                ):
                    upper_module_name = name.rsplit(".", 1)[0]
                    upper_module = upper_modules[upper_module_name]
                    if hasattr(upper_module, "head_dim"):
                        head_dim = upper_module.head_dim
                    else:
                        raise ValueError(
                            f"Expected upper module {upper_module_name} to have attribute 'head_dim'"
                        )
                    assert isinstance(
                        sub_module, torch.nn.Linear
                    ), f"Expected module {name} to be torch.nn.Linear"
                    head = sub_module.out_features // head_dim
                    if head < device_mesh.size():
                        for n, param in sub_module.named_parameters():
                            original_shape = param.shape
                            # param = param.view(
                            #     -1,
                            #     head_dim,
                            #     sub_module.in_features,
                            # )
                            param = param.redistribute(
                                device_mesh,
                                [UnevenShard(0, head)],
                            )
                            dist_param = torch.nn.Parameter(
                                param,
                            )
                            logger.info(
                                f"Registered uneven shard parameter {n} with original shape {dist_param.__create_chunk_list__()}"
                            )
                            # model.add_uneven_shard_params(".".join([module_name, name, n]) , original_shape, [[0, 1]])
                            sub_module.register_parameter(n, dist_param)


@torch.no_grad()
def gradient_reduce_across_dp_replicas_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    comm: "HighAvailabilitylNccl",
):
    """
    Reduce a tensor across data parallel replicas.
    TODO, we need make sure this function is atomic.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will reduce gradients.
        comm_idx (int): The nccl communicator id for the reduction.
    """

    grads = [p.grad for p in parameters if p.grad is not None]

    # We only need to reduce DTensor's local grad, this is to avoid tensor.grad == nullptr
    for i, g in enumerate(grads):
        if isinstance(g, DTensor):
            grads[i] = g.to_local()

    # create bucket for all grads, we can allreduce them in one go
    # NOTE: why we don't set DTensor as bucket view?
    # This is becuase we can't be sure that the training framework
    # never release grad, or clean grad by set None.
    # Create temporary bucket is a more reliable solution.
    buckets: dict[torch.dtype, list[torch.Tensor]] = {}
    for g in grads:
        if g.dtype not in buckets:
            buckets[g.dtype] = []
        buckets[g.dtype].append(g.view(-1))

    # move all grad into one bucket
    comm.wait_comm_ready()

    for bucket in buckets.values():
        BUCKET_SIZE = 200 * 1024 * 1024
        sub_buckets = []
        current_bucket = []
        current_size = 0
        for tensor in bucket:
            n_bytes = tensor.numel() * tensor.element_size()
            if current_size + n_bytes > BUCKET_SIZE:
                current_bucket.append(tensor)
                sub_buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
                continue
            current_bucket.append(tensor)
            current_size += n_bytes
        if current_size > 0:
            sub_buckets.append(current_bucket)
        del current_bucket
        del current_size

        for sub_bucket in sub_buckets:
            tmp_buffer = torch.cat(sub_bucket, dim=0).contiguous()
            # Convert to float32 to keep precision
            original_dtype = tmp_buffer.dtype
            original_device = tmp_buffer.device
            tmp_buffer = tmp_buffer.float()

            # Make sure the buffer is on GPU before calling nccl allreduce
            if tmp_buffer.device == torch.device("cpu"):
                tmp_buffer = tmp_buffer.cuda()

            # TODO a risk here, when comm is rebuilt, the reduce result will be wrong.
            # For the first time to build mesh, we set a longer timeout (30 minutes) to avoid lost some slower replicas
            timeout_ms = get_nccl_timeout_ms()
            if gradient_reduce_across_dp_replicas_.first_invoke:
                timeout_ms = 30 * 60 * 1000
                gradient_reduce_across_dp_replicas_.first_invoke = False

            comm.allreduce(
                tmp_buffer, tmp_buffer, dist.ReduceOp.AVG, timeout_ms=timeout_ms
            )
            tmp_buffer = tmp_buffer.to(original_dtype).to(original_device)

            # copy the result back to original grad
            offset = 0
            for g in sub_bucket:
                size = g.numel()
                g.copy_(tmp_buffer[offset : offset + size].view_as(g))
                offset += size
                assert (
                    offset <= tmp_buffer.numel()
                ), "offset should be equal to total size"


gradient_reduce_across_dp_replicas_.first_invoke = True


@torch.no_grad()
def gradient_norm_clipping(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[DeviceMesh] = None,
    return_norm_only: bool = False,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    # Group the parameters by their device meshes.
    parameters_by_mesh = defaultdict(list)
    for param in parameters:
        if param.grad is not None:
            # If one parameter belongs to multiple meshes, use a flattened mesh name
            # by concatenating all the mesh names together.
            if hasattr(param, "device_mesh"):
                device_mesh_str = "-".join(list(param.device_mesh.mesh_dim_names))
            else:
                device_mesh_str = "default"
            parameters_by_mesh[device_mesh_str].append(param)
    # Compute the norm for each mesh group
    per_mesh_norm_list = []
    for mesh, params in parameters_by_mesh.items():
        grads = [p.grad for p in params if p.grad is not None]
        mesh_norm = (
            torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
            if len(grads) > 0
            else torch.tensor(0.0).to(torch.cuda.current_device()).float()
        )
        # If mesh_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced mesh_norm tensor whose .item() would return the correct value
        if isinstance(mesh_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, mesh_norm will be a local tensor.

            # Remove FT replicate dimension if it exists.
            mesh_norm = mesh_norm.full_tensor()
        # Make the norm to be a 1D tensor so we can call cat() later.
        if mesh_norm.ndim == 0:
            mesh_norm = mesh_norm.reshape(1)
        per_mesh_norm_list.append(mesh_norm)

    # Compute the total norm among all meshes.
    if len(per_mesh_norm_list) > 1:
        per_mesh_norm_tensor = torch.cat(per_mesh_norm_list)
        if math.isinf(norm_type):
            total_norm = torch.max(per_mesh_norm_tensor)
        else:
            per_mesh_norm_tensor **= norm_type
            total_norm = torch.sum(per_mesh_norm_tensor)
            total_norm **= 1.0 / norm_type
    else:
        assert per_mesh_norm_list[0].numel() == 1, "total_norm should be a scalar"
        total_norm = per_mesh_norm_list[0].view(-1)[0]

    # Reduce the norm among the PP ranks.
    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    if not return_norm_only:
        # Perform clipping on each mesh group
        for mesh, params in parameters_by_mesh.items():
            torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach)
    return total_norm


def _dist_reduce(x: torch.Tensor, reduceOp: str, mesh: DeviceMesh) -> float:
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()
    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh)


def dist_mean(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh)


class ReplicateParallel(ParallelStyle):
    def __init__(
        self, *, use_local_output: bool = True, input_layout: Optional[Placement] = None
    ):
        super().__init__()
        self.use_local_output = use_local_output
        self.input_layout = input_layout or Replicate()

    def _replicate_module_fn(
        self, name: str, module: torch.nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(input_layout, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            return DTensor.from_local(
                input_tensor, device_mesh, [input_layout], run_check=False
            )
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, tuple):
            return tuple([o.to_local() if use_local_output else o for o in outputs])
        else:
            return outputs.to_local() if use_local_output else outputs

    def _apply(
        self, module: torch.nn.Module, device_mesh: DeviceMesh
    ) -> torch.nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.input_layout),
            partial(self._prepare_output_fn, self.use_local_output),
        )


def broadcast_object_cpu(
    obj, src=0, device=torch.device("cpu"), group=None, group_src=None
):
    """
    Broadcast an object from the source process to all processes.
    The object is first converted to a list and then broadcasted.
    """
    self_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        return obj

    if group_src is None:
        obj_lst = [obj if self_rank == src else None]
    else:
        src = None  # src is ignored when group_src is specified
        obj_lst = [obj if group.rank() == group_src else None]
    dist.broadcast_object_list(
        obj_lst, src=src, device=device, group=group, group_src=group_src
    )
    return obj_lst[0]


def all_gather_object_cpu(obj, device=torch.device("cpu"), group=None):
    """
    Gather an object from all processes.
    The object is first converted to a list and then gathered.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        return [obj]

    obj_lst = [None for i in range(world_size)]
    dist.all_gather_object(obj_lst, obj, group=group)
    return obj_lst


class HighAvailabilitylNccl:
    DESTROY_CMD = "destroy"

    def __init__(self, replica_name: str, global_rank: int, api_client: APIClient):
        self.replica_name = replica_name
        self.global_rank = global_rank
        self.api_client = api_client
        # max retry times for nccl op after nccl comm is rebuilt
        self.max_retry = 3
        self.default_timeout_ms = get_nccl_timeout_ms()

        # The nccl group info
        self.comm_idx: int = -1
        self.nccl_comm_count_map = {}
        self.replica_name_to_rank: Dict[str, int] = {}

        # For background thread
        self.build_mesh_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.is_single_peer = threading.Event()
        self.is_single_peer.clear()
        self.is_comm_ready = threading.Event()
        self.is_comm_ready.clear()
        self.is_first_time_build_mesh = True
        self.cmd_queue = Queue()
        self.build_mesh_thread = threading.Thread(
            target=self.__run_background_thread,
            daemon=True,
            name=f"HA_NCCL-{self.replica_name}-#{self.global_rank}",
        )
        self.build_mesh_thread.start()

    def __get_mesh_unique_key(self, replica_name_to_rank: Dict[str, int]):
        return (
            "_".join(
                [
                    k
                    for k, _ in sorted(
                        replica_name_to_rank.items(), key=lambda item: item[1]
                    )
                ]
            )
            + "_"
            + str(self.global_rank)
        )

    def __log_prefix(self):
        if self.replica_name in self.replica_name_to_rank:
            return f"[HA_NCCL][global_rank {self.global_rank}, replica_rank {self.replica_name_to_rank[self.replica_name]}] {self.replica_name}"
        else:
            return f"[HA_NCCL][global_rank {self.global_rank}] {self.replica_name}"

    def __run_background_thread(self):
        # new thread will reset current device to 0, we fix it here.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        while not self.shutdown_event.is_set():
            try:
                # non-blocking get the command from the queue
                cmd = self.cmd_queue.get(timeout=1)
            except Empty:
                continue

            # lock the build_mesh_lock to avoid abort in-flight nccl comm
            with self.build_mesh_lock:
                # 1. destory nccl comm immediately when receive any command
                # need_abort = True if cmd == self.DESTROY_CMD else False
                self.__execute_destroy_nccl_comm(abort=True)

                # 2. build nccl comm if it is a buildmesh command
                if isinstance(cmd, BuildMeshCommand):
                    # first, destroy the nccl comm if it exists, then build the new nccl comm
                    self.__execute_build_mesh(cmd)

    def __execute_destroy_nccl_comm(self, abort: bool = False):
        self.is_comm_ready.clear()
        if self.comm_idx != -1:
            logger.info(f"{self.__log_prefix()} destroy nccl comm_idx: {self.comm_idx}")
            try:
                # most time, we don't need to abort the nccl comm, because the nccl comm is aborted by watchdog
                # but if `nccl_timeout_watchdog` not used, we need to abort the nccl comm manually
                if abort:
                    nccl_abort(self.comm_idx)
            except Exception as e:
                logger.error(f"{self.__log_prefix()} Failed in destroy nccl comm: {e}")
            finally:
                self.comm_idx = -1

    def __execute_build_mesh(self, cmd: BuildMeshCommand) -> bool:
        logger.debug(
            f"{self.__log_prefix()} build mesh with {cmd.replica_name_to_rank}"
        )

        if len(cmd.replica_name_to_rank) == 1:
            self.replica_name_to_rank = cmd.replica_name_to_rank
            assert self.replica_name in cmd.replica_name_to_rank
            self.is_single_peer.set()
            self.is_comm_ready.set()
            return

        # continue to build nccl comm
        assert self.replica_name in cmd.replica_name_to_rank
        rank = cmd.replica_name_to_rank[self.replica_name]
        nccl_group_id = None
        unique_pair_name = self.__get_mesh_unique_key(cmd.replica_name_to_rank)
        if rank == 0:
            # initialize nccl handle for building mesh among policies
            # only replica_rank == 0 have the right to generate nccl id.
            nccl_group_id = create_nccl_uid()
            self.api_client.post_nccl_comm_initiator(unique_pair_name, nccl_group_id)
            logger.debug(
                f"{self.__log_prefix()} post nccl group_id to controller: {unique_pair_name}"
            )
        else:
            # other replicas should query the nccl group id from controller
            # all ranks need to wait for the rollout replica 0 finished the group_id post
            # and then they can get the group_id from controller
            # But we don't have something like dist.barrier(), so just while True loop to query it like synchronize.
            # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
            nccl_group_id = self.api_client.post_nccl_comm_acceptor(unique_pair_name)

        # create nccl comm, any error will be reported to the controller
        try:
            self.comm_idx = create_nccl_comm(
                nccl_group_id, rank, len(cmd.replica_name_to_rank)
            )
            self.is_first_time_build_mesh = False
        except Exception as e:
            # report the error to the controller
            self.api_client.post_nccl_comm_error(self.replica_name, e)
            logger.error(
                f"{self.__log_prefix()} failed in create nccl comm , report to controller: {e}"
            )
            self.__execute_destroy_nccl_comm(abort=True)
            return

        # Also need delete old nccl handler
        self.replica_name_to_rank = cmd.replica_name_to_rank
        self.is_single_peer.clear()
        self.is_comm_ready.set()
        logger.debug(
            f"{self.__log_prefix()} created nccl_comm for replica_rank {rank} with total {len(cmd.replica_name_to_rank)} ranks."
        )

        # To prevent following rebuild mesh with same unique_pair_name,
        # we need to clear the kv store of the old mesh.
        if self.replica_name_to_rank.get(self.replica_name) == 0:
            self.api_client.post_clear_nccl_comm_store(unique_pair_name)

    def __do_nccl_op_with_retry(self, func: Callable, timeout_ms: int, **kwargs):
        if self.is_single_peer.is_set():
            # single peer, no need to do nccl op
            return

        for i in range(self.max_retry):
            try:
                self.wait_comm_ready()
                timeout_ms = (
                    timeout_ms if timeout_ms is not None else self.default_timeout_ms
                )
                with (
                    nccl_timeout_watchdog(wait_stream=True, timeout_ms=timeout_ms),
                    self.build_mesh_lock,
                ):
                    func(
                        comm_idx=self.comm_idx,
                        timeout_ms=timeout_ms,
                        **kwargs,
                    )

                # if success, break the loop
                break
            except Exception as e:
                # mark the communicator is not ready
                self.is_comm_ready.clear()

                # report the error to the controller
                # the communicator will destroy before buildmesh
                self.api_client.post_nccl_comm_error(self.replica_name, e)
                logger.error(
                    f"{self.__log_prefix()} recovering nccl op '{func.__name__}' with kwargs {kwargs} after {i} retries: {e}"
                )

    def destroy_nccl_comm(self):
        self.cmd_queue.put(self.DESTROY_CMD)

    def push_cmd(self, cmd: BuildMeshCommand):
        self.cmd_queue.put(cmd)

    def shutdown(self):
        self.shutdown_event.set()
        self.build_mesh_thread.join()

    def is_ready(self):
        """
        Check if the nccl comm is ready.
        This is non-blocking check, user should ensure the nccl op won't be skipped.
        """
        return self.is_comm_ready.is_set()

    def wait_comm_ready(self, timeout: float = 0):
        """
        Wait for the nccl comm to be ready.

        Args:
            timeout (float): The timeout in seconds.
        """
        start_time = time.time()

        if timeout == 0:
            while not self.is_comm_ready.is_set():
                time.sleep(0.1)
        else:
            done = self.is_comm_ready.wait(timeout=timeout)
            if not done:
                raise TimeoutError(
                    f"{self.__log_prefix()} wait for nccl comm ready timeout, current time: {time.time()}, start time: {start_time}, timeout: {timeout}"
                )

    def world_size(self):
        """
        Get the world size of the nccl comm.
        """
        if self.is_single_peer.is_set():
            return 1

        if not self.is_ready():
            raise RuntimeError(
                f"{self.__log_prefix()} nccl comm is not ready, please wait for the nccl comm to be ready"
            )

        try:
            # TODO(zjx): there will be a risk, if the nccl comm destroyed while get_nccl_comm_count,
            ws = get_nccl_comm_nranks(self.comm_idx)
        except Exception as e:
            ws = -1
            logger.warning(
                f"{self.__log_prefix()} failed in get nccl comm count: {e}, please try again after the nccl comm is ready"
            )

        return ws

    def get_replica_rank(self, replica_name: str):
        self.wait_comm_ready()
        return self.replica_name_to_rank[replica_name]

    def broadcast(self, tensor: torch.Tensor, src_replica: str, timeout_ms: int = None):
        src_rank = self.get_replica_rank(src_replica)
        self.__do_nccl_op_with_retry(
            func=nccl_broadcast,
            tensor=tensor,
            rank=src_rank,
            timeout_ms=timeout_ms,
        )

    def allreduce(
        self,
        sendbuff: torch.Tensor,
        recvbuff: torch.Tensor,
        op: dist.ReduceOp,
        timeout_ms: int = None,
    ):
        self.__do_nccl_op_with_retry(
            func=nccl_allreduce,
            sendbuff=sendbuff,
            recvbuff=recvbuff,
            op=op,
            timeout_ms=timeout_ms,
        )

    def send(self, tensor: torch.Tensor, dst_replica: str, timeout_ms: int = None):
        dst_rank = self.get_replica_rank(dst_replica)
        self.__do_nccl_op_with_retry(
            func=nccl_send,
            tensor=tensor,
            peer=dst_rank,
            timeout_ms=timeout_ms,
        )

    def recv(self, tensor: torch.Tensor, src_replica: str, timeout_ms: int = None):
        src_rank = self.get_replica_rank(src_replica)
        self.__do_nccl_op_with_retry(
            func=nccl_recv,
            tensor=tensor,
            peer=src_rank,
            timeout_ms=timeout_ms,
        )


class DistKVStore:
    def __init__(
        self,
        group: dist.ProcessGroup,
        master_rank: int,
        shutdown_event: threading.Event,
    ):
        self.group = group
        self.rank = self.group.rank()
        self.world_size = self.group.size()
        self.master_rank = master_rank if -1 < master_rank < self.world_size else 0
        self.counter = 0
        self.lock = threading.Lock()
        self.shutdown_event = shutdown_event
        self.local_store = None
        self.__init_local_store()

    def __init_local_store(self):
        if self.world_size == 1:
            return

        if self.rank == self.master_rank:
            local_ips = network_util.get_eth_ips()
            assert len(local_ips) > 0, "No IP addresses found"
            local_ip = local_ips[0]
            free_port = network_util.find_available_port(22000)

            max_retry = 300
            for _ in range(max_retry):
                try:
                    # Init TCPStore may fail when multi processes concurrently init TCPStore, so we need to retry to find another available port
                    self.local_store = dist.TCPStore(
                        host_name="0.0.0.0",
                        port=free_port,
                        # world_size=self.world_size,
                        is_master=True,
                        timeout=timedelta(seconds=constant.COSMOS_TCP_STORE_TIMEOUT),
                    )
                    break
                except Exception:
                    logger.error(
                        f"[DistKVStore] Failed to bind port {free_port}, try another"
                    )
                    time.sleep(1)
                    free_port = network_util.find_available_port(20000)

            logger.info(f"Local store started at {local_ip}:{free_port}")
            dist.broadcast_object_list(
                [local_ip, free_port],
                src=self.master_rank,
                device=torch.device("cpu"),
                group=self.group,
            )
        else:
            broadcast_object_list = [None, None]
            dist.broadcast_object_list(
                broadcast_object_list,
                src=self.master_rank,
                device=torch.device("cpu"),
                group=self.group,
            )
            local_ip, local_port = broadcast_object_list
            assert (
                local_ip is not None and local_port is not None
            ), "Failed to broadcast local store info"

            while True:
                try:
                    self.local_store = dist.TCPStore(
                        host_name=local_ip,
                        port=local_port,
                        is_master=False,
                        # world_size=self.world_size,
                        timeout=timedelta(seconds=constant.COSMOS_TCP_STORE_TIMEOUT),
                    )
                    break
                except Exception as e:
                    logger.error(f"Failed to connect to local store: {e}")
                    time.sleep(3)
                    continue

    def blocking_wait(self, keys: list[str]):
        assert self.world_size > 1, "Only master rank can wait for command"
        # retry every 10 seconds
        timeout = 10
        n_max_retries = max(1, int(constant.COSMOS_TCP_STORE_TIMEOUT / timeout))
        for _ in range(n_max_retries):
            try:
                self.local_store.wait(keys, timedelta(seconds=timeout))
                return
            except Exception as e:
                logger.debug(f"Failed to wait for kv store blocking wait: {e}")
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    raise RuntimeError("Stop signal received")
        raise RuntimeError("Failed to wait for kv store blocking wait")

    def broadcast_command(self, command: Command, src: int = 0) -> Command:
        """
        Broadcast a command to all ranks.
        """
        if self.world_size == 1:
            return command

        __key = f"#BROADCAST-{self.counter}"
        __key_dones = [f"{__key}-done-{i}" for i in range(self.world_size)]

        __last_key = f"#BROADCAST-{self.counter - 1}"
        __last_key_dones = [f"{__last_key}-done-{i}" for i in range(self.world_size)]

        error_raised = False
        cmd = None
        while self.shutdown_event is None or not self.shutdown_event.is_set():
            try:
                if src == self.rank:
                    self.local_store.set(__key, command.pack())
                else:
                    self.blocking_wait([__key])

                cmd_raw = self.local_store.get(__key)
                cmd = Command.depack(cmd_raw)

                self.local_store.set(__key_dones[self.rank], "1")
                self.blocking_wait(__key_dones)
            except Exception as e:
                if self.rank == src:
                    # Only log error when the rank is the source rank
                    # Else it is normal if there is no command to broadcast
                    logger.error(f"Failed to broadcast command: {e}")
                error_raised = True
                continue
            finally:
                if not error_raised:
                    if self.rank == src:
                        self.local_store.delete_key(__last_key)
                        for _d in __last_key_dones:
                            self.local_store.delete_key(_d)
                    self.counter += 1
                    break
        return cmd
