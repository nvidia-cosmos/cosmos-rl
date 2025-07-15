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

import inspect
import types
import torch

from typing import Any, Tuple, Callable
from functools import partial

from torch.distributed.device_mesh import DeviceMesh
import torch.distributed as dist
import torch.nn as nn

from cosmos_rl.utils.attn_util import repeat_kv
from cosmos_rl.utils.parallelism import ParallelDims


def all_to_all_tensor(
    local_input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    cp_mesh: DeviceMesh,
    async_op: bool = False,
) -> torch.Tensor:
    """
    All‐to‐all via all_to_all_single.
    Splits `local_input` along scatter_dim into cp_world_size pieces,
    exchanges them, then concatenates received pieces along gather_dim.
    Returns a tensor or, if async_op=True, returns a callable that waits and returns.
    """
    group = cp_mesh.get_group()
    world_size = cp_mesh.size()
    # 1) Split into per‐rank chunks and record each chunk's shape & numel
    splits = torch.tensor_split(local_input, world_size, dim=scatter_dim)
    split_shapes = [s.shape for s in splits]
    split_nelems = [s.numel() for s in splits]

    # 2) Flatten & concat all send‐chunks into one 1‐D tensor
    flat_input = torch.cat([s.contiguous().view(-1) for s in splits], dim=0)
    flat_output = torch.empty_like(flat_input)

    # 3) Exchange
    work = dist.all_to_all_single(
        flat_output,  # output
        flat_input,  # input
        output_split_sizes=split_nelems,
        input_split_sizes=split_nelems,
        group=group,
        async_op=async_op,
    )

    # helper to rebuild & cat along gather_dim
    def _build_result():
        out_chunks = []
        offset = 0
        for shape, ne in zip(split_shapes, split_nelems):
            chunk = flat_output[offset : offset + ne].view(shape)
            out_chunks.append(chunk)
            offset += ne
        return torch.cat(out_chunks, dim=gather_dim).contiguous()

    if async_op:

        def wait_and_get():
            work.wait()
            return _build_result()

        return wait_and_get

    # synchronous
    return _build_result()


def all_gather_tensor(
    local_tensor: torch.Tensor, cp_mesh: DeviceMesh, async_op: bool = False
) -> torch.Tensor:
    """
    This function performs an all-gather communication operation on a tensor.
    It splits the input tensor into `cp_world_size` parts along the specified scatter dimension,
    and then performs an all-to-all communication operation on these parts.
    The output is a tensor of the same shape as the input, but with the specified gather dimension
    concatenated.

    Args:
        local_tensor (torch.Tensor): The input tensor to be gathered.
        cp_mesh (DeviceMesh): The device mesh to use for the all-gather communication.
        async_op (bool, optional): Whether to perform the operation asynchronously. Defaults to False.

    Returns:
        torch.Tensor: The output tensor of the same shape as the input, but with the specified gather dimension concatenated.
    """
    group = cp_mesh.get_group()
    cp_world_size = cp_mesh.size()
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * cp_world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    comm = dist.all_gather_into_tensor(
        output, local_tensor, group=group, async_op=async_op
    )
    if async_op:

        def wait():
            comm.wait()
            return output

        return wait
    return output


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        cp_mesh: DeviceMesh,
        local_tensor: torch.Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> torch.Tensor:
        # Record the autograd context.
        ctx.cp_mesh = cp_mesh
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        cp_world_size = cp_mesh.size()
        ctx.cp_world_size = cp_world_size

        cp_rank = cp_mesh.get_local_rank()
        ctx.cp_rank = cp_rank

        local_shape = local_tensor.shape
        split_size = local_shape[0]
        ctx.cp_part_size = local_shape[gather_dim]

        output = all_gather_tensor(local_tensor, cp_mesh, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.cp_world_size
        return (
            None,
            grad_output.split(ctx.cp_part_size, dim=ctx.gather_dim)[
                ctx.cp_rank
            ].contiguous(),
            None,
            None,
            None,
            None,
        )


class SeqAllToAll(torch.autograd.Function):
    """
    This class implements the all-to-all communication operation for a sequence of tensors.
    """

    @staticmethod
    def forward(
        ctx: Any,
        cp_mesh: DeviceMesh,
        local_input: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> torch.Tensor:
        ctx.cp_mesh = cp_mesh
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(
            local_input, scatter_dim, gather_dim, cp_mesh, async_op
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None, None]:
        input_t = (
            torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
            if ctx.async_op
            else grad_output[0]
        )
        return (
            None,
            all_to_all_tensor(
                input_t, ctx.gather_dim, ctx.scatter_dim, ctx.cp_mesh, False
            ),
            None,
            None,
            None,
            None,
        )


def slice_input_tensor(
    data: torch.Tensor, dim: int, cp_mesh: DeviceMesh
) -> torch.Tensor:
    # We must use local rank, not get_rank()
    cp_world_size, cp_rank = cp_mesh.size(), cp_mesh.get_local_rank()
    partial_size = data.size(dim) // cp_world_size
    slc = [slice(None)] * len(data.shape)
    slc[dim] = slice(cp_rank * partial_size, (cp_rank + 1) * partial_size)
    return data[slc].contiguous()


def slice_inputs_for_ulysses(
    input_tensors: list[torch.Tensor | None], cp_mesh: DeviceMesh
) -> list[torch.Tensor]:
    """
    The input tensors are already padded by cosmos-rl datapacker.
    Args:
        input_tensors: Input tensors with shape of [bsz, seqlen]
        cp_mesh: ulysses sequence parallelism size

    Returns:
        list[torch.Tensor]: Input tensors for current CP rank.
        torch.Tensor: position_ids for current rank
    """
    return [
        slice_input_tensor(t, dim=1, cp_mesh=cp_mesh) if t is not None else None
        for t in input_tensors
    ]


def gather_outputs_for_ulysses(
    output: torch.Tensor,
    gather_dim: int,
    cp_mesh: DeviceMesh,
    grad_scaler: bool = True,
) -> torch.Tensor:
    return Gather.apply(cp_mesh, output, gather_dim, grad_scaler)


def gather_seq_scatter_heads(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_mesh: DeviceMesh,
) -> torch.Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel.
    gather sequence dimension and scatter head dim:
    For example, when seq_dim is 1, head_dim is 2, the transformation is:
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    Args:
        x: shape of [bsz, seq/n, h, ...]
        seq_dim: the dimension to gather
        head_dim: the dimension to scatter
        cp_mesh: ulysses sequence parallelism size
    Returns:
        torch.Tensor: shape of gathered and scattered tensor
    """
    x = SeqAllToAll.apply(cp_mesh, x, head_dim, seq_dim)
    return x


def gather_heads_scatter_seq(
    x: torch.Tensor, head_dim: int, seq_dim: int, cp_mesh: DeviceMesh
) -> torch.Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel.
    gather head dimension and scatter seq dim:
    For example, when seq_dim is 1, head_dim is 2, the transformation is:
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]

    Args:
        x (torch.Tensor): shape of [bsz, seq, h/n, ...]
        head_dim (int): the dimension to gather
        seq_dim (int): the dimension to scatter
        cp_mesh (DeviceMesh): ulysses sequence parallelism size

    Returns:
        torch.Tensor: shape of [bsz, seq/n, h, ...]
    """
    return SeqAllToAll.apply(cp_mesh, x, seq_dim, head_dim, False)


def ulysses_wrapper_of_attn_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cp_mesh: DeviceMesh,
    original_attn_func: Callable,
    *args,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): [batch_size, seqlen/cp_size, nheads, head_dim]
        key_states (torch.Tensor): [batch_size, seqlen/cp_size, nheads_k, head_dim]
        value_states (torch.Tensor): [batch_size, seqlen/cp_size, nheads_k, head_dim]
        cp_mesh (DeviceMesh): ulysses sequence parallelism device mesh
        original_attn_func: the original attention function

    Returns:
        torch.Tensor: [batch_size, seqlen/cp_size, nheads, head_dim]
    """
    cp_world_size = cp_mesh.size()
    assert cp_world_size > 1, "CP world size must be greater than 1"

    # AlltoAll for Ulysses
    # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
    # we choose to repeat cp_size//nheads_k, since flash_attention supports MQA/GQA.
    # For example:
    # - nheads_k=4, sp=8, repeats=2
    # - nheads_k=8, sp=8, repeats=1
    # - nheads_k=16, sp=8, repeats=1
    repeats = max(cp_world_size // key_states.size(2), 1)
    key_states = repeat_kv(key_states, repeats)
    value_states = repeat_kv(value_states, repeats)

    # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
    query_states = gather_seq_scatter_heads(
        query_states, seq_dim=1, head_dim=2, cp_mesh=cp_mesh
    )
    key_states = gather_seq_scatter_heads(
        key_states, seq_dim=1, head_dim=2, cp_mesh=cp_mesh
    )
    value_states = gather_seq_scatter_heads(
        value_states, seq_dim=1, head_dim=2, cp_mesh=cp_mesh
    )

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = original_attn_func(
        query_states, key_states, value_states, *args, **kwargs
    )

    # AlltoAll for Ulysses
    # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
    attn_output = gather_heads_scatter_seq(
        attn_output, seq_dim=1, head_dim=2, cp_mesh=cp_mesh
    )

    return attn_output


def ulysses_attn_func(
    original_attn_func: Callable,
    cp_mesh: DeviceMesh,
):
    return partial(
        ulysses_wrapper_of_attn_func,
        original_attn_func=original_attn_func,
        cp_mesh=cp_mesh,
    )


def swizzle_cp_forward(model: nn.Module, parallel_dims: ParallelDims):
    cp_mesh = parallel_dims.mesh["cp"]
    if parallel_dims.pp_enabled:
        # because `_swizzle_pp_grpo_forward` is replaced as the `forward` of model,
        # So hook wo't work in pp. We do a cp replacement, too.
        # substitute the original forward of model of last stage
        if parallel_dims.pp_coord[0] == parallel_dims.pp_coord[1] - 1:  # last stage
            origin_forward = model.forward
            full_signature = list(inspect.signature(origin_forward).parameters.keys())

            def gather_output_forward(*args, **kwargs):
                args = args[1:]
                n_args = len(args)
                if n_args > 0:
                    signature = full_signature[:n_args]
                    for key in signature:
                        if key in kwargs:
                            kwargs.pop(key)
                outputs = origin_forward(*args, **kwargs)
                return gather_outputs_for_ulysses(
                    outputs, gather_dim=1, cp_mesh=cp_mesh
                )

            model.forward = types.MethodType(gather_output_forward, model)
    else:
        # non-pp case, just use hook is perfect.
        def gather_output_hook(model, args, output):
            return gather_outputs_for_ulysses(output, gather_dim=1, cp_mesh=cp_mesh)

        model.register_forward_hook(gather_output_hook)
