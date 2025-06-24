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

from contextlib import contextmanager
import torch
from typing import Any, Tuple, Callable, Dict
from functools import partial
import torch.distributed as dist

from torch.distributed.device_mesh import DeviceMesh

from cosmos_rl.utils.attn_util import repeat_kv
from cosmos_rl.utils.parallelism import ParallelDims


COSMOS_ATTN_FUNC = None


def all_to_all_tensor(
    local_input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    cp_mesh: DeviceMesh,
    async_op: bool = False,
) -> torch.Tensor:
    """
    This function performs an all-to-all communication operation on a tensor.
    It splits the input tensor into `cp_world_size` parts along the specified scatter dimension,
    and then performs an all-to-all communication operation on these parts.
    The output is a tensor of the same shape as the input, but with the specified gather dimension
    concatenated.

    Args:
        local_input (torch.Tensor): The input tensor to be scattered and gathered.
        scatter_dim (int): The dimension along which to scatter the input tensor.
        gather_dim (int): The dimension along which to gather the scattered tensor.
        cp_mesh (DeviceMesh): The device mesh to use for the all-to-all communication.
        async_op (bool, optional): Whether to perform the operation asynchronously. Defaults to False.

    Returns:
        torch.Tensor: The output tensor of the same shape as the input, but with the specified gather dimension concatenated.
    """
    group = cp_mesh.get_group()
    cp_world_size = cp_mesh.size()
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(local_input, cp_world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(cp_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait

    return torch.cat(output_list, dim=gather_dim).contiguous()


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
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    # FIXME: lms: should we wait for the async_op?
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
    dim_size = data.size(dim)
    partial_size = dim_size // cp_world_size
    slc = [slice(None)] * len(data.shape)
    slc[dim] = slice(cp_rank * partial_size, (cp_rank + 1) * partial_size)
    return data[slc].contiguous()


def slice_input_for_ulysses(
    input_ids: torch.Tensor, position_ids: torch.Tensor, cp_mesh: DeviceMesh
) -> Tuple[torch.Tensor, torch.Tensor]:
    """


    Note both input_ids_rmpad and position_ids_rmpad will be padded and sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
    """
    input_for_current_rank = slice_input_tensor(input_ids, dim=1, cp_mesh=cp_mesh)
    position_ids_for_current_rank = slice_input_tensor(
        position_ids, dim=1, cp_mesh=cp_mesh
    )
    return input_for_current_rank, position_ids_for_current_rank


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
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    x = SeqAllToAll.apply(cp_mesh, x, head_dim, seq_dim)
    return x


def gather_heads_scatter_seq(
    x: torch.Tensor, head_dim: int, seq_dim: int, cp_mesh: DeviceMesh
) -> torch.Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    return SeqAllToAll.apply(cp_mesh, x, seq_dim, head_dim, False)
    # return x


def ulysses_wrapper_of_flash_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cp_mesh: DeviceMesh,
    *args,
    original_attn_func: Callable,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    cp_world_size = cp_mesh.size()
    assert cp_world_size > 1, "CP world size must be greater than 1"

    ########## AlltoAll for Ulysses ##########
    # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
    # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
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

    # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
    # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
    # https://github.com/huggingface/transformers/pull/33932

    # (bsz, seq_len/n) -> (bsz, seq_len)
    # position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
    # torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
    # position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = original_attn_func(
        query_states, key_states, value_states, *args, **kwargs
    )

    ########## AlltoAll for Ulysses ##########
    # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
    attn_output = gather_heads_scatter_seq(
        attn_output, seq_dim=1, head_dim=2, cp_mesh=cp_mesh
    )

    return attn_output


def ulysses_attn_func(original_attn_func: Callable, cp_mesh: DeviceMesh):
    return partial(
        ulysses_wrapper_of_flash_attn,
        original_attn_func=original_attn_func,
        cp_mesh=cp_mesh,
    )


@contextmanager
def ulysses_cp_context(user_mini_batch: Dict[str, Any], parallel_dims: ParallelDims):
    cp_mesh = parallel_dims.mesh["cp"]
    dp_mesh = parallel_dims.mesh["dp"]

    cp_world_size = cp_mesh.size()
    if cp_world_size == 1:
        # do nothing if
        yield
    else:
        if dp_mesh.size() == 1:
            # do nothing if dp_world_size == 1
            yield
        else:
            # slice input_ids and position_ids for each dp
            input_ids_per_dp = user_mini_batch["input_ids"]  # [bs, seqlen]
            position_ids_per_dp = user_mini_batch["position_ids"]  # [bs, seqlen]

            try:
                # user_mini_batch["input_ids"] = input_ids
                # user_mini_batch["position_ids"] = position_ids
                yield
            finally:
                user_mini_batch["input_ids"] = input_ids_per_dp
                user_mini_batch["position_ids"] = position_ids_per_dp
