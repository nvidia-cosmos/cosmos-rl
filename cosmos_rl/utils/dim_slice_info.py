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

from typing import Dict, Any, Union, List, Tuple
from functools import reduce
from math import gcd
import torch


class DimSliceInfo:
    """
    A class to represent the slice information of a tensor along a specific dimension.
    This class contains the offset, total size, dimension name, and length of the slice.
    """

    offset: int
    total_size: int
    dim: str
    length: int = 1

    def __init__(self, offset: int, total_size: int, dim: str = "", length: int = 1):
        """
        Initialize the DimSliceInfo with the given offset, total size, dimension name, and length.
        """
        self.offset = offset
        self.total_size = total_size
        self.dim = dim
        self.length = length

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create a DimSliceInfo object from a dictionary.
        :param data: A dictionary containing the keys 'offset', 'total_size', 'dim', and 'length'.
        :return: A DimSliceInfo object.
        """
        return DimSliceInfo(
            offset=data["offset"],
            total_size=data["total_size"],
            dim=data.get("dim", ""),
            length=data.get("length", 1),
        )

    def simplify(self):
        common = reduce(gcd, [self.offset, self.total_size, self.length])  # noqa: E741
        return DimSliceInfo(
            offset=self.offset // common,
            total_size=self.total_size // common,
            dim=self.dim,
            length=self.length // common,
        )


def slice_tensor_with_strategy(
    tensor: torch.Tensor, idx: int, tensor_split_strategy: DimSliceInfo
):
    """
    Slices a tensor according to the given strategy at one dimension index.
    :param tensor: The tensor to be sliced.
    :param idx: The index of the dimension to slice.
    :param tensor_split_strategy: The strategy for slicing the tensor.
    :return: A sliced view of the tensor for the given dimension index.
    """

    view = tensor
    assert (
        view.shape[idx] % tensor_split_strategy.total_size == 0
    ), f"Tensor shape {view.shape} on dim {idx} must be divisible by {tensor_split_strategy.total_size}"
    start = (
        view.shape[idx]
        // tensor_split_strategy.total_size
        * tensor_split_strategy.offset
    )
    length = (
        view.shape[idx]
        // tensor_split_strategy.total_size
        * tensor_split_strategy.length
    )
    dim = view.dim()
    assert idx < view.dim(), f"Invalid index {idx} for {dim}D tensor."
    slices = (
        [slice(None, None)] * idx
        + [slice(start, start + length)]
        + [slice(None, None)] * (dim - idx - 1)
    )
    return view[slices]


def slice_tensor_with_strategies(
    self: torch.Tensor, strategys: Dict[int, Union[DimSliceInfo, Dict[str, Any]]]
) -> torch.Tensor:
    """
    Slices the tensor according to the given strategies at all dimension indices.
    :param tensor: The tensor to be sliced.
    :param strategys: A dictionary mapping dimension indices to DimSliceInfo objects.
    :return: The sliced tensor.
    """
    view = self
    for idx, split in strategys.items():
        idx = int(idx)
        if isinstance(split, dict):
            split = DimSliceInfo.from_dict(split)
        view = slice_tensor_with_strategy(view, idx, split)
    return view


torch.Tensor.cosmos_slice = slice_tensor_with_strategies


def get_unified_rank_info(
    a: DimSliceInfo, b: DimSliceInfo
) -> Tuple[DimSliceInfo, DimSliceInfo]:
    """
    Get the unified slice information with the same total size for two DimSliceInfo objects.
    :param a: The first DimSliceInfo object.
    :param b: The second DimSliceInfo object.
    :return: A tuple containing the unified slice information for both objects.
    """
    size = max(a.total_size, b.total_size)
    assert (
        size % a.total_size == 0 and size % b.total_size == 0
    ), "Sizes are not compatible for unification"
    scale_a = size // a.total_size
    scale_b = size // b.total_size
    scaled_a_size = a.total_size * scale_a
    scaled_b_size = b.total_size * scale_b
    scaled_a_rank = a.offset * scale_a
    scaled_b_rank = b.offset * scale_b
    unified_a = DimSliceInfo(scaled_a_rank, scaled_a_size, a.dim, a.length * scale_a)
    unified_b = DimSliceInfo(scaled_b_rank, scaled_b_size, b.dim, b.length * scale_b)
    return unified_a, unified_b


def rank_overlap(a: DimSliceInfo, b: DimSliceInfo) -> DimSliceInfo:
    """
    Check if the parts of two DimSliceInfo objects overlap.

    :param a: The first DimSliceInfo object.
    :param b: The second DimSliceInfo object.
    :return: A DimSliceInfo object representing the overlap, or None if there is no overlap.
    """
    a_new, b_new = get_unified_rank_info(a, b)
    assert a_new.total_size == b_new.total_size, "Sizes do not match after unification"

    left = max(a_new.offset, b_new.offset)
    right = min(
        a_new.offset + a_new.length,
        b_new.offset + b_new.length,
    )
    overlapped = None
    if left < right:
        overlapped = DimSliceInfo(left, a_new.total_size, a_new.dim, right - left)
    return overlapped


def relative_rank(smaller: DimSliceInfo, larger: DimSliceInfo) -> DimSliceInfo:
    """
    Get the relative slice information of two DimSliceInfo objects.
    :param smaller: The smaller DimSliceInfo object.
    :param larger: The larger DimSliceInfo object.
    :return: A DimSliceInfo object representing the relative slice of the smaller on in the larger one.
    """
    s, l = get_unified_rank_info(smaller, larger)  # noqa: E741
    assert s.offset >= l.offset, "Smaller rank is not less than or equal to larger rank"
    assert (
        s.offset + s.length <= l.offset + l.length
    ), "Smaller rank does not fit within larger rank"
    rank = s.offset - l.offset
    size = l.length
    length = s.length
    return DimSliceInfo(rank, size, s.dim, length)


def merge_rank(outter: DimSliceInfo, inner: DimSliceInfo) -> DimSliceInfo:
    """
    Merge two nested DimSliceInfo objects into one.
    :param outter: The DimSliceInfo object at a outter dimension.
    :param inner: The DimSliceInfo object at an inner dimension.
    :return: A DimSliceInfo object representing the merged slice information.
    """
    assert outter.length == 1 or outter.length == 0, "Outer rank length must be 1 or 0"
    size = outter.total_size * inner.total_size
    rank = outter.offset * inner.total_size + inner.offset
    length = inner.length * outter.length
    return DimSliceInfo(rank, size, outter.dim, length)


def tensor_overlap_info_at_dim(
    policy_rank: Dict[int, DimSliceInfo],
    rollout_rank: Dict[int, DimSliceInfo],
    dim: int,
) -> Tuple[DimSliceInfo, DimSliceInfo]:
    """
    Get the tensor overlap information at one dimension index.
    :param policy_rank: The sharded slice information for the given tensor from policy.
    :param rollout_rank: The sharded slice information for the given tensor from rollout.
    :param dim: The dimension index to check for overlap.
    :return: A tuple containing the overlap information between the given policy and rollout tensors.
    """
    if dim not in policy_rank:
        p = DimSliceInfo(0, 1)
    else:
        p = policy_rank[dim]
    if dim not in rollout_rank:
        r = DimSliceInfo(0, 1)
    else:
        r = rollout_rank[dim]

    p_new, r_new = get_unified_rank_info(p, r)
    overlap = rank_overlap(p_new, r_new)

    if overlap is None:
        return None, None
    overlap_r = relative_rank(overlap, r_new)
    overlap_p = relative_rank(overlap, p_new)
    return overlap_p.simplify(), overlap_r.simplify()


def extract_infomation_from_dtensor(
    param: torch.Tensor, name: str
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    """
    Extract the slice information from a DTensor parameter.
    :param param: The DTensor parameter to extract information from.
    :param name: The name of the parameter.
    :return: A dictionary mapping dimension indices to DimSliceInfo objects and a dictionary mapping mesh dimensions to their corresponding dimension indices.
    """
    dims_rank_info = {}
    dims_map = {}
    global_shape = tuple(param.shape)
    if isinstance(param, torch.distributed.tensor.DTensor):
        mesh = param.device_mesh
        placements = param.placements
        assert (
            len(placements) == len(mesh.mesh_dim_names)
        ), f"Number of placements {placements} does not match number of mesh dimensions {mesh}."
        for dim, placement in zip(mesh.mesh_dim_names, placements):
            if placement.is_shard():
                dims_map[dim] = placement.dim
            elif placement.is_replicate():
                pass
            else:
                raise ValueError(f"Unsupported placement type: {placement}")
        chunk_meta_list = param.__create_chunk_list__()
        local = param.to_local()
        assert (
            len(chunk_meta_list) == 1
        ), f"Expected only one chunk meta, but got {len(chunk_meta_list)} for {name}."
        meta = chunk_meta_list[0]
        assert (
            len(meta.offsets)
            == len(meta.sizes)
            == len(global_shape)
            == len(tuple(local.shape))
        ), f"Offsets {meta.offsets} and sizes {meta.sizes} must match global shape {global_shape} and local shape {tuple(local.shape)}."

        no_shard_dims_rank_info = {}
        for idx, g_size in enumerate(global_shape):
            offset = int(meta.offsets[idx])
            total_size = int(g_size)
            length = int(meta.sizes[idx])
            if total_size == length:
                assert (
                    offset == 0
                ), f"Expected rank 0 for full size dimension {idx}, but got {offset}."
                no_shard_dims_rank_info[idx] = DimSliceInfo(
                    offset=0,
                    total_size=total_size,
                    length=length,
                )
            else:
                dims_rank_info[idx] = DimSliceInfo(
                    offset=offset,
                    total_size=total_size,
                    length=length,
                ).__dict__

    return dims_rank_info, dims_map


def get_local_weight_shard_with_DTensor(
    target: torch.Tensor,
    dest_name: str,
    tensor: torch.Tensor,
    splited_dims: List[List[int]] = None,
    mesh_dims: List[str] = ["tp", "dp_shard"],
):
    """
    Get the local weight shard for the given tensor based on the provided DTensor information.
    :param target: The target tensor in DTensor format to copy into.
    :param dest_name: The name of the destination tensor.
    :param tensor: The input tensor to be sliced.
    :param splited_dims: The list of dimensions that are split from the same original dimension.
    :param mesh_dims: The list of mesh dimensions to consider for slicing.
    :return: The local weight shard for the given tensor.
    """
    dims_rank_info, dims_map = extract_infomation_from_dtensor(
        target, dest_name, splited_dims
    )
    dims_rank_info = {
        int(k): DimSliceInfo.from_dict(v) for k, v in dims_rank_info.items()
    }
    for mdim in mesh_dims:
        if mdim in dims_map:
            tp_dim = dims_map[mdim]
            dims_rank_info[tp_dim].offset = min(
                tensor.shape[tp_dim], dims_rank_info[tp_dim].offset
            )
            dims_rank_info[tp_dim].length = min(
                tensor.shape[tp_dim] - dims_rank_info[tp_dim].offset,
                dims_rank_info[tp_dim].length,
            )
            dims_rank_info[tp_dim].total_size = min(
                tensor.shape[tp_dim], dims_rank_info[tp_dim].total_size
            )
    shard = slice_tensor_with_strategies(
        tensor,
        dims_rank_info,
    )
    return shard


def merge_continuous_dims(
    tensor: torch.Tensor, dims_list: List[List[int]]
) -> torch.Tensor:
    """
    Merge continuous dimensions from start_dim to end_dim (inclusive) into one dimension

    Args:
        tensor: Input tensor
        dims: List of dimensions to merge (must be continuous)
    """
    shape = list(tensor.shape)

    merged = {}
    # Calculate the size of the merged dimension
    for dims in dims_list:
        merged_size = 1
        for i in dims:
            merged_size *= shape[i]
        for i in dims:
            merged[i] = (merged_size, dims)

    # Create new shape
    new_shape = []
    handled_dims = set()
    for i in range(len(shape)):
        if i in merged:
            if i not in handled_dims:
                new_shape.append(merged[i][0])
                handled_dims.update(merged[i][1])
        else:
            new_shape.append(shape[i])

    return tensor.view(*new_shape)
