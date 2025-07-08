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
from cosmos_rl.utils.parallelism import ParallelDims
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils.logging import logger
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    ColumnParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from torch.nn.parameter import Parameter
from math import gcd
from functools import reduce, partial


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
    assert view.shape[idx] % tensor_split_strategy.total_size == 0
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


class ParallelTopoMapper:
    """
    A class used for weight sharing topology map for weight synchronization.
    """

    ordered_dims: List[str] = ["tp", "dp_shard_cp", "dp_cp_tp"]

    def __init__(
        self,
        parallelism: Optional[ParallelDims],
        parallelism_strategy: Optional[Callable],
        weight_mapper: WeightMapper,
        hf_config: Any,
        is_policy: bool,
        underlying_model: Any,
    ):
        """
        Initialize the ParallelTopoMap with the given parallelism configurations.

        :param parallelism: The parallelism config for the policy or rollout.
        :param parallelism_strategy: The strategy function for the policy parallelism or rollout parallelism if specified.
        :param weight_mapper: The weight mapper to use for mapping weights.
        :param hf_config: The huggingface config.
        :param is_policy: A boolean indicating if this is for policy or rollout.
        :param underlying_model: The underlying model for which the parallelism map is created.
        """
        self.parallelism = parallelism
        self.parallelism_strategy = parallelism_strategy
        ranks = range(self.parallelism.world_size)
        full_mesh_rank_info_map = []
        for r in ranks:
            full_rank = self.get_full_mesh_rank_info(r)
            full_mesh_rank_info_map.append(full_rank)
        self.full_mesh_rank_info_map = full_mesh_rank_info_map
        self.hf_config = hf_config
        self.weight_mapper = weight_mapper
        self.is_policy = is_policy
        self.underlying_model = underlying_model

    def get_full_mesh_rank_info(self, global_rank: int) -> Dict[str, DimSliceInfo]:
        """
        Get the full mesh rank info of the given global rank in the simulation map.

        :param global_rank: The global rank to get the full rank for.
        :return: A dictionary mapping each parallel mesh dimension to its mesh rank info represented using DimSliceInfo.
        """
        full_mesh_rank_info = {}
        for dim in self.ordered_dims:
            full_mesh_rank_info[dim] = DimSliceInfo(
                self.parallelism.get_rank_in_dim(dim, global_rank),
                self.parallelism.get_size_in_dim(dim),
                dim,
            )
        full_mesh_rank_info["pp"] = DimSliceInfo(
            self.parallelism.get_rank_in_dim("pp", global_rank),
            self.parallelism.get_size_in_dim("pp"),
            "pp",
        )
        return full_mesh_rank_info

    @classmethod
    def get_unified_rank_info(
        cls, a: DimSliceInfo, b: DimSliceInfo
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
        unified_a = DimSliceInfo(
            scaled_a_rank, scaled_a_size, a.dim, a.length * scale_a
        )
        unified_b = DimSliceInfo(
            scaled_b_rank, scaled_b_size, b.dim, b.length * scale_b
        )
        return unified_a, unified_b

    @classmethod
    def rank_overlap(cls, a: DimSliceInfo, b: DimSliceInfo) -> DimSliceInfo:
        """
        Check if the parts of two DimSliceInfo objects overlap.

        :param a: The first DimSliceInfo object.
        :param b: The second DimSliceInfo object.
        :return: A DimSliceInfo object representing the overlap, or None if there is no overlap.
        """
        a_new, b_new = cls.get_unified_rank_info(a, b)
        assert (
            a_new.total_size == b_new.total_size
        ), "Sizes do not match after unification"

        left = max(a_new.offset, b_new.offset)
        right = min(
            a_new.offset + a_new.length,
            b_new.offset + b_new.length,
        )
        overlapped = None
        if left < right:
            overlapped = DimSliceInfo(left, a_new.total_size, a_new.dim, right - left)
        return overlapped

    @classmethod
    def relative_rank(cls, smaller: DimSliceInfo, larger: DimSliceInfo) -> DimSliceInfo:
        """
        Get the relative slice information of two DimSliceInfo objects.
        :param smaller: The smaller DimSliceInfo object.
        :param larger: The larger DimSliceInfo object.
        :return: A DimSliceInfo object representing the relative slice of the smaller on in the larger one.
        """
        s, l = cls.get_unified_rank_info(smaller, larger)  # noqa: E741
        assert (
            s.offset >= l.offset
        ), "Smaller rank is not less than or equal to larger rank"
        assert (
            s.offset + s.length <= l.offset + l.length
        ), "Smaller rank does not fit within larger rank"
        rank = s.offset - l.offset
        size = l.length
        length = s.length
        return DimSliceInfo(rank, size, s.dim, length)

    @classmethod
    def merge_rank(cls, outter: DimSliceInfo, inner: DimSliceInfo) -> DimSliceInfo:
        """
        Merge two nested DimSliceInfo objects into one.
        :param outter: The DimSliceInfo object at a outter dimension.
        :param inner: The DimSliceInfo object at an inner dimension.
        :return: A DimSliceInfo object representing the merged slice information.
        """
        assert outter.length == 1, "Outer rank length must be 1"
        size = outter.total_size * inner.total_size
        rank = outter.offset * inner.total_size + inner.offset
        length = inner.length
        return DimSliceInfo(rank, size, outter.dim, length)

    @classmethod
    def tensor_overlap_info_at_dim(
        cls,
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

        p_new, r_new = cls.get_unified_rank_info(p, r)
        overlap = cls.rank_overlap(p_new, r_new)

        if overlap is None:
            return None, None
        overlap_r = cls.relative_rank(overlap, r_new)
        overlap_p = cls.relative_rank(overlap, p_new)
        return overlap_p.simplify(), overlap_r.simplify()

    def shard_info_at_dim(
        self,
        rank_infos: Dict[str, DimSliceInfo],
        dim: str,
    ) -> DimSliceInfo:
        """
        Get the sharded slice information at one mesh dimension.
        :param rank_infos: The shared slice information for all related mesh dimensions.
        :param dim: The dimension to get the shard information for.
        :return: A DimSliceInfo object representing the sharded slice information for the given dimension.
        """
        if dim not in rank_infos:
            p = DimSliceInfo(0, 1, dim)
        else:
            p = rank_infos[dim]

        return p

    def merged_shard_info_at_dim(
        self,
        rank_info: Dict[str, DimSliceInfo],
    ) -> DimSliceInfo:
        """
        Get the merged sharded slice information for different mesh dimensions.
        :param rank_info: The slice information for the mesh dimensions.
        :return: A DimSliceInfo object representing the merged sharded slice information for the given dimensions.
        """

        if self.ordered_dims[0] not in rank_info:
            rank_info[self.ordered_dims[0]] = DimSliceInfo(0, 1, self.ordered_dims[0])
        if self.ordered_dims[1] not in rank_info:
            rank_info[self.ordered_dims[1]] = DimSliceInfo(0, 1, self.ordered_dims[1])
        # Merge the ranks of the two dimensions
        p = self.merge_rank(
            rank_info[self.ordered_dims[0]], rank_info[self.ordered_dims[1]]
        )
        return p

    @classmethod
    def get_global_ranks_for_given_mesh_rank(
        cls, parallel_dims: ParallelDims, mesh_rank: Dict[str, int]
    ) -> List[int]:
        """
        Get the global ranks for a given mesh rank in the parallelism configuration.
        mesh_rank is subset of parallel_dims, so there could be multiple devices have
        the same mesh_rank.
        :param parallel_dims: The parallelism configuration.
        :param mesh_rank: The mesh rank to get the global ranks whose mesh rank matches the given mesh_rank.
        :return: A list of global ranks.
        """
        if len(mesh_rank) == 0:
            return list(range(parallel_dims.world_size))
        global_ranks = []
        for rank in range(parallel_dims.world_size):
            if all(
                [
                    parallel_dims.get_rank_in_dim(dim, rank) == dimr
                    for dim, dimr in mesh_rank.items()
                ]
            ):
                global_ranks.append(rank)
        return global_ranks

    def duplicate_ranks_at_given_dimensions(
        self, dims: List[str], global_rank: int
    ) -> List[int]:
        """
        Get the duplicate global ranks with same mesh rank info with the given global rank at the specified dimensions.
        :param dims: The dimensions to check for duplicate ranks.
        :param global_rank: The global rank to check.
        :return: A list of duplicate global ranks.
        """
        dims_map = {}
        for dim in dims:
            dims_map[dim] = self.parallelism.get_rank_in_dim(dim, global_rank)

        return ParallelTopoMapper.get_global_ranks_for_given_mesh_rank(
            self.parallelism, dims_map
        )

    @classmethod
    def policy_to_rollout_assign(
        cls, policys: List[int], rollouts: List[int]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Assign policy ranks to rollout ranks sharing the same sharded part.
        :param policys: The list of policy ranks sharing the same sharded part.
        :param rollouts: The list of rollout ranks sharing the same sharded part.
        :return: A tuple containing two dictionaries: policy assignment and rollout assignment.
        """
        p_assignment = {}
        r_assignment = {}
        if len(policys) >= len(rollouts):
            for i, p in enumerate(policys):
                if i >= len(rollouts):
                    break
                p_assignment[p] = [rollouts[i]]
                r_assignment[rollouts[i]] = [p]
        else:
            group_size = ((len(rollouts) - 1) // len(policys)) + 1
            for i, p in enumerate(policys):
                rs = rollouts[
                    i * group_size : min(i * group_size + group_size, len(rollouts))
                ]
                p_assignment[p] = rs
                for r in rs:
                    if r not in r_assignment:
                        r_assignment[r] = []
                    r_assignment[r].append(p)
        for p in policys:
            if p not in p_assignment:
                p_assignment[p] = []
        for r in rollouts:
            if r not in r_assignment:
                r_assignment[r] = []
        return p_assignment, r_assignment

    def generate_local_shard_info(
        self,
        dim_to_parallel: Dict[int, list[str]],
        rank_info: Dict[str, DimSliceInfo],
    ) -> Dict[int, Dict]:
        """
        Generate detailed shard info for the given dimensions and ranks.
        :param dim_to_parallel: A dictionary mapping dimension indices to parallel mesh dimensions.
        :param rank_info: The shard information of the parallel mesh dimensions.
        :return: A dictionary mapping dimension indices to DimSliceInfo objects to represent the detailed shard strategy.
        """
        shard_dim_info = {}
        for idx, dims in dim_to_parallel.items():
            if len(dims) == 1:
                shard_dim_info[idx] = self.shard_info_at_dim(
                    rank_info, dims[0]
                ).__dict__
            elif len(dims) == 2:
                assert (
                    self.ordered_dims[2] not in dims
                ), f"Invalid dimension mapping: {dims} in generate_slice_strategies for merge"
                assert (
                    self.ordered_dims[0] in dims and self.ordered_dims[1] in dims
                ), f"Invalid dimension mapping: {dims} in generate_slice_strategies for merge"
                shard_dim_info[idx] = self.merged_shard_info_at_dim(rank_info).__dict__
            else:
                raise ValueError(
                    f"Invalid dimension mapping: {dims} in generate_slice_strategies"
                )
        return shard_dim_info

    def local_shard_info_for_params(
        self,
        params: List[List[Tuple[str, int]]],
        global_rank: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate local shard information for the given parameters.
        :param params: The parameters to generate local shard information for.
        :param global_rank: The global rank to generate local shard information for.
        :return: A list containing the generated local shard information.
        """
        local_shard_info_all_params = []
        if self.is_policy:
            self.parallelism_info_for_dtensor_params()
        else:
            self.parallelism_info_for_vllm_params()

        for param_group in params:
            group_info = []
            for dest_name, shape in param_group:
                """
                param group may contain multiple parameters with some connections such as from the same original param.
                """

                split_dim_map, dim_to_parallel, pp_rank, dims_rank_info = (
                    self.parallelism_info_for_param(dest_name)
                )

                if (
                    split_dim_map is None
                    and dim_to_parallel is None
                    and pp_rank is None
                ):
                    if self.parallelism_strategy is not None:
                        split_dim_map, dim_to_parallel, pp_rank = (
                            self.parallelism_strategy(
                                shape, dest_name, self.parallelism, self.hf_config
                            )
                        )
                if (
                    split_dim_map is None
                    and dim_to_parallel is None
                    and pp_rank is None
                ):
                    continue

                ranks = self.full_mesh_rank_info_map[global_rank]
                if ranks["pp"].offset != pp_rank:
                    # group_info.append(
                    #     {
                    #         "name": dest_name,
                    #     }
                    # )
                    continue

                group_info.append(
                    {
                        "name": dest_name,
                        "shard_info": self.generate_local_shard_info(
                            dim_to_parallel, ranks
                        )
                        if dims_rank_info is None
                        else dims_rank_info,
                    }
                )
            if group_info:
                local_shard_info_all_params.append(group_info)
        # Return a list of dictionaries containing the local shard info for each parameter group
        return local_shard_info_all_params

    def parallelism_info_for_param(
        self,
        param_name: str,
    ):
        """
        Get the parallelism info for a specific parameter.
        This method returns a tuple of four elements:
            - dims_map: Dict[str, int]: A mapping of mesh names to corresponding dimension index of the param.
            - tensor_dim_to_parallel_map: Dict[int, List[str]]: A mapping of tensor dimension index to their related parallel mesh dimensions.
            - p_rank: int: The pipeline rank of the parameter.
            - dim_slice_info: Optional[Dict[int, Dict]]: A dictionary containing information for the specific slice of the parameter at each dimension index.
        If the parameter name is not found, it returns None for all elements.
        Args:
            param_name (str): The name of the parameter for which to retrieve parallelism info.
        Returns:
            Tuple[Dict[str, int], Dict[int, List[str]], int, Optional[Dict[int, Dict]]]:
            A tuple containing the dimensions map, tensor dimension to parallel map, pipeline rank, and optional dimension slice info.
        """
        if hasattr(self, "parallelism_info_for_params"):
            if param_name in self.parallelism_info_for_params:
                return self.parallelism_info_for_params[param_name]
        else:
            logger.error("No parallelism info found for the given parameter name.")
        return None, None, None, None

    def insert_to_parallelism_info(
        self,
        param_name: str,
        dims_map: Dict[str, int],
        name_to_hf: Callable,
        packed_modules_mapping: Dict[str, Any] = {},
        dims_rank_info: Optional[Dict[int, Dict]] = None,
    ):
        """
        Insert the parallelism info for the policy model parameters.
        This method updates the `policy_parallelism_info_for_params` dictionary with the parameter name,
        its dimensions map, tensor dimension to parallel map, pipeline rank, and optional dimension slice info.
        Args:
            param_name (str): The name of the parameter.
            dims_map (Dict[str, int]): The dimensions map for the parameter.
            name_to_hf (Callable): A function to map parameter names to Hugging Face names.
            packed_modules_mapping (Dict[str, Any], optional): A mapping of packed module names.
            dims_rank_info (Optional[Dict[int, Dict]], optional): A dictionary containing information for the specific slice of the parameter at each dimension index.
        """
        tensor_dim_to_parallel_map = {}
        for k, v in dims_map.items():
            if v not in tensor_dim_to_parallel_map:
                tensor_dim_to_parallel_map[v] = []
            tensor_dim_to_parallel_map[v].append(k)
        p_rank = self.parallelism.pp_coord[0]
        for k, v in packed_modules_mapping.items():
            assert (
                dims_rank_info is None
            ), f"Packed modules mapping {packed_modules_mapping} should not be used with dims_rank_info {dims_rank_info}."
            if k in param_name:
                for rename in v:
                    name = name_to_hf(param_name).replace(k, rename)
                    self.parallelism_info_for_params[name] = (
                        dims_map,
                        tensor_dim_to_parallel_map,
                        p_rank,
                        dims_rank_info,
                    )
                return
        name = name_to_hf(param_name)
        self.parallelism_info_for_params[name] = (
            dims_map,
            tensor_dim_to_parallel_map,
            p_rank,
            dims_rank_info,
        )

    def parallelism_info_for_dtensor_params(self) -> None:
        """
        Get the parallelism info for the dtensor model parameters.
        Normally, this is used for policy model parameters.
        It analyzes the model's named parameters and extracts the parallel dimensions and their shard information.
        The method checks if the model parameters are distributed tensors (DTensor) and extracts their detailed shard information from DTensor specifications.
        This method updates a dictionary with parameter names as keys and their parallel dimensions with shard information as values.
        """
        assert (
            self.is_policy
        ), "parallelism_info_for_dtensor_params should only be called for policy model."
        if hasattr(self, "parallelism_info_for_params"):
            return self.parallelism_info_for_params
        self.parallelism_info_for_params = {}
        for name, param in self.underlying_model.named_parameters():
            is_dist_tensor = isinstance(param, torch.distributed.tensor.DTensor)
            dims_rank_info = {}
            if not is_dist_tensor:
                dims_map = {}
            else:
                dims_map = {}
                global_shape = tuple(param.shape)
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

                for idx, g_size in enumerate(global_shape):
                    offset = int(meta.offsets[idx])
                    total_size = int(g_size)
                    length = int(meta.sizes[idx])
                    if total_size == length:
                        assert (
                            offset == 0
                        ), f"Expected rank 0 for full size dimension {idx}, but got {offset}."
                    else:
                        dims_rank_info[idx] = DimSliceInfo(
                            offset=offset,
                            total_size=total_size,
                            length=length,
                        ).__dict__
                reformatted = (
                    self.weight_mapper.policy_map_param_to_transformed_params_for_sync(
                        self.weight_mapper.policy_map_local_key_to_hf_key(name)
                    )
                )
                if reformatted:
                    for part_name, part_slice in reformatted:
                        splitted_dim_rank_info = {}
                        part_in_local = {}
                        part_slice = {
                            len(global_shape) + k if k < 0 else k: v
                            for k, v in part_slice.items()
                        }
                        all_dims = part_slice.keys() | dims_rank_info.keys()
                        for dim in all_dims:
                            if dim not in part_slice:
                                dim_slice = DimSliceInfo(
                                    offset=0,
                                    total_size=1,
                                )
                            else:
                                dim_slice = DimSliceInfo.from_dict(part_slice[dim])
                            if dim not in dims_rank_info:
                                assert (
                                    len(global_shape) > dim
                                ), f"Dimension {dim} is out of bounds for global shape {global_shape}."
                                local_part = DimSliceInfo(offset=0, total_size=1)
                            else:
                                local_part = DimSliceInfo.from_dict(dims_rank_info[dim])
                            slice_in_splited, overlap_in_local = (
                                self.tensor_overlap_info_at_dim(
                                    {dim: dim_slice}, {dim: local_part}, dim
                                )
                            )
                            if slice_in_splited is None:
                                splitted_dim_rank_info = None
                                break

                            splitted_dim_rank_info[dim] = slice_in_splited.__dict__
                            part_in_local[dim] = overlap_in_local
                        if splitted_dim_rank_info is not None:
                            self.insert_to_parallelism_info(
                                part_name,
                                dims_map,
                                self.weight_mapper.policy_map_local_key_to_hf_key,
                                dims_rank_info=splitted_dim_rank_info,
                            )

                            def slice_tensor_with_part(
                                local: torch.Tensor,
                                part_in_local: Dict[int, DimSliceInfo],
                            ) -> torch.Tensor:
                                """
                                Slice the local tensor with the part in local information.
                                :param local: The local tensor to be sliced.
                                :param part_in_local: The part in local information for slicing.
                                :return: The sliced tensor.
                                """
                                return slice_tensor_with_strategies(
                                    local, part_in_local
                                )

                            self.weight_mapper.set_transform_func_from_local_param_for_sync(
                                self.weight_mapper.policy_map_local_key_to_hf_key(
                                    part_name
                                ),
                                partial(
                                    slice_tensor_with_part, part_in_local=part_in_local
                                ),
                            )

                    continue
            self.insert_to_parallelism_info(
                name,
                dims_map,
                self.weight_mapper.policy_map_local_key_to_hf_key,
                dims_rank_info=dims_rank_info,
            )

    def parallelism_info_for_vllm_params(self):
        """
        Get the parallelism info for the vllm rollout model parameters by analyzing the model's named parameters with vllm instance check.
        Normally, this is used for rollout model parameters.
        It extracts the parallel dimensions and their shard information from the vllm model parameters.
        The method checks if the model parameters are instances of vllm parallel layers (like QKVParallelLinear, MergedColumnParallelLinear, etc.)
        and extracts their detailed shard information from the parameters.
        This method updates a dictionary with parameter names as keys and their parallel dimensions with shard information as values.
        """
        assert (
            not self.is_policy
        ), "parallelism_info_for_vllm_params should only be called for rollout model."
        if hasattr(self, "parallelism_info_for_params"):
            return self.parallelism_info_for_params
        self.parallelism_info_for_params = {}

        for param_name, param in self.underlying_model.named_parameters():
            name_parts = param_name.split(".")
            part = self.underlying_model
            is_bias = False
            for part_name in name_parts:
                if hasattr(part, part_name):
                    if isinstance(getattr(part, part_name), Parameter):
                        if part_name == "bias":
                            is_bias = True
                        elif part_name == "weight":
                            is_bias = False
                        else:
                            raise ValueError(
                                f"Part {part_name} is not a Parameter. Skipping."
                            )
                        break
                    part = getattr(part, part_name)
                elif str.isdigit(part_name):
                    part = part[int(part_name)]
                else:
                    raise ValueError(f"Part {part_name} not found in {part}. Skipping.")
            dims_map = {}
            if isinstance(part, (QKVParallelLinear)):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    output_dim is not None
                ), f"QKVParallelLinear {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
                assert any(
                    [
                        k in param_name
                        for k in self.weight_mapper.packed_modules_mapping.keys()
                    ]
                ), f"QKVParallelLinear {param_name} is not in packed_modules_mapping {self.weight_mapper.packed_modules_mapping}."
            elif isinstance(part, (MergedColumnParallelLinear)):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    output_dim is not None
                ), f"MergedColumnParallelLinear {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
                assert any(
                    [
                        k in param_name
                        for k in self.weight_mapper.packed_modules_mapping.keys()
                    ]
                ), f"MergedColumnParallelLinear {param_name} is not in packed_modules_mapping {self.weight_mapper.packed_modules_mapping}."
            elif isinstance(part, (RowParallelLinear)):
                input_dim = getattr(param, "input_dim", None)
                if not is_bias:
                    assert (
                        input_dim is not None
                    ), f"RowParallelLinear {param_name} has no input_dim attribute."
                    dims_map["tp"] = input_dim
            elif isinstance(part, (ColumnParallelLinear)):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    output_dim is not None
                ), f"ColumnParallelLinear {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
            elif isinstance(part, VocabParallelEmbedding):
                output_dim = getattr(param, "output_dim", None)
                assert (
                    not is_bias
                ), f"VocabParallelEmbedding {param_name} should not have bias."
                assert (
                    output_dim is not None
                ), f"VocabParallelEmbedding {param_name} has no output_dim attribute."
                dims_map["tp"] = output_dim
            else:
                assert (
                    "Parallel" not in part.__class__.__name__
                ), f"Part {part.__class__.__name__} is not a parallel layer. Skipping."

            self.insert_to_parallelism_info(
                param_name,
                dims_map,
                self.weight_mapper._rollout_vllm_name_to_hf,
                self.weight_mapper.packed_modules_mapping,
            )


class ParallelTopoMapperGroup:
    """
    A class to represent a group of weight sharing topology maps used for weight synchronization.
    This class manages multiple ParallelTopoMapper instances, each corresponding to a different parallelism strategy.
    Different model parts may have different parallelism strategies in one whole model.
    It is used to prepare local shard information for parameters based on the parallelism configuration.
    It clusters parameters by model part and prepares local shard information for each part.
    """

    def __init__(
        self,
        global_parallelism: ParallelDims,
        hf_config: Any,
        is_policy: bool,
        underlying_model: Any,
        weight_mapper: Optional[WeightMapper] = None,
    ):
        """
        Initialize the ParallelTopoMapperGroup with the given parallelism configurations.

        :param global_parallelism: The parallelism config for the policy or rollout.
        :param hf_config: The huggingface config.
        :param is_policy: A boolean indicating if this is for policy parallelism.
        :param underlying_model: The underlying model for which the parallelism map is created.
        :param weight_mapper: An optional WeightMapper instance. If None, a default mapper is used based on the model type from hf_config.
        """
        self.hf_config = hf_config
        model_type = hf_config.model_type
        self.mapper_group: List[ParallelTopoMapper] = []

        if weight_mapper is None:
            weight_mapper_fn = WeightMapper.get_weight_mapper(model_type)
            self.weight_mapper = weight_mapper_fn(hf_config)
        else:
            self.weight_mapper = weight_mapper

        # Note: policy_strategies and rollout_strategies callable to decide if or how to parallel
        # the param tensor of a give name if given as a counterpart of the automatic inferred parallelism strategy.

        # The replica has its single global parallelism config
        # But there may be different parallelism strategies executed by different model parts
        # For example: VLM has 2 parts: the visual encoder and the language encoder.
        #   the visual encoder may merge TP and DP meshes, while the language encoder may not
        # So we may need a ParallelTopoMapper for each model part with its own parallelism strategy
        # So the stategies might be a list containing multiple strategies for different model parts.

        if is_policy:
            strategies = self.weight_mapper.get_policy_parallelism_strategy()
        else:
            strategies = self.weight_mapper.get_rollout_parallelism_strategy()

        if strategies:
            for strategy in strategies:
                self.mapper_group.append(
                    ParallelTopoMapper(
                        global_parallelism,
                        strategy,
                        weight_mapper,
                        hf_config,
                        is_policy=is_policy,
                        underlying_model=underlying_model,
                    )
                )
        else:
            # If no parallelism strategy is provided, reserve it as None to use the default automatic parallelism infer.
            self.mapper_group.append(
                ParallelTopoMapper(
                    global_parallelism,
                    None,
                    weight_mapper,
                    hf_config,
                    is_policy=is_policy,
                    underlying_model=underlying_model,
                )
            )

    def _cluster_params_by_model_part(
        self, params: List[List[Tuple[str, int]]]
    ) -> List[List[List[Tuple[str, int]]]]:
        """
        Resort the parameters based on the name mapper.
        :param params: The parameters to resort.
        :return: A list of tuples containing the resorted parameters.
        """
        if len(self.mapper_group) == 1:
            return [params]
        x = [[] for _ in self.mapper_group]

        for param_group in params:
            idx = None
            for name, _ in param_group:
                if idx is not None:
                    assert idx == self.weight_mapper.name_to_model_part_index(name), (
                        f"Parameter {name} is assigned to different model parts in one same group {param_group}: "
                        f"{idx} and {self.weight_mapper.name_to_model_part_index(name)}"
                    )
                else:
                    idx = self.weight_mapper.name_to_model_part_index(name)
            x[idx].append(param_group)
        return x

    def prepare_local_shard_infos(
        self,
        hf_key_n_rank: List[List[Tuple[str, int]]],
        global_rank: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Prepare local shard information for the given parameters based on the parallelism configuration.
        :param hf_key_n_rank: A list of tuples containing the parameter names and their shape ranks.
        :param global_rank: The global rank to prepare local shard information for.
        :return: A list of dictionaries containing the local shard information for each parameter group.
        """
        x = self._cluster_params_by_model_part(hf_key_n_rank)
        insts = []
        for model_index, p in enumerate(x):
            insts.extend(
                self.mapper_group[model_index].local_shard_info_for_params(
                    p, global_rank
                )
            )
        return insts


class ParallelizedShardMapper:
    def __init__(self):
        """
        Initialize the ParallelizedShardMapper with empty shard info lists.
        ParallelizedShardMapper is used to gather the shard information for policy and rollout then generate the send and receive instructions for policy and rollout.
        """
        self.policy_all_rank_shard_infos: Optional[List[List[List[Dict[str, Any]]]]] = (
            None
        )
        self.rollout_all_rank_shard_infos: Optional[
            List[List[List[Dict[str, Any]]]]
        ] = None
        self.send_insts_for_policy: Optional[List[List[List[List[Dict[str, Any]]]]]] = (
            None
        )
        self.recv_insts_for_rollout: Optional[
            List[List[List[List[Dict[str, Any]]]]]
        ] = None

    def post_set_shard_infos(self):
        """
        Post-process the shard infos after they are set.
        This method generates the send and receive instructions for policy and rollout based on the shard information provided.
        It initializes the send and receive instruction lists for policy and rollout, respectively.
        """
        self.send_insts_for_policy = []
        self.recv_insts_for_rollout = []
        if (
            self.policy_all_rank_shard_infos is not None
            and self.rollout_all_rank_shard_infos is not None
        ):
            self.sort_param_with_groups()
            self.policy_shard_dicts = [
                {
                    r["name"]: r
                    for r_info_group in r_infos_per_rank
                    for r in r_info_group
                }
                for r_infos_per_rank in self.policy_all_rank_shard_infos
            ]
            self.rollout_shard_dicts = [
                {
                    r["name"]: r
                    for r_info_group in r_infos_per_rank
                    for r in r_info_group
                }
                for r_infos_per_rank in self.rollout_all_rank_shard_infos
            ]
            for p_rank in range(len(self.policy_all_rank_shard_infos)):
                self.send_insts_for_policy.append(
                    self.generate_parallelized_shard_send_insts_for_policy(p_rank)
                )
            for r_rank in range(len(self.rollout_all_rank_shard_infos)):
                self.recv_insts_for_rollout.append(
                    self.generate_parallelized_shard_recv_insts_for_rollout(r_rank)
                )

    def set_shard_infos_of_policy(
        self,
        policy_all_rank_shard_infos: List[List[List[Dict[str, Any]]]],
    ):
        """
        Set the shard information for the policy.
        :param policy_all_rank_shard_infos: A list of dictionaries containing shard info for each rank. For each rank, it contains a list of parameter groups, and each group contains a list of dictionaries with shard info. Each parameter group may contain multiple parameters with some connections such as from the same original param.
        """
        self.policy_all_rank_shard_infos = policy_all_rank_shard_infos
        self.post_set_shard_infos()

    def set_shard_infos_of_rollout(
        self,
        rollout_all_rank_shard_infos: List[List[List[Dict[str, Any]]]],
    ):
        """
        Set the shard information for the rollout.
        :param policy_all_rank_shard_infos: A list of dictionaries containing shard info for each rank. For each rank, it contains a list of parameter groups, and each group contains a list of dictionaries with shard info. Each parameter group may contain multiple parameters with some connections such as from the same original param.
        """
        self.rollout_all_rank_shard_infos = rollout_all_rank_shard_infos
        self.post_set_shard_infos()

    def sort_param_with_groups(
        self,
    ):
        """
        Sort the parameters with groups from policy and rollout shard infos.
        Merge the parameter groups from policy and rollout shard infos into a single sorted order.
        Consider the grouping of parameters specified by policy and rollout shard infos.
        """
        group_key_map = {}
        param_group_map = {}

        policy_params = set()
        rollout_params = set()
        for all_rank_shard_infos, params_set in zip(
            [self.policy_all_rank_shard_infos, self.rollout_all_rank_shard_infos],
            [policy_params, rollout_params],
        ):
            for p_rank in all_rank_shard_infos:
                for p_group in p_rank:
                    group_key = ";".join(sorted([p["name"] for p in p_group]))
                    if len(p_group) > 1:
                        if p_group[0]["name"] not in param_group_map:
                            assert (
                                group_key not in group_key_map
                            ), f"Parameter {p_group[0]['name']} is not in any group, but group {group_key} already exists."
                        for p_info in p_group:
                            params_set.add(p_info["name"])
                            if p_info["name"] not in param_group_map:
                                param_group_map[p_info["name"]] = group_key
                                group_key_map[group_key] = p_group
                            else:
                                assert (
                                    param_group_map[p_info["name"]] == group_key
                                ), f"Parameter {p_info['name']} is in different groups: {param_group_map[p_info['name']]} and {group_key}"
                    else:
                        params_set.add(p_group[0]["name"])

        groups_map = {}
        for all_rank_shard_infos in [
            self.policy_all_rank_shard_infos,
            self.rollout_all_rank_shard_infos,
        ]:
            for p_rank in all_rank_shard_infos:
                for p_group in p_rank:
                    group_key = ";".join(sorted([p["name"] for p in p_group]))
                    if group_key in group_key_map and group_key not in groups_map:
                        assert (
                            len(p_group) > 1
                        ), f"Parameter group {group_key} should have more than one parameter, but has only {len(p_group)}."
                        groups_map[group_key] = p_group
                    elif len(p_group) == 1 and p_group[0]["name"] in param_group_map:
                        pass
                    else:
                        if group_key not in groups_map:
                            groups_map[group_key] = p_group

        self.sorted_param_groups = [groups_map[key] for key in sorted(groups_map)]
        assert (
            sum(len(sublist) for sublist in self.sorted_param_groups)
            == len(policy_params)
            and sum(len(sublist) for sublist in self.sorted_param_groups)
            == len(rollout_params)
        ), "The total number of parameters in sorted_param_groups does not match the number of parameters in policy and rollout shard infos."

    def generate_parallelized_shard_send_insts_for_policy(
        self, p_rank: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate the send instructions for policy based on the shard information.
        :param p_rank: The rank of the policy to generate send instructions for.
        :return: A list of send instructions for policy.
        In the returned list, each element corresponds to a parameter group.
        Each parameter group contains a list of instructions for each parameter in the group.
        One parameter group contains the paramters with some connections such as from the same original param.
        Each instruction is a dictionary containing the parameter name and the corresponding sharded tensor split strategies.
        """

        policy_to_rollout_insts = []
        for info_group in self.sorted_param_groups:
            insts_for_group = []
            for info in info_group:
                dest_name = info["name"]
                if dest_name not in self.policy_shard_dicts[p_rank]:
                    continue
                p_info = self.policy_shard_dicts[p_rank][dest_name]
                insts_for_param_name = []
                for r_rank, r_infos in enumerate(self.rollout_shard_dicts):
                    if dest_name not in r_infos:
                        continue
                    r_info = r_infos[dest_name]
                    if "shard_info" not in p_info or "shard_info" not in r_info:
                        continue

                    shard_info = {
                        k: DimSliceInfo.from_dict(v)
                        for k, v in p_info["shard_info"].items()
                    }

                    p_dup_ranks = self.get_dup_ranks_for_policy(p_rank, dest_name)
                    r_shard_info = {
                        k: DimSliceInfo.from_dict(v)
                        for k, v in r_info["shard_info"].items()
                    }
                    r_dup_ranks = self.get_dup_ranks_for_rollout(r_rank, dest_name)

                    all_dims = shard_info.keys() | r_shard_info.keys()

                    p_tensor_split_strategys = {}
                    for d in all_dims:
                        p_tensor_split_strategy, r_tensor_split_strategy = (
                            ParallelTopoMapper.tensor_overlap_info_at_dim(
                                shard_info, r_shard_info, d
                            )
                        )
                        if p_tensor_split_strategy is None:
                            assert r_tensor_split_strategy is None
                            p_tensor_split_strategys = None
                            break
                        p_tensor_split_strategys[d] = p_tensor_split_strategy
                    if p_tensor_split_strategys is None:
                        continue
                    else:
                        assignments, _ = ParallelTopoMapper.policy_to_rollout_assign(
                            p_dup_ranks, r_dup_ranks
                        )
                        assignment = assignments[p_rank]
                        for r in assignment:
                            if r == r_rank:
                                insts_for_param_name.append(
                                    (p_rank, r, p_tensor_split_strategys)
                                )
                insts_for_group.append(
                    {"name": dest_name, "insts": insts_for_param_name}
                )
            policy_to_rollout_insts.append(insts_for_group)
        return policy_to_rollout_insts

    def get_dup_ranks_for_policy(
        self,
        p_rank: int,
        name: str,
    ) -> List[List[Dict[str, Any]]]:
        """
        Get the duplicate ranks for a param in policy of the given rank.
        :param p_rank: The rank of the policy to get duplicate ranks for.
        :param name: The name of the parameter to get duplicate ranks for.
        :return: A list of duplicate ranks for policy.
        """
        if name in self.policy_shard_dicts[p_rank]:
            p_info = self.policy_shard_dicts[p_rank][name]
        else:
            return []

        ranks = []
        for p_rank, p_infos in enumerate(self.policy_shard_dicts):
            if name in p_infos:
                if p_infos[name]["shard_info"] == p_info["shard_info"]:
                    ranks.append(p_rank)
        return ranks

    def get_dup_ranks_for_rollout(
        self,
        p_rank: int,
        name: str,
    ) -> List[List[Dict[str, Any]]]:
        """
        Get the duplicate ranks for a param in rollout of the given rank.
        :param p_rank: The rank of the rollout to get duplicate ranks for.
        :param name: The name of the parameter to get duplicate ranks for.
        :return: A list of duplicate ranks for rollout.
        """
        if name in self.rollout_shard_dicts[p_rank]:
            p_info = self.rollout_shard_dicts[p_rank][name]
        else:
            return []

        ranks = []
        for p_rank, p_infos in enumerate(self.rollout_shard_dicts):
            if name in p_infos:
                if p_infos[name]["shard_info"] == p_info["shard_info"]:
                    ranks.append(p_rank)
        return ranks

    def generate_parallelized_shard_recv_insts_for_rollout(
        self, r_rank: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate the receive instructions for rollout based on the shard information.
        :param r_rank: The rank of the rollout to generate receive instructions for.
        :return: A list of receive instructions for rollout.
        In the returned list, each element corresponds to a parameter group.
        Each parameter group contains a list of instructions for each parameter in the group.
        One parameter group contains the paramters with some connections such as from the same original param.
        Each instruction is a dictionary containing the parameter name and the corresponding sharded tensor split strategies.
        """

        rollout_from_policy_insts = []

        for info_group in self.sorted_param_groups:
            insts_for_group = []
            for info in info_group:
                dest_name = info["name"]
                if dest_name not in self.rollout_shard_dicts[r_rank]:
                    continue
                r_info = self.rollout_shard_dicts[r_rank][dest_name]
                insts_for_param_name = []
                for p_rank, p_infos in enumerate(self.policy_shard_dicts):
                    if dest_name not in p_infos:
                        continue
                    p_info = p_infos[dest_name]

                    if "shard_info" not in p_info or "shard_info" not in r_info:
                        continue

                    shard_info = {
                        k: DimSliceInfo.from_dict(v)
                        for k, v in p_info["shard_info"].items()
                    }

                    p_dup_ranks = self.get_dup_ranks_for_policy(p_rank, dest_name)
                    r_shard_info = {
                        k: DimSliceInfo.from_dict(v)
                        for k, v in r_info["shard_info"].items()
                    }
                    r_dup_ranks = self.get_dup_ranks_for_rollout(r_rank, dest_name)

                    all_dims = shard_info.keys() | r_shard_info.keys()

                    r_tensor_split_strategys = {}
                    for d in all_dims:
                        p_tensor_split_strategy, r_tensor_split_strategy = (
                            ParallelTopoMapper.tensor_overlap_info_at_dim(
                                shard_info, r_shard_info, d
                            )
                        )
                        if r_tensor_split_strategy is None:
                            assert p_tensor_split_strategy is None
                            r_tensor_split_strategys = None
                            break
                        r_tensor_split_strategys[d] = r_tensor_split_strategy
                    if r_tensor_split_strategys is None:
                        continue
                    else:
                        _, assignments = ParallelTopoMapper.policy_to_rollout_assign(
                            p_dup_ranks, r_dup_ranks
                        )
                        assignment = assignments[r_rank]

                        for p in assignment:
                            if p == p_rank:
                                insts_for_param_name.append(
                                    (p_rank, r_rank, r_tensor_split_strategys)
                                )
                insts_for_group.append(
                    {"name": dest_name, "insts": insts_for_param_name}
                )
            rollout_from_policy_insts.append(insts_for_group)
        return rollout_from_policy_insts

    def get_send_insts_for_policy(self, rank: int) -> List[List[Dict[str, Any]]]:
        """
        Get the send instructions for policy of the given rank.
        :return: A list of send instructions for policy.
        """
        return self.send_insts_for_policy[rank]

    def get_recv_insts_for_rollout(self, rank: int) -> List[List[Dict[str, Any]]]:
        """
        Get the receive instructions for rollout of the given rank.
        :return: A list of receive instructions for rollout.
        """
        return self.recv_insts_for_rollout[rank]
