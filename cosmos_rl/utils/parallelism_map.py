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

import msgpack
import torch
from cosmos_rl.utils.parallelism import ParallelDims
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from cosmos_rl.utils.constant import COSMOS_HF_MODEL_TYPES
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
from functools import reduce
import asyncio
from cosmos_rl.utils import util
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from cosmos_rl.policy.config import Config as CosmosConfig


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


class WeightSyncInstruction:
    """
    A class to represent a weight synchronization instruction for a specific parameter.
    This class contains the parameter name, the global rank, and the instruction.
    """

    def __init__(
        self,
        policy_rank: int,
        rollout_rank: int,
        slice_strategy: Dict[int, DimSliceInfo],
    ):
        """
        Initialize the WeightSyncInstruction with the given policy rank, rollout rank, and slice strategy.
        :param policy_rank: The rank of the policy in the parallelism configuration.
        :param rollout_rank: The rank of the rollout in the parallelism configuration.
        :param slice_strategy: A dictionary mapping dimension indices to DimSliceInfo objects representing the slicing strategy.
        """
        self.policy_rank = policy_rank
        self.rollout_rank = rollout_rank
        self.slice_strategy = slice_strategy

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"


class WeightSyncInstructionsPerParam:
    """
    A class to represent a collection of weight synchronization instructions for a specific param.
    This class contains the parameter name and a list of synchronization instructions.
    """

    def __init__(self, param_name: str, instructions: List[WeightSyncInstruction]):
        """
        Initialize the WeightSyncInstructionsPerParam with the given parameter name and instructions.
        :param param_name: The name of the parameter for which the instructions are created.
        :param instructions: A list of WeightSyncInstruction objects representing the synchronization instructions.
        """
        self.param_name = param_name
        self.instructions = instructions

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"


class WeightSyncInstructionsGroup:
    """
    A class to represent a group of weight synchronization instructions for multiple parameters.
    This class contains a list of WeightSyncInstructionsPerParam objects.
    """

    def __init__(self, param_instructions: List[WeightSyncInstructionsPerParam]):
        """
        Initialize the WeightSyncInstructionsGroup with the given instructions.
        :param param_instructions: A list of WeightSyncInstructionsPerParam objects representing the synchronization instructions for multiple parameters in one group.
        """
        self.param_instructions = param_instructions

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create a WeightSyncInstructionsGroup object from a dictionary.
        :param data: A dictionary containing the keys 'instructions' and 'param_names'.
        :return: A WeightSyncInstructionsGroup object.
        """
        instructions = [
            WeightSyncInstructionsPerParam(
                param_name=insts["param_name"],
                instructions=[
                    WeightSyncInstruction(
                        policy_rank=inst["policy_rank"],
                        rollout_rank=inst["rollout_rank"],
                        slice_strategy={
                            k: DimSliceInfo.from_dict(v)
                            for k, v in inst["slice_strategy"].items()
                        },
                    )
                    for inst in insts["instructions"]
                ],
            )
            for insts in data["param_instructions"]
        ]
        return cls(instructions)


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
        cls, policys: List[int], rollouts: List[int], p_rank: int, r_rank: int
    ) -> Tuple[List[int], List[int]]:
        """
        Assign policy ranks to rollout ranks sharing the same sharded part related between the given policy and rollout rank.
        :param policys: The list of policy ranks sharing the same sharded part.
        :param rollouts: The list of rollout ranks sharing the same sharded part.
        :param p_rank: The given parallelism rank of the policy to consider.
        :param r_rank: The given parallelism rank of the rollout to consider.
        :return: A tuple containing two list: policy assignment and rollout assignment related between the given policy and rollout rank.
        """
        p_assignment = []
        r_assignment = []
        if len(policys) >= len(rollouts):
            for i, p in enumerate(policys):
                if i >= len(rollouts):
                    break
                if p == p_rank and rollouts[i] == r_rank:
                    p_assignment = [r_rank]
                    r_assignment = [p_rank]
        else:
            group_size = ((len(rollouts) - 1) // len(policys)) + 1
            for i, p in enumerate(policys):
                if p == p_rank:
                    rs = rollouts[
                        i * group_size : min(i * group_size + group_size, len(rollouts))
                    ]
                    if r_rank in rs:
                        p_assignment = [r_rank]
                        r_assignment = [p_rank]
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
        params: List[Tuple[str, int]],
        global_rank: int,
    ) -> Dict[str, Any]:
        """
        Generate local shard information for the given parameters.
        :param params: The parameters to generate local shard information for.
        :param global_rank: The global rank to generate local shard information for.
        :return: A dictionary containing the generated local shard information.
        """
        local_shard_info_all_params = {}
        if self.is_policy:
            self.parallelism_info_for_dtensor_params()
        else:
            self.parallelism_info_for_vllm_params()

        for dest_name, shape in params:
            split_dim_map, dim_to_parallel, pp_rank, dims_rank_info = (
                None,
                None,
                None,
                None,
            )
            if self.parallelism_strategy is not None:
                # If custom parallelism strategy is provided, use it to get the split dimensions and parallel mapping.
                # The custom parallelism strategy has high priority.
                # The custom parallelism strategy is specified from `get_policy_parallelism_strategy` and `get_rollout_parallelism_strategy` in weight mapper.
                split_dim_map, dim_to_parallel, pp_rank = self.parallelism_strategy(
                    shape, dest_name, self.parallelism, self.hf_config
                )

            if split_dim_map is None and dim_to_parallel is None and pp_rank is None:
                # If no custom parallelism strategy is provided, use the default automatically inferred parallelism info for the parameter.
                split_dim_map, dim_to_parallel, pp_rank, dims_rank_info = (
                    self.parallelism_info_for_param(dest_name)
                )

            if split_dim_map is None and dim_to_parallel is None and pp_rank is None:
                continue

            ranks = self.full_mesh_rank_info_map[global_rank]
            if ranks["pp"].offset != pp_rank:
                continue

            local_shard_info_all_params[dest_name] = (
                self.generate_local_shard_info(dim_to_parallel, ranks)
                if dims_rank_info is None
                else dims_rank_info
            )
        # Return a dictionary containing the local shard info for each parameter
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
                global_shape = tuple(param.shape)
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
                decomposed_key_and_slices = (
                    self.weight_mapper.policy_decompose_param_1_to_n_for_sync(
                        self.weight_mapper.policy_map_local_key_to_hf_key(name)
                    )
                )
                if decomposed_key_and_slices:
                    for part_name, part_slice in decomposed_key_and_slices:
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
            should_skip = False
            for part_name in name_parts:
                if hasattr(part, part_name):
                    if isinstance(getattr(part, part_name), Parameter):
                        if part_name == "bias":
                            is_bias = True
                        elif part_name == "weight":
                            is_bias = False
                        else:
                            logger.warning(
                                f"Part {part_name} is not a Parameter. Skipping."
                            )
                            should_skip = True
                        break
                    part = getattr(part, part_name)
                elif str.isdigit(part_name):
                    part = part[int(part_name)]
                else:
                    raise ValueError(f"Part {part_name} not found in {part}. Skipping.")
            if should_skip:
                continue
            dims_map = {}
            if isinstance(part, (QKVParallelLinear)):
                output_dim = getattr(param, "output_dim", 0)
                dims_map["tp"] = output_dim
                assert any(
                    [
                        k in param_name
                        for k in self.weight_mapper.packed_modules_mapping.keys()
                    ]
                ), f"QKVParallelLinear {param_name} is not in packed_modules_mapping {self.weight_mapper.packed_modules_mapping}."
            elif isinstance(part, (MergedColumnParallelLinear)):
                output_dim = getattr(param, "output_dim", 0)
                dims_map["tp"] = output_dim
                assert any(
                    [
                        k in param_name
                        for k in self.weight_mapper.packed_modules_mapping.keys()
                    ]
                ), f"MergedColumnParallelLinear {param_name} is not in packed_modules_mapping {self.weight_mapper.packed_modules_mapping}."
            elif isinstance(part, (RowParallelLinear)):
                input_dim = getattr(param, "input_dim", 1)
                if not is_bias:
                    assert (
                        input_dim is not None
                    ), f"RowParallelLinear {param_name} has no input_dim attribute."
                    dims_map["tp"] = input_dim
            elif isinstance(part, (ColumnParallelLinear)):
                output_dim = getattr(param, "output_dim", 0)
                dims_map["tp"] = output_dim
            elif isinstance(part, VocabParallelEmbedding):
                output_dim = getattr(param, "output_dim", 0)
                assert (
                    not is_bias
                ), f"VocabParallelEmbedding {param_name} should not have bias."
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
            if model_type not in WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY:
                logger.warning(
                    f"[ParallelTopoMapperGroup] can not find {model_type} in weight mapper, use {COSMOS_HF_MODEL_TYPES} model type instead."
                )
                model_type = COSMOS_HF_MODEL_TYPES
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
        self, params: List[Tuple[str, int]]
    ) -> List[List[Tuple[str, int]]]:
        """
        Resort the parameters based on the name mapper.
        :param params: The parameters to resort.
        :return: A list of tuples containing the resorted parameters separated by different model parts.
        """
        if len(self.mapper_group) == 1:
            return [params]
        x = [[] for _ in self.mapper_group]
        for name, rank in params:
            idx = self.weight_mapper.name_to_model_part_index(name)
            x[idx].append((name, rank))
        return x

    def prepare_local_shard_infos(
        self,
        hf_key_n_rank: List[Tuple[str, int]],
        global_rank: int,
    ) -> Dict[str, Any]:
        """
        Prepare local shard information for the given parameters based on the parallelism configuration.
        :param hf_key_n_rank: A list of tuples containing the parameter names and their shape ranks.
        :param global_rank: The global rank to prepare local shard information for.
        :return: A dictionary containing the local shard information for each parameter of that rank.
        """
        x = self._cluster_params_by_model_part(hf_key_n_rank)
        insts = {}
        for model_index, p in enumerate(x):
            insts.update(
                self.mapper_group[model_index].local_shard_info_for_params(
                    p, global_rank
                )
            )
        return insts


class ParallelizedShardMapper:
    """
    A class to manage the parallelized shard mapping for policy and rollout models.
    This class is responsible for gathering shard information for policy and rollout, generating send and receive instructions,
    and managing the parallelism configuration for both policy and rollout models.
    It uses multiprocessing to handle the unpacking of shard information in parallel.
    """

    def __init__(self, config: CosmosConfig, max_processes: Optional[int] = None):
        """
        Initialize the ParallelizedShardMapper with empty shard info lists.
        ParallelizedShardMapper is used to gather the shard information for policy and rollout then generate the send and receive instructions for policy and rollout.
        :param config: The CosmosConfig instance containing the configuration for the parallelism.
        :param max_processes: The maximum number of processes to use for parallel processing.
        """
        self.policy_all_rank_shard_infos: Optional[List[Dict[str, Any]]] = None
        self.rollout_all_rank_shard_infos: Optional[List[Dict[str, Any]]] = None

        self.send_insts_for_policy: Optional[
            List[List[WeightSyncInstructionsGroup]]
        ] = None  # List of instructions of different ranks and for each rank it contains a list of instructions for each parameter group.
        self.recv_insts_for_rollout: Optional[
            List[List[WeightSyncInstructionsGroup]]
        ] = None  # List of instructions of different ranks and for each rank it contains a list of instructions for each parameter group.
        self.config = config
        self.param_to_param_groups = {}
        self.param_groups = set()
        self.policy_raw_info_bytes = None
        self.rollout_raw_info_bytes = None
        self.max_processes = max_processes
        self.scheme_generation_done = asyncio.Event()
        self.scheme_generation_done.clear()

    @classmethod
    def get_instance(
        cls, config: CosmosConfig = None, max_processes: Optional[int] = None
    ) -> "ParallelizedShardMapper":
        """
        Get the singleton instance of ParallelizedShardMapper.
        :param config: The CosmosConfig instance containing the configuration for the parallelism.
        :param max_processes: The maximum number of processes to use for parallel processing.
        :return: The singleton instance of ParallelizedShardMapper.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls(config, max_processes)
        return cls._instance

    @staticmethod
    def unpack_and_set_shard_infos(
        is_policy: bool = True,
    ):
        """
        Unpack and set the shard information for policy and rollout base on raw bytes received.
        :param is_policy: A boolean indicating if the unpacking is for policy or rollout.
        This method is called to extrac the shard information for policy and rollout after they are unpacked from the bytes.
        """
        instance = ParallelizedShardMapper.get_instance()
        if is_policy:
            unpacked_data = msgpack.unpackb(
                instance.policy_raw_info_bytes, strict_map_key=False
            )
        else:
            unpacked_data = msgpack.unpackb(
                instance.rollout_raw_info_bytes, strict_map_key=False
            )
        return unpacked_data

    async def post_set_shard_infos(self):
        """
        Post-process the shard infos after they are set.
        This method generates the send and receive instructions for policy and rollout based on the shard information provided.
        It initializes the send and receive instruction lists for policy and rollout, respectively.
        """
        if (
            self.policy_raw_info_bytes is not None
            and self.rollout_raw_info_bytes is not None
            and self.send_insts_for_policy is None
            and self.recv_insts_for_rollout is None
        ):
            logger.debug(
                "[ParallelizedShardMapper] Start unpacking shard infos for both policy and rollout."
            )
            self.send_insts_for_policy = []
            self.recv_insts_for_rollout = []
            loop = asyncio.get_event_loop()
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(mp_context=ctx, max_workers=2) as pool:
                policy_unpack = loop.run_in_executor(
                    pool,
                    ParallelizedShardMapper.unpack_and_set_shard_infos,
                    True,
                )
                rollout_unpack = loop.run_in_executor(
                    pool,
                    ParallelizedShardMapper.unpack_and_set_shard_infos,
                    False,
                )
                policy_info = await policy_unpack
                rollout_info = await rollout_unpack
            self.policy_raw_info_bytes = None
            self.rollout_raw_info_bytes = None

            self.policy_all_rank_shard_infos = policy_info["shard_infos"]
            for group in policy_info["param_groups"]:
                for param_name in group:
                    self.param_to_param_groups[param_name] = group
                # Each group has already been sorted in replica.
                self.param_groups.add(tuple(group))
            self.sorted_params_all_rank_policy = policy_info["sorted_params"]
            self.rollout_all_rank_shard_infos = rollout_info["shard_infos"]
            for group in rollout_info["param_groups"]:
                for param_name in group:
                    self.param_to_param_groups[param_name] = group
                # Each group has already been sorted in replica.
                self.param_groups.add(tuple(group))
            self.sorted_params_all_rank_rollout = rollout_info["sorted_params"]
            logger.debug(
                "[ParallelizedShardMapper] Finished unpacking shard infos for both policy and rollout."
            )
            self.param_groups = sorted(self.param_groups, key=lambda x: x[0])
            policy_parallelism = ParallelDims.from_config_for_analysis(
                self.config.policy.parallelism,
                self.policy_world_size,
            )
            rollout_parallelism = ParallelDims.from_config_for_analysis(
                self.config.rollout.parallelism,
                self.rollout_world_size,
            )
            self.pp_rank_map_policy = [
                policy_parallelism.get_rank_in_dim("pp", p_rank)
                for p_rank in range(self.policy_world_size)
            ]
            self.pp_rank_map_rollout = [
                rollout_parallelism.get_rank_in_dim("pp", r_rank)
                for r_rank in range(self.rollout_world_size)
            ]
            self.send_insts_for_policy = [None for _ in range(self.policy_world_size)]
            self.recv_insts_for_rollout = [None for _ in range(self.rollout_world_size)]
            ctx = multiprocessing.get_context("fork")
            self.multiprocessing_pool = ProcessPoolExecutor(
                mp_context=ctx, max_workers=self.max_processes
            )
            self.policy_results = [
                loop.run_in_executor(
                    self.multiprocessing_pool,
                    ParallelizedShardMapper.policy_insts_generation_per_rank,
                    rank,
                )
                for rank in range(self.policy_world_size)
            ]
            self.rollout_results = [
                loop.run_in_executor(
                    self.multiprocessing_pool,
                    ParallelizedShardMapper.rollout_insts_generation_per_rank,
                    rank,
                )
                for rank in range(self.rollout_world_size)
            ]
            self.scheme_generation_done.set()
            logger.debug(
                "[ParallelizedShardMapper] Started generating send and receive instructions for policy and rollout asynchronously."
            )

    async def set_shard_infos_of_policy(
        self,
        policy_raw_info_bytes: bytes,
        n_gpus_per_policy: int,
    ):
        """
        Set the shard information for the policy.
        :param policy_raw_info_bytes: The raw bytes containing the shard information for the policy. include:
            shard_infos: A list of dictionaries containing shard info for each rank.
            sorted_params: A list of sorted parameter names for each rank.
            param_groups: A list of parameter groups for the policy. Each group contains multiple parameters with some connections such as from the same original param.
        :param n_gpus_per_policy: The number of GPUs per policy.
        """
        if self.policy_raw_info_bytes is None:
            self.policy_world_size = n_gpus_per_policy
            self.policy_raw_info_bytes = policy_raw_info_bytes
            util.create_async_task(self.post_set_shard_infos())

    async def set_shard_infos_of_rollout(
        self,
        rollout_raw_info_bytes: bytes,
        n_gpus_per_rollout: int,
    ):
        """
        Set the shard information for the rollout.
        :param rollout_raw_info_bytes: The raw bytes containing the shard information for the rollout. include:
            shard_infos: A list of dictionaries containing shard info for each rank.
            sorted_params: A list of sorted parameter names for each rank.
            param_groups: A list of parameter groups for the rollout. Each group contains multiple parameters with some connections such as from the same original param.
        :param n_gpus_per_rollout: The number of GPUs per rollout.
        """
        if self.rollout_raw_info_bytes is None:
            self.rollout_world_size = n_gpus_per_rollout
            self.rollout_raw_info_bytes = rollout_raw_info_bytes
            util.create_async_task(self.post_set_shard_infos())

    def generate_parallelized_shard_send_insts_for_policy(
        self, p_rank: int
    ) -> List[WeightSyncInstructionsGroup]:
        """
        Generate the send instructions for policy based on the shard information.
        :param p_rank: The rank of the policy to generate send instructions for.
        :return: A list of send instructions for policy.
        In the returned list, each element corresponds to a parameter group.
        Each parameter group contains a list of instructions for each parameter in a group represented by `WeightSyncInstructionsGroup`.
        One parameter group includes the parameters with some connections such as from the same original param.
        Correspondingly, each `WeightSyncInstructionsGroup` includes `WeightSyncInstructionsPerTensor` for each parameter in the group.
        Each instruction is a `WeightSyncInstruction` containing the parameter name and the corresponding sharded tensor split strategies.
        """
        sorted_params = self.sorted_params_all_rank_policy[
            self.pp_rank_map_policy[p_rank]
        ]
        policy_shard_dicts = self.policy_all_rank_shard_infos[p_rank]
        policy_to_rollout_insts = []
        name_in_group = set()
        for dest_name in sorted_params:
            insts_for_group = []
            if dest_name not in policy_shard_dicts:
                continue
            if dest_name in self.param_to_param_groups:
                name_in_group.add(dest_name)
                continue
            p_info = policy_shard_dicts[dest_name]
            insts_for_param_name = []
            shard_info = {k: DimSliceInfo.from_dict(v) for k, v in p_info.items()}
            p_dup_ranks = self.get_dup_ranks_for_policy(p_rank, dest_name)
            for r_rank, r_infos in enumerate(self.rollout_all_rank_shard_infos):
                if dest_name not in r_infos:
                    continue
                r_info = r_infos[dest_name]
                r_shard_info = {k: DimSliceInfo.from_dict(v) for k, v in r_info.items()}
                r_dup_ranks = self.get_dup_ranks_for_rollout(r_rank, dest_name)

                all_dims = shard_info.keys() | r_shard_info.keys()

                p_tensor_split_strategys = {}
                r_tensor_split_strategys = {}
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
                    p_tensor_split_strategys[d] = p_tensor_split_strategy.__dict__
                    r_tensor_split_strategys[d] = r_tensor_split_strategy.__dict__
                if p_tensor_split_strategys is None:
                    continue
                else:
                    p_assignments, r_assignments = (
                        ParallelTopoMapper.policy_to_rollout_assign(
                            p_dup_ranks, r_dup_ranks, p_rank, r_rank
                        )
                    )
                    for r in p_assignments:
                        insts_for_param_name.append(
                            WeightSyncInstruction(
                                p_rank, r, p_tensor_split_strategys
                            ).__dict__
                        )
            if insts_for_param_name:
                insts_for_group.append(
                    WeightSyncInstructionsPerParam(
                        dest_name, insts_for_param_name
                    ).__dict__
                )
            if insts_for_group:
                policy_to_rollout_insts.append(
                    WeightSyncInstructionsGroup(insts_for_group).__dict__
                )
            else:
                logger.warning(
                    f"No send instructions generated for parameter {dest_name} in policy rank {p_rank}."
                )
        for group in self.param_groups:
            insts_for_group = []
            for dest_name in group:
                if dest_name not in policy_shard_dicts:
                    continue
                if dest_name in name_in_group:
                    name_in_group.remove(dest_name)
                p_info = policy_shard_dicts[dest_name]
                insts_for_param_name = []
                shard_info = {k: DimSliceInfo.from_dict(v) for k, v in p_info.items()}
                p_dup_ranks = self.get_dup_ranks_for_policy(p_rank, dest_name)
                for r_rank, r_infos in enumerate(self.rollout_all_rank_shard_infos):
                    if dest_name not in r_infos:
                        continue
                    r_info = r_infos[dest_name]
                    r_shard_info = {
                        k: DimSliceInfo.from_dict(v) for k, v in r_info.items()
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
                        p_tensor_split_strategys[d] = p_tensor_split_strategy.__dict__
                    if p_tensor_split_strategys is None:
                        continue
                    else:
                        p_assignments, _ = ParallelTopoMapper.policy_to_rollout_assign(
                            p_dup_ranks, r_dup_ranks, p_rank, r_rank
                        )
                        for r in p_assignments:
                            insts_for_param_name.append(
                                WeightSyncInstruction(
                                    p_rank, r, p_tensor_split_strategys
                                ).__dict__
                            )
                if insts_for_param_name:
                    insts_for_group.append(
                        WeightSyncInstructionsPerParam(
                            dest_name, insts_for_param_name
                        ).__dict__
                    )
            if insts_for_group:
                policy_to_rollout_insts.append(
                    WeightSyncInstructionsGroup(insts_for_group).__dict__
                )
        if len(name_in_group) > 0:
            logger.warning(
                f"No send instructions generated for parameters {name_in_group} in policy rank {p_rank}."
            )
        # Pack the instructions into msgpack format for efficient serialization.
        return msgpack.packb(policy_to_rollout_insts)

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
        if name in self.policy_all_rank_shard_infos[p_rank]:
            p_info = self.policy_all_rank_shard_infos[p_rank][name]
        else:
            return []

        ranks = []
        for p_rank, p_infos in enumerate(self.policy_all_rank_shard_infos):
            if name in p_infos:
                if p_infos[name] == p_info:
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
        if name in self.rollout_all_rank_shard_infos[p_rank]:
            p_info = self.rollout_all_rank_shard_infos[p_rank][name]
        else:
            return []

        ranks = []
        for p_rank, p_infos in enumerate(self.rollout_all_rank_shard_infos):
            if name in p_infos:
                if p_infos[name] == p_info:
                    ranks.append(p_rank)
        return ranks

    def generate_parallelized_shard_recv_insts_for_rollout(
        self, r_rank: int
    ) -> List[WeightSyncInstructionsGroup]:
        """
        Generate the receive instructions for rollout based on the shard information.
        :param r_rank: The rank of the rollout to generate receive instructions for.
        :return: A list of receive instructions for rollout.
        In the returned list, each element corresponds to a parameter group.
        Each parameter group contains a list of instructions for each parameter in a group represented by `WeightSyncInstructionsGroup`.
        One parameter group includes the parameters with some connections such as from the same original param.
        Correspondingly, each `WeightSyncInstructionsGroup` includes `WeightSyncInstructionsPerTensor` for each parameter in the group.
        Each instruction is a `WeightSyncInstruction` containing the parameter name and the corresponding sharded tensor split strategies.
        """
        sorted_params = self.sorted_params_all_rank_rollout[
            self.pp_rank_map_rollout[r_rank]
        ]
        rollout_shard_dicts = self.rollout_all_rank_shard_infos[r_rank]
        rollout_from_policy_insts = []
        name_in_group = set()
        for dest_name in sorted_params:
            insts_for_group = []
            if dest_name not in rollout_shard_dicts:
                continue
            if dest_name in self.param_to_param_groups:
                name_in_group.add(dest_name)
                continue
            r_info = rollout_shard_dicts[dest_name]
            insts_for_param_name = []
            shard_info = {k: DimSliceInfo.from_dict(v) for k, v in r_info.items()}
            r_dup_ranks = self.get_dup_ranks_for_rollout(r_rank, dest_name)
            for p_rank, p_infos in enumerate(self.policy_all_rank_shard_infos):
                if dest_name not in p_infos:
                    continue
                p_info = p_infos[dest_name]
                p_shard_info = {k: DimSliceInfo.from_dict(v) for k, v in p_info.items()}
                p_dup_ranks = self.get_dup_ranks_for_policy(p_rank, dest_name)

                all_dims = p_shard_info.keys() | shard_info.keys()
                r_tensor_split_strategys = {}
                for d in all_dims:
                    p_tensor_split_strategy, r_tensor_split_strategy = (
                        ParallelTopoMapper.tensor_overlap_info_at_dim(
                            p_shard_info, shard_info, d
                        )
                    )
                    if r_tensor_split_strategy is None:
                        assert p_tensor_split_strategy is None
                        r_tensor_split_strategys = None
                        break
                    r_tensor_split_strategys[d] = r_tensor_split_strategy.__dict__
                if r_tensor_split_strategys is None:
                    continue
                else:
                    _, r_assignments = ParallelTopoMapper.policy_to_rollout_assign(
                        p_dup_ranks, r_dup_ranks, p_rank, r_rank
                    )
                    for p in r_assignments:
                        insts_for_param_name.append(
                            WeightSyncInstruction(
                                p, r_rank, r_tensor_split_strategys
                            ).__dict__
                        )
            if insts_for_param_name:
                insts_for_group.append(
                    WeightSyncInstructionsPerParam(
                        dest_name, insts_for_param_name
                    ).__dict__
                )
            if insts_for_group:
                rollout_from_policy_insts.append(
                    WeightSyncInstructionsGroup(insts_for_group).__dict__
                )
            else:
                raise ValueError(
                    f"No recv instructions generated for parameter {dest_name} in rollout rank {r_rank}."
                )
        for group in self.param_groups:
            insts_for_group = []
            for dest_name in group:
                if dest_name not in rollout_shard_dicts:
                    continue
                if dest_name in name_in_group:
                    name_in_group.remove(dest_name)
                r_info = rollout_shard_dicts[dest_name]
                insts_for_param_name = []
                shard_info = {k: DimSliceInfo.from_dict(v) for k, v in r_info.items()}
                r_dup_ranks = self.get_dup_ranks_for_rollout(r_rank, dest_name)
                for p_rank, p_infos in enumerate(self.policy_all_rank_shard_infos):
                    if dest_name not in p_infos:
                        continue
                    p_info = p_infos[dest_name]
                    p_shard_info = {
                        k: DimSliceInfo.from_dict(v) for k, v in p_info.items()
                    }
                    p_dup_ranks = self.get_dup_ranks_for_policy(p_rank, dest_name)

                    all_dims = p_shard_info.keys() | shard_info.keys()
                    r_tensor_split_strategys = {}
                    for d in all_dims:
                        p_tensor_split_strategy, r_tensor_split_strategy = (
                            ParallelTopoMapper.tensor_overlap_info_at_dim(
                                p_shard_info, shard_info, d
                            )
                        )
                        if r_tensor_split_strategy is None:
                            assert p_tensor_split_strategy is None
                            r_tensor_split_strategys = None
                            break
                        r_tensor_split_strategys[d] = r_tensor_split_strategy.__dict__
                    if r_tensor_split_strategys is None:
                        continue
                    else:
                        _, r_assignments = ParallelTopoMapper.policy_to_rollout_assign(
                            p_dup_ranks, r_dup_ranks, p_rank, r_rank
                        )
                        for p in r_assignments:
                            insts_for_param_name.append(
                                WeightSyncInstruction(
                                    p, r_rank, r_tensor_split_strategys
                                ).__dict__
                            )
                if insts_for_param_name:
                    insts_for_group.append(
                        WeightSyncInstructionsPerParam(
                            dest_name, insts_for_param_name
                        ).__dict__
                    )
            if insts_for_group:
                rollout_from_policy_insts.append(
                    WeightSyncInstructionsGroup(insts_for_group).__dict__
                )
        assert (
            len(name_in_group) == 0
        ), f"No recv instructions generated for parameters {name_in_group} in rollout rank {r_rank}."
        # Pack the instructions into msgpack format for efficient serialization.
        return msgpack.packb(rollout_from_policy_insts)

    @staticmethod
    def policy_insts_generation_per_rank(
        rank: int,
    ) -> Tuple[List[WeightSyncInstructionsGroup], List[Dict[str, Any]]]:
        """
        Generate the send instructions for policy in parallel for a given rank.
        This method is called in parallel for each rank to generate the send instructions.
        :param rank: The rank of the current process.
        :return: The generated send instructions for policy of that rank and metadata used for the rollout from policy instructions.
        """
        logger.debug(
            f"[ParallelizedShardMapper] Start generating send instructions for rank {rank}"
        )
        mapper = ParallelizedShardMapper.get_instance()
        if rank < mapper.policy_world_size:
            send_insts = mapper.generate_parallelized_shard_send_insts_for_policy(rank)
        else:
            send_insts = None
        logger.debug(
            f"[ParallelizedShardMapper] Finished generating send instructions for rank {rank}"
        )
        return send_insts

    @staticmethod
    def rollout_insts_generation_per_rank(
        rank: int,
    ) -> List[WeightSyncInstructionsGroup]:
        """
        Generate the receive instructions for rollout in parallel for a given rank.
        This method is called in parallel for each rank to generate the receive instructions.
        :param rank: The rank of the current process.
        :return: The generated receive instructions for rollout of that rank.
        """
        logger.debug(
            f"[ParallelizedShardMapper] Start generating receive instructions for rank {rank}"
        )
        mapper = ParallelizedShardMapper.get_instance()
        if rank < mapper.rollout_world_size:
            recv_insts = mapper.generate_parallelized_shard_recv_insts_for_rollout(rank)
        else:
            recv_insts = None
        logger.debug(
            f"[ParallelizedShardMapper] Finished generating receive instructions for rank {rank}"
        )
        return recv_insts

    async def get_send_insts_for_policy(
        self, rank: int
    ) -> List[WeightSyncInstructionsGroup]:
        """
        Get the send instructions for policy of the given rank.
        :param rank: The rank of the policy to get send instructions for.
        :return: A list of send instructions for policy.
        """
        if self.send_insts_for_policy[rank] is None:
            self.send_insts_for_policy[rank] = await self.policy_results[rank]
        return self.send_insts_for_policy[rank]

    async def get_recv_insts_for_rollout(
        self, rank: int
    ) -> List[WeightSyncInstructionsGroup]:
        """
        Get the receive instructions for rollout of the given rank.
        :param rank: The rank of the rollout to get receive instructions for.
        :return: A list of receive instructions for rollout.
        """
        if self.recv_insts_for_rollout[rank] is None:
            self.recv_insts_for_rollout[rank] = await self.rollout_results[rank]
        return self.recv_insts_for_rollout[rank]
