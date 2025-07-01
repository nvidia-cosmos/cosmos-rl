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
from cosmos_rl.utils.parallelism import ParallelDims, ParallelismConfig
from typing import Dict, List, Tuple, Callable, Any, Optional
from cosmos_rl.policy.model.base import WeightMapper


class DimRankInfo:
    rank: int
    size: int
    dim: str
    length: int = 1

    def __init__(self, rank: int, size: int, dim: str = "", length: int = 1):
        """
        Initialize the DimRankInfo with the given rank, size, and dimension.
        """
        self.rank = rank
        self.size = size
        self.dim = dim
        self.length = length

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create a DimRankInfo object from a dictionary.
        :param data: A dictionary containing 'rank', 'size', 'dim', and optionally 'length'.
        :return: A DimRankInfo object.
        """
        return DimRankInfo(
            rank=data["rank"],
            size=data["size"],
            dim=data["dim"],
            length=data.get("length", 1),
        )


def slice_tensor_with_strategy(
    tensor: torch.Tensor, idx: int, tensor_split_strategy: DimRankInfo
):
    view = tensor
    assert view.shape[idx] % tensor_split_strategy.size == 0
    start = view.shape[idx] // tensor_split_strategy.size * tensor_split_strategy.rank
    length = (
        view.shape[idx] // tensor_split_strategy.size * tensor_split_strategy.length
    )
    if view.dim() == 1:
        if idx == 0:
            view = view[start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 1D tensor.")
    elif view.dim() == 2:
        if idx == 0:
            view = view[start : start + length, :]
        elif idx == 1:
            view = view[:, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 2D tensor.")
    elif view.dim() == 3:
        if idx == 0:
            view = view[start : start + length, :, :]
        elif idx == 1:
            view = view[:, start : start + length, :]
        elif idx == 2:
            view = view[:, :, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 3D tensor.")
    elif view.dim() == 4:
        if idx == 0:
            view = view[start : start + length, :, :, :]
        elif idx == 3:
            view = view[:, :, :, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 4D tensor.")
    elif view.dim() == 5:
        if idx == 0:
            view = view[start : start + length, :, :, :, :]
        elif idx == 4:
            view = view[:, :, :, :, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 5D tensor.")
    else:
        raise ValueError(f"Invalid tensor dimension {view.dim()}.")
    return view


def slice_tensor_with_strategies(
    self: torch.Tensor, strategys: Dict[int, DimRankInfo]
) -> torch.Tensor:
    """
    Slices the tensor according to the given strategies.
    :param tensor: The tensor to be sliced.
    :param strategys: A dictionary mapping dimension indices to DimRankInfo objects.
    :return: The sliced tensor.
    """
    view = self
    for idx, split in strategys.items():
        view = slice_tensor_with_strategy(view, idx, split)
    return view


torch.Tensor.cosmos_slice = slice_tensor_with_strategies


class ParallelTopoMapper:
    """
    A class to represent a weight sharing topology map for weight synchronization.
    """

    ordered_dims: List[str] = ["tp", "dp_shard_cp"]

    def __init__(
        self,
        parallelism: Optional[ParallelismConfig],
        world_size: int,
        parallelism_strategy: Callable,
        hf_config: Any,
    ):
        """
        Initialize the ParallelTopoMap with the given parallelism configurations.

        :param parallelism: The parallelism configuration for the policy or rollout.
        :param world_size: The world size for the policy or rollout.
        :param parallelism_strategy: The strategy for the policy parallelism or rollout parallelism.
        :param hf_config: The huggingface config.
        """
        self.parallelism = ParallelDims.from_config_for_analysis(
            parallelism, world_size
        )
        self.parallelism.build_mesh_info()
        self.parallelism_strategy = parallelism_strategy
        ranks = range(self.parallelism.world_size)
        full_rank_map = []
        for r in ranks:
            full_rank = self.get_full_rank(r)
            full_rank_map.append(full_rank)
        self.full_rank_map = full_rank_map
        self.hf_config = hf_config

    def get_full_rank(self, global_rank: int) -> Dict[str, DimRankInfo]:
        """
        Get the full rank of the given rank in the simulation map.

        :param rank: The rank to get the full rank for.
        :return: A dictionary mapping each parallel dimension to its full rank.
        """
        full_rank = {}
        for dim in self.ordered_dims:
            full_rank[dim] = DimRankInfo(
                self.parallelism.get_rank_in_dim(dim, global_rank),
                self.parallelism.get_size_in_dim(dim),
                dim,
            )
        full_rank["pp"] = DimRankInfo(
            self.parallelism.get_rank_in_dim("pp", global_rank),
            self.parallelism.get_size_in_dim("pp"),
            "pp",
        )
        return full_rank

    @classmethod
    def get_unified_rank_info(
        cls, a: DimRankInfo, b: DimRankInfo
    ) -> Tuple[DimRankInfo, DimRankInfo]:
        """
        Get the unified rank information for two DimRankInfo objects.
        :param a: The first DimRankInfo object.
        :param b: The second DimRankInfo object.
        :return: A tuple containing the unified rank information for both objects.
        """
        size = max(a.size, b.size)
        assert (
            size % a.size == 0 and size % b.size == 0
        ), "Sizes are not compatible for unification"
        scale_a = size // a.size
        scale_b = size // b.size
        scaled_a_size = a.size * scale_a
        scaled_b_size = b.size * scale_b
        scaled_a_rank = a.rank * scale_a
        scaled_b_rank = b.rank * scale_b
        unified_a = DimRankInfo(scaled_a_rank, scaled_a_size, a.dim, a.length * scale_a)
        unified_b = DimRankInfo(scaled_b_rank, scaled_b_size, b.dim, b.length * scale_b)
        return unified_a, unified_b

    @classmethod
    def rank_overlap(cls, a: DimRankInfo, b: DimRankInfo) -> DimRankInfo:
        """
        Check if the ranks of two DimRankInfo objects overlap.

        :param a: The first DimRankInfo object.
        :param b: The second DimRankInfo object.
        :return: A DimRankInfo object representing the overlap, or None if there is no overlap.
        """
        a_new, b_new = cls.get_unified_rank_info(a, b)
        assert a_new.size == b_new.size, "Sizes do not match after unification"
        left = max(a_new.rank, b_new.rank)
        right = min(
            a_new.rank + a_new.length,
            b_new.rank + b_new.length,
        )
        overlapped = None
        if left < right:
            overlapped = DimRankInfo(left, a_new.size, a_new.dim, right - left)
        return overlapped

    @classmethod
    def relative_rank(cls, smaller: DimRankInfo, larger: DimRankInfo) -> DimRankInfo:
        """
        Get the relative rank of two DimRankInfo objects.
        :param smaller: The smaller DimRankInfo object.
        :param larger: The larger DimRankInfo object.
        :return: A DimRankInfo object representing the relative rank.
        """
        s, l = cls.get_unified_rank_info(smaller, larger)  # noqa: E741
        assert s.rank >= l.rank, "Smaller rank is not less than or equal to larger rank"
        assert (
            s.rank + s.length <= l.rank + l.length
        ), "Smaller rank does not fit within larger rank"
        rank = s.rank - l.rank
        size = l.length
        length = s.length
        return DimRankInfo(rank, size, s.dim, length)

    @classmethod
    def merge_rank(cls, outter: DimRankInfo, inner: DimRankInfo) -> DimRankInfo:
        """
        Merge two DimRankInfo objects into one.
        :param outter: The outer DimRankInfo object.
        :param inner: The inner DimRankInfo object.
        :return: A DimRankInfo object representing the merged rank.
        """
        assert outter.length == 1, "Outer rank length must be 1"
        size = outter.size * inner.size
        rank = outter.rank * inner.size + inner.rank
        length = inner.length
        return DimRankInfo(rank, size, outter.dim, length)

    @classmethod
    def tensor_sharing_overlap_at_dim(
        cls,
        policy_rank: Dict[int, DimRankInfo],
        rollout_rank: Dict[int, DimRankInfo],
        dim: int,
    ) -> Tuple[DimRankInfo, DimRankInfo]:
        """
        Get the tensor sharing information one different dimensions.
        :param policy_rank: The policy rank information.
        :param rollout_rank: The rollout rank information.
        :return: A set of dimensions that are different.
        """
        if dim not in policy_rank:
            p = DimRankInfo(0, 1)
        else:
            p = policy_rank[dim]
        if dim not in rollout_rank:
            r = DimRankInfo(0, 1)
        else:
            r = rollout_rank[dim]

        p_new, r_new = cls.get_unified_rank_info(p, r)
        overlap = cls.rank_overlap(p_new, r_new)
        if overlap is None:
            # logger.warning(f"No rank overlap detected in {dim}: {p} and {r}")
            return None, None
        overlap_r = cls.relative_rank(overlap, r_new)
        overlap_p = cls.relative_rank(overlap, p_new)
        return overlap_p, overlap_r

    def shard_dim_info(
        self,
        rank_infos: Dict[str, DimRankInfo],
        dim: str,
    ) -> DimRankInfo:
        """
        Get the tensor sharing information one different dimensions.
        :param policy_rank: The policy rank information.
        :param rollout_rank: The rollout rank information.
        :return: A set of dimensions that are different.
        """
        if dim not in rank_infos:
            p = DimRankInfo(0, 1, dim)
        else:
            p = rank_infos[dim]

        return p

    def shard_merged_dim_info(
        self,
        rank_info: Dict[str, DimRankInfo],
    ) -> DimRankInfo:
        """
        Get the tensor sharing information for the same dimensions.
        :param rank_info: The rank information.
        :return: A set of dimensions that are the same.
        """
        assert len(self.ordered_dims) == 2, "Only two dimensions are supported"
        if self.ordered_dims[0] not in rank_info:
            rank_info[self.ordered_dims[0]] = DimRankInfo(0, 1, self.ordered_dims[0])
        if self.ordered_dims[1] not in rank_info:
            rank_info[self.ordered_dims[1]] = DimRankInfo(0, 1, self.ordered_dims[1])
        # Merge the ranks of the two dimensions
        p = self.merge_rank(
            rank_info[self.ordered_dims[0]], rank_info[self.ordered_dims[1]]
        )
        return p

    @classmethod
    def get_global_ranks_for_given_group_rank(
        cls, parallel_dims: ParallelDims, group_rank: Dict[str, int]
    ) -> List[int]:
        """
        Get the global ranks for a given group rank in the parallelism configuration.
        group_rank is subset of parallel_dims, so there could be multiple devices have
        the same group_rank.
        :param parallel_dims: The parallelism configuration.
        :param group_rank: The group rank to get the global ranks for.
        :return: A list of global ranks.
        """
        if len(group_rank) == 0:
            return list(range(parallel_dims.world_size))
        global_ranks = []
        for rank in range(parallel_dims.world_size):
            if all(
                [
                    parallel_dims.get_rank_in_dim(dim, rank) == dimr
                    for dim, dimr in group_rank.items()
                ]
            ):
                global_ranks.append(rank)
        return global_ranks

    def duplicate_ranks_at_given_dimensions(
        self, dims: List[str], global_rank: int
    ) -> List[int]:
        """
        Get the duplicate ranks at the given dimensions.
        :param dims: The dimensions to check.
        :param global_rank: The global rank to check.
        :return: A list of duplicate ranks.
        """
        dims_map = {}
        for dim in dims:
            dims_map[dim] = self.parallelism.get_rank_in_dim(dim, global_rank)

        return ParallelTopoMapper.get_global_ranks_for_given_group_rank(
            self.parallelism, dims_map
        )

    @classmethod
    def policy_to_rollout_assign(
        cls, policys: List[int], rollouts: List[int]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Assign policy ranks to rollout ranks based on the sharing map.
        :param policys: The list of policy ranks.
        :param rollouts: The list of rollout ranks.
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
        rank_info: Dict[str, DimRankInfo],
    ) -> Dict[int, Dict]:
        """
        Generate detailed shard info for the given dimensions and ranks.
        :param dim_to_parallel: A dictionary mapping dimension indices to parallel dimensions.
        :param rank_info: The rank information.
        :return: A dictionary mapping dimension indices to DimRankInfo objects.
        """
        shard_dim_info = {}
        for idx, dims in dim_to_parallel.items():
            if len(dims) == 1:
                shard_dim_info[idx] = self.shard_dim_info(rank_info, dims[0]).__dict__
            elif len(dims) == 2:
                shard_dim_info[idx] = self.shard_merged_dim_info(rank_info).__dict__
            else:
                raise ValueError(
                    f"Invalid dimension mapping: {dims} in generate_slice_strategies"
                )
        return shard_dim_info

    def params_local_shard_info(
        self,
        params: List[Tuple[str, Tuple[int]]],
        global_rank: int,
    ) -> List[Tuple[int, int, Dict[int, Dict], str, Tuple[int]]]:
        """
        Generate loca
        :param params: The parameters to generate local shard info for.
        :param global_rank: The global rank to generate local shard info for.
        :return: A list of tuples containing the generated local shard info.
        Tuple meaning: [sender rank, receiver rank, tensor split strategies, tensor name, tensor shape]
        """
        local_shard_info_all_params = []
        for dest_name, shape in params:
            split_dim_map, dim_to_parallel, pp_rank = self.parallelism_strategy(
                shape, dest_name, self.parallelism, self.hf_config
            )
            dup_ranks = self.duplicate_ranks_at_given_dimensions(
                list(split_dim_map.keys()) + ["pp"], global_rank
            )
            ranks = self.full_rank_map[global_rank]
            if ranks["pp"].rank != pp_rank:
                local_shard_info_all_params.append(
                    {
                        "name": dest_name,
                    }
                )
                continue

            local_shard_info_all_params.append(
                {
                    "name": dest_name,
                    "shard_info": self.generate_local_shard_info(
                        dim_to_parallel, ranks
                    ),
                    "dup_ranks": dup_ranks,
                }
            )
        return local_shard_info_all_params


class ParallelTopoMapperGroup:
    """
    A class to represent a group of weight sharing topology maps for weight synchronization.
    """

    def __init__(
        self,
        global_parallelism: ParallelismConfig,
        world_size: int,
        hf_config: Any,
        is_policy: bool,
        weight_mapper: Optional[WeightMapper] = None,
    ):
        """
        Initialize the ParallelTopoMapperGroup with the given parallelism configurations.

        :param global_parallelism: The parallelism configuration for the policy or rollout.
        :param world_size: The world size for the policy or rollout.
        :param hf_config: The huggingface config.
        """
        self.hf_config = hf_config
        model_type = hf_config.model_type
        self.mapper_group: List[ParallelTopoMapper] = []

        if weight_mapper is None:
            weight_mapper_fn = WeightMapper.get_weight_mapper(model_type)
            self.weight_mapper = weight_mapper_fn(hf_config)
        else:
            self.weight_mapper = weight_mapper

        # The replica has its single global parallelism config
        # But there may be different parallelism strategies executed by different model parts
        # For example: VLM has 2 parts: the visual encoder and the language encoder.
        #   the visual encoder may merge TP and DP meshes, while the language encoder may not
        # Note: policy_strategies and rollout_strategies callable to decide if or how to parallel
        # the param tensor of a give name.
        if is_policy:
            configs: List[ParallelismConfig] = (
                self.weight_mapper.get_policy_parallelism(global_parallelism)
            )
            strategies = self.weight_mapper.get_policy_parallelism_strategy()
        else:
            configs: List[ParallelismConfig] = (
                self.weight_mapper.get_rollout_parallelism(global_parallelism)
            )
            strategies = self.weight_mapper.get_rollout_parallelism_strategy()

        assert len(configs) == len(strategies)
        for config, strategy in zip(configs, strategies):
            self.mapper_group.append(
                ParallelTopoMapper(
                    config,
                    world_size,
                    strategy,
                    hf_config,
                )
            )

    def _cluster_params_by_model_part(
        self, params: List[Tuple[str, int]]
    ) -> List[List[Tuple[str, int]]]:
        """
        Resort the parameters based on the name mapper.
        :param params: The parameters to resort.
        :return: A list of tuples containing the resorted parameters.
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
    ) -> List[Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]]]:
        x = self._cluster_params_by_model_part(hf_key_n_rank)
        insts = []
        for model_index, p in enumerate(x):
            insts.extend(
                self.mapper_group[model_index].params_local_shard_info(p, global_rank)
            )
        return insts


class ParallelizedShardMapper:
    def __init__(self):
        """
        Initialize the ParallelizedShardMapper with empty shard info lists.
        """
        self.policy_all_rank_shard_infos: Optional[List[List[Dict[str, Any]]]] = None
        self.rollout_all_rank_shard_infos: Optional[List[List[Dict[str, Any]]]] = None
        self.send_insts_for_policy: List[
            List[Tuple[int, int, Dict[int, DimRankInfo], str]]
        ] = None
        self.recv_insts_for_rollout: List[
            List[Tuple[int, int, Dict[int, DimRankInfo], str]]
        ] = None

    def post_set_shard_infos(self):
        """
        Post-process the shard infos after they are set.
        """
        self.send_insts_for_policy = []
        self.recv_insts_for_rollout = []
        if (
            self.policy_all_rank_shard_infos is not None
            and self.rollout_all_rank_shard_infos is not None
        ):
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
        policy_all_rank_shard_infos: List[List[Dict[str, Any]]],
    ):
        """
        Set the shard infos for the policy.
        :param policy_all_rank_shard_infos: A list of dictionaries containing shard info for each rank.
        """
        self.policy_all_rank_shard_infos = policy_all_rank_shard_infos
        self.post_set_shard_infos()

    def set_shard_infos_of_rollout(
        self,
        rollout_all_rank_shard_infos: List[List[Dict[str, Any]]],
    ):
        self.rollout_all_rank_shard_infos = rollout_all_rank_shard_infos
        self.post_set_shard_infos()

    def generate_parallelized_shard_send_insts_for_policy(
        self, p_rank: int
    ) -> List[Dict[str, Any]]:
        policy_to_rollout_insts = []

        rollout_shard_dicts = [
            {r["name"]: r for r in r_info}
            for r_info in self.rollout_all_rank_shard_infos
        ]

        for p_info in self.policy_all_rank_shard_infos[p_rank]:
            insts_for_param_name = []
            dest_name = p_info["name"]
            for r_rank, r_infos in enumerate(rollout_shard_dicts):
                r_info = r_infos[dest_name]
                if "shard_info" not in p_info or "shard_info" not in r_info:
                    continue

                shard_info = {
                    k: DimRankInfo.from_dict(v) for k, v in p_info["shard_info"].items()
                }
                p_dup_ranks = p_info["dup_ranks"]

                r_shard_info = {
                    k: DimRankInfo.from_dict(v) for k, v in r_info["shard_info"].items()
                }
                r_dup_ranks = r_info["dup_ranks"]

                all_dims = shard_info.keys() | r_shard_info.keys()

                p_tensor_split_strategys = {}
                for d in all_dims:
                    p_tensor_split_strategy, r_tensor_split_strategy = (
                        ParallelTopoMapper.tensor_sharing_overlap_at_dim(
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
            policy_to_rollout_insts.append(
                {"name": dest_name, "insts": insts_for_param_name}
            )
        return policy_to_rollout_insts

    def generate_parallelized_shard_recv_insts_for_rollout(
        self, r_rank: int
    ) -> List[Dict[str, Any]]:
        rollout_from_policy_insts = []

        policy_shard_dicts = [
            {r["name"]: r for r in r_info}
            for r_info in self.policy_all_rank_shard_infos
        ]

        for r_info in self.rollout_all_rank_shard_infos[r_rank]:
            insts_for_param_name = []
            dest_name = r_info["name"]
            for p_rank, p_infos in enumerate(policy_shard_dicts):
                p_info = p_infos[dest_name]
                if "shard_info" not in p_info or "shard_info" not in r_info:
                    continue

                shard_info = {
                    k: DimRankInfo.from_dict(v) for k, v in p_info["shard_info"].items()
                }
                p_dup_ranks = p_info["dup_ranks"]

                r_shard_info = {
                    k: DimRankInfo.from_dict(v) for k, v in r_info["shard_info"].items()
                }
                r_dup_ranks = r_info["dup_ranks"]

                all_dims = shard_info.keys() | r_shard_info.keys()

                r_tensor_split_strategys = {}
                for d in all_dims:
                    p_tensor_split_strategy, r_tensor_split_strategy = (
                        ParallelTopoMapper.tensor_sharing_overlap_at_dim(
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
            rollout_from_policy_insts.append(
                {"name": dest_name, "insts": insts_for_param_name}
            )
        return rollout_from_policy_insts

    def get_send_insts_for_policy(self, rank: int) -> List[Dict[str, Any]]:
        """
        Get the send instructions for policy.
        :return: A list of send instructions for policy.
        """
        return self.send_insts_for_policy[rank]

    def get_recv_insts_for_rollout(self, rank: int) -> List[Dict[str, Any]]:
        """
        Get the receive instructions for rollout.
        :return: A list of receive instructions for rollout.
        """
        return self.recv_insts_for_rollout[rank]
