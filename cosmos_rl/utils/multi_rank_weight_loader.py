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

"""
Multi-rank weight loading utility for distributed model weight loading.

This module provides a utility class to handle multi-rank loading of model weights
from safetensors files, distributing the I/O workload across multiple ranks and
broadcasting tensors to all ranks.
"""

import os
from typing import Dict, List, Set, Tuple, Optional, Callable, Iterator
import torch
import torch.distributed as dist
from safetensors import safe_open
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.distributed import broadcast_object_cpu


class MultiRankWeightLoader:
    """Utility class for multi-rank loading of model weights from safetensors files."""

    def __init__(self, parallel_dims: ParallelDims):
        """
        Initialize the multi-rank weight loader.

        Args:
            parallel_dims: Parallel dimensions definition.
        """
        # Get current rank and world size for distributed loading
        # When dp_replicate > 1, we need to use a process group that excludes dp_replicate
        # since each replica is independent and should load weights separately
        if dist.is_initialized():
            assert hasattr(parallel_dims, "mesh"), "parallel_dims.mesh is not found"
            # Use weight_loading mesh which excludes dp_replicate
            # This ensures we only communicate within the same replica
            try:
                self.group = parallel_dims.mesh.get_group("weight_loading")
                self.rank = dist.get_rank(self.group)
                self.world_size = dist.get_world_size(self.group)
            except (KeyError, AttributeError):
                raise ValueError(
                    "[MultiRankWeightLoader] weight_loading group not found in parallel_dims.mesh"
                )
        else:
            self.rank = 0
            self.world_size = 1
            self.group = None

    def load_files_parallel(
        self,
        model_path: str,
        device: torch.device,
        safetensors_files: List[str],
        name_converter: Optional[Callable[[str], str]] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, Tuple[list, int]],
        Set[str],
    ]:
        """
        Load safetensors files in parallel across ranks.

        Args:
            model_path: Path to the model directory.
            device: Device to load tensors on.
            safetensors_files: List of safetensors file names.
            name_converter: Optional function to convert tensor names (e.g., for checkpoint conversion).

        Returns:
            Tuple of (rank_tensors, rank_tensor_metadata, weights_of_ckpt_names):
            - rank_tensors: Dict mapping tensor names to tensors loaded by this rank.
            - rank_tensor_metadata: Dict mapping tensor names to (shape, dtype_int) tuples.
            - weights_of_ckpt_names: Set of all tensor names found by this rank.
        """
        rank_tensors = {}  # {tensor_name: tensor_data} for this rank
        rank_tensor_metadata = {}  # {tensor_name: (shape, dtype)} for this rank
        weights_of_ckpt_names = set()

        # If COSMOS_MULTI_RANK_WEIGHT_LOADER_ON_CPU is set to 1, load tensors to CPU to avoid GPU OOM
        # Otherwise, load tensors to the specified device
        # Note: This may cause performance degradation due to CPU-GPU transfer overhead
        COSMOS_MULTI_RANK_WEIGHT_LOADER_ON_CPU = (
            os.getenv("COSMOS_MULTI_RANK_WEIGHT_LOADER_ON_CPU", "0") == "1"
        )
        loading_device = "cpu" if COSMOS_MULTI_RANK_WEIGHT_LOADER_ON_CPU else device

        with torch.device(loading_device):
            for file_idx, f in enumerate(safetensors_files):
                file_rank = file_idx % self.world_size
                if self.rank == file_rank:
                    # This rank is responsible for reading this file
                    ckpt = safe_open(
                        os.path.join(model_path, f),
                        framework="pt",
                        device=str(loading_device),
                    )
                    keys = list(ckpt.keys())
                    for name in keys:
                        ckpt_tensor = ckpt.get_tensor(name)
                        # Apply name converter if provided
                        if name_converter is not None:
                            name = name_converter(name)
                        weights_of_ckpt_names.add(name)
                        rank_tensors[name] = ckpt_tensor
                        rank_tensor_metadata[name] = (
                            ckpt_tensor.shape,
                            ckpt_tensor.dtype,
                        )
                    del ckpt

        return rank_tensors, rank_tensor_metadata, weights_of_ckpt_names

    def gather_tensor_names_and_build_mapping(
        self, weights_of_ckpt_names: Set[str], rank_tensors: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Dict[str, int]]:
        """
        Gather all tensor names from all ranks and build a tensor-to-rank mapping.

        Args:
            weights_of_ckpt_names: Set of tensor names found by this rank.
            rank_tensors: Dict of tensors loaded by this rank.

        Returns:
            Tuple of (all_tensor_names, tensor_to_rank_map):
            - all_tensor_names: Set of all tensor names across all ranks.
            - tensor_to_rank_map: Dict mapping tensor names to the rank that loaded them.
        """
        if self.world_size > 1:
            # all_gather_object requires output list to be pre-initialized with world_size
            all_tensor_names_lists = [None] * self.world_size
            dist.all_gather_object(
                all_tensor_names_lists, list(weights_of_ckpt_names), group=self.group
            )
            # Flatten the list and create a set
            all_tensor_names = set()
            for names_list in all_tensor_names_lists:
                if names_list is not None:
                    all_tensor_names.update(names_list)

            # Build tensor-to-rank mapping: gather which rank has which tensors
            # Create a dict mapping tensor_name -> rank for this rank
            local_tensor_to_rank = {name: self.rank for name in rank_tensors.keys()}
            all_tensor_to_rank_dicts = [None] * self.world_size
            dist.all_gather_object(
                all_tensor_to_rank_dicts, local_tensor_to_rank, group=self.group
            )

            # Merge all dicts to create global mapping
            tensor_to_rank_map = {}
            for rank_idx, tensor_dict in enumerate(all_tensor_to_rank_dicts):
                if tensor_dict is not None:
                    for tensor_name, _ in tensor_dict.items():
                        if tensor_name not in tensor_to_rank_map:
                            tensor_to_rank_map[tensor_name] = rank_idx
                        # If duplicate, keep the first one (shouldn't happen, but just in case)
        else:
            all_tensor_names = weights_of_ckpt_names
            tensor_to_rank_map = {name: 0 for name in rank_tensors.keys()}

        return all_tensor_names, tensor_to_rank_map

    def broadcast_tensor(
        self,
        name: str,
        tensor_rank: int,
        rank_tensors: Dict[str, torch.Tensor],
        rank_tensor_metadata: Dict[str, Tuple[list, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Broadcast a tensor from the rank that has it to all ranks.

        Args:
            name: Name of the tensor to broadcast.
            tensor_rank: Rank that has the tensor.
            rank_tensors: Dict of tensors loaded by this rank.
            rank_tensor_metadata: Dict of tensor metadata (shape, dtype) for this rank.
            device: Device to create tensors on.

        Returns:
            The broadcasted tensor (same on all ranks).
        """
        # Get tensor from the rank that has it
        if self.rank == tensor_rank:
            ckpt_tensor = rank_tensors[name]
            meta_data = rank_tensor_metadata[name]

            # Move tensor from CPU to GPU if needed (tensors are loaded to CPU to avoid OOM)
            if ckpt_tensor.device.type != device.type:
                ckpt_tensor = ckpt_tensor.to(device)
        else:
            ckpt_tensor = None
            meta_data = None

        # Broadcast tensor metadata (shape, dtype) from the rank that has it
        if self.world_size > 1:
            meta_data = broadcast_object_cpu(
                meta_data, group=self.group, group_src=tensor_rank
            )
            tensor_shape, tensor_dtype = meta_data

            if self.rank != tensor_rank:
                ckpt_tensor = torch.empty(
                    tensor_shape, dtype=tensor_dtype, device=device
                )

            # Broadcast the actual tensor data
            dist.broadcast(ckpt_tensor, group=self.group, group_src=tensor_rank)
        else:
            # Single rank case: ensure tensor is on the correct device
            if ckpt_tensor is not None and ckpt_tensor.device.type != device.type:
                ckpt_tensor = ckpt_tensor.to(device)

        # Ensure ckpt_tensor is not None
        if ckpt_tensor is None:
            raise ValueError(
                f"Failed to get tensor {name} on rank {self.rank}. "
                f"tensor_rank={tensor_rank}, world_size={self.world_size}, "
                f"group={self.group}"
            )

        return ckpt_tensor

    def iterate_tensors(
        self,
        all_tensor_names: Set[str],
        tensor_to_rank_map: Dict[str, int],
        rank_tensors: Dict[str, torch.Tensor],
        rank_tensor_metadata: Dict[str, Tuple[list, int]],
        device: torch.device,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Iterate over all tensors, broadcasting them as needed.

        Args:
            all_tensor_names: Set of all tensor names across all ranks.
            tensor_to_rank_map: Dict mapping tensor names to the rank that loaded them.
            rank_tensors: Dict of tensors loaded by this rank.
            rank_tensor_metadata: Dict of tensor metadata (shape, dtype) for this rank.
            device: Device to create tensors on.

        Yields:
            Tuple of (tensor_name, tensor) for each tensor.
        """
        for name in sorted(all_tensor_names):
            tensor_rank = tensor_to_rank_map.get(name)
            if tensor_rank is None:
                logger.error(
                    f"Tensor {name} not found in tensor_to_rank_map which is unexpected."
                )
                continue

            tensor = self.broadcast_tensor(
                name, tensor_rank, rank_tensors, rank_tensor_metadata, device
            )
            yield name, tensor
