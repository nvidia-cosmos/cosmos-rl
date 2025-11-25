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

from cosmos_rl.utils.parallelism import ParallelDims
import torch
from typing import Tuple, Optional


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    parallel_dims: ParallelDims,
    tp_slice_dim: Optional[int] = None,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    load_weight_test = not hasattr(parallel_dims, "mesh")
    if not load_weight_test:
        dp_shard_rank = parallel_dims.mesh[tuple(("dp_shard_cp",))].get_local_rank()
        dp_shard_size = parallel_dims.mesh[tuple(("dp_shard_cp",))].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    if tp_slice_dim is not None:
        tp_rank, tp_size = parallel_dims.tp_coord
        shard = tensor.tensor_split(tp_size, dim=tp_slice_dim)[tp_rank]
    else:
        shard = tensor

    dest_name = name
    # Do FSDP sharding
    shard = shard.contiguous()
    row_size = shard.shape[0]
    if row_size % dp_shard_size != 0:
        average_row_size = (row_size + dp_shard_size - 1) // dp_shard_size
        start_idx = dp_shard_rank * average_row_size
        end_idx = min(start_idx + average_row_size, row_size)
        shard = shard[start_idx:end_idx]
        return dest_name, shard
    else:
        shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
        return dest_name, shard.contiguous()
