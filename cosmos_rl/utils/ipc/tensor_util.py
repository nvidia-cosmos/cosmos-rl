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
from torch.multiprocessing.reductions import reduce_tensor
from typing import Any, Dict


def tensor_ipc_serialize(tensor: torch.Tensor) -> Any:
    """
    Convert the tensor to the IPC handle.
    """
    return reduce_tensor(tensor)


def tensor_ipc_deserialize(ipc_data: Any) -> torch.Tensor:
    """
    Convert the IPC handle to the tensor.

    Args:
        ipc_data: the IPC data returned by tensor_ipc_serialize

    Returns:
        original tensor shared same physical memory in same device
    """
    rebuild_func, args = ipc_data
    return rebuild_func(*args)


def named_tensors_to_serialize(
    named_tensors: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Convert the state dict to the IPC data.

    Args:
        named_tensors: the named tensors to convert

    Returns:
        the IPC data
    """
    state_dict_ipc = {}
    for name, tensor in named_tensors.items():
        state_dict_ipc[name] = tensor_ipc_serialize(tensor)
    return state_dict_ipc


def named_tensors_from_serialize(
    named_tensors_ipc: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Convert the named tensors IPC to the named tensors.

    Args:
        named_tensors_ipc: the IPC data returned by named_tensors_to_serialize

    Returns:
        the named tensors
    """
    named_tensors = {}
    for name, ipc_data in named_tensors_ipc.items():
        named_tensors[name] = tensor_ipc_deserialize(ipc_data)
    return named_tensors
