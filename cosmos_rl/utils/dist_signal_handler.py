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

import signal
import torch


def get_world_size():
    assert torch.distributed.is_initialized(), "Torch Distributed is not initialized"
    return torch.distributed.get_world_size()


def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if "nccl" in backend:
        if local_rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{local_rank}")
    elif "gloo" in backend:
        device = torch.device("cpu")
    else:
        raise RuntimeError
    return device


def all_gather_item(
    item, dtype, group=None, async_op=False, local_rank=None
) -> list[bool]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    def __init__(self, sig: signal.Signals = signal.SIGTERM):
        self.sig = sig

        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self._signal_received = True

        signal.signal(self.sig, handler)

    def signals_received(self):
        all_received = all_gather_item(self._signal_received, dtype=torch.int32)
        return all_received

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
