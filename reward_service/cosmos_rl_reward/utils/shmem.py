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

from multiprocessing import shared_memory
import torch
import numpy as np
import msgpack
import enum
from torch.multiprocessing.reductions import rebuild_cuda_tensor
import time
from cosmos_rl_reward.utils.logging import logger


class CrossProcessStatus(enum.Enum):
    DEFAULT = 0
    READY = 1
    DECODE = 2
    INFERENCE = 3
    INITIALIZE = 4


class CrossProcessMaster:
    """
    Initialize shared memory for cross-process communication, including CUDA and CPU memory.
    Used between multiple worker processes to share tensors and information.
    """

    def __init__(self, name, cuda_maxbytes, cpu_maxbytes):
        """
        Initialize the CrossProcessMaster with shared memory for CUDA and CPU.

        Args:
            name (str): The name of the shared memory segment.
            cuda_maxbytes (int): The maximum size in bytes for the CUDA tensor.
            cpu_maxbytes (int): The maximum size in bytes for the CPU shared memory.
        """
        self.name = name
        self.cuda_maxbytes = cuda_maxbytes
        self.cpu_maxbytes = cpu_maxbytes
        self.shm_cpu = shared_memory.SharedMemory(
            create=True,
            size=cpu_maxbytes,
        )
        self.tensor_cuda = torch.empty(
            cuda_maxbytes,
            dtype=torch.uint8,
            device="cuda",
        )

        self.status = np.ndarray((1,), dtype=np.int32, buffer=self.shm_cpu.buf)
        self.status[0] = CrossProcessStatus.DEFAULT.value
        self.byte_len = np.ndarray((1,), dtype=np.int32, buffer=self.shm_cpu.buf[4:8])
        self.cpu_maxbytes = self.shm_cpu.size - 8

    def __del__(self):
        """
        Clean up the shared memory resources.
        """
        try:
            self.shm_cpu.close()
            self.shm_cpu.unlink()
        except FileNotFoundError:
            pass

    def init_memory(self):
        """
        Initialize the shared memory for the CUDA tensor.
        """
        (
            storage_device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = self.tensor_cuda.untyped_storage()._share_cuda_()
        self.shm_cuda_info = {
            "dtype": str(self.tensor_cuda.dtype),
            "tensor_size": self.tensor_cuda.size(),
            "tensor_stride": self.tensor_cuda.stride(),
            "tensor_offset": self.tensor_cuda.storage_offset(),
            "storage_cls": type(self.tensor_cuda.untyped_storage()).__name__,
            "storage_device": storage_device,
            "storage_handle": storage_handle,
            "storage_size_bytes": storage_size_bytes,
            "storage_offset_bytes": storage_offset_bytes,
            "requires_grad": self.tensor_cuda.requires_grad,
            "ref_counter_handle": ref_counter_handle,
            "ref_counter_offset": ref_counter_offset,
            "event_handle": event_handle,
            "event_sync_required": event_sync_required,
        }
        logger.info(
            f"Initializing shared memory for {self.tensor_cuda.shape}, "
            f"dtype: {self.tensor_cuda.dtype}, "
            f"device: {self.tensor_cuda.device}"
        )
        self.set_cpu_info(self.shm_cuda_info)
        self.status[0] = CrossProcessStatus.INITIALIZE.value

    def set_cuda_tensor(self, tensor):
        """
        Set the CUDA tensor in shared memory.

        Args:
            tensor (torch.Tensor): The CUDA tensor to set.
        """
        if tensor.numel() * tensor.element_size() > self.cuda_maxbytes:
            raise ValueError(
                f"Tensor size {tensor.numel() * tensor.element_size()} exceeds max size {self.cuda_maxbytes}."
            )
        self.tensor_cuda.copy_(tensor)

    def set_cpu_info(self, info):
        """
        Set the CPU info in shared memory.

        Args:
            info (dict): The information to set in shared memory.
        """
        msgpack_data = msgpack.packb(info)
        if len(msgpack_data) > self.cpu_maxbytes:
            raise ValueError(
                f"Info size {len(msgpack_data)} exceeds max size {self.cpu_maxbytes}."
            )
        self.byte_len[0] = len(msgpack_data)
        self.shm_cpu.buf[8 : 8 + len(msgpack_data)] = msgpack_data


class CrossProcessHandler:
    """
    Manages shared memory for cross-process communication from a worker process.
    Rely on the shared memory initialized by CrossProcessMaster.
    Multiple CrossProcessHandler can be created to share the same memory segment initialized by a CrossProcessMaster.
    Decode and reward calculation processes hold its own CrossProcessHandler instance to share the memory created from a CrossProcessMaster.
    Used to read and write tensors and information in shared memory.
    Controls the state of the shared memory communication.
    Controls synchronization between the decode and reward calculation inference processes.
    """

    def __init__(self, name, shm_name, index, total_handler):
        """
        Initialize the CrossProcessHandler with shared memory for CUDA and CPU.

        Args:
            name (str): The name of the shared memory segment.
            shm_name (str): The name of the CPU shared memory segment.
            index (int): The index of this handler.
            total_handler (int): The total number of handlers.
        """
        self.name = name
        self.shm_cpu = shared_memory.SharedMemory(
            name=shm_name,
        )
        self.status = np.ndarray((1,), dtype=np.int32, buffer=self.shm_cpu.buf)
        self.byte_len = np.ndarray((1,), dtype=np.int32, buffer=self.shm_cpu.buf[4:8])
        self.cpu_maxbytes = self.shm_cpu.size - 8
        self.total_handler = total_handler
        self.index = index

    def set_cuda_tensor(self, tensor):
        """
        Set the CUDA tensor in shared memory.
        Args:
            tensor (torch.Tensor): The CUDA tensor to set.
        """
        if tensor.numel() * tensor.element_size() > self.cuda_maxbytes:
            raise ValueError(
                f"Tensor size {tensor.numel() * tensor.element_size()} exceeds max size {self.cuda_maxbytes}."
            )
        tensor = tensor.contiguous().view(self.tensor_cuda.dtype).reshape(-1)
        self.tensor_cuda[: tensor.numel()].copy_(tensor)

    def set_cpu_info(self, info):
        """
        Set the CPU info in shared memory.

        Args:
            info (dict): The information to set in shared memory.
        """
        msgpack_data = msgpack.packb(info)
        if len(msgpack_data) > self.cpu_maxbytes:
            raise ValueError(
                f"Info size {len(msgpack_data)} exceeds max size {self.cpu_maxbytes}."
            )
        self.byte_len[0] = len(msgpack_data)
        self.shm_cpu.buf[8 : 8 + len(msgpack_data)] = msgpack_data

    def get_data_info(self):
        """
        Get the data information from shared memory.

        Returns:
            dict: The unpacked data information.
        """
        data_info = msgpack.unpackb(self.shm_cpu.buf[8 : 8 + self.byte_len[0]])
        return data_info

    def init_memory(self):
        """
        Initialize the shared memory for CUDA and CPU.
        """
        while self.status[0] != (CrossProcessStatus.INITIALIZE.value + self.index):
            time.sleep(0.001)
        self.shm_cuda_info = self.get_data_info()
        if isinstance(self.shm_cuda_info["storage_cls"], str):
            self.shm_cuda_info["storage_cls"] = getattr(
                torch.storage, self.shm_cuda_info["storage_cls"]
            )
        if isinstance(self.shm_cuda_info["dtype"], str):
            if self.shm_cuda_info["dtype"].startswith("torch."):
                self.shm_cuda_info["dtype"] = getattr(
                    torch, self.shm_cuda_info["dtype"].split(".")[-1]
                )
        logger.debug(f"Rebuilding CUDA tensor with info: {self.shm_cuda_info}")
        self.tensor_cuda = rebuild_cuda_tensor(torch.Tensor, **self.shm_cuda_info)
        logger.info(
            f"Rebuilt CUDA tensor from shared memory: {self.tensor_cuda.shape}, "
            f"dtype: {self.tensor_cuda.dtype}, "
            f"device: {self.tensor_cuda.device}, "
            f"index: {self.index}, total: {self.total_handler}"
        )
        self.cuda_maxbytes = self.tensor_cuda.numel() * self.tensor_cuda.element_size()
        if self.index + 1 == self.total_handler:
            self.status[0] = CrossProcessStatus.DECODE.value
        else:
            self.status[0] = CrossProcessStatus.INITIALIZE.value + self.index + 1

    def wait_tensor_in_inference(self):
        """
        Wait for the tensor to be ready from decoding in the reward calculation inference stage.

        Returns:
            torch.Tensor: The tensor from shared memory.
            dict: The associated information.
        """
        while self.status[0] != CrossProcessStatus.INFERENCE.value:
            time.sleep(0.001)
        info = self.get_data_info()
        tensor_info = info.get("decoded_info", {})
        assert (
            "dtype" in tensor_info and "shape" in tensor_info
        ), "Tensor info must contain 'dtype' and 'shape'."
        if isinstance(tensor_info["dtype"], str):
            if tensor_info["dtype"].startswith("torch."):
                dtype = getattr(torch, tensor_info["dtype"].split(".")[-1])
            else:
                dtype = getattr(torch, tensor_info["dtype"])
        else:
            dtype = tensor_info["dtype"]
        size = (
            dtype.itemsize
            * np.prod(tensor_info["shape"])
            // self.tensor_cuda.element_size()
        )
        tensor = (
            self.tensor_cuda[:size].view(dtype=dtype).reshape(*tensor_info["shape"])
        )
        return tensor, info

    def set_tensor_in_decode(self, tensor, info):
        """
        Set the tensor and associated information in the decode stage after decoding.

        Args:
            tensor (torch.Tensor): The tensor to set in shared memory.
            info (dict): The associated information.
        """
        assert (
            self.status[0] == CrossProcessStatus.DECODE.value
        ), f"Expected status {CrossProcessStatus.DECODE.value}, but got {self.status[0]}."
        if tensor.numel() * tensor.element_size() > self.cuda_maxbytes:
            needed = tensor.numel() * tensor.element_size()
            raise ValueError(
                "Tensor size "
                f"{needed} exceeds shared CUDA buffer size {self.cuda_maxbytes}. "
                "Increase the shared buffer and restart the reward service, e.g. set "
                "COSMOS_RL_REWARD_SHMEM_CUDA_MAXBYTES (bytes)."
            )
        self.set_cuda_tensor(tensor)
        self.set_cpu_info(info)
        self.status[0] = CrossProcessStatus.INFERENCE.value

    def set_finish_in_inference(self):
        """
        Set the finish status in the reward calculation inference stage after calculation.
        """
        assert (
            self.status[0] == CrossProcessStatus.INFERENCE.value
        ), f"Expected status {CrossProcessStatus.INFERENCE.value}, but got {self.status[0]}."
        self.status[0] = CrossProcessStatus.DECODE.value

    def wait_start_in_decode(self):
        """
        Wait for the decode stage to start when the memory is ready.
        """
        while self.status[0] != CrossProcessStatus.DECODE.value:
            time.sleep(0.001)
