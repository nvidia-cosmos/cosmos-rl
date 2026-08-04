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

import unittest

import torch
import torch.multiprocessing as mp
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_send,
    nccl_recv,
    nccl_broadcast,
    nccl_allreduce,
)
from cosmos_rl.utils.pynccl_wrapper import ncclRedOpTypeEnum


def setup_nccl_comm(rank, world_size, nccl_uid):
    """Setup NCCL communicator for a process."""
    # Set device for this process (now using the visible device)
    torch.cuda.set_device(rank)  # Always use first visible device
    # Create NCCL communicator
    comm_idx = create_nccl_comm(nccl_uid, rank, world_size)
    return comm_idx


# Dtypes exercised by the movement collectives.  bool and int16 have no entry in
# ncclDataTypeEnum -- NCCL has no int16 type at all -- so they pass only because
# broadcast/send/recv are issued as byte counts.
MOVEMENT_DTYPES = [
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.int8,
    torch.bfloat16,
    torch.float64,
    torch.bool,
    torch.int16,
]

# Reductions stay typed, so bool/int16 are excluded here by design.
REDUCTION_DTYPES = MOVEMENT_DTYPES[:8]


def make_payload(dtype, size, device, seed):
    """Deterministic payload of `dtype` that varies with `seed`."""
    if dtype == torch.bool:
        # Must be a *mixed* pattern: an all-False payload is indistinguishable
        # from a buffer that was never written, which would make the assertion
        # vacuous.  The +seed shifts the pattern so different senders differ.
        return (torch.arange(size, device=device) + seed) % 3 == 0
    return torch.arange(size, dtype=dtype, device=device) * seed


def assert_payload_eq(actual, expected, message):
    """Exact comparison -- movement collectives copy bytes, so nothing drifts."""
    assert torch.equal(actual, expected), message


class TestNCCLBidirectionalSendRecv(unittest.TestCase):
    @staticmethod
    def run_bidirectional_sender(rank, world_size, nccl_uid, dtypes):
        """Run sender part of bidirectional NCCL send/recv test."""
        comm_idx = setup_nccl_comm(rank, world_size, nccl_uid)

        for dtype in dtypes:
            # Create test tensor
            tensor_size = 1000
            device = f"cuda:{rank}"
            send_tensor = make_payload(dtype, tensor_size, device, rank + 1)
            recv_tensor = torch.zeros(tensor_size, dtype=dtype, device=device)

            send_rank = 0
            # Send to other rank and receive from other rank
            other_rank = 1 - rank
            if rank == send_rank:
                nccl_send(send_tensor, other_rank, comm_idx)
                nccl_recv(recv_tensor, other_rank, comm_idx)
            else:
                nccl_recv(recv_tensor, other_rank, comm_idx)
                nccl_send(recv_tensor, other_rank, comm_idx)

            # Verify received data: rank 0's payload is echoed back by rank 1.
            expected = make_payload(dtype, tensor_size, device, send_rank + 1)
            assert_payload_eq(
                recv_tensor, expected, f"send/recv failed for dtype {dtype}"
            )

    def test_nccl_bidirectional_send_recv(self):
        """Test bidirectional NCCL send/recv operations between two processes with different CUDA devices."""
        world_size = 2

        # Create NCCL unique ID
        nccl_uid = create_nccl_uid()

        # Define functions for each process (same function but different rank)
        functions = self.run_bidirectional_sender
        # Spawn processes with different functions
        dtypes = MOVEMENT_DTYPES
        mp.spawn(
            functions,
            args=(world_size, nccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )


class TestNCCLBroadcast(unittest.TestCase):
    @staticmethod
    def run_broadcast(rank, world_size, nccl_uid, dtypes):
        """Run broadcast test for different data types."""
        comm_idx = setup_nccl_comm(rank, world_size, nccl_uid)

        for dtype in dtypes:
            # Test broadcasting from each rank
            for root_rank in range(world_size):
                # Create test tensor
                tensor_size = 1000
                device = f"cuda:{rank}"
                if rank == root_rank:  # Root rank
                    # Create tensor with unique values based on root rank
                    tensor = make_payload(dtype, tensor_size, device, root_rank + 1)
                else:
                    # Create empty tensor for receiving
                    tensor = torch.zeros(tensor_size, dtype=dtype, device=device)

                # Perform broadcast from current root rank
                nccl_broadcast(tensor, root_rank, comm_idx)

                # Verify received data
                expected = make_payload(dtype, tensor_size, device, root_rank + 1)
                assert_payload_eq(
                    tensor,
                    expected,
                    f"Broadcast from rank {root_rank} failed for dtype {dtype}",
                )

    def test_nccl_broadcast(self):
        """Test NCCL broadcast operations between multiple processes with different CUDA devices."""
        world_size = 4

        # Create NCCL unique ID
        nccl_uid = create_nccl_uid()

        # Define data types to test
        dtypes = MOVEMENT_DTYPES

        # Spawn processes
        mp.spawn(
            self.run_broadcast,
            args=(world_size, nccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )


class TestNCCLAllreduce(unittest.TestCase):
    @staticmethod
    def run_allreduce(rank, world_size, nccl_uid, dtypes):
        """Run allreduce test for different data types."""
        comm_idx = setup_nccl_comm(rank, world_size, nccl_uid)

        for dtype in dtypes:
            # Create test tensor
            tensor_size = 1000
            # Each rank creates a tensor with its rank value
            tensor = torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * (
                rank + 1
            )
            op = ncclRedOpTypeEnum.from_torch(torch.distributed.ReduceOp.SUM)
            # Perform allreduce (sum)
            nccl_allreduce(tensor, tensor, op, comm_idx)

            # Verify result
            # For 4 ranks, the sum should be 1 + 2 + 3 + 4 = 10
            expected_sum = (
                torch.ones(tensor_size, dtype=dtype, device=f"cuda:{rank}") * 10
            )
            assert torch.allclose(tensor, expected_sum)

    def test_nccl_allreduce(self):
        """Test NCCL allreduce operations between multiple processes with different CUDA devices."""
        world_size = 4

        # Create NCCL unique ID
        nccl_uid = create_nccl_uid()

        # Define data types to test
        dtypes = REDUCTION_DTYPES

        # Spawn processes
        mp.spawn(
            self.run_allreduce,
            args=(world_size, nccl_uid, dtypes),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
