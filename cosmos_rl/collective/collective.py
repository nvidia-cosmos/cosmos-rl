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


import zmq
import os
import torch
from abc import ABC, abstractmethod
from typing import Optional
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_send,
    nccl_recv,
)
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
from cosmos_rl.utils import network_util as net
from cosmos_rl.utils.ipc.tensor_util import tensor_ipc_serialize, tensor_ipc_deserialize


class _P2PChannelBase(ABC):
    """A fixed one-way point-to-point channel.

    The channel always contains exactly two participants:
    rank 0 is the sender side, and rank 1 is the receiver side.
    """

    def __init__(
        self,
        channel_key: str,
        *,
        local_channel_rank: int,
        replica_name: str,
        role_name: str,
    ):
        self.channel_key = channel_key
        self.local_channel_rank = local_channel_rank
        self.replica_name = replica_name
        self.role_name = role_name
        self.send_peer = 1
        self.recv_peer = 0
        self.is_sender = local_channel_rank == 0

    @abstractmethod
    def send(self, tensor: torch.Tensor):
        pass

    @abstractmethod
    def recv(self, tensor: torch.Tensor):
        pass

    def close(self):
        pass

    @property
    def comm_index(self) -> Optional[int]:
        return None


class _NCCLP2PChannelBase(_P2PChannelBase):
    """A fixed one-way NCCL point-to-point channel."""

    def __init__(
        self,
        channel_key: str,
        nccl_unique_id,
        *,
        stream: torch.cuda.Stream,
        local_channel_rank: int,
        replica_name: str,
        role_name: str,
    ):
        super().__init__(
            channel_key,
            local_channel_rank=local_channel_rank,
            replica_name=replica_name,
            role_name=role_name,
        )
        self.stream = stream
        self._comm_index: Optional[int] = None
        self._nccl_unique_id = nccl_unique_id

        self._setup_communicator()

    def _setup_communicator(self):
        if self._comm_index is not None:
            return

        if self._nccl_unique_id is None:
            raise RuntimeError(
                f"[{self.role_name}] Missing nccl_unique_id for channel {self.channel_key}"
            )

        self._comm_index = create_nccl_comm(
            self._nccl_unique_id,
            self.local_channel_rank,
            2,
        )
        logger.info(
            "[%s] Created NCCL communicator for point-to-point channel %s.",
            self.role_name,
            self.channel_key,
        )

    @property
    def comm_index(self) -> int:
        if self._comm_index is None:
            raise RuntimeError(
                f"[{self.role_name}] NCCL communicator for channel {self.channel_key} is not initialized."
            )
        return self._comm_index


class NCCLChannel(_NCCLP2PChannelBase):
    """A fixed one-way NCCL point-to-point channel."""

    def __init__(
        self,
        channel_key: str,
        nccl_unique_id,
        replica_name: str,
        *,
        is_sender: bool,
        stream: torch.cuda.Stream,
    ):
        super().__init__(
            channel_key,
            nccl_unique_id,
            stream=stream,
            local_channel_rank=0 if is_sender else 1,
            replica_name=replica_name,
            role_name=(
                f"NCCLSendChannel:{replica_name}"
                if is_sender
                else f"NCCLRecvChannel:{replica_name}"
            ),
        )

    def send(self, tensor: torch.Tensor):
        if not self.is_sender:
            raise RuntimeError(
                f"[{self.role_name}] Recv-only channel does not support send."
            )
        nccl_send(tensor, self.send_peer, self.comm_index, stream=self.stream)

    def recv(self, tensor: torch.Tensor):
        if self.is_sender:
            raise RuntimeError(
                f"[{self.role_name}] Send-only channel does not support recv."
            )
        nccl_recv(tensor, self.recv_peer, self.comm_index, stream=self.stream)


class _IPCP2PChannelBase(_P2PChannelBase):
    """A fixed one-way IPC point-to-point channel."""

    def __init__(
        self,
        channel_key: str,
        zmq_context: zmq.Context,
        ipc_addr: str,
        *,
        local_channel_rank: int,
        replica_name: str,
        role_name: str,
    ):
        super().__init__(
            channel_key,
            local_channel_rank=local_channel_rank,
            replica_name=replica_name,
            role_name=role_name,
        )
        self.zmq_context = zmq_context
        self.ipc_addr = ipc_addr
        self._socket: Optional[zmq.Socket] = None

        self._setup_socket()

    def _setup_socket(self):
        if self._socket is not None:
            return

        if not self.ipc_addr:
            raise RuntimeError(
                f"[{self.role_name}] Missing ipc_addr for channel {self.channel_key}"
            )

        socket = self.zmq_context.socket(zmq.PAIR)
        if self.local_channel_rank == 0:
            socket.bind(self.ipc_addr)
        else:
            socket.connect(self.ipc_addr)
        self._socket = socket

        logger.info(
            "[%s] Created IPC socket for point-to-point channel %s.",
            self.role_name,
            self.channel_key,
        )

    @property
    def socket(self) -> zmq.Socket:
        if self._socket is None:
            raise RuntimeError(
                f"[{self.role_name}] IPC socket for channel {self.channel_key} is not initialized."
            )
        return self._socket

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class IPCChannel(_IPCP2PChannelBase):
    """A fixed one-way IPC point-to-point channel."""

    def __init__(
        self,
        channel_key: str,
        zmq_context: zmq.Context,
        ipc_addr: str,
        replica_name: str,
        *,
        is_sender: bool,
    ):
        super().__init__(
            channel_key,
            zmq_context,
            ipc_addr,
            local_channel_rank=0 if is_sender else 1,
            replica_name=replica_name,
            role_name=(
                f"IPCSendChannel:{replica_name}"
                if is_sender
                else f"IPCRecvChannel:{replica_name}"
            ),
        )

    def send(self, tensor: torch.Tensor):
        if not self.is_sender:
            raise RuntimeError(
                f"[{self.role_name}] Recv-only channel does not support send."
            )
        self.socket.send_pyobj(tensor_ipc_serialize(tensor))

    def recv(self, tensor: torch.Tensor):
        if self.is_sender:
            raise RuntimeError(
                f"[{self.role_name}] Send-only channel does not support recv."
            )
        ipc_data = self.socket.recv_pyobj()
        tensor.copy_(tensor_ipc_deserialize(ipc_data).cuda())


class P2RCollectiveManager:
    """Point-to-point P2R communication manager for one policy/rollout rank pair."""

    def __init__(
        self,
        replica_name: str,
        parallel_dims: ParallelDims,
        config: CosmosConfig,
        api_client: APIClient,
        role: Role,
        command: PolicyToRolloutUnicastCommand,
        policy_rank: int,
        rollout_rank: int,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        self.config = config
        self.replica_name = replica_name
        self.parallel_dims = parallel_dims
        self.world_size = parallel_dims.world_size
        self.api_client = api_client
        self.role = role
        self.rl_mode = config.mode
        self.command = command
        self.policy_rank = policy_rank
        self.rollout_rank = rollout_rank

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))

        self.zmq_context: Optional[zmq.Context] = None
        self.channel: Optional[_P2PChannelBase] = None
        self.stream: Optional[torch.cuda.Stream] = stream

        self._validate_pair_membership()
        self._setup_backend()

    def __del__(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None
        if self.zmq_context is not None:
            self.zmq_context.term()
            self.zmq_context = None

    @property
    def base_mesh_key(self) -> str:
        return self.command.src_replica_name + "_" + self.command.dst_replica_name

    @property
    def pair_mesh_key(self) -> str:
        return self.base_mesh_key + f"_{self.policy_rank}_{self.rollout_rank}"

    @property
    def pair_ipc_key(self) -> str:
        return self.pair_mesh_key + "_ipc"

    @property
    def uses_ipc(self) -> bool:
        return (
            self.rl_mode == "colocated_separated"
            and self.policy_rank == self.rollout_rank
        )

    def _validate_pair_membership(self):
        if self.role == Role.ROLLOUT:
            if self.command.dst_replica_name != self.replica_name:
                raise RuntimeError(
                    f"[Rollout] Replica {self.replica_name} doesn't match command destination: {self.command.dst_replica_name}"
                )
            if self.global_rank != self.rollout_rank:
                raise RuntimeError(
                    f"[Rollout] Local global rank {self.global_rank} must match rollout_rank {self.rollout_rank} for pair manager."
                )
        else:
            if self.command.src_replica_name != self.replica_name:
                raise RuntimeError(
                    f"[Policy] Replica {self.replica_name} doesn't match command source: {self.command.src_replica_name}"
                )
            if self.global_rank != self.policy_rank:
                raise RuntimeError(
                    f"[Policy] Local global rank {self.global_rank} must match policy_rank {self.policy_rank} for pair manager."
                )

    def _setup_backend(self):
        self.channel = (
            self._build_ipc_channel() if self.uses_ipc else self._build_nccl_channel()
        )

    def _build_nccl_channel(self) -> _P2PChannelBase:
        if self.stream is None:
            self.stream = torch.cuda.Stream()

        if self.role == Role.ROLLOUT:
            nccl_unique_id = self.api_client.post_nccl_comm_acceptor(self.pair_mesh_key)
            if nccl_unique_id is None:
                raise RuntimeError(
                    f"[Rollout] Failed to query nccl unique id from controller for {self.pair_mesh_key}"
                )
            return NCCLChannel(
                channel_key=self.pair_mesh_key,
                nccl_unique_id=nccl_unique_id,
                replica_name=self.replica_name,
                is_sender=False,
                stream=self.stream,
            )
        nccl_unique_id = create_nccl_uid()
        self.api_client.post_nccl_comm_initiator(self.pair_mesh_key, nccl_unique_id)
        return NCCLChannel(
            channel_key=self.pair_mesh_key,
            nccl_unique_id=nccl_unique_id,
            replica_name=self.replica_name,
            is_sender=True,
            stream=self.stream,
        )

    def _build_ipc_channel(self) -> _P2PChannelBase:
        self.zmq_context = zmq.Context()

        if self.role == Role.ROLLOUT:
            ipc_addr = self.api_client.query_ipc_info(self.pair_ipc_key)
            return IPCChannel(
                channel_key=self.pair_ipc_key,
                zmq_context=self.zmq_context,
                ipc_addr=ipc_addr,
                replica_name=self.replica_name,
                is_sender=False,
            )
        local_ip = net.get_local_ip()[0]
        port_range = 65535 - 23000
        base_port_offset = port_range // self.world_size
        start_port = 23000 + base_port_offset * self.policy_rank
        end_port = start_port + base_port_offset
        free_port = net.find_available_port(start_port, end_port)
        ipc_addr = f"tcp://{local_ip}:{free_port}"
        self.api_client.post_ipc_info(self.pair_ipc_key, ipc_addr)
        return IPCChannel(
            channel_key=self.pair_ipc_key,
            zmq_context=self.zmq_context,
            ipc_addr=ipc_addr,
            replica_name=self.replica_name,
            is_sender=True,
        )

    def query_nccl_comm_index(self) -> int:
        if self.channel is None or self.channel.comm_index is None:
            raise ValueError(
                f"NCCL communicator index not found for pair key: {self.pair_mesh_key}"
            )
        return self.channel.comm_index

    def send(self, tensor: torch.Tensor):
        assert self.role == Role.POLICY, "Only policy can send data."
        assert self.channel is not None, (
            f"Channel is not initialized for pair key: {self.pair_mesh_key}"
        )
        self.channel.send(tensor)

    def recv(self, tensor: torch.Tensor):
        assert self.role == Role.ROLLOUT, "Only rollout can receive data."
        assert self.channel is not None, (
            f"Channel is not initialized for pair key: {self.pair_mesh_key}"
        )
        self.channel.recv(tensor)
