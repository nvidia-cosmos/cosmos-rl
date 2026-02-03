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

import cosmos_rl.utils.distributed as dist_util


class P2RCollectiveManager:
    """
    Send and Recv operations for Policy to Rollout communication.
    """

    def __init__(
        self,
        replica_name: str,
        parallel_dims: ParallelDims,
        config: CosmosConfig,
        api_client: APIClient,
        role: Role,
    ):
        """
        Initialize the CollectiveManager.
        """
        self.config = config
        self.replica_name = replica_name
        self.parallel_dims = parallel_dims
        self.world_size = parallel_dims.world_size
        self.api_client = api_client
        self.role = role
        self.rl_mode = config.mode

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))

        # Shared ZMQ context
        self.zmq_context = zmq.Context()

        self.rl_mode = config.mode

        # UniqueIds cache
        self.unique_ids_cache = {}
        # nccl comm index cache
        self.nccl_comm_cache = {}

        # ipc socket cache
        self.ipc_comm_cache = {}

    def _setup_inter_replica_communicators(
        self, command: PolicyToRolloutUnicastCommand
    ):
        # init replica to replica communicators
        mesh_key = self.generate_mesh_key(command)
        nccl_unique_id = None
        if self.role != Role.ROLLOUT:
            # policy initialization
            assert (
                command.src_replica_size == self.world_size
            ), "The source replica size should be the same as the world size."
            if not command.src_replica_name == self.replica_name:
                raise RuntimeError(
                    f"[Policy] Replica {self.replica_name} doesn't match command source: {command.src_replica_name}"
                )
            # create the communication group ID
            if mesh_key not in self.unique_ids_cache:
                if self.global_rank == 0:
                    nccl_unique_id = create_nccl_uid()
                    logger.debug(f"[Policy] Creating nccl group id for {mesh_key}")
                    self.api_client.post_nccl_comm_initiator(mesh_key, nccl_unique_id)

                # broadcast the nccl group id to all ranks
                nccl_unique_id = dist_util.broadcast_object_cpu(nccl_unique_id)
                self.unique_ids_cache[mesh_key] = nccl_unique_id
            else:
                nccl_unique_id = self.unique_ids_cache[mesh_key]
        else:
            # rollout initialization
            if command.dst_replica_name != self.replica_name:
                raise RuntimeError(
                    f"[Rollout] Replica {self.replica_name} doesn't match command destionation: {command.dst_replica_name}"
                )
            if mesh_key not in self.unique_ids_cache:
                # query the nccl group id from controller
                nccl_unique_id = self.api_client.post_nccl_comm_acceptor(mesh_key)
                if nccl_unique_id is None:
                    raise RuntimeError(
                        f"[Rollout] Failed to query nccl group_id from controller for {mesh_key}"
                    )
                self.unique_ids_cache[mesh_key] = nccl_unique_id
            else:
                nccl_unique_id = self.unique_ids_cache[mesh_key]

        if self.role == Role.ROLLOUT:
            group_size = self.world_size + command.src_replica_size
            rank_in_group = self.global_rank + command.src_replica_size
        else:
            group_size = self.world_size + command.dst_replica_size
            rank_in_group = self.global_rank
        # create the nccl communicator
        if mesh_key not in self.nccl_comm_cache:
            nccl_comm_index = create_nccl_comm(
                nccl_unique_id,
                rank_in_group,
                group_size,
            )
            self.nccl_comm_cache[mesh_key] = nccl_comm_index
            logger.info(
                f"Creating nccl communicator for {mesh_key} in {self.role} side."
            )

    def _setup_p2p_communicators(self, command: PolicyToRolloutUnicastCommand):
        # init p2p communicators, in colocated separated mode, policy and rollout shares the same devices.
        if self.role != Role.ROLLOUT:
            if command.src_replica_name != self.replica_name:
                raise RuntimeError(
                    f"[Policy] Replica {self.replica_name} doesn't match command source: {command.src_replica_name}"
                )
            # policy
            p_rank = self.global_rank
            for r_rank in range(command.dst_replica_size):
                if p_rank != r_rank:
                    mesh_key = self.generate_mesh_key(
                        command, p_rank, r_rank, is_p2p=True
                    )
                    if mesh_key not in self.unique_ids_cache:
                        # create p2p unique id for each non-same device pair
                        p2p_unique_id = create_nccl_uid()
                        logger.debug(f"[Policy] Creating nccl unique id for {mesh_key}")
                        self.unique_ids_cache[mesh_key] = p2p_unique_id
                        self.api_client.post_nccl_comm_initiator(
                            mesh_key, p2p_unique_id
                        )

                    # create communicator for each non-same device pair
                    if mesh_key not in self.nccl_comm_cache:
                        nccl_comm_index = create_nccl_comm(
                            p2p_unique_id,
                            0,  # policy rank is always 0
                            2,  # group size of two devices is always 2
                        )
                        self.nccl_comm_cache[mesh_key] = nccl_comm_index
                        logger.debug(
                            f"[Policy] Creating nccl communicator for {mesh_key}"
                        )
        else:
            # rollout
            if command.dst_replica_name != self.replica_name:
                raise RuntimeError(
                    f"[Rollout] Replica {self.replica_name} doesn't match command destination: {command.dst_replica_name}"
                )
            r_rank = self.global_rank
            for p_rank in range(command.src_replica_size):
                if r_rank != p_rank:
                    mesh_key = self.generate_mesh_key(
                        command, p_rank, r_rank, is_p2p=True
                    )
                    if mesh_key not in self.unique_ids_cache:
                        nccl_unique_id = self.api_client.post_nccl_comm_acceptor(
                            mesh_key
                        )
                        if nccl_unique_id is None:
                            raise RuntimeError(
                                f"[Rollout] Failed to query nccl group_id from controller for {mesh_key}"
                            )
                        self.unique_ids_cache[mesh_key] = nccl_unique_id
                    else:
                        nccl_unique_id = self.unique_ids_cache[mesh_key]

                    if mesh_key not in self.nccl_comm_cache:
                        nccl_comm_index = create_nccl_comm(
                            nccl_unique_id,
                            1,  # rollout rank is always 1
                            2,  # group size of two devices is always 2
                        )
                        self.nccl_comm_cache[mesh_key] = nccl_comm_index
                        logger.debug(
                            f"[Rollout] Creating nccl communicator for {mesh_key}"
                        )

    def _setup_nccl(self, command: PolicyToRolloutUnicastCommand):
        """
        Arguments:
            command: PolicyToRolloutUnicastCommand
            The command is used to initialize the nccl communicator. It contains the source(Policy) and destination(Rollout) replica names.
        """

        if self.rl_mode == "colocated_separated":
            self._setup_p2p_communicators(command)
        else:
            self._setup_inter_replica_communicators(command)

    def query_nccl_comm_index(self, mesh_key: str):
        """
        Query the nccl communicator index for a given mesh key.
        """
        if mesh_key not in self.nccl_comm_cache:
            raise ValueError(
                f"NCCL communicator index not found for mesh key: {mesh_key}"
            )
        return self.nccl_comm_cache[mesh_key]

    def _setup_ipc(self, command: PolicyToRolloutUnicastCommand):
        if self.rl_mode != "colocated_separated":
            return

        if self.role != Role.ROLLOUT:
            # Policy initialization
            if not command.src_replica_name == self.replica_name:
                raise RuntimeError(
                    f"[Policy] Replica {self.replica_name} doesn't match command source: {command.src_replica_name}"
                )
            # Each pair of policy and rollout share the same device will have one IPC socket.
            p_rank = self.global_rank
            r_rank = self.global_rank
            mesh_key = self.generate_mesh_key(command, p_rank, r_rank, is_ipc=True)
            if mesh_key not in self.ipc_comm_cache:
                logger.info(
                    f"Setting up IPC for {self.role} side with mode {self.rl_mode}"
                )
                local_ip = net.get_local_ip()[0]
                # To avoid conflict with other processes, we use a fixed port range.
                port_range = 65535 - 23000
                base_port_offset = port_range // self.world_size
                start_port = 23000 + base_port_offset * p_rank
                end_port = start_port + base_port_offset
                free_port = net.find_available_port(start_port, end_port)

                ipc_addr = f"tcp://{local_ip}:{free_port}"
                # create zmq socket
                socket = self.zmq_context.socket(zmq.PAIR)
                socket.bind(ipc_addr)
                self.ipc_comm_cache[mesh_key] = socket

                # Post ipc address to controller
                self.api_client.post_ipc_info(mesh_key, ipc_addr)
        else:
            # Rollout initialization
            if not command.dst_replica_name == self.replica_name:
                raise RuntimeError(
                    f"[Rollout] Replica {self.replica_name} doesn't match command destination: {command.dst_replica_name}"
                )
            r_rank = self.global_rank
            p_rank = self.global_rank
            mesh_key = self.generate_mesh_key(command, p_rank, r_rank, is_ipc=True)
            if mesh_key not in self.ipc_comm_cache:
                logger.info(
                    f"Setting up IPC for {self.role} side with mode {self.rl_mode}"
                )
                # query ipc addr from controller
                ipc_addr = self.api_client.query_ipc_info(mesh_key)
                # create zmq socket
                socket = self.zmq_context.socket(zmq.PAIR)
                socket.connect(ipc_addr)
                self.ipc_comm_cache[mesh_key] = socket

    def setup_manager(self, command: PolicyToRolloutUnicastCommand):
        # Setup NCCL
        self._setup_nccl(command)
        # Setup IPC
        self._setup_ipc(command)

    def __del__(self):
        # close the zmq context
        for socket in self.ipc_comm_cache.values():
            socket.close()
        self.ipc_comm_cache.clear()

        if self.zmq_context is not None:
            self.zmq_context.term()
            self.zmq_context = None

    def generate_mesh_key(
        self,
        base_mesh_key_or_command: PolicyToRolloutUnicastCommand | str,
        p_rank: Optional[int] = None,
        r_rank: Optional[int] = None,
        is_p2p: bool = False,
        is_ipc: bool = False,
    ):
        """
        Generate the mesh key for the collective communication.
        Arguments:
            base_mesh_key_or_command: PolicyToRolloutUnicastCommand | str If it is a str, it should be in format of "{src_replica_name}_{dst_replica_name}"
            p_rank: Optional[int] The rank of the policy
            r_rank: Optional[int] The rank of the rollout
            is_p2p: bool Whether the communication is a p2p communication in colocated separated mode
            is_ipc: bool Whether the communication is a ipc communication in colocated separated mode
        """
        if isinstance(base_mesh_key_or_command, PolicyToRolloutUnicastCommand):
            base_mesh_key = (
                base_mesh_key_or_command.src_replica_name
                + "_"
                + base_mesh_key_or_command.dst_replica_name
            )
        else:
            base_mesh_key = base_mesh_key_or_command

        if is_p2p:
            return base_mesh_key + f"_{p_rank}_{r_rank}"

        if is_ipc:
            return base_mesh_key + f"_{p_rank}_{r_rank}_ipc"

        return base_mesh_key

    def send(self, mesh_key: str, tensor: torch.Tensor, r_rank: int):
        """
        Send data to a peer.
        Arguments:
            mesh_key: str The mesh key shoule in format of "{src_replica_name}_{dst_replica_name}"
            tensor: torch.Tensor
            r_rank: int
        """
        assert self.role == Role.POLICY, "Only policy can send data."

        if self.rl_mode == "colocated_separated":
            # in colocated separated mode, we use two ways to send data:
            # 1. for the same device, we use IPC
            # 2. for the different device, we use NCCL
            p_rank = self.global_rank
            if p_rank == r_rank:
                # for the same device, we use IPC
                ipc_mesh_key = self.generate_mesh_key(
                    mesh_key, p_rank, r_rank, is_ipc=True
                )
                assert (
                    ipc_mesh_key in self.ipc_comm_cache
                ), "IPC socket not found for mesh key: {ipc_mesh_key}"
                socket = self.ipc_comm_cache[ipc_mesh_key]
                ipc_data = tensor_ipc_serialize(tensor)
                socket.send_pyobj(ipc_data)
            else:
                # for the different device, we use NCCL
                nccl_mesh_key = self.generate_mesh_key(
                    mesh_key, p_rank, r_rank, is_p2p=True
                )
                assert (
                    nccl_mesh_key in self.nccl_comm_cache
                ), "NCCL communicator index not found for mesh key: {nccl_mesh_key}"
                comm_index = self.nccl_comm_cache[nccl_mesh_key]
                nccl_send(tensor, 1, comm_index)
        else:
            assert (
                mesh_key in self.nccl_comm_cache
            ), "NCCL communicator index not found for mesh key: {mesh_key_or_comm_index}"
            comm_index = self.nccl_comm_cache[mesh_key]
            nccl_send(tensor, self.world_size + r_rank, comm_index)

    def recv(self, mesh_key: str, tensor: torch.Tensor, p_rank: int):
        """
        Receive data from a peer.
        Arguments:
            mesh_key: str The mesh key shoule in format of "{src_replica_name}_{dst_replica_name}"
            tensor: torch.Tensor
            p_rank: int
        """
        assert self.role == Role.ROLLOUT, "Only rollout can receive data."

        if self.rl_mode == "colocated_separated":
            r_rank = self.global_rank
            if r_rank == p_rank:
                # for the same device, we use IPC
                ipc_mesh_key = self.generate_mesh_key(
                    mesh_key, p_rank, r_rank, is_ipc=True
                )
                assert (
                    ipc_mesh_key in self.ipc_comm_cache
                ), "IPC socket not found for mesh key: {ipc_mesh_key}"
                socket = self.ipc_comm_cache[ipc_mesh_key]
                ipc_data = socket.recv_pyobj()
                tensor.copy_(tensor_ipc_deserialize(ipc_data).cuda())
            else:
                # for the different device, we use NCCL
                nccl_mesh_key = self.generate_mesh_key(
                    mesh_key, p_rank, r_rank, is_p2p=True
                )
                assert (
                    nccl_mesh_key in self.nccl_comm_cache
                ), "NCCL communicator index not found for mesh key: {nccl_mesh_key}"
                comm_index = self.nccl_comm_cache[nccl_mesh_key]
                nccl_recv(tensor, 0, comm_index)
        else:
            assert (
                mesh_key in self.nccl_comm_cache
            ), "NCCL communicator index not found for mesh key: {mesh_key}"
            comm_index = self.nccl_comm_cache[mesh_key]
            nccl_recv(tensor, p_rank, comm_index)
