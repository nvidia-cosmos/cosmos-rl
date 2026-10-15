# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C-9: actual policy/rollout P2P setup with cached IDs but absent handles."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.collective import collective
from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
from cosmos_rl.dispatcher.protocol import Role


@pytest.mark.parametrize("role", [Role.POLICY, Role.ROLLOUT])
@pytest.mark.parametrize("cached", [(), (1,), (2,), (1, 2)])
def test_each_pair_rebuilds_with_its_own_cached_uid(monkeypatch, role, cached):
    peer = object.__new__(collective.P2RCollectiveManager)
    peer.role, peer.global_rank, peer.world_size = role, 0, 3
    peer.replica_name = "policy" if role == Role.POLICY else "rollout"
    peer.unique_ids_cache, peer.nccl_comm_cache = {}, {}
    peer.ipc_comm_cache, peer.zmq_context = {}, None
    peer._wait_transfer_ready = Mock()
    peer.api_client = SimpleNamespace(
        post_nccl_comm_initiator=Mock(),
        post_nccl_comm_acceptor=lambda key: [1000 + len(key)],
    )
    command = PolicyToRolloutUnicastCommand("policy", "rollout", 3, 3)
    keys = {
        rank: peer.generate_mesh_key(
            command,
            0 if role == Role.POLICY else rank,
            rank if role == Role.POLICY else 0,
            is_p2p=True,
        )
        for rank in (1, 2)
    }
    for rank in cached:
        peer.unique_ids_cache[keys[rank]] = [rank]
    generated = iter(([101], [102]))
    monkeypatch.setattr(collective, "create_nccl_uid", lambda: next(generated))
    builds = []

    def build(uid, rank, size, **kwargs):
        builds.append((uid, rank, size))
        return len(builds)

    monkeypatch.setattr(collective, "create_nccl_comm", build)
    peer._setup_p2p_communicators(command)
    assert builds == [
        (peer.unique_ids_cache[keys[rank]], int(role == Role.ROLLOUT), 2)
        for rank in (1, 2)
    ]
    assert [peer.nccl_comm_cache[keys[rank]] for rank in (1, 2)] == [1, 2]
    peer._setup_p2p_communicators(command)
    assert len(builds) == 2, "warm handles must not be rebuilt"
