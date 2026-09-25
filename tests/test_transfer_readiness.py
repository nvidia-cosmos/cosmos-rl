# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import pytest

from cosmos_rl.collective import collective
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.command import Command, PolicyToRolloutUnicastCommand
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.utils import pynccl
from cosmos_rl.dispatcher.transfer_readiness import (
    TransferReadiness,
    TransferReadyRequest,
)


def declaration(**kwargs):
    return TransferReadyRequest(
        **{
            "operation_id": "command-1",
            "src": "policy",
            "dst": "rollout",
            "src_size": 1,
            "dst_size": 1,
            "side": "source",
            "rank": 0,
            "expires_at": 100,
            "needs_build": True,
            "uid": [1, 2, 3],
            **kwargs,
        }
    )


def test_duplicate_arrival_does_not_complete_agreement():
    ready = TransferReadiness(clock=lambda: 0)
    for _ in range(3):
        assert ready.arrive(declaration())["state"] == "waiting"
    response = ready.arrive(declaration(side="receiver", uid=None))
    assert response == {"state": "ready", "build": True, "uid": [1, 2, 3]}
    assert ready.arrive(declaration()) == response


@pytest.mark.parametrize(
    "mutation",
    [
        {"src": "other"},
        {"expires_at": 101},
        {"needs_build": False},
        {"uid": [4, 5]},
        {"error": "peer stopped"},
    ],
)
def test_failed_or_changed_arrival_poisoned_for_every_rank(mutation):
    ready = TransferReadiness(clock=lambda: 0)
    ready.arrive(declaration())
    assert ready.arrive(declaration(**mutation))["state"] == "failed"
    assert ready.arrive(declaration(side="receiver", uid=None))["state"] == "failed"


def test_expired_agreement_cannot_be_resurrected_after_pruning():
    now = [0]
    ready = TransferReadiness(clock=lambda: now[0], capacity=1)
    ready.arrive(declaration())
    assert ready.arrive(declaration(operation_id="new"))["state"] == "failed"
    now[0] = 101
    assert (
        ready.arrive(declaration(operation_id="new", expires_at=200))["state"]
        == "waiting"
    )
    assert ready.arrive(declaration())["state"] == "failed"
    assert len(ready._operations) == 1


@pytest.mark.parametrize("sizes", [(1, 1), (2, 3)])
def test_agreement_requires_every_rank_and_rebuilds_if_any_cache_missing(sizes):
    ready = TransferReadiness(clock=lambda: 0)
    requests = [
        declaration(
            src_size=sizes[0],
            dst_size=sizes[1],
            side=side,
            rank=rank,
            uid=[1] if (side, rank) == ("source", 0) else None,
            needs_build=(side, rank) == ("receiver", sizes[1] - 1),
        )
        for side, size in zip(("source", "receiver"), sizes)
        for rank in range(size)
    ]
    for request in requests[:-1]:
        assert ready.arrive(request)["state"] == "waiting"
    assert ready.arrive(requests[-1])["build"]


def test_command_serialization_preserves_readiness_deadline():
    command = PolicyToRolloutUnicastCommand("policy", "rollout", 1, 1)
    loaded = Command.depack(command.pack())
    assert loaded.ready_deadline == command.ready_deadline
    assert loaded.uuid_value == command.uuid_value


def manager(role, api):
    result = collective.P2RCollectiveManager.__new__(collective.P2RCollectiveManager)
    result.replica_name = "policy" if role == Role.POLICY else "rollout"
    result.role = role
    result.world_size = 1
    result.global_rank = 0
    result.api_client = api
    result.unique_ids_cache = {}
    result.nccl_comm_cache = {}
    result.ipc_comm_cache = {}
    result.zmq_context = None
    return result


@pytest.mark.parametrize("warm", [False, True])
def test_actual_manager_waits_for_busy_receiver_before_native_init(monkeypatch, warm):
    registry = pynccl._CommunicatorRegistry()
    monkeypatch.setattr(pynccl, "_COMM_REGISTRY", registry)
    ready = TransferReadiness()
    source_waiting, receiver_entered = threading.Event(), threading.Event()
    builds = []

    def wait(payload):
        if payload["side"] == "source":
            source_waiting.set()
        while True:
            result = ready.arrive(TransferReadyRequest(**payload))
            if result["state"] != "waiting":
                assert result["state"] == "ready", result
                return result
            time.sleep(0.001)

    def build(uid, rank, size, timeout_ms):
        assert receiver_entered.is_set(), (
            "Native timeout began before receiver readiness"
        )
        builds.append((uid, rank, size))
        return 10 + rank

    monkeypatch.setattr(collective, "create_nccl_uid", lambda: [7])
    monkeypatch.setattr(collective, "create_nccl_comm", build)
    peers = [
        manager(role, SimpleNamespace(wait_p2r_ready=wait))
        for role in (Role.POLICY, Role.ROLLOUT)
    ]
    if warm:
        for rank, peer in enumerate(peers):
            peer.nccl_comm_cache["policy_rollout"] = registry.register(
                object(), rank, 2
            )
            peer.unique_ids_cache["policy_rollout"] = [7]
    command = PolicyToRolloutUnicastCommand("policy", "rollout", 1, 1)
    with ThreadPoolExecutor(2) as pool:
        sending = pool.submit(peers[0]._setup_inter_replica_communicators, command)
        assert source_waiting.wait(2)
        assert not sending.done()
        assert builds == []
        receiver_entered.set()
        receiving = pool.submit(peers[1]._setup_inter_replica_communicators, command)
        sending.result(3)
        receiving.result(3)
    assert len(builds) == (0 if warm else 2)


@pytest.mark.parametrize("aborted_side", [None, "source", "receiver", "both"])
def test_registered_cache_state_participates_in_existing_readiness(
    monkeypatch, aborted_side
):
    registry = pynccl._CommunicatorRegistry()
    monkeypatch.setattr(pynccl, "_COMM_REGISTRY", registry)
    aborted = []
    monkeypatch.setattr(pynccl, "_nccl", SimpleNamespace(ncclCommAbort=aborted.append))
    ready = TransferReadiness()
    declarations, builds = [], []

    def wait(payload):
        declarations.append(payload)
        while True:
            response = ready.arrive(TransferReadyRequest(**payload))
            if response["state"] != "waiting":
                assert response["state"] == "ready", response
                return response
            time.sleep(0.001)

    def build(uid, rank, size, **kwargs):
        builds.append((uid, rank, size))
        return registry.register(object(), rank, size)

    monkeypatch.setattr(collective, "create_nccl_uid", lambda: [42])
    monkeypatch.setattr(collective, "create_nccl_comm", build)
    peers = [
        manager(role, SimpleNamespace(wait_p2r_ready=wait))
        for role in (Role.POLICY, Role.ROLLOUT)
    ]
    for rank, (side, peer) in enumerate(zip(("source", "receiver"), peers)):
        index = registry.register(object(), rank, 2)
        peer.nccl_comm_cache["policy_rollout"] = index
        peer.unique_ids_cache["policy_rollout"] = [1]
        if aborted_side in (side, "both"):
            pynccl.nccl_abort(index)

    command = PolicyToRolloutUnicastCommand("policy", "rollout", 1, 1)
    with ThreadPoolExecutor(2) as pool:
        operations = [
            pool.submit(peer._setup_inter_replica_communicators, command)
            for peer in peers
        ]
        for operation in operations:
            operation.result(3)

    assert len(builds) == (0 if aborted_side is None else 2)
    for declaration_ in declarations:
        assert declaration_["needs_build"] == (
            aborted_side in (declaration_["side"], "both")
        )
    if aborted_side is not None:
        assert all(uid == [42] and size == 2 for uid, _, size in builds)
        assert all(peer.unique_ids_cache["policy_rollout"] == [42] for peer in peers)
        assert len(aborted) == 2, "old native handles are each aborted only once"


def test_client_deadline_and_controller_failure_are_not_native_timeouts(monkeypatch):
    client = APIClient(Role.POLICY, ["localhost"], 12345)
    payload = declaration(expires_at=time.time() - 1).model_dump()
    with pytest.raises(TimeoutError, match="readiness expired"):
        client.wait_p2r_ready(payload)
    payload["expires_at"] = time.time() + 5
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"state": "failed", "error": "peer stopped"},
        ),
    )
    with pytest.raises(RuntimeError, match="peer stopped"):
        client.wait_p2r_ready(payload)
