# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The native weight-copy harness uses real, command-scoped HTTP readiness."""

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from cosmos_rl.collective import collective
from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
from cosmos_rl.dispatcher.protocol import Role
from p2r_test_support import P2RReadinessServer, p2r_test_client, p2r_test_command


def test_all_eight_cached_participants_share_command_and_wait_for_last(monkeypatch):
    command = PolicyToRolloutUnicastCommand(
        "policy", "rollout", 4, 4, trainable_only=True, ready_deadline=time.time() + 10
    )
    server = P2RReadinessServer(command)
    try:
        for key, value in server.environment().items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(collective, "create_nccl_uid", lambda: [1, 2, 3])

        def arrive(role, rank):
            loaded = p2r_test_command()
            assert loaded._serialize() == command._serialize()
            manager = object.__new__(collective.P2RCollectiveManager)
            manager.role, manager.global_rank, manager.world_size = role, rank, 4
            manager.replica_name = "policy" if role == Role.POLICY else "rollout"
            manager.api_client = p2r_test_client(role)
            manager.zmq_context, manager.ipc_comm_cache, manager.nccl_comm_cache = (
                None,
                {},
                {},
            )
            return manager._wait_transfer_ready(loaded, needs_build=False)

        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = [pool.submit(arrive, Role.POLICY, rank) for rank in range(4)]
            pending += [pool.submit(arrive, Role.ROLLOUT, rank) for rank in range(3)]
            deadline = time.monotonic() + 5
            while True:
                with server.readiness._lock:
                    operation = server.readiness._operations.get(command.uuid_value)
                    count = 0 if operation is None else len(operation["arrivals"])
                if count == 7:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert not any(future.done() for future in pending)
            pending.append(pool.submit(arrive, Role.ROLLOUT, 3))
            assert [future.result(timeout=5) for future in pending] == [
                {"state": "ready", "build": False, "uid": [1, 2, 3]}
            ] * 8
        server.assert_all_ready()
    finally:
        server.close()


def test_missing_parent_configuration_is_not_a_readiness_bypass(monkeypatch):
    monkeypatch.delenv("COSMOS_TEST_P2R_READY_PORT", raising=False)
    with pytest.raises(KeyError, match="COSMOS_TEST_P2R_READY_PORT"):
        p2r_test_client(Role.POLICY)
    monkeypatch.delenv("COSMOS_TEST_P2R_COMMAND", raising=False)
    with pytest.raises(KeyError, match="COSMOS_TEST_P2R_COMMAND"):
        p2r_test_command()
