# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real Redis command reads must not delay a ready colocated rollout."""

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from unittest.mock import Mock

import pytest
import redis
import torch.distributed as dist

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.colocated.utils import CommandDispatcher
from cosmos_rl.dispatcher.command import (
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    TrainingCompleteCommand,
)
from cosmos_rl.policy.worker.colocated.policy_control import (
    ColocatedPolicyControlWorker,
)
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from cosmos_rl.utils.constant import RedisStreamConstant
from cosmos_rl.utils.redis_stream import RedisStreamHandler


@contextmanager
def redis_endpoint():
    # Keep the Unix socket below its platform path-length limit.
    with tempfile.TemporaryDirectory(prefix="cosmos-poll-") as directory:
        socket = Path(directory) / "redis.sock"
        server = subprocess.Popen(
            [
                "redis-server",
                "--port",
                "0",
                "--save",
                "",
                "--appendonly",
                "no",
                "--unixsocket",
                str(socket),
                "--unixsocketperm",
                "700",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        client = redis.Redis(
            unix_socket_path=str(socket), socket_timeout=12, protocol=2
        )
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    client.ping()
                    break
                except redis.ConnectionError:
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            yield client
        finally:
            client.close()
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def controller_for_client(client, rank=0):
    controller = object.__new__(ColocatedController)
    for role, cls in (
        ("policy", ColocatedPolicyControlWorker),
        ("rollout", ColocatedRolloutControlWorker),
    ):
        worker = object.__new__(cls)
        worker.replica_name, worker.global_rank = role, rank
        worker.redis_for_remote = object.__new__(RedisStreamHandler)
        worker.redis_for_remote.latest_id_command = "0-0"
        spy = Mock(wraps=client) if client is not None else Mock()
        if client is None:
            spy.xread.side_effect = AssertionError("Only rank zero may read Redis")
        worker.redis_for_remote.redis_clients = [spy]
        setattr(controller, role, worker)
    controller.remote_command_manager = CommandDispatcher(["policy", "rollout"])
    return controller


def sync_command(step=1):
    return RolloutToRolloutBroadcastCommand("rollout", ["rollout"], step, 10, False)


def terminal_command():
    return TrainingCompleteCommand(
        replica_name="policy", global_step=2, total_steps=2, remain_samples_num=8
    )


@pytest.fixture
def endpoint(monkeypatch):
    # Keep a regressed blocking read bounded without changing its semantics.
    monkeypatch.setattr(RedisStreamConstant, "CMD_READING_TIMEOUT_MS", 150)
    with redis_endpoint() as client:
        yield client


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("remove", [False, True])
def test_ready_rollout_uses_nonblocking_policy_poll(endpoint, queued, remove):
    controller = controller_for_client(endpoint)
    for step in (1, 2):
        command = sync_command(step)
        if queued:
            controller.remote_command_manager.publish_command(command.pack(), "rollout")
        else:
            endpoint.xadd("rollout_command", {"command": command.pack()})
        actual = controller.wait_for_remote_command(controller.rollout, remove=remove)
        assert actual.uuid_value == command.uuid_value
        if not remove:
            pending = controller.remote_command_manager.pop_command("rollout")
            assert pending.uuid_value == command.uuid_value
    reads = controller.policy.redis_for_remote.redis_clients[0].xread.call_args_list
    # A remote command is fetched at the bottom of the loop and rechecked on
    # its next pass; both policy-side checks must be nonblocking.
    assert len(reads) == (2 if queued else 4)
    assert all("block" not in call.kwargs for call in reads)
    rollout_reads = controller.rollout.redis_for_remote.redis_clients[
        0
    ].xread.call_args_list
    assert len(rollout_reads) == (0 if queued else 2)
    assert all(call.kwargs["block"] == 150 for call in rollout_reads)


@pytest.mark.parametrize("queued_sync", [False, True])
def test_real_policy_terminal_precedes_ready_rollout(endpoint, queued_sync):
    controller = controller_for_client(endpoint)
    if queued_sync:
        controller.remote_command_manager.publish_command(
            PolicyToRolloutUnicastCommand("policy", "rollout", 1, 1).pack(), "policy"
        )
    ready, terminal = sync_command(), terminal_command()
    controller.remote_command_manager.publish_command(ready.pack(), "rollout")
    endpoint.xadd("policy_command", {"command": terminal.pack()})
    actual = controller.wait_for_remote_command(controller.rollout)
    assert actual.uuid_value == terminal.uuid_value
    assert (
        controller.remote_command_manager.front_command("rollout").uuid_value
        == ready.uuid_value
    )
    controller.rollout.redis_for_remote.redis_clients[0].xread.assert_not_called()


def test_terminal_arriving_during_rollout_wait_is_observed(endpoint):
    controller = controller_for_client(endpoint)
    terminal = terminal_command()

    def wait_for_rollout(*args, **kwargs):
        endpoint.xadd("policy_command", {"command": terminal.pack()})
        return endpoint.xread(*args, **kwargs)

    rollout_client = controller.rollout.redis_for_remote.redis_clients[0]
    rollout_client.xread.side_effect = wait_for_rollout
    actual = controller.wait_for_remote_command(controller.rollout)
    assert actual.uuid_value == terminal.uuid_value
    assert rollout_client.xread.call_count == 1
    policy_reads = controller.policy.redis_for_remote.redis_clients[
        0
    ].xread.call_args_list
    assert len(policy_reads) == 2
    assert all("block" not in call.kwargs for call in policy_reads)


def test_ordinary_policy_wait_retains_blocking_read(endpoint):
    controller = controller_for_client(endpoint)
    terminal = terminal_command()
    endpoint.xadd("policy_command", {"command": terminal.pack()})
    actual = controller.wait_for_remote_command(controller.policy)
    assert actual.uuid_value == terminal.uuid_value
    reads = controller.policy.redis_for_remote.redis_clients[0].xread.call_args_list
    assert len(reads) == 1
    assert reads[0].kwargs["block"] == 150


def distributed_canary():
    """Keep all ranks in the real wait loop; only the leader accesses Redis."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    RedisStreamConstant.CMD_READING_TIMEOUT_MS = 150
    try:
        with redis_endpoint() if rank == 0 else nullcontext(None) as client:
            for case in (
                "healthy-queued",
                "healthy-remote",
                "terminal-precedes-ready",
                "terminal-after-empty-poll",
            ):
                if rank == 0:
                    client.flushdb()
                dist.barrier()
                controller = controller_for_client(client, rank)
                terminal = case.startswith("terminal")
                commands = [terminal_command() if terminal else sync_command()]
                dist.broadcast_object_list(commands, src=0)
                expected = commands[0]
                if case == "healthy-queued":
                    controller.remote_command_manager.publish_command(
                        expected.pack(), "rollout"
                    )
                elif case == "healthy-remote" and rank == 0:
                    client.xadd("rollout_command", {"command": expected.pack()})
                elif case == "terminal-precedes-ready":
                    controller.remote_command_manager.publish_command(
                        PolicyToRolloutUnicastCommand("policy", "rollout", 1, 1).pack(),
                        "policy",
                    )
                    controller.remote_command_manager.publish_command(
                        sync_command().pack(), "rollout"
                    )
                    if rank == 0:
                        client.xadd("policy_command", {"command": expected.pack()})
                elif case == "terminal-after-empty-poll" and rank == 0:

                    def read_after_stop(*args, **kwargs):
                        client.xadd("policy_command", {"command": expected.pack()})
                        return client.xread(*args, **kwargs)

                    controller.rollout.redis_for_remote.redis_clients[
                        0
                    ].xread.side_effect = read_after_stop
                dist.barrier()
                start = time.monotonic()
                actual = controller.wait_for_remote_command(controller.rollout)
                elapsed = time.monotonic() - start
                assert actual.uuid_value == expected.uuid_value, case
                if case == "terminal-precedes-ready":
                    assert (
                        controller.remote_command_manager.front_command("rollout")
                        is not None
                    )
                reads = controller.policy.redis_for_remote.redis_clients[0].xread
                if rank:
                    reads.assert_not_called()
                    controller.rollout.redis_for_remote.redis_clients[
                        0
                    ].xread.assert_not_called()
                else:
                    assert reads.call_count
                    assert all(
                        "block" not in call.kwargs for call in reads.call_args_list
                    )
                dist.barrier()
                print(
                    json.dumps(
                        {
                            "passed": True,
                            "case": case,
                            "rank": rank,
                            "elapsed_ms": elapsed * 1000,
                        }
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    assert int(os.environ.get("WORLD_SIZE", 1)) >= 2
    distributed_canary()
