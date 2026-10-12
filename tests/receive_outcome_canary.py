# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real NCCL missing/rejection outcomes, leases and following healthy transfers.

Run under torchrun with two ranks on one or two nodes. Uses a private temporary
Redis process and real producer buffers. Gloo coordinates the test, not the
production data plane. Unknown/native-completion failures are separate terminal
fault tests, not safely rejected samples.
"""

from datetime import timedelta
import os
import socket
import subprocess
import time
from types import SimpleNamespace
import uuid

import redis
import torch
import torch.distributed as dist

from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
from cosmos_rl.utils.payload_transport.receive_memory import ReceiveMemoryError
from cosmos_rl.utils.transport_failure import TransportUnusableError
from test_nccl_e2e import _ConsumerPacker, _make_trajectory


def start_redis():
    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [
            "redis-server",
            "--port",
            str(port),
            "--bind",
            "0.0.0.0",
            "--protected-mode",
            "no",
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
    )
    probe = redis.Redis(host="127.0.0.1", port=port, socket_timeout=1)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                probe.ping()
                return server, port
            except redis.ConnectionError:
                assert server.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
    except BaseException:
        server.terminate()
        server.wait(timeout=5)
        raise
    finally:
        probe.close()


def resolve(packer, refs, prepared):
    def prepare():
        return [packer.get_policy_input(rollout_output=ref) for ref in refs]

    if prepared:
        future = packer.start_prepared_prefetch(refs, prepare)
        values = future.result(timeout=60)
        packer.release_prepared_prefetch(future)
        return values
    packer.start_prefetch(refs)
    packer.wait_prefetch()
    return prepare()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.cuda.set_device(device)
    server = producer = packer = client = None
    try:
        connection = [None]
        if rank == 0:
            server, port = start_redis()
            connection[0] = (os.environ["MASTER_ADDR"], port, uuid.uuid4().hex)
        dist.broadcast_object_list(connection, src=0)
        host, port, run_id = connection[0]
        client = redis.Redis(host=host, port=port, decode_responses=True)
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name=f"receive-outcomes-{run_id}"),
            custom={
                "nccl_receive_budget_bytes": 8192,
                "nccl_receive_admission_timeout": 30,
            },
        )
        if rank == 0:
            producer = NCCLRolloutMixin()
            producer.setup_nccl(
                replica_id=f"producer-{run_id}",
                rollout_idx=0,
                redis_client=client,
                config=config,
                sender_rank=0,
                device=device,
                max_steps=8,
                obs_dim=4,
                action_dim=2,
                registry_capacity=64,
            )
        else:
            packer = _ConsumerPacker()
            packer._setup_nccl_data_packer(
                device=device,
                redis_client=client,
                config=config,
                prefetch_timeout=90,
                recv_timeout=10,
                first_transfer_timeout=30,
            )

            # Any fallback violates bounded ownership, including a second read.
            def unexpected_refetch(*args, **kwargs):
                raise AssertionError("safe outcome attempted synchronous refetch")

            packer._sync_fetch = unexpected_refetch

        for prepared in (False, True):
            for outcome in ("missing", "no_schema"):
                for mixed in (False, True):
                    for healthy in (False, True):
                        transfer = [None]
                        if rank == 0:
                            bad = producer.write_to_buffer(_make_trajectory(device))
                            assert bad is not None
                            if not healthy:
                                entry = producer._nccl_registry.get(bad["_transfer_id"])
                                assert entry is not None
                                if outcome == "missing":
                                    assert producer._nccl_registry.free(
                                        bad["_transfer_id"]
                                    )
                                else:
                                    bad.pop("_schema")
                            refs = [bad]
                            if mixed:
                                good = producer.write_to_buffer(
                                    _make_trajectory(device)
                                )
                                assert good is not None
                                refs.append(good)
                            torch.cuda.synchronize(device)
                            transfer[0] = refs
                        dist.broadcast_object_list(transfer, src=0)
                        refs = transfer[0]
                        if rank == 1:
                            values = resolve(packer, refs, prepared)
                            lease = packer._prefetch_cache
                            expected_rejected = (
                                set() if healthy else {refs[0]["_transfer_id"]}
                            )
                            assert lease.rejected_keys == expected_rejected
                            for i, value in enumerate(values):
                                assert (
                                    packer.get_policy_input(rollout_output=refs[i])
                                    is value
                                )
                                if not healthy and i == 0:
                                    assert value is None
                                else:
                                    expected = _make_trajectory(device)
                                    for name in ("observations", "actions", "rewards"):
                                        torch.testing.assert_close(
                                            value[name], expected[name]
                                        )
                            del value, values
                            packer.release_prefetch()
                            assert lease.released and not lease.rejected_keys
                            stats = packer._transport_strategy.receive_memory_stats()
                            assert (
                                stats["reserved_bytes"]
                                == stats["current_tensor_bytes"]
                                == 0
                            )
                            assert stats["peak_reserved_bytes"] <= stats["budget_bytes"]
                            assert not packer._transport_strategy._receive_budget.closed
                            if not healthy:
                                try:
                                    packer._transport_strategy.sync_fetch(refs[0])
                                except ReceiveMemoryError:
                                    pass
                                else:
                                    raise AssertionError(
                                        "bounded unleased refetch was accepted"
                                    )
                        dist.barrier()
                    print(
                        f"RECEIVE_OUTCOME_PASS rank={rank} outcome={outcome} prepared={prepared} mixed={mixed} healthy_followup=True",
                        flush=True,
                    )
        # A completed receive with an invalid identity is terminal, unlike a
        # pre-accept rejection. Call the worker's fetch primitive directly so
        # the test can inspect its retained closed-budget state before exit.
        transfer = [None]
        if rank == 0:
            meta = producer.write_to_buffer(_make_trajectory(device))
            assert meta is not None
            entry = producer._nccl_registry.get(meta["_transfer_id"])
            assert entry is not None
            entry.buffer[0] = 0
            torch.cuda.synchronize(device)
            transfer[0] = meta
        dist.broadcast_object_list(transfer, src=0)
        if rank == 1:
            try:
                packer._transport_strategy.fetch_batch([(0, transfer[0])])
            except TransportUnusableError as error:
                assert "Accepted payload identity mismatch" in str(error)
                assert packer._transport_strategy._receive_budget.closed
            else:
                raise AssertionError("accepted header mismatch was not terminal")
        dist.barrier()
        print(f"RECEIVE_HEADER_TERMINAL_PASS rank={rank}", flush=True)
    finally:
        if packer is not None:
            packer.shutdown_nccl_data_packer()
        if producer is not None:
            producer.cleanup_nccl()
        if client is not None:
            client.close()
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
