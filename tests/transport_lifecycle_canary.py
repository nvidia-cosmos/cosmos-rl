# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank payload lifecycle canary; one CUDA GPU per process, across nodes.

Run using torchrun with two ranks; select --backend nccl (default) or ucxx.
Requires redis-server on rank zero and
network reachability between ranks. Uses an isolated ephemeral Redis instance.
Tests exact payloads, repeated close, and reattachment across three cycles.
This is healthy teardown validation, not native-fault recovery validation.
"""

import json
import argparse
import sys
import os
import socket
import subprocess
import time
from datetime import timedelta
from types import SimpleNamespace

import redis
import torch
import torch.distributed as dist

from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
from cosmos_rl.utils.payload_transport.nccl.strategy import compose_nccl_transport
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin


class BasePacker:
    def get_policy_input(self, sample=None, rollout_output=None, *args, **kwargs):
        return rollout_output


class Packer(PrefetchDataPackerMixin, BasePacker):
    pass


def trajectory(device):
    return {
        "observations": torch.arange(32, dtype=torch.float32, device=device).reshape(
            8, 4
        ),
        "actions": torch.ones(8, 2, device=device),
        "rewards": torch.arange(8, dtype=torch.float32, device=device),
        "episode_length": 8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nccl", "ucxx"), default="nccl")
    parser.add_argument("--server-threads", type=int, default=4)
    args = parser.parse_args()
    backend = args.backend
    print(
        json.dumps(
            {
                "ucx_environment": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith(("UCX_", "UCXPY_"))
                }
            }
        ),
        flush=True,
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    server = None
    client = None
    producer = None
    packer = None
    try:
        endpoint = [None]
        if rank == 0:
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
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
                ]
            )
            endpoint[0] = (os.environ["MASTER_ADDR"], port)
        dist.broadcast_object_list(endpoint, src=0)
        host, port = endpoint[0]
        client = redis.Redis(host=host, port=port, socket_timeout=5)
        deadline = time.monotonic() + 10
        while True:
            try:
                client.ping()
                break
            except redis.ConnectionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name="transport-lifecycle-canary"),
            custom={"nccl_max_steps": 8, "nccl_obs_dim": 4, "nccl_action_dim": 2},
        )
        if backend == "ucxx":
            from cosmos_rl.utils.payload_transport.ucxx.mixins import UCXXRolloutMixin
            from cosmos_rl.utils.payload_transport.ucxx.strategy import (
                compose_ucxx_transport,
            )
            from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
                UCXXBufferConfig,
            )
            from cosmos_rl.utils.payload_transport.ucxx.transport import (
                UCXXPayloadTransport,
            )

            producer = UCXXRolloutMixin() if rank == 0 else None
        else:
            producer = NCCLRolloutMixin() if rank == 0 else None
        packer = Packer() if rank == 1 else None
        for cycle in range(3):
            metadata = [None]
            if rank == 0:
                if backend == "ucxx":
                    producer.setup_ucxx(
                        replica_id=f"lifecycle-{os.getpid()}-{cycle}",
                        max_steps=8,
                        obs_dim=4,
                        action_dim=2,
                        port=31000 + cycle * 16,
                        config=UCXXBufferConfig(
                            max_entries=8,
                            entry_size_bytes=4096,
                            n_server_threads=args.server_threads,
                        ),
                    )
                else:
                    producer.setup_nccl(
                        replica_id="lifecycle-producer",
                        rollout_idx=0,
                        redis_client=client,
                        config=config,
                        sender_rank=0,
                        device=device,
                        max_steps=8,
                        obs_dim=4,
                        action_dim=2,
                    )
                metadata[0] = producer.write_to_buffer(trajectory(device))
                assert metadata[0] is not None
            else:
                packer._nccl_dp_receiver_replica = "lifecycle-consumer"
                if backend == "ucxx":
                    compose_ucxx_transport(packer, device=device, read_timeout=10)
                else:
                    compose_nccl_transport(
                        packer,
                        device=device,
                        redis_client=client,
                        config=config,
                        prefetch_timeout=30,
                        max_attempts=2,
                        recv_timeout=10,
                    )
            dist.broadcast_object_list(metadata, src=0)
            if rank == 1:
                result = packer.get_policy_input(rollout_output=metadata[0])
                assert result is not None
                for key in ("observations", "actions", "rewards"):
                    torch.testing.assert_close(
                        result[key], trajectory(device)[key], rtol=0, atol=0
                    )
                packer.close_transport(timeout=30)
                packer.close_transport(timeout=0)
                assert packer._prefetch_thread is None
                assert packer._transport_strategy is None
            dist.barrier()
            if rank == 0:
                if backend == "ucxx":
                    buffer = producer._ucxx_buffer
                    UCXXPayloadTransport().close_producer(producer, timeout=30)
                    UCXXPayloadTransport().close_producer(producer, timeout=0)
                    assert producer._ucxx_buffer is None
                    assert not buffer._server_threads
                    buffer.unlink()
                else:
                    producer.cleanup_nccl(timeout=30)
                    producer.cleanup_nccl(timeout=0)
                    assert all(
                        not thread.is_alive() for thread in producer._nccl_threads
                    )
                    assert not producer._nccl_retained_entries
            dist.barrier()
            print(
                json.dumps(
                    {"rank": rank, "cycle": cycle, "exact_payload_and_close": True}
                ),
                flush=True,
            )
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if packer is not None:
                packer.close_transport(timeout=30)
            if producer is not None:
                if backend == "ucxx":
                    UCXXPayloadTransport().close_producer(producer, timeout=30)
                else:
                    producer.cleanup_nccl(timeout=30)
        except Exception as error:
            print(f"rank={rank} explicit cleanup failed: {error!r}", flush=True)
            if not active_error:
                raise
        if client is not None:
            client.close()
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
