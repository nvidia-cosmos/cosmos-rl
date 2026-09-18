# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank NCCL lifecycle canary; one CUDA GPU per process, across nodes.

Run using torchrun with two ranks. Requires redis-server on rank zero and
network reachability between ranks. Uses an isolated ephemeral Redis instance.
Tests exact payloads, repeated close, and reattachment across three cycles.
This is healthy teardown validation, not native-fault recovery validation.
"""

import json
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
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    server = None
    client = None
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
        producer = NCCLRolloutMixin() if rank == 0 else None
        packer = Packer() if rank == 1 else None
        for cycle in range(3):
            metadata = [None]
            if rank == 0:
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
                producer.cleanup_nccl(timeout=30)
                producer.cleanup_nccl(timeout=0)
                assert all(not thread.is_alive() for thread in producer._nccl_threads)
                assert not producer._nccl_retained_entries
            dist.barrier()
            print(
                json.dumps(
                    {"rank": rank, "cycle": cycle, "exact_payload_and_close": True}
                ),
                flush=True,
            )
    finally:
        if client is not None:
            client.close()
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
