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

"""Gated 2-rank end-to-end test for the NCCL payload transport.

Spawns two processes on two GPUs (rank 0 = rollout producer, rank 1 =
trainer consumer) that rendezvous over a real Redis and move one
trajectory GPU→GPU via ``nccl_send`` / ``nccl_recv``.  Self-skips unless
CUDA is available with ``>= 2`` devices and a Redis is reachable — so it is
a no-op on CPU-only / single-GPU CI and only runs where the full path can
actually be exercised.

Run explicitly on a 2-GPU box with Redis on localhost:6379:

    pytest tests/test_nccl_e2e.py -q
"""

import json
import os
import time
import unittest

import torch

REDIS_HOST = os.environ.get("COSMOS_TEST_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("COSMOS_TEST_REDIS_PORT", "6379"))
_EXPERIMENT = "nccl_e2e"
_META_KEY = "nccl_e2e:meta"
_DONE_KEY = "nccl_e2e:done"


def _redis_reachable() -> bool:
    try:
        import redis

        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


def _make_trajectory(device, ep_len=8, obs_dim=4, action_dim=2):
    return {
        "observations": torch.arange(
            ep_len * obs_dim, dtype=torch.float32, device=device
        ).reshape(ep_len, obs_dim),
        "actions": torch.ones(ep_len, action_dim, dtype=torch.float32, device=device),
        "rewards": torch.arange(ep_len, dtype=torch.float32, device=device),
        "episode_length": ep_len,
    }


def _worker(rank: int, world_size: int, err_queue):
    """Entry point for each spawned rank.  Reports failures via err_queue."""
    try:
        import redis

        from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

        os.environ["RANK"] = str(rank)
        os.environ["SLURM_JOB_ID"] = "nccl_e2e_job"
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        config = _Config()
        dims = dict(max_steps=8, obs_dim=4, action_dim=2)

        if rank == 0:
            # Producer.
            producer = NCCLRolloutMixin()
            producer.setup_nccl(
                replica_id="rollout-0",
                rollout_idx=0,
                redis_client=client,
                config=config,
                sender_rank=0,
                device=device,
                **dims,
            )
            traj = _make_trajectory(device)
            meta = producer.write_to_buffer(traj)
            assert meta is not None, "producer failed to pack buffer"
            client.set(_META_KEY, json.dumps(meta))
            # Serve until the consumer signals completion (bounded).
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not client.get(_DONE_KEY):
                time.sleep(0.05)
            producer.cleanup_nccl()
        else:
            # Consumer.
            packer = _ConsumerPacker()
            packer._setup_nccl_data_packer(
                device=device,
                redis_client=client,
                config=config,
                prefetch_timeout=30.0,
                max_attempts=3,
                recv_timeout=10.0,
            )
            # Wait for the producer's metadata.
            raw = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and raw is None:
                raw = client.get(_META_KEY)
                if raw is None:
                    time.sleep(0.05)
            assert raw is not None, "consumer never saw producer metadata"
            meta = json.loads(raw)

            resolved = packer.get_policy_input(rollout_output=meta)
            assert resolved is not None, "consumer failed to resolve NCCL ref"
            obs = resolved["observations"]
            expected = _make_trajectory(device)["observations"]
            assert torch.allclose(obs.float(), expected), "payload mismatch"
            client.set(_DONE_KEY, "1")
            packer.shutdown_nccl_data_packer()
    except Exception as e:  # pragma: no cover - surfaced to the parent
        import traceback

        err_queue.put(f"rank {rank}: {e}\n{traceback.format_exc()}")


class _Config:
    """Minimal config surface the mixins read."""

    class _Logging:
        experiment_name = _EXPERIMENT

    logging = _Logging()
    custom = {"nccl_max_steps": 8, "nccl_obs_dim": 4, "nccl_action_dim": 2}


# Defined at module scope so the consumer packer is import/spawn-friendly.
try:
    from cosmos_rl.utils.payload_transport.nccl.data_packer_mixin import (
        NCCLDataPackerMixin as _NCCLDataPackerMixin,
    )

    class _BaseTrajPacker:
        # Signature must match the real DataPacker.get_policy_input: the
        # PrefetchDataPackerMixin delegates via super() with
        # (sample, resolved, n_ignore_prefix_tokens) positionally.
        def get_policy_input(
            self, sample=None, rollout_output=None, n_ignore_prefix_tokens=0, **kw
        ):
            return rollout_output

    class _ConsumerPacker(_NCCLDataPackerMixin, _BaseTrajPacker):
        pass

except Exception:  # pragma: no cover - import guarded for CPU-only collection
    _ConsumerPacker = None


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "requires >= 2 CUDA devices for NCCL P2P",
)
@unittest.skipUnless(_redis_reachable(), f"requires Redis at {REDIS_HOST}:{REDIS_PORT}")
class TestNcclE2E(unittest.TestCase):
    def setUp(self):
        import redis

        self.client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        for key in (_META_KEY, _DONE_KEY):
            self.client.delete(key)

    def test_two_rank_roundtrip(self):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        err_queue = ctx.Queue()
        procs = [
            ctx.Process(target=_worker, args=(rank, 2, err_queue)) for rank in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)

        errors = []
        while not err_queue.empty():
            errors.append(err_queue.get())
        for p in procs:
            if p.is_alive():
                p.terminate()
                errors.append("a rank hung and was terminated")
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
