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
trajectory GPU→GPU via ``nccl_send`` / ``nccl_recv``.  Self-skips unless CUDA
is available with ``>= 2`` devices — so it is a no-op on CPU-only / single-GPU
CI and only runs where the full path can actually be exercised.

Redis is *not* a precondition: if nothing is listening, :func:`_ensure_redis`
starts a throwaway ``redis-server`` on a free port (see its docstring for why
depending on an ambient one silently disabled this test).  Point it at an
existing instance with ``COSMOS_TEST_REDIS_HOST`` / ``COSMOS_TEST_REDIS_PORT``.

    pytest tests/test_nccl_e2e.py -q
"""

import json
import os
import uuid
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


#: Extra schema field for the strided case, mirroring the per-generation
#: sampled-mode column an NDAS producer emits.
STRIDED_FIELD = "sampled_mode"
STRIDED_VALUE = 3


def _strided_scalar(device):
    """``shape=(1,)`` int64 whose last stride is 8, yet reports contiguous.

    A size-1 dimension may hold any stride, so ``.contiguous()`` leaves this
    untouched and ``view(torch.uint8)`` then rejects it.  Producers emit
    exactly this when they slice one column out of a per-generation tensor.
    """
    row = torch.arange(8, dtype=torch.int64, device=device).reshape(1, 8)
    value = row[:, STRIDED_VALUE]
    assert value.is_contiguous() and value.stride(-1) != 1
    return value


def _make_strided_trajectory(device, ep_len=8, obs_dim=4, action_dim=2):
    """The ordinary trajectory plus one field with a non-unit last stride.

    Dims stay exactly as the passing roundtrips use them: the stride is the
    only variable, so a failure here cannot be blamed on an unusual schema.
    """
    traj = _make_trajectory(device, ep_len, obs_dim, action_dim)
    traj[STRIDED_FIELD] = _strided_scalar(device)
    return traj


def _worker(
    rank: int,
    world_size: int,
    err_queue,
    composed: bool = False,
    strided: bool = False,
    run_id: str = "",
    action_dim: int = 2,
):
    """Entry point for each spawned rank.  Reports failures via err_queue.

    ``run_id`` namespaces the Redis keys.  Module-level keys let a producer
    that outlived its own test -- it serves for up to 60s -- be picked up by
    the next test in the session, which then hangs waiting on a peer that is
    already tearing down.  One namespace per roundtrip removes that coupling.
    """
    meta_key = "%s:%s" % (_META_KEY, run_id) if run_id else _META_KEY
    done_key = "%s:%s" % (_DONE_KEY, run_id) if run_id else _DONE_KEY
    try:
        import redis

        from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

        os.environ["RANK"] = str(rank)
        os.environ["SLURM_JOB_ID"] = "nccl_e2e_job"
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        config = _Config()
        dims = dict(max_steps=8, obs_dim=4, action_dim=action_dim)

        def build(dev):
            fn = _make_strided_trajectory if strided else _make_trajectory
            return fn(dev, action_dim=action_dim)

        if rank == 0:
            # Producer.
            if strided:
                # Extend the schema the way a backend does -- before setup, so
                # the mixin sizes its buffers from it.  The consumer needs no
                # change: the producer ships the schema in its metadata.
                import numpy as np

                import cosmos_rl.utils.payload_transport.nccl.mixins as _mixins
                from cosmos_rl.utils.trajectory import TensorSpec

                _base_build = _mixins.build_trajectory_schema

                def _build_with_strided_field(dims_arg):
                    schema = _base_build(dims_arg)
                    schema.append(
                        TensorSpec(name=STRIDED_FIELD, shape=(1,), dtype=np.int64)
                    )
                    return schema

                _mixins.build_trajectory_schema = _build_with_strided_field
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
            traj = build(device)
            meta = producer.write_to_buffer(traj)
            assert meta is not None, "producer failed to pack buffer"
            client.set(meta_key, json.dumps(meta))
            # Serve until the consumer signals completion (bounded).
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not client.get(done_key):
                time.sleep(0.05)
            producer.cleanup_nccl()
        else:
            # Consumer, built one of the two supported ways.  Both must move
            # real bytes GPU->GPU; the composed path is otherwise only covered
            # by CPU tests with a fake transport.
            if composed:
                from cosmos_rl.utils.payload_transport.nccl.strategy import (
                    compose_nccl_transport,
                )

                packer = _ComposedPacker()
                compose_nccl_transport(
                    packer,
                    device=device,
                    redis_client=client,
                    config=config,
                    prefetch_timeout=30.0,
                    max_attempts=3,
                    recv_timeout=10.0,
                )
            else:
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
                raw = client.get(meta_key)
                if raw is None:
                    time.sleep(0.05)
            assert raw is not None, "consumer never saw producer metadata"
            meta = json.loads(raw)

            resolved = packer.get_policy_input(rollout_output=meta)
            assert resolved is not None, "consumer failed to resolve NCCL ref"
            expected_traj = build(device)
            obs = resolved["observations"]
            assert torch.allclose(obs.float(), expected_traj["observations"]), (
                "payload mismatch"
            )
            if strided:
                # The point of the strided run: the non-unit-stride field must
                # arrive intact rather than taking the pack fallback.
                got = resolved[STRIDED_FIELD].reshape(-1)[0].item()
                assert got == STRIDED_VALUE, (
                    "strided field mismatch: got %r want %r" % (got, STRIDED_VALUE)
                )
            client.set(done_key, "1")
            if composed:
                # No transport-specific teardown to call: shutdown_prefetch
                # defaults before_join to the attached strategy, which is the
                # abort that lets a parked recv fail fast.
                packer.shutdown_prefetch()
                packer._transport_strategy.shutdown()
            else:
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

    from cosmos_rl.utils.payload_transport.prefetch_mixin import (
        PrefetchDataPackerMixin as _PrefetchDataPackerMixin,
    )

    class _ComposedPacker(_PrefetchDataPackerMixin, _BaseTrajPacker):
        """No NCCL ancestry -- the transport arrives as an attached strategy."""

except Exception:  # pragma: no cover - import guarded for CPU-only collection
    _ConsumerPacker = None
    _ComposedPacker = None


def _ensure_redis() -> bool:
    """Make a Redis available, starting a throwaway one if nothing is listening.

    Without this the test only ran where a Redis happened to already be bound
    to :data:`REDIS_PORT`, and self-skipped everywhere else -- including CI,
    where nothing binds 6379 (the dispatcher starts its own server on a *free*
    port).  A skip there is indistinguishable from a pass, so the transport's
    only end-to-end guard would have been silently absent.

    ``redis-server`` ships in the CI image, so start one rather than depend on
    ambient state.  The chosen port is exported so the spawned ranks -- which
    re-import this module -- connect to the same instance instead of each
    starting another; that env var is also why re-entry here is a no-op in the
    children.
    """
    global REDIS_PORT

    if _redis_reachable():
        return True

    import atexit
    import shutil
    import socket
    import subprocess

    exe = shutil.which("redis-server")
    if exe is None:
        return False

    with socket.socket() as probe:
        probe.bind((REDIS_HOST, 0))
        port = probe.getsockname()[1]

    try:
        proc = subprocess.Popen(
            [exe, "--port", str(port), "--save", "", "--appendonly", "no"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False

    REDIS_PORT = port
    os.environ["COSMOS_TEST_REDIS_PORT"] = str(port)
    atexit.register(lambda: (proc.terminate(), proc.wait(timeout=10)))

    deadline = time.time() + 15.0
    while time.time() < deadline:
        if proc.poll() is not None:  # died on startup; nothing to wait for
            return False
        if _redis_reachable():
            return True
        time.sleep(0.1)
    return False


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "requires >= 2 CUDA devices for NCCL P2P",
)
@unittest.skipUnless(
    _ensure_redis(),
    f"requires Redis at {REDIS_HOST}:{REDIS_PORT} (and no redis-server)",
)
class TestNcclE2E(unittest.TestCase):
    def setUp(self):
        import redis

        self.client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        for key in (_META_KEY, _DONE_KEY):
            self.client.delete(key)

    def _run_roundtrip(
        self, composed: bool, strided: bool = False, action_dim: int = 2
    ):
        import torch.multiprocessing as mp

        # One Redis namespace per roundtrip: a producer serves for up to 60s,
        # so with shared keys it can outlive its own test and be picked up by
        # the next one, which then waits on a peer that is tearing down.
        run_id = uuid.uuid4().hex[:8]
        ctx = mp.get_context("spawn")
        err_queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_worker,
                args=(rank, 2, err_queue, composed, strided, run_id, action_dim),
            )
            for rank in range(2)
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

    def test_two_rank_roundtrip(self):
        """Consumer built by subclassing the NCCL mixin (the original path)."""
        self._run_roundtrip(composed=False)

    def test_two_rank_roundtrip_composed(self):
        """Consumer built by COMPOSING the transport -- no NCCL ancestry.

        The composed path's other coverage is CPU-only with a stand-in
        transport, so without this it never actually moves bytes between two
        GPUs. Same assertions as above: if composition wires anything
        differently, the payload comes back wrong or not at all.
        """
        self._run_roundtrip(composed=True)

    def test_two_rank_roundtrip_narrow_action_dim(self):
        """A payload schema narrower than the consumer config must still resolve.

        The consumer decodes from the schema the producer ships in its
        metadata, so ``action_dim=1`` against a config declaring 2 should be
        transparent.  When this shape was first tried the payload packed and
        the consumer then dropped the reference inside ``_rendezvous_one``
        within 18ms -- no bytes moved and the episode fell back to Redis.
        Ordinary trajectory here, so only the dim differs: if this fails, the
        consumer path is still sizing from config rather than from the
        reference, and any backend whose schema differs loses payloads
        silently.
        """
        self._run_roundtrip(composed=False, action_dim=1)

    def test_two_rank_roundtrip_strided_field(self):
        """A field with a non-unit last stride must survive the real transfer.

        CPU tests cover the packer in isolation; this proves the same payload
        packs, sends, and unpacks GPU->GPU instead of failing in
        ``write_to_buffer`` and dropping the run to the disk fallback.
        """
        self._run_roundtrip(composed=False, strided=True)


if __name__ == "__main__":
    unittest.main()
