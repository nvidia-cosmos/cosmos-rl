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

"""Gated 2-process end-to-end test for the UCXX payload transport.

The NCCL counterpart (``test_nccl_e2e.py``) has existed for a while; UCXX had
no equivalent, so every claim about it rested on CPU unit tests driving a
stand-in client.  This moves a real trajectory between two GPUs over a real
UCXX endpoint: rank 0 writes into its shared ring buffer and serves it, rank 1
resolves the reference through the consumer packer and compares the payload
against the source tensor.

Both consumer construction paths are covered -- subclassing ``UCXXDataPackerMixin``
and composing ``UCXXTransportStrategy`` -- because a transport that wires
differently under composition would surface exactly here and nowhere else.

Unlike NCCL, UCXX needs no control plane: the rendezvous *is* the metadata the
producer returns (worker ip, port, slot).  A ``multiprocessing.Queue`` carries
it between the two ranks, so this test has no Redis dependency at all.

Self-skips unless CUDA has >= 2 devices and the ``ucxx`` extra is installed.

    pytest tests/test_ucxx_e2e.py -q
"""

import unittest

import torch

_EP_LEN = 8
_OBS_DIM = 4
_ACTION_DIM = 2

try:
    from cosmos_rl.utils.payload_transport.ucxx import UCXX_AVAILABLE
except Exception:  # pragma: no cover - package absent entirely
    UCXX_AVAILABLE = False


def _make_trajectory(device, ep_len=_EP_LEN):
    return {
        "observations": torch.arange(
            ep_len * _OBS_DIM, dtype=torch.float32, device=device
        ).reshape(ep_len, _OBS_DIM),
        "actions": torch.ones(ep_len, _ACTION_DIM, dtype=torch.float32, device=device),
        "rewards": torch.arange(ep_len, dtype=torch.float32, device=device),
        "episode_length": ep_len,
    }


def _producer(meta_q, done_q, err_q, direct: bool):
    """Rank 0: serve one trajectory out of a real UCXX endpoint."""
    try:
        from cosmos_rl.utils.payload_transport.ucxx.producer import (
            UCXXPayloadProducer,
        )
        from cosmos_rl.utils.payload_transport import PayloadContext

        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

        rollout = UCXXPayloadProducer()
        rollout.setup(
            replica_id="rollout-0",
            max_steps=_EP_LEN,
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            port=0,
        )
        if direct:
            context = PayloadContext("ucxx", rollout, rollout.default_layout)
            payload = context.acquire()
            trajectory = _make_trajectory("cpu")
            for name, target in payload.tensors.items():
                source = trajectory.get(name)
                if source is None:
                    target.zero_()
                    continue
                if isinstance(source, torch.Tensor):
                    target.copy_(source)
                else:
                    target.fill_(source)
            payload.mark_all_written()
            meta = context.publish(payload).to_dict()
        else:
            context = None
            meta = rollout.publish(_make_trajectory(device))
        assert meta is not None, "producer failed to write the trajectory"
        assert meta.get("_ucxx"), f"metadata is not a UCXX reference: {meta!r}"
        port = meta.get("_ucxx_port")
        ports = meta.get("_ports")
        assert isinstance(port, int) and port > 1024, (
            f"producer advertised an unusable auto-assigned port {port!r}"
        )
        assert isinstance(ports, list) and len(ports) == 4
        assert len(set(ports)) == len(ports)
        assert port in ports and all(p > 1024 for p in ports)
        assert "_buffer_handle" not in meta
        assert meta["_layout_id"] == rollout.default_layout.layout_id
        assert isinstance(meta["_slot_generation"], int)
        assert meta["_sender_incarnation"]
        meta_q.put(meta)

        # Serve until the consumer confirms, bounded so a dead peer cannot hang
        # the suite.
        try:
            done_q.get(timeout=120)
        except Exception:
            pass
        if context is not None:
            context.close()
        else:
            rollout.close()
    except Exception as e:  # pragma: no cover - surfaced to the parent
        import traceback

        err_q.put(f"producer: {e}\n{traceback.format_exc()}")


def _consumer(meta_q, done_q, err_q, composed: bool):
    """Rank 1: resolve the reference and check the bytes actually arrived."""
    try:
        torch.cuda.set_device(1)
        device = torch.device("cuda:1")

        class _BaseTrajPacker:
            # Mirrors the real DataPacker signature: the prefetch mixin
            # delegates positionally via super().
            def get_policy_input(
                self, sample=None, rollout_output=None, n_ignore_prefix_tokens=0, **kw
            ):
                return rollout_output

        if composed:
            from cosmos_rl.utils.payload_transport.prefetch_mixin import (
                PrefetchDataPackerMixin,
            )
            from cosmos_rl.utils.payload_transport.ucxx.strategy import (
                compose_ucxx_transport,
            )

            class _Packer(PrefetchDataPackerMixin, _BaseTrajPacker):
                """No UCXX ancestry -- the transport is an attached strategy."""

            packer = _Packer()
            compose_ucxx_transport(
                packer, device=device, prefetch_timeout=60.0, read_timeout=30.0
            )
        else:
            from cosmos_rl.utils.payload_transport.ucxx import UCXXDataPackerMixin

            class _Packer(UCXXDataPackerMixin, _BaseTrajPacker):
                pass

            packer = _Packer()
            packer._setup_ucxx_data_packer(
                device=device, prefetch_timeout=60.0, read_timeout=30.0
            )

        meta = meta_q.get(timeout=120)
        assert packer._should_intercept(meta), (
            f"consumer did not recognise the producer's reference: {meta!r}"
        )

        resolved = packer.get_policy_input(rollout_output=meta)
        assert resolved is not None, "consumer failed to resolve the UCXX ref"
        obs = resolved["observations"]
        expected = _make_trajectory(device)["observations"]
        assert obs.shape == expected.shape, f"shape {obs.shape} != {expected.shape}"
        assert torch.allclose(obs.float(), expected), "payload mismatch"

        done_q.put(1)
        if composed:
            # No transport-specific teardown exists for a composed packer:
            # shutdown_prefetch defaults before_join to the attached strategy,
            # and UCXX deliberately closes its client afterwards rather than
            # aborting mid-flight.
            packer.shutdown_prefetch()
            packer._transport_strategy.shutdown()
        else:
            packer.shutdown_ucxx_data_packer()
    except Exception as e:  # pragma: no cover - surfaced to the parent
        import traceback

        err_q.put(f"consumer: {e}\n{traceback.format_exc()}")
        try:
            done_q.put(1)  # never leave the producer parked on our account
        except Exception:
            pass


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "requires >= 2 CUDA devices for a real UCXX transfer",
)
@unittest.skipUnless(UCXX_AVAILABLE, "requires the 'ucxx' extra")
class TestUcxxE2E(unittest.TestCase):
    def _run_roundtrip(self, composed: bool, direct: bool = False):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        meta_q, done_q, err_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        procs = [
            ctx.Process(target=_producer, args=(meta_q, done_q, err_q, direct)),
            ctx.Process(target=_consumer, args=(meta_q, done_q, err_q, composed)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)

        errors = []
        while not err_q.empty():
            errors.append(err_q.get())
        for p in procs:
            if p.is_alive():
                p.terminate()
                errors.append("a rank hung and was terminated")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_two_process_roundtrip(self):
        """Consumer built by subclassing UCXXDataPackerMixin."""
        self._run_roundtrip(composed=False)

    def test_two_process_roundtrip_composed(self):
        """Consumer built by COMPOSING the transport -- no UCXX ancestry."""
        self._run_roundtrip(composed=True)

    def test_two_process_direct_storage_roundtrip(self):
        """Public owned-storage publication commits the shared-memory slot."""
        self._run_roundtrip(composed=True, direct=True)


if __name__ == "__main__":
    unittest.main()
