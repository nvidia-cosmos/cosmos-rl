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

"""Tests for the UCXX data-packer mixin (MR5).

The mixin's networking path requires the optional ``ucxx-cu12`` extra
plus a real UCX server, so the test suite focuses on:

* MRO / surface contract (``UCXXDataPackerMixin`` mixed with a stub
  ``BaseDataPacker``-shaped class produces the expected method
  resolution order).
* The double-buffer state machine
  (``is_cold_start`` → ``defer_prefetch`` → steady state →
  ``collect_prefetch`` → ``defer_prefetch`` → ...) without any actual
  UCXX traffic.
* ``get_policy_input`` short-circuits to ``super()`` for plain (non-
  UCXX-flagged) inputs and routes to the cache for ``_ucxx`` refs.
* The cache-key helper is stable across worker_ip / port / slot.

The full end-to-end UCXX path is exercised by integration tests in the
downstream consumer.
"""

import unittest
from typing import Any, List

from cosmos_rl.utils.payload_transport.ucxx.data_packer_mixin import (
    UCXXDataPackerMixin,
)


class _StubDataPacker:
    """Minimal stand-in for a ``BaseDataPacker`` subclass.

    Records the arguments :meth:`get_policy_input` was invoked with so
    tests can verify the mixin properly delegates after resolving (or
    skipping) UCXX refs.
    """

    def __init__(self):
        self.calls: List[dict] = []

    def get_policy_input(
        self,
        sample: Any = None,
        rollout_output: Any = None,
        n_ignore_prefix_tokens: int = 0,
        **kwargs,
    ) -> Any:
        self.calls.append(
            {
                "sample": sample,
                "rollout_output": rollout_output,
                "n_ignore_prefix_tokens": n_ignore_prefix_tokens,
                "kwargs": kwargs,
            }
        )
        # Echo so tests can verify what was forwarded.
        return rollout_output


class _Packer(UCXXDataPackerMixin, _StubDataPacker):
    """MRO: UCXXDataPackerMixin first, then _StubDataPacker."""


# ---------------------------------------------------------------------------
# Method resolution order
# ---------------------------------------------------------------------------


class TestMro(unittest.TestCase):
    def test_mixin_intercepts_first(self):
        mro = [c.__name__ for c in _Packer.__mro__]
        # Mixin must precede the concrete packer for ``get_policy_input``
        # to intercept.
        self.assertLess(
            mro.index("UCXXDataPackerMixin"),
            mro.index("_StubDataPacker"),
        )


# ---------------------------------------------------------------------------
# Double-buffer state machine (no UCXX traffic)
# ---------------------------------------------------------------------------


class TestPrefetchStateMachine(unittest.TestCase):
    """Drive the public prefetch API without actually starting any UCXX
    fetches.  ``_ucxx_dp_enabled`` stays False so ``start_prefetch`` is
    a no-op (which is the right behavior on a cold-start of a worker
    where UCXX setup hasn't happened yet)."""

    def setUp(self) -> None:
        self.p = _Packer()

    def test_initial_cold_start(self):
        self.assertTrue(self.p.is_cold_start)
        self.assertIsNone(self.p.prefetch_buffer)

    def test_defer_seeds_buffer_on_cold_start(self):
        rollouts = ["r0", "r1", "r2"]
        self.p.defer_prefetch(rollouts)
        # Cold start: defer just seeds the buffer (no pending wait).
        self.assertFalse(self.p.is_cold_start)
        self.assertEqual(self.p.prefetch_buffer, rollouts)
        self.assertFalse(self.p._prefetch_pending)

    def test_defer_after_seed_marks_pending(self):
        # First defer seeds; second defer marks pending so the next
        # ``collect_prefetch`` will block until the wait resolves.
        self.p.defer_prefetch(["r0"])
        self.p.defer_prefetch(["r1"])
        self.assertTrue(self.p._prefetch_pending)
        self.assertEqual(self.p._prefetch_rollouts, ["r1"])

    def test_collect_no_op_when_nothing_pending(self):
        # No pending defer => collect just returns the buffer (None).
        out = self.p.collect_prefetch()
        self.assertIsNone(out)
        # Buffer still cold-start.
        self.assertTrue(self.p.is_cold_start)

    def test_collect_returns_buffer_after_seed(self):
        self.p.defer_prefetch(["r0", "r1"])
        out = self.p.collect_prefetch()
        self.assertEqual(out, ["r0", "r1"])

    def test_start_prefetch_is_noop_when_disabled(self):
        # ``_ucxx_dp_enabled`` defaults to False; start should not raise
        # and should leave buffer state unchanged.
        before = self.p.prefetch_buffer
        self.p.start_prefetch(["r0", "r1"])
        self.assertEqual(self.p.prefetch_buffer, before)


# ---------------------------------------------------------------------------
# get_policy_input dispatch
# ---------------------------------------------------------------------------


class TestGetPolicyInputDispatch(unittest.TestCase):
    def setUp(self) -> None:
        self.p = _Packer()

    def test_plain_dict_delegates_to_super(self):
        traj = {"observations": [1, 2, 3], "rewards": [0.1, 0.2]}
        out = self.p.get_policy_input(rollout_output=traj)
        self.assertIs(out, traj)
        self.assertEqual(len(self.p.calls), 1)
        self.assertIs(self.p.calls[0]["rollout_output"], traj)

    def test_non_ucxx_string_delegates_to_super(self):
        out = self.p.get_policy_input(rollout_output="just a string")
        self.assertEqual(out, "just a string")
        self.assertEqual(len(self.p.calls), 1)

    def test_ucxx_ref_without_cache_skips_episode(self):
        """When UCXX is disabled and the cache is empty, the sync
        fetch returns None and the mixin skips the episode (returns
        None without delegating to super()).  This is the documented
        behavior in :meth:`UCXXDataPackerMixin.get_policy_input`."""
        ref = {
            "_ucxx": True,
            "_worker_ip": "10.0.0.1",
            "_ucxx_port": 7000,
            "_slot": 5,
        }
        out = self.p.get_policy_input(rollout_output=ref)
        self.assertIsNone(out)
        # super() was never invoked because nothing was resolvable.
        self.assertEqual(len(self.p.calls), 0)

    def test_ucxx_ref_with_cache_hit_delegates_to_super(self):
        """When the prefetch cache has the resolved tensors, the mixin
        forwards the resolved dict to super() instead of the original
        UCXX metadata."""
        resolved = {"observations": [1, 2, 3], "rewards": [0.1, 0.2]}
        ref = {
            "_ucxx": True,
            "_worker_ip": "10.0.0.1",
            "_ucxx_port": 7000,
            "_slot": 5,
        }
        cache_key = UCXXDataPackerMixin._ucxx_dp_cache_key(ref)
        # Inject directly into the cache (bypass UCXX entirely).
        self.p._ucxx_dp_prefetch_cache = {cache_key: resolved}

        out = self.p.get_policy_input(rollout_output=ref)
        self.assertIs(out, resolved)
        self.assertEqual(len(self.p.calls), 1)
        # super() received the resolved dict, not the metadata stub.
        self.assertIs(self.p.calls[0]["rollout_output"], resolved)


# ---------------------------------------------------------------------------
# Cache-key helper
# ---------------------------------------------------------------------------


class TestCacheKey(unittest.TestCase):
    def test_cache_key_format(self):
        key = UCXXDataPackerMixin._ucxx_dp_cache_key(
            {
                "_worker_ip": "10.0.0.1",
                "_ucxx_port": 7000,
                "_slot": 5,
            }
        )
        self.assertEqual(key, "10.0.0.1:7000:5")

    def test_cache_key_handles_missing_fields(self):
        # Missing fields should yield a still-deterministic string so
        # collisions show up rather than crashes.
        key = UCXXDataPackerMixin._ucxx_dp_cache_key({"_worker_ip": "x"})
        self.assertIn("x:", key)


# ---------------------------------------------------------------------------
# Transport-driven invocation (MR3a + MR3b unification)
# ---------------------------------------------------------------------------


class TestSetupViaTransportAttach(unittest.TestCase):
    """``UCXXPayloadTransport.attach_data_packer`` invokes
    ``_setup_ucxx_data_packer`` so the trainer no longer has to call
    it manually.  These tests assert that the transport-driven path
    produces the same final mixin state as a direct call (at least for
    the recordable arguments) and that the backward-compat alias
    forwards correctly.
    """

    def test_attach_via_transport_invokes_setup_with_same_args(self):
        # Capture what _setup_ucxx_data_packer receives via the
        # transport hook vs a direct call.  We patch the method on the
        # mixin to avoid requiring a real UCX server / ucxx-cu12.
        from types import SimpleNamespace
        from unittest import mock

        from cosmos_rl.utils.payload_transport.ucxx import UCXXPayloadTransport

        captured_via_transport = {}
        captured_direct = {}

        class _AttachPacker(UCXXDataPackerMixin):
            pass

        def _record_into(target):
            def _fake(self, **kwargs):
                target.update(kwargs)

            return _fake

        # Path 1: transport-driven attach.
        config = SimpleNamespace(
            custom={
                "ucxx_prefetch_timeout": 7.5,
                "ucxx_read_max_attempts": 4,
                "ucxx_read_timeout": 45.0,
            }
        )
        with mock.patch.object(
            UCXXDataPackerMixin,
            "_setup_ucxx_data_packer",
            _record_into(captured_via_transport),
        ):
            UCXXPayloadTransport().attach_data_packer(
                _AttachPacker(), config=config, device="cuda:2"
            )

        # Path 2: direct call with the same args.
        with mock.patch.object(
            UCXXDataPackerMixin,
            "_setup_ucxx_data_packer",
            _record_into(captured_direct),
        ):
            packer = _AttachPacker()
            packer._setup_ucxx_data_packer(
                device="cuda:2",
                prefetch_timeout=7.5,
                max_attempts=4,
                read_timeout=45.0,
            )

        # Final captured state is identical on both paths.
        self.assertEqual(captured_via_transport, captured_direct)

    def test_backward_compat_alias_forwards_to_underscored(self):
        # Downstream code that still calls ``setup_ucxx_data_packer``
        # (the original public name) must keep working: the alias
        # forwards positional args to ``_setup_ucxx_data_packer``
        # using kwargs.  Historical signature is ``(device,
        # prefetch_timeout)``; newer kwargs default through.
        from unittest import mock

        captured = {}

        def _fake(self, **kwargs):
            captured.update(kwargs)

        class _AliasPacker(UCXXDataPackerMixin):
            pass

        packer = _AliasPacker()
        with mock.patch.object(UCXXDataPackerMixin, "_setup_ucxx_data_packer", _fake):
            # Positional call (the historical public surface).
            packer.setup_ucxx_data_packer("cuda:0", 5.0)
        self.assertEqual(
            captured,
            {
                "device": "cuda:0",
                "prefetch_timeout": 5.0,
                "max_attempts": 2,
                "read_timeout": 5.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
