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

"""Regression tests for the 55745c NCCL packer contract through the
unified ``CommMixin._attach_payload_transport`` integration path.

These tests exist as a separate file (rather than in
``test_payload_transport.py``) because they exercise the full call
path from ``CommMixin`` -> ``PayloadTransportRegistry`` ->
``NcclPayloadTransport.attach_data_packer`` -> packer state.  They
guard against breaking downstream NCCL packers (notably Yuxiao's
``PolicyDataPacker``) when refactoring the data-packer attachment
flow.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from cosmos_rl.comm.base import CommMixin
from cosmos_rl.utils.payload_transport import (
    PAYLOAD_TRANSFER_KEY,
    PayloadTransportRegistry,
)


class _NcclAwarePacker:
    """Mirrors the 55745c contract.

    Note ``redis_client`` starts None; ``post_redis_injection`` records
    what value was set on the instance at the moment the hook ran so
    tests can assert the assignment-then-hook order.
    """

    def __init__(self):
        self.redis_client = None
        self.post_called = False
        self.client_at_post_call = None

    def post_redis_injection(self):
        self.client_at_post_call = self.redis_client
        self.post_called = True


class _NoOpPacker:
    """Packer with no ``redis_client`` attribute -- attach must skip it."""


class _FakeRedis:
    def __init__(self, *, ping_raises=None):
        self._ping_raises = ping_raises
        self.pinged = False

    def ping(self):
        self.pinged = True
        if self._ping_raises is not None:
            raise self._ping_raises
        return True


class _FakeRedisFactory:
    def __init__(self, fake):
        self.fake = fake
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.fake


class _CommHarness(CommMixin):
    """Minimal CommMixin host that bypasses init_data_packer machinery.

    The full ``init_data_packer`` flow requires heavy dependencies
    (transformers, model configs, distributed state) that these tests
    do not need to exercise.  This harness attaches the packer(s) and
    a minimal ``config`` directly so we can call
    ``_attach_payload_transport`` in isolation.
    """

    def __init__(self, *, mode, data_packer, val_data_packer=None, parallelism=None):
        self.role = "test_role"
        self.config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: mode})
        if parallelism is not None:
            self.config.policy = SimpleNamespace(
                parallelism=SimpleNamespace(**parallelism)
            )
        self.data_packer = data_packer
        if val_data_packer is not None:
            self.val_data_packer = val_data_packer

    # No real Redis controller; ``_build_redis_endpoint`` will fall
    # back to a synthetic localhost endpoint, which the patched
    # ``_redis_lib`` factory accepts.


class TestAttachPayloadTransportContract(unittest.TestCase):
    """Full-integration regression guards for the 55745c contract."""

    def _patch_nccl_redis(self, fake_factory):
        return mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._redis_lib",
            SimpleNamespace(Redis=fake_factory),
        )

    def test_assignment_then_hook_order_via_attach(self):
        # Drives the full integration: CommMixin -> registry -> NCCL
        # transport -> packer.  Asserts the assignment-then-hook order
        # that downstream NCCL packers depend on (PR #670, 55745c).
        packer = _NcclAwarePacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        harness = _CommHarness(mode="nccl", data_packer=packer)
        with self._patch_nccl_redis(factory):
            harness._attach_payload_transport()
        self.assertIs(packer.redis_client, fake)
        self.assertTrue(packer.post_called)
        # Order assertion: the hook saw the assigned client.
        self.assertIs(packer.client_at_post_call, fake)

    def test_receiver_replica_plumbed_from_worker_replica_name(self):
        # Multi-policy-replica addressing: the worker's globally-unique
        # replica_name must land on an NCCL-aware packer's
        # ``_nccl_dp_receiver_replica`` BEFORE attach runs setup, so the
        # producer keys comms per receiver replica.
        class _ReplicaAwarePacker(_NcclAwarePacker):
            _nccl_dp_receiver_replica = None

        packer = _ReplicaAwarePacker()
        factory = _FakeRedisFactory(_FakeRedis())
        harness = _CommHarness(mode="nccl", data_packer=packer)
        harness.replica_name = "policy-XYZ"
        with self._patch_nccl_redis(factory):
            harness._attach_payload_transport()
        self.assertEqual(packer._nccl_dp_receiver_replica, "policy-XYZ")

    def test_receiver_replica_plumbing_skips_packer_without_attr(self):
        # A packer that does not expose the attribute must not gain one.
        packer = _NcclAwarePacker()  # no _nccl_dp_receiver_replica
        factory = _FakeRedisFactory(_FakeRedis())
        harness = _CommHarness(mode="nccl", data_packer=packer)
        harness.replica_name = "policy-XYZ"
        with self._patch_nccl_redis(factory):
            harness._attach_payload_transport()
        self.assertFalse(hasattr(packer, "_nccl_dp_receiver_replica"))


class TestSingleReceiverTopologyGuard(unittest.TestCase):
    """The nccl/ucxx single-receiver guard, through the real attach path.

    Both transports deliver a payload to exactly ONE receiver, while the policy
    hands the same rollout reference to every rank sharing a DP coordinate.  The
    guard must reject that topology at attach time -- before any transfer -- and
    must do so regardless of whether the transport attach itself would succeed.
    """

    def _patch_nccl_redis(self, fake_factory):
        return mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._redis_lib",
            SimpleNamespace(Redis=fake_factory),
        )

    def test_attach_rejects_model_parallel_policy(self):
        for mode in ("nccl", "ucxx"):
            for dim in ("tp_size", "cp_size", "pp_size"):
                with self.subTest(mode=mode, dim=dim):
                    harness = _CommHarness(
                        mode=mode,
                        data_packer=_NcclAwarePacker(),
                        parallelism={dim: 2},
                    )
                    with self.assertRaises(ValueError) as ctx:
                        harness._attach_payload_transport()
                    self.assertIn(f"{dim}=2", str(ctx.exception))

    def test_guard_raises_even_when_mode_not_explicit_is_irrelevant(self):
        # The guard runs OUTSIDE the attach try/except, so it is never
        # downgraded to a warning the way an attach ImportError/RuntimeError is
        # for a non-explicitly-selected transport.
        harness = _CommHarness(
            mode="ucxx",  # ucxx-cu12 need not be installed; guard fires first
            data_packer=_NoOpPacker(),
            parallelism={"tp_size": 8},
        )
        with self.assertRaises(ValueError):
            harness._attach_payload_transport()

    def test_attach_allows_pure_data_parallel(self):
        # tp/cp/pp all 1 -> exactly one receiver per transfer -> attach proceeds
        # normally and still wires the packer.
        packer = _NcclAwarePacker()
        fake = _FakeRedis()
        harness = _CommHarness(
            mode="nccl",
            data_packer=packer,
            parallelism={"tp_size": 1, "cp_size": 1, "pp_size": 1},
        )
        with self._patch_nccl_redis(_FakeRedisFactory(fake)):
            harness._attach_payload_transport()
        self.assertIs(packer.redis_client, fake)
        self.assertTrue(packer.post_called)

    def test_attach_allows_model_parallel_under_redis(self):
        # redis GETs are repeatable, so model parallelism is fine there.
        harness = _CommHarness(
            mode="redis",
            data_packer=_NoOpPacker(),
            parallelism={"tp_size": 8, "cp_size": 2, "pp_size": 4},
        )
        harness._attach_payload_transport()  # must not raise

    def test_ping_failure_leaves_packer_unchanged(self):
        # If Redis ping fails, the packer must not be left in a half-
        # configured state (assigned client without a working ping).
        packer = _NcclAwarePacker()
        fake = _FakeRedis(ping_raises=ConnectionError("nope"))
        factory = _FakeRedisFactory(fake)
        harness = _CommHarness(mode="nccl", data_packer=packer)
        with self._patch_nccl_redis(factory):
            harness._attach_payload_transport()
        self.assertIsNone(packer.redis_client)
        self.assertFalse(packer.post_called)

    def test_non_redis_aware_packer_untouched(self):
        # Packers that don't expose a redis_client attribute must not
        # be modified by the attach flow.
        packer = _NoOpPacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        harness = _CommHarness(mode="nccl", data_packer=packer)
        with self._patch_nccl_redis(factory):
            harness._attach_payload_transport()
        self.assertFalse(hasattr(packer, "redis_client"))
        # No client construction was attempted (NCCL attach skipped early).
        self.assertEqual(factory.calls, [])

    def test_deprecation_shim_routes_through_attach(self):
        # _inject_redis_into_data_packers must remain a one-line shim
        # that produces the same observable outcome as the new entry
        # point.  Downstream forks may still call this name.
        packer = _NcclAwarePacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        harness = _CommHarness(mode="nccl", data_packer=packer)
        with self._patch_nccl_redis(factory):
            harness._inject_redis_into_data_packers()
        self.assertIs(packer.redis_client, fake)
        self.assertTrue(packer.post_called)

    def test_explicit_mode_with_failing_attach_raises(self):
        # explicit_fatal failure policy: when the user explicitly sets
        # payload_transfer in config and the transport's attach raises
        # ImportError/RuntimeError, the failure must propagate.
        class _ExplodingTransport:
            name = "_test_explode"
            completion_prefix = None

            def attach_data_packer(self, packer, **kwargs):
                raise RuntimeError("attach exploded")

        explode = _ExplodingTransport()
        try:
            PayloadTransportRegistry.register(explode)
            packer = _NoOpPacker()
            harness = _CommHarness(mode="_test_explode", data_packer=packer)
            with self.assertRaises(RuntimeError):
                harness._attach_payload_transport()
        finally:
            PayloadTransportRegistry._registry.pop("_test_explode", None)

    def test_inferred_mode_with_failing_attach_warns(self):
        # When the active mode came from the default fallback (NOT
        # user-explicit), an attach failure must be logged-and-swallowed
        # rather than re-raised.  This is the warn-and-continue branch
        # of the explicit_fatal policy.
        class _SneakyTransport:
            """Reuses a default-resolved name ("redis") to ensure the
            mode is inferred, not explicit, from the harness config."""

            name = "redis"
            completion_prefix = None
            attach_calls = 0

            def attach_data_packer(self, packer, **kwargs):
                type(self).attach_calls += 1
                raise RuntimeError("attach exploded but should be swallowed")

        original = PayloadTransportRegistry._registry.get("redis")
        try:
            PayloadTransportRegistry._registry["redis"] = _SneakyTransport()
            # Build a harness whose config does NOT explicitly set
            # payload_transfer -- triggers the inferred-mode branch.
            harness = _CommHarness(mode=None, data_packer=_NoOpPacker())
            harness.config = SimpleNamespace(custom={})  # no explicit key
            # Must NOT raise even though attach raised.
            harness._attach_payload_transport()
            self.assertEqual(_SneakyTransport.attach_calls, 1)
        finally:
            if original is not None:
                PayloadTransportRegistry._registry["redis"] = original
            else:
                PayloadTransportRegistry._registry.pop("redis", None)

    def test_attach_visits_val_data_packer_when_distinct(self):
        # Both data packer and val_data_packer must receive the attach
        # call when they are distinct instances.
        train = _NcclAwarePacker()
        val = _NcclAwarePacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        harness = _CommHarness(mode="nccl", data_packer=train, val_data_packer=val)
        with self._patch_nccl_redis(factory):
            harness._attach_payload_transport()
        self.assertIs(train.redis_client, fake)
        self.assertIs(val.redis_client, fake)
        self.assertTrue(train.post_called)
        self.assertTrue(val.post_called)

    def test_attach_skips_val_data_packer_when_same_object(self):
        # Avoid double-attaching when val_data_packer is the same
        # instance as data_packer.
        shared = _NcclAwarePacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        harness = _CommHarness(mode="nccl", data_packer=shared, val_data_packer=shared)
        with self._patch_nccl_redis(factory):
            harness._attach_payload_transport()
        # post_redis_injection should fire exactly once even though both
        # slots point at the same object.
        self.assertIs(shared.redis_client, fake)
        # Sanity: factory was invoked exactly once for the shared packer.
        self.assertEqual(len(factory.calls), 1)


class TestOpportunisticRedisInjection(unittest.TestCase):
    """The non-NCCL compat fallback in ``_attach_payload_transport``.

    Compatibility surface (2): pre-55745c convention allowed downstream
    packers to receive a Redis client even when NCCL was not selected.
    The fallback preserves that for the redis-default mode while
    explicitly skipping NCCL mode (to avoid double-injection).
    """

    def _patch_comm_redis(self, fake_factory):
        return mock.patch(
            "cosmos_rl.comm.base._redis_lib",
            SimpleNamespace(Redis=fake_factory),
        )

    def test_fires_for_redis_default_mode(self):
        packer = _NcclAwarePacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        # Mode = "redis" (default).  Attach is no-op for the default
        # transport, so the only path that can wire redis_client is the
        # opportunistic fallback.
        harness = _CommHarness(mode="redis", data_packer=packer)
        with self._patch_comm_redis(factory):
            harness._attach_payload_transport()
        self.assertIs(packer.redis_client, fake)
        self.assertTrue(packer.post_called)

    def test_skipped_for_nccl_mode_to_avoid_double_inject(self):
        # In NCCL mode the NCCL transport's attach is responsible for
        # the assignment.  The opportunistic fallback must NOT also
        # fire, otherwise downstream packers see two different clients
        # depending on which call wins.
        packer = _NcclAwarePacker()

        # Track all client constructions across both code paths so we
        # can assert exactly one client was made.
        all_calls = []

        def _factory(**kwargs):
            all_calls.append(kwargs)
            return _FakeRedis()

        with (
            mock.patch(
                "cosmos_rl.utils.payload_transport.nccl.transport._redis_lib",
                SimpleNamespace(Redis=_factory),
            ),
            mock.patch(
                "cosmos_rl.comm.base._redis_lib",
                SimpleNamespace(Redis=_factory),
            ),
        ):
            harness = _CommHarness(mode="nccl", data_packer=packer)
            harness._attach_payload_transport()
        # Exactly one client construction (NCCL's), not two.
        self.assertEqual(len(all_calls), 1)

    def test_skipped_when_packer_already_has_client(self):
        # If something else already wired in a client, the
        # opportunistic fallback must not stomp on it.
        sentinel = object()

        class _PreWiredPacker:
            def __init__(self):
                self.redis_client = sentinel
                self.post_called = False

            def post_redis_injection(self):
                self.post_called = True

        packer = _PreWiredPacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        harness = _CommHarness(mode="redis", data_packer=packer)
        with self._patch_comm_redis(factory):
            harness._attach_payload_transport()
        # Pre-existing client preserved, no replacement.
        self.assertIs(packer.redis_client, sentinel)
        self.assertFalse(packer.post_called)


if __name__ == "__main__":
    unittest.main()
