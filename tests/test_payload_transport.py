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

"""Tests for the payload-transport registry refactor (MR3a).

These tests validate:

1. The default Redis backend and the new NCCL backend are registered
   out of the box.
2. ``get_payload_transfer_mode`` honors both the new
   ``[custom].payload_transfer`` string and the deprecated
   ``[custom].nccl_payload_transfer`` boolean.
3. The legacy ``cosmos_rl.utils.nccl_transfer_protocol`` module still
   re-exports the public names so external importers do not break.
4. The unified worker-side hook ``PayloadTransport.attach_data_packer``
   preserves the 55745c contract for NCCL-aware data packers (assign
   ``redis_client`` then call ``post_redis_injection``) and degrades
   safely on Redis ping failure.
5. The unified controller-side dispatch
   ``PayloadTransportRegistry.handle_discarded`` partitions discards
   correctly by completion prefix, isolates per-transport failures,
   and skips transports with ``completion_prefix=None``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from cosmos_rl.utils.payload_transport import (
    PAYLOAD_TRANSFER_KEY,
    LEGACY_NCCL_KEY,
    P2P_TRANSFER_MODES,
    PayloadTransport,
    PayloadTransportRegistry,
    RedisEndpoint,
    get_payload_transfer_mode,
    is_payload_transfer_mode_explicit,
    validate_single_receiver_topology,
)
from cosmos_rl.utils.payload_transport.nccl.context import (
    DEFAULT_MAX_LIVE_COMMS,
    resolve_custom_int,
    resolve_max_live_comms,
)
from cosmos_rl.utils.payload_transport.nccl import (
    NCCL_COMPLETION_PREFIX,
    NcclPayloadTransport,
    build_cleanup_channel,
    build_nccl_prefix,
    build_rollout_prefix,
    build_transfer_rollout_candidates,
)


class TestRegistryBootstrap(unittest.TestCase):
    def test_default_redis_registered(self):
        transport = PayloadTransportRegistry.get("redis")
        self.assertIsInstance(transport, PayloadTransport)
        self.assertEqual(transport.name, "redis")
        self.assertIsNone(transport.completion_prefix)

    def test_nccl_registered(self):
        transport = PayloadTransportRegistry.get("nccl")
        self.assertIsInstance(transport, NcclPayloadTransport)
        self.assertEqual(transport.completion_prefix, NCCL_COMPLETION_PREFIX)

    def test_unknown_lookup_raises(self):
        with self.assertRaises(KeyError):
            PayloadTransportRegistry.get("definitely_not_registered")

    def test_get_optional_returns_none_for_unknown(self):
        self.assertIsNone(
            PayloadTransportRegistry.get_optional("definitely_not_registered")
        )

    def test_active_for_completion_matches_nccl_prefix(self):
        transport = PayloadTransportRegistry.active_for_completion("nccl:0:abcdef")
        self.assertIsInstance(transport, NcclPayloadTransport)

    def test_active_for_completion_returns_none_for_plain_text(self):
        self.assertIsNone(PayloadTransportRegistry.active_for_completion("hello world"))

    def test_active_for_completion_rejects_non_string(self):
        self.assertIsNone(PayloadTransportRegistry.active_for_completion(None))
        self.assertIsNone(PayloadTransportRegistry.active_for_completion(42))

    def test_active_for_completion_matches_dict_metadata_form(self):
        # NCCL packers return dict metadata whose "completion" field carries
        # the nccl: string; controller cleanup must still detect it (else the
        # producer never frees discarded buffers -- Codex P2 finding).
        meta = {"_nccl": True, "_transfer_id": "0:abc", "completion": "nccl:0:abc"}
        transport = PayloadTransportRegistry.active_for_completion(meta)
        self.assertIsInstance(transport, NcclPayloadTransport)

    def test_active_for_completion_dict_without_nccl_string_is_none(self):
        self.assertIsNone(
            PayloadTransportRegistry.active_for_completion({"observations": [1]})
        )
        self.assertIsNone(
            PayloadTransportRegistry.active_for_completion({"completion": "plain"})
        )


class TestGetPayloadTransferMode(unittest.TestCase):
    def test_default_when_custom_empty(self):
        config = SimpleNamespace(custom={})
        self.assertEqual(get_payload_transfer_mode(config), "redis")

    def test_default_when_custom_missing(self):
        config = SimpleNamespace()
        self.assertEqual(get_payload_transfer_mode(config), "redis")

    def test_string_form_redis(self):
        config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: "redis"})
        self.assertEqual(get_payload_transfer_mode(config), "redis")

    def test_string_form_nccl(self):
        config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: "nccl"})
        self.assertEqual(get_payload_transfer_mode(config), "nccl")

    def test_string_form_normalized(self):
        # Capitalization / whitespace shouldn't matter -- normalization
        # avoids spurious config-error noise.
        config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: "  NCCL  "})
        self.assertEqual(get_payload_transfer_mode(config), "nccl")

    def test_legacy_boolean_resolves_to_nccl(self):
        config = SimpleNamespace(custom={LEGACY_NCCL_KEY: True})
        self.assertEqual(get_payload_transfer_mode(config), "nccl")

    def test_legacy_boolean_false_falls_back_to_default(self):
        config = SimpleNamespace(custom={LEGACY_NCCL_KEY: False})
        self.assertEqual(get_payload_transfer_mode(config), "redis")

    def test_string_form_takes_precedence_over_legacy(self):
        # If both are set, the explicit string form wins.
        config = SimpleNamespace(
            custom={
                PAYLOAD_TRANSFER_KEY: "redis",
                LEGACY_NCCL_KEY: True,
            }
        )
        self.assertEqual(get_payload_transfer_mode(config), "redis")

    def test_unknown_string_raises(self):
        config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: "carrier_pigeon"})
        with self.assertRaises(ValueError):
            get_payload_transfer_mode(config)


class TestNcclProtocolHelpers(unittest.TestCase):
    """Lightweight regression tests for the NCCL protocol helpers."""

    def test_build_nccl_prefix(self):
        self.assertEqual(
            build_nccl_prefix(experiment_name="exp", job_id="42"),
            "cosmos_rl:exp:42",
        )

    def test_build_rollout_prefix_and_channels(self):
        prefix = build_nccl_prefix(experiment_name="exp", job_id="42")
        rprefix = build_rollout_prefix(prefix, 7)
        self.assertEqual(rprefix, "cosmos_rl:exp:42:rollout_comm:7")
        self.assertEqual(
            build_cleanup_channel(rprefix),
            "cosmos_rl:exp:42:rollout_comm:7:nccl_cleanup",
        )

    def test_build_transfer_rollout_candidates_valid(self):
        ids = build_transfer_rollout_candidates(transfer_id="3:abcdef")
        self.assertEqual(ids, [3])

    def test_build_transfer_rollout_candidates_not_bounded_by_replica_count(self):
        # A replica added mid-run has an index beyond the INITIAL count.  It
        # must still get a cleanup channel -- bounding here silently withheld
        # the publish and pinned the producer's send buffer until eviction.
        ids = build_transfer_rollout_candidates(transfer_id="9999:abcdef")
        self.assertEqual(ids, [9999])

    def test_build_transfer_rollout_candidates_invalid(self):
        ids = build_transfer_rollout_candidates(transfer_id="garbage")
        self.assertEqual(ids, [])


class TestNcclTransportPublishCleanup(unittest.TestCase):
    class _FakeRedis:
        def __init__(self):
            self.published = []

        def publish(self, channel, payload):
            self.published.append((channel, payload))

    def test_publish_cleanup_routes_to_correct_channel(self):
        transport = NcclPayloadTransport()
        redis = self._FakeRedis()
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name="exp"),
            rollout=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=4)),
        )
        n = transport.publish_cleanup_for_discarded(
            transfer_ids=["2:abc", "3:def"],
            config=config,
            redis_client=redis,
        )
        self.assertEqual(n, 2)
        # Two transfer ids → two messages, one per replica index.
        self.assertEqual(len(redis.published), 2)
        channels = {ch for ch, _ in redis.published}
        self.assertIn("cosmos_rl:exp:test:rollout_comm:2:nccl_cleanup", channels)
        self.assertIn("cosmos_rl:exp:test:rollout_comm:3:nccl_cleanup", channels)

    def test_publish_cleanup_no_redis_is_safe(self):
        transport = NcclPayloadTransport()
        n = transport.publish_cleanup_for_discarded(
            transfer_ids=["0:abc"], config=SimpleNamespace(), redis_client=None
        )
        self.assertEqual(n, 0)

    def test_publish_cleanup_empty_list_skips_redis(self):
        transport = NcclPayloadTransport()
        redis = self._FakeRedis()
        n = transport.publish_cleanup_for_discarded(
            transfer_ids=[], config=SimpleNamespace(), redis_client=redis
        )
        self.assertEqual(n, 0)
        self.assertEqual(redis.published, [])


class TestBackwardCompatShim(unittest.TestCase):
    def test_legacy_module_reexports(self):
        # ``cosmos_rl.utils.nccl_transfer_protocol`` was renamed to
        # ``cosmos_rl.utils.payload_transport.nccl`` -- the old path
        # must continue to work for existing importers.
        import cosmos_rl.utils.nccl_transfer_protocol as legacy

        self.assertEqual(legacy.NCCL_COMPLETION_PREFIX, "nccl:")
        self.assertEqual(
            legacy.build_nccl_prefix(experiment_name="exp", job_id="42"),
            "cosmos_rl:exp:42",
        )
        # Same class instance is reachable through both paths.
        self.assertIs(legacy.NcclPayloadTransport, NcclPayloadTransport)


# ---------------------------------------------------------------------------
# MR3a unification tests
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Minimal redis.Redis substitute that records ping/publish calls."""

    def __init__(self, *, ping_raises=None):
        self._ping_raises = ping_raises
        self.pinged = False
        self.published = []

    def ping(self):
        self.pinged = True
        if self._ping_raises is not None:
            raise self._ping_raises
        return True

    def publish(self, channel, payload):
        self.published.append((channel, payload))


class _FakeRedisFactory:
    """Stand-in for ``redis.Redis`` that returns a configured fake."""

    def __init__(self, fake):
        self.fake = fake
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.fake


class _NcclAwarePacker:
    """Mirrors the 55745c contract: redis_client + post_redis_injection."""

    def __init__(self):
        self.redis_client = None
        self.post_called = False
        self.client_at_post_call = None

    def post_redis_injection(self):
        # Capture state at hook time to assert the assignment-then-hook
        # ordering in tests.
        self.client_at_post_call = self.redis_client
        self.post_called = True


class _NoOpPacker:
    """Packer with no redis_client attribute -- attach must be a no-op."""


class TestNcclAttachDataPacker(unittest.TestCase):
    """Regression guards for the 55745c NCCL packer contract."""

    def setUp(self):
        self.endpoint = RedisEndpoint(host="r.example", port=6380, db=2)
        self.config = SimpleNamespace(custom={})
        self.transport = NcclPayloadTransport()

    def _patch_redis_lib(self, fake_factory):
        return mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._redis_lib",
            SimpleNamespace(Redis=fake_factory),
        )

    def test_attach_assigns_redis_client_then_calls_post_hook(self):
        # Regression guard for the 55745c contract: assignment must
        # complete before post_redis_injection() is called.
        packer = _NcclAwarePacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        with self._patch_redis_lib(factory):
            self.transport.attach_data_packer(
                packer,
                config=self.config,
                redis_endpoint=self.endpoint,
            )
        self.assertIs(packer.redis_client, fake)
        self.assertTrue(packer.post_called)
        # Most important assertion: the hook ran AFTER assignment.
        self.assertIs(packer.client_at_post_call, fake)
        # Endpoint was passed through.
        self.assertEqual(factory.calls[0]["host"], "r.example")
        self.assertEqual(factory.calls[0]["port"], 6380)
        self.assertEqual(factory.calls[0]["db"], 2)
        # decode_responses is True for parity with pre-55745c behavior.
        self.assertTrue(factory.calls[0]["decode_responses"])

    def test_attach_skips_when_no_redis_attr(self):
        # Non-NCCL-aware packer is not modified.
        packer = _NoOpPacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        with self._patch_redis_lib(factory):
            self.transport.attach_data_packer(
                packer,
                config=self.config,
                redis_endpoint=self.endpoint,
            )
        self.assertFalse(hasattr(packer, "redis_client"))
        # No client construction attempted.
        self.assertEqual(factory.calls, [])

    def test_attach_skips_on_ping_failure(self):
        # Ping failure must NOT crash and must NOT mutate the packer.
        packer = _NcclAwarePacker()
        fake = _FakeRedis(ping_raises=ConnectionError("nope"))
        factory = _FakeRedisFactory(fake)
        with self._patch_redis_lib(factory):
            self.transport.attach_data_packer(
                packer,
                config=self.config,
                redis_endpoint=self.endpoint,
            )
        self.assertIsNone(packer.redis_client)
        self.assertFalse(packer.post_called)

    def test_attach_no_endpoint_is_safe(self):
        packer = _NcclAwarePacker()
        self.transport.attach_data_packer(
            packer, config=self.config, redis_endpoint=None
        )
        self.assertIsNone(packer.redis_client)

    def test_attach_no_redis_lib_is_safe(self):
        packer = _NcclAwarePacker()
        with mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._redis_lib", None
        ):
            self.transport.attach_data_packer(
                packer, config=self.config, redis_endpoint=self.endpoint
            )
        self.assertIsNone(packer.redis_client)

    def test_attach_post_hook_exception_is_logged_not_raised(self):
        class _BadPacker:
            def __init__(self):
                self.redis_client = None

            def post_redis_injection(self):
                raise RuntimeError("boom in hook")

        packer = _BadPacker()
        fake = _FakeRedis()
        factory = _FakeRedisFactory(fake)
        with self._patch_redis_lib(factory):
            # Must NOT raise -- a buggy downstream hook cannot kill init.
            self.transport.attach_data_packer(
                packer, config=self.config, redis_endpoint=self.endpoint
            )
        # Client is still assigned (assignment happens before the hook).
        self.assertIs(packer.redis_client, fake)


class TestHandleDiscarded(unittest.TestCase):
    """Centralized controller-side cleanup dispatch."""

    def test_returns_zero_for_no_discards(self):
        rollouts = [SimpleNamespace(completion="nccl:0:abc")]
        published = PayloadTransportRegistry.handle_discarded(
            rollouts,
            rollouts,  # nothing filtered out
            config=SimpleNamespace(),
            redis_client=_FakeRedis(),
        )
        self.assertEqual(published, 0)

    def test_returns_zero_for_empty_rollouts(self):
        published = PayloadTransportRegistry.handle_discarded(
            [], [], config=SimpleNamespace(), redis_client=_FakeRedis()
        )
        self.assertEqual(published, 0)

    def test_partitions_discards_by_completion_prefix(self):
        # Two discarded NCCL rollouts plus one untracked discard.
        rollouts = [
            SimpleNamespace(completion="nccl:0:keep"),
            SimpleNamespace(completion="nccl:0:drop1"),
            SimpleNamespace(completion="nccl:1:drop2"),
            SimpleNamespace(completion="plaintext-discard"),
        ]
        filtered = [rollouts[0]]  # only the first survives
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name="exp"),
            rollout=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=4)),
        )
        redis = _FakeRedis()
        published = PayloadTransportRegistry.handle_discarded(
            rollouts, filtered, config=config, redis_client=redis
        )
        # NCCL routes 2 discards (its prefix matches 2 entries).  Plain
        # text gets no transport, so it's silently ignored.
        self.assertEqual(published, 2)
        self.assertEqual(len(redis.published), 2)

    def test_skips_transports_with_none_completion_prefix(self):
        # Register a transport with completion_prefix=None and one
        # discard whose completion is a dict (UCXX-style).  Neither
        # should match; published should be zero.
        class _NoPrefixTransport(PayloadTransport):
            name = "noprefix"
            completion_prefix = None

            def __init__(self):
                self.called = False

            def publish_cleanup_for_discarded(self, **kwargs):
                self.called = True
                return 5  # would be counted if it were reached

        np_transport = _NoPrefixTransport()
        try:
            PayloadTransportRegistry.register(np_transport)
            rollouts = [
                SimpleNamespace(completion="keep"),
                SimpleNamespace(completion={"_ucxx": True, "_slot": 3}),
            ]
            published = PayloadTransportRegistry.handle_discarded(
                rollouts,
                [rollouts[0]],
                config=SimpleNamespace(),
                redis_client=_FakeRedis(),
            )
            self.assertEqual(published, 0)
            self.assertFalse(np_transport.called)
        finally:
            # Clean up to avoid contaminating other tests.
            PayloadTransportRegistry._registry.pop("noprefix", None)

    def test_continues_after_per_transport_failure(self):
        # Register a "broken" prefix that always raises; ensure NCCL
        # still gets a chance to publish for its own discards.
        class _BoomTransport(PayloadTransport):
            name = "boom"
            completion_prefix = "boom:"

            def publish_cleanup_for_discarded(self, **kwargs):
                raise RuntimeError("simulated transport failure")

        boom = _BoomTransport()
        try:
            PayloadTransportRegistry.register(boom)
            rollouts = [
                SimpleNamespace(completion="nccl:0:abc"),
                SimpleNamespace(completion="boom:99"),
            ]
            config = SimpleNamespace(
                logging=SimpleNamespace(experiment_name="exp"),
                rollout=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=4)),
            )
            redis = _FakeRedis()
            published = PayloadTransportRegistry.handle_discarded(
                rollouts, [], config=config, redis_client=redis
            )
            # NCCL's one discard still got published; boom failed cleanly.
            self.assertEqual(published, 1)
            self.assertEqual(len(redis.published), 1)
        finally:
            PayloadTransportRegistry._registry.pop("boom", None)


class TestRegistryHygiene(unittest.TestCase):
    def test_register_class_validates_name(self):
        class _Empty(PayloadTransport):
            name = ""

        with self.assertRaises(ValueError):
            PayloadTransportRegistry.register_class(_Empty)

    def test_register_validates_name(self):
        class _Empty(PayloadTransport):
            name = ""

        with self.assertRaises(ValueError):
            PayloadTransportRegistry.register(_Empty())

    def test_re_registration_replaces_instance(self):
        class _T(PayloadTransport):
            name = "_test_re_reg"

        a = _T()
        b = _T()
        try:
            PayloadTransportRegistry.register(a)
            self.assertIs(PayloadTransportRegistry.get("_test_re_reg"), a)
            PayloadTransportRegistry.register(b)
            self.assertIs(PayloadTransportRegistry.get("_test_re_reg"), b)
        finally:
            PayloadTransportRegistry._registry.pop("_test_re_reg", None)


class TestIsPayloadTransferModeExplicit(unittest.TestCase):
    def test_default_is_not_explicit(self):
        # Default fallback to "redis" is NOT explicit.
        self.assertFalse(is_payload_transfer_mode_explicit(SimpleNamespace(custom={})))
        self.assertFalse(is_payload_transfer_mode_explicit(SimpleNamespace()))

    def test_string_form_is_explicit(self):
        config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: "nccl"})
        self.assertTrue(is_payload_transfer_mode_explicit(config))

    def test_legacy_boolean_is_explicit(self):
        config = SimpleNamespace(custom={LEGACY_NCCL_KEY: True})
        self.assertTrue(is_payload_transfer_mode_explicit(config))

    def test_legacy_boolean_false_is_not_explicit(self):
        config = SimpleNamespace(custom={LEGACY_NCCL_KEY: False})
        self.assertFalse(is_payload_transfer_mode_explicit(config))

    def test_empty_string_is_not_explicit(self):
        config = SimpleNamespace(custom={PAYLOAD_TRANSFER_KEY: "   "})
        self.assertFalse(is_payload_transfer_mode_explicit(config))


def _cfg_parallelism(*, tp_size=1, cp_size=1, pp_size=1, **extra):
    """Config exposing only ``policy.parallelism`` (transport mode is passed
    to the validator directly, so ``custom`` is irrelevant here)."""
    return SimpleNamespace(
        policy=SimpleNamespace(
            parallelism=SimpleNamespace(
                tp_size=tp_size, cp_size=cp_size, pp_size=pp_size, **extra
            )
        )
    )


class TestValidateSingleReceiverTopology(unittest.TestCase):
    """nccl/ucxx deliver each payload to exactly ONE receiver, but the policy
    hands the same rollout reference to every rank sharing a DP coordinate
    (``cp * tp * pp`` of them).  Any non-DP policy parallelism must fail loud
    at startup rather than dropping episodes and hanging mid-run."""

    def test_pure_data_parallel_is_allowed(self):
        for mode in P2P_TRANSFER_MODES:
            with self.subTest(mode=mode):
                validate_single_receiver_topology(_cfg_parallelism(), mode)

    def test_tensor_parallel_rejected(self):
        for mode in P2P_TRANSFER_MODES:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as ctx:
                    validate_single_receiver_topology(_cfg_parallelism(tp_size=2), mode)
                self.assertIn("tp_size=2", str(ctx.exception))

    def test_context_parallel_rejected(self):
        # cp is NOT covered by a tp_size==1 check -- it splits the same batch
        # across ranks just like tp, so it must be caught too.
        with self.assertRaises(ValueError) as ctx:
            validate_single_receiver_topology(_cfg_parallelism(cp_size=2), "nccl")
        self.assertIn("cp_size=2", str(ctx.exception))

    def test_pipeline_parallel_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_single_receiver_topology(_cfg_parallelism(pp_size=4), "ucxx")
        self.assertIn("pp_size=4", str(ctx.exception))

    def test_error_reports_every_offending_dim_and_receiver_count(self):
        with self.assertRaises(ValueError) as ctx:
            validate_single_receiver_topology(
                _cfg_parallelism(tp_size=2, cp_size=2, pp_size=2), "nccl"
            )
        msg = str(ctx.exception)
        for field in ("tp_size=2", "cp_size=2", "pp_size=2"):
            self.assertIn(field, msg)
        # cp * tp * pp = 8 ranks would race for one payload.
        self.assertIn("8 ranks", msg)
        # Actionable: names both escape hatches.
        self.assertIn("redis", msg)

    def test_redis_is_unaffected(self):
        # A redis GET is non-destructive/repeatable, so model parallelism is
        # fine there -- the guard must not fire.
        validate_single_receiver_topology(
            _cfg_parallelism(tp_size=8, cp_size=2, pp_size=4), "redis"
        )

    def test_missing_policy_config_is_tolerated(self):
        # Rollout/reference workers may carry a config without policy dims;
        # the guard must not crash on them.
        validate_single_receiver_topology(SimpleNamespace(), "nccl")
        validate_single_receiver_topology(SimpleNamespace(policy=None), "nccl")

    def test_absent_or_none_dims_default_to_one(self):
        # Sparse configs (missing keys, explicit None) must read as 1, not fail.
        cfg = SimpleNamespace(policy=SimpleNamespace(parallelism=SimpleNamespace()))
        validate_single_receiver_topology(cfg, "nccl")
        cfg_none = SimpleNamespace(
            policy=SimpleNamespace(
                parallelism=SimpleNamespace(tp_size=None, cp_size=None, pp_size=None)
            )
        )
        validate_single_receiver_topology(cfg_none, "nccl")


class TestCommCapSizing(unittest.TestCase):
    """``max_live`` is resource hygiene, not a correctness bound, and the comm
    working set is topology-bounded -- so sizing it blindly is what makes LRU
    eviction dangerous.  Derive it from the actual peer fan-out instead."""

    @staticmethod
    def _cfg(*, peers=None, custom=None, role="policy"):
        cfg = SimpleNamespace(custom=custom or {})
        if peers is not None:
            setattr(
                cfg,
                role,
                SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=peers)),
            )
        return cfg

    def test_defaults_to_historical_cap_without_config(self):
        self.assertEqual(
            resolve_max_live_comms(SimpleNamespace(), peer_role="policy"),
            DEFAULT_MAX_LIVE_COMMS,
        )

    def test_small_fan_out_never_lowers_the_cap(self):
        # Auto-sizing must only ever RAISE the ceiling -- a 2-replica job must
        # not end up with a cap of 8.
        self.assertEqual(
            resolve_max_live_comms(self._cfg(peers=2), peer_role="policy"),
            DEFAULT_MAX_LIVE_COMMS,
        )

    def test_large_fan_out_raises_the_cap_above_the_peer_count(self):
        # 200 peers would thrash a cap of 128: cycling over N+1 keys with
        # capacity N is the pathological LRU case (~100% miss).
        self.assertEqual(
            resolve_max_live_comms(self._cfg(peers=200), peer_role="policy"), 800
        )

    def test_reads_the_role_that_owns_the_peers(self):
        # A producer's peers are policy replicas; a consumer's are rollout ones.
        cfg = self._cfg(peers=500, role="rollout")
        self.assertEqual(resolve_max_live_comms(cfg, peer_role="rollout"), 2000)
        self.assertEqual(
            resolve_max_live_comms(cfg, peer_role="policy"), DEFAULT_MAX_LIVE_COMMS
        )

    def test_explicit_custom_override_wins(self):
        cfg = self._cfg(peers=500, custom={"nccl_max_live_comms": 64})
        self.assertEqual(resolve_max_live_comms(cfg, peer_role="policy"), 64)

    def test_unparseable_override_falls_back_to_derivation(self):
        cfg = self._cfg(peers=200, custom={"nccl_max_live_comms": "not-a-number"})
        self.assertEqual(resolve_max_live_comms(cfg, peer_role="policy"), 800)


class TestResolveCustomInt(unittest.TestCase):
    def test_reads_and_clamps(self):
        cfg = SimpleNamespace(custom={"nccl_num_sender_threads": 8})
        self.assertEqual(resolve_custom_int(cfg, "nccl_num_sender_threads", 2), 8)
        # Clamped so a 0/negative config cannot disable the sender pool.
        cfg0 = SimpleNamespace(custom={"nccl_num_sender_threads": 0})
        self.assertEqual(resolve_custom_int(cfg0, "nccl_num_sender_threads", 2), 1)

    def test_missing_or_bad_values_use_the_default(self):
        self.assertEqual(resolve_custom_int(SimpleNamespace(), "k", 3), 3)
        self.assertEqual(resolve_custom_int(SimpleNamespace(custom={}), "k", 3), 3)
        self.assertEqual(
            resolve_custom_int(SimpleNamespace(custom={"k": "twelve"}), "k", 3), 3
        )


class TestPayloadTransportDefaults(unittest.TestCase):
    def test_default_attach_is_noop(self):
        # Default base-class attach must not raise on any packer shape.
        class _T(PayloadTransport):
            name = "_default_attach"

        t = _T()
        # Should silently no-op for arbitrary packers.
        t.attach_data_packer(object(), config=SimpleNamespace())
        t.attach_data_packer(_NcclAwarePacker(), config=SimpleNamespace())

    def test_default_publish_cleanup_returns_zero(self):
        class _T(PayloadTransport):
            name = "_default_publish"

        t = _T()
        self.assertEqual(
            t.publish_cleanup_for_discarded(
                transfer_ids=["a", "b"],
                config=SimpleNamespace(),
                redis_client=_FakeRedis(),
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
