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

These tests validate three things:

1. The default Redis backend and the new NCCL backend are registered
   out of the box.
2. ``get_payload_transfer_mode`` honors both the new
   ``[custom].payload_transfer`` string and the deprecated
   ``[custom].nccl_payload_transfer`` boolean.
3. The legacy ``cosmos_rl.utils.nccl_transfer_protocol`` module still
   re-exports the public names so external importers do not break.
"""

import unittest
from types import SimpleNamespace

from cosmos_rl.utils.payload_transport import (
    PAYLOAD_TRANSFER_KEY,
    LEGACY_NCCL_KEY,
    PayloadTransport,
    PayloadTransportRegistry,
    get_payload_transfer_mode,
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
        transport = PayloadTransportRegistry.active_for_completion(
            "nccl:0:abcdef"
        )
        self.assertIsInstance(transport, NcclPayloadTransport)

    def test_active_for_completion_returns_none_for_plain_text(self):
        self.assertIsNone(
            PayloadTransportRegistry.active_for_completion("hello world")
        )

    def test_active_for_completion_rejects_non_string(self):
        self.assertIsNone(PayloadTransportRegistry.active_for_completion(None))
        self.assertIsNone(PayloadTransportRegistry.active_for_completion(42))


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
        config = SimpleNamespace(custom={
            PAYLOAD_TRANSFER_KEY: "redis",
            LEGACY_NCCL_KEY: True,
        })
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
        ids = build_transfer_rollout_candidates(
            transfer_id="3:abcdef", num_rollout_replicas=4
        )
        self.assertEqual(ids, [3])

    def test_build_transfer_rollout_candidates_out_of_range(self):
        ids = build_transfer_rollout_candidates(
            transfer_id="9:abcdef", num_rollout_replicas=4
        )
        self.assertEqual(ids, [])

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
            rollout=SimpleNamespace(
                parallelism=SimpleNamespace(n_init_replicas=4)
            ),
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
        self.assertIn(
            "cosmos_rl:exp:test:rollout_comm:2:nccl_cleanup", channels
        )
        self.assertIn(
            "cosmos_rl:exp:test:rollout_comm:3:nccl_cleanup", channels
        )

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


if __name__ == "__main__":
    unittest.main()
