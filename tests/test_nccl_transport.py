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

"""Tests for the NCCL transport backend: registration, cleanup dispatch,
and the two ``attach_data_packer`` paths (in-tree mixin vs legacy)."""

import unittest
from types import SimpleNamespace
from unittest import mock

from cosmos_rl.utils.payload_transport import PayloadTransportRegistry, RedisEndpoint
from cosmos_rl.utils.payload_transport.nccl import (
    NCCL_COMPLETION_PREFIX,
    NcclPayloadTransport,
    build_cleanup_channel,
    build_nccl_prefix,
    build_rollout_prefix,
)


class FakeRedis:
    def __init__(self):
        self.published = []

    def publish(self, channel, payload):
        self.published.append((channel, payload))
        return 1


class TestRegistration(unittest.TestCase):
    def test_registered_with_prefix(self):
        t = PayloadTransportRegistry.get("nccl")
        self.assertIsInstance(t, NcclPayloadTransport)
        self.assertEqual(t.completion_prefix, NCCL_COMPLETION_PREFIX)

    def test_active_for_completion(self):
        t = PayloadTransportRegistry.active_for_completion("nccl:0:abc")
        self.assertIsInstance(t, NcclPayloadTransport)
        self.assertIsNone(PayloadTransportRegistry.active_for_completion("plain"))
        self.assertIsNone(
            PayloadTransportRegistry.active_for_completion({"_nccl": True})
        )


class TestPublishCleanup(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            logging=SimpleNamespace(experiment_name="exp"),
            rollout=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=4)),
        )

    def test_publishes_to_rollout_cleanup_channel(self):
        redis = FakeRedis()
        t = NcclPayloadTransport()
        n = t.publish_cleanup_for_discarded(
            transfer_ids=["0:abc", "2:def"],
            config=self._config(),
            redis_client=redis,
        )
        self.assertEqual(n, 2)
        prefix = build_nccl_prefix(experiment_name="exp", job_id="test")
        ch0 = build_cleanup_channel(build_rollout_prefix(prefix, 0))
        ch2 = build_cleanup_channel(build_rollout_prefix(prefix, 2))
        channels = {c for c, _ in redis.published}
        self.assertIn(ch0, channels)
        self.assertIn(ch2, channels)

    def test_out_of_range_replica_skipped(self):
        redis = FakeRedis()
        t = NcclPayloadTransport()
        # rollout idx 9 >= n_init_replicas(4) -> no candidate -> nothing published.
        n = t.publish_cleanup_for_discarded(
            transfer_ids=["9:abc"], config=self._config(), redis_client=redis
        )
        self.assertEqual(n, 1)  # counted as handled...
        self.assertEqual(redis.published, [])  # ...but nothing published

    def test_no_redis_client_returns_zero(self):
        t = NcclPayloadTransport()
        self.assertEqual(
            t.publish_cleanup_for_discarded(
                transfer_ids=["0:a"], config=self._config(), redis_client=None
            ),
            0,
        )


class TestAttachDataPacker(unittest.TestCase):
    def test_mixin_path_preferred(self):
        t = NcclPayloadTransport()
        calls = {}

        class _MixinPacker:
            def _setup_nccl_data_packer(self, **kwargs):
                calls.update(kwargs)

        with mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client",
            return_value="client",
        ):
            t.attach_data_packer(
                _MixinPacker(),
                config=SimpleNamespace(custom={}),
                device="cuda:0",
                redis_endpoint=RedisEndpoint("h", 1),
            )
        self.assertEqual(calls["device"], "cuda:0")
        self.assertEqual(calls["redis_client"], "client")

    def test_mixin_path_raises_when_no_control_plane(self):
        # Gap 5: an unreachable/absent Redis control plane must RAISE (not
        # silently no-op), so _attach_payload_transport's explicit_fatal policy
        # can surface an EXPLICIT payload_transfer="nccl" misconfig loudly.
        t = NcclPayloadTransport()

        class _MixinPacker:
            def _setup_nccl_data_packer(self, **kwargs):  # pragma: no cover
                raise AssertionError("setup must not run without a client")

        with mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                t.attach_data_packer(
                    _MixinPacker(),
                    config=SimpleNamespace(custom={}),
                    device="cuda:0",
                    redis_endpoint=RedisEndpoint("h", 1),
                )

    def test_legacy_path_assigns_client_then_hooks(self):
        t = NcclPayloadTransport()
        order = []

        class _LegacyPacker:
            redis_client = None

            def post_redis_injection(self):
                # Client must already be assigned when the hook runs.
                order.append(("hook", self.redis_client))

        packer = _LegacyPacker()
        with mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client",
            return_value="legacy-client",
        ):
            t.attach_data_packer(
                packer,
                config=SimpleNamespace(custom={}),
                redis_endpoint=RedisEndpoint("h", 1),
            )
        self.assertEqual(packer.redis_client, "legacy-client")
        self.assertEqual(order, [("hook", "legacy-client")])

    def test_no_op_for_unrelated_packer(self):
        t = NcclPayloadTransport()

        class _Plain:
            pass

        # Neither surface present -> no error, no redis client built.
        with mock.patch(
            "cosmos_rl.utils.payload_transport.nccl.transport._build_redis_client"
        ) as build:
            t.attach_data_packer(
                _Plain(),
                config=SimpleNamespace(custom={}),
                redis_endpoint=RedisEndpoint("h", 1),
            )
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
