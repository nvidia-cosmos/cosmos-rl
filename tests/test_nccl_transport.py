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

"""Tests for the NCCL transport backend.

Scoped to behaviour unique to this module.  Registration, ``active_for_completion``
dispatch, the legacy attach path, and the no-op-for-unrelated-packer contract are
covered once in ``test_payload_transport.py`` with stronger assertions, so they
are deliberately not duplicated here.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from cosmos_rl.utils.payload_transport import RedisEndpoint
from cosmos_rl.utils.payload_transport.nccl import NcclPayloadTransport


class FakeRedis:
    def __init__(self):
        self.published = []

    def publish(self, channel, payload):
        self.published.append((channel, payload))
        return 1


class TestPublishCleanup(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            logging=SimpleNamespace(experiment_name="exp"),
            rollout=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=4)),
        )

    def test_replica_beyond_initial_count_still_gets_cleanup(self):
        # Elastic scale-up: rollout idx 9 is past n_init_replicas(4).  Cleanup
        # used to be withheld for exactly these transfers, so the producer held
        # its send buffer until capacity eviction.  Publishing to a channel
        # nobody subscribes to is a harmless no-op; withholding pins GPU memory.
        redis = FakeRedis()
        t = NcclPayloadTransport()
        n = t.publish_cleanup_for_discarded(
            transfer_ids=["9:abc"], config=self._config(), redis_client=redis
        )
        self.assertEqual(n, 1)
        self.assertEqual(len(redis.published), 1)
        channel, payload = redis.published[0]
        self.assertIn("rollout_comm:9", channel)
        self.assertIn("9:abc", payload)


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
