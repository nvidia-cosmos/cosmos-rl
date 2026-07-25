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

"""Regression: the RL policy worker tears down its payload-transport data
packer at shutdown.

The bug this guards against: the policy trainer HUNG at exit after a full
NCCL disaggregated run, because ``shutdown_nccl_data_packer`` (which aborts
the mixin's 2-rank payload comms) was never called -- the leftover comms
wedged the NCCL / process-group teardown.  ``handle_shutdown`` now calls
``_shutdown_payload_data_packers`` first.
"""

import unittest

from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker


class _StubNCCLPacker:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown_nccl_data_packer(self):
        self.shutdown_calls += 1


class _StubUCXXPacker:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown_ucxx_data_packer(self):
        self.shutdown_calls += 1


class _PlainPacker:
    pass


def _worker():
    # Bypass the heavy __init__; we only exercise _shutdown_payload_data_packers.
    return RLPolicyWorker.__new__(RLPolicyWorker)


class TestShutdownPayloadDataPackers(unittest.TestCase):
    def test_calls_nccl_shutdown(self):
        w = _worker()
        p = _StubNCCLPacker()
        w.data_packer = p
        w.val_data_packer = p  # same object
        w._shutdown_payload_data_packers()
        self.assertEqual(p.shutdown_calls, 1)  # deduped by identity

    def test_distinct_packers_both_shut_down(self):
        w = _worker()
        w.data_packer = _StubNCCLPacker()
        w.val_data_packer = _StubNCCLPacker()
        w._shutdown_payload_data_packers()
        self.assertEqual(w.data_packer.shutdown_calls, 1)
        self.assertEqual(w.val_data_packer.shutdown_calls, 1)

    def test_ucxx_packer_also_covered(self):
        w = _worker()
        p = _StubUCXXPacker()
        w.data_packer = p
        w.val_data_packer = None
        w._shutdown_payload_data_packers()
        self.assertEqual(p.shutdown_calls, 1)

    def test_plain_packer_is_noop(self):
        w = _worker()
        w.data_packer = _PlainPacker()  # no shutdown_* methods
        w.val_data_packer = None
        w._shutdown_payload_data_packers()  # must not raise

    def test_shutdown_error_is_swallowed(self):
        class _Raiser:
            def shutdown_nccl_data_packer(self):
                raise RuntimeError("boom")

        w = _worker()
        w.data_packer = _Raiser()
        w.val_data_packer = None
        w._shutdown_payload_data_packers()  # best-effort: must not raise

    def test_handle_shutdown_invokes_packer_teardown_first(self):
        # handle_shutdown must call the packer teardown; we stop it right
        # after by making the next step (shutdown_signal) raise, and assert
        # the packer was already torn down.
        w = _worker()
        p = _StubNCCLPacker()
        w.data_packer = p
        w.val_data_packer = p

        class _Boom(Exception):
            pass

        class _Sig:
            def set(self_):
                raise _Boom

        w.shutdown_signal = _Sig()
        with self.assertRaises(_Boom):
            w.handle_shutdown()
        self.assertEqual(p.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
