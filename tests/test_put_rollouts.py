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

"""
Regression tests for ``PolicyStatusManager.put_rollouts``.

Pins the fix for the on-policy rollout-drop deadlock: ``put_rollouts`` must
admit every rollout in an incoming batch regardless of the
``on_policy_rollout_completed`` flag, because that flag is a trainer
notification primitive, not a producer-side admission gate.

The previous buggy behaviour was:
  1. Early-return at the top of ``put_rollouts`` when the flag was already
     ``True``, dropping the entire batch.
  2. ``break`` inside the loop when the pending queue drained, dropping the
     remaining rollouts in the same batch.

Either drop, combined with the controller's prompt-dispatch racing ahead of
the trainer's flag reset, deadlocks the on-policy producer-consumer pipeline.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from cosmos_rl.dispatcher.status import PolicyStatusManager


def _make_config(on_policy: bool) -> SimpleNamespace:
    """Build the minimal config surface ``put_rollouts`` reads."""
    return SimpleNamespace(
        train=SimpleNamespace(
            train_policy=SimpleNamespace(
                on_policy=on_policy,
                rollout_as_token_ids=True,
            ),
            non_text=False,
        ),
    )


def _make_rollout(n_tokens: int = 4) -> MagicMock:
    """A rollout stub that ``put_rollouts`` can introspect.

    With ``rollout_as_token_ids=True`` ``put_rollouts`` only reads
    ``rollout.completion_token_ids``; nothing else on the rollout is touched.
    """
    rollout = MagicMock(spec=["completion_token_ids"])
    rollout.completion_token_ids = list(range(n_tokens))
    return rollout


class _ManagerFixture:
    """Wraps a real ``PolicyStatusManager`` with its heavy deps replaced.

    The methods under test (``put_rollouts``) call ``self.put_rollout`` and
    ``self.total_pending_rollouts``; both are replaced with controllable
    stand-ins so the tests can drive the pending-queue state precisely without
    standing up the full dispatcher.
    """

    def __init__(self, on_policy: bool, pending_after_each_put):
        self.manager = PolicyStatusManager()
        self.manager.config = _make_config(on_policy=on_policy)
        self.manager.remain_samples_num = 1000

        self.put_rollout_calls = []

        def fake_put_rollout(rollout):
            self.put_rollout_calls.append(rollout)

        # ``pending_after_each_put`` is the sequence of values
        # ``total_pending_rollouts`` should return on successive calls (one per
        # admitted rollout). This lets a test express e.g. "the queue drains on
        # the 2nd rollout of a 4-rollout batch".
        pending_iter = iter(pending_after_each_put)

        def fake_total_pending_rollouts():
            return next(pending_iter)

        self.manager.put_rollout = fake_put_rollout  # type: ignore[assignment]
        self.manager.total_pending_rollouts = (  # type: ignore[assignment]
            fake_total_pending_rollouts
        )


class TestPutRolloutsAdmissionGate(unittest.TestCase):
    """``put_rollouts`` must not use ``on_policy_rollout_completed`` as a gate."""

    def test_stale_flag_does_not_drop_batch(self):
        """Regression: stale flag must not drop the incoming batch.

        Failure timeline this guards against:
          - Step N's last rollout flips ``on_policy_rollout_completed=True``.
          - Controller dispatches step N+1 prompts immediately.
          - Step N+1 rollouts come back BEFORE the trainer resets the flag.
          - Buggy code early-returned and dropped them -> trainer deadlocks.
        """
        rollouts = [_make_rollout() for _ in range(3)]
        # Queue is non-empty after each put (next step is in progress; flag
        # should not flip back to True mid-batch).
        fixture = _ManagerFixture(on_policy=True, pending_after_each_put=[1, 2, 3])
        fixture.manager.on_policy_rollout_completed = True  # stale flag
        remain_before = fixture.manager.remain_samples_num

        _, n_samples = fixture.manager.put_rollouts(rollouts)

        self.assertEqual(n_samples, 3, "all rollouts must be admitted")
        self.assertEqual(len(fixture.put_rollout_calls), 3)
        self.assertEqual(
            fixture.manager.remain_samples_num,
            remain_before,
            "remain_samples_num must not be decremented as if rollouts were "
            "filtered; the buggy path did `remain_samples_num -= len(rollouts)` "
            "which destroyed budget accounting.",
        )

    def test_mid_batch_drain_does_not_break(self):
        """Regression: ``break`` after flag flip must not drop remaining rollouts.

        When the pending queue happens to drain partway through a batch, the
        buggy code set the flag AND broke out of the loop, silently discarding
        the rest of the rollouts. They are valid step-N+1 data and must be
        admitted.
        """
        rollouts = [_make_rollout() for _ in range(4)]
        # After the 2nd ``put_rollout`` the pending queue is empty (drains
        # mid-batch); the remaining two rollouts must still be admitted.
        fixture = _ManagerFixture(on_policy=True, pending_after_each_put=[2, 0, 0, 0])
        fixture.manager.on_policy_rollout_completed = False

        _, n_samples = fixture.manager.put_rollouts(rollouts)

        self.assertEqual(n_samples, 4, "no rollout in the batch may be dropped")
        self.assertEqual(len(fixture.put_rollout_calls), 4)
        self.assertEqual(fixture.put_rollout_calls, rollouts)
        self.assertTrue(
            fixture.manager.on_policy_rollout_completed,
            "flag must still be set when the pending queue drains so the "
            "trainer's notification semantics are preserved",
        )

    def test_flag_set_when_queue_drains(self):
        """Notification semantics: flag flips to True iff the queue drains."""
        rollouts = [_make_rollout() for _ in range(2)]
        # Last put empties the queue.
        fixture = _ManagerFixture(on_policy=True, pending_after_each_put=[1, 0])
        fixture.manager.on_policy_rollout_completed = False

        fixture.manager.put_rollouts(rollouts)

        self.assertTrue(fixture.manager.on_policy_rollout_completed)

    def test_flag_not_set_when_queue_nonempty(self):
        """Flag stays False as long as rollouts remain pending."""
        rollouts = [_make_rollout() for _ in range(2)]
        fixture = _ManagerFixture(on_policy=True, pending_after_each_put=[5, 4])
        fixture.manager.on_policy_rollout_completed = False

        fixture.manager.put_rollouts(rollouts)

        self.assertFalse(fixture.manager.on_policy_rollout_completed)

    def test_off_policy_admits_all_and_ignores_flag(self):
        """``on_policy=False`` runs are unaffected by the flag entirely."""
        rollouts = [_make_rollout() for _ in range(3)]
        # ``total_pending_rollouts`` is not consulted on the off-policy path,
        # but provide a value defensively in case that changes.
        fixture = _ManagerFixture(on_policy=False, pending_after_each_put=[0, 0, 0])
        # Flag set; off-policy must ignore it.
        fixture.manager.on_policy_rollout_completed = True

        _, n_samples = fixture.manager.put_rollouts(rollouts)

        self.assertEqual(n_samples, 3)
        self.assertEqual(len(fixture.put_rollout_calls), 3)
        self.assertTrue(
            fixture.manager.on_policy_rollout_completed,
            "off-policy must not mutate the flag",
        )


if __name__ == "__main__":
    unittest.main()
