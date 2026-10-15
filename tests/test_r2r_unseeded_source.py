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

"""An R2R source must hold real weights before it broadcasts them.

The controller chooses the R2R source from
``weights_loaded_in_view_of_command``, which records that a P2R was *published*
and is never cleared when one fails.  ``WeightSyncThread._run`` turns a failed
P2R into a log line and then executes the next queued command, so the source
would go on to broadcast a buffer that was never seeded.  Every destination
accepts it, sets the sticky ``weight_synced`` bit, and generates against base
weights while reporting the current version -- silently.

These tests pin the guard that stops it, and the cancellation path that keeps
the peers from paying the 120 s barrier timeout plus a full
``COSMOS_NCCL_TIMEOUT_MS`` inside ``ncclBroadcast``.
"""

import unittest
from unittest import mock

from cosmos_rl.rollout.worker import weight_sync as ws


class FakeRedis:
    """Enough of the redis client for the barrier: keys plus one pub/sub."""

    def __init__(self):
        self.store = {}
        self.published = []
        self.counters = {}
        self._queued_messages = []
        self.incr_calls = 0

    # --- key/value ---
    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def expire(self, key, ttl):
        return True

    def incr(self, key):
        self.incr_calls += 1
        self.counters[key] = self.counters.get(key, 0) + 1
        self.store[key] = str(self.counters[key])
        return self.counters[key]

    # --- pub/sub ---
    def publish(self, channel, payload):
        self.published.append((channel, payload))

    def pubsub(self):
        return FakePubSub(self)


class FakePubSub:
    def __init__(self, client):
        self.client = client
        self.subscribed = []
        self.closed = False

    def subscribe(self, channel):
        self.subscribed.append(channel)

    def unsubscribe(self, channel):
        pass

    def close(self):
        self.closed = True

    def get_message(self, timeout=None):
        if self.client._queued_messages:
            return self.client._queued_messages.pop(0)
        return None


class FakeWorker:
    def __init__(self, replica_name="rollout-a", buffer_version=1):
        self.replica_name = replica_name
        self._buffer_version = buffer_version
        self._r2r_redis = FakeRedis()
        self._r2r_barrier_prefix = "cosmos:r2r"
        self._r2r_world_size = 3
        self.replica_name_to_rank = {"rollout-a": 0, "rollout-b": 1, "rollout-c": 2}
        self.device = "cpu"


class FakeCommand:
    def __init__(self, src="rollout-a", dsts=("rollout-a", "rollout-b", "rollout-c")):
        self.src_replica_name = src
        self.dst_replica_names = list(dsts)
        self.weight_step = 7
        self.total_steps = 20

    def replica_should_stop(self):
        return False


def make_thread(worker):
    """A WeightSyncThread without starting its thread or touching CUDA."""
    thread = ws.WeightSyncThread.__new__(ws.WeightSyncThread)
    thread._worker = worker
    thread._p2r_failed = False
    thread._task_failed = False
    return thread


def make_last_arriver(worker, weight_step=7, world_size=3):
    """Pre-seed the barrier counter so this worker completes the quorum.

    ``r2r_barrier`` returns as soon as the last arriver publishes the go
    signal.  A non-final arriver instead waits out the real
    ``_R2R_BARRIER_TIMEOUT_S`` and only then proceeds, so a test that cares
    about the abort check rather than the wait would take two minutes and
    reach its assertion down the timeout path instead of the intended one.
    """
    barrier_key, _, _ = ws._round_keys(worker._r2r_barrier_prefix, weight_step)
    worker._r2r_redis.counters[barrier_key] = world_size - 1


class TestTheSourceRefusesToBroadcastWeightsItDoesNotHave(unittest.TestCase):
    def test_never_seeded_source_cancels_the_round(self):
        worker = FakeWorker(buffer_version=0)
        thread = make_thread(worker)
        with self.assertRaises(ws.R2RAborted) as caught:
            thread._assert_seeded_before_broadcast(FakeCommand(), 7)
        self.assertIn("no weight sync has ever completed", str(caught.exception))

    def test_source_whose_p2r_failed_this_round_cancels(self):
        worker = FakeWorker(buffer_version=5)
        thread = make_thread(worker)
        thread._p2r_failed = True
        with self.assertRaises(ws.R2RAborted) as caught:
            thread._assert_seeded_before_broadcast(FakeCommand(), 7)
        self.assertIn("P2R staging this round failed", str(caught.exception))

    def test_a_seeded_source_proceeds(self):
        thread = make_thread(FakeWorker(buffer_version=3))
        thread._assert_seeded_before_broadcast(FakeCommand(), 7)

    def test_destinations_are_not_judged_by_their_own_buffer(self):
        """A destination has no weights yet -- that is what R2R is for."""
        worker = FakeWorker(replica_name="rollout-b", buffer_version=0)
        thread = make_thread(worker)
        thread._p2r_failed = True
        thread._assert_seeded_before_broadcast(FakeCommand(src="rollout-a"), 7)

    def test_the_cancellation_reaches_redis(self):
        worker = FakeWorker(buffer_version=0)
        thread = make_thread(worker)
        with self.assertRaises(ws.R2RAborted):
            thread._assert_seeded_before_broadcast(FakeCommand(), 7)
        # The marker is scoped to the source: the controller re-issues at the
        # same step off a different replica, and a step-only key would cancel
        # that round too (job 2143219).
        self.assertIn("cosmos:r2r:abort:7", worker._r2r_redis.store)
        self.assertEqual(
            worker._r2r_redis.published, [("cosmos:r2r:go:7", ws._R2R_ABORT_SIGNAL)]
        )


class TestASingleMemberRoundIsNotCancelled(unittest.TestCase):
    """One replica broadcasts to nobody, so there is no peer to protect.

    ``_execute_r2r`` skips both the barrier and the collective when the
    recipient set has one member; cancelling there would break a legitimate
    round -- it only bumps the local version and validation bookkeeping.
    """

    def test_lone_replica_with_an_empty_buffer_still_proceeds(self):
        worker = FakeWorker(buffer_version=0)
        worker.current_weight_version = 0
        worker.state = mock.MagicMock()
        worker.state.weight_synced.return_value = False
        worker.config = mock.MagicMock()
        worker.config.validation.enable = False
        thread = make_thread(worker)
        thread._stream = object()
        thread._queue = mock.MagicMock()
        thread._queue.qsize.return_value = 0
        thread._executed = 0
        command = FakeCommand(src="rollout-a", dsts=("rollout-a",))
        with (
            mock.patch.object(ws, "r2r_barrier") as barrier,
            mock.patch.object(ws, "do_nccl_broadcast_grouped") as broadcast,
            mock.patch("torch.cuda.Event"),
        ):
            thread._execute_r2r(command)
        barrier.assert_not_called()
        broadcast.assert_not_called()
        self.assertEqual(worker._buffer_version, 1)


class TestNoNcclWorkIsLaunchedForACancelledRound(unittest.TestCase):
    def test_execute_r2r_aborts_before_the_barrier_and_the_broadcast(self):
        worker = FakeWorker(buffer_version=0)
        thread = make_thread(worker)
        with (
            mock.patch.object(ws, "r2r_barrier") as barrier,
            mock.patch.object(ws, "do_nccl_broadcast_grouped") as broadcast,
        ):
            with self.assertRaises(ws.R2RAborted):
                thread._execute_r2r(FakeCommand())
        barrier.assert_not_called()
        broadcast.assert_not_called()


class TestWaitingWorkersLearnQuickly(unittest.TestCase):
    def test_barrier_refuses_a_round_already_marked_aborted(self):
        worker = FakeWorker()
        worker._r2r_redis.store["cosmos:r2r:abort:7"] = (
            ws._R2R_ABORT_MARKER + "source had no weights"
        )
        with self.assertRaises(ws.R2RAborted) as caught:
            ws.r2r_barrier(worker, 7, expected_world_size=3)
        self.assertIn("source had no weights", str(caught.exception))

    def test_an_aborted_round_does_not_count_the_worker_towards_go(self):
        worker = FakeWorker()
        worker._r2r_redis.store["cosmos:r2r:abort:7"] = ws._R2R_ABORT_MARKER + "nope"
        with self.assertRaises(ws.R2RAborted):
            ws.r2r_barrier(worker, 7, expected_world_size=3)
        self.assertEqual(worker._r2r_redis.incr_calls, 0)

    def test_abort_published_while_waiting_wakes_the_worker(self):
        worker = FakeWorker()
        redis = worker._r2r_redis
        redis.store["cosmos:r2r:abort:7"] = (
            ws._R2R_ABORT_MARKER + "source had no weights"
        )
        redis._queued_messages.append(
            {"type": "message", "data": ws._R2R_ABORT_SIGNAL.encode()}
        )
        # Not yet marked when the barrier first looks; only the message carries it.
        real_get = redis.get
        first = {"n": 0}

        def get_hiding_the_key_at_first(key):
            if key.startswith("cosmos:r2r:abort:7") and first["n"] < 2:
                first["n"] += 1
                return None
            return real_get(key)

        redis.get = get_hiding_the_key_at_first
        with self.assertRaises(ws.R2RAborted):
            ws.r2r_barrier(worker, 7, expected_world_size=3)

    def test_abort_landing_between_the_check_and_subscribe_is_caught(self):
        worker = FakeWorker()
        redis = worker._r2r_redis
        real_get = redis.get
        state = {"seen": 0}

        def get_that_appears_after_the_first_look(key):
            if key.startswith("cosmos:r2r:abort:7"):
                state["seen"] += 1
                if state["seen"] == 1:
                    return None
                return ws._R2R_ABORT_MARKER + "raced in"
            return real_get(key)

        redis.get = get_that_appears_after_the_first_look
        with self.assertRaises(ws.R2RAborted) as caught:
            ws.r2r_barrier(worker, 7, expected_world_size=3)
        self.assertIn("raced in", str(caught.exception))


class TestOnlyAnAbortRecordCancelsARound(unittest.TestCase):
    """A GET answered for a key that was never set must not cancel the round.

    Caught on the Slurm CI run: a redis double whose ``get`` ignores the key
    returned the barrier counter for the abort key, and an ``is not None``
    check read that as a cancellation, failing three pre-existing barrier
    tests. Requiring the marker makes the check about the value, not the key.
    """

    def test_a_client_answering_every_get_does_not_cancel(self):
        worker = FakeWorker()
        worker._r2r_redis.get = lambda key: 2  # the barrier count, for any key
        make_last_arriver(worker)
        self.assertTrue(ws.r2r_barrier(worker, 7, expected_world_size=3))

    def test_an_unmarked_value_is_not_an_abort(self):
        worker = FakeWorker()
        worker._r2r_redis.store["cosmos:r2r:abort:7"] = "some other writer"
        make_last_arriver(worker)
        self.assertTrue(ws.r2r_barrier(worker, 7, expected_world_size=3))

    def test_the_stored_record_carries_the_marker(self):
        worker = FakeWorker()
        ws.abort_r2r_round(worker, 7, "because")
        stored = worker._r2r_redis.store["cosmos:r2r:abort:7"]
        self.assertTrue(stored.startswith(ws._R2R_ABORT_MARKER))
        self.assertIn("because", stored)


class TestARedisOutageStillDoesNotBlockWeightSync(unittest.TestCase):
    """Pre-existing behaviour: a broken Redis must not stop the round."""

    def test_barrier_still_proceeds_when_redis_raises(self):
        worker = FakeWorker()

        def boom(*a, **k):
            raise ConnectionError("redis down")

        worker._r2r_redis.get = boom
        make_last_arriver(worker)
        self.assertTrue(ws.r2r_barrier(worker, 7, expected_world_size=3))

    def test_publishing_an_abort_without_redis_is_not_fatal(self):
        worker = FakeWorker()
        worker._r2r_redis = None
        self.assertFalse(ws.abort_r2r_round(worker, 7, "reason"))

    def test_publishing_an_abort_survives_a_redis_error(self):
        worker = FakeWorker()

        def boom(*a, **k):
            raise ConnectionError("redis down")

        worker._r2r_redis.set = boom
        self.assertFalse(ws.abort_r2r_round(worker, 7, "reason"))


class TestACancelledRoundFailsTheJob(unittest.TestCase):
    """Drives the real ``_run`` loop, not a copy of its except clauses.

    A cancelled round is terminal. Waiting does not help: the cancelled round
    is exactly what stops the trainer reaching the next sync boundary, so the
    job stalls silently while holding its allocation -- on job 2142899 the
    last completed sync was step 6 and the controller then soft-throttled out
    to 195s until it was killed. Failing here keeps the cause attributable.
    """

    def _run_once_raising(self, exc):
        import queue as queue_mod
        import threading

        worker = FakeWorker()
        worker.shutdown_signal = threading.Event()
        worker.shutdown_mp_signal = threading.Event()
        thread = make_thread(worker)
        thread._queue = queue_mod.PriorityQueue()
        thread._stop = threading.Event()
        thread._idle = threading.Event()
        thread._seq = 0

        def explode(command):
            # Stop after this one command so ``_run`` returns.
            thread._stop.set()
            raise exc

        thread._execute_r2r = explode
        thread._queue.put((1, 1, ("r2r", FakeCommand())))
        with mock.patch("torch.cuda.set_device"):
            ws.WeightSyncThread._run(thread)
        return thread, worker

    def test_a_cancellation_brings_the_worker_down(self):
        _, worker = self._run_once_raising(ws.R2RAborted("cancelled"))
        self.assertTrue(worker.shutdown_signal.is_set())
        self.assertTrue(worker.shutdown_mp_signal.is_set())

    def test_cancellation_does_not_latch_a_failure(self):
        """The latch would cost the next mesh rebuild a spurious discard."""
        thread, _ = self._run_once_raising(ws.R2RAborted("cancelled"))
        self.assertFalse(thread._task_failed)

    def test_an_ordinary_error_latches_and_does_not_fail_the_job(self):
        """Only a cancelled round is terminal; other failures stay recoverable."""
        thread, worker = self._run_once_raising(RuntimeError("nccl exploded"))
        self.assertTrue(thread._task_failed)
        self.assertFalse(worker.shutdown_signal.is_set())

    def test_a_worker_without_shutdown_signals_does_not_raise(self):
        worker = FakeWorker()
        ws._fail_the_job(worker)


class TestP2ROutcomeIsRecordedSeparatelyFromTheStickyFlag(unittest.TestCase):
    def _thread_with_p2r(self, recv):
        worker = FakeWorker()
        worker._execute_p2r_recv = recv
        worker.current_weight_version = 0
        thread = make_thread(worker)
        thread._stream = mock.MagicMock()
        thread._executed = 0
        thread._queue = mock.MagicMock()
        thread._queue.qsize.return_value = 0
        return thread

    def test_a_failed_p2r_is_recorded(self):
        def boom(command, stream):
            raise RuntimeError("p2r failed")

        thread = self._thread_with_p2r(boom)
        with mock.patch("torch.cuda.Event"):
            with self.assertRaises(RuntimeError):
                thread._execute_p2r(FakeCommand())
        self.assertTrue(thread._p2r_failed)

    def test_a_successful_p2r_clears_the_record(self):
        thread = self._thread_with_p2r(lambda command, stream: None)
        thread._p2r_failed = True
        with mock.patch("torch.cuda.Event"):
            thread._execute_p2r(FakeCommand())
        self.assertFalse(thread._p2r_failed)


class TestTheP2RDrainDoesNotReportSuccessAfterAborting(unittest.TestCase):
    """A drain that had to abort every communicator is not a completed sync.

    ``bounded_drain_or_abort`` returns False after a timeout/abort or when a
    CUDA event cannot prove completion.
    The P2R call site used to discard that, so the policy reported the sync as
    successful and returned to a main loop with no communicators: it never
    unregistered, the controller never saw a dead policy, and the job held its
    nodes until the wall clock. Job 2148080 sat silent for 16 minutes that way
    while all seven rollouts had already exited 13s after the injection.
    """

    def _call(self, drained):
        from cosmos_rl.policy.worker import rl_worker

        with mock.patch.object(
            rl_worker, "bounded_drain_or_abort", return_value=drained
        ) as drain:
            rl_worker._drain_or_fail(
                "stream", 120.0, "policy_P2R[src@step8]", "src", "dst", 8
            )
        return drain

    def test_a_clean_drain_returns_quietly(self):
        drain = self._call(True)
        drain.assert_called_once_with("stream", 120.0, "policy_P2R[src@step8]")

    def test_an_aborted_drain_fails_the_replica(self):
        from cosmos_rl.policy.worker import rl_worker

        with self.assertRaises(rl_worker.P2RDrainAborted) as caught:
            self._call(False)
        message = str(caught.exception)
        self.assertIn("dst", message)
        self.assertIn("step 8", message)
        self.assertIn("aborted", message)

    def test_the_p2r_call_site_uses_the_checked_helper(self):
        """The unchecked call must not come back; the failure it causes is
        silent, so nothing else would catch its return."""
        import inspect
        from cosmos_rl.policy.worker import rl_worker

        body = inspect.getsource(
            rl_worker.RLPolicyWorker.execute_policy_to_rollout_unicast
        )
        self.assertIn("_drain_or_fail(", body)
        self.assertNotIn("bounded_drain_or_abort(", body)


if __name__ == "__main__":
    unittest.main()
