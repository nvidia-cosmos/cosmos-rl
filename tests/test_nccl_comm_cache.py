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

"""Tests for the lazy 2-rank communicator cache (CPU-only, fake build/abort).

``build_fn`` / ``abort_fn`` are injected so the LRU, quarantine cooldown,
per-pair build coalescing, and init semaphore are all exercised without a
CUDA context.
"""

import threading
import time
import unittest

from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache


class _FakeBuilder:
    """Assigns monotonically increasing comm indices; records calls."""

    def __init__(self, delay=0.0):
        self.calls = []
        self._next = 100
        self._lock = threading.Lock()
        self._delay = delay
        self.concurrent = 0
        self.peak_concurrent = 0

    def __call__(self, uid_chars, local_rank):
        with self._lock:
            self.concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
            idx = self._next
            self._next += 1
            self.calls.append((tuple(uid_chars), local_rank))
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.concurrent -= 1
        return idx


class TestGetOrCreate(unittest.TestCase):
    def test_builds_once_and_caches(self):
        builder = _FakeBuilder()
        cache = CommCache(build_fn=builder, abort_fn=lambda i: None)
        idx1 = cache.get_or_create((0, 1), uid_chars=[1, 2], local_rank=0)
        idx2 = cache.get_or_create((0, 1), uid_chars=[1, 2], local_rank=0)
        self.assertEqual(idx1, idx2)
        self.assertEqual(len(builder.calls), 1)
        self.assertIn((0, 1), cache)
        self.assertEqual(len(cache), 1)

    def test_distinct_pairs_build_separately(self):
        builder = _FakeBuilder()
        cache = CommCache(build_fn=builder, abort_fn=lambda i: None)
        a = cache.get_or_create((0, 1), uid_chars=[1], local_rank=0)
        b = cache.get_or_create((2, 3), uid_chars=[2], local_rank=1)
        self.assertNotEqual(a, b)
        self.assertEqual(len(cache), 2)

    def test_build_from_empty_uid_raises_not_wedges(self):
        # Invariant: never build a comm from an empty UID (would wedge 600s in
        # create_nccl_comm).  Fail fast so the caller renegotiates.
        builder = _FakeBuilder()
        cache = CommCache(build_fn=builder, abort_fn=lambda i: None)
        with self.assertRaises(ValueError):
            cache.get_or_create((0, 1), uid_chars=[], local_rank=0)
        self.assertEqual(len(builder.calls), 0)  # never called the builder
        self.assertNotIn((0, 1), cache)  # nothing cached

    def test_empty_uid_ok_when_pair_already_cached(self):
        # A cached pair returns without building, so an empty UID is irrelevant
        # (the warm-reuse path must not trip the guard).
        builder = _FakeBuilder()
        cache = CommCache(build_fn=builder, abort_fn=lambda i: None)
        idx = cache.get_or_create((0, 1), uid_chars=[1, 2], local_rank=0)
        self.assertEqual(cache.get_or_create((0, 1), uid_chars=[], local_rank=0), idx)
        self.assertEqual(len(builder.calls), 1)

    def test_fresh_uid_rebuilds_stale_comm(self):
        # Comm-generation recovery: a non-empty UID that differs from the one
        # the cached comm was built with means the PEER rebuilt its half (e.g.
        # after quarantine).  We must abort our stale half and rebuild in
        # lockstep, or split-brain (peer on new comm, us on old) -> hang.
        aborted = []
        builder = _FakeBuilder()
        cache = CommCache(build_fn=builder, abort_fn=lambda i: aborted.append(i))
        idx1 = cache.get_or_create((0, 1), uid_chars=[1, 2], local_rank=0)
        # Same UID -> reuse, no rebuild, no abort.
        self.assertEqual(
            cache.get_or_create((0, 1), uid_chars=[1, 2], local_rank=0), idx1
        )
        self.assertEqual(len(builder.calls), 1)
        self.assertEqual(aborted, [])
        # Fresh (different) UID -> abort stale + rebuild.
        idx2 = cache.get_or_create((0, 1), uid_chars=[9, 9], local_rank=0)
        self.assertNotEqual(idx2, idx1)
        self.assertEqual(aborted, [idx1])
        self.assertEqual(len(builder.calls), 2)
        self.assertIn((0, 1), cache)
        # Empty UID afterward (warm reuse) -> keep the current comm, no rebuild.
        self.assertEqual(cache.get_or_create((0, 1), uid_chars=[], local_rank=0), idx2)
        self.assertEqual(len(builder.calls), 2)


class TestLruEviction(unittest.TestCase):
    def test_evicts_lru_when_over_cap(self):
        aborted = []
        builder = _FakeBuilder()
        cache = CommCache(
            max_live=2, build_fn=builder, abort_fn=lambda i: aborted.append(i)
        )
        i01 = cache.get_or_create((0, 1), uid_chars=[1], local_rank=0)
        cache.get_or_create((2, 3), uid_chars=[1], local_rank=0)
        # Touch (0,1) so (2,3) becomes LRU.
        cache.get((0, 1))
        cache.get_or_create(
            (4, 5), uid_chars=[1], local_rank=0
        )  # over cap -> evict LRU

        self.assertEqual(len(cache), 2)
        self.assertNotIn((2, 3), cache)  # (2,3) was LRU
        self.assertIn((0, 1), cache)
        self.assertIn((4, 5), cache)
        # Build order gave (0,1)->100, (2,3)->101, (4,5)->102; the evicted
        # LRU (2,3) held comm idx 101.
        self.assertEqual(aborted, [101])
        self.assertEqual(i01, 100)

    def test_evict_aborts_outside_lock(self):
        # Regression: eviction must abort the LRU comm AFTER releasing
        # self._lock (like every other abort path).  A slow/hung nccl_abort
        # under the lock would otherwise freeze the whole cache (get/stats/
        # other builds) for every thread.  We assert the lock is free at the
        # moment abort_fn runs.
        builder = _FakeBuilder()
        lock_free_during_abort = []

        def abort_fn(idx):
            got = cache._lock.acquire(blocking=False)
            lock_free_during_abort.append(got)
            if got:
                cache._lock.release()

        cache = CommCache(max_live=1, build_fn=builder, abort_fn=abort_fn)
        cache.get_or_create((0, 1), uid_chars=[1], local_rank=0)
        cache.get_or_create((2, 3), uid_chars=[1], local_rank=0)  # evicts (0,1)

        self.assertEqual(lock_free_during_abort, [True])  # abort ran lock-free
        self.assertNotIn((0, 1), cache)
        self.assertIn((2, 3), cache)


class TestPinnedEviction(unittest.TestCase):
    """A send/recv holds a bare ``comm_idx`` for the whole collective, so LRU
    eviction must never abort a pinned pair.  Deliberate aborts still must,
    since they are what unwedge a stuck collective."""

    def test_pinned_comm_is_not_evicted(self):
        # Also the only test that pins ``_evict_if_needed_locked(protect=...)``:
        # with (0,1) held, the freshly-built (2,3) is the sole eligible victim,
        # so dropping either the pin skip or ``protect`` aborts a comm here.
        aborted = []
        builder = _FakeBuilder()
        cache = CommCache(
            max_live=1, build_fn=builder, abort_fn=lambda i: aborted.append(i)
        )
        with cache.leased((0, 1), uid_chars=[1], local_rank=0) as pinned_idx:
            # Building a second pair puts the cache over cap, but the only
            # candidate is in use -> hold the surplus instead of aborting it.
            fresh = cache.get_or_create((2, 3), uid_chars=[1], local_rank=0)
            self.assertEqual(aborted, [])
            self.assertIn((0, 1), cache)
            self.assertIn((2, 3), cache)  # the new comm survived its own insert
            self.assertNotIn(fresh, aborted)
            self.assertEqual(len(cache), 2)  # deliberately over max_live
        # Pin released -> the next build may now evict it.
        self.assertEqual(cache.pinned_count((0, 1)), 0)
        cache.get_or_create((4, 5), uid_chars=[1], local_rank=0)
        self.assertIn(pinned_idx, aborted)

    def test_unpinned_neighbour_is_evicted_instead(self):
        # With a pinned LRU entry, eviction must skip it and take the next
        # unpinned candidate rather than giving up.
        aborted = []
        builder = _FakeBuilder()
        cache = CommCache(
            max_live=2, build_fn=builder, abort_fn=lambda i: aborted.append(i)
        )
        with cache.leased((0, 1), uid_chars=[1], local_rank=0):  # LRU, pinned
            idx23 = cache.get_or_create((2, 3), uid_chars=[1], local_rank=0)
            cache.get_or_create((4, 5), uid_chars=[1], local_rank=0)  # over cap
            self.assertEqual(aborted, [idx23])  # skipped (0,1), took (2,3)
            self.assertIn((0, 1), cache)

    def test_deliberate_aborts_ignore_pins(self):
        # Quarantine/teardown must fire on a PINNED pair: the wedged operation
        # is what holds the pin, so deferring the abort until the refcount drops
        # would deadlock.  Every deliberate abort path must behave this way.
        for name, fire in (
            ("abort", lambda c: c.abort((0, 1))),
            ("abort_endpoint", lambda c: c.abort_endpoint((0,))),
            ("abort_all", lambda c: c.abort_all()),
        ):
            with self.subTest(path=name):
                aborted = []
                cache = CommCache(
                    build_fn=lambda u, r: 7, abort_fn=lambda i: aborted.append(i)
                )
                with cache.leased((0, 1), uid_chars=[1], local_rank=0):
                    fire(cache)
                    self.assertEqual(aborted, [7])
                    self.assertNotIn((0, 1), cache)

    def test_pins_are_refcounted_and_released(self):
        cache = CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None)
        with cache.leased((0, 1), uid_chars=[1], local_rank=0):
            self.assertEqual(cache.pinned_count((0, 1)), 1)
            with cache.leased((0, 1), uid_chars=[1], local_rank=0):
                self.assertEqual(cache.pinned_count((0, 1)), 2)
            self.assertEqual(cache.pinned_count((0, 1)), 1)
        self.assertEqual(cache.pinned_count((0, 1)), 0)

    def test_pin_released_when_body_raises(self):
        cache = CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None)
        with self.assertRaises(RuntimeError):
            with cache.leased((0, 1), uid_chars=[1], local_rank=0):
                raise RuntimeError("send blew up")
        self.assertEqual(cache.pinned_count((0, 1)), 0)


class TestQuarantine(unittest.TestCase):
    def test_quarantine_with_cooldown(self):
        cache = CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None)
        cache.quarantine(("replica-a", 3), cooldown=100.0)
        self.assertTrue(cache.is_quarantined(("replica-a", 3)))
        self.assertFalse(cache.is_quarantined(("replica-b", 3)))

    def test_zero_cooldown_not_quarantined(self):
        cache = CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None)
        cache.quarantine(("replica-a", 3), cooldown=0.0)
        # Cooldown already elapsed -> not quarantined, and the entry is GC'd.
        self.assertFalse(cache.is_quarantined(("replica-a", 3)))
        self.assertEqual(cache.stats()["quarantined"], 0)

    def test_quarantine_pairkey_aborts_live_comm(self):
        aborted = []
        cache = CommCache(
            build_fn=lambda u, r: 42, abort_fn=lambda i: aborted.append(i)
        )
        cache.get_or_create((0, 1), uid_chars=[1], local_rank=0)
        self.assertIn((0, 1), cache)
        cache.quarantine((0, 1), cooldown=50.0)
        self.assertNotIn((0, 1), cache)  # comm aborted + dropped
        self.assertEqual(aborted, [42])
        self.assertTrue(cache.is_quarantined((0, 1)))

    def test_clear_quarantine(self):
        cache = CommCache(build_fn=lambda u, r: 1, abort_fn=lambda i: None)
        cache.quarantine(("x", 0), cooldown=100.0)
        cache.clear_quarantine(("x", 0))
        self.assertFalse(cache.is_quarantined(("x", 0)))


class TestEndpointAbort(unittest.TestCase):
    """quarantine/abort_endpoint must tear down the real 3-tuple comm keys.

    Regression: comm keys are 3-tuples ((sender_replica, sender_rank,
    receiver_rank) on the receiver; (sender_rank, receiver_replica,
    receiver_rank) on the sender), but quarantine used to only abort exact
    2-tuples, so a quarantined endpoint's dead comm stayed cached and was
    reused after the transient failure.  Now an endpoint is a *prefix* of the
    comm key and prefix-matching aborts every comm it owns.
    """

    def _cache(self, aborted):
        idx = {"n": 0}

        def build(u, r):
            idx["n"] += 1
            return idx["n"]

        return CommCache(build_fn=build, abort_fn=lambda i: aborted.append(i))

    def test_receiver_endpoint_prefix_aborts_only_its_comm(self):
        aborted = []
        cache = self._cache(aborted)
        # Receiver keys: (sender_replica, sender_rank, receiver_rank).
        a = cache.get_or_create(("rA", 0, 5), uid_chars=[1], local_rank=1)
        cache.get_or_create(("rB", 0, 5), uid_chars=[1], local_rank=1)  # other sender
        cache.get_or_create(("rA", 1, 5), uid_chars=[1], local_rank=1)  # other rank
        # Quarantine the sender endpoint (sender_replica, sender_rank).
        cache.quarantine(("rA", 0), cooldown=50.0)
        self.assertNotIn(("rA", 0, 5), cache)  # aborted
        self.assertIn(("rB", 0, 5), cache)  # different sender untouched
        self.assertIn(("rA", 1, 5), cache)  # different rank untouched
        self.assertEqual(aborted, [a])
        self.assertTrue(cache.is_quarantined(("rA", 0)))

    def test_sender_full_pair_quarantine_aborts(self):
        aborted = []
        cache = self._cache(aborted)
        # Sender keys: (sender_rank, receiver_replica, receiver_rank).
        p = cache.get_or_create((0, "pA", 1), uid_chars=[1], local_rank=0)
        cache.quarantine((0, "pA", 1), cooldown=50.0)
        self.assertNotIn((0, "pA", 1), cache)
        self.assertEqual(aborted, [p])

    def test_abort_endpoint_counts_and_noops_on_no_match(self):
        aborted = []
        cache = self._cache(aborted)
        cache.get_or_create(("rA", 0, 5), uid_chars=[1], local_rank=1)
        cache.get_or_create(("rA", 0, 6), uid_chars=[1], local_rank=1)
        # Endpoint (rA, 0) owns both receiver ranks 5 and 6.
        self.assertEqual(cache.abort_endpoint(("rA", 0)), 2)
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.abort_endpoint(("rZ", 9)), 0)  # no match


class TestAbort(unittest.TestCase):
    def test_abort_idempotent(self):
        aborted = []
        cache = CommCache(build_fn=lambda u, r: 7, abort_fn=lambda i: aborted.append(i))
        cache.get_or_create((0, 1), uid_chars=[1], local_rank=0)
        self.assertTrue(cache.abort((0, 1)))
        self.assertFalse(cache.abort((0, 1)))  # already gone
        self.assertEqual(aborted, [7])

    def test_abort_all(self):
        aborted = []
        builder = _FakeBuilder()
        cache = CommCache(build_fn=builder, abort_fn=lambda i: aborted.append(i))
        cache.get_or_create((0, 1), uid_chars=[1], local_rank=0)
        cache.get_or_create((2, 3), uid_chars=[1], local_rank=0)
        n = cache.abort_all()
        self.assertEqual(n, 2)
        self.assertEqual(len(cache), 0)
        self.assertEqual(len(aborted), 2)


class TestConcurrency(unittest.TestCase):
    def test_same_pair_builds_once_under_race(self):
        builder = _FakeBuilder(delay=0.05)
        cache = CommCache(build_fn=builder, abort_fn=lambda i: None)
        results = []
        results_lock = threading.Lock()

        def worker():
            idx = cache.get_or_create((0, 1), uid_chars=[9], local_rank=0)
            with results_lock:
                results.append(idx)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one build despite 8 racing threads; all see the same comm.
        self.assertEqual(len(builder.calls), 1)
        self.assertEqual(len(set(results)), 1)

    def test_init_semaphore_bounds_concurrency(self):
        builder = _FakeBuilder(delay=0.05)
        cache = CommCache(
            max_concurrent_init=2, build_fn=builder, abort_fn=lambda i: None
        )

        def worker(pair):
            cache.get_or_create(pair, uid_chars=[pair[0]], local_rank=0)

        threads = [
            threading.Thread(target=worker, args=((i, i + 100),)) for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(builder.calls), 6)
        # Never more than max_concurrent_init builds in flight at once.
        self.assertLessEqual(builder.peak_concurrent, 2)


if __name__ == "__main__":
    unittest.main()
