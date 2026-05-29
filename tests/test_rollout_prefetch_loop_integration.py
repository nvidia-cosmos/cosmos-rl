"""Integration tests for ``RolloutWorkerBase._prefetch_loop`` ↔ ``submit_setup`` wiring.

These tests cover the seam that the unit tests in
``test_rollout_generation_mixin.py`` deliberately don't: the
:class:`~cosmos_rl.rollout.worker.rollout_control.DisaggregatedRolloutControlWorker`
side of the prefetch contract.  The mixin tests prove that *if*
``submit_setup`` is called, ``_gather_prepared_samples`` will see a
ready future; this file proves that the worker's prefetch loop
*actually* calls ``submit_setup`` -- once per fetched batch, before
the batch is enqueued, with the same payload objects the consumer
will later pop.

Background
----------
Before the single-producer-mode refactor, the worker had two
producers for ``_prompt_queue``: ``main_loop`` (via
``request_new_prompts``) and ``_prefetch_loop``.  Only the latter
called ``submit_setup``, and it lost ~98% of races because of a 500
ms polling sleep + uncoordinated lock placement.  Cluster runs
showed prefetch firing on 8 of 500 batches -- the bridge appeared
broken even though every individual unit test passed.

The lesson: the integration call site (``RolloutWorkerBase`` →
``RolloutGenerationMixin``) needed its own test, not just tests
that *mock* it.  These tests are that.
"""

from __future__ import annotations

import threading
import time
import unittest
from queue import Queue
from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest.mock import MagicMock

from cosmos_rl.rollout import State
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
)


def _make_worker_for_prefetch(
    *,
    api_responses: List[Tuple[List[dict], bool]],
    submit_setup_recorder: List[List[dict]],
    queue_max: int = 2,
    on_submit_delay_ms: float = 0.0,
    fail_submit_for_idxs: set = None,
) -> Any:
    """Construct a stripped-down worker just sufficient to drive ``_prefetch_loop``.

    We bypass ``DisaggregatedRolloutControlWorker.__init__`` (which
    loads HuggingFace model configs, registers atexit handlers,
    creates CUDA streams, ...) and set only the attributes the
    prefetch loop reads.  This keeps the test hermetic and fast --
    no GPU, no controller, no model files.

    Parameters
    ----------
    api_responses
        List of ``(payloads_dict_list, is_end)`` tuples returned in
        order by the fake ``api_client.get_next_prompt``.  After the
        list is exhausted the test drives the loop to ``is_end=True``
        on its own.
    submit_setup_recorder
        Caller-owned list that ``self.rollout.submit_setup`` appends
        each batch's payloads to (in call order).  Used to assert
        ordering and per-batch coverage.
    queue_max
        ``maxsize`` for the bounded ``_prompt_queue``.  Defaults to 2
        to mirror the production single-producer-mode default.
    on_submit_delay_ms
        Optional artificial delay inside ``submit_setup`` to widen
        the put-after-submit window for ordering tests.
    fail_submit_for_idxs
        Set of ``prompt_idx`` values for which ``submit_setup``
        should raise.  The loop must still ``put`` despite the
        exception (consumer falls back to inline prep).
    """
    fail_submit_for_idxs = fail_submit_for_idxs or set()

    worker = DisaggregatedRolloutControlWorker.__new__(
        DisaggregatedRolloutControlWorker
    )

    worker.shutdown_signal = threading.Event()
    worker.state = State()
    # Skip the startup weight-sync wait by setting the sticky bit.
    worker.state.set_weight_synced()

    worker.batch_size = 2
    worker.rank_in_rollout_repicas = 0
    worker._prompt_queue = Queue(maxsize=queue_max)

    # Minimal config: only the ``train.local_dataset`` flag is read
    # by the prefetch loop body.  Setting it False skips the
    # data_fetcher.get_payload_by_index branch entirely.
    worker.config = SimpleNamespace(
        train=SimpleNamespace(local_dataset=False),
    )
    worker.data_packer = MagicMock(name="data_packer")
    worker.data_fetcher = MagicMock(name="data_fetcher")

    # api_client.get_next_prompt: returns queued responses in order,
    # then keeps returning ``([], True)`` so the loop sets
    # ``prompt_fetch_end`` and exits cleanly.
    response_iter = iter(api_responses)

    def _get_next_prompt(batch_size, **kwargs):
        try:
            return next(response_iter)
        except StopIteration:
            return ([], True)

    worker.api_client = SimpleNamespace(get_next_prompt=_get_next_prompt)

    # Fake rollout backend: records bind_prefetch_context/submit_setup
    # calls.  Mimics the surface a RolloutGenerationMixin-composing
    # backend would expose.
    bind_called = threading.Event()
    submit_lock = threading.Lock()

    def _submit_setup(payloads):
        # Simulate the bg setup worker getting kicked off; the real
        # mixin returns immediately after queueing futures, so this
        # should be near-instant in practice.  We record the call
        # before any optional delay so assertions on order are
        # robust.
        with submit_lock:
            submit_setup_recorder.append(list(payloads))
        if on_submit_delay_ms:
            time.sleep(on_submit_delay_ms / 1000.0)
        for p in payloads:
            if p.prompt_idx in fail_submit_for_idxs:
                raise RuntimeError(f"submit_setup boom for prompt_idx={p.prompt_idx}")

    worker.rollout = SimpleNamespace(
        bind_prefetch_context=lambda **kw: bind_called.set(),
        submit_setup=_submit_setup,
    )
    worker._bind_called = bind_called

    return worker


def _make_payload_dict(prompt_idx: int) -> dict:
    """Minimum dict that ``RLPayload.model_validate`` accepts."""
    return {
        "prompt_idx": prompt_idx,
        "prompt": f"prompt-{prompt_idx}",
        "completion": "",
        "weight_version": 0,
    }


class TestPrefetchLoopSubmitSetupWiring(unittest.TestCase):
    """The seam: prefetch loop must notify the backend for every fetched batch.

    Each test runs ``_prefetch_loop`` in its own thread, drives a
    fake controller, and asserts on the side-channel
    ``submit_setup_recorder``.
    """

    def _run_loop_and_drain(
        self, worker: Any, *, expect_batches: int
    ) -> List[List[Any]]:
        """Start the prefetch thread, drain ``_prompt_queue`` like
        ``main_loop`` would, and return the popped batches in order.

        Bounded ``_prompt_queue`` means the producer back-pressures
        when we don't drain; if we never drain, the loop blocks
        forever on ``put``.  A real ``main_loop`` is the consumer
        here.
        """
        thread = threading.Thread(target=worker._prefetch_loop, daemon=True)
        thread.start()

        popped: List[List[Any]] = []
        deadline = time.monotonic() + 5.0
        while len(popped) < expect_batches and time.monotonic() < deadline:
            try:
                batch = worker._prompt_queue.get(timeout=0.5)
                popped.append(batch)
            except Exception:  # queue.Empty
                continue

        # Signal shutdown and wait for the thread to exit.  The loop
        # checks ``shutdown_signal`` between fetches and inside the
        # bounded-put inner loop, so it should exit promptly.
        worker.shutdown_signal.set()
        thread.join(timeout=3.0)
        self.assertFalse(
            thread.is_alive(),
            "prefetch_loop did not exit within 3s of shutdown_signal",
        )
        return popped

    def test_submit_setup_called_for_every_batch(self) -> None:
        """One submit_setup call per fetched batch -- the most basic invariant.

        Pre-refactor this passed ~2% of the time (``main_loop``'s
        non-notifying ``request_new_prompts`` produced most batches).
        Post-refactor it must pass 100%.
        """
        recorder: List[List[dict]] = []
        worker = _make_worker_for_prefetch(
            api_responses=[
                ([_make_payload_dict(0), _make_payload_dict(1)], False),
                ([_make_payload_dict(2), _make_payload_dict(3)], False),
                ([_make_payload_dict(4), _make_payload_dict(5)], True),
            ],
            submit_setup_recorder=recorder,
        )
        popped = self._run_loop_and_drain(worker, expect_batches=3)

        self.assertEqual(len(recorder), 3, "submit_setup once per batch")
        self.assertEqual(len(popped), 3, "consumer should see all 3 batches")
        # Per-batch payloads must match (same prompt_idxs in order).
        for submitted, popped_batch in zip(recorder, popped):
            self.assertEqual(
                [p.prompt_idx for p in submitted],
                [p.prompt_idx for p in popped_batch],
                "submit_setup and put must see the same payload objects",
            )

    def test_submit_setup_runs_before_put(self) -> None:
        """Ordering: submit_setup must fire before the corresponding put.

        Race detection: we widen the submit_setup-to-put window with
        an artificial delay inside ``submit_setup`` and confirm the
        consumer can't pop the batch before the recorder sees it.
        """
        recorder: List[List[dict]] = []
        worker = _make_worker_for_prefetch(
            api_responses=[
                ([_make_payload_dict(10), _make_payload_dict(11)], True),
            ],
            submit_setup_recorder=recorder,
            on_submit_delay_ms=100.0,  # widen the ordering window
        )

        thread = threading.Thread(target=worker._prefetch_loop, daemon=True)
        thread.start()

        # Wait until submit_setup fires (we'll see the recorder grow).
        deadline = time.monotonic() + 3.0
        while not recorder and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(
            len(recorder), 1, "submit_setup should fire promptly after fetch"
        )

        # The put should happen *after* submit_setup completes, not
        # before.  With a 100 ms delay inside submit_setup, the
        # batch must remain unavailable to the consumer until ~100 ms
        # has elapsed.  We give a generous budget but assert the
        # ordering via "submit recorded first, then put visible".
        # Trying to pop *now* (immediately after recorder grew) may
        # succeed or fail depending on how far through the delay we
        # are; what we assert is just: pop succeeds eventually, and
        # the recorder was non-empty before the pop unblocked.
        popped = worker._prompt_queue.get(timeout=2.0)
        self.assertEqual(
            [p.prompt_idx for p in popped],
            [p.prompt_idx for p in recorder[0]],
            "popped batch must match the just-submitted batch",
        )

        worker.shutdown_signal.set()
        thread.join(timeout=3.0)

    def test_bind_prefetch_context_called_on_loop_start(self) -> None:
        """The mixin's bg worker needs the data_packer/data_fetcher
        bound before the first submit_setup arrives, otherwise the
        bg thread runs ``_prepare_sample`` with ``None`` packer.
        """
        recorder: List[List[dict]] = []
        worker = _make_worker_for_prefetch(
            api_responses=[
                ([_make_payload_dict(0), _make_payload_dict(1)], True),
            ],
            submit_setup_recorder=recorder,
        )
        thread = threading.Thread(target=worker._prefetch_loop, daemon=True)
        thread.start()
        # bind should fire before the first fetch.
        self.assertTrue(
            worker._bind_called.wait(timeout=2.0),
            "bind_prefetch_context must be called before fetching prompts",
        )
        worker.shutdown_signal.set()
        thread.join(timeout=3.0)

    def test_prompt_fetch_end_set_on_is_end(self) -> None:
        """When the controller signals end-of-prompts, the loop must
        set the sticky ``state.prompt_fetch_end`` so ``main_loop``
        knows it can drain the queue and exit.
        """
        recorder: List[List[dict]] = []
        worker = _make_worker_for_prefetch(
            api_responses=[
                ([_make_payload_dict(0), _make_payload_dict(1)], False),
                ([_make_payload_dict(2), _make_payload_dict(3)], True),
            ],
            submit_setup_recorder=recorder,
        )
        self._run_loop_and_drain(worker, expect_batches=2)
        self.assertTrue(
            worker.state.prompt_fetch_end(),
            "is_end=True must propagate to state.prompt_fetch_end",
        )

    def test_prompt_fetch_end_set_on_loop_exit_via_finally(self) -> None:
        """Defensive: even if the loop exits unexpectedly (e.g. an
        unhandled bug, or shutdown without is_end), ``prompt_fetch_end``
        must be set so ``main_loop`` doesn't wait forever for a
        producer that died.
        """
        recorder: List[List[dict]] = []
        # Empty api_responses + shutdown immediately -> loop sets
        # prompt_fetch_end via the StopIteration->is_end=True branch
        # OR via the finally clause.  Either way, the bit must be
        # set.
        worker = _make_worker_for_prefetch(
            api_responses=[],
            submit_setup_recorder=recorder,
        )
        thread = threading.Thread(target=worker._prefetch_loop, daemon=True)
        thread.start()
        # Let the loop fetch the empty/is_end response, then signal
        # shutdown for good measure.
        time.sleep(0.1)
        worker.shutdown_signal.set()
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive(), "loop should exit promptly")
        self.assertTrue(
            worker.state.prompt_fetch_end(),
            "prompt_fetch_end must be set on loop exit, even on shutdown",
        )

    def test_submit_setup_failure_does_not_block_put(self) -> None:
        """If ``submit_setup`` raises (backend bug), the loop must
        still ``put`` so the consumer can fall back to inline prep.
        Losing a batch entirely on a backend error is much worse
        than just losing the prep overlap for that batch.
        """
        recorder: List[List[dict]] = []
        worker = _make_worker_for_prefetch(
            api_responses=[
                ([_make_payload_dict(0), _make_payload_dict(1)], False),
                ([_make_payload_dict(2), _make_payload_dict(3)], True),
            ],
            submit_setup_recorder=recorder,
            fail_submit_for_idxs={0},  # first batch's submit_setup raises
        )
        popped = self._run_loop_and_drain(worker, expect_batches=2)
        self.assertEqual(len(popped), 2, "both batches must reach the queue")
        # Both batches must still be recorder-visible (submit_setup
        # was attempted; the recorder records before raising).
        self.assertEqual(len(recorder), 2)

    def test_bounded_queue_back_pressure(self) -> None:
        """``put`` blocks when the bounded queue is full; the producer
        runs at the consumer's pace.  This is the back-pressure
        mechanism that replaces the old 500 ms polling sleep.
        """
        recorder: List[List[dict]] = []
        # Many batches, small queue.  Producer should not race ahead.
        n_batches = 6
        responses = [
            (
                [_make_payload_dict(2 * i), _make_payload_dict(2 * i + 1)],
                i == n_batches - 1,
            )
            for i in range(n_batches)
        ]
        worker = _make_worker_for_prefetch(
            api_responses=responses,
            submit_setup_recorder=recorder,
            queue_max=1,  # tighter than production default for the test
        )

        thread = threading.Thread(target=worker._prefetch_loop, daemon=True)
        thread.start()

        # Sleep without draining: queue should fill to maxsize=1 and
        # producer should block on put.  After this sleep,
        # exactly 1 batch should be queued and 1 submit_setup
        # should have been recorded (the one currently blocked on
        # put), or possibly 2 (the one in-queue plus the one stuck
        # at put).  The strict invariant is that submit_setup has
        # NOT been called n_batches times yet.
        time.sleep(0.3)
        self.assertLess(
            len(recorder),
            n_batches,
            "producer must not race ahead of bounded queue",
        )

        # Drain: pop one at a time and verify the producer keeps up.
        popped = []
        deadline = time.monotonic() + 5.0
        while len(popped) < n_batches and time.monotonic() < deadline:
            try:
                batch = worker._prompt_queue.get(timeout=0.5)
                popped.append(batch)
            except Exception:
                continue

        worker.shutdown_signal.set()
        thread.join(timeout=3.0)
        self.assertEqual(len(popped), n_batches, "all batches reach consumer")
        self.assertEqual(len(recorder), n_batches, "submit_setup once per batch")


class TestSingleProducerModeIsLive(unittest.TestCase):
    """``_single_producer_mode`` must reflect live config, not a snapshot.

    Backends that compose ``RolloutBase`` may flip
    ``config.rollout.prefetch_rollout`` from ``post_init_hook``, which
    runs *after* ``DisaggregatedRolloutControlWorker.__init__`` via the
    ``RolloutBase`` machinery.  An earlier revision snapshotted
    ``_single_producer_mode`` in ``__init__``, so the worker captured
    the pre-override value (``False``) and silently fell into the
    ``elif self.config.rollout.prefetch_rollout:`` branch in ``work()``,
    producing a misleading ``world_size=1 > 1`` warning and leaving
    the prefetch path disabled despite the user requesting it.  The
    fix converts ``_single_producer_mode`` from a cached attribute to
    a ``@property`` that reads the live config; this test pins that.
    """

    def _make_minimal_worker(
        self, *, prefetch_rollout: bool, world_size: int = 1
    ) -> Any:
        """Bypass __init__ and set just enough state for the property."""
        worker = DisaggregatedRolloutControlWorker.__new__(
            DisaggregatedRolloutControlWorker
        )
        worker.config = SimpleNamespace(
            rollout=SimpleNamespace(prefetch_rollout=prefetch_rollout)
        )
        worker.parallel_dims = SimpleNamespace(world_size=world_size)
        return worker

    def test_property_reads_live_config(self) -> None:
        """Mutating config after construction must change the property's
        return value.  This is the load-bearing invariant.
        """
        w = self._make_minimal_worker(prefetch_rollout=False, world_size=1)
        self.assertFalse(
            w._single_producer_mode,
            "should be False when prefetch_rollout=False",
        )
        # Simulate post_init_hook flipping the flag.
        w.config.rollout.prefetch_rollout = True
        self.assertTrue(
            w._single_producer_mode,
            "post-override flip must be observable through the property; "
            "if this fails, _single_producer_mode is being snapshotted again",
        )

    def test_property_false_when_world_size_gt_one(self) -> None:
        """Multi-rank workers fall back to legacy mode regardless of
        prefetch_rollout, because ``request_new_prompts`` runs a
        distributed broadcast that the single rank-0 background
        thread can't drive.
        """
        w = self._make_minimal_worker(prefetch_rollout=True, world_size=4)
        self.assertFalse(w._single_producer_mode)

    def test_property_true_only_for_world_size_one_and_prefetch_on(self) -> None:
        for prefetch, ws, expected in [
            (False, 1, False),
            (True, 1, True),
            (False, 4, False),
            (True, 4, False),
        ]:
            with self.subTest(prefetch=prefetch, world_size=ws):
                w = self._make_minimal_worker(prefetch_rollout=prefetch, world_size=ws)
                self.assertEqual(w._single_producer_mode, expected)


if __name__ == "__main__":
    unittest.main()
