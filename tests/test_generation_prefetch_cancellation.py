# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation must not strand the sole preparation worker's future queue."""

import threading
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest

from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin


class ControlledBackend(RolloutGenerationMixin):
    def __init__(self):
        self.config = SimpleNamespace(rollout=SimpleNamespace(prefetch_rollout=True))
        self.started = threading.Event()
        self.release = threading.Event()
        self.prepared = []
        self.retry_value = None
        self.calls = {}
        self.setup_generation()

    def _prepare_sample(self, payload, **kwargs):
        occurrence = id(payload)
        count = self.calls.get(occurrence, 0)
        self.calls[occurrence] = count + 1
        value = self.retry_value if count and self.retry_value else payload.value
        if value == "first":
            self.started.set()
            assert self.release.wait(10)
        if value == "error":
            raise ValueError("bad sample")
        self.prepared.append(value)
        return value


def payload(index, value):
    return SimpleNamespace(prompt_idx=index, value=value)


@pytest.mark.parametrize("replacement", ["second", "error"])
def test_replace_an_actively_preparing_future(monkeypatch, replacement):
    failures = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: failures.append(args.exc_value)
    )
    backend = ControlledBackend()
    try:
        first = payload(1, "first")
        backend.submit_setup([first])
        assert backend.started.wait(5)
        prior = backend._setup_futures[backend._setup_key(first)]
        backend.retry_value = replacement
        backend.submit_setup([first])
        current = backend._setup_futures[backend._setup_key(first)]
        backend.release.set()
        assert prior.result(timeout=5) == "first"
        if replacement == "error":
            with pytest.raises(ValueError, match="bad sample"):
                current.result(timeout=5)
        else:
            assert current.result(timeout=5) == "second"
        followup = payload(2, "follow-up")
        backend.submit_setup([followup])
        assert (
            backend._setup_futures[backend._setup_key(followup)].result(timeout=5)
            == "follow-up"
        )
        assert backend._setup_thread.is_alive()
        assert not failures
    finally:
        backend.release.set()
        backend.shutdown_generation()


def test_cancel_queued_work_without_cancelling_running_preparation():
    backend = ControlledBackend()
    try:
        backend.submit_setup([payload(1, "first")])
        assert backend.started.wait(5)
        queued_payload = payload(2, "replacement")
        backend.submit_setup([queued_payload])
        queued = backend._setup_futures[backend._setup_key(queued_payload)]
        backend.submit_setup([queued_payload])
        replacement = backend._setup_futures[backend._setup_key(queued_payload)]
        assert queued.cancelled()
        backend.release.set()
        assert replacement.result(timeout=5) == "replacement"
        assert backend.prepared.count("replacement") == 1
    finally:
        backend.release.set()
        backend.shutdown_generation()


def test_shutdown_keeps_active_future_owned_until_preparation_finishes(monkeypatch):
    failures = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: failures.append(args.exc_value)
    )
    backend = ControlledBackend()
    try:
        first, pending = payload(1, "first"), payload(2, "queued")
        backend.submit_setup([first])
        assert backend.started.wait(5)
        active = backend._setup_futures[backend._setup_key(first)]
        backend.submit_setup([pending])
        queued = backend._setup_futures[backend._setup_key(pending)]
        with pytest.raises(TimeoutError, match="still using backend resources"):
            backend.shutdown_generation()
        assert queued.cancelled()
        assert not active.cancelled()
        backend.release.set()
        assert active.result(timeout=5) == "first"
        backend._setup_thread.join(5)
        assert not backend._setup_thread.is_alive()
        assert not failures
    finally:
        backend.release.set()
        backend._setup_thread.join(5)
        backend.shutdown_generation()


def test_shutdown_cancels_queued_futures_already_claimed_by_consumer(monkeypatch):
    backend = ControlledBackend()
    first, pending = payload(1, "first"), payload(2, "queued")
    consumed = threading.Event()
    outcomes = []
    consumer = None
    queued = None
    try:
        backend.submit_setup([first, pending])
        assert backend.started.wait(5)
        active = backend._setup_futures[backend._setup_key(first)]
        queued = backend._setup_futures[backend._setup_key(pending)]
        original_result = active.result

        def await_active(*args, **kwargs):
            consumed.set()
            return original_result(*args, **kwargs)

        monkeypatch.setattr(active, "result", await_active)

        def consume():
            try:
                outcomes.append(
                    backend._gather_prepared_samples(
                        [first, pending],
                        data_packer=None,
                        data_fetcher=None,
                        is_validation=False,
                    )
                )
            except BaseException as error:
                outcomes.append(error)

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        assert consumed.wait(5)
        assert not backend._setup_futures
        with pytest.raises(TimeoutError, match="still using backend resources"):
            backend.shutdown_generation()
        assert queued.cancelled()
        backend.release.set()
        consumer.join(5)
        assert not consumer.is_alive()
        assert len(outcomes) == 1 and isinstance(outcomes[0], CancelledError)
    finally:
        if queued is not None:
            queued.cancel()
        backend.release.set()
        if consumer is not None:
            consumer.join(5)
        backend._setup_thread.join(5)
        backend.shutdown_generation()


def test_setup_cannot_restart_while_previous_callback_is_running():
    backend = ControlledBackend()
    old_worker = backend._setup_thread
    try:
        backend.submit_setup([payload(1, "first")])
        assert backend.started.wait(5)
        with pytest.raises(TimeoutError, match="still using backend resources"):
            backend.shutdown_generation()
        assert old_worker.is_alive()
        with pytest.raises(RuntimeError, match="shutdown is incomplete"):
            backend.setup_generation()
        backend.release.set()
        old_worker.join(5)
        backend.shutdown_generation()
        backend.setup_generation()
        followup = payload(2, "follow-up")
        backend.submit_setup([followup])
        assert (
            backend._setup_futures[backend._setup_key(followup)].result(5)
            == "follow-up"
        )
    finally:
        backend.release.set()
        backend.shutdown_generation()
        old_worker.join(5)


@pytest.mark.parametrize("restart", [False, True])
def test_shutdown_fences_in_progress_submission(monkeypatch, restart):
    backend = ControlledBackend()
    entered = threading.Event()
    release = threading.Event()
    original_key = backend._setup_key
    submitter = None

    def paused_key(value):
        entered.set()
        assert release.wait(5)
        return original_key(value)

    monkeypatch.setattr(backend, "_setup_key", paused_key)
    try:
        submitter = threading.Thread(
            target=backend.submit_setup, args=([payload(1, "late")],), daemon=True
        )
        submitter.start()
        assert entered.wait(5)
        backend.shutdown_generation()
        if restart:
            backend.setup_generation()
        release.set()
        submitter.join(5)
        assert not submitter.is_alive()
        assert not backend._setup_futures
        assert not backend._setup_payloads
        assert backend._setup_request_queue.empty()
    finally:
        release.set()
        if submitter is not None:
            submitter.join(5)
        backend.shutdown_generation()
