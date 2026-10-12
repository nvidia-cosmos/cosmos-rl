# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual training entrypoint and prefetch ownership; CPU fake wire traffic."""

import threading
import time
import weakref
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    FixedRolloutBatching,
    prefetch_training_batch,
    run_training_step,
)
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBudget,
    ReceivedBatch,
    ReceiveMemoryError,
)


def eventually(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.001)


@pytest.fixture
def consumer():
    budget = ReceiveBudget(64, 0.05)
    allocated = []

    class Packer(PrefetchDataPackerMixin):
        def _filter_prefetch_tasks(self, rollouts):
            return [(0, rollouts[0])]

        def _fetch_batch(self, tasks):
            budget.reserve(64)
            tensor = torch.full((16,), float(tasks[0][1]))
            allocated.append(weakref.ref(tensor))
            budget.attribute(decoded=64)
            return ReceivedBatch({"ref": {"x": tensor}}, budget, 64)

    packer = Packer()
    packer._transport_strategy = SimpleNamespace(
        _receive_budget=budget,
        before_join=budget.close,
        on_prefetch_complete=lambda *args: None,
    )
    packer._setup_prefetch(prefetch_timeout=0.1)
    trainer = SimpleNamespace(
        data_packer=packer,
        batching_contract=ExpandedSampleBatching(partial_tail="include"),
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=16, mu_iterations=1)
            )
        ),
    )

    def prepare(rollouts):
        cache = getattr(packer._preparation_local, "cache", packer._prefetch_cache)
        return ExpandedTrainingBatch(((cache["ref"]["x"][:8],),))

    trainer.prepare_training_batch = prepare
    yield trainer, budget, allocated
    packer.shutdown_prefetch(join_timeout=2)
    packer.release_prefetch()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("fixed", [None, 1])
def test_two_updates_drop_all_prepared_aliases_before_capacity_returns(
    consumer, prepared, fixed
):
    trainer, budget, allocated = consumer
    trainer.batching_contract = ExpandedSampleBatching(
        partial_tail="include", fixed_minibatches=fixed
    )
    original_release = budget.release

    def release(size):
        assert all(ref() is None for ref in allocated)
        original_release(size)

    budget.release = release
    seen = []

    def train(batch, **kwargs):
        assert budget.used == budget.consumer_bytes == 64
        seen.append(batch.minibatches[0][0].tolist())
        return {"loss": batch.minibatches[0][0].sum().item()}

    trainer.step_expanded_training = train
    for value in (1, 2):
        rollouts = [value]
        if prepared:
            assert prefetch_training_batch(trainer, rollouts)
        result = run_training_step(trainer, rollouts=rollouts)
        assert result["loss"] == value * 8
        assert budget.used == budget.decoded == budget.consumer_bytes == 0
    assert seen == [[1.0] * 8, [2.0] * 8]


@pytest.mark.parametrize("prepared", [False, True])
def test_slow_consumer_can_outlive_admission_and_transport_deadlines(
    consumer, prepared
):
    trainer, budget, allocated = consumer
    packer = trainer.data_packer
    upcoming = [2]

    def train(batch, **kwargs):
        if batch.minibatches[0][0][0] == 1:
            if prepared:
                prefetch_training_batch(trainer, upcoming)
            else:
                packer.start_prefetch(upcoming)
            eventually(lambda: budget.waiting_for_consumer)
            time.sleep(0.35)  # > 3 watchdog periods and 7 admission periods.
            assert len(allocated) == 1  # No next receive/allocation yet.
            assert packer._prefetch_failure is None
        return {}

    trainer.step_expanded_training = train
    run_training_step(trainer, rollouts=[1])
    run_training_step(trainer, rollouts=upcoming)
    assert len(allocated) == 2
    assert budget.used == 0
    assert packer._prefetch_failure is None


def test_native_hang_after_admission_still_trips_independent_watchdog(
    consumer, monkeypatch
):
    trainer, budget, _ = consumer
    packer = trainer.data_packer
    entered, finish, failed = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(
        "cosmos_rl.utils.payload_transport.prefetch_mixin.fail_transport",
        lambda message: failed.set(),
    )
    original = packer._fetch_batch
    packer.start_prefetch([1])
    packer.wait_prefetch()

    def fetch(tasks):
        result = original(tasks)
        entered.set()
        assert finish.wait(2)
        return result

    packer._fetch_batch = fetch
    packer.start_prefetch([2])
    eventually(lambda: budget.waiting_for_consumer)
    time.sleep(0.25)
    assert not failed.is_set()
    packer.release_prefetch()
    try:
        assert entered.wait(1)
        assert failed.wait(1)  # No wait_prefetch call is needed to detect the hang.
    finally:
        finish.set()
    packer.shutdown_prefetch(join_timeout=2)
    # A mocked fatal callback returning must not permit cleanup/reuse. The real
    # callback exits the process, which owns this uncertain result until exit.
    assert budget.used == 64
    assert any(
        isinstance(owner, ReceivedBatch) and not owner.released
        for owner in packer._prefetch_terminal_owners
    )


@pytest.mark.parametrize("failure", ["step", "alias", "autograd"])
def test_failed_step_or_invalid_report_keeps_lease(consumer, failure):
    trainer, budget, _ = consumer

    def train(batch, **kwargs):
        if failure == "step":
            raise ValueError("training failed")
        value = batch.minibatches[0][0]
        if failure == "autograd":
            value = (value * torch.ones(8, requires_grad=True)).sum()
        return {"bad": value}

    trainer.step_expanded_training = train
    with pytest.raises((ValueError, ReceiveMemoryError)):
        run_training_step(trainer, rollouts=[1])
    assert budget.used == budget.consumer_bytes == 64


def test_legacy_contract_rejected_before_contacting_a_sender(consumer):
    trainer, budget, allocated = consumer
    trainer.batching_contract = FixedRolloutBatching()
    trainer.step_training = Mock(side_effect=AssertionError("must not train"))
    with pytest.raises(ValueError, match="explicit final-reader integration"):
        run_training_step(trainer, rollouts=[1])
    assert not allocated and budget.used == 0


def test_training_and_additional_streams_reach_final_release(consumer, monkeypatch):
    trainer, budget, _ = consumer
    trainer.train_stream = "training"
    trainer.training_payload_streams = lambda: ("reward", "visualization")
    trainer.step_expanded_training = lambda batch, **kwargs: {}
    recorded, waited = [], []

    class Event:
        def record(self, stream):
            recorded.append(stream)

        def synchronize(self):
            assert len(recorded) == 3 and budget.used == 64
            waited.append(True)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    run_training_step(trainer, rollouts=[1])
    assert recorded == ["reward", "visualization", "training"]
    assert len(waited) == 3 and budget.used == 0


def test_prepared_consumer_waits_until_worker_drops_aliases(consumer, monkeypatch):
    trainer, budget, _ = consumer
    packer = trainer.data_packer
    packer._prefetch_timeout_s = 2
    published, relinquish, consumed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    class SlowHandoff(Future):
        def set_result(self, result):
            super().set_result(result)
            published.set()
            assert relinquish.wait(2)

    monkeypatch.setattr(
        "cosmos_rl.utils.payload_transport.prefetch_mixin.Future", SlowHandoff
    )
    trainer.step_expanded_training = lambda batch, **kwargs: {}
    prefetch_training_batch(trainer, [1])
    assert published.wait(1)
    errors = []

    def consume():
        try:
            run_training_step(trainer, rollouts=[1])
            consumed.set()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert not consumed.wait(0.05)
        assert budget.used == 64
    finally:
        relinquish.set()
        thread.join(2)
    assert not thread.is_alive() and not errors
    assert consumed.is_set() and budget.used == 0


def test_shutdown_cancels_safe_consumer_backpressure(consumer):
    trainer, budget, _ = consumer
    packer = trainer.data_packer
    packer.start_prefetch([1])
    packer.wait_prefetch()
    packer.start_prefetch([2])
    eventually(lambda: budget.waiting_for_consumer)
    time.sleep(0.15)
    packer.shutdown_prefetch(join_timeout=1)
    assert packer._prefetch_thread is None
    assert budget.used == budget.consumer_bytes == 64
    packer.release_prefetch()
    assert budget.used == budget.consumer_bytes == 0
