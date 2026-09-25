# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading
import time

import pytest
import torch

from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    RecoverablePreparationError,
    prefetch_training_batch,
    run_training_step,
)
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin


class BasePacker:
    def get_policy_input(
        self, sample=None, rollout_output=None, n_ignore_prefix_tokens=0, **kwargs
    ):
        return rollout_output


class Packer(PrefetchDataPackerMixin, BasePacker):
    def _should_intercept(self, value):
        return isinstance(value, str)

    def _cache_key(self, value):
        return value

    def _fetch_batch(self, tasks):
        return {ref: [1.0, float("nan"), 2.0] for _, ref in tasks}

    def _sync_fetch(self, ref):
        return self._fetch_batch([(0, ref)])[ref]


@pytest.mark.parametrize("fixed", [None, 2])
@pytest.mark.parametrize(
    "case", ["healthy", "empty", "nonfinite", "recoverable", "bug"]
)
def test_prefetch_on_off_and_absent_have_identical_training_semantics(fixed, case):
    observations = []
    for mode in ("on", "off", "absent"):
        packer = Packer()
        if mode == "on":
            packer._setup_prefetch(prefetch_timeout=2)
        main = threading.get_ident()
        prepare_threads, trained, scheduler_calls = [], [], []
        candidate = SimpleNamespace(
            batching_contract=ExpandedSampleBatching(
                partial_tail="include", fixed_minibatches=fixed
            ),
            config=SimpleNamespace(
                train=SimpleNamespace(
                    train_policy=SimpleNamespace(mini_batch=3, mu_iterations=2)
                )
            ),
        )
        if mode != "absent":
            candidate.data_packer = packer

        def prepare(rollouts):
            prepare_threads.append(threading.get_ident())
            if case == "recoverable":
                raise RecoverablePreparationError("missing")
            if case == "bug":
                raise ValueError("bug")
            if case == "empty":
                return ExpandedTrainingBatch(())
            return ExpandedTrainingBatch(
                ((1.0, float("nan") if case == "nonfinite" else 2.0),)
            )

        def train(batch, **kwargs):
            assert threading.get_ident() == main
            trained.append((batch, kwargs))
            return {"updates": len(batch.minibatches) * batch.mu_iterations}

        candidate.prepare_training_batch = prepare
        candidate.step_expanded_training = train
        rollouts = [SimpleNamespace(completion="ref")]
        try:
            assert prefetch_training_batch(candidate, rollouts) == (mode == "on")
            if mode != "on":
                assert not prepare_threads  # No eager work or thread creation.
                assert getattr(candidate, "_prepared_training_batch", None) is None
            if case == "bug":
                with pytest.raises(ValueError, match="bug"):
                    run_training_step(candidate, rollouts=rollouts)
                assert not trained
            else:
                report = run_training_step(
                    candidate,
                    rollouts=rollouts,
                    before_step=lambda: scheduler_calls.append(True),
                    do_save_checkpoint=True,
                )
                observations.append((trained, scheduler_calls, report))
            assert len(prepare_threads) == 1
            assert (prepare_threads[0] != main) == (mode == "on")
            assert getattr(candidate, "_prepared_training_batch", None) is None
        finally:
            packer.shutdown_prefetch(join_timeout=2)
    if observations:
        assert observations[0] == observations[1] == observations[2]


@pytest.mark.parametrize("enabled", [True, False])
def test_transport_resolution_with_or_without_prefetch(enabled):
    packer = Packer()
    if enabled:
        packer._setup_prefetch(prefetch_timeout=2)
    candidate = SimpleNamespace(
        data_packer=packer,
        batching_contract=ExpandedSampleBatching(partial_tail="include"),
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=3, mu_iterations=1)
            )
        ),
        step_expanded_training=Mock(return_value={}),
        prepare_training_batch=lambda rollouts: ExpandedTrainingBatch(
            tuple(
                packer.get_policy_input(rollout_output=r.completion) for r in rollouts
            )
        ),
    )
    rollouts = [SimpleNamespace(completion="ref")]
    try:
        assert prefetch_training_batch(candidate, rollouts) == enabled
        report = run_training_step(candidate, rollouts=rollouts)
        assert candidate.step_expanded_training.call_args.args[0].minibatches == (
            (1.0, 2.0),
        )
        assert report["batching/dropped_samples"] == 1
    finally:
        packer.shutdown_prefetch(join_timeout=2)


@pytest.fixture
def trainer():
    packer = Packer()
    packer._setup_prefetch(prefetch_timeout=2)
    trainer = SimpleNamespace(
        batching_contract=ExpandedSampleBatching(
            partial_tail="include", fixed_minibatches=1
        ),
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=3, mu_iterations=1)
            )
        ),
        data_packer=packer,
        step_expanded_training=Mock(return_value={}),
    )
    trainer.prepare_training_batch = lambda rollouts: ExpandedTrainingBatch(
        tuple(packer.get_policy_input(rollout_output=r.completion) for r in rollouts)
    )
    yield trainer
    packer.shutdown_prefetch(join_timeout=2)


def test_fetch_prepare_filter_overlaps_current_training_without_cache_rotation(trainer):
    current = [SimpleNamespace(completion="current")]
    upcoming = [SimpleNamespace(completion="next")]
    entered, release = threading.Event(), threading.Event()
    main = threading.get_ident()
    original = trainer.prepare_training_batch

    def prepare(rollouts):
        if rollouts[0] is upcoming[0]:
            assert threading.get_ident() != main
            entered.set()
            assert release.wait(2)
        return original(rollouts)

    trainer.prepare_training_batch = prepare
    prefetch_training_batch(trainer, current)
    trainer.data_packer._prefetch_cache = {"current": "active-cache"}

    def train(batch, **kwargs):
        assert threading.get_ident() == main
        assert batch.minibatches == ((1.0, 2.0),)
        prefetch_training_batch(trainer, upcoming)
        try:
            assert entered.wait(2)  # Next CPU preparation runs during this step.
            assert trainer.data_packer._prefetch_cache == {"current": "active-cache"}
            with pytest.raises(RuntimeError, match="one unconsumed"):
                prefetch_training_batch(trainer, upcoming)
        finally:
            release.set()
        return {}

    trainer.step_expanded_training = train
    with patch(
        "cosmos_rl.policy.trainer.batching._gather",
        side_effect=AssertionError("agreement already sealed"),
    ):
        run_training_step(trainer, rollouts=current)
        trainer.step_expanded_training = Mock(return_value={})
        report = run_training_step(trainer, rollouts=upcoming)
    assert report["batching/dropped_samples"] == 1
    assert trainer._prepared_training_batch is None
    assert trainer.data_packer._prepared_prefetch_future is None


@pytest.mark.parametrize("recoverable", [True, False])
def test_background_errors_reach_training_boundary(trainer, recoverable):
    trainer.prepare_training_batch = Mock(
        side_effect=(
            RecoverablePreparationError("missing") if recoverable else ValueError("bug")
        )
    )
    rollouts = [SimpleNamespace(completion="r")]
    prefetch_training_batch(trainer, rollouts)
    if recoverable:
        result = run_training_step(trainer, rollouts=rollouts)
        assert result["batching/preparation_failed"] == 1
        assert trainer.step_expanded_training.call_args.args[0].minibatches == ((),)
    else:
        with pytest.raises(ValueError, match="bug"):
            run_training_step(trainer, rollouts=rollouts)
    assert trainer._prepared_training_batch is None


def test_cannot_consume_another_commands_prepared_batch(trainer):
    rollouts = [SimpleNamespace(completion="r")]
    prefetch_training_batch(trainer, rollouts)
    with pytest.raises(ValueError, match="does not match"):
        run_training_step(trainer, rollouts=[SimpleNamespace(completion="r")])
    run_training_step(trainer, rollouts=rollouts)


def test_background_preparation_rejects_non_cpu_output(trainer):
    trainer.prepare_training_batch = lambda _: ExpandedTrainingBatch(
        ((torch.empty(1, device="meta"),),)
    )
    rollouts = []
    prefetch_training_batch(trainer, rollouts)
    with pytest.raises(ValueError, match="CPU samples"):
        run_training_step(trainer, rollouts=rollouts)


def test_preparation_deadline_is_terminal_and_retains_running_ownership(trainer):
    entered, release = threading.Event(), threading.Event()

    def prepare(_):
        entered.set()
        assert release.wait(2)
        return ExpandedTrainingBatch(())

    trainer.prepare_training_batch = prepare
    trainer.data_packer._prefetch_timeout_s = 0.01
    prefetch_training_batch(trainer, [])
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError):
            run_training_step(trainer, rollouts=[])
        assert trainer._prepared_training_batch is not None
        with pytest.raises(RuntimeError, match="one unconsumed"):
            prefetch_training_batch(trainer, [])
        # Future.result's consumer timeout and the independent watchdog run
        # on different threads. Keep preparation blocked until the watchdog
        # has sealed failure; releasing it first races with its completion.
        assert trainer.data_packer._prefetch_shutdown.wait(2)
    finally:
        release.set()
    with pytest.raises(TimeoutError):
        run_training_step(trainer, rollouts=[])
    with pytest.raises(TimeoutError):
        trainer.data_packer.start_prepared_prefetch([], lambda: None)


def test_completed_preparation_disarms_watchdog_before_delayed_consumption(trainer):
    trainer.data_packer._prefetch_timeout_s = 0.1
    prefetch_training_batch(trainer, [])
    trainer._prepared_training_batch[1].result(timeout=2)
    time.sleep(0.2)
    assert not trainer.data_packer._prefetch_timers
    assert trainer.data_packer._prefetch_failure is None
    run_training_step(trainer, rollouts=[])


@pytest.mark.parametrize("uncentralized", [True, False])
@pytest.mark.parametrize("count", [1, 3])
def test_dispatch_rejects_uneven_collection_without_dequeuing(uncentralized, count):
    worker = SimpleNamespace(
        trainer=SimpleNamespace(batching_contract=ExpandedSampleBatching()),
        replica_batch_for_this_step=count,
        dp_world_size=2,
        world_size=2,
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(uncentralized_training=uncentralized)
            )
        ),
        data_queue=Queue(),
    )
    for item in range(count):
        worker.data_queue.put(item)
    with pytest.raises(ValueError, match="refusing to silently round"):
        RLPolicyWorker.dispatch_rollouts(worker)
    assert worker.data_queue.qsize() == count


@pytest.mark.parametrize("uncentralized", [True, False])
def test_dispatch_consumes_every_validly_sharded_episode(uncentralized):
    episodes = [
        SimpleNamespace(prompt_idx=i, teacher_result_uuid=None) for i in range(6)
    ]
    shared_queue = Queue()
    for episode in episodes:
        shared_queue.put(episode)
    scattered = []
    received = []
    for rank in range(2):
        local_queue = Queue()
        for episode in episodes[rank::2]:
            local_queue.put(episode)
        worker = SimpleNamespace(
            trainer=SimpleNamespace(batching_contract=ExpandedSampleBatching()),
            replica_batch_for_this_step=6,
            dp_world_size=2,
            world_size=2,
            global_rank=rank,
            parallel_dims=SimpleNamespace(get_rank_in_dim=lambda dim, r: r),
            config=SimpleNamespace(
                train=SimpleNamespace(
                    local_dataset=False,
                    train_policy=SimpleNamespace(uncentralized_training=uncentralized),
                )
            ),
            data_queue=local_queue if uncentralized else shared_queue,
            prepare_teacher_uuids_for_prefetch=Mock(return_value=0),
        )

        def scatter(out, inputs, src):
            if rank == 0:
                scattered.extend(inputs)
            out[0] = scattered[rank]

        with patch(
            "cosmos_rl.policy.worker.rl_worker.dist.scatter_object_list",
            side_effect=scatter,
        ):
            received.extend(RLPolicyWorker.dispatch_rollouts(worker))
        assert worker.data_queue.empty()
    assert sorted(r.prompt_idx for r in received) == list(range(6))
