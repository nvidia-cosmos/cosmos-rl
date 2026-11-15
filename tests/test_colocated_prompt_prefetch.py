# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Colocated lookahead stays inside a single shared-model rollout phase."""

from queue import Queue
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.colocated.rl_worker import ColocatedRLControlWorker
from cosmos_rl.rollout import State
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from test_rollout_generation_mixin import _FakeBackend, _FakePayload


@pytest.fixture
def worker_factory(monkeypatch):
    backends = []
    monkeypatch.setattr(
        "cosmos_rl.rollout.worker.rollout_control.dist_utils.broadcast_object_cpu",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cosmos_rl.rollout.worker.colocated.rollout_control.dist_util.all_gather_object_cpu",
        lambda value: [value],
    )

    def make(*, prefetch=True, depth=2, counts=(1, 1), eof=True, backend=None):
        worker = object.__new__(ColocatedRolloutControlWorker)
        worker.global_rank = worker.rank_in_rollout_repicas = 0
        worker.batch_size = 1
        worker.current_weight_version = 0
        worker._prompt_fetch_lock = threading.Lock()
        worker._prompt_queue = Queue(maxsize=depth)
        worker.state = State()
        worker.shutdown_signal = threading.Event()
        worker.config = NS(
            train=NS(
                local_dataset=False,
                train_policy=NS(
                    data_dispatch_as_rank_in_mesh=False, uncentralized_training=False
                ),
            ),
            rollout=NS(
                prefetch_rollout=prefetch, prefetch_queue_maxsize=depth, n_generation=1
            ),
        )
        worker.parallel_dims = NS(world_size=1, mesh={"dp": NS(size=lambda: 1)})
        worker.data_packer = object()
        worker.data_fetcher = object()
        worker.rollout = backend or _FakeBackend(prefetch=prefetch)
        backends.append(worker.rollout)
        worker.rollout.submit_setup = Mock(wraps=worker.rollout.submit_setup)
        fetched = []
        generated = []

        def fetch(*args, **kwargs):
            index = len(fetched)
            assert index < len(counts), "leased past the declared prompt stream"
            start = sum(counts[:index])
            payloads = [
                dict(prompt_idx=i, prompt=str(i), weight_version=0)
                for i in range(start, start + counts[index])
            ]
            fetched.append(len(payloads))
            return payloads, eof and index == len(counts) - 1

        worker.api_client = NS(get_next_prompt=Mock(side_effect=fetch))

        def generate():
            generated.append((len(fetched), worker._prompt_queue.qsize()))
            batch = worker._prompt_queue.get_nowait()
            return batch, worker.rollout.rollout_generation(batch)

        worker.one_step_generation = generate
        return worker, fetched, generated

    yield make
    for backend in backends:
        backend.shutdown_generation()


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_bounded_lookahead_and_final_batch_drain(worker_factory, prefetch, depth):
    worker, fetched, generated = worker_factory(
        prefetch=prefetch, depth=depth, counts=(1,) * 5
    )
    for remaining in range(5, 0, -1):
        end, count = worker.rollout_for_one_minor_step(remaining_samples=remaining)
        assert count == 1
        assert end == (remaining == 1)
        assert worker._prompt_queue.qsize() <= min(depth - 1, remaining - 1)
    assert worker._prompt_queue.empty()
    assert len(fetched) == len(generated) == 5
    assert generated[0] == (min(depth, 5) if prefetch else 1,) * 2
    assert worker.rollout.submit_setup.call_count == (5 if prefetch else 0)
    if prefetch:
        assert set(worker.rollout.thread_names["_prepare_sample"]) == {
            "FakeBackendPrefetch"
        }
        assert worker.rollout._prefetch_data_packer is worker.data_packer
        assert worker.rollout._prefetch_data_fetcher is worker.data_fetcher
        assert not worker.rollout._setup_futures


@pytest.mark.parametrize("counts", [(2, 3, 1), (1, 3, 1)])
def test_actual_fetched_count_prevents_leasing_into_next_update(worker_factory, counts):
    worker, fetched, _ = worker_factory(counts=counts, depth=4, eof=False)
    pending = 0
    while pending < 4:
        end, count = worker.rollout_for_one_minor_step(remaining_samples=4 - pending)
        assert not end
        pending += count
    assert len(fetched) == 2
    assert worker._prompt_queue.empty()
    assert not worker.rollout._setup_futures


def test_rejection_refills_only_after_consumed_work_is_reported(worker_factory):
    worker, fetched, _ = worker_factory(counts=(1,) * 4, depth=4)
    worker.rollout._fail_prepare_for = {0}
    end, count = worker.rollout_for_one_minor_step(remaining_samples=2)
    assert (end, count, len(fetched)) == (False, 0, 2)
    # The rejected batch did not provide a training sample. A replacement can
    # now be admitted, while the existing second batch remains first in FIFO.
    assert worker.rollout_for_one_minor_step(remaining_samples=2) == (False, 1)
    assert len(fetched) == 3
    assert worker.rollout_for_one_minor_step(remaining_samples=1) == (False, 1)
    assert len(fetched) == 3
    assert worker._prompt_queue.empty()


@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("n_generation", [1, 4])
def test_decentralized_uneven_queue_bounds_per_rank_capacity(
    worker_factory, monkeypatch, depth, n_generation
):
    worker, fetched, _ = worker_factory(counts=(3, 2), depth=depth, eof=False)
    worker.config.train.train_policy.uncentralized_training = True
    worker.config.rollout.n_generation = n_generation
    worker.parallel_dims.world_size = 2
    worker.parallel_dims.mesh["dp"] = NS(
        size=lambda: 2, get_local_rank=lambda: 0, get_group=lambda: None
    )

    def scatter(output, values, **kwargs):
        output[0] = values[0]

    monkeypatch.setattr("torch.distributed.scatter_object_list", scatter)
    # After an asymmetric rejection the backlogs can be [0, n_generation].
    # A three-prompt batch can supply [2, 1] * n_generation and finish this
    # update. Its global count of three must not admit another queued batch.
    worker.rollout_for_one_minor_step(remaining_samples=4 * n_generation)
    assert fetched == [3]
    assert worker._prompt_queue.empty()
    assert not worker.rollout._setup_futures


def test_transient_empty_fetch_does_not_spin_or_claim_eof(worker_factory):
    worker, fetched, _ = worker_factory(counts=(0, 1))
    assert worker.rollout_for_one_minor_step(remaining_samples=2) == (False, 0)
    assert fetched == [0]
    assert worker.rollout_for_one_minor_step(remaining_samples=2) == (True, 1)
    assert fetched == [0, 1]


def test_real_worker_overlaps_next_preparation_with_current_generation(worker_factory):
    backend = _FakeBackend(prefetch=True)
    generation_started = threading.Event()
    next_prepared = threading.Event()
    original_prepare = backend._prepare_sample
    original_generate = backend._generate

    def prepare(payload, **kwargs):
        if payload.prompt_idx == 1:
            assert generation_started.wait(5), "generation never overlapped setup"
            next_prepared.set()
        return original_prepare(payload, **kwargs)

    def generate(batch, **kwargs):
        if not generation_started.is_set():
            generation_started.set()
            assert next_prepared.wait(5), "next setup was never scheduled"
        return original_generate(batch, **kwargs)

    backend._prepare_sample = prepare
    backend._generate = generate
    worker, fetched, _ = worker_factory(backend=backend)
    assert worker.rollout_for_one_minor_step(remaining_samples=2) == (False, 1)
    assert next_prepared.is_set() and len(fetched) == 2
    assert worker.rollout_for_one_minor_step(remaining_samples=1) == (True, 1)


@pytest.mark.parametrize("prefetch", [False, True])
def test_model_preparation_keeps_rollout_no_grad_context(worker_factory, prefetch):
    worker, _, _ = worker_factory(prefetch=prefetch, counts=(1,))
    weight = torch.nn.Parameter(torch.ones(1))
    prepared = []
    original_prepare = worker.rollout._prepare_sample

    def prepare(payload, **kwargs):
        prepared.append(weight.square())
        return original_prepare(payload, **kwargs)

    worker.rollout._prepare_sample = prepare
    assert worker.rollout_for_one_minor_step(remaining_samples=1) == (True, 1)
    assert len(prepared) == 1
    assert not prepared[0].requires_grad, "rollout setup retained an autograd graph"


@pytest.mark.parametrize("replace_running", [False, True])
@pytest.mark.parametrize("same_batch", [False, True])
def test_failed_preparation_fences_own_running_callbacks(replace_running, same_batch):
    backend = _FakeBackend(prefetch=True)
    running = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    errors = []
    # A backend may recover with usable fallback results, letting the learner
    # train immediately after this call. The failed batch still owns its setup.
    backend._on_generation_error = lambda err, payloads: [
        ("fallback", payload.prompt_idx) for payload in payloads
    ]

    def prepare(payload, **kwargs):
        if payload.prompt_idx == 0:
            raise ValueError("first sample failed")
        running.set()
        assert release.wait(5)
        return payload.prompt_idx

    backend._prepare_sample = prepare
    payloads = [_FakePayload(i) for i in range(2)]
    backend.submit_setup(payloads)
    assert running.wait(5)
    if replace_running:
        backend.submit_setup([payloads[1]])

    def consume():
        try:
            errors.append(
                backend.rollout_generation(payloads if same_batch else payloads[:1])
            )
        finally:
            returned.set()

    consumer = threading.Thread(target=consume)
    consumer.start()
    try:
        if same_batch:
            assert not returned.wait(0.1), (
                "failed batch escaped with setup still running"
            )
        else:
            assert returned.wait(1), "failed batch waited for another batch's setup"
        release.set()
        consumer.join(5)
        assert returned.is_set() and errors == [
            [
                ("fallback", p.prompt_idx)
                for p in (payloads if same_batch else payloads[:1])
            ]
        ]
        if not same_batch:
            assert backend.rollout_generation(payloads[1:]) == [("result", 1)]
        assert not backend._setup_futures
    finally:
        release.set()
        consumer.join(5)
        backend.shutdown_generation()


def test_colocated_teardown_fences_preparation_before_shared_policy_resources():
    events = []
    worker = NS(
        rollout=NS(rollout=NS(shutdown_generation=lambda: events.append("setup"))),
        controller=NS(requested_stop_complete=False),
        policy=NS(destroy_worker=lambda: events.append("policy")),
    )
    ColocatedRLControlWorker.destroy_worker(worker)
    assert events == ["setup", "policy"]


@pytest.mark.parametrize("depth", [2, 4])
def test_multiple_completions_use_existing_generation_count(worker_factory, depth):
    worker, fetched, _ = worker_factory(depth=depth, counts=(1, 1, 1), eof=False)
    worker.config.rollout.n_generation = 8
    assert worker.rollout_for_one_minor_step(remaining_samples=16) == (False, 1)
    assert fetched == [1, 1]
    assert worker.rollout_for_one_minor_step(remaining_samples=8) == (False, 1)
    assert fetched == [1, 1] and worker._prompt_queue.empty()


def test_future_weight_prompt_is_not_prepared_early(worker_factory):
    worker, _, _ = worker_factory(counts=(1,))
    worker.api_client.get_next_prompt.side_effect = None
    worker.api_client.get_next_prompt.return_value = (
        [dict(prompt_idx=0, prompt="p", weight_version=2)],
        True,
    )
    assert worker.rollout_for_one_minor_step(remaining_samples=1) == (True, 0)
    worker.rollout.submit_setup.assert_not_called()


def test_global_count_survives_uneven_scatter(worker_factory, monkeypatch):
    worker, fetched, _ = worker_factory(counts=(3, 1), depth=4, eof=False)
    worker.parallel_dims.mesh["dp"] = NS(
        size=lambda: 2, get_local_rank=lambda: 0, get_group=lambda: None
    )

    def scatter(output, values, **kwargs):
        output[0] = values[0]

    monkeypatch.setattr("torch.distributed.scatter_object_list", scatter)
    assert not worker._prefetch_colocated_prompts(3)
    assert fetched == [3], "local DP slice count must not cause another lease"
    batch = worker._prompt_queue.get_nowait()
    assert len(batch) == 2 and batch.global_prompt_count == 3
    worker.rollout.rollout_generation(batch)


def test_ragged_eof_does_not_start_unconsumable_preparation(
    worker_factory, monkeypatch
):
    worker, fetched, _ = worker_factory(counts=(1,))
    worker.rollout.supports_empty_dp_batches = True
    worker.parallel_dims.mesh["dp"] = NS(
        size=lambda: 2, get_local_rank=lambda: 0, get_group=lambda: None
    )

    def scatter(output, values, **kwargs):
        output[0] = values[0]

    monkeypatch.setattr("torch.distributed.scatter_object_list", scatter)
    monkeypatch.setattr(
        "cosmos_rl.rollout.worker.colocated.rollout_control.dist_util.all_gather_object_cpu",
        lambda value: [value, (False, True)],
    )
    assert worker.rollout_for_one_minor_step(remaining_samples=2) == (True, 0)
    assert fetched == [1]
    worker.rollout.submit_setup.assert_not_called()
    assert not worker.rollout._setup_futures


def test_idle_setup_worker_releases_consumed_preparation(worker_factory, monkeypatch):
    import gc
    import weakref

    worker, _, _ = worker_factory(counts=(1,))
    prepared_refs = []

    class Prepared:
        pass

    def prepare(payload, **kwargs):
        sample = Prepared()
        prepared_refs.append(weakref.ref(sample))
        return sample

    worker.rollout._prepare_sample = prepare
    idle = threading.Event()
    requests = worker.rollout._setup_request_queue
    original_get = requests.get

    def get(*args, **kwargs):
        if prepared_refs:
            idle.set()
        return original_get(*args, **kwargs)

    monkeypatch.setattr(requests, "get", get)
    assert worker.rollout_for_one_minor_step(remaining_samples=1) == (True, 1)
    assert idle.wait(5)
    assert not worker.rollout._setup_futures
    # Leave the setup worker alive and idle, as it is during learner training.
    gc.collect()
    assert prepared_refs[0]() is None, "idle setup thread retains the last sample"


def test_nonterminal_ragged_batch_fails_before_rank_queue_divergence(
    worker_factory, monkeypatch
):
    worker, _, generated = worker_factory(counts=(1, 2), eof=False)
    worker.rollout.supports_empty_dp_batches = True
    worker.parallel_dims.mesh["dp"] = NS(
        size=lambda: 2, get_local_rank=lambda: 0, get_group=lambda: None
    )

    def scatter(output, values, **kwargs):
        output[0] = values[0]

    monkeypatch.setattr("torch.distributed.scatter_object_list", scatter)
    with pytest.raises(ValueError, match="nonempty DP slices before dataset EOF"):
        worker.rollout_for_one_minor_step(remaining_samples=4)
    worker.rollout.submit_setup.assert_not_called()
    assert generated == []
