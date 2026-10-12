# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Cosmos acceptance tests: real Redis/NCCL and two CUDA devices.

These tests exercise the public consumer contract without NDAS. Explicit fault
injection uses real received CUDA storage; it does not claim a physical device OOM.
"""

import gc
import time
import uuid
import weakref
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from cosmos_rl.utils.payload_transport.nccl import strategy as nccl
from cosmos_rl.utils.payload_transport.receive_memory import ReceiveMemoryError
from cosmos_rl.utils.trajectory import build_trajectory_schema, schema_layout
from cosmos_rl.utils.transport_failure import TransportUnusableError
from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
    prefetch_training_batch,
    run_training_step,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two real CUDA devices and Redis/NCCL",
)

LENGTHS = (17, 23, 31)
OBS_DIM = 63
MAX_BYTES = schema_layout(
    build_trajectory_schema(dict(max_steps=max(LENGTHS), obs_dim=OBS_DIM, action_dim=3))
)[1]


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before deadline")


def trajectory(seed, length, device):
    return {
        "observations": torch.arange(
            length * OBS_DIM, device=device, dtype=torch.float32
        ).reshape(length, OBS_DIM)
        / 1000
        + seed / 100,
        "actions": torch.arange(length * 3, device=device, dtype=torch.float32).reshape(
            length, 3
        )
        / 100,
        "rewards": torch.full((length,), 0.5, device=device),
        "episode_length": length - 2,
    }


@pytest.fixture
def endpoint():
    import redis
    import test_nccl_e2e as e2e
    from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

    assert e2e._ensure_redis(), "Redis unavailable: refusing an inconclusive GPU test"
    endpoints = []

    def create(budget):
        token = uuid.uuid4().hex
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name=f"bounded-acceptance-{token}"),
            custom=dict(
                nccl_receive_budget_bytes=budget, nccl_receive_admission_timeout=15
            ),
        )
        client = redis.Redis(
            host=e2e.REDIS_HOST, port=e2e.REDIS_PORT, decode_responses=True
        )
        producers = []
        packer = e2e._ConsumerPacker()
        endpoints.append((producers, packer))
        torch.cuda.set_device(0)
        for index, length in enumerate(LENGTHS):
            producer = NCCLRolloutMixin()
            producer.setup_nccl(
                replica_id=f"producer-{token}-{index}",
                rollout_idx=index,
                redis_client=client,
                config=config,
                device=torch.device("cuda:0"),
                sender_rank=0,
                max_steps=length,
                obs_dim=OBS_DIM,
                action_dim=3,
                registry_capacity=64,
            )
            producers.append(producer)
        torch.cuda.set_device(1)
        packer._nccl_dp_receiver_replica = f"consumer-{token}"
        packer._setup_nccl_data_packer(
            device=torch.device("cuda:1"),
            redis_client=client,
            config=config,
            prefetch_timeout=30,
            recv_timeout=5,
            first_transfer_timeout=30,
        )

        def batch(count=6, seed=0):
            torch.cuda.set_device(0)
            metas = []
            for i in range(count):
                which = i % len(producers)
                data = trajectory(seed + i, LENGTHS[which], "cuda:0")
                meta = producers[which].write_to_buffer(data)
                assert meta is not None
                metas.append(meta)
            torch.cuda.set_device(1)
            return metas

        return SimpleNamespace(
            packer=packer,
            strategy=packer._transport_strategy,
            producers=producers,
            batch=batch,
        )

    yield create
    for producers, packer in reversed(endpoints):
        torch.cuda.set_device(1)
        packer.shutdown_nccl_data_packer()
        # Tests must drop their aliases themselves. This is failure cleanup only.
        packer.release_prefetch()
        for producer in producers:
            producer.cleanup_nccl()
    torch.cuda.set_device(0)


def collect(env, refs):
    env.packer.start_prefetch(refs)
    env.packer.wait_prefetch()
    assert len(env.packer._prefetch_cache) == len(refs)


def train_step(packer, refs, seed, weight):
    reference_weight = weight.detach().clone().requires_grad_()
    received_losses, reference_losses = [], []
    for i, ref in enumerate(refs):
        payload = packer.get_policy_input(rollout_output=ref)
        assert packer.get_policy_input(rollout_output=ref) is payload
        expected = trajectory(seed + i, LENGTHS[i % len(LENGTHS)], "cuda:1")
        length = expected["episode_length"]
        for name in ("observations", "actions", "rewards"):
            torch.testing.assert_close(
                payload[name], expected[name][:length], rtol=0, atol=0
            )
        received_losses.append(
            (payload["observations"] @ weight - payload["actions"].sum(-1))
            .square()
            .mean()
        )
        reference_losses.append(
            (
                expected["observations"][:length] @ reference_weight
                - expected["actions"][:length].sum(-1)
            )
            .square()
            .mean()
        )
    loss = torch.stack(received_losses).mean()
    reference_loss = torch.stack(reference_losses).mean()
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)
    with torch.no_grad():
        weight.add_(weight.grad, alpha=-0.0001)
    weight.grad = None
    return loss.item()


@pytest.mark.parametrize("bounded", [False, True])
def test_twenty_training_steps_match_direct_data_loss_and_gradients(endpoint, bounded):
    env = endpoint(8 * MAX_BYTES + 32 if bounded else 0)
    weight = torch.linspace(-0.01, 0.01, OBS_DIM, device="cuda:1", requires_grad=True)
    for step in range(20):
        refs = env.batch(count=6, seed=step * 10)
        collect(env, refs)
        train_step(env.packer, refs, step * 10, weight)
        env.packer.release_prefetch()
        if bounded:
            stats = env.strategy.receive_memory_stats()
            assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0
            assert stats["peak_tensor_bytes"] <= stats["budget_bytes"]


def pending_reader(packer, refs):
    stream = torch.cuda.Stream(device=1)
    ready = torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        # Deliberately keep a final reader pending while the next prefetch runs.
        torch.cuda._sleep(3_000_000_000)
        values = [
            packer.get_policy_input(rollout_output=ref)["observations"][::2].sum()
            for ref in refs
        ]
        done = torch.cuda.Event()
        done.record()
    return stream, done, values


def test_prefetch_finishes_while_prior_cuda_reader_is_pending(endpoint):
    env = endpoint(32 * MAX_BYTES + 64)
    # Warm all three producer pairs before making a timing-dependent assertion.
    collect(env, env.batch())
    env.packer.release_prefetch()
    current = env.batch(seed=10)
    following = env.batch(seed=20)
    collect(env, current)
    stream, done, values = pending_reader(env.packer, current)
    env.packer.start_prefetch(following)
    eventually(lambda: not env.packer._prefetch_result_queue.empty())
    assert not done.query(), "next prefetch did not complete during the delayed reader"
    env.packer.release_prefetch(streams=[stream])
    assert done.query()
    assert all(torch.isfinite(value).item() for value in values)
    env.packer.wait_prefetch()
    assert len(env.packer._prefetch_cache) == len(following)
    env.packer.release_prefetch()
    stats = env.strategy.receive_memory_stats()
    assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0
    assert stats["admission_waits"] == 0


def test_slow_final_reader_backpressures_then_allows_progress(endpoint):
    env = endpoint(7 * MAX_BYTES + 32)
    current = env.batch(seed=10)
    following = env.batch(seed=20)
    collect(env, current)
    stream, done, values = pending_reader(env.packer, current)
    env.packer.start_prefetch(following)
    eventually(lambda: env.strategy.receive_memory_stats()["admission_waits"] == 1)
    assert env.packer._prefetch_result_queue.empty()
    assert not done.query()
    env.packer.release_prefetch(streams=[stream])
    assert done.query()
    assert all(torch.isfinite(value).item() for value in values)
    env.packer.wait_prefetch()
    env.packer.release_prefetch()
    stats = env.strategy.receive_memory_stats()
    assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0
    assert stats["peak_tensor_bytes"] <= stats["budget_bytes"]


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("fixed", [None, 1])
def test_generic_training_boundary_releases_real_payloads(endpoint, prepared, fixed):
    env = endpoint(7 * MAX_BYTES + 32)
    # Warm native setup before shortening the steady-state watchdog.
    collect(env, env.batch())
    env.packer.release_prefetch()
    env.packer._prefetch_timeout_s = 2
    env.strategy._receive_budget.timeout = 0.1
    current, upcoming = env.batch(seed=10), env.batch(seed=20)
    stream = torch.cuda.Stream(device=1)
    reader_done = torch.cuda.Event()
    weight = torch.linspace(-0.01, 0.01, OBS_DIM, device="cuda:1", requires_grad=True)
    trainer = SimpleNamespace(
        data_packer=env.packer,
        train_stream=stream,
        batching_contract=ExpandedSampleBatching(
            partial_tail="include", fixed_minibatches=fixed
        ),
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=6, mu_iterations=1)
            )
        ),
    )

    def prepare(rollouts):
        samples = []
        for ref in rollouts:
            payload = env.packer.get_policy_input(rollout_output=ref)
            values = (payload["observations"], payload["actions"])
            # Background preparation is CPU-only. Inline preparation retains
            # genuine received GPU views until the generic final-reader boundary.
            samples.append(
                tuple(value.cpu() for value in values) if prepared else values
            )
        return ExpandedTrainingBatch((tuple(samples),))

    trainer.prepare_training_batch = prepare
    steps = []
    reader_values = []

    def train(batch, **kwargs):
        assert env.strategy._receive_budget.consumer_bytes > 0
        producing_stream = torch.cuda.current_stream(1)
        with torch.cuda.stream(stream):
            stream.wait_stream(producing_stream)
            reference = weight.detach().clone().requires_grad_()
            losses, expected_losses = [], []
            seed = 10 if not steps else 20
            for index, (observations, actions) in enumerate(batch.minibatches[0]):
                observations, actions = observations.to("cuda:1"), actions.to("cuda:1")
                expected = trajectory(seed + index, LENGTHS[index % 3], "cuda:1")
                length = expected["episode_length"]
                losses.append((observations @ weight - actions.sum(-1)).square().mean())
                expected_losses.append(
                    (
                        expected["observations"][:length] @ reference
                        - expected["actions"][:length].sum(-1)
                    )
                    .square()
                    .mean()
                )
            loss, expected_loss = (
                torch.stack(losses).mean(),
                torch.stack(expected_losses).mean(),
            )
            loss.backward()
            expected_loss.backward()
            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            torch.testing.assert_close(weight.grad, reference.grad, rtol=0, atol=0)
            with torch.no_grad():
                weight.add_(weight.grad, alpha=-0.0001)
            weight.grad = None
        if not steps:
            if prepared:
                prefetch_training_batch(trainer, upcoming)
            else:
                env.packer.start_prefetch(upcoming)
            eventually(lambda: env.strategy._receive_budget.waiting_for_consumer)
            time.sleep(4.5)  # Healthy training exceeds both configured deadlines.
            assert env.packer._prefetch_failure is None
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            reader_values.append(batch.minibatches[0][0][0].to("cuda:1").sum())
            reader_done.record()
        steps.append(seed)
        return {"loss": loss.item()}

    trainer.step_expanded_training = train
    if prepared:
        prefetch_training_batch(trainer, current)
    for refs in (current, upcoming):
        run_training_step(trainer, rollouts=refs)
        assert reader_done.query(), "training stream was not included in final release"
        assert torch.isfinite(reader_values[-1]).item()
        assert env.strategy._receive_budget.consumer_bytes == 0
    assert steps == [10, 20]
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_shutdown_wakes_real_prefetch_admission_without_releasing_consumer(endpoint):
    env = endpoint(7 * MAX_BYTES + 32)
    collect(env, env.batch())
    leased = env.strategy.receive_memory_stats()["decoded_leased_bytes"]
    env.packer.start_prefetch(env.batch(seed=10))
    eventually(lambda: env.strategy.receive_memory_stats()["admission_waits"] == 1)
    start = time.monotonic()
    env.packer.shutdown_nccl_data_packer()
    assert time.monotonic() - start < 5
    assert env.packer._prefetch_thread is None
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == leased
    env.packer.release_prefetch()
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_oversized_payload_never_contacts_sender(endpoint):
    env = endpoint(1)
    with mock.patch.object(
        env.strategy, "_rendezvous_one", wraps=env.strategy._rendezvous_one
    ) as negotiate:
        env.packer.start_prefetch(env.batch(count=1))
        with pytest.raises(ReceiveMemoryError, match="exceeding"):
            env.packer.wait_prefetch()
        negotiate.assert_not_called()
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_decode_fault_cleans_real_cuda_allocations(endpoint, monkeypatch):
    env = endpoint(32 * MAX_BYTES + 32)
    refs = env.batch()
    # Use the same collection boundary as the post-fault measurement; earlier
    # trainer fixtures can have cycles retaining unrelated CUDA tensors.
    gc.collect()
    torch.cuda.synchronize(1)
    before = torch.cuda.memory_allocated(1)
    original = nccl._unpack
    count = 0

    def faulty_unpack(*args, **kwargs):
        nonlocal count
        payload = original(*args, **kwargs)
        count += 1
        if count == 2:
            raise torch.OutOfMemoryError("injected after real CUDA decode")
        return payload

    monkeypatch.setattr(nccl, "_unpack", faulty_unpack)
    env.packer.start_prefetch(refs)
    with pytest.raises(ReceiveMemoryError, match="injected after real CUDA decode"):
        env.packer.wait_prefetch()
    torch.cuda.synchronize(1)
    gc.collect()
    assert torch.cuda.memory_allocated(1) == before
    stats = env.strategy.receive_memory_stats()
    assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0


def test_mid_window_fault_keeps_actual_pending_nccl_storage(endpoint, monkeypatch):
    """A real NCCL receive stays owned when the next allocation cannot proceed.

    Call below the worker's fatal-exit boundary to inspect ownership. The test
    subsequently proves stream completion for fixture teardown; production must
    terminate and must not treat catching this error as permission to continue.
    """
    import cosmos_rl.utils.pynccl as pynccl

    env = endpoint(32 * MAX_BYTES + 32)
    # Warm communicator creation before injecting a pending device operation.
    collect(env, env.batch(count=1))
    env.packer.release_prefetch()
    refs = [
        (i, nccl._parse_ref(meta, default_schema=env.strategy._schema))
        for i, meta in enumerate(env.batch(count=2, seed=10))
    ]
    original_recv = pynccl.nccl_recv
    original_rendezvous = env.strategy._rendezvous_one
    done = torch.cuda.Event()
    raw_refs, streams = [], []

    def delayed_recv(tensor, *args, stream, **kwargs):
        raw_refs.append(weakref.ref(tensor))
        streams.append(stream)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(3_000_000_000)
        original_recv(tensor, *args, stream=stream, **kwargs)
        done.record(stream)

    def fail_second(ref, module):
        if ref["transfer_id"] == refs[1][1]["transfer_id"]:
            assert not done.query(), "fault must occur with actual pending GPU work"
            raise torch.OutOfMemoryError("injected mid-window allocation failure")
        return original_rendezvous(ref, module)

    monkeypatch.setattr(pynccl, "nccl_recv", delayed_recv)
    monkeypatch.setattr(env.strategy, "_rendezvous_one", fail_second)
    with pytest.raises(TransportUnusableError, match="before native completion"):
        env.strategy._fetch_all(refs)
    gc.collect()
    assert len(raw_refs) == 1 and raw_refs[0]() is not None
    stats = env.strategy.receive_memory_stats()
    assert stats["reserved_bytes"] > 0 and stats["raw_receive_bytes"] > 0
    assert env.strategy._receive_budget.closed
    pair = nccl._pair_key(refs[0][1], env.strategy._receiver_rank)
    assert env.strategy._comm_cache.pinned_count(pair) == 1
    # Test-only completion proof; never free on the basis of abort/timeout alone.
    streams[0].synchronize()
    assert done.query()


def test_missing_payload_preserves_other_payloads_and_following_progress(endpoint):
    env = endpoint(32 * MAX_BYTES + 32)
    refs = env.batch(count=3)
    assert env.producers[1]._nccl_registry.free(refs[1]["_transfer_id"])
    env.packer.start_prefetch(refs)
    env.packer.wait_prefetch()
    assert set(env.packer._prefetch_cache) == {refs[i]["_transfer_id"] for i in (0, 2)}
    env.packer.release_prefetch()
    collect(env, env.batch(count=3, seed=10))
    env.packer.release_prefetch()
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_dictionary_mutation_cannot_drop_storage_before_final_cuda_reader(endpoint):
    env = endpoint(32 * MAX_BYTES + 32)
    refs = env.batch()
    collect(env, refs)
    stream, done, values = pending_reader(env.packer, refs)
    allocated = torch.cuda.memory_allocated(1)
    for payload in env.packer._prefetch_cache.values():
        payload.clear()
    del payload
    gc.collect()
    assert not done.query()
    assert torch.cuda.memory_allocated(1) == allocated
    env.packer.release_prefetch(streams=[stream])
    assert done.query()
    assert all(torch.isfinite(value).item() for value in values)
    assert torch.cuda.memory_allocated(1) < allocated
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_concurrent_window_and_prepared_cpu_lease(endpoint, monkeypatch):
    env = endpoint(64 * MAX_BYTES + 256)
    widths = []
    original = env.strategy._fetch_unbounded

    def receive(refs):
        widths.append(len(refs))
        return original(refs)

    monkeypatch.setattr(env.strategy, "_fetch_unbounded", receive)
    refs = env.batch(count=6)

    def prepare():
        return [
            {key: tensor.cpu() for key, tensor in payload.items()}
            for payload in env.packer._preparation_local.cache.values()
        ]

    future = env.packer.start_prepared_prefetch(refs, prepare)
    prepared = future.result(timeout=30)
    assert widths == [6]
    assert len(prepared) == 6
    assert all(t.device.type == "cpu" for p in prepared for t in p.values())
    stats = env.strategy.receive_memory_stats()
    assert 0 < stats["reserved_bytes"] <= stats["budget_bytes"]
    assert stats["peak_tensor_bytes"] <= stats["peak_reserved_bytes"]
    env.packer.release_prepared_prefetch(future)
    assert env.strategy.receive_memory_stats()["reserved_bytes"] > 0
    del future, prepared
    env.packer.release_prefetch()
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0
