# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests; NCCL traffic is faked, allocation and unpack are real."""

import gc
import queue
import threading
import time
import weakref
from collections import deque
from unittest import mock

import numpy as np
import pytest
import torch

from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBudget,
    ReceiveMemoryError,
    ReceivedBatch,
    storage_bytes,
)
from cosmos_rl.utils.payload_transport.nccl import strategy as nccl
from cosmos_rl.utils.payload_transport.nccl.comm_cache import CommCache
from cosmos_rl.utils.payload_transport.nccl.header import HEADER_NBYTES, build_header
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.trajectory import TensorSpec, schema_layout
from cosmos_rl.utils.transport_failure import TransportUnusableError


@pytest.mark.parametrize("limit", [-1, 0, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        ReceiveBudget(limit, 1)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        ReceiveBudget(100, timeout)


def test_oversize_does_not_wait_or_reserve():
    budget = ReceiveBudget(100, 1)
    with pytest.raises(ReceiveMemoryError, match="reduce batch/payload size"):
        budget.reserve(101)
    assert budget.used == budget.waits == 0


def test_admission_wait_release_and_shutdown():
    budget = ReceiveBudget(100, 2)
    budget.reserve(70)
    admitted = threading.Event()
    errors = []

    def worker():
        try:
            budget.reserve(50)
            admitted.set()
        except ReceiveMemoryError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=worker)
    t.start()
    deadline = time.monotonic() + 1
    while not budget.waits and time.monotonic() < deadline:
        time.sleep(0.001)
    assert budget.waits == 1
    assert not admitted.is_set()
    budget.release(70)
    t.join(1)
    assert admitted.is_set()
    assert budget.used == 50
    budget.reserve(50)
    t = threading.Thread(target=worker)
    t.start()
    budget.close()
    t.join(1)
    assert not t.is_alive()
    assert errors and "shutdown" in errors[0]
    assert budget.peak <= budget.limit


def test_admission_timeout_is_actionable():
    budget = ReceiveBudget(100, 0.01)
    budget.reserve(100)
    with pytest.raises(ReceiveMemoryError, match="release_prefetch"):
        budget.reserve(1)
    assert budget.used == 100


def test_unique_backing_storage_and_final_release():
    tensor = torch.arange(16)
    payload = {"a": tensor[:2], "b": tensor[5:]}
    size = storage_bytes([payload, payload])
    assert size == 16 * 8
    budget = ReceiveBudget(size, 1)
    budget.reserve(size)
    budget.attribute(decoded=size)
    lease = ReceivedBatch({"episode": payload}, budget, size)
    assert lease["episode"]["a"].tolist() == [0, 1]
    assert lease["episode"]["a"].tolist() == [0, 1]
    del tensor, payload
    lease.release()
    lease.release()
    assert not lease
    assert budget.used == budget.decoded == 0


def test_release_records_all_streams_before_waiting_and_keeps_charge():
    tensor = torch.ones(4)
    budget = ReceiveBudget(16, 1)
    budget.reserve(16)
    budget.attribute(decoded=16)
    lease = ReceivedBatch({"episode": {"a": tensor}}, budget, 16)
    calls = []

    class Event:
        def record(self, stream):
            calls.append(("record", stream))

        def synchronize(self):
            assert budget.used == 16
            assert lease
            calls.append(("wait", None))

    with mock.patch("torch.cuda.Event", Event):
        lease.release(streams=["training", "reward"])
    assert calls == [
        ("record", "training"),
        ("record", "reward"),
        ("wait", None),
        ("wait", None),
    ]
    assert budget.used == 0


def make_receiver(monkeypatch, limit=1000):
    receiver = nccl.NCCLTransportStrategy()
    receiver._device = torch.device("cpu")
    receiver._rendezvous = object()
    receiver._receiver_rank = 0
    receiver._warm_pairs = set()
    receiver._recv_lock = threading.Lock()
    receiver._bounded_fetch_lock = threading.Lock()
    receiver._receive_budget = ReceiveBudget(limit, 0.2)
    receiver._comm_cache = CommCache(build_fn=lambda u, r: 5, abort_fn=lambda i: None)
    receiver._comm_cache.get_or_create(("sender", 0, 0), uid_chars=[1], local_rank=1)
    raw_refs = []
    seen = []

    def rendezvous(ref, pynccl):
        schema = ref["schema"]
        _, size = schema_layout(schema)
        raw = nccl._alloc_recv_buffer(schema, "cpu")
        receiver._receive_budget.attribute(raw=raw.untyped_storage().nbytes())
        raw[:HEADER_NBYTES] = torch.tensor(
            list(build_header(transfer_id=ref["transfer_id"], payload_nbytes=size)),
            dtype=torch.uint8,
        )
        raw[HEADER_NBYTES:] = torch.arange(size, dtype=torch.uint8)
        raw_refs.append(weakref.ref(raw))
        seen.append(ref["transfer_id"])
        return 5, raw

    monkeypatch.setattr(receiver, "_rendezvous_one", rendezvous)
    import cosmos_rl.utils.pynccl as pynccl

    monkeypatch.setattr(pynccl, "nccl_recv", lambda *a, **kw: None)
    monkeypatch.setattr(nccl, "record_event", lambda stream: None)
    monkeypatch.setattr(nccl, "wait_event", lambda stream, event: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return receiver, raw_refs, seen


def refs_for(sizes):
    return [
        (
            i,
            {
                "transfer_id": f"episode-{i}",
                "sender_replica": "sender",
                "sender_rank": 0,
                "schema": [
                    TensorSpec(name="odd", shape=(n,), dtype=np.uint8),
                    TensorSpec(name="aligned", shape=(1,), dtype=np.int64),
                ],
            },
        )
        for i, n in enumerate(sizes)
    ]


def test_incremental_receive_alignment_equivalence_and_budget(monkeypatch):
    receiver, raw_refs, seen = make_receiver(monkeypatch, limit=90)
    refs = refs_for([3, 5, 9])
    batch, nbytes, _ = receiver._fetch_all(refs)
    assert isinstance(batch, ReceivedBatch)
    assert seen == ["episode-0", "episode-1", "episode-2"]
    assert all(r() is None for r in raw_refs)
    for idx, ref in refs:
        _, size = schema_layout(ref["schema"])
        expected = nccl._unpack(
            torch.arange(size, dtype=torch.uint8), ref["schema"], "cpu"
        )
        for key in expected:
            torch.testing.assert_close(batch[idx][key], expected[key])
    stats = receiver.receive_memory_stats()
    assert stats["raw_receive_bytes"] == 0
    assert stats["decoded_leased_bytes"] == 41
    assert stats["reserved_bytes"] == 41
    assert stats["peak_tensor_bytes"] == 90
    assert stats["peak_reserved_bytes"] == 90
    assert nbytes == 41 + 3 * HEADER_NBYTES
    batch.release()
    assert receiver.receive_memory_stats()["reserved_bytes"] == 0


def test_oversized_batch_rejected_before_rendezvous(monkeypatch):
    receiver, _, seen = make_receiver(monkeypatch, limit=60)
    with pytest.raises(ReceiveMemoryError, match="exceeding"):
        receiver._fetch_all(refs_for([3, 5, 9]))
    assert seen == []
    assert receiver._receive_budget.used == 0


def test_partial_decode_failure_releases_all_accounting(monkeypatch):
    receiver, raw_refs, _ = make_receiver(monkeypatch)
    original = nccl._unpack
    calls = 0

    def unpack(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            raise torch.OutOfMemoryError("injected decode OOM")
        return result

    monkeypatch.setattr(nccl, "_unpack", unpack)
    with pytest.raises(ReceiveMemoryError, match="injected decode OOM"):
        receiver._fetch_all(refs_for([3, 5, 9]))
    gc.collect()
    assert all(r() is None for r in raw_refs)
    stats = receiver.receive_memory_stats()
    assert (
        stats["reserved_bytes"]
        == stats["decoded_leased_bytes"]
        == stats["raw_receive_bytes"]
        == 0
    )


def test_mid_window_allocation_failure_retains_pending_receive(monkeypatch):
    receiver, raw_refs, _ = make_receiver(monkeypatch)
    rendezvous = receiver._rendezvous_one

    def fail_second_allocation(ref, pynccl):
        if ref["transfer_id"] == "episode-1":
            raise torch.OutOfMemoryError("injected allocation after pending recv")
        return rendezvous(ref, pynccl)

    monkeypatch.setattr(receiver, "_rendezvous_one", fail_second_allocation)
    with pytest.raises(TransportUnusableError, match="before native completion"):
        receiver._fetch_all(refs_for([3, 5, 9]))
    gc.collect()
    # nccl_recv enqueued work; no receive-complete event was recorded before
    # the next allocation failed. A compute-stream sync cannot prove this
    # independent transfer stream stopped using its raw receive storage.
    assert len(raw_refs) == 1 and raw_refs[0]() is not None
    assert receiver._receive_budget.closed
    assert receiver._receive_budget.used > 0
    assert receiver._receive_budget.raw > 0


@pytest.mark.parametrize("failure", ["enqueue", "record", "wait", "completion"])
def test_uncertain_receive_retains_pins_storage_and_budget(monkeypatch, failure):
    receiver, raw_refs, _ = make_receiver(monkeypatch)
    # Isolate test quarantine; production never clears terminal ownership.
    monkeypatch.setattr(nccl, "_FAILED_RECEIVE_OWNERS", [])
    rendezvous = receiver._rendezvous_one
    pair = ("sender", 0, 0)
    cache = receiver._comm_cache

    def pinned_rendezvous(ref, pynccl):
        cache.get_or_create(pair, uid_chars=[1], local_rank=1, pin=True)
        return rendezvous(ref, pynccl)

    def fail(*args, **kwargs):
        raise RuntimeError("injected uncertain native completion")

    monkeypatch.setattr(receiver, "_rendezvous_one", pinned_rendezvous)
    import cosmos_rl.utils.pynccl as pynccl

    if failure == "enqueue":
        monkeypatch.setattr(pynccl, "nccl_recv", fail)
    elif failure == "record":
        monkeypatch.setattr(nccl, "record_event", fail)
    elif failure == "wait":
        monkeypatch.setattr(nccl, "wait_event", fail)
    else:
        monkeypatch.setattr(nccl, "record_event", lambda *args: mock.Mock(query=fail))

    budget, cache = receiver._receive_budget, receiver._comm_cache
    with pytest.raises(TransportUnusableError):
        receiver._fetch_all(refs_for([3]))
    assert cache.pinned_count(pair) == 1
    # Even a caller incorrectly swallowing the terminal error and destroying
    # its strategy cannot free operands behind an uncertain native operation.
    del receiver
    gc.collect()
    assert raw_refs[0]() is not None
    assert budget.closed and budget.used > 0 and budget.raw > 0


def test_failed_decode_completion_retains_allocated_tensor_storages(monkeypatch):
    receiver, raw_refs, _ = make_receiver(monkeypatch)
    monkeypatch.setattr(nccl, "_FAILED_RECEIVE_OWNERS", [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    synchronize = mock.Mock(side_effect=RuntimeError("decode completion failed"))
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args: mock.Mock(synchronize=synchronize)
    )
    decoded_refs = []
    original = nccl._unpack

    def fail_after_clone(*args, on_allocate, **kwargs):
        def retain(tensor):
            decoded_refs.append(weakref.ref(tensor))
            on_allocate(tensor)

        original(*args, on_allocate=retain, **kwargs)
        raise torch.OutOfMemoryError("decode allocation failure")

    monkeypatch.setattr(nccl, "_unpack", fail_after_clone)
    budget = receiver._receive_budget
    with pytest.raises(TransportUnusableError, match="CUDA completion failed"):
        receiver._fetch_all(refs_for([3]))
    del receiver
    gc.collect()
    assert all(ref() is not None for ref in raw_refs + decoded_refs)
    assert decoded_refs and budget.closed and budget.decoded > 0 and budget.used > 0


def test_previous_lease_backpressures_next_batch(monkeypatch):
    receiver, _, _ = make_receiver(monkeypatch, limit=90)
    first, _, _ = receiver._fetch_all(refs_for([3, 5, 9]))
    with pytest.raises(ReceiveMemoryError, match="release_prefetch"):
        receiver._fetch_all(refs_for([3, 5, 9]))
    first.release()
    second, _, _ = receiver._fetch_all(refs_for([3, 5, 9]))
    second.release()
    assert receiver._receive_budget.peak == 90


def test_prefetch_error_propagation_and_unreleased_cache_guard():
    packer = PrefetchDataPackerMixin()
    packer._prefetch_enabled = True
    packer._prefetch_result_queue = queue.Queue()
    packer._prefetch_outstanding = deque([0])
    packer._prefetch_cache = {}
    packer._prefetch_result_queue.put((0, ReceiveMemoryError("oversized"), 0))
    with pytest.raises(ReceiveMemoryError, match="oversized"):
        packer.wait_prefetch()
    packer._prefetch_failure = None  # independent unreleased-cache scenario
    budget = ReceiveBudget(10, 1)
    budget.reserve(10)
    budget.attribute(decoded=10)
    packer._prefetch_cache = ReceivedBatch({}, budget, 10)
    with pytest.raises(ReceiveMemoryError, match="release_prefetch"):
        packer.wait_prefetch()
    packer.release_prefetch()
    assert budget.used == 0


def test_worker_drops_previous_result_reference():
    refs = []
    entered_second = threading.Event()
    finish_second = threading.Event()

    class Packer(PrefetchDataPackerMixin):
        def _fetch_batch(self, tasks):
            if refs:
                entered_second.set()
                finish_second.wait(2)
            tensor = torch.ones(1)
            refs.append(weakref.ref(tensor))
            return {"x": tensor}

    packer = Packer()
    packer._setup_prefetch()
    try:
        packer._prefetch_request_queue.put((0, []))
        packer._prefetch_outstanding.append(0)
        packer.wait_prefetch()
        packer.release_prefetch()
        packer._prefetch_request_queue.put((1, []))
        packer._prefetch_outstanding.append(1)
        assert entered_second.wait(1)
        gc.collect()
        assert refs[0]() is None
    finally:
        finish_second.set()
        packer.shutdown_prefetch()


@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_repeated_batches_have_bounded_peaks_without_leaks(monkeypatch, batch_size):
    sizes = [3 + i % 5 for i in range(batch_size)]
    decoded = sum(n + 8 for n in sizes)
    required = decoded + HEADER_NBYTES + max(sizes) + 8
    receiver, raw_refs, _ = make_receiver(monkeypatch, limit=required)
    for _ in range(5):
        batch, _, _ = receiver._fetch_all(refs_for(sizes))
        assert storage_bytes(batch.values()) == decoded
        batch.release()
        assert receiver._receive_budget.used == 0
    assert all(ref() is None for ref in raw_refs)
    stats = receiver.receive_memory_stats()
    assert stats["peak_tensor_bytes"] <= required
    assert stats["peak_reserved_bytes"] == required
    assert stats["peak_reserved_bytes"] == required


def test_completed_batch_keeps_lease_when_cache_keys_are_remapped(monkeypatch):
    from cosmos_rl.utils.trajectory import serialize_schema

    receiver, _, _ = make_receiver(monkeypatch)
    tasks = [
        (
            i,
            {
                "_nccl": True,
                "_transfer_id": ref["transfer_id"],
                "_sender_replica": "sender",
                "_sender_rank": 0,
                "_schema": serialize_schema(ref["schema"]),
            },
        )
        for i, ref in refs_for([3, 5])
    ]
    batch = receiver.fetch_batch(tasks)
    assert isinstance(batch, ReceivedBatch)
    assert set(batch) == {"episode-0", "episode-1"}
    assert receiver._receive_budget.used == 24
    batch.release()
    assert receiver._receive_budget.used == 0


@pytest.mark.parametrize("shutdown", ["shutdown_prefetch", "close_transport"])
def test_shutdown_releases_queued_but_not_consumer_owned_leases(shutdown):
    budget = ReceiveBudget(20, 1)
    budget.reserve(20)
    budget.attribute(decoded=20)
    packer = PrefetchDataPackerMixin()
    packer._prefetch_result_queue = queue.Queue()
    packer._prefetch_result_queue.put((0, ReceivedBatch({}, budget, 10), 0))
    packer._prefetch_cache = ReceivedBatch({}, budget, 10)
    getattr(packer, shutdown)()
    assert budget.used == 10
    assert isinstance(packer._prefetch_cache, ReceivedBatch)
    packer.release_prefetch()
    assert budget.used == 0


def test_close_timeout_retains_queued_and_consumer_leases_until_worker_exits():
    budget = ReceiveBudget(20, 1)
    budget.reserve(20)
    budget.attribute(decoded=20)
    packer = PrefetchDataPackerMixin()
    strategy = mock.Mock()
    packer.set_transport_strategy(strategy)
    packer._prefetch_result_queue = queue.Queue()
    queued = ReceivedBatch({}, budget, 10)
    consumer = ReceivedBatch({}, budget, 10)
    packer._prefetch_result_queue.put((0, queued, 0))
    packer._prefetch_cache = consumer
    finished = threading.Event()
    packer._prefetch_thread = threading.Thread(target=finished.wait)
    packer._prefetch_thread.start()
    try:
        with pytest.raises(TimeoutError):
            packer.close_transport(timeout=0.01)
        assert budget.used == 20
        assert not queued.released and not consumer.released
        assert packer._prefetch_cache is consumer
        strategy.shutdown.assert_not_called()
    finally:
        finished.set()
        packer.close_transport()
    strategy.before_join.assert_called_once()
    strategy.shutdown.assert_called_once()
    assert queued.released and not consumer.released
    assert packer._prefetch_cache is consumer
    assert budget.used == 10
    packer.release_prefetch()
    assert budget.used == 0


def test_close_failure_keeps_unconsumed_lease_owned():
    budget = ReceiveBudget(16, 1)
    budget.reserve(16)
    budget.attribute(decoded=16)
    lease = ReceivedBatch({}, budget, 16)
    packer = PrefetchDataPackerMixin()
    packer._prefetch_result_queue = queue.Queue()
    packer._prefetch_result_queue.put((0, lease, 0))
    with mock.patch.object(lease, "release", side_effect=RuntimeError("event failed")):
        with pytest.raises(RuntimeError, match="transport teardown failed"):
            packer.close_transport()
    assert budget.used == 16 and not lease.released
    assert packer._prefetch_result_queue.get_nowait()[1] is lease
    assert not packer._transport_close_operation.completed
    lease.release()


def test_stream_completion_failure_preserves_lease_and_charge():
    budget = ReceiveBudget(16, 1)
    budget.reserve(16)
    budget.attribute(decoded=16)
    batch = ReceivedBatch({"e": {"a": torch.ones(4)}}, budget, 16)
    event = mock.Mock()
    event.synchronize.side_effect = RuntimeError("stream failure")
    with mock.patch("torch.cuda.Event", return_value=event):
        with pytest.raises(RuntimeError, match="stream failure"):
            batch.release(streams=["reader"])
    assert batch and not batch.released
    assert budget.used == budget.decoded == 16
    batch.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_cuda_views_and_multiple_reader_streams():
    schema = [TensorSpec(name="x", shape=(1024,), dtype=np.float32)]
    raw = torch.arange(4096, device="cuda", dtype=torch.uint8)
    payload = nccl._unpack(raw, schema, "cuda")
    # Use a known finite value so equality also checks all asynchronous readers.
    payload["x"].fill_(2)
    ready = torch.cuda.Event()
    ready.record()
    readers = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in readers:
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            view = payload["x"][::2]
            outputs.append(view.sum())
    size = storage_bytes([payload])
    budget = ReceiveBudget(size, 1)
    budget.reserve(size)
    budget.attribute(decoded=size)
    batch = ReceivedBatch({"e": payload}, budget, size)
    del payload, view
    batch.release(streams=readers)
    assert budget.used == 0
    assert [out.item() for out in outputs] == [1024, 1024]


def test_bounded_timeout_cannot_consume_late_result_as_next_batch():
    packer = PrefetchDataPackerMixin()
    packer._setup_prefetch(prefetch_timeout=0.01)
    try:
        # No work is queued: simulate an outstanding receive that never returns.
        batch_id = packer._arm_prefetch_deadline()
        packer._prefetch_outstanding.append(batch_id)
        assert packer._prefetch_shutdown.wait(2)
        with pytest.raises(TimeoutError, match="exceeded"):
            packer.wait_prefetch()
        packer._prefetch_result_queue.put((batch_id, {"late": True}, 0))
        with pytest.raises(TimeoutError, match="exceeded"):
            packer.wait_prefetch()
        assert packer._prefetch_cache == {}
    finally:
        packer.shutdown_prefetch()


@pytest.mark.parametrize("limit,expected", [(90, [1, 1, 1]), (150, [2, 1]), (200, [3])])
def test_receive_window_scales_with_available_budget(monkeypatch, limit, expected):
    receiver, raw_refs, _ = make_receiver(monkeypatch, limit)
    widths = []
    original = receiver._fetch_unbounded

    def fetch(refs):
        assert all(ref() is None for ref in raw_refs)
        widths.append(len(refs))
        return original(refs)

    monkeypatch.setattr(receiver, "_fetch_unbounded", fetch)
    batch, _, _ = receiver._fetch_all(refs_for([3, 5, 9]))
    assert widths == expected
    assert receiver._receive_budget.peak <= limit
    assert receiver._receive_budget.peak_live <= receiver._receive_budget.peak
    batch.release()
    assert receiver._receive_budget.used == 0


@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("consume", [False, True])
@pytest.mark.parametrize("shutdown", ["shutdown_prefetch", "close_transport"])
def test_prepared_lease_survives_until_consumer_release_or_shutdown(
    error, consume, shutdown
):
    budget = ReceiveBudget(64, 1)

    class Packer(PrefetchDataPackerMixin):
        def _filter_prefetch_tasks(self, rollouts):
            return [(0, "ref")]

        def _fetch_batch(self, tasks):
            budget.reserve(64)
            budget.attribute(decoded=64)
            return ReceivedBatch({"ref": {"x": torch.ones(16)}}, budget, 64)

    packer = Packer()
    packer._setup_prefetch()

    def prepare():
        if error:
            raise ValueError("bad preparation")
        return packer._preparation_local.cache["ref"]["x"][:2]

    try:
        future = packer.start_prepared_prefetch([], prepare)
        if error:
            with pytest.raises(ValueError, match="bad preparation"):
                future.result(timeout=2)
        else:
            future.result(timeout=2)
        assert budget.used == 64
        if consume:
            packer.release_prepared_prefetch(future)
            assert budget.used == 64
            assert isinstance(packer._prefetch_cache, ReceivedBatch)
            del future  # Drop prepared aliases before final-reader release.
            packer.release_prefetch()
            assert budget.used == 0
    finally:
        getattr(packer, shutdown)()
    assert budget.used == 0


def test_lease_keeps_storage_alive_when_consumer_mutates_dictionary():
    tensor = torch.ones(16)
    storage = weakref.ref(tensor.untyped_storage())
    budget = ReceiveBudget(64, 1)
    budget.reserve(64)
    budget.attribute(decoded=64)
    batch = ReceivedBatch({"episode": {"tensor": tensor}}, budget, 64)
    del tensor
    batch["episode"].clear()
    gc.collect()
    assert storage() is not None
    assert budget.used == 64
    batch.release()
    gc.collect()
    assert storage() is None
    assert budget.used == 0
