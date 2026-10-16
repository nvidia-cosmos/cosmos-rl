# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_rl.utils.payload_transport.context import PayloadContext, PayloadRef
from cosmos_rl.utils.payload_transport.layout import TensorLayout
from cosmos_rl.utils.payload_transport.storage import WritablePayload
from cosmos_rl.utils.payload_transport.nccl.transport import NcclPayloadTransport
from cosmos_rl.utils.payload_transport.ucxx.transport import UCXXPayloadTransport
from cosmos_rl.utils.trajectory import TensorSpec


def _schema(name="value"):
    return [TensorSpec(name=name, shape=(4,), dtype=np.float32)]


class _Producer:
    def __init__(self, default_layout):
        self.default_layout = default_layout
        self.registered = []
        self.published = []
        self.abandoned = []
        self.closed = 0
        self.reject_registration = False
        self.reject_close = False
        self.active = 0
        self.peak_active = 0
        self._counter_lock = threading.Lock()

    def register_layout(self, layout):
        if self.reject_registration:
            raise ValueError("backend rejected layout")
        self.registered.append(layout)

    def publish(self, tensors, *, layout, copy_policy="batched"):
        with self._counter_lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        time.sleep(0.001)
        with self._counter_lock:
            self.active -= 1
        self.published.append((tensors, layout, copy_policy))
        return {"transfer": len(self.published), "layout": layout.layout_id}

    def acquire(self, *, layout):
        storage = torch.empty(layout.payload_nbytes, dtype=torch.uint8)
        return WritablePayload(
            owner=self,
            layout=layout,
            storage=storage,
            tensors=layout.views(storage),
            opaque=storage,
            on_abandon=lambda: self.abandoned.append(storage),
        )

    def publish_storage(self, payload, *, ready_event=None):
        payload._submit(self)
        return {"transfer": "direct", "layout": payload.layout.layout_id}

    def close(self, timeout=None):
        self.closed += 1
        if self.reject_close:
            raise RuntimeError("backend close failed")


def _context(*, layout_cache_capacity=64):
    layout = TensorLayout.compile(_schema())
    producer = _Producer(layout)
    return (
        PayloadContext(
            "nccl",
            producer,
            layout,
            layout_cache_capacity=layout_cache_capacity,
        ),
        producer,
    )


def test_context_requires_explicit_layout_registration():
    context, producer = _context()
    unregistered = TensorLayout.compile(_schema("other"))

    with pytest.raises(KeyError, match="unknown tensor layout"):
        context.publish_tensors(
            {"other": torch.ones(4, dtype=torch.float32)}, layout=unregistered
        )

    registered = context.register_layout(_schema("other"))
    reference = context.publish_tensors(
        {"other": torch.ones(4, dtype=torch.float32)}, layout=registered
    )
    assert isinstance(reference, PayloadRef)
    assert producer.registered == [registered]


def test_failed_backend_registration_does_not_enter_context_cache():
    context, producer = _context()
    layout = TensorLayout.compile(_schema("rejected"))
    producer.reject_registration = True

    with pytest.raises(ValueError, match="backend rejected"):
        context.register_layout(_schema("rejected"))
    with pytest.raises(KeyError, match="unknown tensor layout"):
        context.publish_tensors(
            {"rejected": torch.ones(4, dtype=torch.float32)}, layout=layout
        )


def test_layout_registry_is_bounded_without_evicting_live_identities():
    context, producer = _context(layout_cache_capacity=1)
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        context.register_layout(_schema("other"))

    reference = context.publish_tensors({"value": torch.ones(4, dtype=torch.float32)})

    assert reference["layout"] == producer.default_layout.layout_id


@pytest.mark.parametrize("submit_first", [False, True])
def test_failed_direct_publish_releases_only_unsubmitted_storage(submit_first):
    context, producer = _context()
    payload = context.acquire()
    payload.mark_all_written()

    def fail(candidate, *, ready_event=None):
        if submit_first:
            candidate._submit(producer)
        raise RuntimeError("publication failed")

    producer.publish_storage = fail
    with pytest.raises(RuntimeError, match="publication failed"):
        context.publish(payload)

    assert len(producer.abandoned) == (0 if submit_first else 1)
    context.close()


def test_context_serializes_concurrent_publication_and_close_owns_slots():
    context, producer = _context()
    with torch.no_grad():
        with pytest.raises(ValueError, match="not acquired"):
            foreign = producer.acquire(layout=producer.default_layout)
            context.publish(foreign)
        foreign.close()

    errors = []

    def publish():
        try:
            context.publish_tensors({"value": torch.ones(4, dtype=torch.float32)})
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=publish) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert producer.peak_active == 1

    abandoned_before_close = len(producer.abandoned)
    outstanding = context.acquire()
    context.close()
    assert len(producer.abandoned) == abandoned_before_close + 1
    assert producer.abandoned[-1] is outstanding.storage
    assert producer.closed == 1
    with pytest.raises(RuntimeError, match="closing or closed"):
        context.acquire()


def test_close_attempts_every_owner_and_reports_failures():
    context, producer = _context()
    first = context.acquire()
    second = context.acquire()
    failures = 0

    def fail_once():
        nonlocal failures
        failures += 1
        if failures == 1:
            raise RuntimeError("abandon failed")
        producer.abandoned.append(first.storage)

    first._on_abandon = fail_once

    with pytest.raises(RuntimeError, match="1 unpublished operation"):
        context.close()

    assert any(storage is second.storage for storage in producer.abandoned)
    assert producer.closed == 0
    context.close()
    assert any(storage is first.storage for storage in producer.abandoned)
    assert producer.closed == 1


def test_backend_close_failure_remains_retryable():
    context, producer = _context()
    producer.reject_close = True

    with pytest.raises(RuntimeError, match="shutdown failed"):
        context.close()

    assert producer.closed == 1
    producer.reject_close = False
    context.close()
    assert producer.closed == 2


@pytest.mark.parametrize(
    ("transport", "close_attribute"),
    [
        (NcclPayloadTransport(), "_nccl_close_operation"),
        (UCXXPayloadTransport(), "_ucxx_close_operation"),
    ],
)
def test_backend_close_hook_supports_composed_and_bare_producers(
    transport, close_attribute
):
    calls = []
    context = SimpleNamespace(
        close=lambda *, timeout: calls.append(("context", timeout))
    )
    transport.close_producer(SimpleNamespace(payload_context=context), timeout=7)

    bare = SimpleNamespace(close=lambda *, timeout: calls.append(("bare", timeout)))
    setattr(bare, close_attribute, object())
    transport.close_producer(bare, timeout=11)

    assert calls == [("context", 7), ("bare", 11)]


def test_submit_and_abandon_are_atomic():
    context, producer = _context()
    payload = context.acquire()
    payload.mark_all_written()
    barrier = threading.Barrier(2)
    outcomes = []

    def publish():
        barrier.wait()
        try:
            context.publish(payload)
            outcomes.append("published")
        except RuntimeError:
            outcomes.append("closed")

    thread = threading.Thread(target=publish)
    thread.start()
    barrier.wait()
    payload.close()
    thread.join()

    assert outcomes in (["published"], ["closed"])
    assert len(producer.abandoned) in (0, 1)
    assert not (outcomes == ["published"] and producer.abandoned)
    context.close()


def test_direct_publish_rejects_unattested_fields_before_backend_ownership():
    context, producer = _context()
    payload = context.acquire()
    payload.tensors["value"].fill_(3)

    with pytest.raises(RuntimeError, match="not marked fully written"):
        context.publish(payload)

    assert producer.abandoned == [payload.storage]
    assert producer.published == []


def test_direct_field_attestation_is_explicit_and_complete():
    layout = TensorLayout.compile(_schema("first") + _schema("second"))
    producer = _Producer(layout)
    context = PayloadContext("nccl", producer, layout)
    payload = context.acquire()
    payload.tensors["first"].zero_()
    payload.mark_written("first")
    assert payload.missing_fields == {"second"}

    with pytest.raises(RuntimeError, match="second"):
        context.publish(payload)

    with pytest.raises(RuntimeError, match="no longer writable"):
        payload.mark_written("second")


def test_borrowed_storage_reports_backend_capability():
    context, _ = _context()

    with pytest.raises(NotImplementedError, match="does not accept borrowed"):
        context.storage_requirements()
    with pytest.raises(NotImplementedError, match="does not accept borrowed"):
        context.borrow(torch.empty(64, dtype=torch.uint8))
