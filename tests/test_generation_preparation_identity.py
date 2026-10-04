# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Preparation belongs to queued occurrences, not repeating dataset indices."""

import threading
from types import SimpleNamespace

import pytest

from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin


class Backend(RolloutGenerationMixin):
    def __init__(self, prefetch, constant_key=False):
        self.config = SimpleNamespace(
            rollout=SimpleNamespace(prefetch_rollout=prefetch)
        )
        self.constant_key = constant_key
        self.second_prepared = threading.Event()
        self._engine_initialized = True
        self.setup_generation()

    def _payload_key(self, payload):
        return "custom-key" if self.constant_key else super()._payload_key(payload)

    def _prepare_sample(self, payload, **kwargs):
        if payload.value == "fail":
            raise ValueError("injected preparation failure")
        if payload.value == "second":
            self.second_prepared.set()
        return payload.value

    def _collate_batch(self, samples, **kwargs):
        return samples

    def _generate(self, batch, **kwargs):
        return batch

    def _postprocess(self, raw, payloads, **kwargs):
        return [
            (payload.value, value) for payload, value in zip(payloads, raw, strict=True)
        ]


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("separate_batches", [False, True])
@pytest.mark.parametrize(
    "repeated_index,constant_key", [(True, False), (False, False), (False, True)]
)
def test_consumers_keep_occurrence_identity(
    prefetch, separate_batches, repeated_index, constant_key
):
    backend = Backend(prefetch, constant_key)
    first = SimpleNamespace(prompt_idx=1, value="first")
    second = SimpleNamespace(prompt_idx=1 if repeated_index else 2, value="second")
    batches = [[first], [second]] if separate_batches else [[first, second]]
    try:
        for batch in batches:
            backend.submit_setup(batch)
        if prefetch:
            assert backend.second_prepared.wait(5)
        actual = [
            result for batch in batches for result in backend.rollout_generation(batch)
        ]
        assert actual == [("first", "first"), ("second", "second")]
        assert not backend._setup_futures
        assert not backend._setup_payloads
    finally:
        backend.shutdown_generation()


@pytest.mark.parametrize("repeated_index", [False, True])
def test_failed_batch_releases_every_occurrence_but_keeps_other_queued_work(
    repeated_index,
):
    backend = Backend(True)
    failed = SimpleNamespace(prompt_idx=1, value="fail")
    later = SimpleNamespace(prompt_idx=1 if repeated_index else 2, value="second")
    other = SimpleNamespace(prompt_idx=1, value="other")
    try:
        backend.submit_setup([failed, later])
        backend.submit_setup([other])
        assert backend.second_prepared.wait(5)
        assert backend.rollout_generation([failed, later]) == []
        assert set(backend._setup_futures) == {backend._setup_key(other)}
        assert set(backend._setup_payloads) == {backend._setup_key(other)}
        assert backend.rollout_generation([other]) == [("other", "other")]
        assert not backend._setup_futures and not backend._setup_payloads
    finally:
        backend.shutdown_generation()


def test_cold_payload_cannot_take_another_occurrences_preparation():
    backend = Backend(True)
    prepared = SimpleNamespace(prompt_idx=1, value="second")
    cold = SimpleNamespace(prompt_idx=1, value="cold")
    try:
        backend.submit_setup([prepared])
        assert backend.second_prepared.wait(5)
        assert backend.rollout_generation([cold]) == [("cold", "cold")]
        assert backend.rollout_generation([prepared]) == [("second", "second")]
        assert not backend._setup_futures
    finally:
        backend.shutdown_generation()
