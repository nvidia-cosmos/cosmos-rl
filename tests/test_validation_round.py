# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.validation_round import ValidationRound


def fetch(round_, indices, *, replica="a", sequence=0, end=True):
    return round_.fetch(
        round_.round_id,
        replica,
        sequence,
        (4, None),
        lambda: ([RLPayload(prompt_idx=index) for index in indices], end),
    )


def results(batch):
    return [payload.model_copy(update={"rewards": [1.0]}) for payload in batch.payloads]


def report(round_, payloads, *, reporter=("a", 0), sequence=0, end=False):
    return round_.report(round_.round_id, reporter, sequence, payloads, end)


@pytest.mark.parametrize("indices", [[], [7], [7, 7], list(range(9))])
def test_actual_issued_work_completes_including_empty_and_repeated_indices(indices):
    round_ = ValidationRound(2, {("a", 0)}, 1)
    batch = fetch(round_, indices)
    assert len({payload.validation_work_id for payload in batch.payloads}) == len(
        indices
    )
    if indices:
        assert report(round_, results(batch))
        assert not round_.complete
    assert report(round_, [], sequence=int(bool(indices)), end=True)
    assert round_.complete
    assert round_.reported_prompts == len(indices)
    assert not round_._fetch_receipts
    assert not round_._pending
    assert not report(round_, [], sequence=int(bool(indices)), end=True)


def test_lost_fetch_reply_reuses_exact_work_without_advancing_or_sharing_mutations():
    round_ = ValidationRound(2, {("a", 0)}, 1)
    producer = Mock(return_value=([RLPayload(prompt_idx=7, prompt="original")], False))
    first = round_.fetch(round_.round_id, "a", 0, (1, None), producer)
    first.payloads[0].prompt = "mutated"
    replay = round_.fetch(round_.round_id, "a", 0, (1, None), producer)
    producer.assert_called_once()
    assert replay.payloads[0].prompt == "original"
    assert replay.payloads[0].validation_work_id == first.payloads[0].validation_work_id
    with pytest.raises(ValueError, match="changed"):
        round_.fetch(round_.round_id, "a", 0, (2, None), producer)
    with pytest.raises(ValueError, match="Out-of-order"):
        round_.fetch(round_.round_id, "a", 2, (1, None), producer)
    producer.assert_called_once()


def test_lost_report_reply_does_not_count_twice_or_complete_another_assignment():
    round_ = ValidationRound(2, {("a", 0)}, 1)
    payloads = results(fetch(round_, [7, 8]))
    assert report(round_, payloads[:1])
    assert not report(round_, payloads[:1])
    assert round_.reported_prompts == 1
    assert not round_.complete
    with pytest.raises(ValueError, match="changed"):
        report(round_, payloads[1:])
    assert report(round_, payloads[1:], sequence=1)
    assert report(round_, [], sequence=2, end=True)
    assert round_.complete


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "duplicate", "wrong_prompt", "missing_generation", "nonfinite"],
)
def test_invalid_report_is_atomic(mutation):
    round_ = ValidationRound(2, {("a", 0)}, 1)
    payloads = results(fetch(round_, [7, 8]))
    bad = copy.deepcopy(payloads)
    if mutation == "unknown":
        bad[1].validation_work_id = 999
    elif mutation == "duplicate":
        bad[1] = copy.deepcopy(bad[0])
    elif mutation == "wrong_prompt":
        bad[1].prompt_idx = 7
    elif mutation == "missing_generation":
        bad[1].rewards = []
    else:
        bad[1].rewards = [float("nan")]
    with pytest.raises(ValueError):
        report(round_, bad)
    assert round_.reported_prompts == 0
    assert len(round_._pending) == 2
    assert not round_._report_receipts
    assert report(round_, payloads, end=True)
    assert round_.complete


def test_every_sealed_reporter_including_empty_ranks_must_end():
    round_ = ValidationRound(2, {("a", 0), ("a", 1), ("b", 0)}, 1)
    a = results(fetch(round_, [7]))
    fetch(round_, [], replica="b")
    with pytest.raises(ValueError, match="own pending"):
        report(round_, a, reporter=("b", 0))
    assert report(round_, a, end=True)
    assert report(round_, [], reporter=("a", 1), end=True)
    assert not round_.complete
    assert report(round_, [], reporter=("b", 0), end=True)
    assert round_.complete


def test_early_or_incomplete_terminal_report_cannot_erase_pending_work():
    round_ = ValidationRound(2, {("a", 0)}, 1)
    payloads = results(fetch(round_, [7], end=False))
    with pytest.raises(ValueError, match="before fetch exhaustion"):
        report(round_, payloads, end=True)
    assert round_.reported_prompts == 0
    fetch(round_, [], sequence=1)
    with pytest.raises(ValueError, match="outstanding"):
        report(round_, [], end=True)
    assert not round_._ended
    assert report(round_, payloads, end=True)


@pytest.mark.parametrize("operation", ["fetch", "report"])
def test_an_old_execution_or_round_cannot_touch_a_new_round_at_the_same_step(operation):
    old = ValidationRound(2, {("a", 0)}, 1)
    current = ValidationRound(2, {("a", 0)}, 1)
    producer = Mock(return_value=([], True))
    with pytest.raises(ValueError, match="Stale"):
        if operation == "fetch":
            current.fetch(old.round_id, "a", 0, (1, None), producer)
        else:
            current.report(old.round_id, ("a", 0), 0, [], True)
    producer.assert_not_called()
    assert not current._report_receipts
    assert not current.complete


def test_retired_report_sequence_is_rejected_not_readmitted():
    round_ = ValidationRound(2, {("a", 0)}, 1)
    payloads = results(fetch(round_, [7, 8]))
    report(round_, payloads[:1])
    report(round_, payloads[1:], sequence=1)
    with pytest.raises(ValueError, match="Out-of-order"):
        report(round_, payloads[:1])
    assert round_.reported_prompts == 2
    assert len(round_._report_receipts) == 1


def test_failed_sampler_is_not_retried_into_a_new_batch():
    round_ = ValidationRound(2, {("a", 0)}, 1)
    producer = Mock(side_effect=RuntimeError("sampler failed after advancing"))
    with pytest.raises(RuntimeError, match="sampler failed"):
        round_.fetch(round_.round_id, "a", 0, (1, None), producer)
    with pytest.raises(RuntimeError, match="after sampler admission"):
        round_.fetch(round_.round_id, "a", 0, (1, None), producer)
    with pytest.raises(RuntimeError, match="after sampler admission"):
        report(round_, [], end=True)
    producer.assert_called_once()
    assert not round_.complete
