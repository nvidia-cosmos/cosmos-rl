# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Temporal context, padding, admission and gradients for application batches."""

import copy

import pytest
import torch

from cosmos_rl.execution.sequences import (
    SequenceSample,
    SequenceWindow,
    collate_sequence_windows,
    iter_sequence_batches,
    sequence_windows,
)


def sample(name, length, ends=()):
    terminal = torch.zeros(length, dtype=torch.bool)
    for end in ends:
        terminal[end - 1] = True
    return SequenceSample(
        name,
        {"observations": torch.randn(length, 4), "actions": torch.randn(length, 2)},
        {"supervised": torch.arange(length) % 3 != 1},
        terminal,
    )


def test_context_never_crosses_episode_and_each_target_contributes_once():
    row = sample("episodes", 11, ends=(3, 8))
    coverage = torch.zeros(row.length, dtype=torch.int32)
    batches = list(
        iter_sequence_batches(
            [row], window_steps=2, context_steps=2, batch_size=3, max_bytes=4096
        )
    )
    for batch in batches:
        for i, window in enumerate(batch.windows):
            assert batch.reset_mask[i].sum() == 1
            assert batch.reset_mask[i, 0]
            assert not row.episode_ends[window.start : window.stop - 1].any()
            for j in batch.loss_masks["supervised"][i].nonzero().flatten().tolist():
                coverage[window.start + j] += 1
            assert not batch.loss_masks["supervised"][
                i, : window.loss_start - window.start
            ].any()
        assert not (batch.loss_masks["supervised"] & ~batch.valid_mask).any()
    assert torch.equal(coverage, row.loss_masks["supervised"].int())


def test_padded_temporal_gradients_match_individual_windows():
    torch.manual_seed(37)
    rows = [sample("a", 7, ends=(4,)), sample("b", 3)]
    initial = torch.nn.GRU(4, 2, batch_first=True)
    count = sum(int(row.loss_masks["supervised"].sum()) for row in rows)
    reference = copy.deepcopy(initial)
    losses = []
    for row in rows:
        for w in sequence_windows(row, window_steps=3, context_steps=2):
            output, _ = reference(row.fields["observations"][None, w.start : w.stop])
            values = (
                (output[0] - row.fields["actions"][w.start : w.stop]).square().mean(-1)
            )
            mask = row.loss_masks["supervised"][w.start : w.stop].clone()
            mask[: w.loss_start - w.start] = False
            losses.append(values[mask].sum() / count)
    sum(losses).backward()
    actual = copy.deepcopy(initial)
    actual_loss = 0
    for batch in iter_sequence_batches(
        rows, window_steps=3, context_steps=2, batch_size=3, max_bytes=4096
    ):
        output, _ = actual(batch.fields["observations"])
        values = (output - batch.fields["actions"]).square().mean(-1)
        loss = values[batch.loss_masks["supervised"]].sum() / count
        actual_loss += float(loss.detach())
        loss.backward()
    assert actual_loss == pytest.approx(float(sum(losses).detach()), abs=3e-7)
    for (name, a), (_, b) in zip(
        actual.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5, msg=name)


def test_budget_splits_before_padding_and_rejects_before_allocation(monkeypatch):
    row = sample("a", 8)
    # Each step: 6 float32 values plus valid/reset/objective booleans = 27B.
    batches = list(
        iter_sequence_batches([row], window_steps=3, batch_size=4, max_bytes=162)
    )
    assert [len(batch.windows) for batch in batches] == [2, 1]
    for batch in batches:
        tensors = [
            *batch.fields.values(),
            batch.valid_mask,
            batch.reset_mask,
            *batch.loss_masks.values(),
        ]
        assert batch.allocated_bytes == sum(
            t.numel() * t.element_size() for t in tensors
        )
        assert batch.allocated_bytes <= 162

    def forbidden(*args, **kwargs):
        raise AssertionError("Allocation happened before byte admission")

    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        collate_sequence_windows([SequenceWindow(row, 0, 3, 0)], max_bytes=80)


def test_direct_windows_reject_episode_crossing():
    row = sample("a", 6, ends=(3,))
    with pytest.raises(ValueError, match="episode boundary"):
        collate_sequence_windows([SequenceWindow(row, 1, 5, 1)], max_bytes=4096)


def test_partial_empty_and_invalid_sequences():
    empty = sample("empty", 0)
    assert list(iter_sequence_batches([empty], window_steps=2, max_bytes=1024)) == []
    row = sample("a", 3)
    batches = list(
        iter_sequence_batches([row], window_steps=2, batch_size=4, max_bytes=1024)
    )
    assert [w.length for w in batches[0].windows] == [2, 1]
    assert not batches[0].valid_mask[1, 1]
    with pytest.raises(ValueError, match="boolean vectors"):
        SequenceSample("bad", {"x": torch.zeros(2, 3)}, {"supervised": torch.ones(2)})
    with pytest.raises(ValueError, match="time dimension"):
        SequenceSample("bad", {"x": torch.zeros(2, 3), "y": torch.zeros(3)}, {})
