# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Optional temporal batching helpers for application-owned recorded trainers.

Applications choose fields, temporal alignment, episode boundaries and loss
masks. These helpers never load payloads or interpret observations/actions.
"""

from dataclasses import dataclass
from math import prod
from typing import Mapping

import torch


@dataclass(frozen=True)
class SequenceSample:
    """Owned, time-aligned tensors from one record; dimension zero is time."""

    sample_id: str
    fields: Mapping[str, torch.Tensor]
    loss_masks: Mapping[str, torch.Tensor]
    episode_ends: torch.Tensor | None = None

    def __post_init__(self):
        if not self.fields:
            raise ValueError("A sequence requires time-aligned tensor fields")
        lengths = set()
        for value in self.fields.values():
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise ValueError(
                    "Sequence fields must be tensors with a time dimension"
                )
            lengths.add(len(value))
        if len(lengths) != 1:
            raise ValueError("Sequence fields disagree on their time dimension")
        for mask in (*self.loss_masks.values(), self.episode_ends):
            if mask is not None and (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.shape != (self.length,)
            ):
                raise ValueError("Sequence masks must be aligned boolean vectors")

    @property
    def length(self):
        return len(next(iter(self.fields.values())))


@dataclass(frozen=True)
class SequenceWindow:
    sample: SequenceSample
    start: int
    stop: int
    loss_start: int

    @property
    def length(self):
        return self.stop - self.start


@dataclass(frozen=True)
class SequenceBatch:
    fields: Mapping[str, torch.Tensor]
    valid_mask: torch.Tensor
    loss_masks: Mapping[str, torch.Tensor]
    reset_mask: torch.Tensor
    windows: tuple[SequenceWindow, ...]
    allocated_bytes: int


def sequence_windows(sample, *, window_steps, context_steps=0):
    """Partition contributing steps once; repeat context only within an episode.

    A terminal step belongs to the ending episode. ``window_steps`` counts new
    steps, with up to ``context_steps`` preceding steps excluded from every loss.
    Every emitted window starts a fresh model state; context is its warmup.
    """
    if type(window_steps) is not int or window_steps < 1:
        raise ValueError("window_steps must be a positive integer")
    if type(context_steps) is not int or context_steps < 0:
        raise ValueError("context_steps must be a nonnegative integer")
    ends = (
        []
        if sample.episode_ends is None
        else (sample.episode_ends.nonzero().flatten() + 1).tolist()
    )
    if not ends or ends[-1] != sample.length:
        ends.append(sample.length)
    episode_start = 0
    for episode_stop in ends:
        for loss_start in range(episode_start, episode_stop, window_steps):
            yield SequenceWindow(
                sample,
                max(episode_start, loss_start - context_steps),
                min(loss_start + window_steps, episode_stop),
                loss_start,
            )
        episode_start = episode_stop


def _batch_bytes(windows):
    first = windows[0].sample
    for window in windows:
        sample = window.sample
        if (
            sample.fields.keys() != first.fields.keys()
            or sample.loss_masks.keys() != first.loss_masks.keys()
        ):
            raise ValueError("Sequence batch fields and objective names must agree")
        for name, value in sample.fields.items():
            other = first.fields[name]
            if value.shape[1:] != other.shape[1:] or value.dtype != other.dtype:
                raise ValueError("Sequence batch field shapes and dtypes must agree")
    per_step = sum(
        prod(value.shape[1:]) * value.element_size() for value in first.fields.values()
    )
    # valid, reset, and one boolean mask per objective are also allocated.
    per_step += 2 + len(first.loss_masks)
    return len(windows) * max(window.length for window in windows) * per_step


def collate_sequence_windows(windows, *, max_bytes, device=None):
    """Pad one batch, checking its exact tensor allocation before any copy.

    ``max_bytes`` bounds this batch's tensors, excluding source-owned payloads,
    model state and application workspace. Padding values are zero; valid and
    objective masks define which positions the application may use.
    """
    windows = tuple(windows)
    if not windows:
        raise ValueError("Cannot collate an empty sequence batch")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    for window in windows:
        if (
            not 0
            <= window.start
            <= window.loss_start
            < window.stop
            <= window.sample.length
        ):
            raise ValueError("Invalid sequence window bounds")
        ends = window.sample.episode_ends
        if ends is not None and ends[window.start : window.stop - 1].any():
            raise ValueError("A sequence window cannot cross an episode boundary")
    size = _batch_bytes(windows)
    if size > max_bytes:
        raise ValueError(
            f"Prepared sequence batch requires {size} bytes, exceeding its {max_bytes} byte budget"
        )
    first = windows[0].sample
    device = next(iter(first.fields.values())).device if device is None else device
    shape = (len(windows), max(window.length for window in windows))
    fields = {
        name: torch.zeros((*shape, *value.shape[1:]), dtype=value.dtype, device=device)
        for name, value in first.fields.items()
    }
    valid = torch.zeros(shape, dtype=torch.bool, device=device)
    reset = torch.zeros_like(valid)
    masks = {name: torch.zeros_like(valid) for name in first.loss_masks}
    for i, window in enumerate(windows):
        n = window.length
        valid[i, :n] = True
        reset[i, 0] = True
        for name, value in window.sample.fields.items():
            fields[name][i, :n].copy_(value[window.start : window.stop])
        offset = window.loss_start - window.start
        for name, mask in window.sample.loss_masks.items():
            masks[name][i, offset:n].copy_(mask[window.loss_start : window.stop])
    return SequenceBatch(fields, valid, masks, reset, windows, size)


def iter_sequence_batches(
    samples, *, window_steps, context_steps=0, batch_size=1, max_bytes, device=None
):
    """Stream bounded batches without materializing all trajectory windows.

    A batch is flushed when either its window count or padded byte budget would
    be exceeded. An individual oversized window fails before allocation.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    pending = []
    for sample in samples:
        for window in sequence_windows(
            sample, window_steps=window_steps, context_steps=context_steps
        ):
            if _batch_bytes([window]) > max_bytes:
                raise ValueError("A sequence window exceeds its prepared byte budget")
            if pending and (
                len(pending) == batch_size
                or _batch_bytes([*pending, window]) > max_bytes
            ):
                yield collate_sequence_windows(
                    pending, max_bytes=max_bytes, device=device
                )
                pending = []
            pending.append(window)
    if pending:
        yield collate_sequence_windows(pending, max_bytes=max_bytes, device=device)
