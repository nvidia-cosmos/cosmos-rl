# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative replay boundaries for application-owned sampling.

No rollout payloads are stored. Callers serialize snapshots with the trainer
checkpoint, and settle a completion only after its update has completed (or it
has been terminally discarded). Operations must share the application's sampling
and checkpoint synchronization boundary.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from typing import Any


class SamplingBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    epoch: StrictInt = Field(ge=1)
    remaining_completions: StrictInt = Field(ge=0)
    sampler_state: dict[str, Any]


@dataclass
class _Issued:
    after: SamplingBoundary
    completions: int
    settled: set[int] = field(default_factory=set)


class SamplingReplayLedger:
    """Advance a replay cursor only over a contiguous settled prefix.

    `after` is the sampler state *after* issuance, not a training counter. Its
    remaining count covers the entire sampling suffix from that position.
    Out-of-order settlement never skips an earlier outstanding request; later
    settled prompts may consequently be repeated on resume. Construct a fresh
    ledger from the saved boundary on restart: old execution tokens are rejected.
    This helper does not acknowledge training or intercept a sampler for callers.
    """

    def __init__(self, boundary: SamplingBoundary):
        self._boundary = boundary.model_copy(deep=True)
        self._issued: OrderedDict[str, _Issued] = OrderedDict()
        self._execution = uuid4().hex
        self._sequence = 0

    def issue(self, after: SamplingBoundary, completions: int = 1) -> str:
        if type(completions) is not int or completions < 1:
            raise ValueError("completions must be a positive integer")
        before = (
            next(reversed(self._issued.values())).after
            if self._issued
            else self._boundary
        )
        if before.remaining_completions - after.remaining_completions != completions:
            raise ValueError(
                "sampling boundary must account for every issued completion"
            )
        if after.epoch < before.epoch:
            raise ValueError("sampling epoch must not move backwards")
        token = f"{self._execution}:{self._sequence}"
        self._sequence += 1
        self._issued[token] = _Issued(after.model_copy(deep=True), completions)
        return token

    def settle(self, token: str, completion: int = 0) -> bool:
        entry = self._issued.get(token)
        if entry is None:
            # Includes old-run tokens and retries for an already retired prefix.
            return False
        if type(completion) is not int or not 0 <= completion < entry.completions:
            raise ValueError("completion index is outside this issued request")
        if completion in entry.settled:
            return False
        entry.settled.add(completion)
        while self._issued:
            first = next(iter(self._issued.values()))
            if len(first.settled) != first.completions:
                break
            self._boundary = first.after
            self._issued.popitem(last=False)
        return True

    def snapshot(self) -> SamplingBoundary:
        """Copy the safe boundary; never read or advance the live sampler."""
        return deepcopy(self._boundary)
