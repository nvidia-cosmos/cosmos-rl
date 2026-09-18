# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instance-owned admission contracts and bounded completion identity tracking."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cosmos_rl.dispatcher.data.schema import Rollout


class CompletionIdentity(BaseModel):
    """Sequence is allocated before generation, once per source rank incarnation.

    Retries and failure reports must reuse it. Diagnostic reasons are not part
    of identity. Sources may deliver out of order within the configured window.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int = Field(ge=0, strict=True)
    weight_version: int = Field(ge=0, strict=True)


class CompletionDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    outcome: Literal["accepted", "rejected"]
    reason: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_reason(self):
        if (self.outcome == "rejected") != (self.reason is not None):
            raise ValueError("Only a rejected completion must have a reason")
        return self


class CompletionFailure(BaseModel):
    """Producer terminal failure, using the identity allocated before generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    identity: CompletionIdentity
    reason: str = Field(min_length=1, max_length=64)


@dataclass(frozen=True)
class CompletionContext:
    source_replica: str
    source_rank: int
    identity: CompletionIdentity


class CompletionAdmission(Protocol):
    def admit(
        self, context: CompletionContext, rollout: Rollout
    ) -> CompletionDisposition:
        """Decide synchronously, without mutating controller or rollout state.

        Exceptions reject the whole report before accounting changes; callers
        may retry the unchanged report. This is not a fail-open hook.
        """
        ...


@dataclass
class SourceWindow:
    """Bounded replay protection which never re-admits an evicted sequence.

    A sequence below the floor is rejected as expired, including an unseen
    out-of-order completion. The producer must bound reordering accordingly;
    silently treating old reports as new would double-settle accounting.
    """

    capacity: int = 4096
    highest: int = -1
    versions: dict[int, int] = field(default_factory=dict)
    failed_without_payload: set[int] = field(default_factory=set)

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError("Completion identity window must be positive")

    def unseen(self, identities: list[CompletionIdentity]) -> list[bool]:
        if len(identities) > self.capacity:
            raise ValueError("Completion report exceeds identity window")
        sequences = [identity.sequence for identity in identities]
        if len(set(sequences)) != len(sequences):
            raise ValueError("Completion report contains duplicate identities")
        floor = max(0, self.highest - self.capacity + 1)
        result = []
        for identity in identities:
            if identity.sequence < floor:
                raise ValueError("Completion identity expired; do not resubmit as new")
            previous = self.versions.get(identity.sequence)
            if previous is not None and previous != identity.weight_version:
                raise ValueError("Completion identity changed its originating version")
            result.append(previous is None)
        # A single report must not expire one of its own identities on commit.
        highest = max([self.highest, *sequences])
        if sequences and min(sequences) < highest - self.capacity + 1:
            raise ValueError("Completion report spans more than the identity window")
        return result

    def commit(self, identities: list[CompletionIdentity]) -> None:
        self.unseen(identities)
        for identity in identities:
            self.versions[identity.sequence] = identity.weight_version
            self.highest = max(self.highest, identity.sequence)
        floor = self.highest - self.capacity + 1
        self.versions = {
            sequence: version
            for sequence, version in self.versions.items()
            if sequence >= floor
        }
        self.failed_without_payload.intersection_update(self.versions)
