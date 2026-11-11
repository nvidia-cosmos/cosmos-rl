# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Persistent sample identities, separate from controller reservations."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class ModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["learner", "actor", "teacher"]
    name: str
    revision: str | None = None
    artifact_digest: str | None = None
    representation_revision: str | None = None
    behavior_version: StrictInt | None = Field(default=None, ge=0)


class SampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1)
    occurrence_id: str = Field(min_length=1)
    group_id: str | None = None
    group_size: StrictInt | None = Field(default=None, ge=1)
    group_slot: StrictInt | None = Field(default=None, ge=0)
    producer: ModelIdentity | None = None
    inputs: Any
    targets: Any = None
    masks: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    behavior_logprobs: Any = None
    sampling: dict[str, Any] = Field(default_factory=dict)
    attempt_id: str | None = None

    def require_policy_gradient(self):
        if self.producer is None or self.behavior_logprobs is None:
            raise ValueError(
                "Recorded RL requires original producer identity and behavior log probabilities"
            )


def output_id(run_id: str, occurrence: int, slot: int) -> str:
    if (
        type(occurrence) is not int
        or occurrence < 0
        or type(slot) is not int
        or slot < 0
    ):
        raise ValueError("Output occurrence and slot must be nonnegative integers")
    # JSON keeps user-provided run IDs unambiguous (including slashes/colons).
    import json

    return json.dumps([run_id, occurrence, slot], separators=(",", ":"))
