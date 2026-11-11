# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Persistent sample identities, separate from controller reservations."""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    model_validator,
)
from cosmos_rl.dispatcher.data.schema import Rollout


# Storage v2 deliberately excludes stream markers, prompt indices, transient
# teacher handles, runtime telemetry and the duplicate live producer/version.
DURABLE_ROLLOUT_FIELDS = (
    "prompt",
    "conversation",
    "completion",
    "completed_conversation",
    "completion_token_ids",
    "completion_logprobs",
    "prompt_token_ids",
    "prompt_logprobs",
    "teacher_logprobs",
    "reward",
    "advantage",
    "filter_reward",
    "n_ignore_prefix_tokens",
    "extra_info",
)


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

    schema_version: Literal[2] = 2
    sample_id: str = Field(min_length=1)
    occurrence_id: str = Field(min_length=1)
    group_id: str | None = None
    group_size: StrictInt | None = Field(default=None, ge=1)
    group_slot: StrictInt | None = Field(default=None, ge=0)
    producer: ModelIdentity | None = None
    rollout: Rollout
    masks: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    sampling: dict[str, Any] = Field(default_factory=dict)
    attempt_id: str | None = None

    @model_validator(mode="after")
    def durable_content(self):
        value = self.rollout
        if value.is_end:
            raise ValueError("A stream end marker is not a durable sample")
        if value.teacher_result_uuid:
            raise ValueError("Resolve transient teacher targets before durable storage")
        if value.producer_model is not None:
            identity = ModelIdentity.model_validate(value.producer_model)
            if self.producer is not None and identity != self.producer:
                raise ValueError("Rollout and durable producer identities differ")
            self.producer = identity
        # Keep legacy live defaults unchanged. The durable boundary preserves
        # absence, and shallow-copies only the envelope, never tensor storage.
        updates = {
            name: None
            for name in ("reward", "advantage", "filter_reward")
            if name not in value.model_fields_set
        }
        updates["weight_version"] = (
            self.producer.behavior_version if self.producer else None
        )
        self.rollout = value.model_copy(update=updates)
        return self

    @field_serializer("rollout")
    def serialize_rollout(self, value):
        # Rollout content is mutable. Recheck on publication, including values
        # returned by application model_copy adapters after construction.
        if value.is_end or value.teacher_result_uuid:
            raise ValueError(
                "Resolve stream markers and transient teacher handles before durable storage"
            )
        if (
            value.producer_model is not None
            and ModelIdentity.model_validate(value.producer_model) != self.producer
        ):
            raise ValueError("Rollout and durable producer identities differ")
        return {name: getattr(value, name) for name in DURABLE_ROLLOUT_FIELDS}

    def require_policy_gradient(self):
        if self.producer is None or self.rollout.completion_logprobs is None:
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
