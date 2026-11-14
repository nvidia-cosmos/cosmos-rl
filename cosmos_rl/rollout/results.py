# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Borrowed generation results for an application's synchronous result sink."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CompletionOutcome:
    slot: int
    status: Literal["generated", "generation_failed"]
    completion: Any = None
    score: float | None = None
    quality_trainable: bool | None = None
    quality_reason: str | None = None
    failure_reason: str | None = None
    completed_conversation: Any = None
    token_ids: Any = None
    logprobs: Any = None
    scorer_metrics: Any = None


@dataclass(frozen=True)
class RolloutResultEvent:
    """One reserved prompt; nested values are borrowed until callback return.

    Copy/persist the data synchronously before returning. The frozen envelope
    does not make tensor storage or nested application objects immutable.
    """

    execution_id: str
    replica_name: str
    global_rank: int
    work_id: str
    prompt_idx: int
    model_name_or_path: str
    model_revision: str | None
    weight_version: int
    prompt: Any
    conversation: Any
    outcomes: tuple[CompletionOutcome, ...]
    extra_info: Any = None
