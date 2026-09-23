# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller metadata and sampler restoration for application checkpoints."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cosmos_rl.policy.config import Config


class ControllerResumeMetadata(BaseModel):
    """Version-one metadata. Counts deliberately use distinct units.

    Training steps are completed Cosmos DataFetch commands, not necessarily
    optimizer updates. Remaining completions include rollout.n_generation.
    Epoch is one-based, matching ControllerDataFetcher. The effective sampler
    owns the entire stream, including its nested sampler when batch sampling.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    checkpoint_path: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    completed_training_steps: StrictInt = Field(ge=0)
    completed_optimizer_updates: StrictInt = Field(ge=0)
    remaining_completions: StrictInt = Field(ge=0)
    epoch: StrictInt = Field(ge=1)
    sampling_owner: Literal["sampler", "batch_sampler"]
    sampler_state: dict[str, Any]

    def to_checkpoint_extra_info(self) -> dict[str, Any]:
        return {
            "step": self.completed_training_steps,
            "optimizer_updates": self.completed_optimizer_updates,
            "remain_samples_num": self.remaining_completions,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_id": self.checkpoint_id,
            "controller_resume_schema_version": self.schema_version,
        }


class ControllerResumeAdapter(Protocol):
    """An instance passed to controller main/setup, never a process-global hook.

    load_metadata is called once only when train.resume is enabled. An explicit
    path must take precedence over discovery; errors must propagate. Return
    None only when auto-discovery found no checkpoint (fresh bootstrap).

    restore_sampler runs after sampler construction and initial set_epoch,
    before DataLoader construction or iterator consumption. Restore the whole
    effective stream without probing/advancing it; for a batch sampler this
    includes any nested sampler. Model/optimizer/RNG loading stays in Trainer.

    Metadata must come from a committed checkpoint. checkpoint_id identifies
    the shared immutable save, not the current process or a reusable directory.
    sampler_state must be a safe replay boundary: fetching a prompt does not
    commit it. remaining_completions includes repeated work after a conservative
    rewind. The application's trainer validates the same save before loading.
    """

    def load_metadata(self, config: Config) -> ControllerResumeMetadata | None: ...

    def restore_sampler(
        self, sampler: Any, metadata: ControllerResumeMetadata
    ) -> None: ...
