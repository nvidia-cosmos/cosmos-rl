# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The controller/worker checkpoint agreement contract."""


class ResumeMetadataMismatch(ValueError):
    """The controller and worker cannot agree on resumable training state."""


class NoCheckpointFound(FileNotFoundError):
    """Discovery found no committed checkpoint, before loading any state.

    Only automatic discovery may treat this as a fresh start. Missing artifacts
    or errors after selecting a checkpoint are NOT this outcome.
    """


def controller_checkpoint_metadata(checkpoint: dict) -> dict:
    """Separate trainer-owned reference state from the shared contract.

    Trainers validate and consume these fields locally during restore. Reference
    tensors can be rank-local and must not travel over the controller JSON API.
    Do not filter unknown fields: application sampling/progress remains subject
    to exact controller/worker agreement.
    """
    local_fields = {
        "grpo_reference_enabled",
        "grpo_reference_state",
        "grpo_reference_reset_step",
        "dpo_reference_policy",
        "dpo_reference_state",
    }
    return {key: value for key, value in checkpoint.items() if key not in local_fields}


def validate_resume_metadata(expected: dict, actual: dict) -> None:
    """Preserve exact agreement, reporting keys without leaking checkpoint data.

    This dictionary is a resume contract, not an arbitrary logging envelope.
    Rank-local RNG state is restored locally and excluded by the checkpoint
    reader. Applications must likewise exclude rank-local/diagnostic fields.
    Unknown contract fields are not silently ignored: they may affect sampling
    or trainer state in an application we cannot interpret.
    """
    if expected != actual:
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        changed = sorted(
            key
            for key in expected.keys() & actual.keys()
            if expected[key] != actual[key]
        )
        raise ResumeMetadataMismatch(
            "Checkpoint resume agreement failed; continuation is unsafe: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
