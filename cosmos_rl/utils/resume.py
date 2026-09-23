# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The controller/worker checkpoint agreement contract."""


class ResumeMetadataMismatch(ValueError):
    """The controller and worker cannot agree on resumable training state."""


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
