# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small detached application diagnostics, independent of payload ownership."""

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def detached_metadata(value: Any) -> dict:
    """Copy bounded JSON primitives without invoking application serializers.

    In particular, tensors, arrays and transport leases are never traversed,
    stringified or copied to the CPU to produce reporting metadata.
    """

    nodes = 0

    def copy(item, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 8 or nodes > 1024:
            raise ValueError("Completion diagnostics exceed the structure limit")
        kind = type(item)
        if item is None or kind in (bool, int, str):
            return item
        if kind is float and math.isfinite(item):
            return item
        if kind is list:
            return [copy(child, depth + 1) for child in item]
        if kind is dict and all(type(key) is str for key in item):
            return {key: copy(child, depth + 1) for key, child in item.items()}
        raise ValueError("Completion diagnostics must contain finite JSON primitives")

    if type(value) is not dict:
        raise ValueError("Completion diagnostics must be a dictionary")
    result = copy(value)
    if len(json.dumps(result, ensure_ascii=True).encode("utf-8")) > 4096:
        raise ValueError("Completion diagnostics exceed 4096 bytes")
    return result


class CompletionDiagnostics(BaseModel):
    """Original group position survives native selection and rejected reports."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always"
    )
    original_index: int = Field(ge=0, strict=True)
    metadata: dict[str, Any]

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value):
        return detached_metadata(value)
