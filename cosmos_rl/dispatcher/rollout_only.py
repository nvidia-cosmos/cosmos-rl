# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Controller-owned rollout execution state, independent of trainer budgets."""

from dataclasses import dataclass, field
from cosmos_rl.dispatcher.data.schema import RLPayload


@dataclass
class RolloutOnlyState:
    prompts_issued: int = 0
    pending_prompts: list[RLPayload] = field(default_factory=list)
    input_ended: bool = False
    outstanding_completions: int = 0

    def settle(self, count: int) -> None:
        if type(count) is not int or not 0 <= count <= self.outstanding_completions:
            raise ValueError("Rollout-only settlement exceeds outstanding reservations")
        self.outstanding_completions -= count
