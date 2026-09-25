# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Controller-owned, sealed membership for one real training dispatch.

Membership is captured before publication, never inferred from live status
flags. This is in-memory execution accounting, not checkpoint queue recovery.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import hashlib
import json


@dataclass
class TrainingDispatch:
    step: int
    total_steps: int
    participants: frozenset[str]
    rollout_count: int
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_digests: dict[str, str] = field(default_factory=dict)
    settled: bool = False

    def __post_init__(self):
        if not self.participants or self.step < 0 or self.rollout_count < 0:
            raise ValueError("Invalid training dispatch")

    def acknowledge(self, replica: str, step: int, total_steps: int, report: dict):
        """Validate before mutation; return whether this is a new receipt.

        An exact retry is a no-op, including after settlement. Changed repeats
        and receipts from nonparticipants must never mutate counters/statuses.
        """
        if step != self.step or total_steps != self.total_steps:
            raise ValueError("Training ACK does not match the dispatched schedule")
        if replica not in self.participants:
            raise ValueError("Training ACK is not from a dispatched participant")
        digest = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if replica in self.report_digests:
            if self.report_digests[replica] != digest:
                raise ValueError("Training ACK retry changed its report")
            return False
        if self.settled:
            raise ValueError("Training ACK arrived after dispatch settlement")
        self.reports[replica] = deepcopy(report)
        self.report_digests[replica] = digest
        return True

    @property
    def complete(self):
        return self.report_digests.keys() == self.participants

    def settle(self):
        if not self.complete or self.settled:
            raise ValueError("Training dispatch cannot be settled")
        self.settled = True
        reports, self.reports = self.reports, {}
        return list(reports.values())
