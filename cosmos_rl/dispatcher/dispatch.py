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
    preparation_required: bool = False
    preparation_reports: dict[str, bool] = field(default_factory=dict)
    cancelled: bool = False

    def prepare(self, replica: str, step: int, total_steps: int, ready: bool):
        """Seal permission to execute only after every colocated replica prepares.

        False means terminal input exhaustion, not a transient empty queue.
        Reports are immutable so retries cannot revoke execution permission.
        """
        if not self.preparation_required:
            raise ValueError("Dispatch does not use colocated preparation")
        if step != self.step or total_steps != self.total_steps:
            raise ValueError("Preparation does not match the dispatched schedule")
        if replica not in self.participants or type(ready) is not bool:
            raise ValueError("Invalid preparation participant or readiness")
        if replica in self.preparation_reports:
            if self.preparation_reports[replica] != ready:
                raise ValueError("Preparation retry changed readiness")
        elif self.settled or self.reports:
            raise ValueError("Preparation arrived after training started")
        else:
            self.preparation_reports[replica] = ready
        return self.preparation_decision

    @property
    def preparation_decision(self):
        if self.preparation_reports.keys() != self.participants:
            return "wait"
        return "train" if all(self.preparation_reports.values()) else "stop"

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
        if self.preparation_required and self.preparation_decision != "train":
            raise ValueError("Training ACK arrived without preparation agreement")
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
