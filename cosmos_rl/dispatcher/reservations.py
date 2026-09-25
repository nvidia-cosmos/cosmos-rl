# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Controller-owned, producer-held prompt slots, not accepted payload storage.

The controller event loop owns this ledger under its lifecycle lock. Once a
report's admission/accounting succeeds, settlement transfers those slots away
from the producer. Retirement releases only the remainder. Dataset indices and
actual generation versions are deliberately not reservation identities.
"""

from collections import Counter
from dataclasses import dataclass, field
from uuid import uuid4
from cosmos_rl.dispatcher.data.schema import TrainingCompletionIdentity


@dataclass(frozen=True)
class ReservationPlan:
    owner: object = field(repr=False)
    revision: int
    slots: tuple[tuple[str, tuple[int, ...]], ...]
    requested_version_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ReportReservations:
    ownership: ReservationPlan
    rejected_version_counts: tuple[tuple[int, int], ...]
    requested_versions: dict[tuple[str, int], int]


class ProducerReservations:
    def __init__(self):
        self.incarnation = uuid4().hex
        self._next_work = 0
        self._revision = 0
        self._owner = object()
        self._work: dict[str, tuple[int, set[int]]] = {}
        self.retired = False

    @property
    def outstanding(self) -> int:
        return sum(len(slots) for _, slots in self._work.values())

    def issue(self, requested_weight_version: int, count: int) -> str:
        if self.retired:
            raise ValueError("Cannot reserve work for a retired producer")
        if type(count) is not int or count < 1:
            raise ValueError("Reservation count must be a positive integer")
        if type(requested_weight_version) is not int or requested_weight_version < 0:
            raise ValueError("Reservation requires its original requested version")
        work_id = f"{self.incarnation}:{self._next_work}"
        self._next_work += 1
        self._work[work_id] = (requested_weight_version, set(range(count)))
        self._revision += 1
        return work_id

    def prepare(self, outcomes: list[tuple[str, list[int]]]) -> ReservationPlan:
        """Validate an entire report before any admission/accounting mutation."""
        if self.retired:
            raise ValueError("Cannot settle work for a retired producer")
        selected: dict[str, set[int]] = {}
        counts = Counter()
        for work_id, slots in outcomes:
            if work_id not in self._work:
                raise ValueError("Unknown or already settled training work")
            version, pending = self._work[work_id]
            claimed = selected.setdefault(work_id, set())
            if not slots:
                raise ValueError("A training outcome must identify completion slots")
            for slot in slots:
                if type(slot) is not int or slot not in pending or slot in claimed:
                    raise ValueError("Unknown, duplicate or settled completion slot")
                claimed.add(slot)
                counts[version] += 1
        return ReservationPlan(
            self._owner,
            self._revision,
            tuple(
                (work_id, tuple(sorted(slots))) for work_id, slots in selected.items()
            ),
            tuple(sorted(counts.items())),
        )

    def commit(self, plan: ReservationPlan) -> None:
        if (
            self.retired
            or plan.owner is not self._owner
            or plan.revision != self._revision
        ):
            raise RuntimeError("Reservation settlement lost its source/revision")
        for work_id, slots in plan.slots:
            _, pending = self._work[work_id]
            pending.difference_update(slots)
            if not pending:
                del self._work[work_id]
        self._revision += 1

    def prepare_report(self, request, *, application_plan=None, is_dapo=False):
        """Bind report outcomes to owned work before legacy/application mutation.

        Application replay windows decide which identities are new; their
        immutable reservation binding also permits late cleanup after a failure
        without reserving or settling that slot twice.
        """
        delivered = []
        for payload in request.payloads:
            if (
                payload.training_work_id is None
                or payload.training_completion_slots is None
            ):
                raise ValueError(
                    "Training report lacks controller reservation identity"
                )
            if len(payload.training_completion_slots) != len(payload.completions):
                raise ValueError("Training report slots must be completion-aligned")
            delivered.extend(
                TrainingCompletionIdentity(work_id=payload.training_work_id, slot=slot)
                for slot in payload.training_completion_slots
            )
        rejected = request.training_rejections
        claims = delivered + rejected
        if len({(item.work_id, item.slot) for item in claims}) != len(claims):
            raise ValueError("Training report repeats a controller reservation")
        if application_plan is not None:
            identities = request.completion_identities
            if delivered != [identity.reservation for identity in identities]:
                raise ValueError(
                    "Application and payload reservation identities disagree"
                )
            failure_reservations = [
                failure.identity.reservation for failure in request.completion_failures
            ]
            if any(item is None for item in failure_reservations) or sorted(
                (item.work_id, item.slot) for item in failure_reservations
            ) != sorted((item.work_id, item.slot) for item in rejected):
                raise ValueError("Application failure reservation identities disagree")
            _, _, decisions, _, _ = application_plan
            claims = [identity.reservation for identity, _, _ in decisions]
            rejected = [
                identity.reservation
                for identity, _, disposition in decisions
                if disposition.outcome == "rejected"
            ]
        else:
            if request.completion_identities is not None or request.completion_failures:
                raise ValueError(
                    "Identified completions require completion_admission=True"
                )
            expected = 0
            keys = ["discarded_samples"]
            if is_dapo:
                keys += ["filtered_positive", "filtered_negative"]
            for key in keys:
                count = request.metrics.get(key, 0)
                if type(count) is not int or count < 0:
                    raise ValueError("Discard metrics must be nonnegative integers")
                expected += count
            if len(rejected) != expected:
                raise ValueError("Discard metrics and controller reservations disagree")
        plan = self.prepare([(item.work_id, [item.slot]) for item in claims])
        versions = {
            (item.work_id, item.slot): self._work[item.work_id][0] for item in claims
        }
        rejected_counts = Counter(
            versions[(item.work_id, item.slot)] for item in rejected
        )
        return ReportReservations(
            plan, tuple(sorted(rejected_counts.items())), versions
        )

    def retire(self) -> dict[int, int]:
        """Return unreported counts under their original requested versions once."""
        if self.retired:
            return {}
        counts = Counter()
        for version, slots in self._work.values():
            counts[version] += len(slots)
        self._work.clear()
        self.retired = True
        self._revision += 1
        return dict(counts)
