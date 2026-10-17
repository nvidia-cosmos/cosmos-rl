# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation ownership and retry accounting, serialized by the controller.

One round owns actual sampler work, not dataset indices or a nominal length.
Participants are sealed before work is issued. A single serialized producer per
replica fetches prompts, while each reporting rank has its own report sequence.
The caller must fence requests with ``round_id`` and retain bounded terminal
receipts across round transitions so a lost final POST response can be retried.
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from cosmos_rl.dispatcher.data.schema import RLPayload


def validation_report_digest(payloads: list[RLPayload], is_end: bool) -> str:
    return hashlib.sha256(
        json.dumps(
            [is_end, [payload.model_dump(mode="json") for payload in payloads]],
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ValidationBatch:
    payloads: tuple[RLPayload, ...]
    is_end: bool


class ValidationRound:
    def __init__(self, step: int, reporters: set[tuple[str, int]], generations: int):
        if step < 0 or generations < 1 or not reporters:
            raise ValueError("Validation requires a step, generations and reporters")
        self.step = step
        self.round_id = uuid4().hex
        self.reporters = frozenset(reporters)
        self.replicas = frozenset(replica for replica, _ in reporters)
        self.generations = generations
        self._next_work_id = 0
        self._pending: dict[int, tuple[str, int]] = {}
        self._fetch_receipts: dict[str, tuple[int, tuple, ValidationBatch]] = {}
        self._report_receipts: dict[tuple[str, int], tuple[int, str]] = {}
        self._exhausted: set[str] = set()
        self._ended: set[tuple[str, int]] = set()
        self._fetch_failed = False
        self.reported_prompts = 0

    @property
    def complete(self) -> bool:
        return self._ended == self.reporters and not self._pending

    def _check_round(self, round_id: str) -> None:
        if round_id != self.round_id:
            raise ValueError("Stale validation round")
        if self._fetch_failed:
            raise RuntimeError("Validation fetch failed after sampler admission")

    def completed_receipts(self):
        if not self.complete:
            raise ValueError("Cannot archive an unfinished validation round")
        return dict(self._report_receipts)

    def fetch(
        self,
        round_id: str,
        replica: str,
        sequence: int,
        request: tuple,
        get_payloads: Callable[[], tuple[list[RLPayload], bool]],
    ) -> ValidationBatch:
        """Retry the last identical fetch without advancing the sampler again.

        ``request`` is the immutable (batch size, rank-in-mesh) request signature.
        The retained response is isolated from later reward/serialization edits.
        Only one response per replica is retained, not an entire dataset copy.
        """
        self._check_round(round_id)
        if replica not in self.replicas or self.complete:
            raise ValueError("Validation fetch outside the sealed participant set")
        previous = self._fetch_receipts.get(replica)
        if previous is not None and sequence == previous[0]:
            if request != previous[1]:
                raise ValueError("Validation fetch retry changed its request")
            return copy.deepcopy(previous[2])
        expected = 0 if previous is None else previous[0] + 1
        if sequence != expected or replica in self._exhausted:
            raise ValueError("Out-of-order or exhausted validation fetch")
        try:
            payloads, is_end = get_payloads()
            payloads = copy.deepcopy(payloads)
        except Exception:
            # The sampler may already have advanced. Repeating the callback
            # would silently skip work rather than replay the failed request.
            self._fetch_failed = True
            raise
        for payload in payloads:
            payload.validation_work_id = self._next_work_id
            self._pending[self._next_work_id] = (replica, payload.prompt_idx)
            self._next_work_id += 1
        result = ValidationBatch(tuple(payloads), is_end)
        self._fetch_receipts[replica] = (sequence, request, result)
        if is_end:
            self._exhausted.add(replica)
        return copy.deepcopy(result)

    def report(
        self,
        round_id: str,
        reporter: tuple[str, int],
        sequence: int,
        payloads: list[RLPayload],
        is_end: bool,
    ) -> bool:
        """Validate an entire report before settling any work; false is a retry.

        Each reporter serializes POSTs, retaining its last request until ACK.
        Old/changed receipts are rejected, never reinterpreted as new work.
        Every reporter sends a final empty report, including empty assignments.
        """
        self._check_round(round_id)
        if reporter not in self.reporters:
            raise ValueError("Unknown validation reporter")
        digest = validation_report_digest(payloads, is_end)
        previous = self._report_receipts.get(reporter)
        if previous is not None and sequence == previous[0]:
            if digest != previous[1]:
                raise ValueError("Validation report retry changed its result")
            return False
        expected = 0 if previous is None else previous[0] + 1
        if sequence != expected or reporter in self._ended:
            raise ValueError("Out-of-order or completed validation reporter")
        if not payloads and not is_end:
            raise ValueError("Empty validation report must be terminal")
        identities = [payload.validation_work_id for payload in payloads]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate work in validation report")
        for payload in payloads:
            if self._pending.get(payload.validation_work_id) != (
                reporter[0],
                payload.prompt_idx,
            ):
                raise ValueError("Validation result does not own pending work")
            if payload.rewards is None or len(payload.rewards) != self.generations:
                raise ValueError("Validation result has incomplete generations")
        if is_end:
            if reporter[0] not in self._exhausted:
                raise ValueError("Validation reporter ended before fetch exhaustion")
            if self._ended | {reporter} == self.reporters and len(identities) != len(
                self._pending
            ):
                raise ValueError("Validation ended with outstanding issued work")
        for work_id in identities:
            del self._pending[work_id]
        self.reported_prompts += len(payloads)
        self._report_receipts[reporter] = (sequence, digest)
        if is_end:
            self._ended.add(reporter)
        if self.complete:
            self._fetch_receipts.clear()
        return True
