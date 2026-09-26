# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""One serialized caller's bounded, in-memory mutation receipt.

Callers retain a request until its ACK and advance the sequence only afterwards.
The controller must authenticate the source incarnation before using a receipt.
This does not replay a partially applied mutation or persist across restart.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any


def request_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MutationToken:
    sequence: int
    digest: str


class OrderedReceipt:
    """Retain only the last ACK, rejecting expired and changed requests."""

    def __init__(self):
        self.sequence = -1
        self.digest: str | None = None
        self.response: Any = None
        self.pending: MutationToken | None = None
        self.failed = False

    def begin(self, sequence: int, digest: str) -> tuple[MutationToken | None, Any]:
        if self.failed:
            raise RuntimeError("Mutation receipt is unusable after a failed operation")
        if type(sequence) is not int or sequence < 0 or not digest:
            raise ValueError("Mutation requires a nonnegative sequence and digest")
        if self.pending is not None:
            raise ValueError("Mutation source already has an unacknowledged operation")
        if sequence == self.sequence:
            if digest != self.digest:
                raise ValueError("Mutation retry changed its request")
            return None, deepcopy(self.response)
        if sequence != self.sequence + 1:
            raise ValueError("Mutation sequence is expired or out of order")
        token = MutationToken(sequence, digest)
        self.pending = token
        return token, None

    def commit(self, token: MutationToken, response: Any) -> None:
        if self.failed or token is not self.pending:
            raise RuntimeError("Mutation commit does not own the pending operation")
        # Snapshot can itself fail after an external mutation. Keep the token
        # pending and poison the receipt; do not permit the operation to rerun.
        try:
            snapshot = deepcopy(response)
        except BaseException:
            self.failed = True
            raise
        self.sequence, self.digest, self.response = (
            token.sequence,
            token.digest,
            snapshot,
        )
        self.pending = None

    def fail(self, token: MutationToken) -> None:
        if token is not self.pending:
            raise RuntimeError("Mutation failure does not own the pending operation")
        self.failed = True
