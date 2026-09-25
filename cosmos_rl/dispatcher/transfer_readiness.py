# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, command-scoped P2R readiness. No native calls or blocking waits."""

import math
import threading
import time

from pydantic import BaseModel, Field


class TransferReadyRequest(BaseModel):
    operation_id: str = Field(min_length=1, max_length=128)
    src: str = Field(min_length=1, max_length=256)
    dst: str = Field(min_length=1, max_length=256)
    src_size: int = Field(ge=1, le=65536)
    dst_size: int = Field(ge=1, le=65536)
    side: str
    rank: int = Field(ge=0)
    expires_at: float
    needs_build: bool
    uid: list[int] | None = None
    error: str | None = Field(default=None, max_length=2048)


class TransferReadiness:
    """Retries register one participant, not another arrival.

    Expiry is sealed in the published command, so a late request cannot recreate
    a retired operation. Capacity exhaustion rejects new work rather than
    evicting an active agreement. Only source rank zero supplies the mesh UID.
    """

    def __init__(self, capacity=4096, clock=time.time):
        self._operations = {}
        self._lock = threading.Lock()
        self._capacity = capacity
        self._clock = clock

    def arrive(self, request: TransferReadyRequest):
        now = self._clock()
        deadline = request.expires_at
        if not math.isfinite(deadline) or deadline <= now:
            return {"state": "failed", "error": "P2R readiness deadline expired"}
        size = {"source": request.src_size, "receiver": request.dst_size}.get(
            request.side, 0
        )
        if request.rank >= size:
            raise ValueError("Invalid P2R readiness participant")
        identity = request.side, request.rank
        if request.uid is not None and identity != ("source", 0):
            raise ValueError("Only source rank zero may publish a P2R UID")
        signature = (
            request.src,
            request.dst,
            request.src_size,
            request.dst_size,
            deadline,
        )
        arrival = request.needs_build, tuple(request.uid or ())
        with self._lock:
            for key in list(self._operations):
                if self._operations[key]["deadline"] <= now:
                    del self._operations[key]
            op = self._operations.get(request.operation_id)
            if op is None:
                if len(self._operations) >= self._capacity:
                    return {
                        "state": "failed",
                        "error": "P2R readiness capacity exhausted",
                    }
                op = {
                    "signature": signature,
                    "deadline": deadline,
                    "arrivals": {},
                    "error": None,
                }
                self._operations[request.operation_id] = op
            if signature != op["signature"]:
                op["error"] = "P2R command metadata mismatch"
            previous = op["arrivals"].get(identity)
            if previous is not None and previous != arrival:
                op["error"] = "P2R participant changed its readiness declaration"
            if request.error:
                op["error"] = request.error
            if op["error"]:
                return {"state": "failed", "error": op["error"]}
            op["arrivals"][identity] = arrival
            if len(op["arrivals"]) != request.src_size + request.dst_size:
                return {"state": "waiting"}
            build = any(value[0] for value in op["arrivals"].values())
            uid = op["arrivals"][("source", 0)][1]
            if build and not uid:
                op["error"] = "P2R build requires a source UID"
                return {"state": "failed", "error": op["error"]}
            return {"state": "ready", "build": build, "uid": list(uid)}
