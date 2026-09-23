# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared step permission for workers which own their training loop."""

import asyncio


class StepBoundary:
    """One controller-event-loop barrier, with an idempotent previous reply.

    A stop never revokes permission already granted for an update. All replicas
    rendezvous again after that update/active validation, before any next update.
    Missing participants cannot produce a successful stop.
    """

    def __init__(self):
        self.step = None
        self.participants = set()
        self.arrived = set()
        self.future = None
        self.previous = None
        self.stopped_step = None
        self.completed = set()

    async def arrive(self, replica, step, participants, stop_reason):
        if type(step) is not int or step < 0 or replica not in participants:
            raise ValueError("Invalid training boundary participant or step")
        if self.stopped_step is not None and step != self.stopped_step:
            raise ValueError("Training cannot advance past the stop boundary")
        if self.previous is not None and step == self.previous[0]:
            if replica not in self.previous[2]:
                raise ValueError("Participant was not in the previous boundary")
            return self.previous[1]
        if self.future is None:
            self.step = step
            self.participants = set(participants)
            self.future = asyncio.get_running_loop().create_future()
        if self.future.done() and step == self.step + 1:
            self.previous = (self.step, self.future.result(), self.participants)
            self.step = step
            self.arrived = set()
            self.future = asyncio.get_running_loop().create_future()
        if step != self.step or set(participants) != self.participants:
            error = ValueError("Training boundary step or membership disagreement")
            if not self.future.done():
                self.future.set_exception(error)
            raise error
        self.arrived.add(replica)
        if self.arrived == self.participants and not self.future.done():
            reason = stop_reason()
            self.stopped_step = step if reason is not None else None
            self.future.set_result(
                {"stop": reason is not None, "reason": reason, "step": step}
            )
        return await asyncio.shield(self.future)

    def complete(self, replica, step):
        if (
            type(step) is not int
            or step < 0
            or self.future is None
            or not self.future.done()
        ):
            raise ValueError("No agreed training boundary for completion")
        self.future.result()
        if self.stopped_step is None:
            # Natural completion can race a stop request during final saving.
            # Require every participant to confirm the same completed step.
            if (
                self.step is None
                or replica not in self.participants
                or type(step) is not int
                or step not in (self.step, self.step + 1)
            ):
                raise ValueError("Invalid final checkpoint acknowledgement")
            self.stopped_step = step
        if self.stopped_step != step or replica not in self.participants:
            raise ValueError("Completion does not match the agreed stop boundary")
        self.completed.add(replica)
        return self.completed == self.participants
