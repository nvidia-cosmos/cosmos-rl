# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Local teacher-result ownership, including cancellation of skipped batches."""

import threading
import time

from cosmos_rl.utils.teacher_channel import deadline_after


class TeacherResultInbox:
    def __init__(self):
        self._condition = threading.Condition()
        self._results = {}

    def admit(self, identity):
        with self._condition:
            self._results.setdefault(identity, None)

    def pending(self, identity):
        with self._condition:
            return identity in self._results and self._results[identity] is None

    def complete(self, identity, value):
        with self._condition:
            if identity not in self._results:
                return False  # A skipped/retired update cannot recreate cache entries.
            self._results[identity] = value
            self._condition.notify_all()
            return True

    def wait(self, identities, timeout):
        deadline = deadline_after(timeout)
        with self._condition:
            while any(self.pending(identity) for identity in identities):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return [self._results.get(identity) for identity in identities]

    def retire(self, identities):
        with self._condition:
            for identity in identities:
                self._results.pop(identity, None)
            self._condition.notify_all()
