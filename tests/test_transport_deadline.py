# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from cosmos_rl.utils.transport_failure import TransportDeadline, TransportUnusableError


def test_completion_seals_deadline_before_callback():
    expired = []
    deadline = TransportDeadline(60, "test", fatal=expired.append)
    deadline.close()
    deadline._expire()
    assert expired == []


def test_callback_seals_timeout_before_late_completion():
    expired = []
    deadline = TransportDeadline(60, "test", fatal=expired.append)
    deadline._expire()
    with pytest.raises(TransportUnusableError, match="completion after"):
        deadline.close()
    assert len(expired) == 1


def test_budget_does_not_reset_between_phases():
    now = [100.0]
    deadline = TransportDeadline(60, "test", fatal=lambda _: None, clock=lambda: now[0])
    try:
        assert deadline.remaining_ms() == 60000
        now[0] += 10
        assert deadline.remaining_ms() == 50000
    finally:
        deadline.close()


@pytest.mark.parametrize("phase", ["queued", "raw_init", "device_completion"])
def test_independent_deadline_exits_without_caller_progress(phase):
    repo = str(Path(__file__).resolve().parents[1])
    program = (
        "import threading, time\n"
        "from cosmos_rl.utils.transport_failure import TransportDeadline\n"
        "print('DEADLINE_ARMED', time.monotonic(), flush=True)\n"
        f"TransportDeadline(0.1, {phase!r})\n"
        "threading.Event().wait(30)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONPATH": repo},
        capture_output=True,
        text=True,
        # Package imports can be slow on a cold container filesystem. Measure
        # the enforced operation budget separately; never relax its 0.1s timer.
        timeout=120,
    )
    assert result.returncode == 86, result.stderr
    assert phase in result.stderr
    armed = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("DEADLINE_ARMED ")
    )
    assert time.monotonic() - float(armed.split()[1]) < 5


def test_pending_device_event_detects_correlated_peer_failure():
    deadline = TransportDeadline(60, "test")
    try:

        class Event:
            def query(self):
                return False

        def failed():
            raise TransportUnusableError("sender failed")

        with pytest.raises(TransportUnusableError, match="sender failed"):
            deadline.wait_event(Event(), check_peer=failed)
    finally:
        deadline.close()
