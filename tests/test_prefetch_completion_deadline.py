# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The absolute completion deadline cannot depend on timer scheduling."""

import threading
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.utils.payload_transport import prefetch_mixin as module
from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBudget,
    ReceivedBatch,
)
from cosmos_rl.utils.transport_failure import TransportUnusableError


class PausedTimer:
    def __init__(self, interval, function, args=()):
        self.interval = interval
        self.function = function
        self.args = args
        self.cancelled = False

    def start(self):
        # Deterministic starvation of the independent timer callback.
        pass

    def cancel(self):
        self.cancelled = True


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("strategy", [False, True])
@pytest.mark.parametrize("late", [False, True])
def test_completion_seals_absolute_deadline_before_publishing(
    monkeypatch, prepared, strategy, late
):
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module.threading, "Timer", PausedTimer)
    fatal = []
    monkeypatch.setattr(module, "fail_transport", fatal.append)
    entered, release = threading.Event(), threading.Event()
    payload = {"ref": object()}

    class Packer(module.PrefetchDataPackerMixin):
        def _filter_prefetch_tasks(self, rollouts):
            return [(0, "ref")]

        def _fetch_batch(self, tasks):
            entered.set()
            assert release.wait(2)
            return payload

    packer = Packer()
    if strategy:
        packer._transport_strategy = SimpleNamespace(
            before_join=lambda: None, on_prefetch_complete=lambda *a: None
        )
    packer._setup_prefetch(prefetch_timeout=5)
    try:
        if prepared:
            future = packer.start_prepared_prefetch(["ref"], lambda: payload)
        else:
            packer.start_prefetch(["ref"])
        assert entered.wait(1)
        clock[0] = 106.0 if late else 101.0
        release.set()
        if prepared:
            if late:
                with pytest.raises(TimeoutError, match="exceeded"):
                    future.result(timeout=1)
            else:
                assert future.result(timeout=1) is payload
            assert future._prefetch_handoff.wait(1)
        else:
            item = packer._prefetch_result_queue.get(timeout=1)
            packer._prefetch_result_queue.put(item)
            # Collection latency after on-time completion is not active work.
            clock[0] = 200.0
            if late:
                with pytest.raises(TimeoutError, match="exceeded"):
                    packer.wait_prefetch()
            else:
                packer.wait_prefetch()
                assert packer._prefetch_cache is payload
        assert bool(packer._prefetch_failure) is late
        assert len(fatal) == int(strategy and late)
        if late:
            assert not packer._prefetch_timers
            assert any(owner is payload for owner in packer._prefetch_terminal_owners)
            with pytest.raises(TimeoutError):
                packer.start_prefetch(["ref"])
    finally:
        release.set()
        packer.shutdown_prefetch(join_timeout=2)


@pytest.mark.parametrize("prepared", [False, True])
def test_completion_respects_proven_consumer_backpressure_extension(
    monkeypatch, prepared
):
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module.threading, "Timer", PausedTimer)
    fatal = []
    monkeypatch.setattr(module, "fail_transport", fatal.append)
    entered, release = threading.Event(), threading.Event()

    class Packer(module.PrefetchDataPackerMixin):
        def _filter_prefetch_tasks(self, rollouts):
            return [(0, "ref")]

        def _fetch_batch(self, tasks):
            entered.set()
            assert release.wait(2)
            return {"ref": "ready"}

    packer = Packer()
    packer._transport_strategy = SimpleNamespace(
        _receive_budget=SimpleNamespace(
            watchdog_delay=lambda deadline, timeout: max(deadline, 104.0 + timeout)
            - clock[0]
        ),
        before_join=lambda: None,
        on_prefetch_complete=lambda *a: None,
    )
    packer._setup_prefetch(prefetch_timeout=5)
    try:
        if prepared:
            future = packer.start_prepared_prefetch(["ref"], lambda: "ready")
        else:
            packer.start_prefetch(["ref"])
        assert entered.wait(1)
        clock[0] = 106.0
        release.set()
        if prepared:
            assert future.result(timeout=1) == "ready"
            assert future._prefetch_handoff.wait(1)
        else:
            item = packer._prefetch_result_queue.get(timeout=1)
            packer._prefetch_result_queue.put(item)
            packer.wait_prefetch()
        assert not fatal and packer._prefetch_failure is None
    finally:
        release.set()
        packer.shutdown_prefetch(join_timeout=2)


@pytest.mark.parametrize("prepared", [False, True])
def test_terminal_late_lease_is_not_released_by_shutdown(monkeypatch, prepared):
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module.threading, "Timer", PausedTimer)
    fatal = []
    monkeypatch.setattr(module, "fail_transport", fatal.append)
    budget = ReceiveBudget(8, 1)
    budget.reserve(8)
    budget.attribute(decoded=8)
    lease = ReceivedBatch({"ref": {"x": torch.ones(2)}}, budget, 8)

    class Packer(module.PrefetchDataPackerMixin):
        def _filter_prefetch_tasks(self, rollouts):
            return [(0, "ref")]

        def _fetch_batch(self, tasks):
            clock[0] = 106.0
            return lease

    packer = Packer()
    cleanup = []
    packer._transport_strategy = SimpleNamespace(
        before_join=lambda: cleanup.append(True)
    )
    packer._setup_prefetch(prefetch_timeout=5)
    try:
        if prepared:
            future = packer.start_prepared_prefetch(["ref"], lambda: lease)
            with pytest.raises(TimeoutError):
                future.result(timeout=1)
            assert future._prefetch_handoff.wait(1)
            with pytest.raises(TimeoutError):
                packer.release_prepared_prefetch(future)
            assert packer._prepared_prefetch_lease is lease
        else:
            packer.start_prefetch(["ref"])
            packer._prefetch_result_queue.get(timeout=1)
        packer.shutdown_prefetch(join_timeout=2)
        assert len(fatal) == 1 and not lease.released
        assert budget.used == budget.decoded == 8
        assert any(owner is lease for owner in packer._prefetch_terminal_owners)
        with pytest.raises(RuntimeError, match="teardown failed") as failure:
            packer.close_transport(timeout=1)
        assert isinstance(failure.value.__cause__, TransportUnusableError)
        assert not cleanup
    finally:
        packer.shutdown_prefetch(join_timeout=2)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("case", ["healthy", "late"])
def test_starved_timer_completion_has_real_process_outcome(prepared, case):
    command = [
        sys.executable,
        str(Path(__file__).with_name("prefetch_completion_canary.py")),
        "--worker",
        "--cpu",
        "--case",
        case,
    ]
    if prepared:
        command.append("--prepared")
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == (86 if case == "late" else 0), result.stderr
    assert "UNSAFE_LATE_RESULT" not in result.stdout
    if case == "late":
        assert "[Transport FATAL] prefetch batch" in result.stderr
    else:
        assert "PREFETCH_HEALTHY_PASS" in result.stdout
