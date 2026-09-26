# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A scheduled cancellation is not a completed native drain."""

from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from cosmos_rl.utils.payload_transport.ucxx import operation, ucxx_buffer
from cosmos_rl.utils.transport_failure import TransportUnusableError


@pytest.mark.parametrize(
    "fault", ["missing_cancel", "missing_size", "cancel_error", "size_error", "pending"]
)
def test_unproven_drain_never_reports_success(fault):
    worker = SimpleNamespace(
        cancel_inflight_requests=Mock(return_value=2),
        get_canceling_size=Mock(return_value=0),
    )
    if fault == "missing_cancel":
        del worker.cancel_inflight_requests
    elif fault == "missing_size":
        del worker.get_canceling_size
    elif fault == "cancel_error":
        worker.cancel_inflight_requests.side_effect = RuntimeError("cancel failed")
    elif fault == "size_error":
        worker.get_canceling_size.side_effect = RuntimeError("query failed")
    else:
        worker.get_canceling_size.return_value = 1
    with pytest.raises((RuntimeError, TimeoutError)):
        ucxx_buffer._drain_inflight_requests(worker, timeout_s=0.03)


def test_native_zero_is_required_after_cancellation():
    events = []
    sizes = iter([2, 1, 0])
    worker = SimpleNamespace(
        cancel_inflight_requests=lambda: events.append("cancel"),
        get_canceling_size=lambda: events.append("query") or next(sizes),
    )
    ucxx_buffer._drain_inflight_requests(worker, timeout_s=1)
    assert events == ["cancel", "query", "query", "query"]


@pytest.fixture
def context(monkeypatch):
    worker = SimpleNamespace(
        cancel_inflight_requests=Mock(return_value=0),
        get_canceling_size=Mock(return_value=0),
        stop_progress_thread=Mock(),
    )
    native = SimpleNamespace(
        core=SimpleNamespace(_ctx=SimpleNamespace(worker=worker)),
        init=Mock(),
        reset=Mock(),
    )
    failures = []
    monkeypatch.setattr(ucxx_buffer, "ucxx", native)
    monkeypatch.setattr(ucxx_buffer, "UCXX_AVAILABLE", True)
    monkeypatch.setattr(ucxx_buffer, "_CONTEXT_OWNERS", {})
    monkeypatch.setattr(ucxx_buffer, "_CONTEXT_FAILURE", None)
    monkeypatch.setattr(operation, "_TERMINAL_OPERATIONS", [])
    monkeypatch.setattr(operation, "fail_transport", failures.append)
    return native, worker, failures


def test_global_reset_cannot_cancel_another_live_owner(context):
    native, worker, failures = context
    first, second = object(), object()
    ucxx_buffer._acquire_ucxx_context(first)
    ucxx_buffer._acquire_ucxx_context(second)
    ucxx_buffer.reset_ucxx_context()
    ucxx_buffer._release_ucxx_context(first)
    worker.cancel_inflight_requests.assert_not_called()
    worker.stop_progress_thread.assert_not_called()
    native.reset.assert_not_called()
    assert ucxx_buffer._CONTEXT_OWNERS == {id(second): second}
    ucxx_buffer._release_ucxx_context(second)
    # Managed owners already awaited native retirement, so no speculative
    # global cancellation or unsupported cancellation-count query is needed.
    worker.cancel_inflight_requests.assert_not_called()
    worker.stop_progress_thread.assert_called_once()
    native.reset.assert_called_once()
    assert not failures and not ucxx_buffer._CONTEXT_OWNERS


@pytest.mark.parametrize("stage", ["drain", "stop", "reset"])
def test_uncertain_global_teardown_is_terminal_and_blocks_readmission(context, stage):
    native, worker, failures = context
    target = {
        "drain": worker.cancel_inflight_requests,
        "stop": worker.stop_progress_thread,
        "reset": native.reset,
    }[stage]
    target.side_effect = RuntimeError("injected uncertainty")
    with pytest.raises(TransportUnusableError, match="uncertain"):
        ucxx_buffer.reset_ucxx_context()
    assert failures and operation._TERMINAL_OPERATIONS
    if stage == "drain":
        worker.stop_progress_thread.assert_not_called()
    if stage != "reset":
        native.reset.assert_not_called()
    with pytest.raises(TransportUnusableError):
        ucxx_buffer._acquire_ucxx_context(object())
    native.init.assert_not_called()


def test_last_owner_reset_and_new_admission_are_serialized(context):
    native, _, failures = context
    entered, release, admitted = threading.Event(), threading.Event(), threading.Event()
    first, second = object(), object()
    ucxx_buffer._acquire_ucxx_context(first)
    native.init.reset_mock()

    def reset():
        entered.set()
        assert release.wait(2)

    def admit():
        ucxx_buffer._acquire_ucxx_context(second)
        admitted.set()

    native.reset.side_effect = reset
    closer = threading.Thread(target=ucxx_buffer._release_ucxx_context, args=(first,))
    reader = threading.Thread(target=admit)
    closer.start()
    try:
        assert entered.wait(1)
        reader.start()
        assert not admitted.wait(0.03)
        native.init.assert_not_called()
    finally:
        release.set()
        closer.join(2)
        reader.join(2)
    assert admitted.is_set() and not failures
    native.init.assert_called_once()
    ucxx_buffer._release_ucxx_context(second)
