# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Managed enqueue ownership; mocked native calls do not prove GPU completion."""

import threading
import subprocess
import sys
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.utils import pynccl
from cosmos_rl.utils.transport_failure import TransportDeadline


class FatalProbe(BaseException):
    pass


@pytest.fixture
def group_runtime(monkeypatch):
    registry = pynccl._CommunicatorRegistry()
    comm = registry.register(object(), 0, 2)
    monkeypatch.setattr(pynccl, "_COMM_REGISTRY", registry)
    monkeypatch.setattr(pynccl, "_tls", threading.local())
    start, end, fatal = Mock(), Mock(), Mock(side_effect=FatalProbe)
    monkeypatch.setattr(pynccl, "nccl_group_start", start)
    monkeypatch.setattr(pynccl, "nccl_group_end", end)
    monkeypatch.setattr(pynccl, "fail_transport", fatal)
    timers = []

    def deadline(*args):
        timer = TransportDeadline(*args, fatal=fatal)
        timers.append(timer)
        return timer

    monkeypatch.setattr(pynccl, "TransportDeadline", deadline)
    yield SimpleNamespace(
        registry=registry, comm=comm, start=start, end=end, fatal=fatal
    )
    for timer in timers:
        timer._timer.cancel()


def test_healthy_group_releases_claim_after_end(group_runtime):
    rt = group_runtime
    observed = []
    rt.end.side_effect = lambda *a, **kw: observed.append(
        rt.registry.get(rt.comm).group_owner is not None
    )
    with pynccl.nccl_group(rt.comm, timeout_ms=1000):
        assert rt.registry.get(rt.comm).group_owner is not None
    assert observed == [True]
    assert rt.registry.get(rt.comm).group_owner is None
    assert pynccl._tls.group_comm is None
    rt.start.assert_called_once()
    rt.end.assert_called_once()
    rt.fatal.assert_not_called()
    assert (
        0
        < rt.end.call_args.kwargs["timeout_ms"]
        <= rt.start.call_args.kwargs["timeout_ms"]
        <= 1000
    )


@pytest.mark.parametrize("phase", ["start", "body", "end"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_group_failure_retains_claim_and_never_attempts_cleanup(
    group_runtime, phase, error_type
):
    rt = group_runtime
    error = error_type("original group failure")
    if phase != "body":
        getattr(rt, phase).side_effect = error
    with pytest.raises(FatalProbe):
        with pynccl.nccl_group(rt.comm):
            if phase == "body":
                raise error
    assert rt.registry.get(rt.comm).group_owner is not None
    rt.fatal.assert_called_once()
    assert (
        f"{error_type.__name__}: original group failure" in rt.fatal.call_args.args[0]
    )
    assert rt.end.call_count == (phase == "end")


def test_abort_cannot_pop_or_destroy_a_group_owned_handle(group_runtime, monkeypatch):
    rt = group_runtime
    native = Mock()
    monkeypatch.setattr(pynccl, "_nccl", native)
    owner = object()
    rt.registry.claim_group(rt.comm, owner)
    with pytest.raises(FatalProbe):
        pynccl.nccl_abort(rt.comm)
    assert rt.registry.get(rt.comm).group_owner is owner
    native.ncclCommAbort.assert_not_called()
    native.ncclCommDestroy.assert_not_called()
    assert "active NCCL group" in rt.fatal.call_args.args[0]


def test_retired_handle_rejected_before_group_start(group_runtime):
    rt = group_runtime
    rt.registry.pop(rt.comm)
    with pytest.raises(KeyError):
        with pynccl.nccl_group(rt.comm):
            pytest.fail("retired communicator entered group")
    rt.start.assert_not_called()
    rt.fatal.assert_not_called()


def test_concurrent_claim_is_rejected_without_changing_owner(group_runtime):
    rt = group_runtime
    owner = object()
    rt.registry.claim_group(rt.comm, owner)
    with pytest.raises(RuntimeError, match="already has"):
        rt.registry.claim_group(rt.comm, object())
    assert rt.registry.get(rt.comm).group_owner is owner


@pytest.mark.parametrize("other_comm,inline", [(None, True), (42, True), (0, False)])
def test_group_disallows_other_communicators_or_worker_threads(
    group_runtime, other_comm, inline
):
    rt = group_runtime
    raw = Mock()
    with pytest.raises(FatalProbe):
        with pynccl.nccl_group(rt.comm):
            pynccl._submit_nccl(raw, 1000, other_comm, run_inline=inline)
    raw.assert_not_called()
    rt.end.assert_not_called()


def test_other_thread_cannot_submit_into_a_claimed_group(group_runtime):
    rt = group_runtime
    raw = Mock()
    errors = []

    def submit():
        try:
            pynccl._submit_nccl(raw, 1000, rt.comm)
        except RuntimeError as error:
            errors.append(str(error))

    with pynccl.nccl_group(rt.comm):
        thread = threading.Thread(target=submit)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert len(errors) == 1 and "another thread" in errors[0]
    raw.assert_not_called()
    rt.fatal.assert_not_called()


def test_nested_group_does_not_open_another_native_scope(group_runtime):
    rt = group_runtime
    with pytest.raises(FatalProbe):
        with pynccl.nccl_group(rt.comm):
            with pynccl.nccl_group(rt.comm):
                pytest.fail("nested group entered")
    rt.start.assert_called_once()
    rt.end.assert_not_called()


@pytest.mark.parametrize("failure", ["body", "deadline", "abort"])
def test_real_fatal_path_exits_without_native_cleanup(failure):
    # Isolated process: never replace the real non-returning fatal handler.
    code = """
import sys, threading
from types import SimpleNamespace
from cosmos_rl.utils import pynccl
comm = pynccl._COMM_REGISTRY.register(object(), 0, 2)
pynccl.nccl_group_start = lambda *a, **k: print('GROUP_STARTED', flush=True)
pynccl.nccl_group_end = lambda *a, **k: print('UNSAFE_END', flush=True)
pynccl._nccl = SimpleNamespace(
    ncclCommAbort=lambda *a: print('UNSAFE_ABORT', flush=True),
    ncclCommDestroy=lambda *a: print('UNSAFE_DESTROY', flush=True),
)
with pynccl.nccl_group(comm, timeout_ms=100):
    if sys.argv[1] == 'body':
        raise ValueError('injected body failure')
    if sys.argv[1] == 'abort':
        thread = threading.Thread(target=pynccl.nccl_abort, args=(comm,))
        thread.start()
    threading.Event().wait(10)
print('UNSAFE_CONTINUATION', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, failure],
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 86, output
    assert "GROUP_STARTED" in output and "[Transport FATAL]" in output
    assert "UNSAFE_" not in output
    if failure == "body":
        assert "ValueError: injected body failure" in output
    elif failure == "deadline":
        assert "accepted operation deadline expired" in output
    else:
        assert "abort requested inside active NCCL group" in output


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize(
    "filename",
    [
        "policy/worker/rl_worker.py",
        "rollout/worker/rollout_control.py",
        "rollout/worker/weight_sync.py",
        "rollout/trtllm_rollout/trtllm_worker.py",
    ],
)
def test_actual_caller_scope_contains_body_failure(group_runtime, filename, grouped):
    """Execute the actual group region, without importing TRT/model backends."""
    rt = group_runtime
    source = Path(pynccl.__file__).parents[1] / filename
    scopes = [
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "nccl_group"
            for item in node.items
            for child in ast.walk(item.context_expr)
        )
    ]
    assert len(scopes) == 1
    failure = ValueError("injected caller body failure")
    worker = SimpleNamespace(
        global_rank=0,
        recv_weight_shard=Mock(side_effect=failure),
        p2r_collective_manager=SimpleNamespace(send=Mock(side_effect=failure)),
    )
    namespace = dict(
        nccl_group=pynccl.nccl_group,
        nullcontext=nullcontext,
        grouped=grouped,
        group_unpacked=grouped,
        p2r_group_size=int(grouped),
        comm_id=rt.comm,
        comm_idx=rt.comm,
        communicator_index=rt.comm,
        self=worker,
        logger=Mock(),
        base_mesh_key="test-pair",
        grouped_send_ops=[(SimpleNamespace(shape=(1,), dtype="float32"), 0, "weight")],
        transfer_tensors=[object()],
        src_rank=0,
        nccl_broadcast=Mock(side_effect=failure),
        sync_round=[object()],
        command=SimpleNamespace(trainable_only=False, do_weight_sync_check=False),
    )
    with pytest.raises(FatalProbe if grouped else ValueError):
        exec(
            compile(ast.Module(body=scopes, type_ignores=[]), str(source), "exec"),
            namespace,
        )
    rt.end.assert_not_called()
    if grouped:
        rt.start.assert_called_once()
        assert "ValueError: injected caller body failure" in rt.fatal.call_args.args[0]
    else:
        rt.start.assert_not_called()
        rt.fatal.assert_not_called()
