# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from functools import partial
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from cosmos_rl.utils import cuda_cache


@pytest.fixture
def policy(monkeypatch):
    policy = cuda_cache._CacheCleanupPolicy()
    monkeypatch.setattr(cuda_cache, "_policy", policy)
    return policy


def test_before_transport_preserves_flush_and_errors(policy, monkeypatch):
    flush = Mock()
    monkeypatch.setattr(cuda_cache.torch.cuda, "empty_cache", flush)
    assert cuda_cache.empty_cuda_cache()
    flush.assert_called_once()
    flush.side_effect = RuntimeError("allocator error")
    with pytest.raises(RuntimeError, match="allocator error"):
        cuda_cache.empty_cuda_cache()


def test_suppression_is_process_wide_sticky_and_does_not_synchronize(
    policy, monkeypatch
):
    flush = Mock(side_effect=AssertionError("unsafe flush"))
    sync = Mock(side_effect=AssertionError("must not synchronize"))
    monkeypatch.setattr(cuda_cache.torch.cuda, "empty_cache", flush)
    monkeypatch.setattr(cuda_cache.torch.cuda, "synchronize", sync)
    cuda_cache.suppress_cuda_cache_cleanup()
    cuda_cache.suppress_cuda_cache_cleanup()
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(cuda_cache.empty_cuda_cache()))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert results == [False] * 8
    flush.assert_not_called()
    sync.assert_not_called()


def test_transport_start_waits_for_preexisting_flush(policy, monkeypatch):
    entered, release, started = (threading.Event() for _ in range(3))

    def flush():
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(cuda_cache.torch.cuda, "empty_cache", flush)
    flushing = threading.Thread(target=cuda_cache.empty_cuda_cache)
    flushing.start()
    assert entered.wait(2)

    def startup():
        cuda_cache.suppress_cuda_cache_cleanup()
        started.set()

    transport = threading.Thread(target=startup)
    transport.start()
    try:
        assert not started.wait(0.05)
    finally:
        release.set()
        flushing.join(2)
        transport.join(2)
    assert started.is_set()
    assert not cuda_cache.empty_cuda_cache()


def test_fork_reinitializes_lock_without_enabling_cleanup(policy):
    policy.suppress()
    old_lock = policy._lock
    old_lock.acquire()
    try:
        policy.after_fork()
        assert not policy.empty_cache()
    finally:
        old_lock.release()


def test_generic_prefetch_suppresses_before_thread_start_and_after_close(
    policy, monkeypatch
):
    from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin

    packer = PrefetchDataPackerMixin()
    flush = Mock(side_effect=AssertionError("unsafe flush"))
    monkeypatch.setattr(cuda_cache.torch.cuda, "empty_cache", flush)
    original = threading.Thread.start

    def start(thread):
        assert not cuda_cache.empty_cuda_cache()
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    packer._setup_prefetch()
    packer.shutdown_prefetch()
    assert not cuda_cache.empty_cuda_cache()
    flush.assert_not_called()


@pytest.mark.parametrize(
    "backend,kind",
    [
        ("nccl", "strategy"),
        ("nccl", "producer"),
        ("ucxx", "strategy"),
        ("ucxx", "producer"),
    ],
)
def test_native_setup_suppresses_before_partial_acquisition(
    policy, monkeypatch, backend, kind
):
    # Inject immediately at the suppression boundary: no optional native library
    # or resource acquisition should be reached first, even on failed startup.
    class StartupStopped(Exception):
        pass

    def suppress():
        policy.suppress()
        raise StartupStopped()

    monkeypatch.setattr(cuda_cache, "suppress_cuda_cache_cleanup", suppress)
    if backend == "nccl":
        if kind == "strategy":
            from cosmos_rl.utils.payload_transport.nccl.strategy import (
                NCCLTransportStrategy,
            )

            call = partial(
                NCCLTransportStrategy().setup,
                device=None,
                redis_client=None,
                config=None,
            )
        else:
            from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

            call = partial(
                NCCLRolloutMixin().setup_nccl,
                replica_id="p",
                rollout_idx=0,
                redis_client=None,
                config=None,
            )
    elif kind == "strategy":
        from cosmos_rl.utils.payload_transport.ucxx.strategy import (
            UCXXTransportStrategy,
        )

        call = partial(UCXXTransportStrategy().setup, device=None)
    else:
        from cosmos_rl.utils.payload_transport.ucxx.mixins import UCXXRolloutMixin

        call = partial(UCXXRolloutMixin().setup_ucxx, "p", 1, 1, 1)
    with pytest.raises(StartupStopped):
        call()
    assert not cuda_cache.empty_cuda_cache()


def test_cosmos_has_no_bypass_of_explicit_cache_policy():
    root = Path(__file__).resolve().parents[1] / "cosmos_rl"
    bypasses = []
    for path in root.rglob("*.py"):
        if path == root / "utils/cuda_cache.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "empty_cache"
            ):
                bypasses.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not bypasses, bypasses
