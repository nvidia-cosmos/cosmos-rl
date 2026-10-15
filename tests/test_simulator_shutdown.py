"""Real child-process coverage for terminal shutdown and failed state saving."""

import multiprocessing as mp
import queue
import signal
import threading
import time
import ctypes
import gc
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.simulators.env_manager import EnvManager
from cosmos_rl.simulators.env_manager import _simulator_worker
import torch


def unresponsive_child(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    threading.Event().wait()


@pytest.mark.parametrize("preserve_state", [False, True])
def test_unresponsive_child_is_killed_and_reaped(preserve_state):
    context = mp.get_context("spawn")
    manager = object.__new__(EnvManager)
    manager.env = None
    manager.state_buffer = None
    manager.command_queue = context.Queue()
    manager.result_queue = context.Queue()
    ready = context.Event()
    process = context.Process(target=unresponsive_child, args=(ready,))
    manager.process = process
    process.start()
    try:
        assert ready.wait(20)
        start = time.monotonic()
        if preserve_state:
            with pytest.raises(queue.Empty):
                manager.stop_simulator(state_timeout=0.05, join_timeout=0.1)
        else:
            manager.stop_simulator(preserve_state=False, join_timeout=0.1)
        assert time.monotonic() - start < 5
        assert not process.is_alive()
        assert process.exitcode == -signal.SIGKILL
        assert manager.process is None
        assert manager.command_queue is None
        assert manager.result_queue is None
        manager.stop_simulator(preserve_state=False)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()


def test_dead_child_is_reaped_and_queues_closed():
    context = mp.get_context("spawn")
    manager = object.__new__(EnvManager)
    manager.env = None
    manager.command_queue = context.Queue()
    manager.result_queue = context.Queue()
    manager.process = None
    manager.stop_simulator(preserve_state=False)
    assert manager.command_queue is None
    assert manager.result_queue is None


@pytest.mark.parametrize("constructor_fails", [False, True])
def test_worker_exit_never_invalidates_live_tensor_references(
    monkeypatch, constructor_fails
):
    # Exercise the old destructive path without actually corrupting Python's
    # heap. The owner and queue can still hold tensor references during finally.
    tensor = SimpleNamespace(is_cuda=True)
    decref = Mock()
    empty_cache = Mock()
    commands = SimpleNamespace(get=lambda: {"method": "shutdown"}, close=Mock())
    results = SimpleNamespace(put=Mock(), close=Mock())

    def env_cls(cfg, count):
        if constructor_fails:
            raise RuntimeError("constructor failed")
        return SimpleNamespace(tensor=tensor)

    with monkeypatch.context() as patch:
        patch.setattr(gc, "get_objects", lambda: [tensor])
        patch.setattr(torch, "is_tensor", lambda value: value is tensor)
        patch.setattr(ctypes.pythonapi, "Py_DecRef", decref)
        patch.setattr(torch.cuda, "empty_cache", empty_cache)
        _simulator_worker(0, {}, 1, env_cls, commands, results, None, bind_numa=False)
    decref.assert_not_called()
    empty_cache.assert_not_called()
    commands.close.assert_called_once()
    results.close.assert_called_once()


def worker_with_live_tensor(device, constructor_fails, report):
    tensor = torch.arange(8, device=device)
    commands = SimpleNamespace(get=lambda: {"method": "shutdown"}, close=lambda: None)
    results = SimpleNamespace(put=lambda value: None, close=lambda: None)

    def env_cls(cfg, count):
        if constructor_fails:
            raise RuntimeError("constructor failed with a live tensor owner")
        return SimpleNamespace(tensor=tensor)

    _simulator_worker(0, {}, 1, env_cls, commands, results, None, bind_numa=False)
    # This owner outlives the worker function, just like a queue feeder or a
    # backend finalizer. Its storage must remain valid until normal destruction.
    tensor.add_(1)
    report.put(int(tensor.sum().item()))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("constructor_fails", [False, True])
def test_worker_process_preserves_live_tensors_until_process_exit(
    device, constructor_fails
):
    if device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("COSMOS_REQUIRE_CUDA") == "1":
            pytest.fail("Required native CUDA cleanup control may not skip")
        pytest.skip("CUDA unavailable")
    context = mp.get_context("spawn")
    report = context.Queue()
    process = context.Process(
        target=worker_with_live_tensor, args=(device, constructor_fails, report)
    )
    process.start()
    try:
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert report.get(timeout=5) == 36
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()
        report.close()
        report.join_thread()
