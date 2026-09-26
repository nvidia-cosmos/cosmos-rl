"""Real child-process coverage for terminal shutdown and failed state saving."""

import multiprocessing as mp
import queue
import signal
import threading
import time

import pytest

from cosmos_rl.simulators.env_manager import EnvManager


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
