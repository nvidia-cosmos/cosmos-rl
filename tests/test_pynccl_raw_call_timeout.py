# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for bounding the raw NCCL host call and the group wrappers."""

import threading
import time
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from cosmos_rl.utils import pynccl
from cosmos_rl.utils.pynccl_wrapper import ncclResultEnum


class TestRawCallTimeout(unittest.TestCase):
    """A raw host call that never returns must still hit the task timeout."""

    def _blocking_task(self, timeout_ms: int, comm_idx):
        """Build a task whose functor blocks until the abort releases it."""
        released = threading.Event()

        def _functor():
            # Stands in for ``ncclRecv`` against a peer that never sends: the
            # call only returns once the communicator is aborted.
            released.wait(timeout=30.0)
            return Mock()

        task = pynccl._Task(_functor, timeout_ms, comm_idx)
        return task, released

    def test_blocked_raw_call_aborts_communicator_and_times_out(self):
        task, released = self._blocking_task(timeout_ms=200, comm_idx=7)
        abort = Mock(side_effect=lambda _idx: released.set())

        started = time.monotonic()
        with patch.object(pynccl, "nccl_abort", abort):
            pynccl.run_task(task)
        elapsed = time.monotonic() - started

        abort.assert_called_once_with(7)
        self.assertTrue(task.timed_out.is_set())
        self.assertTrue(task.done.is_set())
        # Bounded by the task timeout, not by the functor's own 30s ceiling.
        self.assertLess(elapsed, 10.0)

    def test_submit_raises_timeout_error_for_blocked_raw_call(self):
        released = threading.Event()

        def _functor():
            released.wait(timeout=30.0)
            return Mock()

        with ExitStack() as stack:
            stack.enter_context(patch.object(pynccl, "_worker_started", True))
            stack.enter_context(
                patch.object(
                    pynccl,
                    "nccl_abort",
                    Mock(side_effect=lambda _idx: released.set()),
                )
            )
            with self.assertRaises(TimeoutError):
                pynccl._submit_nccl(_functor, 200, 3)

    def test_returning_raw_call_is_not_aborted(self):
        task, _ = self._blocking_task(timeout_ms=60_000, comm_idx=7)
        task.functor = lambda: Mock()
        abort = Mock()
        query = Mock(return_value=ncclResultEnum.ncclSuccess)

        with ExitStack() as stack:
            stack.enter_context(patch.object(pynccl, "nccl_abort", abort))
            stack.enter_context(
                patch.object(pynccl._nccl, "ncclCommGetAsyncError", query)
            )
            pynccl.run_task(task)

        abort.assert_not_called()
        self.assertFalse(task.timed_out.is_set())

    def test_communicator_creation_has_nothing_to_abort(self):
        """``comm_idx=None`` means the communicator does not exist yet."""
        task = pynccl._Task(lambda: Mock(), 200, None)
        abort = Mock()
        query = Mock(return_value=ncclResultEnum.ncclSuccess)

        with ExitStack() as stack:
            stack.enter_context(patch.object(pynccl, "nccl_abort", abort))
            stack.enter_context(
                patch.object(pynccl._nccl, "ncclCommGetAsyncError", query)
            )
            pynccl.run_task(task)

        abort.assert_not_called()
        self.assertFalse(task.timed_out.is_set())


class TestGroupTimeoutForwarding(unittest.TestCase):
    """The group wrappers must honour the caller's budget, not the default."""

    def _record_submit(self, call, timeout_ms):
        submit = Mock()
        meta = pynccl._CommMeta(comm=Mock(), rank=0, world_size=2)
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(pynccl._CommunicatorRegistry, "get", return_value=meta)
            )
            stack.enter_context(patch.object(pynccl, "_submit_nccl", submit))
            if timeout_ms is None:
                call(4)
            else:
                call(4, timeout_ms=timeout_ms)
        return submit.call_args

    def test_group_start_forwards_explicit_timeout(self):
        args, _ = self._record_submit(pynccl.nccl_group_start, 1_800_000)
        self.assertEqual(args[1:], (1_800_000, 4))

    def test_group_end_forwards_explicit_timeout(self):
        args, _ = self._record_submit(pynccl.nccl_group_end, 1_800_000)
        self.assertEqual(args[1:], (1_800_000, 4))

    def test_group_helpers_still_default_to_environment_timeout(self):
        for call in (pynccl.nccl_group_start, pynccl.nccl_group_end):
            args, _ = self._record_submit(call, None)
            self.assertEqual(args[1:], (None, 4))


if __name__ == "__main__":
    unittest.main()
