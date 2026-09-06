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
            # Real NCCL does not return quietly once its communicator is
            # aborted: the blocked call returns an error code and NCCL_CHECK
            # raises. Returning a Mock here would let the test pass without
            # exercising what the abort actually produces, and would not
            # notice a timeout being reported as that error instead.
            released.wait(timeout=30.0)
            raise RuntimeError("NCCL error: unhandled system error (aborted)")

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


class TestFailureCauseSurvivesTheWorker(unittest.TestCase):
    """A failure must be reported as itself, not as a timeout.

    ``run_task`` sets ``timed_out`` for every way a task can end badly, so
    before this distinction existed ``_submit_nccl`` reported an invalid
    argument, a fabric error and a genuinely absent peer identically -- as
    ``TimeoutError: NCCL: non-blocking enqueue timed out``. Callers act on that
    difference (a timeout means a peer never arrived; nothing else does), and
    the original message, the only thing that named the real fault, was
    discarded.
    """

    def _submit(self, functor, comm_idx=None):
        with patch.object(pynccl, "_worker_started", True):
            pynccl._submit_nccl(functor, 200, comm_idx=comm_idx)

    def test_raised_exception_is_re_raised_unchanged(self):
        boom = ValueError("ncclInvalidArgument: bad rank")

        def _functor():
            raise boom

        with self.assertRaises(ValueError) as caught:
            self._submit(_functor)

        # The very same object, so type, message and traceback all survive.
        self.assertIs(caught.exception, boom)

    def test_a_real_deadline_still_raises_timeout(self):
        # Nothing raised; the async-error poll simply never reports success.
        # This is the case that genuinely means "a peer never arrived".
        def _functor():
            return Mock()

        with patch.object(pynccl, "_nccl") as nccl:
            nccl.ncclCommGetAsyncError.return_value = ncclResultEnum.ncclInProgress
            with patch.object(pynccl, "nccl_abort", Mock()):
                with self.assertRaises(TimeoutError):
                    self._submit(_functor, comm_idx=3)

    def test_async_error_is_reported_as_an_error_not_a_timeout(self):
        def _functor():
            return Mock()

        with patch.object(pynccl, "_nccl") as nccl:
            nccl.ncclCommGetAsyncError.return_value = ncclResultEnum.ncclSystemError
            with patch.object(pynccl, "nccl_abort", Mock()):
                with self.assertRaises(RuntimeError) as caught:
                    self._submit(_functor, comm_idx=3)

        self.assertNotIsInstance(caught.exception, TimeoutError)
        self.assertIn("asynchronous error", str(caught.exception))

    def test_successful_task_records_no_error(self):
        def _functor():
            return Mock()

        with patch.object(pynccl, "_nccl") as nccl:
            nccl.ncclCommGetAsyncError.return_value = ncclResultEnum.ncclSuccess
            task = pynccl._Task(_functor, 200, 3)
            pynccl.run_task(task)

        self.assertIsNone(task.error)
        self.assertFalse(task.timed_out.is_set())


if __name__ == "__main__":
    unittest.main()
