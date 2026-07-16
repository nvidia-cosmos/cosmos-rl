# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the optional point-to-point NCCL phase observer."""

import threading
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from cosmos_rl.utils import pynccl
from cosmos_rl.utils.pynccl_wrapper import NCCLLibrary, ncclResultEnum


class TestP2PPhaseObserver(unittest.TestCase):
    def _invoke_p2p(
        self,
        direction: str,
        *,
        observer=None,
        raw_result: int = ncclResultEnum.ncclSuccess,
        query_results=((ncclResultEnum.ncclSuccess, ncclResultEnum.ncclSuccess),),
    ) -> dict[str, Mock]:
        tensor = Mock()
        tensor.numel.return_value = 8
        meta = pynccl._CommMeta(comm=Mock(), rank=1, world_size=2)
        legacy_raw = Mock()
        result_raw = Mock(return_value=raw_result)
        legacy_query = Mock(return_value=ncclResultEnum.ncclSuccess)
        result_query = Mock(side_effect=query_results)
        call = pynccl.nccl_send if direction == "send" else pynccl.nccl_recv
        legacy_raw_name = "ncclSend" if direction == "send" else "ncclRecv"
        result_raw_name = (
            "_ncclSendResult" if direction == "send" else "_ncclRecvResult"
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(pynccl, "_worker_started", True))
            stack.enter_context(patch.object(pynccl, "_check_tensor"))
            stack.enter_context(
                patch.object(pynccl, "_stream_ptr", return_value=Mock())
            )
            stack.enter_context(patch.object(pynccl, "_buf", return_value=Mock()))
            stack.enter_context(patch.object(pynccl, "_dtype_enum", return_value=7))
            stack.enter_context(
                patch.object(pynccl._CommunicatorRegistry, "get", return_value=meta)
            )
            stack.enter_context(patch.object(pynccl._nccl, legacy_raw_name, legacy_raw))
            stack.enter_context(
                patch.object(
                    pynccl._nccl,
                    result_raw_name,
                    result_raw,
                    create=True,
                )
            )
            stack.enter_context(
                patch.object(
                    pynccl._nccl,
                    "ncclCommGetAsyncError",
                    legacy_query,
                )
            )
            stack.enter_context(
                patch.object(
                    pynccl._nccl,
                    "_ncclCommGetAsyncErrorResult",
                    result_query,
                    create=True,
                )
            )
            stack.enter_context(patch.object(pynccl.time, "sleep"))

            kwargs = {"phase_observer": observer} if observer is not None else {}
            call(tensor, 0, 4, **kwargs)

        return {
            "legacy_raw": legacy_raw,
            "result_raw": result_raw,
            "legacy_query": legacy_query,
            "result_query": result_query,
        }

    def test_disabled_path_uses_the_legacy_calls(self):
        for direction in ("send", "recv"):
            with self.subTest(direction=direction):
                calls = self._invoke_p2p(direction)

                calls["legacy_raw"].assert_called_once()
                calls["legacy_query"].assert_called_once()
                calls["result_raw"].assert_not_called()
                calls["result_query"].assert_not_called()

    def test_send_and_recv_report_results_in_order(self):
        expected_events = [
            ("raw_call_enter", None, None),
            ("raw_call_return", ncclResultEnum.ncclSuccess, None),
            ("async_error_query_enter", None, None),
            (
                "async_error_query_return",
                ncclResultEnum.ncclSuccess,
                ncclResultEnum.ncclInProgress,
            ),
            ("async_error_query_enter", None, None),
            (
                "async_error_query_return",
                ncclResultEnum.ncclSuccess,
                ncclResultEnum.ncclSuccess,
            ),
        ]
        for direction in ("send", "recv"):
            with self.subTest(direction=direction):
                events = []
                calls = self._invoke_p2p(
                    direction,
                    observer=lambda *event: events.append(event),
                    query_results=(
                        (ncclResultEnum.ncclSuccess, ncclResultEnum.ncclInProgress),
                        (ncclResultEnum.ncclSuccess, ncclResultEnum.ncclSuccess),
                    ),
                )

                self.assertEqual(events, expected_events)
                calls["result_raw"].assert_called_once()
                self.assertEqual(calls["result_query"].call_count, 2)
                calls["legacy_raw"].assert_not_called()
                calls["legacy_query"].assert_not_called()

    def test_query_api_result_is_distinct_from_communicator_state(self):
        events = []

        self._invoke_p2p(
            "send",
            observer=lambda *event: events.append(event),
            query_results=(
                (ncclResultEnum.ncclSystemError, ncclResultEnum.ncclSuccess),
            ),
        )

        self.assertEqual(
            events[-1],
            (
                "async_error_query_return",
                ncclResultEnum.ncclSystemError,
                ncclResultEnum.ncclSuccess,
            ),
        )

    def test_observer_failure_does_not_change_the_operation(self):
        calls = 0

        def fail_observer(*_event):
            nonlocal calls
            calls += 1
            raise RuntimeError("observer failed")

        self._invoke_p2p("send", observer=fail_observer)

        self.assertGreater(calls, 0)

    def test_query_enter_is_visible_while_the_native_query_is_blocked(self):
        native_query_entered = threading.Event()
        release_native_query = threading.Event()
        events = []
        errors = []

        def blocking_query(_comm):
            native_query_entered.set()
            release_native_query.wait(timeout=2)
            return ncclResultEnum.ncclSuccess, ncclResultEnum.ncclSuccess

        def invoke():
            try:
                self._invoke_p2p(
                    "send",
                    observer=lambda *event: events.append(event),
                    query_results=blocking_query,
                )
            except Exception as error:  # pragma: no cover - asserted below
                errors.append(error)

        thread = threading.Thread(target=invoke)
        thread.start()
        self.assertTrue(native_query_entered.wait(timeout=1))
        self.assertEqual(events[-1], ("async_error_query_enter", None, None))
        release_native_query.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_async_error_reports_abort_without_reordering_timeout_state(self):
        events = []
        task = pynccl._Task(
            functor=lambda: Mock(),
            timeout_ms=100,
            comm_idx=4,
            phase_observer=lambda *event: events.append(event),
        )
        timed_out_during_abort = []

        def abort(_comm_idx, _comm):
            timed_out_during_abort.append(task.timed_out.is_set())

        with (
            patch.object(
                pynccl._nccl,
                "_ncclCommGetAsyncErrorResult",
                return_value=(
                    ncclResultEnum.ncclSuccess,
                    ncclResultEnum.ncclSystemError,
                ),
                create=True,
            ),
            patch.object(pynccl, "_safe_abort", side_effect=abort),
        ):
            pynccl.run_task(task)

        self.assertEqual(
            [event[0] for event in events[-4:]],
            [
                "async_error_query_enter",
                "async_error_query_return",
                "abort_enter",
                "abort_return",
            ],
        )
        self.assertEqual(timed_out_during_abort, [False])
        self.assertTrue(task.timed_out.is_set())

    def test_enqueue_timeout_reports_abort_without_querying(self):
        events = []
        task = pynccl._Task(
            functor=lambda: Mock(),
            timeout_ms=0,
            comm_idx=4,
            phase_observer=lambda *event: events.append(event),
        )

        with (
            patch.object(
                pynccl._nccl,
                "_ncclCommGetAsyncErrorResult",
                create=True,
            ) as query,
            patch.object(pynccl, "_safe_abort") as abort,
        ):
            pynccl.run_task(task)

        self.assertEqual(
            [event[0] for event in events],
            ["abort_enter", "abort_return"],
        )
        query.assert_not_called()
        abort.assert_called_once()
        self.assertTrue(task.timed_out.is_set())


class TestNCCLResultSeams(unittest.TestCase):
    def _library(self, functions: dict[str, Mock]) -> NCCLLibrary:
        library = object.__new__(NCCLLibrary)
        library._funcs = functions
        return library

    def test_send_and_recv_result_seams_preserve_checked_methods(self):
        for name, arguments in (
            ("Send", (None, 8, 7, 1, None, None)),
            ("Recv", (None, 8, 7, 1, None, None)),
        ):
            with self.subTest(name=name):
                native = Mock(return_value=ncclResultEnum.ncclSystemError)
                library = self._library(
                    {
                        f"nccl{name}": native,
                        "ncclGetErrorString": Mock(return_value=b"simulated"),
                    }
                )

                raw_result = getattr(library, f"_nccl{name}Result")(*arguments)
                with self.assertRaisesRegex(RuntimeError, "NCCL error: simulated"):
                    getattr(library, f"nccl{name}")(*arguments)

                self.assertEqual(raw_result, ncclResultEnum.ncclSystemError)

    def test_async_query_result_seam_preserves_compatibility_result(self):
        def query(_comm, state):
            state._obj.value = ncclResultEnum.ncclInProgress
            return ncclResultEnum.ncclSystemError

        native = Mock(side_effect=query)
        library = self._library({"ncclCommGetAsyncError": native})

        exact_result = library._ncclCommGetAsyncErrorResult(None)
        compatibility_result = library.ncclCommGetAsyncError(None)

        self.assertEqual(
            exact_result,
            (ncclResultEnum.ncclSystemError, ncclResultEnum.ncclInProgress),
        )
        self.assertEqual(compatibility_result, ncclResultEnum.ncclInProgress)
        self.assertEqual(native.call_count, 2)


if __name__ == "__main__":
    unittest.main()
