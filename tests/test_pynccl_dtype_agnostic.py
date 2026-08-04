# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests pinning the dtype/count arguments handed to the raw NCCL API.

Movement collectives (broadcast / send / recv / allgather) must be issued as
``ncclUint8`` over a byte count so that every torch dtype transfers -- including
``torch.bool`` and ``torch.int16``, which have no entry in ``ncclDataTypeEnum``
(NCCL has no int16 type at all).  Reductions must stay typed, since their
arithmetic depends on the dtype.

These run without CUDA or a communicator: the raw library calls are mocked and
the recorded arguments are asserted directly.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import torch
from torch.distributed import ReduceOp

from cosmos_rl.utils import pynccl
from cosmos_rl.utils.pynccl_wrapper import ncclDataTypeEnum

# (dtype, itemsize).  bool and int16 are the regression cases -- both are absent
# from ncclDataTypeEnum.from_torch, so they only work via the byte path.
DTYPES = [
    (torch.bool, 1),
    (torch.int8, 1),
    (torch.uint8, 1),
    (torch.int16, 2),
    (torch.float16, 2),
    (torch.bfloat16, 2),
    (torch.int32, 4),
    (torch.float32, 4),
    (torch.int64, 8),
    (torch.float64, 8),
]

# Deliberately not a power of two: a rounding or off-by-one error in the byte
# arithmetic stays visible instead of being masked by a round number.
NUMEL = 7


def _tensor(dtype: torch.dtype, numel: int = NUMEL) -> torch.Tensor:
    return torch.zeros(numel, dtype=dtype)


class _RawCalls(ExitStack):
    """Patch pynccl's plumbing so a wrapper runs inline against mocked C calls."""

    def __init__(self, *names: str):
        super().__init__()
        self._names = names
        self.mocks: dict[str, Mock] = {}

    def __enter__(self) -> "_RawCalls":
        super().__enter__()
        meta = pynccl._CommMeta(comm=Mock(), rank=1, world_size=2)
        self.enter_context(patch.object(pynccl, "_worker_started", True))
        self.enter_context(patch.object(pynccl, "_check_tensor"))
        self.enter_context(patch.object(pynccl, "_stream_ptr", return_value=Mock()))
        self.enter_context(patch.object(pynccl, "_buf", return_value=Mock()))
        self.enter_context(
            patch.object(pynccl._CommunicatorRegistry, "get", return_value=meta)
        )
        self.enter_context(
            patch.object(
                pynccl._nccl,
                "ncclCommGetAsyncError",
                Mock(return_value=0),
            )
        )
        for name in self._names:
            mock = Mock()
            self.enter_context(patch.object(pynccl._nccl, name, mock, create=True))
            self.mocks[name] = mock
        return self


class TestMovementIsDtypeAgnostic(unittest.TestCase):
    """Every movement wrapper sends bytes, whatever the tensor dtype."""

    def _assert_bytes(self, mock: Mock, count_index: int, expected_bytes: int):
        mock.assert_called_once()
        args = mock.call_args.args
        self.assertEqual(args[count_index], expected_bytes)
        self.assertEqual(args[count_index + 1], ncclDataTypeEnum.ncclUint8)

    def test_broadcast_passes_a_byte_count_for_every_dtype(self):
        for dtype, itemsize in DTYPES:
            with self.subTest(dtype=dtype):
                with _RawCalls("ncclBroadcast") as raw:
                    pynccl.nccl_broadcast(_tensor(dtype), 1, 4)
                # ncclBroadcast(sendbuf, recvbuf, count, datatype, root, ...)
                self._assert_bytes(raw.mocks["ncclBroadcast"], 2, NUMEL * itemsize)

    def test_send_and_recv_pass_a_byte_count_for_every_dtype(self):
        for direction, name in (("send", "ncclSend"), ("recv", "ncclRecv")):
            call = pynccl.nccl_send if direction == "send" else pynccl.nccl_recv
            for dtype, itemsize in DTYPES:
                with self.subTest(direction=direction, dtype=dtype):
                    with _RawCalls(name) as raw:
                        call(_tensor(dtype), 0, 4)
                    # nccl{Send,Recv}(buf, count, datatype, peer, ...)
                    self._assert_bytes(raw.mocks[name], 1, NUMEL * itemsize)

    def test_observer_path_passes_the_same_byte_count(self):
        """The phase-observer branch is a separate call site; it must match."""
        for direction, name in (
            ("send", "_ncclSendResult"),
            ("recv", "_ncclRecvResult"),
        ):
            call = pynccl.nccl_send if direction == "send" else pynccl.nccl_recv
            with self.subTest(direction=direction):
                with _RawCalls(name) as raw:
                    raw.mocks[name].return_value = 0
                    raw.enter_context(
                        patch.object(
                            pynccl._nccl,
                            "_ncclCommGetAsyncErrorResult",
                            Mock(return_value=(0, 0)),
                            create=True,
                        )
                    )
                    call(_tensor(torch.bool), 0, 4, phase_observer=lambda *_: None)
                self._assert_bytes(raw.mocks[name], 1, NUMEL)

    def test_alltoall_passes_a_byte_count(self):
        for dtype, itemsize in DTYPES:
            with self.subTest(dtype=dtype):
                with _RawCalls("ncclAllGather") as raw:
                    pynccl.nccl_alltoall(_tensor(dtype), _tensor(dtype, NUMEL * 2), 4)
                # ncclAllGather(sendbuf, recvbuf, sendcount, datatype, ...)
                self._assert_bytes(raw.mocks["ncclAllGather"], 2, NUMEL * itemsize)


class TestReductionsStayTyped(unittest.TestCase):
    """Reductions must keep the real dtype -- their arithmetic depends on it."""

    def test_allreduce_passes_the_typed_enum(self):
        with _RawCalls("ncclAllReduce") as raw:
            buf = _tensor(torch.float32)
            pynccl.nccl_allreduce(buf, buf, ReduceOp.SUM, 4)

        args = raw.mocks["ncclAllReduce"].call_args.args
        # ncclAllReduce(sendbuf, recvbuf, count, datatype, op, ...)
        self.assertEqual(args[2], NUMEL)  # elements, not bytes
        self.assertEqual(args[3], ncclDataTypeEnum.ncclFloat32)

    def test_allreduce_still_rejects_bool(self):
        """A bool allreduce must fail rather than silently summing bytes.

        The dtype lookup happens inside the functor that runs on the NCCL worker
        thread, and _submit_nccl reports a functor failure as a TimeoutError on
        the enqueue path -- so assert on the observable contract (the collective
        never reaches the wire) rather than on the exception type.
        """
        with _RawCalls("ncclAllReduce") as raw:
            buf = _tensor(torch.bool)
            with self.assertRaises(Exception):
                pynccl.nccl_allreduce(buf, buf, ReduceOp.SUM, 4)
            raw.mocks["ncclAllReduce"].assert_not_called()

        # The mapping itself raises precisely, which is what callers see when
        # they resolve the dtype up front.
        with self.assertRaisesRegex(ValueError, "reduction"):
            pynccl._dtype_enum(torch.bool)

    def test_from_torch_still_rejects_bool(self):
        """Intentional, not an oversight.

        ncclSum over bools-as-bytes yields counts rather than a logical OR, so
        bool has no correct reduction mapping.  Movement does not come through
        here, so nothing needs it.
        """
        for dtype in (torch.bool, torch.int16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "reduction"):
                    ncclDataTypeEnum.from_torch(dtype)

    def test_from_torch_still_maps_the_reducible_dtypes(self):
        self.assertEqual(
            ncclDataTypeEnum.from_torch(torch.float32), ncclDataTypeEnum.ncclFloat32
        )
        self.assertEqual(
            ncclDataTypeEnum.from_torch(torch.int64), ncclDataTypeEnum.ncclInt64
        )


if __name__ == "__main__":
    unittest.main()
