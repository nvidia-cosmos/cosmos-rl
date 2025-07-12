# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed import ReduceOp

from cosmos_rl.utils.logging import logger


# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

ncclDataType_t = ctypes.c_int


# --- NCCL config struct (complete v22700) ---
class ncclConfig_t(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),  # sizeof(ncclConfig_t)
        ("magic", ctypes.c_uint),  # constant magic
        ("version", ctypes.c_uint),  # NCCL version code, e.g. 22703
        ("blocking", ctypes.c_int),  # whether operations are blocking (0 / 1)
        ("cgaClusterSize", ctypes.c_int),
        ("minCTAs", ctypes.c_int),
        ("maxCTAs", ctypes.c_int),
        ("netName", ctypes.c_char_p),
        ("splitShare", ctypes.c_int),
        ("trafficClass", ctypes.c_int),
        ("commName", ctypes.c_char_p),
        ("collnetEnable", ctypes.c_int),
        ("CTAPolicy", ctypes.c_int),
        ("shrinkShare", ctypes.c_int),
        ("nvlsCTAs", ctypes.c_int),
    ]


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclFloat8e4m3 = 10
    ncclFloat8e5m2 = 11

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        if dtype == torch.float8_e4m3fn:
            return cls.ncclFloat8e4m3
        if dtype == torch.float8_e5m2:
            return cls.ncclFloat8e5m2
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class NCCLLibrary:
    # names of optional functions (absence tolerated)
    optional_functions = {"ncclCommInitRankConfig"}
    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t  ncclGetVersion(int *version);
        Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "ncclCommInitRank",
            ncclResult_t,
            [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int],
        ),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ncclAllReduce",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ncclAllGather",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ncclReduceScatter",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclSend(
        #   const void* sendbuff, size_t count, ncclDataType_t datatype,
        #   int dest, ncclComm_t comm, cudaStream_t stream);
        Function(
            "ncclSend",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function(
            "ncclRecv",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, int root, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "ncclBroadcast",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
        # NCCL async error query & abort (for enqueue-timeout protection)
        Function(
            "ncclCommGetAsyncError",
            ncclResult_t,
            [ncclComm_t, ctypes.POINTER(ncclResult_t)],
        ),
        Function("ncclCommAbort", ncclResult_t, [ncclComm_t]),
        # ncclResult_t  ncclCommInitRankConfig(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
        #   ncclConfig_t* config);
        Function(
            "ncclCommInitRankConfig",
            ncclResult_t,
            [
                ctypes.POINTER(ncclComm_t),
                ctypes.c_int,
                ncclUniqueId,
                ctypes.c_int,
                ctypes.POINTER(ncclConfig_t),
            ],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding functions
    path_to_funcs_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str):
        try:
            if so_file not in NCCLLibrary.path_to_library_cache:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception:
            logger.error(f"Failed to load so file: {so_file} from NCCL library. ")
            raise

        if so_file not in NCCLLibrary.path_to_funcs_mapping:
            _funcs: dict[str, Any] = {}
            for func in NCCLLibrary.exported_functions:
                try:
                    f = getattr(self.lib, func.name)
                except AttributeError:
                    if func.name in NCCLLibrary.optional_functions:
                        logger.warning(
                            f"Optional NCCL symbol {func.name} not found; falling back to default behavior."
                        )
                        _funcs[func.name] = None
                        continue
                    raise
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_funcs_mapping[so_file] = _funcs
        self._funcs = NCCLLibrary.path_to_funcs_mapping[so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        """Raise RuntimeError if *result* is an error code."""
        if not ncclResultEnum.is_ok(int(result)):
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(
        self, world_size: int, unique_id: ncclUniqueId, rank: int
    ) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["ncclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def ncclCommInitRankConfig(
        self,
        world_size: int,
        unique_id: ncclUniqueId,
        rank: int,
        blocking: int = 0,
    ) -> ncclComm_t:
        """Python wrapper for ncclCommInitRankConfig with *blocking* flag preset.

        Currently only exposes the *blocking* field (0 = non-blocking NCCL launch).
        Additional ncclConfig_t fields are kept at their default zeros for
        simplicity.
        """
        comm = ncclComm_t()
        config = ncclConfig_t()
        # Fill mandatory header fields according to NCCL_CONFIG_INITIALIZER
        NCCL_CONFIG_MAGIC = 0xCAFEBEEF
        # Try to retrieve NCCL version code (e.g., 22703) from library
        try:
            ver_str = self.ncclGetVersion()  # e.g. "2.27.3"
            major, minor, patch = (int(v) for v in ver_str.split("."))
            version_code = (major * 10000) + (minor * 100) + patch
        except Exception:
            # Fallback to hard-coded current header version
            version_code = 22703

        INT_MIN = -2147483648  # NCCL_CONFIG_UNDEF_INT

        config.size = ctypes.sizeof(ncclConfig_t)
        config.magic = NCCL_CONFIG_MAGIC
        config.version = version_code

        # Initialize all integer fields to UNDEF, pointer fields to NULL
        config.blocking = blocking if blocking in (0, 1) else INT_MIN
        config.cgaClusterSize = INT_MIN
        config.minCTAs = INT_MIN
        config.maxCTAs = INT_MIN
        config.netName = None
        config.splitShare = INT_MIN
        config.trafficClass = INT_MIN
        config.commName = None
        config.collnetEnable = INT_MIN
        config.CTAPolicy = INT_MIN
        config.shrinkShare = INT_MIN
        config.nvlsCTAs = INT_MIN

        fn = self._funcs.get("ncclCommInitRankConfig")
        if fn is None:
            logger.debug(
                "ncclCommInitRankConfig symbol missing – falling back to ncclCommInitRank"
            )
            return self.ncclCommInitRank(world_size, unique_id, rank)

        ret = fn(
            ctypes.byref(comm),
            world_size,
            unique_id,
            rank,
            ctypes.byref(config),
        )
        if not ncclResultEnum.is_ok(int(ret)):
            self.NCCL_CHECK(ret)
        return comm

    def ncclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["ncclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["ncclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["ncclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def ncclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def ncclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def ncclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))

    # ------------------ new helpers for HA ------------------

    def ncclCommGetAsyncError(self, comm: ncclComm_t) -> int:
        err = ncclResult_t()
        self._funcs["ncclCommGetAsyncError"](comm, ctypes.byref(err))
        return int(err.value)

    def ncclCommAbort(self, comm: ncclComm_t) -> None:
        # abort can return error itself; ignore it to avoid masking original issue
        self._funcs["ncclCommAbort"](comm)


class ncclResultEnum:
    """Enumeration of NCCL result codes copied from nccl.h."""

    ncclSuccess = 0
    ncclUnhandledCudaError = 1
    ncclSystemError = 2
    ncclInternalError = 3
    ncclInvalidArgument = 4
    ncclInvalidUsage = 5
    ncclRemoteError = 6
    ncclInProgress = 7

    @staticmethod
    def is_ok(result: int) -> bool:
        """Return True if *result* represents a non-error condition."""
        return result in (
            ncclResultEnum.ncclSuccess,
            ncclResultEnum.ncclInProgress,
        )


__all__ = [
    "NCCLLibrary",
    "ncclDataTypeEnum",
    "ncclRedOpTypeEnum",
    "ncclResultEnum",
    "ncclUniqueId",
    "ncclComm_t",
    "cudaStream_t",
    "buffer_type",
    "ncclConfig_t",
]
