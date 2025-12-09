# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from typing import IO

from cosmos_rl.utils.wfm.io.easy_io.handlers.base import BaseFileHandler


class ByteHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file: IO[bytes], **kwargs):
        file.seek(0)
        # extra all bytes and return
        return file.read()

    def dump_to_fileobj(
        self,
        obj: bytes,
        file: IO[bytes],
        **kwargs,
    ):
        # write all bytes to file
        file.write(obj)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
