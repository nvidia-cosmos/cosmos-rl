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

import tarfile

from cosmos_rl.utils.wfm.io.easy_io.handlers.base import BaseFileHandler


class TarHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, mode="r|*", **kwargs):
        return tarfile.open(fileobj=file, mode=mode, **kwargs)

    def load_from_path(self, filepath, mode="r|*", **kwargs):
        return tarfile.open(filepath, mode=mode, **kwargs)

    def dump_to_fileobj(self, obj, file, mode="w", **kwargs):
        with tarfile.open(fileobj=file, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with tarfile.open(filepath, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
