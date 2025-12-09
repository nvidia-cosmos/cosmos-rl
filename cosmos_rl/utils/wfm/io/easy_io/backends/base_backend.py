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

import os
import os.path as osp
from abc import ABCMeta, abstractmethod


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def has_method(obj, method):
    return hasattr(obj, method) and callable(getattr(obj, method))


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: :meth:`get()` and
    :meth:`get_text()`.

    - :meth:`get()` reads the file as a byte stream.
    - :meth:`get_text()` reads the file as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink = False

    @property
    def allow_symlink(self):
        return self._allow_symlink

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass
