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

import csv
from io import StringIO

from cosmos_rl.utils.wfm.io.easy_io.handlers.base import BaseFileHandler


class CsvHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        del kwargs
        reader = csv.reader(file)
        return list(reader)

    def dump_to_fileobj(self, obj, file, **kwargs):
        del kwargs
        writer = csv.writer(file)
        if not all(isinstance(row, list) for row in obj):
            raise ValueError("Each row must be a list")
        writer.writerows(obj)

    def dump_to_str(self, obj, **kwargs):
        del kwargs
        output = StringIO()
        writer = csv.writer(output)
        if not all(isinstance(row, list) for row in obj):
            raise ValueError("Each row must be a list")
        writer.writerows(obj)
        return output.getvalue()
