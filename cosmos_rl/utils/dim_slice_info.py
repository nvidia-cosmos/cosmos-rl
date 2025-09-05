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

from typing import Dict, Any
from functools import reduce
from math import gcd


class DimSliceInfo:
    """
    A class to represent the slice information of a tensor along a specific dimension.
    This class contains the offset, total size, dimension name, and length of the slice.
    """

    offset: int
    total_size: int
    dim: str
    length: int = 1

    def __init__(self, offset: int, total_size: int, dim: str = "", length: int = 1):
        """
        Initialize the DimSliceInfo with the given offset, total size, dimension name, and length.
        """
        self.offset = offset
        self.total_size = total_size
        self.dim = dim
        self.length = length

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create a DimSliceInfo object from a dictionary.
        :param data: A dictionary containing the keys 'offset', 'total_size', 'dim', and 'length'.
        :return: A DimSliceInfo object.
        """
        return DimSliceInfo(
            offset=data["offset"],
            total_size=data["total_size"],
            dim=data.get("dim", ""),
            length=data.get("length", 1),
        )

    def simplify(self):
        common = reduce(gcd, [self.offset, self.total_size, self.length])  # noqa: E741
        return DimSliceInfo(
            offset=self.offset // common,
            total_size=self.total_size // common,
            dim=self.dim,
            length=self.length // common,
        )
