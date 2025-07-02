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


class DimRankInfo:
    rank: int
    size: int
    dim: str
    length: int = 1

    def __init__(self, rank: int, size: int, dim: str, length: int = 1):
        """
        Initialize the DimRankInfo with the given rank, size, and dimension.
        """
        self.rank = rank
        self.size = size
        self.dim = dim
        self.length = length

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"
