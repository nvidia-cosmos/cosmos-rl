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

from torch.utils.data import Sampler
from typing import Iterator


class SkippingSampler(Sampler[int]):
    """
    One-shot wrapper around an index-level Sampler that skips `skip_samples`
    indices once, then behaves like the base sampler thereafter.
    """

    def __init__(self, base_sampler: Sampler[int], skip_samples: int = 0):
        self.base = base_sampler
        self._initial_skip = max(0, int(skip_samples))
        self._remaining_skip = self._initial_skip
        self._used_once = False

    def __iter__(self) -> Iterator[int]:
        it = iter(self.base)
        if self._remaining_skip > 0:
            for _ in range(self._remaining_skip):
                try:
                    next(it)
                except StopIteration:
                    self._remaining_skip = 0
                    self._used_once = True
                    return iter(())
            self._remaining_skip = 0
        self._used_once = True
        return it

    def __len__(self) -> int:
        base_len = len(self.base)
        if not self._used_once:
            return max(0, base_len - self._initial_skip)
        return base_len

    def set_epoch(self, epoch: int):
        self.base.set_epoch(epoch)
