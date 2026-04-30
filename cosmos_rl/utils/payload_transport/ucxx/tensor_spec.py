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

"""``TensorSpec`` — fixed-shape tensor descriptor for flat byte schemas.

Used by :mod:`cosmos_rl.utils.payload_transport.ucxx` to describe the
on-the-wire layout of a single trajectory entry in a
:class:`~cosmos_rl.utils.payload_transport.ucxx.shared_buffer.SharedRingBuffer`.
Lives in its own module so it can be re-used by additional transports
or by callers wanting to declare a flat schema without depending on the
full UCXX stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class TensorSpec:
    """Fixed-shape tensor descriptor.

    Attributes:
        shape: Tuple of dimensions (e.g. ``(max_steps, obs_dim)``).
        dtype: Element type, accepted as a Python type
            (``np.float32``) or :class:`numpy.dtype` instance.  Always
            normalized to :class:`numpy.dtype` after construction.
        name: Identifier used in flat schemas (e.g.
            ``"observations"``).  Required when the spec participates
            in a UCXX schema.
    """

    shape: Tuple[int, ...]
    dtype: Union[type, np.dtype]
    name: str = ""

    def __post_init__(self) -> None:
        # ``dataclass`` is permissive about dtype; normalize once so
        # downstream code can rely on ``self.dtype.itemsize`` etc.
        self.dtype = np.dtype(self.dtype)

    @property
    def nbytes(self) -> int:
        """Byte size of one tensor matching this spec."""
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def contains(self, tensor: np.ndarray) -> bool:
        """Return True if ``tensor`` has matching shape and dtype."""
        if tensor.shape != self.shape:
            return False
        if tensor.dtype != self.dtype:
            return False
        return True


__all__ = ["TensorSpec"]
