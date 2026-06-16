# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(message: str):
    """Best-effort NVTX range that is a no-op when CUDA/NVTX is unavailable."""
    if not torch.cuda.is_available():
        yield
        return

    pushed = False
    try:
        torch.cuda.nvtx.range_push(message)
        pushed = True
    except Exception:
        pushed = False

    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
