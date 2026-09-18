# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy for explicitly opportunistic CUDA allocator flushing only.

An idle Python queue does not prove native transport completion. Once a process
starts asynchronous payload transport, skip opportunistic flush requests.
Intentional memory-release call sites are not intercepted or reclassified.
"""

import os
import threading

import torch


class _CacheCleanupPolicy:
    def __init__(self):
        self._lock = threading.Lock()
        self._suppressed = False

    def suppress(self):
        # Serialize the transition against a flush that started before transport
        # setup. No transport may start native work until this call returns.
        with self._lock:
            self._suppressed = True

    def empty_cache(self) -> bool:
        with self._lock:
            if self._suppressed:
                return False
            torch.cuda.empty_cache()
            return True

    def after_fork(self):
        # Preserve the conservative policy, not a lock held by a vanished thread.
        self._lock = threading.Lock()


_policy = _CacheCleanupPolicy()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_policy.after_fork)


def suppress_opportunistic_cuda_cache_cleanup() -> None:
    """Disable optional flushing before starting asynchronous transport work.

    This affects only maybe_empty_cuda_cache(), not intentional memory release.
    Process-wide and sticky: failed setup, shutdown, or an empty queue do not
    prove CUDA quiescence. Custom transports must call this before startup.
    It does not wait for, synchronize, cancel, or release transport resources.
    """
    _policy.suppress()


def maybe_empty_cuda_cache() -> bool:
    """Request an opportunistic flush that is always safe to omit.

    Use for optional per-wave/periodic cleanup, not model unloading, checkpoint
    memory handoff, or a request required to give memory to another allocator.
    Return whether PyTorch's flush was called, not whether bytes were released.
    Suppressed requests are skipped, not queued for an implicit later flush.
    Live tensors are unaffected; allocator-cached storage remains reusable.
    Intentional release keeps its direct PyTorch call; its owner must establish
    quiescence first. This helper neither proves that boundary nor makes direct
    calls safe. It does not intercept PyTorch calls or third-party cleanup.
    """
    return _policy.empty_cache()
