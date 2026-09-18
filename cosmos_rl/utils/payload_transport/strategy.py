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

"""Transport behaviour as a composed object rather than a subclass mixin.

``PrefetchDataPackerMixin`` owns *scheduling* -- the background worker, the
double-buffered cache, the early-ack handshake -- and delegates every
*transport-specific* decision to six hooks.  Historically those hooks were
supplied by mixing a transport class into the packer, which forces the choice
to compile time: a consumer must pick ``NCCLSimpleRLDataPacker`` or
``UCXXSimpleRLDataPacker`` before it has read any config.

This module turns that same six-hook contract into an object the packer
*holds*, so the transport can be chosen from config at runtime like every other
knob.  The hook set is identical in shape to the mixin's, deliberately: the two
paths are the same contract expressed two ways, and the mixin path stays
supported (see :class:`PrefetchDataPackerMixin` for how the two interact).

Naming drops the leading underscore -- these are the strategy's public API,
whereas the mixin's ``_``-prefixed methods are protected members of the packer.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

__all__ = ["PayloadTransportStrategy"]


class PayloadTransportStrategy(ABC):
    """The transport-shaped half of a prefetching data packer.

    Implementations own everything about moving a payload -- connections,
    buffers, rendezvous, telemetry -- and nothing about *when* it happens.

    Lifetime is ``setup()`` -> many fetches -> ``shutdown()``.  The scheduling
    layer calls :meth:`fetch_batch` on a background thread and everything else
    on the caller's thread, so implementations must make any state shared
    between the two safe (the two shipped strategies keep their counters
    single-writer for this reason).
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, **kwargs: Any) -> None:
        """Acquire connections/buffers.  Keyword-only; transports differ wildly.

        NCCL needs a redis client, a comm cache, rendezvous state and CUDA
        streams; UCXX needs only a client.  Rather than invent a lowest common
        denominator, each implementation documents its own keywords.
        """
        return None

    def shutdown(self) -> None:
        """Release everything acquired by :meth:`setup`.  Must be idempotent."""
        return None

    def before_join(self) -> None:
        """Force in-flight I/O to fail fast, called during teardown.

        The scheduling layer sets its shutdown event, calls this, and only then
        joins the worker.  A worker parked inside :meth:`fetch_batch` observes
        the event only *between* batches, so without this the join would wait
        out a transfer that only the transport can unwedge -- NCCL aborts its
        comms here, UCXX closes its client.  Default: nothing to unwedge.
        """
        return None

    # ------------------------------------------------------------------
    # Wire-format recognition
    # ------------------------------------------------------------------

    @abstractmethod
    def should_intercept(self, rollout_output: Any) -> bool:
        """Return True if ``rollout_output`` is a reference this transport owns.

        Must be cheap and total: it runs on every rollout, including payloads
        belonging to other transports, and must not raise on shapes it does not
        recognise.
        """

    @abstractmethod
    def cache_key(self, rollout_output: Any) -> str:
        """Stable key for the payload behind ``rollout_output``.

        Only called where :meth:`should_intercept` returned True.  Must agree
        with the keys :meth:`fetch_batch` returns, or every fetch reports a
        cache miss and silently degrades to :meth:`sync_fetch`.
        """

    def filter_prefetch_tasks(self, rollouts: List[Any]) -> List[Any]:
        """Pick the prefetchable subset of a batch as ``(idx, ref)`` pairs.

        ``idx`` is opaque to the scheduling layer and simply propagates to
        :meth:`fetch_batch` so implementations can correlate results with their
        source slot.  The default -- everything :meth:`should_intercept`
        accepts -- is what both shipped transports want; overriding it to
        anything narrower means the excluded rollouts take the synchronous
        path on every step, which is a permanent slowdown rather than a
        one-time cost.
        """
        tasks: List[Any] = []
        for i, rollout in enumerate(rollouts):
            ro = rollout.completion if hasattr(rollout, "completion") else rollout
            if self.should_intercept(ro):
                tasks.append((i, ro))
        return tasks

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_batch(self, tasks: List[Any]) -> Dict[str, Any]:
        """Resolve ``tasks`` in bulk; return ``{cache_key: payload}``.

        Runs on the background prefetch thread.  Raising is permitted and is
        how :meth:`before_join` unblocks teardown; the scheduling layer
        contains the exception and the batch resolves as misses.
        """

    def sync_fetch(self, rollout_output: Any) -> Optional[Any]:
        """Blocking single-reference fallback used on a cache miss.

        Returning ``None`` makes the packer skip that episode, which is the
        safe default.  Implementations must not raise: this runs inside
        ``get_policy_input`` with no containment above it, so an escaping
        rendezvous error would abort the training step rather than degrade to
        the packer's fallback.
        """
        return None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def on_prefetch_complete(
        self,
        batch_id: int,
        n_results: int,
        fetch_ms: float,
        step: int,
    ) -> None:
        """Called after each completed prefetch populates the cache.

        ``step`` is the scheduling layer's iteration counter, passed in rather
        than read off the packer so implementations stay independent of it.
        """
        return None

    def on_resolve_failed(self, rollout_output: Any, cache_key: str) -> None:
        """Called when both the cache and :meth:`sync_fetch` came up empty.

        The scheduling layer already logs a warning; this exists for
        transport-specific counters.
        """
        return None
