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

"""``PayloadTransport`` registration + control-plane cleanup for NCCL.

Background
----------
By default, Cosmos-RL transfers rollout completion payloads (token IDs,
log-probs, etc.) from rollout workers to the controller via Redis
streams.  For workloads with large payloads — such as VLA policies that
produce high-dimensional action tensors — the Redis path becomes a
bottleneck.

**NCCL payload transfer** is an opt-in alternative where the payload
tensors are sent directly between GPUs using NCCL point-to-point
operations.  Redis is still used as the *control plane*: receivers
publish per-transfer requests, senders acknowledge them, unique-IDs are
exchanged for lazy 2-rank communicators, and cleanup messages are sent
when stale transfers are discarded so the producer can free the GPU send
buffer immediately.

How it works (in-tree mixin path)
---------------------------------
1. A data packer that subclasses
   :class:`~cosmos_rl.utils.payload_transport.nccl.data_packer_mixin.NCCLDataPackerMixin`
   exposes ``_setup_nccl_data_packer``.  :meth:`attach_data_packer`
   invokes it with the worker's device, a live Redis client (built from
   ``redis_endpoint``), and the ``config.custom`` tunables.

2. Rollout completions transferred via NCCL have their ``completion``
   field prefixed with ``nccl:`` followed by a transfer ID.

3. When the controller discards outdated rollouts it dispatches to
   :meth:`NcclPayloadTransport.publish_cleanup_for_discarded` (via the
   registry) so the rollout worker's cleanup subscriber releases the
   associated GPU buffers immediately.

Legacy (deprecated) packer path
--------------------------------
Before the in-tree mixin existed, NCCL-aware packers exposed a
``redis_client`` attribute and an optional ``post_redis_injection()``
hook (PR #670 / commit 55745c).  :meth:`attach_data_packer` still honors
that contract as a **deprecated fallback** so downstream forks keep
working; new code should use the ``NCCLDataPackerMixin`` path.

Enabling
--------
Set ``[custom].payload_transfer = "nccl"`` in the experiment config.
The legacy ``[custom].nccl_payload_transfer = true`` boolean still works
as a deprecated alias.
"""

from __future__ import annotations

import functools

import json
import os
import time
from typing import Any, List, Optional

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.payload_transport.nccl.protocol import (
    NCCL_COMPLETION_PREFIX,
    build_cleanup_channel,
    build_nccl_prefix,
    build_rollout_prefix,
    build_transfer_rollout_candidates,
)
from cosmos_rl.utils.payload_transport.registry import (
    PayloadTransport,
    PayloadTransportRegistry,
    RedisEndpoint,
)

try:
    import redis as _redis_lib
except ImportError:  # pragma: no cover - exercised only on systems w/o redis
    _redis_lib = None


def _build_redis_client(redis_endpoint: Optional[RedisEndpoint]) -> Any:
    """Construct + ping a decode_responses Redis client, or return None.

    Shared by both the in-tree mixin path and the legacy fallback.  All
    failure modes (no endpoint, no redis package, ping error) log a
    warning and return ``None`` rather than raising, so a Redis hiccup
    cannot kill worker init.
    """
    if redis_endpoint is None:
        logger.warning(
            "[NcclPayloadTransport] No Redis endpoint provided; "
            "cannot attach NCCL data packer"
        )
        return None
    if _redis_lib is None:
        logger.warning(
            "[NcclPayloadTransport] redis package not installed; "
            "cannot attach NCCL data packer"
        )
        return None
    try:
        client = _redis_lib.Redis(
            host=redis_endpoint.host,
            port=redis_endpoint.port,
            db=redis_endpoint.db,
            decode_responses=True,
        )
        client.ping()
    except Exception as exc:
        logger.warning(
            f"[NcclPayloadTransport] Redis ping failed at "
            f"{redis_endpoint.host}:{redis_endpoint.port}/{redis_endpoint.db}: "
            f"{exc}; skipping NCCL packer attach"
        )
        return None
    return client


class NcclPayloadTransport(PayloadTransport):
    """NCCL-based payload-transport backend.

    Identifies its rollouts by the ``nccl:`` completion prefix and
    publishes cleanup messages on the ``nccl_cleanup`` pub/sub channel
    when the controller discards rollouts.
    """

    name = "nccl"
    completion_prefix = NCCL_COMPLETION_PREFIX

    def attach_data_packer(
        self,
        packer: Any,
        *,
        config: Any,
        device: Any = None,
        redis_endpoint: Optional[RedisEndpoint] = None,
    ) -> None:
        """Wire NCCL-specific state into an NCCL-aware data packer.

        Two paths, tried in order:

        1. **In-tree mixin** (preferred).  Packers subclassing
           ``NCCLDataPackerMixin`` expose ``_setup_nccl_data_packer``;
           it is invoked with ``device``, a live Redis client, and the
           tunables resolved from ``config.custom``:

           * ``nccl_prefetch_timeout`` (float, default 30.0): per-batch
             wait ceiling for the prefetch worker's result queue.
           * ``nccl_read_max_attempts`` (int, default 2): total attempts
             per remote transfer (initial + transient retries).
           * ``nccl_recv_timeout`` (float, default 5.0): per-``nccl_recv``
             wall-clock budget (seconds) so a wedged sender engages
             retry / quarantine quickly.
           * ``nccl_first_transfer_timeout`` (float, default 30.0): larger
             budget for the FIRST (comm-creating) transfer of a pair, which
             must wait out the cold-start comm-init storm; floored at
             ``nccl_recv_timeout``.

        2. **Composed** -- a packer that only schedules (exposes
           ``set_transport_strategy``) gets a strategy built and attached,
           which is how a transport can be chosen from config without
           subclassing a per-transport packer class.

        3. **Legacy fallback** (deprecated).  Packers exposing a
           ``redis_client`` attribute get a live client assigned followed
           by an optional ``post_redis_injection()`` call, in that order
           (PR #670 contract).  Superseded by the mixin path; retained
           only so in-flight downstream forks keep working.

        No-op for packers that expose neither surface.
        """
        setup = getattr(packer, "_setup_nccl_data_packer", None)
        if not callable(setup) and callable(
            getattr(packer, "set_transport_strategy", None)
        ):
            # Composed packer: it schedules (PrefetchDataPackerMixin) but has no
            # NCCL ancestry, so synthesise the same setup signature the mixin
            # exposes.  Reusing _attach_via_mixin verbatim is deliberate -- the
            # tunable resolution below it (cold-start floors, batch hint) is
            # subtle enough that a second copy would drift.
            from cosmos_rl.utils.payload_transport.nccl.strategy import (
                compose_nccl_transport,
            )

            setup = functools.partial(compose_nccl_transport, packer)
        if callable(setup):
            self._attach_via_mixin(
                setup, config=config, device=device, redis_endpoint=redis_endpoint
            )
            return
        if hasattr(packer, "redis_client"):
            self._attach_via_legacy(packer, redis_endpoint=redis_endpoint)

    # ------------------------------------------------------------------
    # In-tree mixin attach (preferred)
    # ------------------------------------------------------------------

    def _attach_via_mixin(
        self,
        setup: Any,
        *,
        config: Any,
        device: Any,
        redis_endpoint: Optional[RedisEndpoint],
    ) -> None:
        client = _build_redis_client(redis_endpoint)
        if client is None:
            # Without a control plane the mixin cannot rendezvous.  RAISE rather
            # than silently no-op'ing: _attach_payload_transport applies the
            # explicit_fatal policy to RuntimeError -- so an EXPLICIT
            # payload_transfer="nccl" with a dead/absent Redis fails loudly
            # (misconfig surfaced immediately), while a non-explicit selection
            # is logged and swallowed there, falling through to the underlying
            # packer (Redis path).  (_build_redis_client already logged why.)
            raise RuntimeError(
                "NCCL payload transport requires a reachable Redis control "
                "plane, but none could be established (no endpoint, redis "
                "package missing, or ping failed) -- cannot attach NCCL data "
                "packer"
            )
        custom = getattr(config, "custom", None) or {}
        try:
            prefetch_timeout = float(custom.get("nccl_prefetch_timeout", 30.0))
        except (TypeError, ValueError):
            prefetch_timeout = 30.0
        try:
            max_attempts = int(custom.get("nccl_read_max_attempts", 2))
        except (TypeError, ValueError):
            max_attempts = 2
        if max_attempts < 1:
            max_attempts = 1
        try:
            recv_timeout = float(custom.get("nccl_recv_timeout", 5.0))
        except (TypeError, ValueError):
            recv_timeout = 5.0
        # Cold-start budget: the FIRST transfer for a pair must wait for the
        # producer to build its side of the 2-rank comm, which under a
        # multi-replica init storm can take tens of seconds -- far longer than
        # the steady-state per-recv budget.  A separate, larger timeout for the
        # comm-creating transfer avoids spuriously cancelling + quarantining a
        # slow-to-warm-up (but healthy) endpoint, which otherwise cascades into
        # skipped steps and a stalled run.
        try:
            first_transfer_timeout = float(
                custom.get("nccl_first_transfer_timeout", 30.0)
            )
        except (TypeError, ValueError):
            first_transfer_timeout = 30.0
        if first_transfer_timeout < recv_timeout:
            first_transfer_timeout = recv_timeout
        # Per-batch prefetch fan-out estimate for timeout sizing.  ``_fetch_all``
        # rendezvouses the batch's refs SEQUENTIALLY, so a cold-start batch of
        # ``B`` first-transfers costs up to ``B x max_attempts x
        # first_transfer_timeout`` -- not just one ref's budget.  The prefetch
        # worker's per-batch wait must cover that, or it expires mid-warm-up and
        # the late result is mis-consumed as a later batch (the FIFO is not
        # batch-id correlated) -> skipped episodes + stale cache.  ``B`` is a
        # runtime property of the rollout batch (not cleanly known here), so it
        # is an operator-tunable hint; default 8 covers small batches and is far
        # above the old implicit 1.
        try:
            batch_hint = int(custom.get("nccl_prefetch_batch_hint", 8))
        except (TypeError, ValueError):
            batch_hint = 8
        if batch_hint < 1:
            batch_hint = 1
        # Floor the wait to cover the full cold-start retry budget for the batch.
        min_prefetch = batch_hint * max_attempts * first_transfer_timeout
        if prefetch_timeout < min_prefetch:
            logger.info(
                "[NcclPayloadTransport] raising prefetch_timeout %.1fs -> %.1fs "
                "to cover the cold-start retry budget (batch_hint=%d x "
                "max_attempts=%d x first_transfer_timeout=%.1fs); set "
                "custom.nccl_prefetch_batch_hint to your prefetch batch size",
                prefetch_timeout,
                min_prefetch,
                batch_hint,
                max_attempts,
                first_transfer_timeout,
            )
            prefetch_timeout = min_prefetch
        logger.debug(
            f"[NcclPayloadTransport] Attaching NCCL data packer "
            f"(device={device}, prefetch_timeout={prefetch_timeout}, "
            f"max_attempts={max_attempts}, recv_timeout={recv_timeout}, "
            f"first_transfer_timeout={first_transfer_timeout})"
        )
        setup(
            device=device,
            redis_client=client,
            config=config,
            prefetch_timeout=prefetch_timeout,
            max_attempts=max_attempts,
            recv_timeout=recv_timeout,
            first_transfer_timeout=first_transfer_timeout,
        )

    # ------------------------------------------------------------------
    # Legacy attach (deprecated; PR #670 / commit 55745c contract)
    # ------------------------------------------------------------------

    def _attach_via_legacy(
        self,
        packer: Any,
        *,
        redis_endpoint: Optional[RedisEndpoint],
    ) -> None:
        logger.warning(
            "[NcclPayloadTransport] Attaching NCCL data packer via the "
            "deprecated redis_client/post_redis_injection path; migrate to "
            "NCCLDataPackerMixin (_setup_nccl_data_packer)."
        )
        client = _build_redis_client(redis_endpoint)
        if client is None:
            return

        # Step 1: assign client. Downstream code reads this attribute
        # directly; assignment must complete before step 2.
        packer.redis_client = client

        # Step 2: call the post-injection hook so the packer can do any
        # deferred setup (NCCL communicator wiring, channel subscription,
        # etc.) that depends on the live client.
        hook = getattr(packer, "post_redis_injection", None)
        if callable(hook):
            try:
                hook()
            except Exception as exc:
                # Hook errors are logged but not re-raised so a buggy
                # downstream packer cannot kill worker init.
                logger.warning(
                    f"[NcclPayloadTransport] post_redis_injection raised "
                    f"{type(exc).__name__}: {exc}; continuing"
                )

    # ------------------------------------------------------------------
    # Controller-side discard cleanup
    # ------------------------------------------------------------------

    def publish_cleanup_for_discarded(
        self,
        *,
        transfer_ids: List[str],
        config: Any,
        redis_client: Any,
    ) -> int:
        if not transfer_ids:
            return 0
        if redis_client is None:
            return 0

        experiment_name = "default"
        try:
            experiment_name = config.logging.experiment_name
        except AttributeError:
            pass
        job_id = os.environ.get("SLURM_JOB_ID", "test")
        prefix = build_nccl_prefix(experiment_name=experiment_name, job_id=job_id)

        published = 0
        max_retries = 3
        for transfer_id in transfer_ids:
            try:
                rollout_indices = build_transfer_rollout_candidates(
                    transfer_id=transfer_id
                )
                for rollout_idx in rollout_indices:
                    channel = build_cleanup_channel(
                        build_rollout_prefix(prefix, rollout_idx)
                    )
                    payload = json.dumps({"transfer_id": transfer_id})
                    for attempt in range(max_retries):
                        try:
                            redis_client.publish(channel, payload)
                            break
                        except Exception:
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(0.1 * (attempt + 1))
                published += 1
            except Exception as e:
                logger.warning(
                    f"[NcclPayloadTransport] Failed to publish cleanup for "
                    f"transfer_id={transfer_id}: {e}"
                )
        return published


PayloadTransportRegistry.register_class(NcclPayloadTransport)


__all__ = ["NcclPayloadTransport"]
