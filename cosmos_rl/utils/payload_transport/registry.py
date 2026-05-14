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

"""Payload-transport ABC + registry shared by all backends."""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Type

from cosmos_rl.utils.logging import logger


# Canonical TOML keys read from the experiment ``[custom]`` section.
PAYLOAD_TRANSFER_KEY = "payload_transfer"  # preferred string form
LEGACY_NCCL_KEY = "nccl_payload_transfer"  # deprecated boolean alias

# Active backend when neither key is present.
DEFAULT_TRANSFER_MODE = "redis"


class RedisEndpoint(NamedTuple):
    """Connection coordinates for a Redis server.

    Passed to :meth:`PayloadTransport.attach_data_packer` so backends
    that need a Redis control plane (notably NCCL) can construct their
    own client without re-deriving the endpoint.
    """

    host: str
    port: int
    db: int = 0


class PayloadTransport(ABC):
    """Abstract base class for payload-transport backends.

    Subclasses encapsulate everything cosmos-rl's controller and worker
    code need to know about a non-default transport:

    * ``name``: backend identifier matched against the
      ``payload_transfer`` config string.
    * ``completion_prefix``: optional sentinel prefix attached to the
      ``completion`` field of rollouts that travel via this transport.
      ``None`` for the implicit Redis default and for transports that do
      not participate in controller-side discard cleanup.
    * :meth:`attach_data_packer`: worker-side hook invoked by
      ``CommMixin._attach_payload_transport`` once per data packer to
      give the backend a chance to wire transport-specific state into
      the packer (e.g. inject a Redis client, allocate ring buffers,
      start prefetch threads).
    * :meth:`publish_cleanup_for_discarded`: controller-side hook
      invoked when outdated rollouts that bear :attr:`completion_prefix`
      are discarded so the backend can release any reserved resources.

    Both hooks have safe no-op defaults; backends override only what
    they need.
    """

    name: str = ""
    completion_prefix: Optional[str] = None

    # ------------------------------------------------------------------
    # Worker-side hooks
    # ------------------------------------------------------------------

    def attach_data_packer(
        self,
        packer: Any,
        *,
        config: Any,
        device: Any = None,
        redis_endpoint: Optional[RedisEndpoint] = None,
    ) -> None:
        """Worker-side hook to wire transport state into a data packer.

        Called once per data packer from
        ``CommMixin._attach_payload_transport`` after the packer has
        been created and ``setup()`` has been invoked.  Default: no-op
        (Redis-default transport, and any other transport that needs no
        per-packer state).

        Args:
            packer: The data packer instance to attach to.
            config: The cosmos-rl
                :class:`~cosmos_rl.policy.config.Config` for this run.
            device: Optional torch device (or device-like) the packer
                will run on.  Useful for transports that pre-allocate
                GPU memory (e.g. UCXX prefetch buffers).
            redis_endpoint: Optional :class:`RedisEndpoint` describing
                the worker's Redis connection.  Transports that need a
                Redis control plane (NCCL) construct their own client
                from this; transports that do not (UCXX, redis default)
                ignore it.
        """

    # ------------------------------------------------------------------
    # Controller-side hooks
    # ------------------------------------------------------------------

    def publish_cleanup_for_discarded(
        self,
        *,
        transfer_ids: List[str],
        config: Any,
        redis_client: Any,
    ) -> int:
        """Publish cleanup messages for discarded rollouts.

        Default: no-op returning ``0``.  Override only when the
        transport holds resources on the producer side that the
        controller must signal to release (e.g. NCCL's pinned GPU
        send buffers).  Transports that auto-recycle on the producer
        side (UCXX SHM ring buffers, Redis streams) inherit the no-op.

        Args:
            transfer_ids: List of opaque transport-side transfer ids
                extracted from discarded rollouts (with
                :attr:`completion_prefix` already stripped).
            config: The cosmos-rl :class:`~cosmos_rl.policy.config.Config`.
            redis_client: A connected Redis client.

        Returns:
            Number of cleanup messages successfully published.
        """
        return 0


class _RedisDefaultTransport(PayloadTransport):
    """Sentinel transport for the implicit Redis default.

    Redis is the data plane *and* the control plane for the default
    transfer path — there is nothing extra to publish on cleanup, and
    no per-packer attachment is needed.  Inherits the no-op defaults
    for both :meth:`attach_data_packer` and
    :meth:`publish_cleanup_for_discarded`.
    """

    name = "redis"
    completion_prefix = None


class PayloadTransportRegistry:
    """Singleton-style registry for :class:`PayloadTransport` backends.

    Backends register at import time (typically in their submodule's
    top-level body).  Two lookup helpers are provided:

    * :meth:`get` — look up a single backend by name (raises if not
      registered, since misconfiguration should fail loudly).
    * :meth:`active_for_completion` — given a completion string,
      return the matching backend by ``completion_prefix``.  Used by the
      controller to dispatch cleanup for rollouts it discards.
    """

    _registry: Dict[str, PayloadTransport] = {}

    @classmethod
    def register(cls, transport: PayloadTransport) -> None:
        if not transport.name:
            raise ValueError("PayloadTransport must declare a non-empty name attribute")
        existing = cls._registry.get(transport.name)
        if existing is not None and existing is not transport:
            logger.debug(
                f"PayloadTransport '{transport.name}' re-registered; "
                "replacing previous instance"
            )
        cls._registry[transport.name] = transport

    @classmethod
    def register_class(cls, transport_cls: Type[PayloadTransport]) -> None:
        """Convenience: instantiate ``transport_cls`` with no args and register."""
        cls.register(transport_cls())

    @classmethod
    def register_default_redis(cls) -> None:
        """Idempotently register the implicit Redis sentinel."""
        if "redis" not in cls._registry:
            cls._registry["redis"] = _RedisDefaultTransport()

    @classmethod
    def get(cls, name: str) -> PayloadTransport:
        try:
            return cls._registry[name]
        except KeyError as e:
            raise KeyError(
                f"PayloadTransport '{name}' is not registered. "
                f"Available: {sorted(cls._registry)}"
            ) from e

    @classmethod
    def get_optional(cls, name: str) -> Optional[PayloadTransport]:
        return cls._registry.get(name)

    @classmethod
    def all_with_completion_prefix(cls) -> Iterable[PayloadTransport]:
        """Iterate over registered backends that mark their completions."""
        for transport in cls._registry.values():
            if transport.completion_prefix:
                yield transport

    @classmethod
    def active_for_completion(cls, completion: Any) -> Optional[PayloadTransport]:
        """Return the backend whose ``completion_prefix`` matches ``completion``.

        Returns ``None`` for non-string completions, completions without
        a registered prefix, or when no backend has a prefix attribute.
        """
        if not isinstance(completion, str):
            return None
        for transport in cls.all_with_completion_prefix():
            if completion.startswith(transport.completion_prefix):
                return transport
        return None

    @classmethod
    def reset(cls) -> None:
        """Wipe the registry.  Useful for tests."""
        cls._registry.clear()

    @classmethod
    def handle_discarded(
        cls,
        rollouts: List[Any],
        filtered: List[Any],
        *,
        config: Any,
        redis_client: Any,
    ) -> int:
        """Centralized controller-side cleanup dispatch for discards.

        Replaces the inline grouping logic that previously lived in
        ``PolicyStatusManager._publish_payload_transport_cleanup``.
        Identifies which discarded rollouts (those in ``rollouts`` but
        not in ``filtered``) belong to which transport via
        :meth:`active_for_completion`, then invokes each matched
        transport's :meth:`PayloadTransport.publish_cleanup_for_discarded`.

        Transports with ``completion_prefix=None`` (e.g. UCXX, the
        default Redis transport) are skipped automatically because
        :meth:`active_for_completion` only matches non-None prefixes.

        Failures in one transport are isolated — the remaining
        transports still get a chance to publish cleanup.

        Args:
            rollouts: All rollouts considered for filtering this batch.
            filtered: Subset of ``rollouts`` that survived filtering.
                Rollouts in ``rollouts`` but not in ``filtered`` are the
                "discarded" set this method publishes cleanup for.
            config: The cosmos-rl
                :class:`~cosmos_rl.policy.config.Config` for the run.
            redis_client: A connected Redis client used by transports
                whose cleanup channel rides on Redis pub/sub.  Pass
                ``None`` for transports that do not need it.

        Returns:
            Total number of cleanup messages successfully published
            across all transports.  Returns ``0`` if there are no
            discards or no rollouts match a registered prefix.
        """
        if not rollouts or len(filtered) == len(rollouts):
            return 0

        kept_ids = {id(r) for r in filtered}
        per_transport: Dict[PayloadTransport, List[str]] = {}
        for rollout in rollouts:
            if id(rollout) in kept_ids:
                continue
            completion = getattr(rollout, "completion", None)
            transport = cls.active_for_completion(completion)
            if transport is None:
                continue
            transfer_id = completion[len(transport.completion_prefix) :]
            per_transport.setdefault(transport, []).append(transfer_id)

        if not per_transport:
            return 0

        total_published = 0
        for transport, transfer_ids in per_transport.items():
            try:
                published = transport.publish_cleanup_for_discarded(
                    transfer_ids=transfer_ids,
                    config=config,
                    redis_client=redis_client,
                )
            except Exception as exc:
                # Isolate failures: a buggy/unresponsive transport
                # should not prevent other transports from publishing
                # cleanup for their own discards.
                logger.debug(
                    f"[PayloadTransportRegistry] {transport.name} cleanup "
                    f"raised {type(exc).__name__}: {exc}"
                )
                continue
            if published:
                total_published += published
                logger.debug(
                    f"[PayloadTransportRegistry] Published {transport.name} "
                    f"cleanup for {published}/{len(transfer_ids)} discarded transfers"
                )

        return total_published


# ---------------------------------------------------------------------------
# Config-side helpers
# ---------------------------------------------------------------------------


def get_payload_transfer_mode(config: Any) -> str:
    """Return the active payload-transfer mode for ``config``.

    Resolution order:

    1. ``config.custom.payload_transfer`` (string).  Authoritative when
       set.  Validated against the registry; misspellings fail fast.
    2. ``config.custom.nccl_payload_transfer`` (boolean).  **Deprecated**
       alias retained for backward compatibility — ``True`` resolves to
       ``"nccl"``, ``False`` to the default.
    3. :data:`DEFAULT_TRANSFER_MODE` (``"redis"``).

    Args:
        config: A cosmos-rl :class:`~cosmos_rl.policy.config.Config`-like
            object whose ``custom`` attribute is a mapping.

    Returns:
        The resolved transport name.
    """
    custom = getattr(config, "custom", None) or {}
    try:
        explicit = custom.get(PAYLOAD_TRANSFER_KEY)
    except AttributeError:
        explicit = None

    if isinstance(explicit, str) and explicit:
        normalized = explicit.strip().lower()
        if PayloadTransportRegistry.get_optional(normalized) is None:
            raise ValueError(
                f"[custom].{PAYLOAD_TRANSFER_KEY}={explicit!r} is not a "
                f"registered payload transport. Available: "
                f"{sorted(PayloadTransportRegistry._registry)}"
            )
        return normalized

    try:
        legacy = custom.get(LEGACY_NCCL_KEY)
    except AttributeError:
        legacy = None

    if legacy:
        logger.warning(
            f"[custom].{LEGACY_NCCL_KEY} is deprecated; use "
            f'[custom].{PAYLOAD_TRANSFER_KEY} = "nccl" instead.'
        )
        return "nccl"

    return DEFAULT_TRANSFER_MODE


def is_payload_transfer_mode_explicit(config: Any) -> bool:
    """Return True iff the user explicitly selected a transport in config.

    "Explicit" means either ``[custom].payload_transfer = "<name>"`` or
    the legacy ``[custom].nccl_payload_transfer = true`` is set.  When
    neither is present, :func:`get_payload_transfer_mode` falls back to
    :data:`DEFAULT_TRANSFER_MODE` and this function returns ``False``.

    Used by ``CommMixin._attach_payload_transport`` to decide whether
    a transport-attach failure should be fatal (user opted in and
    expects it to work) or merely a warning (transport happened to be
    registered passively).
    """
    custom = getattr(config, "custom", None) or {}
    try:
        explicit = custom.get(PAYLOAD_TRANSFER_KEY)
    except AttributeError:
        explicit = None
    if isinstance(explicit, str) and explicit.strip():
        return True
    try:
        legacy = custom.get(LEGACY_NCCL_KEY)
    except AttributeError:
        legacy = None
    return bool(legacy)


__all__ = [
    "DEFAULT_TRANSFER_MODE",
    "LEGACY_NCCL_KEY",
    "PAYLOAD_TRANSFER_KEY",
    "PayloadTransport",
    "PayloadTransportRegistry",
    "RedisEndpoint",
    "get_payload_transfer_mode",
    "is_payload_transfer_mode_explicit",
]
