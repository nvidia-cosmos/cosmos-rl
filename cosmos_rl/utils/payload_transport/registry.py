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

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type

from cosmos_rl.utils.logging import logger


# Canonical TOML keys read from the experiment ``[custom]`` section.
PAYLOAD_TRANSFER_KEY = "payload_transfer"        # preferred string form
LEGACY_NCCL_KEY = "nccl_payload_transfer"        # deprecated boolean alias

# Active backend when neither key is present.
DEFAULT_TRANSFER_MODE = "redis"


class PayloadTransport(ABC):
    """Abstract base class for payload-transport backends.

    Subclasses encapsulate everything cosmos-rl's controller and worker
    code need to know about a non-default transport:

    * ``name``: backend identifier matched against the
      ``payload_transfer`` config string.
    * ``completion_prefix``: optional sentinel prefix attached to the
      ``completion`` field of rollouts that travel via this transport.
      ``None`` for the implicit Redis default.
    * :meth:`prepare_data_packer`: hook invoked by ``CommMixin`` after
      the worker injects a Redis client into the packer.
    * :meth:`publish_cleanup_for_discarded`: invoked by the controller
      when outdated rollouts that bear ``completion_prefix`` are
      discarded so the backend can release any reserved resources.

    Backends that do not need a particular hook may inherit the no-op
    default implementation.
    """

    name: str = ""
    completion_prefix: Optional[str] = None

    # ------------------------------------------------------------------
    # Worker-side hooks
    # ------------------------------------------------------------------

    def prepare_data_packer(self, packer: Any) -> None:
        """Hook for transports that need per-packer initialization.

        Called by ``CommMixin._inject_redis_into_data_packers`` after the
        worker successfully injects its Redis client into ``packer``.
        Default: no-op.
        """

    # ------------------------------------------------------------------
    # Controller-side hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def publish_cleanup_for_discarded(
        self,
        *,
        transfer_ids: List[str],
        config: Any,
        redis_client: Any,
    ) -> int:
        """Publish cleanup messages for discarded rollouts.

        Args:
            transfer_ids: List of opaque transport-side transfer ids
                extracted from discarded rollouts (with
                :attr:`completion_prefix` already stripped).
            config: The cosmos-rl :class:`~cosmos_rl.policy.config.Config`.
            redis_client: A connected Redis client.

        Returns:
            Number of cleanup messages successfully published.  Should
            return ``0`` for transports without a cleanup channel.
        """


class _RedisDefaultTransport(PayloadTransport):
    """Sentinel transport for the implicit Redis default.

    Redis is the data plane *and* the control plane for the default
    transfer path — there is nothing extra to publish on cleanup.
    """

    name = "redis"
    completion_prefix = None

    def publish_cleanup_for_discarded(
        self,
        *,
        transfer_ids: List[str],
        config: Any,
        redis_client: Any,
    ) -> int:
        return 0


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
            raise ValueError(
                "PayloadTransport must declare a non-empty name attribute"
            )
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
            f"[custom].{PAYLOAD_TRANSFER_KEY} = \"nccl\" instead."
        )
        return "nccl"

    return DEFAULT_TRANSFER_MODE


__all__ = [
    "DEFAULT_TRANSFER_MODE",
    "LEGACY_NCCL_KEY",
    "PAYLOAD_TRANSFER_KEY",
    "PayloadTransport",
    "PayloadTransportRegistry",
    "get_payload_transfer_mode",
]
