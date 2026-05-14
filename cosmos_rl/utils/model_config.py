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

"""Pluggable model-config loader for non-HuggingFace local model paths.

The default cosmos-rl flow assumes ``policy.model_name_or_path`` is a
HuggingFace repo id or a local directory containing a ``config.json``.
Non-text RL workloads (e.g. Gymnasium-based tasks with a small MLP
policy) often want to point cosmos-rl at a local TOML/YAML file
describing the architecture instead.

This module provides:

* :func:`register_local_model_config` — register a (predicate, factory)
  pair so callers can produce a HuggingFace-compatible
  :class:`~transformers.PretrainedConfig` from a non-HF path.
* :func:`load_model_config` — central entrypoint that consults the
  registry first and falls back to
  :meth:`transformers.AutoConfig.from_pretrained` otherwise.

Existing cosmos-rl call sites that directly invoke
``AutoConfig.from_pretrained`` continue to work unchanged.  New code (and
the next round of upstream migrations) should prefer
:func:`load_model_config` so non-HF model paths route through the
registry.

Example
-------

.. code-block:: python

    from transformers import PretrainedConfig
    from cosmos_rl.utils.model_config import (
        register_local_model_config,
        load_model_config,
    )

    class GymPolicyConfig(PretrainedConfig):
        model_type = "gym_policy"

    register_local_model_config(
        predicate=lambda path: path.endswith(".toml"),
        factory=lambda path: GymPolicyConfig.from_dict(_parse_toml(path)),
    )

    cfg = load_model_config("/path/to/cartpole_mlp.toml")
    assert cfg.model_type == "gym_policy"
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

from transformers import AutoConfig, PretrainedConfig

from cosmos_rl.utils.logging import logger


_CUSTOM_MODEL_CONFIG_LOADERS: List[
    Tuple[Callable[[str], bool], Callable[[str], PretrainedConfig]]
] = []


def register_local_model_config(
    predicate: Callable[[str], bool],
    factory: Callable[[str], PretrainedConfig],
) -> None:
    """Register a custom model-config factory.

    Predicates are evaluated in registration order; the first one that
    returns truthy for ``model_name_or_path`` wins, and its factory
    produces the :class:`~transformers.PretrainedConfig` instance.
    """
    _CUSTOM_MODEL_CONFIG_LOADERS.append((predicate, factory))


def clear_local_model_configs() -> None:
    """Remove all registered local-model-config factories.  Useful for tests."""
    _CUSTOM_MODEL_CONFIG_LOADERS.clear()


def load_model_config(
    model_name_or_path: str,
    **auto_config_kwargs: Any,
) -> PretrainedConfig:
    """Load a model config, preferring the local-model-config registry.

    Args:
        model_name_or_path: HuggingFace repo id, local model directory,
            or any path matched by a registered custom predicate.
        **auto_config_kwargs: Forwarded to
            :meth:`transformers.AutoConfig.from_pretrained` when no
            custom loader matches.  Always passes
            ``trust_remote_code=True`` by default for parity with
            existing cosmos-rl call sites.

    Returns:
        A :class:`PretrainedConfig` instance with a meaningful
        ``model_type`` attribute that downstream registries
        (``WeightMapperRegistry``, ``ModelRegistry``) can key on.
    """
    for predicate, factory in _CUSTOM_MODEL_CONFIG_LOADERS:
        try:
            matched = bool(predicate(model_name_or_path))
        except Exception as e:
            logger.warning(
                f"Local model-config predicate raised for {model_name_or_path}: {e}; skipping"
            )
            continue
        if matched:
            logger.info(
                f"Model config for {model_name_or_path} resolved via registered local loader"
            )
            return factory(model_name_or_path)

    auto_config_kwargs.setdefault("trust_remote_code", True)
    return AutoConfig.from_pretrained(model_name_or_path, **auto_config_kwargs)


__all__ = [
    "clear_local_model_configs",
    "load_model_config",
    "register_local_model_config",
]
