# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application annotation and final reward-reader boundaries."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from cosmos_rl.dispatcher.data.diagnostics import CompletionDiagnostics
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.application_callbacks import (
    CallbackContext,
    CallbackRegistry,
    _invoke_sync,
)


def initialize_reward_callbacks(owner, hook_fns=None, context=None):
    """Bind registrations independently of the application's data packer."""
    context = context or CallbackContext(role="reward")
    owner.reward_callbacks = CallbackRegistry(
        hook_fns, replace(context, is_validation=False)
    )
    owner.validation_reward_callbacks = CallbackRegistry(
        hook_fns, replace(context, is_validation=True)
    )
    for registry in (owner.reward_callbacks, owner.validation_reward_callbacks):
        registry.validate("reward_annotation", borrowed=True)
        registry.validate("reward_read_complete", borrowed=True, required=True)


def reward_callbacks_for(owner, is_validation):
    return getattr(
        owner,
        "validation_reward_callbacks" if is_validation else "reward_callbacks",
        None,
    )


def close_reward_callbacks(owner):
    first_error = None
    for name in ("reward_callbacks", "validation_reward_callbacks"):
        registry = getattr(owner, name, None)
        if registry is not None:
            try:
                registry.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise first_error


@dataclass(frozen=True)
class RewardResult:
    """Borrowed read-only native result in original completion order.

    ``selected_indices`` describes native admission, before whole-group dynamic
    sampling. Rejected members have no advantage. Validation selects all slots.
    Report values remain producer-owned; annotations must detach their metadata.
    """

    is_validation: bool
    selected_indices: tuple[int, ...]
    rewards: tuple[float, ...]
    filter_rewards: tuple[float, ...]
    advantages: tuple[float | None, ...]
    report_metrics: tuple[Mapping[str, Any], ...]
    group_excluded: bool


def annotate_reward_result(
    packer,
    payload,
    admission,
    rewards,
    filter_rewards,
    advantages,
    metrics,
    *,
    is_validation,
    callbacks=None,
):
    if callbacks is not None and callbacks.has("reward_annotation"):

        def callback(*args):
            return callbacks.invoke("reward_annotation", *args, borrowed=True)
    else:
        callback = getattr(packer, "annotate_reward_result", None)
    if callback is None:
        return
    selected = () if admission.group_excluded else tuple(admission.eligible_indices)
    original_advantages = [None] * admission.original_size
    for index, advantage in zip(selected, advantages):
        original_advantages[index] = advantage
    result = RewardResult(
        is_validation=is_validation,
        selected_indices=selected,
        rewards=tuple(rewards),
        filter_rewards=tuple(filter_rewards),
        advantages=tuple(original_advantages),
        report_metrics=tuple(MappingProxyType(dict(item or {})) for item in metrics),
        group_excluded=admission.group_excluded,
    )
    annotations = _invoke_sync(callback, payload, result)
    if annotations is None:
        return
    if type(annotations) is not list or len(annotations) != admission.original_size:
        raise ValueError(
            "Reward annotations must contain one dictionary per original slot"
        )
    diagnostics = [
        CompletionDiagnostics(original_index=index, metadata=metadata)
        for index, metadata in enumerate(annotations)
    ]
    # Bound the total per-group diagnostic envelope as well as each member.
    if sum(len(item.model_dump_json().encode("utf-8")) for item in diagnostics) > 65536:
        raise ValueError("Reward annotations exceed 65536 bytes per group")
    payload.completion_diagnostics = diagnostics


@contextmanager
def reward_read_scope(packer, payloads, *, is_validation, callbacks=None):
    """Finish every submitted group's application reads, preserving first error.

    Cleanup must synchronously finish application readers and release only its
    own reward views. Policy and transport references have independent owners.
    """
    payloads = tuple(payloads)
    error = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        finish_reward_reads(
            packer,
            payloads,
            is_validation=is_validation,
            error=error,
            callbacks=callbacks,
        )


def finish_reward_reads(packer, payloads, *, is_validation, error, callbacks=None):
    """Notify terminal/canceled reads without replacing a primary error."""
    readers = []
    if callbacks is not None and callbacks.has("reward_read_complete"):

        def callback(*args, **kwargs):
            return callbacks.invoke(
                "reward_read_complete", *args, borrowed=True, required=True, **kwargs
            )

        readers.append(callback)
    packer_cleanup = getattr(packer, "on_reward_read_complete", None)
    if packer_cleanup is not None:
        readers.append(packer_cleanup)
    cleanup_error = None
    for payload in payloads:
        # Registered application readers finish before packer-owned reward
        # views retire. Each callback releases only its own reader references.
        for callback in readers:
            try:
                _invoke_sync(
                    callback, payload, is_validation=is_validation, error=error
                )
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.error("Application reward-read cleanup failed", exc_info=True)
    if error is None and cleanup_error is not None:
        raise cleanup_error
    return cleanup_error
