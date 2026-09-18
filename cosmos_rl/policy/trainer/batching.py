# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit rollout collection versus expanded training-sample contracts."""

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import torch
import numpy as np
import torch.distributed as dist
from cosmos_rl.utils.logging import logger


@dataclass(frozen=True)
class FixedRolloutBatching:
    """One collected completion is one training sample; retain startup checks."""


@dataclass(frozen=True)
class ExpandedSampleBatching:
    """Opt in to a shared schedule with empty local contributions."""

    partial_tail: Literal["include", "reject"] = "reject"

    def __post_init__(self):
        if self.partial_tail not in ("include", "reject"):
            raise ValueError("Unsupported partial-tail policy")


@dataclass(frozen=True)
class ExpandedTrainingBatch:
    """Actual ordered minibatches, each an indexable sequence of training samples.

    Do not pass rollout handles or lazy iterators here. Expansion must not
    perform training collectives or update optimizer/scheduler state.
    """

    minibatches: tuple[Sequence, ...]
    global_sample_counts: tuple[int, ...] = ()
    mu_iterations: int = 1

    def mean_gradient_scale(self, index, world_size):
        """Scale a local SUM loss when the trainer averages gradients across ranks."""
        return world_size / self.global_sample_counts[index]


class RecoverablePreparationError(Exception):
    """Unavailable/bad rollout data; participate with empty local contributions.

    Do not wrap programming errors, CUDA errors or failed collectives in this.
    """


def _finite(value):
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, np.floating):
        return bool(np.isfinite(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _describe(batch, mini_batch):
    if not isinstance(batch, ExpandedTrainingBatch):
        raise TypeError("prepare_training_batch must return ExpandedTrainingBatch")
    if type(mini_batch) is not int or mini_batch <= 0:
        raise ValueError("Training mini_batch must be a positive sample count")
    sizes = tuple(len(samples) for samples in batch.minibatches)
    if any(size > mini_batch for size in sizes):
        raise ValueError("Expanded minibatch exceeds configured sample count")
    return sizes


def run_training_step(trainer, *, before_step=None, **kwargs):
    """Worker entrypoint enforcing the declared contract, not a boolean bypass.

    One metadata exchange per expanded update agrees the variable schedule, not
    one exchange per minibatch. Default process groups are replica-local; the
    supported pure-DP topology couples every rank in that group.
    """
    contract = getattr(trainer, "batching_contract", FixedRolloutBatching())
    if isinstance(contract, FixedRolloutBatching):
        if before_step is not None:
            before_step()
        return trainer.step_training(**kwargs)
    if not isinstance(contract, ExpandedSampleBatching):
        raise TypeError("Unknown trainer batching contract")

    policy = trainer.config.train.train_policy
    mu = getattr(policy, "mu_iterations", None)
    dropped = 0
    recovery = None
    try:
        if type(mu) is not int or mu < 1:
            raise ValueError("mu_iterations must be a positive integer")
        try:
            batch = trainer.prepare_training_batch(kwargs["rollouts"])
        except RecoverablePreparationError as error:
            batch = ExpandedTrainingBatch(())
            recovery = str(error)[:512]
            logger.warning(
                "Expanded preparation unavailable; contributing zero: %s", recovery
            )
        _describe(batch, policy.mini_batch)
        cleaned = []
        for samples in batch.minibatches:
            valid = tuple(sample for sample in samples if _finite(sample))
            if contract.partial_tail == "reject" and len(valid) < policy.mini_batch:
                valid = ()
            dropped += len(samples) - len(valid)
            cleaned.append(valid)
        batch = ExpandedTrainingBatch(tuple(cleaned))
        local = {"sizes": tuple(map(len, cleaned)), "mu": mu, "error": None}
    except Exception as error:
        local = {
            "sizes": (),
            "mu": mu,
            "error": f"{type(error).__name__}: {error}"[:512],
        }
    plans = [local]
    if dist.is_initialized():
        plans = [None] * dist.get_world_size()
        dist.all_gather_object(plans, local)
    errors = [(rank, plan["error"]) for rank, plan in enumerate(plans) if plan["error"]]
    if errors:
        raise ValueError(f"Expanded training preflight failed on ranks: {errors}")
    if len({plan["mu"] for plan in plans}) != 1:
        raise ValueError("Expanded training ranks disagree on configured mu_iterations")
    width = max(len(plan["sizes"]) for plan in plans)
    counts = tuple(
        sum(plan["sizes"][i] if i < len(plan["sizes"]) else 0 for plan in plans)
        for i in range(width)
    )
    active = [i for i, count in enumerate(counts) if count]
    batch = ExpandedTrainingBatch(
        tuple(
            batch.minibatches[i] if i < len(batch.minibatches) else () for i in active
        ),
        tuple(counts[i] for i in active),
        mu,
    )
    if active and before_step is not None:
        before_step()
    expanded_kwargs = {key: value for key, value in kwargs.items() if key != "rollouts"}
    # Even an all-empty plan reaches the trainer for checkpoint/control work,
    # but has zero training slots and must not advance optimizer or scheduler.
    result = trainer.step_expanded_training(batch, **expanded_kwargs)
    result.update(
        {
            "batching/skipped_update": int(not active),
            "batching/dropped_samples": dropped,
            "batching/preparation_failed": int(recovery is not None),
        }
    )
    return result
