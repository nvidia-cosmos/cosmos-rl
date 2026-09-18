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


@dataclass(frozen=True)
class FixedRolloutBatching:
    """One collected completion is one training sample; retain startup checks."""


@dataclass(frozen=True)
class ExpandedSampleBatching:
    """Trainer expands first; Cosmos validates every rank before training.

    Partial final batches may be included or rejected. There is no silent
    dropping, padding or algorithm-specific gradient reweighting in Cosmos.
    """

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


def _describe(batch, contract, mini_batch):
    if not isinstance(batch, ExpandedTrainingBatch):
        raise TypeError("prepare_training_batch must return ExpandedTrainingBatch")
    if type(mini_batch) is not int or mini_batch <= 0:
        raise ValueError("Training mini_batch must be a positive sample count")
    sizes = tuple(len(samples) for samples in batch.minibatches)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Expanded training batches cannot be empty")
    if any(size != mini_batch for size in sizes[:-1]):
        raise ValueError("Only the final training minibatch may be partial")
    if sizes[-1] > mini_batch or (
        contract.partial_tail == "reject" and sizes[-1] != mini_batch
    ):
        raise ValueError("Expanded training tail violates the declared policy")
    if not all(_finite(samples) for samples in batch.minibatches):
        raise ValueError("Expanded training inputs contain nonfinite values")
    return sizes


def run_training_step(trainer, *, before_step=None, **kwargs):
    """Worker entrypoint enforcing the declared contract, not a boolean bypass.

    All policy ranks participate in one preflight collective, including ranks
    whose expansion failed or produced no samples. No rank enters trainer
    collectives until every rank has a valid, equally long minibatch plan.
    """
    contract = getattr(trainer, "batching_contract", FixedRolloutBatching())
    if isinstance(contract, FixedRolloutBatching):
        if before_step is not None:
            before_step()
        return trainer.step_training(**kwargs)
    if not isinstance(contract, ExpandedSampleBatching):
        raise TypeError("Unknown trainer batching contract")

    batch = None
    try:
        batch = trainer.prepare_training_batch(kwargs["rollouts"])
        sizes = _describe(batch, contract, trainer.config.train.train_policy.mini_batch)
        local = {"sizes": sizes, "error": None}
    except Exception as error:
        local = {"sizes": (), "error": f"{type(error).__name__}: {error}"[:512]}
    plans = [local]
    if dist.is_initialized():
        plans = [None] * dist.get_world_size()
        dist.all_gather_object(plans, local)
    errors = [(rank, plan["error"]) for rank, plan in enumerate(plans) if plan["error"]]
    if errors:
        raise ValueError(f"Expanded training preflight failed on ranks: {errors}")
    if len({len(plan["sizes"]) for plan in plans}) != 1:
        raise ValueError("Expanded training ranks disagree on minibatch participation")
    if before_step is not None:
        before_step()
    expanded_kwargs = {key: value for key, value in kwargs.items() if key != "rollouts"}
    return trainer.step_expanded_training(batch, **expanded_kwargs)
