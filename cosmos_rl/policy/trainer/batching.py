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
    fixed_minibatches: int | None = None

    def __post_init__(self):
        if self.partial_tail not in ("include", "reject"):
            raise ValueError("Unsupported partial-tail policy")
        if self.fixed_minibatches is not None and (
            type(self.fixed_minibatches) is not int or self.fixed_minibatches < 1
        ):
            raise ValueError("fixed_minibatches must be a positive integer")


@dataclass(frozen=True)
class ExpandedTrainingBatch:
    """Actual ordered minibatches, each an indexable sequence of training samples.

    Do not pass rollout handles or lazy iterators here. Expansion must not
    perform training collectives or update optimizer/scheduler state.
    """

    minibatches: tuple[Sequence, ...]
    global_sample_counts: tuple[int, ...] | None = ()
    mu_iterations: int = 1

    def mean_gradient_scale(self, index, world_size):
        """Scale a local SUM loss when the trainer averages gradients across ranks."""
        if self.global_sample_counts is None:
            raise ValueError("Fixed schedules require trainer-owned sample weighting")
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


def _gather(local):
    if not dist.is_initialized():
        return [local]
    plans = [None] * dist.get_world_size()
    dist.all_gather_object(plans, local)
    return plans


def agree_batching_schedule(trainer):
    """Seal replica-local configuration once, before any expanded preparation.

    Called lazily by the worker entrypoint; may also be called at trainer startup
    after its process group exists. Configuration/group changes require a new
    trainer, not an independently renegotiated schedule on one rank.
    """
    contract = trainer.batching_contract
    policy = trainer.config.train.train_policy
    signature = (contract, policy.mini_batch, getattr(policy, "mu_iterations", None))
    group = dist.group.WORLD if dist.is_initialized() else None
    sealed = getattr(trainer, "_sealed_batching_schedule", None)
    if sealed is not None:
        if sealed != (signature, group):
            raise ValueError("Sealed batching schedule or process group changed")
        return
    signatures = _gather(signature)
    if any(other != signature for other in signatures):
        raise ValueError("Expanded ranks disagree on batching schedule configuration")
    if any(type(value) is not int or value < 1 for value in signature[1:]):
        raise ValueError("mini_batch and mu_iterations must be positive integers")
    trainer._sealed_batching_schedule = (signature, group)


def _require_cpu(value):
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise ValueError("Background batch preparation must return CPU samples")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_cpu(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_cpu(item)


def _prepare_local_batch(trainer, rollouts, *, background=False):
    contract = trainer.batching_contract
    policy = trainer.config.train.train_policy
    recovery = None
    try:
        batch = trainer.prepare_training_batch(rollouts)
    except RecoverablePreparationError as error:
        batch = ExpandedTrainingBatch(())
        recovery = str(error)[:512]
        logger.warning(
            "Expanded preparation unavailable; contributing zero: %s", recovery
        )
    _describe(batch, policy.mini_batch)
    cleaned = []
    dropped = 0
    for samples in batch.minibatches:
        if background:
            _require_cpu(samples)
        valid = tuple(sample for sample in samples if _finite(sample))
        if contract.partial_tail == "reject" and len(valid) < policy.mini_batch:
            valid = ()
        dropped += len(samples) - len(valid)
        cleaned.append(valid)
    if (
        contract.fixed_minibatches is not None
        and len(cleaned) > contract.fixed_minibatches
    ):
        raise ValueError(
            "Prepared batch exceeds sealed fixed_minibatches; trainer must bound preparation"
        )
    return ExpandedTrainingBatch(tuple(cleaned)), dropped, recovery


def prefetch_training_batch(trainer, rollouts):
    """Submit the next owned batch through the trainer's payload prefetcher.

    Call on the training thread when the next batch is available. Existing
    controller ACK/step ordering is unchanged; this never invents a future batch.
    ``run_training_step`` consumes it when called with the same rollout objects.
    """
    if not isinstance(trainer.batching_contract, ExpandedSampleBatching):
        raise TypeError("Background preparation requires expanded batching")
    if getattr(trainer, "_prepared_training_batch", None) is not None:
        raise RuntimeError("Only one unconsumed prepared training batch is allowed")
    agree_batching_schedule(trainer)
    owned = tuple(rollouts)
    packer = trainer.data_packer
    future = packer.start_prepared_prefetch(
        owned, lambda: _prepare_local_batch(trainer, owned, background=True)
    )
    trainer._prepared_training_batch = (owned, future, packer)


def _take_local_batch(trainer, rollouts):
    pending = getattr(trainer, "_prepared_training_batch", None)
    if pending is None:
        return _prepare_local_batch(trainer, rollouts)
    owned, future, packer = pending
    if len(owned) != len(rollouts) or any(a is not b for a, b in zip(owned, rollouts)):
        raise ValueError(
            "Prepared batch does not match the training command's rollouts"
        )
    try:
        return future.result(timeout=packer._prefetch_timeout_s)
    finally:
        if future.done():
            packer.release_prepared_prefetch(future)
            trainer._prepared_training_batch = None


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
    agree_batching_schedule(trainer)

    policy = trainer.config.train.train_policy
    mu = getattr(policy, "mu_iterations", None)
    dropped = 0
    recovery = None
    try:
        if type(mu) is not int or mu < 1:
            raise ValueError("mu_iterations must be a positive integer")
        batch, dropped, recovery = _take_local_batch(trainer, kwargs["rollouts"])
        local = {"sizes": tuple(map(len, batch.minibatches)), "mu": mu, "error": None}
    except Exception as error:
        if contract.fixed_minibatches is not None:
            # The schedule is already sealed: no extra error agreement here.
            # Unexpected errors use the normal worker/cohort failure path.
            raise
        local = {
            "sizes": (),
            "mu": mu,
            "error": f"{type(error).__name__}: {error}"[:512],
        }
    if contract.fixed_minibatches is not None:
        batch = ExpandedTrainingBatch(
            batch.minibatches
            + ((),) * (contract.fixed_minibatches - len(batch.minibatches)),
            None,
            mu,
        )
        return _consume(trainer, batch, before_step, dropped, recovery, kwargs)
    plans = _gather(local)
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
    return _consume(trainer, batch, before_step, dropped, recovery, kwargs)


def _consume(trainer, batch, before_step, dropped, recovery, kwargs):
    if batch.minibatches and before_step is not None:
        before_step()
    expanded_kwargs = {key: value for key, value in kwargs.items() if key != "rollouts"}
    # Even an all-empty plan reaches the trainer for checkpoint/control work,
    # but has zero training slots and must not advance optimizer or scheduler.
    result = trainer.step_expanded_training(batch, **expanded_kwargs)
    result.update(
        {
            "batching/skipped_update": int(not batch.minibatches),
            "batching/dropped_samples": dropped,
            "batching/preparation_failed": int(recovery is not None),
        }
    )
    return result
