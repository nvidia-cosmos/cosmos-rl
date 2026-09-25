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
from cosmos_rl.utils.distributed import collective_scope
from cosmos_rl.policy.trainer.objectives import ObjectiveWindow


@dataclass(frozen=True)
class FixedRolloutBatching:
    """One collected completion is one training sample; retain startup checks."""


@dataclass(frozen=True)
class ExpandedSampleBatching:
    """Opt in to a shared schedule with empty local contributions."""

    partial_tail: Literal["include", "reject"] = "reject"
    fixed_minibatches: int | None = None
    objective_weighting: Literal["sample", "episode"] | None = None
    accumulation_steps: int = 1

    def __post_init__(self):
        if self.partial_tail not in ("include", "reject"):
            raise ValueError("Unsupported partial-tail policy")
        if self.objective_weighting not in (None, "sample", "episode"):
            raise ValueError("Unsupported objective weighting")
        if type(self.accumulation_steps) is not int or self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
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
    episode_ids: tuple[Sequence, ...] | None = None
    objective_windows: tuple = ()

    def mean_gradient_scale(self, index, world_size):
        """Scale a local SUM loss when the trainer averages gradients across ranks."""
        if self.objective_windows:
            raise ValueError("Use objective_windows for accumulation-aware weighting")
        if self.global_sample_counts is None:
            raise ValueError("Fixed schedules require trainer-owned sample weighting")
        return world_size / self.global_sample_counts[index]


@dataclass(frozen=True)
class TrainingObjectiveWindow:
    """One optimizer update; all slots accumulate before stepping."""

    slots: tuple[int, ...]
    objective: ObjectiveWindow
    global_count: int
    gradient_divisor: int = 1

    def loss(self, losses, *, start=0, gradient_divisor=None):
        return self.objective.loss(
            losses,
            start=start,
            global_count=self.global_count,
            gradient_divisor=self.gradient_divisor
            if gradient_divisor is None
            else gradient_divisor,
        )


def _local_objectives(ids, contract):
    if contract.objective_weighting is None:
        return ()
    if contract.objective_weighting == "episode":
        owners = {}
        for slot, identities in enumerate(ids):
            window = slot // contract.accumulation_steps
            for identity in identities:
                if owners.setdefault(identity, window) != window:
                    raise ValueError(
                        "An episode must fit within one optimizer accumulation window"
                    )
    return tuple(
        ObjectiveWindow.prepare(
            [
                identity
                for slot in ids[start : start + contract.accumulation_steps]
                for identity in slot
            ],
            contract.objective_weighting,
        )
        for start in range(0, len(ids), contract.accumulation_steps)
    )


def _objective_windows(batch, plans, contract, gradient_divisor):
    if contract.objective_weighting is None:
        return ()
    local = _local_objectives(batch.episode_ids, contract)
    windows = []
    for start in range(0, len(batch.minibatches), contract.accumulation_steps):
        slots = tuple(
            range(
                start, min(start + contract.accumulation_steps, len(batch.minibatches))
            )
        )
        window = start // contract.accumulation_steps
        windows.append(
            TrainingObjectiveWindow(
                slots,
                local[window],
                sum(
                    plan["objective_counts"][window]
                    if window < len(plan["objective_counts"])
                    else 0
                    for plan in plans
                ),
                gradient_divisor,
            )
        )
    return tuple(windows)


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


def _training_cohort(trainer, comm):
    """Seal the existing gradient communicator, never a payload transport.

    One communicator joins corresponding DP ranks in each policy replica. The
    local preflight is replicated to every DP rank before crossing this cohort.
    Elastic membership is not supported for a sealed optimizer schedule.
    """
    configured = getattr(
        getattr(getattr(trainer.config, "policy", None), "parallelism", None),
        "n_init_replicas",
        1,
    )
    if comm is None:
        sealed = getattr(trainer, "_sealed_training_cohort", None)
        if sealed is not None and sealed[0] is not None:
            raise ValueError(
                "Training gradient communicator disappeared after agreement"
            )
        if configured > 1:
            raise ValueError(
                "Multi-replica expanded schedules require inter_policy_nccl"
            )
        trainer._sealed_training_cohort = (None, (), 1, None)
        return 1
    comm.wait_comm_ready()
    replicas = comm.world_size()
    if type(replicas) is not int or replicas < 1:
        raise ValueError("Invalid training gradient cohort size")
    membership = tuple(sorted(comm.replica_name_to_rank.items()))
    identity = (comm, membership, replicas, getattr(comm, "comm_idx", None))
    sealed = getattr(trainer, "_sealed_training_cohort", None)
    if sealed is not None:
        if sealed != identity:
            raise ValueError("Training gradient cohort changed after agreement")
        return replicas
    if replicas > 1:
        policy = trainer.config.train.train_policy
        contract = trainer.batching_contract
        signature = [
            policy.mini_batch,
            policy.mu_iterations,
            trainer._sealed_batching_dp_size,
            contract.fixed_minibatches or 0,
            contract.accumulation_steps,
            {None: 0, "sample": 1, "episode": 2}[contract.objective_weighting],
            int(contract.partial_tail == "include"),
        ]
        # MAX of both signs compares minima and maxima in one collective.
        expected = signature + [-value for value in signature]
        agreement = torch.tensor(expected, dtype=torch.int64, device=trainer.device)
        comm.allreduce(agreement, agreement, op=dist.ReduceOp.MAX)
        if agreement.tolist() != expected:
            raise ValueError(
                "Policy replicas disagree on training schedule configuration"
            )
    trainer._sealed_training_cohort = identity
    return replicas


def _cohort_training_plans(trainer, plans, comm, current_step):
    """Agree widths/errors, then sum counts across the gradient cohort.

    Only small integer metadata crosses replicas, on the training thread. No
    losses, identities, payloads, or model state are exchanged here.
    """
    contract = trainer.batching_contract
    errors = [plan["error"] for plan in plans if plan["error"]]
    width = contract.fixed_minibatches or max(len(plan["sizes"]) for plan in plans)
    if type(current_step) is not int or current_step < 0:
        errors.append("current_step must be a nonnegative integer")
        current_step = 0
    header = torch.tensor(
        [width, bool(errors), current_step, -current_step],
        dtype=torch.int64,
        device=trainer.device,
    )
    comm.allreduce(header, header, op=dist.ReduceOp.MAX)
    width, failed, high_step, negative_low_step = header.tolist()
    if failed:
        raise ValueError(
            f"Expanded preflight failed in the gradient cohort; local errors: {errors}"
        )
    if high_step != -negative_low_step:
        raise ValueError("Policy replicas disagree on training step")
    n_windows = (
        (width + contract.accumulation_steps - 1) // contract.accumulation_steps
        if contract.objective_weighting is not None
        else 0
    )
    values = [
        sum(plan["sizes"][i] if i < len(plan["sizes"]) else 0 for plan in plans)
        for i in range(width)
    ]
    values += [
        sum(
            plan["objective_counts"][i] if i < len(plan["objective_counts"]) else 0
            for plan in plans
        )
        for i in range(n_windows)
    ]
    if values:
        totals = torch.tensor(values, dtype=torch.int64, device=trainer.device)
        comm.allreduce(totals, totals, op=dist.ReduceOp.SUM)
        values = totals.tolist()
    return [
        {
            "sizes": tuple(values[:width]),
            "objective_counts": tuple(values[width:]),
            "mu": plans[0]["mu"],
            "error": None,
        }
    ]


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
    trainer._sealed_batching_dp_size = len(signatures)


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
    cleaned_ids = []
    dropped = 0
    if batch.episode_ids is not None and (
        len(batch.episode_ids) != len(batch.minibatches)
        or any(
            len(ids) != len(samples)
            for ids, samples in zip(batch.episode_ids, batch.minibatches)
        )
    ):
        raise ValueError("episode_ids must identify every prepared sample")
    if (
        contract.objective_weighting == "episode"
        and batch.episode_ids is None
        and any(batch.minibatches)
    ):
        raise ValueError("Episode weighting requires prepared episode_ids")
    for index, samples in enumerate(batch.minibatches):
        if background:
            _require_cpu(samples)
        ids = (
            batch.episode_ids[index]
            if batch.episode_ids is not None
            else tuple(range(len(samples)))
        )
        retained = tuple(
            (sample, identity)
            for sample, identity in zip(samples, ids)
            if _finite(sample)
        )
        valid = tuple(sample for sample, _ in retained)
        if contract.partial_tail == "reject" and len(valid) < policy.mini_batch:
            valid = ()
            retained = ()
        # IDs are rank-local and CPU metadata, never payload tensors.
        if contract.objective_weighting == "episode" and any(
            type(identity) not in (int, str) for _, identity in retained
        ):
            raise ValueError("Episode IDs must be integer or string metadata")
        dropped += len(samples) - len(valid)
        cleaned.append(valid)
        cleaned_ids.append(tuple(identity for _, identity in retained))
    if (
        contract.fixed_minibatches is not None
        and len(cleaned) > contract.fixed_minibatches
    ):
        raise ValueError(
            "Prepared batch exceeds sealed fixed_minibatches; trainer must bound preparation"
        )
    return (
        ExpandedTrainingBatch(tuple(cleaned), episode_ids=tuple(cleaned_ids)),
        dropped,
        recovery,
    )


def prefetch_training_batch(trainer, rollouts):
    """Submit the next owned batch through the trainer's payload prefetcher.

    Call on the training thread when the next batch is available. Existing
    controller ACK/step ordering is unchanged; this never invents a future batch.
    ``run_training_step`` consumes it when called with the same rollout objects.
    Return False without preparing when payload prefetch is disabled/unavailable;
    the normal entrypoint will prepare synchronously. Return True when submitted.
    """
    if not isinstance(trainer.batching_contract, ExpandedSampleBatching):
        raise TypeError("Background preparation requires expanded batching")
    if getattr(trainer, "_prepared_training_batch", None) is not None:
        raise RuntimeError("Only one unconsumed prepared training batch is allowed")
    packer = getattr(trainer, "data_packer", None)
    if not getattr(packer, "_prefetch_enabled", False):
        return False
    agree_batching_schedule(trainer)
    owned = tuple(rollouts)
    future = packer.start_prepared_prefetch(
        owned, lambda: _prepare_local_batch(trainer, owned, background=True)
    )
    trainer._prepared_training_batch = (owned, future, packer)
    return True


def _take_local_batch(trainer, rollouts):
    pending = getattr(trainer, "_prepared_training_batch", None)
    if pending is None:
        return _prepare_local_batch(trainer, rollouts)
    owned, future, packer = pending
    packer._raise_if_prefetch_failed()
    if len(owned) != len(rollouts) or any(a is not b for a, b in zip(owned, rollouts)):
        raise ValueError(
            "Prepared batch does not match the training command's rollouts"
        )
    try:
        result = future.result(timeout=packer._prefetch_timeout_s)
        packer._raise_if_prefetch_failed()
        return result
    finally:
        if future.done():
            packer.release_prepared_prefetch(future)
            trainer._prepared_training_batch = None


def run_training_step(trainer, *, before_step=None, **kwargs):
    """Worker entrypoint enforcing the declared contract, not a boolean bypass.

    One metadata exchange per expanded update agrees the variable schedule, not
    one exchange per minibatch. Default process groups are replica-local; the
    supported pure-DP topology also includes the inter-policy gradient cohort.
    """
    contract = getattr(trainer, "batching_contract", FixedRolloutBatching())
    if isinstance(contract, FixedRolloutBatching):
        if before_step is not None:
            before_step()
        return trainer.step_training(**kwargs)
    if not isinstance(contract, ExpandedSampleBatching):
        raise TypeError("Unknown trainer batching contract")
    agree_batching_schedule(trainer)

    # Pin native mesh ownership from agreement through optimizer completion.
    # CPU background preparation remains independent of this local lock.
    with collective_scope(kwargs.get("inter_policy_nccl")):
        return _run_expanded_training_step(trainer, before_step, kwargs)


def _run_expanded_training_step(trainer, before_step, kwargs):
    contract = trainer.batching_contract

    policy = trainer.config.train.train_policy
    comm = kwargs.get("inter_policy_nccl")
    replicas = _training_cohort(trainer, comm)
    mu = getattr(policy, "mu_iterations", None)
    dropped = 0
    recovery = None
    try:
        if type(mu) is not int or mu < 1:
            raise ValueError("mu_iterations must be a positive integer")
        batch, dropped, recovery = _take_local_batch(trainer, kwargs["rollouts"])
        local = {
            "sizes": tuple(map(len, batch.minibatches)),
            "objective_counts": tuple(
                objective.count
                for objective in _local_objectives(batch.episode_ids, contract)
            ),
            "mu": mu,
            "error": None,
        }
    except Exception as error:
        if (
            contract.fixed_minibatches is not None
            and contract.objective_weighting is None
        ):
            # The schedule is already sealed: no extra error agreement here.
            # Unexpected errors use the normal worker/cohort failure path.
            raise
        local = {
            "sizes": (),
            "mu": mu,
            "error": f"{type(error).__name__}: {error}"[:512],
        }
    if contract.fixed_minibatches is not None and contract.objective_weighting is None:
        batch = ExpandedTrainingBatch(
            batch.minibatches
            + ((),) * (contract.fixed_minibatches - len(batch.minibatches)),
            None,
            mu,
        )
        return _consume(trainer, batch, before_step, dropped, recovery, kwargs)
    plans = _gather(local)
    gradient_divisor = len(plans) * replicas
    if replicas > 1:
        plans = _cohort_training_plans(
            trainer, plans, comm, kwargs.get("current_step", 0)
        )
        # Native HA meshes are pinned by the enclosing scope. Also reject drift
        # observed in custom adapters; they must own a stable group lifetime.
        _training_cohort(trainer, comm)
    errors = [(rank, plan["error"]) for rank, plan in enumerate(plans) if plan["error"]]
    if errors:
        raise ValueError(f"Expanded training preflight failed on ranks: {errors}")
    if len({plan["mu"] for plan in plans}) != 1:
        raise ValueError("Expanded training ranks disagree on configured mu_iterations")
    width = contract.fixed_minibatches or max(len(plan["sizes"]) for plan in plans)
    counts = tuple(
        sum(plan["sizes"][i] if i < len(plan["sizes"]) else 0 for plan in plans)
        for i in range(width)
    )
    active = (
        list(range(width))
        if contract.fixed_minibatches or contract.objective_weighting is not None
        else [i for i, count in enumerate(counts) if count]
    )
    batch = ExpandedTrainingBatch(
        tuple(
            batch.minibatches[i] if i < len(batch.minibatches) else () for i in active
        ),
        tuple(counts[i] for i in active),
        mu,
        tuple(
            batch.episode_ids[i] if i < len(batch.episode_ids) else () for i in active
        ),
    )
    batch = ExpandedTrainingBatch(
        batch.minibatches,
        batch.global_sample_counts,
        batch.mu_iterations,
        batch.episode_ids if contract.objective_weighting is not None else None,
        _objective_windows(batch, plans, contract, gradient_divisor),
    )
    return _consume(trainer, batch, before_step, dropped, recovery, kwargs)


def _consume(trainer, batch, before_step, dropped, recovery, kwargs):
    active = (
        any(window.global_count for window in batch.objective_windows)
        if batch.objective_windows
        else bool(batch.minibatches)
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
