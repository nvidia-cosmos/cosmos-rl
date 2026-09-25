# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sample/episode means over an entire optimizer accumulation window.

Each input loss is already a scalar sample objective. Episode identities are
local to a data-parallel rank, and an episode must not span optimizer updates.
No token objective or implicit mean-of-minibatch-means is provided.
"""

from dataclasses import dataclass
from collections import Counter
from typing import Literal
from functools import wraps
from inspect import signature

import torch
import torch.distributed as dist
from cosmos_rl.utils.distributed import collective_scope


def pin_vla_objective_cohort(step):
    """Pin native membership from opt-in counts through optimizer completion.

    Legacy VLA calls are unchanged. Signature binding supports both the worker's
    keyword entrypoint and existing direct positional trainer calls.
    """
    parameters = signature(step)

    @wraps(step)
    def wrapped(trainer, *args, **kwargs):
        if trainer.config.vla.objective_weighting is None:
            return step(trainer, *args, **kwargs)
        arguments = parameters.bind(trainer, *args, **kwargs).arguments
        with collective_scope(arguments["inter_policy_nccl"]):
            return step(trainer, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class ObjectiveWindow:
    weights: tuple[float, ...]
    count: int

    @classmethod
    def prepare(cls, episode_ids, weighting: Literal["sample", "episode"]):
        """Prepare after filtering, on CPU; safe in background prefetch.

        Pass one episode identity per retained scalar sample, in loss order.
        An empty episode has no samples and contributes no denominator mass.
        """
        ids = tuple(episode_ids)
        if weighting == "sample":
            return cls((1.0,) * len(ids), len(ids))
        if weighting != "episode":
            raise ValueError("weighting must be 'sample' or 'episode'")
        counts = Counter(ids)
        return cls(tuple(1.0 / counts[key] for key in ids), len(counts))

    def loss(self, losses, *, start=0, global_count, gradient_divisor=1):
        """Scale a slice's sum for the FINAL gradient reduction.

        global_count covers the whole optimizer window, not this microbatch.
        gradient_divisor is the product of participant counts of averaging
        reductions (one for sum reductions). Do not divide by accumulation steps.
        Empty slices still need a model-connected dummy loss/forward when the
        model's distributed wrapper requires matching collectives.
        """
        if losses.ndim != 1:
            raise ValueError("Expected one scalar loss per sample")
        if (
            type(start) is not int
            or start < 0
            or start + losses.numel() > len(self.weights)
        ):
            raise ValueError("Loss slice is outside the prepared objective window")
        if type(global_count) is not int or global_count < self.count:
            raise ValueError("Global count must include the local objective count")
        if type(gradient_divisor) is not int or gradient_divisor < 1:
            raise ValueError("gradient_divisor must be a positive integer")
        if losses.dtype in (torch.float16, torch.bfloat16):
            losses = losses.float()
        weights = losses.new_tensor(self.weights[start : start + losses.numel()])
        return (losses * weights).sum() * (gradient_divisor / max(global_count, 1))


def masked_sample_means(losses, mask):
    """Reduce components within each sample, excluding wholly masked samples.

    The leading axis identifies samples; remaining axes are the model-specific
    components of their scalar objective. Excluded nonfinite components cannot
    contaminate the loss. Nonfinite *valid* components remain visible as errors.
    """
    if losses.shape != mask.shape or losses.ndim < 2 or mask.dtype != torch.bool:
        raise ValueError("Expected equal [sample, ...] loss and boolean mask shapes")
    sizes = mask.flatten(1).sum(1)
    sums = torch.where(mask, losses, 0).flatten(1).sum(1)
    return (sums / sizes.clamp_min(1))[sizes > 0]


def vla_objective_inputs(packer, policy_inputs, max_chunks):
    """Stream CPU masks only when supported; preserve custom packer fallback."""
    masks = getattr(packer, "policy_logprob_masks", None)
    if masks is not None:
        definitions = [vars(packer)] + [vars(cls) for cls in type(packer).__mro__]
        mask_owner = next(
            i for i, attrs in enumerate(definitions) if "policy_logprob_masks" in attrs
        )
        collate_owner = next(
            i for i, attrs in enumerate(definitions) if "policy_collate_fn" in attrs
        )
        # An existing custom collation override has not opted into its inherited
        # mask helper. Its masks may differ: keep the old count path in that case.
        if collate_owner < mask_owner:
            masks = None
    for policy_input in policy_inputs:
        if masks is None:
            yield packer.policy_collate_fn(policy_input, max_chunks)
        else:
            yield {"logprob_masks": masks(policy_input, max_chunks, device="cpu")}


def vla_objective(trainer, episode_data, inter_policy_nccl):
    """Normalize one VLA update across its DP mesh and replica communicator.

    VLA's fixed-rollout loops retain their existing collective schedule. A
    sample is an action chunk's scalar mean loss; an episode is its mean over
    retained chunks. This count exchange precedes backward and is independent
    of the number of microbatches. Expanded trainers instead reuse preflight.
    """
    if any(
        getattr(trainer.parallel_dims, dimension, False)
        for dimension in ("cp_enabled", "tp_enabled", "pp_enabled", "ep_enabled")
    ):
        raise ValueError("VLA sample/episode objectives currently require pure DP")
    ids = []
    slots = 0
    max_chunks = 1
    for episode, data in enumerate(episode_data):
        mask = data["logprob_masks"].bool()
        if mask.ndim < 1:
            raise ValueError("VLA masks require a leading sample dimension")
        # Legacy custom PI05 packers may supply [B], including an empty B.
        valid = mask if mask.ndim == 1 else mask.flatten(1).any(1)
        ids.extend([episode] * int(valid.sum().item()))
        slots += 1
        max_chunks = max(max_chunks, len(valid))
    objective = ObjectiveWindow.prepare(ids, trainer.config.vla.objective_weighting)
    count = torch.tensor(objective.count, dtype=torch.int64, device=trainer.device)
    divisor = 1
    if trainer.parallel_dims.dp_enabled:
        group = trainer.parallel_dims.mesh["dp"].get_group()
        divisor = dist.get_world_size(group)
        local = count.new_tensor([objective.count, max_chunks, slots])
        plans = [torch.empty_like(local) for _ in range(divisor)]
        dist.all_gather(plans, local, group=group)
        plans = torch.stack(plans).cpu().tolist()
        if len({plan[2] for plan in plans}) != 1:
            raise ValueError(
                "Fixed-rollout VLA requires matching episode slots across DP ranks; use masked empty episodes, not missing slots"
            )
        count.fill_(sum(plan[0] for plan in plans))
        max_chunks = max(plan[1] for plan in plans)
    inter_policy_nccl.wait_comm_ready()
    inter_policy_nccl.allreduce(count, count, op=dist.ReduceOp.SUM)
    divisor *= inter_policy_nccl.world_size()
    return objective, int(count.item()), divisor, max_chunks
