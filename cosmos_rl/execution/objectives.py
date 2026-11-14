# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Objective-specific masks and one denominator per optimizer window."""

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class ObjectiveWindow:
    coefficients: tuple[tuple[str, float], ...]
    counts: tuple[float, ...]
    gradient_divisor: int = 1

    @classmethod
    def prepare(
        cls, coefficients, local_counts, *, device="cpu", group=None, inter_policy=None
    ):
        active = tuple(sorted((k, float(v)) for k, v in coefficients.items() if v > 0))
        if any(
            not torch.isfinite(torch.tensor(v)) or v < 0 for v in coefficients.values()
        ):
            raise ValueError("Objective coefficients must be finite and nonnegative")
        if not active:
            raise ValueError("At least one objective must be enabled")
        counts = torch.tensor(
            [local_counts[k] for k, _ in active], device=device, dtype=torch.float64
        )
        if (counts < 0).any() or not torch.isfinite(counts).all():
            raise ValueError("Objective counts cannot be negative")
        divisor = 1
        if dist.is_initialized():
            # Agree term order and weights before a count collective can diverge.
            peers = [None] * dist.get_world_size(group)
            dist.all_gather_object(peers, active, group=group)
            if any(peer != active for peer in peers):
                raise ValueError(
                    "Objective definitions disagree across the gradient cohort"
                )
            dist.all_reduce(counts, group=group)
            divisor *= dist.get_world_size(group)
        if inter_policy is not None:
            inter_policy.allreduce(counts, counts, op=dist.ReduceOp.SUM)
            divisor *= inter_policy.world_size()
        return cls(active, tuple(float(v) for v in counts.tolist()), divisor)

    @property
    def empty(self):
        return not any(self.counts)

    def loss(self, values, masks):
        total = None
        for (name, coefficient), count in zip(self.coefficients, self.counts):
            value, mask = values[name], masks[name]
            if value.shape != mask.shape or mask.dtype != torch.bool:
                raise ValueError(f"Invalid {name} objective mask")
            valid = torch.where(mask, value, 0).float()
            if not torch.isfinite(valid).all():
                raise ValueError(f"Nonfinite contributing {name} loss")
            denominator = count if count > 0 else 1
            loss = valid.sum() * (coefficient * self.gradient_divisor / denominator)
            total = loss if total is None else total + loss
        return total


def clipped_policy_loss(current_logprobs, behavior_logprobs, advantages, *, clip=0.2):
    if (
        current_logprobs.shape != behavior_logprobs.shape
        or current_logprobs.shape != advantages.shape
    ):
        raise ValueError(
            "Policy objective requires aligned original behavior log probabilities and advantages"
        )
    log_ratio = current_logprobs - behavior_logprobs.detach()
    # Clip the selected branch before exp: an unselected overflowing ratio
    # otherwise produces NaN gradients through zero * infinity.
    selected = torch.where(
        advantages >= 0,
        log_ratio.clamp_max(math.log1p(clip)),
        log_ratio.clamp_min(math.log1p(-clip)),
    )
    return -selected.exp() * advantages.detach()


def supervised_token_loss(logits, labels):
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(labels.shape)
