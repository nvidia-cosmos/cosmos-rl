# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in recorded token objectives in the existing SFT training loop."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cosmos_rl.execution.objectives import (
    ObjectiveWindow,
    clipped_policy_loss,
    supervised_token_loss,
)


class TokenObjectiveBatch:
    def __init__(self, samples, config, *, device, group=None, vocab_size=None):
        self.spec = config.execution.objectives
        self.rows = []
        coefficients = {
            name: getattr(self.spec, name)
            for name in ("supervised", "policy_gradient", "distillation")
        }
        counts = {name: 0.0 for name in coefficients}
        error = None
        try:
            for sample in samples:
                labels = torch.as_tensor(
                    sample["label_ids"][: config.policy.model_max_length],
                    dtype=torch.long,
                )[1:]
                n = len(labels)
                if vocab_size is not None and (
                    ((labels < 0) & (labels != -100)).any()
                    or (labels >= vocab_size).any()
                ):
                    raise ValueError("Training labels exceed the student vocabulary")
                record = sample.get("_execution_record")
                masks = {}
                for name, coefficient in coefficients.items():
                    if coefficient <= 0:
                        continue
                    mask = labels != -100
                    explicit = record.masks.get(name) if record else None
                    if explicit is not None:
                        explicit = torch.as_tensor(explicit)
                        if explicit.numel() == 0 and n == 0:
                            explicit = explicit.bool()
                        if (
                            explicit.ndim != 1
                            or explicit.dtype != torch.bool
                            or len(explicit) < n
                        ):
                            raise ValueError(
                                f"{name} mask does not cover the token positions"
                            )
                        mask = mask & explicit[:n]
                    masks[name] = mask
                row = {"masks": masks, "n": n, "weight": 1.0}
                if self.spec.policy_gradient:
                    if (
                        record is not None
                        and record.annotations.get("rl_eligible") is False
                    ):
                        masks["policy_gradient"] = torch.zeros(n, dtype=torch.bool)
                    valid = masks["policy_gradient"]
                    if valid.any():
                        if record is None:
                            raise ValueError(
                                "Recorded policy loss requires a SampleRecord"
                            )
                        record.require_policy_gradient()
                        if "policy_gradient" not in record.masks:
                            raise ValueError(
                                "Recorded policy loss requires an explicit aligned token mask"
                            )
                        old = torch.as_tensor(
                            record.rollout.completion_logprobs, dtype=torch.float32
                        )
                        raw_advantages = record.rollout.advantage
                        if raw_advantages is None:
                            advantage = record.annotations.get("group_advantage")
                            if advantage is None:
                                raise ValueError(
                                    "Recorded policy loss requires advantages or explicit group standardization"
                                )
                            raw_advantages = [advantage] * len(old)
                        advantages = torch.as_tensor(
                            raw_advantages, dtype=torch.float32
                        )
                        if advantages.ndim == 0:
                            advantages = advantages.expand_as(old)
                        if (
                            old.ndim != 1
                            or len(old) < n
                            or advantages.shape != old.shape
                        ):
                            raise ValueError(
                                "Recorded behavior probabilities and advantages must align with next-token positions"
                            )
                        if (
                            not torch.isfinite(old[:n][valid]).all()
                            or not torch.isfinite(advantages[:n][valid]).all()
                        ):
                            raise ValueError(
                                "Contributing behavior probabilities and advantages must be finite"
                            )
                        row.update(
                            behavior=torch.where(valid, old[:n], 0),
                            advantages=torch.where(valid, advantages[:n], 0),
                        )
                    else:
                        row.update(behavior=torch.zeros(n), advantages=torch.zeros(n))
                if self.spec.distillation:
                    valid = masks["distillation"]
                    if valid.any():
                        if record is None or "teacher" not in record.annotations:
                            raise ValueError(
                                "Recorded distillation requires teacher identity and targets"
                            )
                        teacher = torch.as_tensor(
                            record.rollout.teacher_logprobs, dtype=torch.float32
                        )
                        if teacher.ndim != 2 or len(teacher) < n:
                            raise ValueError(
                                "Teacher targets must align with next-token positions"
                            )
                        if vocab_size is not None and teacher.shape[-1] != vocab_size:
                            raise ValueError(
                                "Teacher and student token vocabularies differ"
                            )
                        from cosmos_rl.execution.records import ModelIdentity

                        identity = ModelIdentity.model_validate(
                            record.annotations["teacher"]
                        )
                        if identity.role != "teacher":
                            raise ValueError(
                                "Teacher targets require teacher provenance"
                            )
                        if not torch.isfinite(
                            teacher[:n][valid]
                        ).all() or not torch.allclose(
                            teacher[:n][valid].logsumexp(-1),
                            torch.zeros(int(valid.sum())),
                            atol=1e-4,
                        ):
                            raise ValueError(
                                "Teacher targets must be finite normalized log probabilities"
                            )
                        row["teacher"] = teacher[:n]
                    else:
                        row["teacher"] = None
                source = config.execution.source
                if (
                    source is not None
                    and source.reward_weighting == "nonnegative"
                    and self.spec.supervised
                ):
                    weight = (
                        float(record.rollout.reward)
                        if masks["supervised"].any()
                        else 0.0
                    )
                    if not torch.isfinite(torch.tensor(weight)) or weight < 0:
                        raise ValueError(
                            "Reward-weighted supervision requires finite nonnegative scores"
                        )
                    row["weight"] = weight
                for name, mask in masks.items():
                    counts[name] += mask.sum().item() * (
                        row["weight"] if name == "supervised" else 1
                    )
                self.rows.append(row)
        except Exception as failure:
            error = str(failure)
        if dist.is_initialized():
            errors = [None] * dist.get_world_size(group)
            dist.all_gather_object(errors, error, group=group)
            error = next((failure for failure in errors if failure is not None), None)
        if error is not None:
            raise ValueError(error)
        self.window = ObjectiveWindow.prepare(
            coefficients, counts, device=device, group=group
        )

    def loss(self, logits, labels, *, start, valid_input_len=None):
        shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
        values = {}
        masks = {}
        if valid_input_len is None:
            selected = self.rows[start : start + len(labels)]
            placements = [(i, 0, row["n"]) for i, row in enumerate(selected)]
        else:
            lengths = valid_input_len.tolist()
            if (
                labels.shape[0] != 1
                or any(length < 1 for length in lengths)
                or sum(lengths) != labels.shape[1]
            ):
                raise ValueError("Packed token layout does not match the model output")
            selected = self.rows[start : start + len(lengths)]
            if len(selected) != len(lengths):
                raise ValueError("Packed token layout does not match its records")
            placements, offset = [], 0
            for row, length in zip(selected, lengths):
                n = min(row["n"], length - 1)
                if any(mask[n:].any() for mask in row["masks"].values()):
                    raise ValueError("Packing omitted contributing token positions")
                placements.append((0, offset, n))
                offset += length
        # A mixed objective needs one vocabulary normalization, shared by CE,
        # sampled-token policy loss and KL. Keep the native CE-only fast path.
        logprobs = (
            None
            if self.window.coefficients == (("supervised", self.spec.supervised),)
            else F.log_softmax(shifted_logits.float(), dim=-1)
        )
        for name, _ in self.window.coefficients:
            mask = torch.zeros_like(shifted_labels, dtype=torch.bool)
            for row, (i, offset, n) in zip(selected, placements):
                mask[i, offset : offset + n] = row["masks"][name][:n].to(mask.device)
            masks[name] = mask
            if name == "supervised":
                loss = (
                    supervised_token_loss(shifted_logits, shifted_labels)
                    if logprobs is None
                    else F.nll_loss(
                        logprobs.reshape(-1, logprobs.shape[-1]),
                        shifted_labels.reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).reshape(shifted_labels.shape)
                )
                weights = torch.zeros_like(loss)
                for row, (i, offset, n) in zip(selected, placements):
                    weights[i, offset : offset + n] = row["weight"]
                values[name] = loss * weights
            elif name == "policy_gradient":
                current = logprobs.gather(
                    -1, shifted_labels.clamp_min(0).unsqueeze(-1)
                ).squeeze(-1)
                behavior = torch.zeros_like(current)
                advantages = torch.zeros_like(current)
                for row, (i, offset, n) in zip(selected, placements):
                    behavior[i, offset : offset + n] = row["behavior"][:n].to(
                        current.device
                    )
                    advantages[i, offset : offset + n] = row["advantages"][:n].to(
                        current.device
                    )
                # Excluded padding cannot overflow an importance ratio.
                current = torch.where(mask, current, 0)
                behavior = torch.where(mask, behavior, 0)
                values[name] = clipped_policy_loss(
                    current, behavior, advantages, clip=self.spec.policy_clip
                )
            else:
                target = torch.zeros_like(shifted_logits, dtype=torch.float32)
                for row, (i, offset, n) in zip(selected, placements):
                    if row["teacher"] is None:
                        continue
                    if row["teacher"].shape[-1] != target.shape[-1]:
                        raise ValueError(
                            "Teacher and student token vocabularies differ"
                        )
                    target[i, offset : offset + n] = row["teacher"][:n].to(
                        target.device
                    )
                target = target.exp()
                target = torch.where(mask.unsqueeze(-1), target, 0)
                values[name] = F.kl_div(
                    logprobs,
                    target,
                    reduction="none",
                ).sum(-1)
        return self.window.loss(values, masks)
