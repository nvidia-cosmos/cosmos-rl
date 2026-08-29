# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import async_safe_ce


class _CE(torch.nn.Module):
    def forward(self, output, target, *, ignore_index, reduction, lin_weight=None):
        del lin_weight
        return F.cross_entropy(output.float(), target, ignore_index=ignore_index, reduction=reduction)


def test_async_safe_ce_emits_raw_token_statistics() -> None:
    torch.manual_seed(17)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, -100], [0, 2, 3, 4]])

    loss, numerator, denominator = async_safe_ce(
        logits,
        labels,
        ce_impl=_CE(),
        return_stats=True,
    )

    shifted_logits = logits[:, :-1].reshape(-1, 7)
    shifted_labels = labels[:, 1:].reshape(-1)
    raw = F.cross_entropy(shifted_logits.float(), shifted_labels, ignore_index=-100, reduction="none")
    valid = shifted_labels != -100
    assert denominator.item() == int(valid.sum())
    torch.testing.assert_close(numerator, raw[valid].sum())
    torch.testing.assert_close(loss, numerator / denominator)
