# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from cosmos_rl.policy.model.hf_models.patch import apply_qwen3_vl_patch_embed_compat


class _PatchEmbed(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 3
        self.temporal_patch_size = 2
        self.patch_size = 2
        self.embed_dim = 5
        self.proj = torch.nn.Conv3d(3, 5, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True)

    def forward(self, x):
        x = x.view(-1, 3, 2, 2, 2)
        return self.proj(x).view(-1, 5)


def test_linear_patch_embed_matches_conv3d_and_preserves_state_keys() -> None:
    model = torch.nn.Sequential(_PatchEmbed())
    values = torch.randn(4, 24)
    expected = model(values)
    keys = set(model.state_dict())
    assert apply_qwen3_vl_patch_embed_compat(model, "linear", device_capability=(8, 0))
    torch.testing.assert_close(model(values), expected)
    assert set(model.state_dict()) == keys
