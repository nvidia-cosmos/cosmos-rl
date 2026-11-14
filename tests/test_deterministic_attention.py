# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Backend policy on CPU and actual resolved attention repeatability on GPU."""

import sys
from types import ModuleType

import pytest
import torch

from cosmos_rl.policy.kernel import modeling_utils as kernels


@pytest.fixture(autouse=True)
def isolated_singleton(monkeypatch):
    monkeypatch.setattr(kernels.FlashAttnMetaSingleton, "_instances", {})


@pytest.mark.parametrize(
    "deterministic,requested,compile,expected",
    [
        (False, None, False, 3),
        (False, None, True, 2),
        (True, None, False, 2),
        (True, None, True, 2),
        (True, 2, False, 2),
        (True, 3, False, 3),
        (True, 3, True, 2),
    ],
)
def test_backend_selection_and_deterministic_forwarding(
    monkeypatch, deterministic, requested, compile, expected
):
    monkeypatch.setattr(kernels, "decide_fa_version", lambda: 3)
    monkeypatch.setattr(kernels, "check_if_fa_installed", lambda: True)
    calls, messages = [], []

    def attention(*args, **kwargs):
        calls.append(kwargs)
        return args[0]

    fa2 = ModuleType("flash_attn")
    fa3 = ModuleType("flash_attn_interface")
    rotary = ModuleType("flash_attn.layers.rotary")
    for module in (fa2, fa3):
        module.flash_attn_func = attention
        module.flash_attn_varlen_func = attention
    rotary.apply_rotary_emb = attention
    for name, module in {
        "flash_attn": fa2,
        "flash_attn_interface": fa3,
        "flash_attn_3": ModuleType("flash_attn_3"),
        "flash_attn.layers": ModuleType("flash_attn.layers"),
        "flash_attn.layers.rotary": rotary,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(kernels.logger, "info", messages.append)
    monkeypatch.setattr(kernels.logger, "warning", messages.append)
    kernels.init_flash_attn_meta(deterministic, compile, requested)
    meta = kernels.FlashAttnMeta()
    assert meta.fa_version == expected
    sentinel = object()
    assert meta.flash_attn_func(sentinel, None, None) is sentinel
    assert (
        meta.flash_attn_varlen_func(sentinel, None, None, None, None, 1, 1) is sentinel
    )
    assert all(call["deterministic"] == deterministic for call in calls)
    assert any(f"Using FlashAttention-{expected}." in message for message in messages)
    assert any(f"deterministic={deterministic}" in message for message in messages)
    assert any("overrides deterministic" in message for message in messages) == (
        deterministic and expected == 3
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires native GPU attention"
)
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("kv_heads", [16, 2])
def test_deterministic_auto_backend_repeats_actual_gradients(variable, kv_heads):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("FA2 requires Ampere or newer")
    kernels.init_flash_attn_meta(deterministic=True, compile=False)
    meta = kernels.FlashAttnMeta()
    assert meta.fa_version == 2
    generator = torch.Generator(device="cuda").manual_seed(41)
    shape = (2, 2048, 16, 128)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(
            (2, 2048, kv_heads, 128), generator=generator, device="cuda", dtype=q.dtype
        )
        for _ in range(2)
    ]
    dout = torch.randn(shape, generator=generator, device="cuda", dtype=q.dtype)
    if variable:
        q, k, v, dout = [x.flatten(0, 1) for x in (q, k, v, dout)]
    offsets = torch.tensor([0, 2048, 4096], device="cuda", dtype=torch.int32)
    baseline = None
    for _ in range(8):
        inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        if variable:
            output = meta.flash_attn_varlen_func(
                *inputs, offsets, offsets, 2048, 2048, causal=True
            )
        else:
            output = meta.flash_attn_func(*inputs, causal=True)
        values = (output.detach(), *torch.autograd.grad(output, inputs, dout))
        if baseline is None:
            baseline = [x.clone() for x in values]
        else:
            for actual, expected in zip(values, baseline, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
