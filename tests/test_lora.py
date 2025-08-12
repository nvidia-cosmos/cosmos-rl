# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import unittest

from cosmos_rl.policy.lora.plugin import (
    inject_lora_adapters,
    mark_only_lora_as_trainable,
    merge_lora_weights_,
    unmerge_lora_weights_,
    lora_state_dict,
    load_lora_state_dict,
    LoraInjectedLinear,
)
from cosmos_rl.policy.config import LoraConfig


def _base_state_dict(m):
    # drop any lora_* tensors; keep only base weights/biases
    return {k: v for k, v in m.state_dict().items() if "lora_" not in k}


class TinyBlock(nn.Module):
    def __init__(self, in_f=32, hidden=64, out_f=16):
        super().__init__()
        self.q_proj = nn.Linear(in_f, hidden)
        self.act = nn.GELU()
        self.v_proj = nn.Linear(hidden, out_f)

    def forward(self, x):
        return self.v_proj(self.act(self.q_proj(x)))


class LoRATest(unittest.TestCase):
    def test_injection_and_freeze(self):
        model = TinyBlock()
        cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]
        )
        model, replaced = inject_lora_adapters(model, cfg)

        # Both should be replaced
        assert any("q_proj" in n for n in replaced)
        assert any("v_proj" in n for n in replaced)
        assert isinstance(model.q_proj, LoraInjectedLinear)
        assert isinstance(model.v_proj, LoraInjectedLinear)

        # Freeze base, enable only LoRA params
        mark_only_lora_as_trainable(model, cfg)
        base_params = []
        lora_params = []
        for n, p in model.named_parameters():
            if "lora_" in n:
                lora_params.append((n, p))
            else:
                base_params.append((n, p))
        assert all(not p.requires_grad for _, p in base_params)
        assert all(p.requires_grad for _, p in lora_params)

    def test_forward_equivalence_merge_unmerge(self):
        x = torch.randn(4, 32)
        model = TinyBlock()
        cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]
        )
        model, _ = inject_lora_adapters(model, cfg)
        mark_only_lora_as_trainable(model, cfg)

        # Start: B=0 => LoRA has no effect; merge should be no-op
        with torch.no_grad():
            out0 = model(x)
        merge_lora_weights_(model)
        with torch.no_grad():
            out1 = model(x)
        assert torch.allclose(out0, out1, atol=0, rtol=0)

        # Create a non-zero LoRA: set A random, B random
        for m in model.modules():
            if isinstance(m, LoraInjectedLinear):
                with torch.no_grad():
                    torch.nn.init.normal_(m.lora_B.weight, std=0.02)

        unmerge_lora_weights_(model)  # ensure unmerged mode
        with torch.no_grad():
            out_unmerged = model(x)

        # Merge and check equivalence
        merge_lora_weights_(model)
        with torch.no_grad():
            out_merged = model(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-6, rtol=1e-6)

        # Unmerge should restore pre-merge base weights' effect (so outputs equal again)
        unmerge_lora_weights_(model)
        with torch.no_grad():
            out_unmerged2 = model(x)
        assert torch.allclose(out_unmerged, out_unmerged2, atol=1e-6, rtol=1e-6)

    def test_grads_only_on_lora(self):
        x = torch.randn(8, 32)
        y = torch.randn(8, 16)

        model = TinyBlock()
        cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]
        )
        model, _ = inject_lora_adapters(model, cfg)
        mark_only_lora_as_trainable(model, cfg)

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        loss_fn = nn.MSELoss()

        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        # No grads on base
        for n, p in model.named_parameters():
            if "lora_" not in n:
                assert p.grad is None, f"Base param {n} unexpectedly has grad"
            else:
                print(f"Base param {n} has grad: {p.grad}")
                assert p.grad is not None, f"LoRA param {n} missing grad"

        # Optim step should change only LoRA params
        old_base = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if "lora_" not in n
        }
        old_lora = {
            n: p.clone().detach() for n, p in model.named_parameters() if "lora_" in n
        }

        opt.step()

        with torch.no_grad():
            # Base stays identical
            for n, p in model.named_parameters():
                if "lora_" not in n:
                    assert torch.equal(
                        p, old_base[n]
                    ), f"Base param {n} changed unexpectedly"

            # LoRA must change (use exact inequality, not allclose)
            for n, p in model.named_parameters():
                if "lora_" in n:
                    assert not torch.equal(
                        p, old_lora[n]
                    ), f"LoRA param {n} did not change"

    def test_save_and_load_lora_only(self):
        model = TinyBlock()
        cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]
        )
        model, _ = inject_lora_adapters(model, cfg)
        mark_only_lora_as_trainable(model, cfg)

        # Make non-zero LoRA
        for m in model.modules():
            if isinstance(m, LoraInjectedLinear):
                with torch.no_grad():
                    torch.nn.init.normal_(m.lora_B.weight, std=0.05)

        # Save only LoRA tensors
        state = lora_state_dict(model, prefix="lora")

        # === Important: align base weights ===
        model2 = TinyBlock()
        model2.load_state_dict(_base_state_dict(model), strict=False)

        # Inject LoRA on model2, then load the LoRA-only checkpoint
        inject_lora_adapters(model2, cfg)
        load_lora_state_dict(model2, state, prefix="lora", strict=True)

        # LoRA params should match exactly
        for (n1, m1), (n2, m2) in zip(model.named_modules(), model2.named_modules()):
            if isinstance(m1, LoraInjectedLinear):
                assert isinstance(m2, LoraInjectedLinear)
                assert torch.allclose(m1.lora_A.weight, m2.lora_A.weight)
                assert torch.allclose(m1.lora_B.weight, m2.lora_B.weight)

        # And with the same base, forward should match
        x = torch.randn(5, 32)
        with torch.no_grad():
            y1 = model(x)
            y2 = model2(x)
        assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
