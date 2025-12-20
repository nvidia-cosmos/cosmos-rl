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

import re
import torch.nn as nn
import unittest
from typing import List

from cosmos_rl.policy.model.base import BaseModel


def apply_freeze_pattern(model: nn.Module, freeze_pattern: List[str]) -> dict:
    """Call the real implementation (BaseModel.apply_freeze_pattern) for test coverage."""
    return BaseModel.apply_freeze_pattern(model, freeze_pattern)  # type: ignore[arg-type]


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.ModuleDict({"blocks": nn.ModuleList([_Block(), _Block()])})
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(100, 8),
                "layers": nn.ModuleList([_Block() for _ in range(4)]),
            }
        )
        self.lm_head = nn.Linear(8, 100)


class FreezePatternTest(unittest.TestCase):
    def test_freeze_visual(self):
        """Freeze all visual components."""
        model = _DummyModel()
        apply_freeze_pattern(model, [r"visual\..*"])

        for name, param in model.named_parameters():
            if name.startswith("visual."):
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should be trainable"

    def test_freeze_specific_layers(self):
        """Freeze layers 0-1, keep 2-3 trainable."""
        model = _DummyModel()
        apply_freeze_pattern(model, [r"model\.layers\.[0-1]\."])

        for name, param in model.named_parameters():
            if "model.layers.0." in name or "model.layers.1." in name:
                assert not param.requires_grad, f"{name} should be frozen"
            elif "model.layers.2." in name or "model.layers.3." in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_freeze_attention_only(self):
        """Freeze attention, keep MLP trainable."""
        model = _DummyModel()
        apply_freeze_pattern(model, ["self_attn"])

        for name, param in model.named_parameters():
            if "self_attn" in name:
                assert not param.requires_grad, f"{name} should be frozen"
            elif "mlp" in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_multiple_patterns(self):
        """Multiple patterns in list."""
        model = _DummyModel()
        out = apply_freeze_pattern(model, [r"visual\..*", "embed_tokens"])
        counts = out["pattern_counts"]

        assert counts[r"visual\..*"] > 0
        assert counts["embed_tokens"] > 0

        for name, param in model.named_parameters():
            if name.startswith("visual.") or "embed_tokens" in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_empty_pattern(self):
        """Empty pattern list should not freeze anything."""
        model = _DummyModel()
        out = apply_freeze_pattern(model, [])
        counts = out["pattern_counts"]
        assert counts == {}

        for param in model.parameters():
            assert param.requires_grad

    def test_overlap_pattern_priority_order(self):
        """When multiple patterns match, the first match wins (due to break)."""
        model = _DummyModel()
        p1 = r"self_attn"
        p2 = r".*self_attn.*"
        out = apply_freeze_pattern(model, [p1, p2])
        counts = out["pattern_counts"]

        # All self_attn params should be attributed to the first pattern only.
        assert counts[p1] > 0
        assert counts[p2] == 0

        for name, param in model.named_parameters():
            if "self_attn" in name:
                assert not param.requires_grad, f"{name} should be frozen"

        # Swap order: attribution should flip.
        model2 = _DummyModel()
        out2 = apply_freeze_pattern(model2, [p2, p1])
        counts2 = out2["pattern_counts"]
        assert counts2[p2] > 0
        assert counts2[p1] == 0

    def test_unmatched_pattern_count_is_zero(self):
        """A pattern that matches nothing should have count 0 and freeze nothing."""
        model = _DummyModel()
        out = apply_freeze_pattern(model, [r"does_not_exist_anywhere_12345"])
        counts = out["pattern_counts"]

        assert counts[r"does_not_exist_anywhere_12345"] == 0
        for param in model.parameters():
            assert param.requires_grad

    def test_empty_string_and_none_patterns_are_ignored(self):
        """Empty/None patterns should be ignored (not compiled, not counted)."""
        model = _DummyModel()
        out = apply_freeze_pattern(
            model,
            ["", None, r"visual\..*"],  # type: ignore[list-item]
        )
        counts = out["pattern_counts"]

        assert "" not in counts
        assert None not in counts
        assert counts[r"visual\..*"] > 0

        for name, param in model.named_parameters():
            if name.startswith("visual."):
                assert not param.requires_grad, f"{name} should be frozen"

    def test_invalid_regex_raises(self):
        """Invalid regex patterns should raise re.error during compile."""
        model = _DummyModel()
        with self.assertRaises(re.error):
            apply_freeze_pattern(model, [r"["])


if __name__ == "__main__":
    unittest.main()
