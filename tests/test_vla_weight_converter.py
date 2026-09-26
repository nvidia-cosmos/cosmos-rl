"""Standalone evaluation must retain checkpoint weights without a device mesh."""

import torch

from cosmos_rl.policy.model.vla.weight_converter import convert_weight_from_hf


def test_standalone_proprio_weight_conversion():
    weight = torch.arange(28, dtype=torch.float32).reshape(14, 2).T
    name, converted = convert_weight_from_hf(
        weight, "proprio_projector.fc1.weight", None
    )
    assert name == "proprio_projector.fc1.weight"
    torch.testing.assert_close(converted, weight)
    assert converted.is_contiguous()
