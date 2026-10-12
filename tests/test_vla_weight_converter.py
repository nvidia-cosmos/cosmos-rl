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


def test_export_wrapper_names_match_inner_model_without_losing_weights():
    from cosmos_rl.policy.model.vla.weight_converter import (
        normalize_vla_checkpoint_keys,
    )

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    exported = {
        f"model.{key}": value.clone() for key, value in model.state_dict().items()
    }
    restored = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    restored.load_state_dict(
        normalize_vla_checkpoint_keys(exported, restored.state_dict().keys()),
        strict=True,
    )
    inputs = torch.randn(5, 3)
    torch.testing.assert_close(restored(inputs), model(inputs))
    hf = model.state_dict()
    assert normalize_vla_checkpoint_keys(hf, hf.keys()).keys() == hf.keys()
    try:
        normalize_vla_checkpoint_keys(
            {"model.0.weight": hf["0.weight"], "0.weight": hf["0.weight"]}, hf.keys()
        )
    except ValueError as error:
        assert "Ambiguous" in str(error)
    else:
        raise AssertionError("Ambiguous checkpoint names were accepted")
