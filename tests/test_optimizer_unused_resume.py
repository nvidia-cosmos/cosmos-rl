"""Adam checkpoint resume with trainable parameters unused by the loss."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import torch


def test_resume_preserves_adam_state_with_previously_unused_parameters():
    from cosmos_rl.policy.trainer.optm import OptimizersContainer

    torch.manual_seed(17)
    model = torch.nn.ModuleDict(
        {"used": torch.nn.Linear(2, 2), "unused": torch.nn.Linear(2, 2)}
    )

    def optimizer(module):
        return OptimizersContainer(
            torch.optim.AdamW, [module], [{"lr": 0.01, "fused": True}]
        )

    original = optimizer(model)
    inputs = torch.ones(2, 2)
    model["used"](inputs).sum().backward()
    original.step()
    original.zero_grad(set_to_none=True)
    checkpoint = copy.deepcopy(original.state_dict())
    assert not any(k.startswith("idx-0-state.unused.") for k in checkpoint)
    resumed_model = copy.deepcopy(model)
    resumed = optimizer(resumed_model)
    resumed.load_state_dict(checkpoint)
    # A previously unused layer may receive gradients after resume. Its Adam
    # step must start at zero, while the used layer retains its prior moments.
    for module, opt in [(model, original), (resumed_model, resumed)]:
        (module["used"](inputs) + module["unused"](inputs)).sum().backward()
        opt.step()
    for expected, actual in zip(model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, expected in original.state_dict().items():
        actual = resumed.state_dict()[name]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected


def test_explicit_resume_failure_does_not_fall_back_to_initial_weights():
    from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer

    failure = FileNotFoundError("checkpoint restore failed")
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                resume="/explicit/checkpoint",
                train_policy=SimpleNamespace(kl_beta=0.0),
            )
        ),
        model_resume_from_checkpoint=Mock(side_effect=failure),
        model_load_from_hf=Mock(),
    )
    try:
        GRPOTrainer.weight_resume(trainer)
    except FileNotFoundError as error:
        assert error is failure
    else:
        raise AssertionError("Explicit resume failure was swallowed")
    trainer.model_load_from_hf.assert_not_called()
