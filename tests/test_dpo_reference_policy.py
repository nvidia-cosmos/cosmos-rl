# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reference-free compatibility and opt-in frozen-reference objective/resume."""

import copy
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from cosmos_rl.policy.config import Config
from cosmos_rl.policy.model.base import CosmosModelOutput
from cosmos_rl.policy.trainer.llm_trainer.dpo_trainer import DPOTrainer, dpo_loss


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4, dtype=torch.float64)
        self.head = torch.nn.Linear(4, 5, dtype=torch.float64)
        self.register_buffer("offset", torch.tensor(0.1, dtype=torch.float64))
        self.calls = []

    def get_position_ids(self, input_ids, **kwargs):
        return torch.arange(input_ids.shape[1]).expand_as(input_ids), None, None

    def forward(self, input_ids, **kwargs):
        self.calls.append((torch.is_grad_enabled(), self.training))
        return CosmosModelOutput(
            logits=self.head(self.embedding(input_ids)) + self.offset
        )


def make_trainer(reference=True):
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available(), "Required GPU validation may not skip"
    torch.manual_seed(43)
    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.model = TinyPolicy().to(device)
    trainer.model_parts = [trainer.model]
    trainer.device = device
    trainer.use_reference_policy = reference
    trainer.reference_state_dict = {}
    trainer.beta = 0.3
    trainer.loss_types = ["sigmoid"]
    trainer.loss_weights = [1.0]
    trainer.act_offloading_ctx_manager = nullcontext()
    if reference:
        trainer._capture_reference()
    with torch.no_grad():
        trainer.model.head.weight.add_(0.2)
        trainer.model.offset.add_(0.3)
    return trainer


def batches(device):
    chosen = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [2, 1, 0, 0]], device=device),
        "logprob_masks": torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]], device=device),
    }
    rejected = {
        "input_ids": torch.tensor([[1, 2, 4, 0], [2, 3, 1, 0]], device=device),
        "logprob_masks": torch.tensor([[0, 0, 1, 0], [0, 1, 1, 0]], device=device),
    }
    return chosen, rejected


def response_sums(model, batch):
    logits = model(**batch).logits[:, :-1].log_softmax(-1)
    values = logits.gather(-1, batch["input_ids"][:, 1:, None]).squeeze(-1)
    return (values * batch["logprob_masks"][:, 1:]).sum(-1)


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize(
    "loss_types", [["sigmoid"], ["bco_pair"], ["sigmoid", "bco_pair", "sft"]]
)
def test_actual_forward_loss_and_gradients_match_independent_reference(
    reference, loss_types
):
    trainer = make_trainer(reference)
    trainer.loss_types = loss_types
    trainer.loss_weights = [0.7, 0.2, 0.4][: len(loss_types)]
    chosen, rejected = batches(trainer.device)
    policy = copy.deepcopy(trainer.model)
    chosen_logps, rejected_logps = (
        response_sums(policy, chosen),
        response_sums(policy, rejected),
    )
    if reference:
        frozen = copy.deepcopy(trainer.model)
        frozen.load_state_dict(trainer.reference_state_dict)
        with torch.no_grad():
            ref_chosen = response_sums(frozen, chosen)
            ref_rejected = response_sums(frozen, rejected)
        chosen_logps = chosen_logps - ref_chosen
        rejected_logps = rejected_logps - ref_rejected
    terms = {
        "sigmoid": -F.logsigmoid(trainer.beta * (chosen_logps - rejected_logps)).mean(),
        "bco_pair": (
            -F.logsigmoid(trainer.beta * chosen_logps)
            - F.logsigmoid(-trainer.beta * rejected_logps)
        ).mean(),
    }
    logits = policy(**chosen).logits[:, :-1]
    mask = chosen["logprob_masks"][:, 1:].bool()
    terms["sft"] = F.cross_entropy(logits[mask], chosen["input_ids"][:, 1:][mask])
    expected = sum(
        weight * terms[kind] for weight, kind in zip(trainer.loss_weights, loss_types)
    )
    actual = trainer._dpo_forward_and_loss(chosen, rejected)
    torch.testing.assert_close(actual.squeeze(), expected)
    actual.backward()
    expected.backward()
    for actual_param, expected_param in zip(
        trainer.model.parameters(), policy.parameters()
    ):
        torch.testing.assert_close(actual_param, expected_param)
        torch.testing.assert_close(actual_param.grad, expected_param.grad)
    assert trainer.model.calls == (
        [(False, False), (True, True)] if reference else [(True, True)]
    )
    assert all(
        not value.requires_grad for value in trainer.reference_state_dict.values()
    )


def test_default_is_reference_free_and_reference_logps_are_detached():
    assert not Config().train.train_policy.dpo_reference_policy
    values = [torch.tensor([v], requires_grad=True) for v in (-2.0, -3.0, -4.0, -5.0)]
    loss = dpo_loss(
        *values[:2],
        reference_chosen_logps=values[2],
        reference_rejected_logps=values[3],
    )
    loss.backward()
    assert values[0].grad is not None and values[1].grad is not None
    assert values[2].grad is None and values[3].grad is None
    with pytest.raises(ValueError, match="Both"):
        dpo_loss(*values[:2], reference_chosen_logps=values[2])


def test_frozen_cpu_snapshot_does_not_alias_live_model():
    trainer = make_trainer()
    assert trainer.reference_state_dict["offset"].item() == pytest.approx(0.1)
    assert trainer.model.offset.item() == pytest.approx(0.4)


def test_reference_restore_uses_current_parameter_storage():
    trainer = make_trainer()
    before = copy.deepcopy(trainer.model.state_dict())
    with trainer._reference_forward():
        # FSDP's lazy initialization may replace local shard storage while
        # preserving the Parameter object and optimizer ownership.
        trainer.model.head.weight.data = trainer.model.head.weight.detach().clone()
        trainer.model.offset = trainer.model.offset.clone()
    torch.testing.assert_close(trainer.model.state_dict(), before, rtol=0, atol=0)


def test_reference_failure_restores_policy_buffers_modes_and_rng():
    trainer = make_trainer()
    trainer.model.head.eval()
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    modes = [module.training for module in trainer.model.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        torch.cuda.get_rng_state(trainer.device)
        if trainer.device.type == "cuda"
        else None
    )
    with pytest.raises(RuntimeError, match="injected"):
        with trainer._reference_forward():
            assert not trainer.model.training
            torch.rand(3)
            torch.rand(3, device=trainer.device)
            trainer.model.offset.add_(99)
            raise RuntimeError("injected reference forward failure")
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert modes == [module.training for module in trainer.model.modules()]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(trainer.device))


@pytest.mark.parametrize("bad", ["absent", "missing", "shape", "dtype", "mode"])
def test_resume_rejects_incompatible_reference_state(bad):
    trainer = make_trainer()
    state = copy.deepcopy(trainer.reference_state_dict)
    payload = {"dpo_reference_policy": True, "dpo_reference_state": state}
    if bad == "absent":
        payload = {}
    elif bad == "missing":
        state.pop("offset")
    elif bad == "shape":
        state["offset"] = torch.zeros(3, dtype=torch.float64)
    elif bad == "dtype":
        state["offset"] = state["offset"].float()
    else:
        trainer.use_reference_policy = False
    with pytest.raises(ValueError, match="DPO"):
        trainer._restore_reference(payload)


def test_new_mode_saves_reference_at_normal_and_terminal_boundaries():
    trainer = make_trainer()
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(ckpt=SimpleNamespace(enable_checkpoint=True))
    )
    trainer.parallel_dims = SimpleNamespace(dp_replicate_coord=(0, 1))
    trainer.optimizers, trainer.lr_schedulers = object(), object()
    trainer.ckpt_manager = Mock()
    trainer.checkpointing(total_steps=3, train_step=1, save_freq=2)
    trainer.ckpt_manager.save_checkpoint.assert_not_called()
    trainer.checkpointing(total_steps=3, train_step=2, save_freq=2)
    trainer.checkpointing(total_steps=3, train_step=3, save_freq=2, is_last_step=True)
    calls = trainer.ckpt_manager.save_checkpoint.call_args_list
    assert [call.kwargs["step"] for call in calls] == [2, 3]
    assert [call.kwargs["is_final"] for call in calls] == [False, True]
    assert all(
        call.kwargs["dpo_reference_state"] is trainer.reference_state_dict
        for call in calls
    )
    trainer.use_reference_policy = False
    trainer.checkpointing(total_steps=3, train_step=3, save_freq=2, is_last_step=True)
    assert trainer.ckpt_manager.save_checkpoint.call_count == 3
    assert (
        trainer.ckpt_manager.save_checkpoint.call_args.kwargs["dpo_reference_state"]
        is None
    )
    assert not trainer.ckpt_manager.save_checkpoint.call_args.kwargs[
        "dpo_reference_policy"
    ]


def test_next_optimizer_update_matches_after_serialized_reference_resume(tmp_path):
    trainer = make_trainer()
    chosen, rejected = batches(trainer.device)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.02)
    trainer._dpo_forward_and_loss(chosen, rejected).backward()
    optimizer.step()
    optimizer.zero_grad()
    path = tmp_path / "resume.pt"
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dpo_reference_policy": True,
            "dpo_reference_state": trainer.reference_state_dict,
        },
        path,
    )
    resumed = make_trainer()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    resumed.model.load_state_dict(checkpoint.pop("model"))
    restored_optimizer = torch.optim.AdamW(resumed.model.parameters(), lr=0.9)
    restored_optimizer.load_state_dict(checkpoint.pop("optimizer"))
    resumed._restore_reference(checkpoint)
    expected, actual = [
        item._dpo_forward_and_loss(chosen, rejected) for item in (trainer, resumed)
    ]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.backward()
    actual.backward()
    optimizer.step()
    restored_optimizer.step()
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[key], rtol=0, atol=0
        )


def test_opt_in_resume_error_does_not_fall_back_to_hf():
    trainer = make_trainer()
    trainer.parallel_dims = SimpleNamespace(dp_replicate_enabled=False)
    trainer.config = SimpleNamespace(train=SimpleNamespace(resume="checkpoint"))
    trainer.model_resume_from_checkpoint = Mock(side_effect=ValueError("broken"))
    trainer.model_load_from_hf = Mock()
    with pytest.raises(RuntimeError, match="refusing"):
        trainer.load_model()
    trainer.model_load_from_hf.assert_not_called()


@pytest.mark.parametrize("reference", [False, True])
def test_empty_responses_produce_finite_zero_gradients(reference):
    trainer = make_trainer(reference)
    chosen, rejected = batches(trainer.device)
    chosen["logprob_masks"].zero_()
    rejected["logprob_masks"].zero_()
    loss = trainer._dpo_forward_and_loss(chosen, rejected)
    assert torch.isfinite(loss).all()
    loss.backward()
    for param in trainer.model.parameters():
        assert param.grad is not None and not torch.count_nonzero(param.grad)


@pytest.mark.parametrize("save_mode", ["sync", "async"])
def test_real_checkpoint_manager_and_load_entrypoint_restore_reference(
    tmp_path, save_mode
):
    from cosmos_rl.policy.trainer.optm import build_optimizers, build_lr_schedulers
    from cosmos_rl.utils.checkpoint import CheckpointMananger

    trainer = make_trainer()
    trainer.config = Config.from_dict(
        {
            "train": {
                "output_dir": str(tmp_path / "run"),
                "resume": False,
                "optm_lr": 0.02,
                "optm_name": "AdamW",
                "train_policy": {"dpo_reference_policy": True},
                "ckpt": {
                    "enable_checkpoint": True,
                    "save_mode": save_mode,
                    "upload_s3": False,
                },
            }
        }
    )
    trainer.parallel_dims = SimpleNamespace(
        dp_replicate_enabled=False, dp_replicate_coord=(0, 1), pp_enabled=False
    )
    trainer.optimizers = build_optimizers([trainer.model], trainer.config)
    trainer.lr_schedulers = build_lr_schedulers(trainer.optimizers, trainer.config, 4)
    trainer.ckpt_manager = CheckpointMananger(trainer.config)
    chosen, rejected = batches(trainer.device)
    trainer._dpo_forward_and_loss(chosen, rejected).backward()
    trainer.optimizers.step()
    trainer.optimizers.zero_grad()
    trainer.lr_schedulers.step()
    trainer.checkpointing(total_steps=4, train_step=1, save_freq=1)
    trainer.ckpt_manager._wait_for_pending_async_saves()
    if save_mode == "async":
        trainer.ckpt_manager.executor.shutdown(wait=True)

    resumed = make_trainer()
    resumed.config = trainer.config.model_copy(deep=True)
    resumed.config.train.resume = os.path.join(
        trainer.config.train.output_dir, "checkpoints", "step_1", "policy"
    )
    resumed.parallel_dims = trainer.parallel_dims
    resumed.optimizers = build_optimizers([resumed.model], resumed.config)
    resumed.lr_schedulers = None
    resumed.ckpt_manager = CheckpointMananger(resumed.config)
    try:
        total, step, metadata = resumed.load_model()
        assert (total, step) == (4, 1)
        assert "dpo_reference_state" not in metadata
        for key, value in trainer.reference_state_dict.items():
            torch.testing.assert_close(
                value, resumed.reference_state_dict[key], rtol=0, atol=0
            )
        for item in (trainer, resumed):
            item._dpo_forward_and_loss(chosen, rejected).backward()
            item.optimizers.step()
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(
                value, resumed.model.state_dict()[key], rtol=0, atol=0
            )
    finally:
        if save_mode == "async":
            resumed.ckpt_manager.executor.shutdown(wait=True)
