# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from cosmos_rl.dispatcher.data.schema import Rollout
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cosmos_rl.execution.config import ObjectiveConfig
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from cosmos_rl.execution.source import ExactCorpusSampler
from cosmos_rl.execution.token_objectives import TokenObjectiveBatch


def configuration(**objectives):
    return SimpleNamespace(
        execution=SimpleNamespace(
            objectives=ObjectiveConfig(**objectives), source=None
        ),
        policy=SimpleNamespace(model_max_length=32),
    )


def sample(ids, *, supervised=True, rl=True):
    n = len(ids) - 1
    record = SampleRecord(
        sample_id=str(ids),
        occurrence_id=str(ids),
        group_id="group",
        producer=ModelIdentity(
            role="actor",
            name="old-policy",
            revision="old-checkpoint",
            behavior_version=2,
        ),
        masks={"supervised": [supervised] * n, "policy_gradient": [rl] * n},
        annotations={},
        rollout=Rollout(
            prompt={"image": torch.zeros(3, 2, 2)},
            completion={"input_ids": ids, "label_ids": ids},
            completion_logprobs=[-1.4] * n,
            advantage=[0.3] * n,
        ),
    )
    return {"input_ids": ids, "label_ids": ids, "_execution_record": record}


def test_mixed_gradient_is_independent_of_microbatch_partition():
    rows = [sample([0, 1, 2]), sample([1, 0]), sample([2, 1, 0, 2], rl=False)]
    config = configuration(supervised=0.7, policy_gradient=0.3)
    initial = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 10

    def update(groups):
        model = torch.nn.Embedding.from_pretrained(initial.clone(), freeze=False)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.03
        )
        window = TokenObjectiveBatch(rows, config, device="cpu")
        offset = 0
        for group in groups:
            selected = rows[offset : offset + group]
            length = max(len(r["input_ids"]) for r in selected)
            inputs = torch.tensor(
                [
                    r["input_ids"] + [0] * (length - len(r["input_ids"]))
                    for r in selected
                ]
            )
            labels = torch.tensor(
                [
                    r["label_ids"] + [-100] * (length - len(r["label_ids"]))
                    for r in selected
                ]
            )
            window.loss(model(inputs), labels, start=offset).backward()
            offset += group
        optimizer.step()
        return model.weight.detach(), optimizer.state_dict()

    actual, actual_state = update([1, 2])
    reference_model = torch.nn.Embedding.from_pretrained(initial.clone(), freeze=False)
    optimizer = torch.optim.SGD(
        reference_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.03
    )
    ce, pg = [], []
    for row in rows:
        ids = torch.tensor(row["input_ids"])
        logits = reference_model(ids)[:-1]
        ce.append(F.cross_entropy(logits, ids[1:], reduction="none"))
        if row["_execution_record"].masks["policy_gradient"][0]:
            current = logits.log_softmax(-1).gather(-1, ids[1:, None]).squeeze(-1)
            ratio = (current + 1.4).exp()
            pg.append(-torch.minimum(ratio * 0.3, ratio.clamp(0.8, 1.2) * 0.3))
    (0.7 * torch.cat(ce).mean() + 0.3 * torch.cat(pg).mean()).backward()
    optimizer.step()
    assert torch.allclose(actual, reference_model.weight, atol=1e-7)
    other, _ = update([3])
    assert torch.allclose(actual, other, atol=1e-7)
    assert torch.allclose(
        actual_state["state"][0]["momentum_buffer"],
        optimizer.state_dict()["state"][0]["momentum_buffer"],
        atol=1e-7,
    )


def test_zero_coefficient_does_not_require_behavior_or_teacher_targets():
    row = {"input_ids": [0, 1], "label_ids": [0, 1]}
    window = TokenObjectiveBatch([row], configuration(supervised=1), device="cpu")
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    window.loss(logits, torch.tensor([[0, 1]]), start=0).backward()
    assert logits.grad.abs().sum() > 0


def test_all_masked_window_is_empty_and_quality_does_not_erase_supervision():
    row = sample([0, 1], supervised=False, rl=False)
    window = TokenObjectiveBatch(
        [row], configuration(supervised=1, policy_gradient=1), device="cpu"
    )
    assert window.window.empty
    row = sample([0, 1], supervised=True, rl=False)
    row["_execution_record"].annotations["quality_trainable"] = False
    window = TokenObjectiveBatch(
        [row], configuration(supervised=1, policy_gradient=1), device="cpu"
    )
    assert dict(
        zip((name for name, _ in window.window.coefficients), window.window.counts)
    ) == {"policy_gradient": 0, "supervised": 1}


def test_exact_sampler_keeps_every_real_record_once_and_masks_schedule_padding():
    schedules = [
        list(ExactCorpusSampler(range(5), num_replicas=3, rank=r)) for r in range(3)
    ]
    assert [len(row) for row in schedules] == [2, 2, 2]
    assert sorted(i for row in schedules for i in row if i is not None) == list(
        range(5)
    )
    assert sum(i is None for row in schedules for i in row) == 1


def test_missing_behavior_rejected_before_forward():
    row = sample([0, 1])
    row["_execution_record"].rollout.completion_logprobs = None
    with pytest.raises(ValueError, match="behavior"):
        TokenObjectiveBatch([row], configuration(policy_gradient=1), device="cpu")


def test_fractional_reward_mass_is_the_supervised_denominator():
    row = sample([0, 1])
    row["_execution_record"].rollout.reward = 0.25
    config = configuration(supervised=1)
    config.execution.source = SimpleNamespace(reward_weighting="nonnegative")
    window = TokenObjectiveBatch([row], config, device="cpu")
    logits = torch.tensor([[[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]]], requires_grad=True)
    loss = window.loss(logits, torch.tensor([[0, 1]]), start=0)
    assert torch.allclose(loss, F.cross_entropy(logits[0, :1], torch.tensor([1])))


def test_cached_distillation_and_supervision_match_reference_gradient():
    row = sample([0, 1, 2], rl=False)
    record = row["_execution_record"]
    teacher = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    record.rollout.teacher_logprobs = teacher.log()
    record.annotations.update(
        teacher=ModelIdentity(role="teacher", name="fixed").model_dump()
    )
    config = configuration(supervised=0.4, distillation=0.6)
    logits = torch.randn(1, 3, 3, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    window = TokenObjectiveBatch([row], config, device="cpu")
    window.loss(logits, torch.tensor([[0, 1, 2]]), start=0).backward()
    loss = (
        0.4 * F.cross_entropy(reference[0, :-1], torch.tensor([1, 2]))
        + 0.6
        * F.kl_div(reference[0, :-1].log_softmax(-1), teacher, reduction="sum")
        / 2
    )
    loss.backward()
    assert torch.allclose(logits.grad, reference.grad, atol=1e-7)


def test_clipped_positive_advantage_cannot_overflow_an_unselected_ratio():
    from cosmos_rl.execution.objectives import clipped_policy_loss

    current = torch.tensor([-1.0], requires_grad=True)
    loss = clipped_policy_loss(current, torch.tensor([-10001.0]), torch.tensor([1.0]))
    loss.sum().backward()
    assert loss.item() == pytest.approx(-1.2)
    assert current.grad.item() == 0
