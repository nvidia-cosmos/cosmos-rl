# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Packing preserves causal context, every objective and embedding gradients."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from cosmos_rl.execution.config import ExecutionConfig
from cosmos_rl.execution.token_objectives import TokenObjectiveBatch
from cosmos_rl.utils.sequence_packing import (
    pack_sequences_for_inputs,
    pack_sequences_for_labels,
    pack_sequences_info_collect,
)
from test_execution_native_sft import cpu_runtime, trainer  # noqa: F401
from test_execution_objectives import sample


class CausalModel(torch.nn.Module):
    """Real causal attention, using the production embedding packing helper."""

    def __init__(self):
        super().__init__()
        self.tokens = torch.nn.Embedding(3, 8)
        self.project = torch.nn.Linear(8, 3)

    def forward(self, input_ids, valid_input_len=None, **kwargs):
        h = self.tokens(input_ids)
        if valid_input_len is not None:
            h = pack_sequences_for_inputs(h, valid_input_len)["inputs"]
            segments = torch.repeat_interleave(
                torch.arange(len(valid_input_len)), valid_input_len
            )
            allowed = segments[:, None] == segments[None, :]
        else:
            allowed = torch.ones(h.shape[1:2] * 2, dtype=torch.bool)
        allowed = allowed.tril()
        hidden = F.scaled_dot_product_attention(
            h[:, None], h[:, None], h[:, None], attn_mask=allowed
        )[:, 0]

        class Output(dict):
            __getattr__ = dict.__getitem__

        return Output(logits=self.project(hidden))

    def get_position_ids(self, input_ids, **kwargs):
        return torch.arange(input_ids.shape[1]), input_ids, 1

    def step_hook(self, step):
        return {}


def rows():
    result = [sample(ids) for ids in ([0, 1, 0], [2], [1, 2, 0, 2, 1], [0, 2])]
    for i, row in enumerate(result):
        row["token_ids"] = row["input_ids"]
        n = len(row["input_ids"]) - 1
        record = row["_execution_record"]
        record.annotations.update(
            teacher={"role": "teacher", "name": "teacher"},
            teacher_logprobs=torch.tensor([0.2, -0.3, 0.5])
            .log_softmax(-1)
            .expand(n, -1)
            .clone(),
            score=0.0 if i == 0 else i + 0.5,
        )
        record.masks["distillation"] = [i != 3] * n
        if i == 2:
            record.masks["policy_gradient"][1] = False
    return result


def configuration():
    return NS(
        execution=ExecutionConfig(
            path="train_from_data",
            source={"reward_weighting": "nonnegative"},
            objectives={"supervised": 0.6, "policy_gradient": 0.3, "distillation": 0.2},
        ),
        policy=NS(model_max_length=32),
    )


@pytest.mark.parametrize("groups", [(4,), (1, 3), (2, 2)])
def test_packed_objectives_match_causal_unpacked_gradients_and_optimizer(groups):
    torch.manual_seed(91)
    initial = CausalModel()
    samples = rows()

    def update(packed):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        objective = TokenObjectiveBatch(
            samples, configuration(), device="cpu", vocab_size=3
        )
        start, losses = 0, []
        for size in groups:
            selected = samples[start : start + size]
            width = max(len(row["input_ids"]) for row in selected)
            ids = torch.tensor(
                [r["input_ids"] + [0] * (width - len(r["input_ids"])) for r in selected]
            )
            labels = torch.tensor(
                [
                    r["label_ids"] + [-100] * (width - len(r["label_ids"]))
                    for r in selected
                ]
            )
            lengths = None
            if packed:
                lengths = pack_sequences_info_collect(ids, 0, labels)[
                    "valid_input_len"
                ].clamp_min(1)
                labels = pack_sequences_for_labels(labels, lengths)
            logits = model(ids, valid_input_len=lengths).logits
            loss = objective.loss(logits, labels, start=start, valid_input_len=lengths)
            losses.append(loss.detach())
            loss.backward()
            start += size
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        optimizer.step()
        return model, gradients, optimizer.state_dict(), sum(losses)

    reference, reference_grads, reference_state, reference_loss = update(False)
    actual, gradients, state, loss = update(True)
    torch.testing.assert_close(loss, reference_loss)
    assert gradients["tokens.weight"].abs().sum() > 0
    for name, value in gradients.items():
        torch.testing.assert_close(value, reference_grads[name], atol=2e-7, rtol=2e-5)
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(
            value, reference.state_dict()[name], atol=2e-7, rtol=2e-5
        )
    for key, values in state["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(
                value, reference_state["state"][key][name], atol=2e-7, rtol=2e-5
            )


@pytest.mark.usefixtures("cpu_runtime")
def test_native_sft_loop_packs_recorded_objectives():
    torch.manual_seed(32)
    initial = CausalModel()
    states = []
    for packed in (False, True):
        worker = trainer(copy.deepcopy(initial), mini_batch=3)
        worker.data_packer.pad_token_id = 0
        # The test packer carries an unused image; real attention ignores it.
        samples = rows()
        for row in samples:
            row["image"] = torch.zeros(3, 2, 2)
        worker.config.execution = configuration().execution
        worker.config.train.sequence_packing = packed
        worker.step_training(samples, total_steps=2, train_step=0, save_freq=1)
        states.append(copy.deepcopy(worker.model.state_dict()))
    for name, value in states[0].items():
        torch.testing.assert_close(value, states[1][name], atol=2e-7, rtol=2e-5)


def test_packed_loss_rejects_layout_that_drops_contributing_targets():
    samples = rows()[:1]
    objective = TokenObjectiveBatch(
        samples, configuration(), device="cpu", vocab_size=3
    )
    with pytest.raises(ValueError, match="omitted contributing"):
        objective.loss(
            torch.randn(1, 2, 3),
            torch.tensor([[0, 1]]),
            start=0,
            valid_input_len=torch.tensor([2]),
        )


def test_packing_preserves_embedding_autograd_and_inference_mode():
    embeds = torch.randn(2, 4, 3, requires_grad=True)
    lengths = torch.tensor([2, 3])
    packed = pack_sequences_for_inputs(embeds, lengths)["inputs"]
    packed.square().sum().backward()
    assert embeds.grad[0, 2:].count_nonzero() == 0
    assert embeds.grad[1, 3:].count_nonzero() == 0
    assert embeds.grad[:, :2].abs().sum() > 0
    with torch.no_grad():
        assert not pack_sequences_for_inputs(embeds, lengths)["inputs"].requires_grad


def test_masked_packing_placeholder_keeps_sequence_alignment():
    ids = torch.zeros(2, 8, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    lengths = pack_sequences_info_collect(
        ids, 0, labels, seq_len_multiple=8, min_input_len=1
    )["valid_input_len"]
    assert (lengths >= 1).all()
    assert int(lengths.sum()) % 8 == 0
