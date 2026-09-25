# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual GRPO PP adapter versus the unchanged non-PP minibatch objective."""

from collections import defaultdict
import os
from types import MethodType, SimpleNamespace

import pytest
import torch

from cosmos_rl.policy.config import Config
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import (
    GRPOTrainer,
    _swizzle_pp_grpo_forward,
    compute_loss,
)


@pytest.mark.parametrize("microbatch", [1, 2, 4])
@pytest.mark.parametrize("lengths", ["equal", "unequal", "empty"])
@pytest.mark.parametrize("variant", ["grpo", "gspo"])
@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize(
    "reduction",
    ["seq-mean-token-mean", "seq-mean-token-sum", "token-mean", "token-sum"],
)
def test_microbatches_preserve_existing_minibatch_objective(
    microbatch, lengths, reduction, variant, auxiliary
):
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available()
    config = Config.from_dict(
        {
            "train": {
                "output_dir": "/tmp/grpo-pipeline-probe",
                "logprob_dtype": "float32",
                "train_policy": {
                    "loss_type": reduction,
                    "variant": variant,
                    "kl_beta": 0.0,
                    "entropy_coeff": 0.03 if auxiliary else 0.0,
                    "positive_nll_coef": 0.1 if auxiliary else 0.0,
                },
            }
        }
    )
    trainer = SimpleNamespace(
        config=config,
        parallel_dims=SimpleNamespace(dp_enabled=False),
        old_per_token_logps=defaultdict(lambda: None),
        ref_per_token_logps=defaultdict(lambda: None),
        metrics=defaultdict(lambda: torch.zeros((), device=device)),
    )
    trainer.compute_logprobs = MethodType(GRPOTrainer.compute_logprobs, trainer)
    logits = (
        torch.linspace(-2, 3, 4 * 6 * 7, device=device)
        .sin()
        .reshape(4, 6, 7)
        .requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    ids = torch.arange(24, device=device).reshape(4, 6) % 7
    masks = torch.tensor([[False, True, True, True, True, False]] * 4, device=device)
    if lengths != "equal":
        masks[1, 2:] = False
        masks[2, 3:] = False
    if lengths == "empty":
        masks[3] = False
    positive = torch.tensor([False, True, False, True], device=device)
    advantages = torch.arange(1.0, 5.0, device=device)[:, None].expand_as(masks)
    batch = {"input_ids": ids, "logprob_masks": masks}
    ref_logps, cu, metrics = trainer.compute_logprobs(
        batch, reference, is_full_logits=True
    )
    factor = 0.5
    expected = compute_loss(
        ref_logps, ref_logps.detach(), None, advantages, cu, config, masks
    )[0]
    if auxiliary:
        positive_mask = positive.repeat_interleave(cu[1:] - cu[:-1])
        expected = expected - 0.03 * metrics["effective_entropy"]
        expected = expected - 0.1 * ref_logps[positive_mask].mean()
    expected = factor * expected
    repaired = hasattr(GRPOTrainer, "_prepare_pp_loss")
    if repaired:
        GRPOTrainer._prepare_pp_loss(trainer, masks, positive, None)
    parts = []
    for index, start in enumerate(range(0, 4, microbatch)):
        selection = slice(start, start + microbatch)
        parts.append(
            _swizzle_pp_grpo_forward(
                trainer,
                lambda **kwargs: logits[selection],
                config,
                None,
                None,
                input_ids=ids[selection],
                logprob_masks=masks[selection],
                mini_batch_ids=torch.zeros(microbatch, 1, dtype=torch.int64),
                micro_batch_ids=torch.full((microbatch, 1), index, dtype=torch.int64),
                loss_scaling=torch.full(
                    (microbatch, 1), factor if repaired else factor / microbatch
                ),
                is_computing_ref=torch.zeros(microbatch, dtype=torch.bool),
                is_computing_old_ahead=torch.zeros(microbatch, dtype=torch.bool),
                advantages=advantages[selection],
                positive_flags=positive[selection, None],
            ).sum()
        )
    actual = sum(parts)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("reduction", ["token-mean", "seq-mean-token-mean"])
def test_whole_minibatch_count_collectives_only_when_needed(
    monkeypatch, balanced, reduction
):
    config = Config.from_dict(
        {
            "train": {
                "output_dir": "/tmp/grpo-pipeline-counts",
                "train_policy": {
                    "type": "grpo",
                    "loss_type": reduction,
                    "balance_dp_token": balanced,
                },
            }
        }
    )
    group = object()
    calls = []

    def reduce_dp(value, *, group):
        calls.append("dp")
        value.add_(7)

    def reduce_replicas(source, destination, *, op):
        assert source is destination
        calls.append("replicas")
        destination.add_(11)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_dp)
    trainer = SimpleNamespace(
        config=config,
        parallel_dims=SimpleNamespace(
            dp_enabled=True,
            mesh={"dp": SimpleNamespace(get_group=lambda: group)},
        ),
    )
    masks = torch.tensor([[False, True, True], [False, True, False]])
    GRPOTrainer._prepare_pp_loss(
        trainer,
        masks,
        torch.tensor([True, False]),
        SimpleNamespace(world_size=lambda: 3, allreduce=reduce_replicas),
    )
    reduced = balanced and reduction == "token-mean"
    assert calls == (["dp", "replicas"] if reduced else [])
    assert trainer._pp_loss_normalization["tokens"] == (21 if reduced else 3)
    assert trainer._pp_loss_normalization["dp_workers"] == (6 if reduced else 1)
    assert trainer._pp_loss_normalization["sequences"] == 2
    assert trainer._pp_local_tokens == 3
    assert trainer._pp_positive_tokens == 2


@pytest.mark.parametrize(
    "batch, requested, expected", [(4, 2, 2), (7, 2, 1), (2, 2, 1), (8, 2, 2)]
)
def test_resize_rebuilds_only_changed_pipeline_shape(
    monkeypatch, batch, requested, expected
):
    from cosmos_rl.utils.pipelining import pipelining_utils as pp

    options = {"batch_size": 4, "microbatch_size": 2, "num_stages": 2}
    original = SimpleNamespace(_cosmos_build_kwargs=options)
    builds = []

    def build(**options):
        builds.append(options)
        return SimpleNamespace(_cosmos_build_kwargs=options)

    monkeypatch.setattr(pp, "build_pipeline_schedule", build)
    resized, size = pp.resize_pipeline_schedule(original, batch, requested)
    assert size == expected
    assert len(builds) == (0 if batch == 4 else 1)
    assert (resized is original) == (batch == 4)
    assert resized._cosmos_build_kwargs["batch_size"] == batch
    assert pp.resize_pipeline_schedule(resized, batch, requested)[0] is resized


@pytest.mark.parametrize("batch, microbatch", [(0, 1), (1, 0), (-1, 2)])
def test_resize_rejects_nonpositive_dimensions(batch, microbatch):
    from cosmos_rl.utils.pipelining.pipelining_utils import resize_pipeline_schedule

    with pytest.raises(ValueError, match="must be positive"):
        resize_pipeline_schedule(object(), batch, microbatch)


def test_reference_state_uses_materialized_parts_not_meta_root():
    trainer = SimpleNamespace(
        parallel_dims=SimpleNamespace(pp_enabled=True),
        model=torch.nn.Linear(2, 2, device="meta"),
        model_parts=[
            torch.nn.ModuleDict({"first": torch.nn.Linear(2, 2)}),
            torch.nn.ModuleDict({"last": torch.nn.Linear(2, 2)}),
        ],
        model_module_path=["", ""],
    )
    state = GRPOTrainer._local_policy_state_dict(trainer)
    assert set(state) == {"first.weight", "first.bias", "last.weight", "last.bias"}
    assert all(value.device.type == "cpu" for value in state.values())
    assert (
        state["first.weight"].data_ptr()
        == trainer.model_parts[0]["first"].weight.data_ptr()
    )
    trainer.parallel_dims.pp_enabled = False
    assert all(
        value.device.type == "meta"
        for value in GRPOTrainer._local_policy_state_dict(trainer).values()
    )


def test_pipeline_reference_keys_keep_part_prefixes_and_reject_duplicates():
    trainer = SimpleNamespace(
        parallel_dims=SimpleNamespace(pp_enabled=True),
        model_parts=[torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)],
        model_module_path=["first", "last"],
    )
    state = GRPOTrainer._local_policy_state_dict(trainer)
    assert set(state) == {"first.weight", "first.bias", "last.weight", "last.bias"}
    assert state["first.weight"].data_ptr() == trainer.model_parts[0].weight.data_ptr()
    assert state["last.weight"].data_ptr() == trainer.model_parts[1].weight.data_ptr()
    trainer.model_module_path = ["", ""]
    with pytest.raises(ValueError, match="Duplicate pipeline reference state"):
        GRPOTrainer._local_policy_state_dict(trainer)


def test_rollout_logprob_rows_preserve_ragged_sample_identity():
    from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import _pp_logprob_rows

    masks = torch.tensor(
        [[False, True, True], [False, True, False], [False, False, False]]
    )
    values = [[-1.0, -2.0], [-3.0], []]
    rows = _pp_logprob_rows(masks, values)
    assert rows.tolist() == [[0.0, -1.0, -2.0], [0.0, -3.0, 0.0], [0.0, 0.0, 0.0]]
    with pytest.raises(AssertionError, match="selected tokens"):
        _pp_logprob_rows(masks, [[-1.0], [-3.0], []])
