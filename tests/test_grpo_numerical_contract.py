# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical controls for configured GRPO behavior, not new objectives."""

import os
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer, compute_loss
from cosmos_rl.utils.sequence_packing import (
    generate_mask,
    pack_sequences_for_inputs,
    pack_sequences_for_logprobs,
    pack_sequences_for_masks,
    pack_sequences_info_collect,
)
from cosmos_rl.utils.util import entropy_from_logits_with_chunking


@pytest.mark.parametrize("mismatched_length", [False, True])
def test_gspo_integration_hook_checks_cached_tensor_values(mismatched_length):
    # Execute the actual integration-test hook without loading its 3B model.
    # This keeps a changed cache representation from breaking only in full CI.
    path = Path(__file__).with_name("launch_test_worker.py")
    tree = ast.parse(path.read_text())
    run_gspo = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_gspo_test"
    )
    hook = next(
        node
        for node in run_gspo.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "hooked_execute_all_reduce"
    )
    reductions = []

    def reduce(trainer, comm):
        reductions.append(comm)
        return 2.5

    namespace = {
        "GRPOTrainer": SimpleNamespace(all_reduce_states=reduce),
        "length": [2, 3, 4, 5],
    }
    exec(
        compile(ast.Module(body=[hook], type_ignores=[]), str(path), "exec"), namespace
    )
    trainer = SimpleNamespace(old_per_token_logps=defaultdict(lambda: None))
    trainer.old_per_token_logps[0] = torch.zeros(6 if mismatched_length else 5)
    trainer.old_per_token_logps[1] = torch.zeros(9)
    comm = object()
    if mismatched_length:
        with pytest.raises(AssertionError):
            namespace[hook.name](trainer, comm)
    else:
        assert namespace[hook.name](trainer, comm) == 2.5
        assert trainer.test_hooked_cnt == 4
    assert reductions == [comm]


@pytest.mark.parametrize("full_logits", [False, True])
@pytest.mark.parametrize("coefficient", [0.0, 0.2])
def test_configured_entropy_has_expected_gradient(full_logits, coefficient):
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available()
    logits = (
        torch.linspace(-3, 2, 48, device=device).sin().reshape(2, 4, 6).requires_grad_()
    )
    expected_logits = logits.detach().clone().requires_grad_()
    mask = torch.tensor(
        [[False, True, True, False], [True, False, True, False]], device=device
    )
    ids = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]], device=device)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                logprob_dtype="float32",
                train_policy=SimpleNamespace(entropy_coeff=coefficient),
            )
        )
    )
    logps, _, metrics = GRPOTrainer.compute_logprobs(
        trainer,
        {"input_ids": ids, "logprob_masks": mask},
        logits if full_logits else logits[mask],
        is_full_logits=full_logits,
    )
    loss = logps.sum() * 0
    if coefficient:
        loss = loss - coefficient * metrics["effective_entropy"]
    expected_logps = expected_logits[mask].log_softmax(-1)
    expected_entropy = -(expected_logps.exp() * expected_logps).sum(-1).mean()
    expected_loss = -coefficient * expected_entropy
    torch.testing.assert_close(metrics["effective_entropy"], expected_entropy)
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, expected_logits.grad)
    assert not metrics["entropy"].requires_grad
    assert not torch.count_nonzero(logits.grad[~mask])
    if coefficient:
        assert torch.count_nonzero(logits.grad[mask])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_entropy_backward_recomputes_fp32_chunks_without_saved_probability_matrix(
    dtype,
):
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available()
    logits = (
        torch.linspace(-3, 4, 99, dtype=dtype, device=device)
        .reshape(11, 9)
        .requires_grad_()
    )
    reference = logits.detach().clone().requires_grad_()
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        entropy = entropy_from_logits_with_chunking(
            logits, chunk_size=3, checkpoint_chunks=True
        )
    assert saved
    assert all(
        not t.numel()
        or t.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()
        for t in saved
    )
    logps = reference.float().log_softmax(-1)
    expected = -(logps.exp() * logps).sum(-1)
    torch.testing.assert_close(entropy, expected)
    entropy.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize(
    "loss_type",
    ["token-mean", "token-sum", "seq-mean-token-mean", "seq-mean-token-sum"],
)
@pytest.mark.parametrize("variant", ["grpo", "gspo"])
@pytest.mark.parametrize("off_policy", [None, 0.15])
@pytest.mark.parametrize("full_logits", [False, True])
@pytest.mark.parametrize("max_tokens", [None, 8])
def test_packing_preserves_targets_sequence_objectives_and_gradients(
    loss_type,
    variant,
    off_policy,
    full_logits,
    max_tokens,
):
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    # EOS == pad, unequal response lengths, repeated EOS, and an empty response.
    ids = torch.tensor(
        [[1, 2, 5, 5, 5, 5], [1, 2, 3, 5, 5, 5], [2, 1, 5, 5, 5, 5]], device=device
    )
    mask = torch.tensor(
        [[0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype=torch.bool,
        device=device,
    )
    advantages = torch.tensor([1.0, -2.0, 0.0], device=device)
    policy = SimpleNamespace(
        entropy_coeff=0.0,
        variant=variant,
        aipo_rho=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        lower_bound_ratio=3.0,
        off_policy_masking_delta=off_policy,
        kl_beta=0.1,
        unbiased_kl_estimate=False,
        loss_type=loss_type,
        unbiased_loss_max_tokens=max_tokens,
        balance_dp_token=False,
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(logprob_dtype="float32", train_policy=policy)
        )
    )
    lengths = pack_sequences_info_collect(ids, 5, logprob_masks=mask)["valid_input_len"]
    assert lengths.tolist() == [3, 5, 2]
    valid = generate_mask(lengths, ids, 1, 0)
    batch = {
        "input_ids": pack_sequences_for_inputs(ids, lengths)["inputs"],
        **pack_sequences_for_masks(lengths, lengths),
        **pack_sequences_for_logprobs(mask, lengths, advantages),
    }
    logits = torch.arange(108, device=device).sin().reshape(3, 6, 6).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    packed_logits = logits[valid].unsqueeze(0)
    actual, boundaries, _ = GRPOTrainer.compute_logprobs(
        trainer,
        batch,
        packed_logits if full_logits else packed_logits[batch["logprob_masks"]],
        is_full_logits=full_logits,
    )
    expected, expected_boundaries, _ = GRPOTrainer.compute_logprobs(
        trainer,
        {"input_ids": ids, "logprob_masks": mask},
        reference,
        is_full_logits=True,
    )
    # Independent gather checks next-token targets, not only shared utility parity.
    expected_targets = torch.tensor([5, 3, 5, 5], device=device)
    naive = (
        reference[mask]
        .log_softmax(-1)
        .gather(-1, expected_targets[:, None])
        .squeeze(-1)
    )
    torch.testing.assert_close(actual, naive)
    assert boundaries.tolist() == [0, 1, 4, 4]
    torch.testing.assert_close(boundaries, expected_boundaries)
    old = expected.detach() + torch.tensor([0.1, 0.4, 0.1, 0.3], device=device)
    ref = expected.detach() - 0.1
    actual_loss = compute_loss(
        actual,
        old,
        ref,
        batch["advantages"],
        boundaries,
        trainer.config,
        batch["logprob_masks"],
    )[0]
    expected_loss = compute_loss(
        expected,
        old,
        ref,
        advantages[:, None].expand_as(mask),
        expected_boundaries,
        trainer.config,
        mask,
    )[0]
    assert torch.isfinite(actual_loss)
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("multiple", [1, 4, 8])
def test_packing_target_lengths_survive_alignment_padding(multiple):
    ids = torch.tensor([[1, 2, 5, 5, 5, 5, 5, 5], [2, 5, 5, 5, 5, 5, 5, 5]])
    mask = torch.tensor(
        [[0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    lengths = pack_sequences_info_collect(
        ids, 5, seq_len_multiple=multiple, logprob_masks=mask
    )["valid_input_len"]
    assert (lengths >= torch.tensor([3, 2])).all()
    assert lengths.sum() % multiple == 0


def test_packing_rejects_mask_without_next_token():
    with pytest.raises(ValueError, match="no next-token target"):
        pack_sequences_info_collect(
            torch.tensor([[1, 2]]), 0, logprob_masks=torch.tensor([[False, True]])
        )


@pytest.mark.parametrize("coefficient", [0.0, 0.2])
def test_empty_effective_entropy_is_finite_and_has_zero_gradient(coefficient):
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                logprob_dtype="float32",
                train_policy=SimpleNamespace(entropy_coeff=coefficient),
            )
        )
    )
    logits = torch.zeros(1, 3, 4, device=device, requires_grad=True)
    _, boundaries, metrics = GRPOTrainer.compute_logprobs(
        trainer,
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long, device=device),
            "logprob_masks": torch.zeros(1, 3, dtype=torch.bool, device=device),
        },
        logits,
        is_full_logits=True,
    )
    assert boundaries.tolist() == [0, 0]
    assert metrics["effective_entropy"].item() == 0.0
    (metrics["effective_entropy"] + logits.sum() * 0).backward()
    assert not torch.count_nonzero(logits.grad)
