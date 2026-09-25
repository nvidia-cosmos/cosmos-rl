# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual trainer entrypoints agree with sample/next-token references."""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.policy.trainer.llm_trainer.dpo_trainer import DPOTrainer
from cosmos_rl.policy.trainer.llm_trainer import grpo_trainer as grpo


@pytest.fixture
def device():
    device = torch.device(os.environ.get("COSMOS_ALIGNMENT_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available(), "Required CUDA alignment gate cannot skip"
    return device


@pytest.mark.parametrize("source", ["mask", "labels"])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64])
def test_dpo_response_sums_and_gradients_follow_target_positions(
    device, source, mask_dtype
):
    # Different responses, padding, and an empty response in the same batch.
    ids = torch.tensor(
        [[1, 2, 3, 4, 0], [2, 4, 1, 0, 0], [3, 1, 2, 0, 0]], device=device
    )
    target_mask = torch.tensor(
        [[0, 0, 1, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
        dtype=torch.bool,
        device=device,
    )
    values = torch.arange(75, dtype=torch.float64, device=device).reshape(3, 5, 5)
    logits = (values.sin() * 2).requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    batch = {"input_ids": ids}
    if source == "mask":
        batch["logprob_masks"] = target_mask.to(mask_dtype)
    else:
        batch["label_ids"] = ids.masked_fill(~target_mask, -100)
    actual = DPOTrainer._compute_logprobs_and_sum(None, batch, logits)
    selected = (
        reference_logits[:, :-1]
        .log_softmax(-1)
        .gather(-1, ids[:, 1:, None])
        .squeeze(-1)
    )
    expected = (selected * target_mask[:, 1:]).sum(-1)
    torch.testing.assert_close(actual, expected)
    weights = torch.tensor([1.0, 3.0, 5.0], device=device)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert not torch.count_nonzero(logits.grad[:, -1])
    assert not torch.count_nonzero(logits.grad[2])
    assert torch.equal(batch["input_ids"], ids)


def test_dpo_missing_response_contract_is_rejected(device):
    with pytest.raises(ValueError, match="logprob_masks or label_ids"):
        DPOTrainer._compute_logprobs_and_sum(
            None,
            {"input_ids": torch.ones((1, 3), dtype=torch.long, device=device)},
            torch.zeros((1, 3, 5), device=device),
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("old_from_rollout", [False, True])
@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("positive_nll", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("packing", [False, True])
def test_grpo_chunks_keep_all_sample_fields_aligned(
    monkeypatch,
    device,
    dynamic,
    old_from_rollout,
    teacher,
    positive_nll,
    chunked,
    packing,
    cache_case="normal",
):
    if device.type == "cpu":
        monkeypatch.setattr(
            torch.cuda,
            "Event",
            lambda **kwargs: Mock(query=lambda: True, elapsed_time=lambda other: 1.0),
        )
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    # Exercise the real step_training loop, including reference/old-logprob
    # passes, arrangement reuse, two optimizer chunks and two mu iterations.
    weight = torch.nn.Parameter(torch.tensor(0.25, device=device))
    optimizer = torch.optim.SGD([weight], lr=0.01)
    seen, updates = [], []
    current_ids = []

    class Model:
        def get_position_ids(self, **batch):
            ids = batch["input_ids"]
            return torch.arange(3, device=device).expand_as(ids), ids, 1

        def __call__(self, **batch):
            shape = (
                (1, 3 * len(batch["input_ids"]), 2)
                if packing
                else (len(batch["input_ids"]), 3, 2)
            )
            return SimpleNamespace(logits=weight.expand(*shape))

    def collate(samples, **kwargs):
        current_ids[:] = [sample["id"] for sample in samples]
        return {
            "input_ids": torch.tensor(
                [[i + 1, i + 2, 0] for i in current_ids], device=device
            ),
            "logprob_masks": torch.tensor(
                [[True, True, False]] * len(samples), device=device
            ),
        }

    def logprobs(batch, logits, **kwargs):
        mask = batch["logprob_masks"]
        return (
            logits[..., 0][mask],
            batch["logprob_cu_seqlens"]
            if packing
            else torch.arange(0, 2 * len(mask) + 1, 2, device=device),
            {},
        )

    def loss_fn(current, old, ref, advantages, cu_seqlens, config, mask, **kwargs):
        expected = torch.tensor([i + 10.0 for i in current_ids], device=device)
        torch.testing.assert_close(advantages[mask], expected.repeat_interleave(2))
        expected_behavior = [[-(i + 1) / 10.0] * 2 for i in current_ids]
        assert kwargs["rollout_per_token_logps"] == expected_behavior
        if old_from_rollout:
            torch.testing.assert_close(
                old, torch.tensor(expected_behavior, device=device).flatten()
            )
        seen.append(tuple(current_ids))
        loss = -(current * expected.repeat_interleave(2)).mean()
        return loss, loss.detach(), loss.detach() * 0

    def reduce_states(comm):
        updates.append(float(weight.grad) if weight.grad is not None else 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return torch.tensor(0.0, device=device)

    def teacher_collate(rollouts, processed_samples, **kwargs):
        ids = [sample["id"] for sample in processed_samples]
        assert [rollout.prompt for rollout in rollouts] == ids
        return torch.tensor([[i + 100.0] * 3 for i in ids], device=device)

    def teacher_advantages(**kwargs):
        expected = torch.tensor([[i + 100.0] * 3 for i in current_ids], device=device)
        if packing:
            expected = expected.reshape(1, -1)
        torch.testing.assert_close(kwargs["teacher_logprobs"], expected)
        return kwargs["current_advantages"], {}

    if dynamic:
        # Deterministic non-contiguous chunk-local partitions expose offset and
        # double-offset bugs without changing the unrelated balancer policy.
        def arrange(batch, **kwargs):
            indices = (
                [[i] for i in range(len(batch))]
                if cache_case == "fragmented"
                else [[len(batch) - 1 - i, i] for i in range(len(batch) // 2)]
            )
            return [[batch[i] for i in group] for group in indices], indices

        monkeypatch.setattr(grpo, "rearrange_mini_batches", arrange)
    policy = SimpleNamespace(
        rollout_as_token_ids=False,
        positive_nll_coef=0.3 if positive_nll else 0.0,
        max_token_len_per_mini_batch=16 if dynamic else None,
        use_decoupled_loss=True,
        use_rollout_logprobs_for_loss=old_from_rollout,
        temperature=1.0,
        entropy_coeff=0.0,
        kl_beta=0.0,
    )
    trainer = SimpleNamespace(
        parallel_dims=SimpleNamespace(
            pp_coord=(0, 1),
            pp_enabled=False,
            cp_enabled=False,
            dp_enabled=False,
            dp_shard_coord=(0, 1),
            world_size=1,
            dp_replicate_enabled=False,
            dp_shard_enabled=False,
        ),
        config=SimpleNamespace(
            train=SimpleNamespace(train_policy=policy, sequence_packing=packing),
            rollout=SimpleNamespace(multi_turn_config=SimpleNamespace(enable=False)),
            distillation=SimpleNamespace(enable=teacher, top_k=0),
            logging=SimpleNamespace(
                logger=["test"] if cache_case == "skipped" else [], report_mfu=False
            ),
        ),
        device=device,
        tokenizer=SimpleNamespace(pad_token_id=0),
        train_stream=torch.cuda.current_stream() if device.type == "cuda" else None,
        data_packer=SimpleNamespace(
            get_policy_input=lambda sample, *args: {
                "id": sample,
                "logprob_masks": [True, True, False],
            },
            policy_compute_max_len=lambda samples: 3,
            policy_collate_fn=collate,
        ),
        batch_size_per_optimize=(3 if cache_case == "uneven" else 4)
        if chunked
        else None,
        mini_batch=2,
        mu_iterations=2,
        seq_len_multiple=1,
        mini_step=0,
        _swap_model_state_dict=lambda: (True, 0.0),
        set_model_eval=lambda: None,
        set_model_train=lambda: None,
        forward_model=Model(),
        act_offloading_ctx_manager=nullcontext(),
        compute_logprobs=logprobs,
        loss_fn=loss_fn,
        all_reduce_states=reduce_states,
        lr_schedulers=Mock(),
        reference_reset=lambda step: None,
        clear_teacher_result_cache=lambda: None,
        global_rank=0,
        fetch_teacher_logprobs=lambda **kwargs: None,
        collate_teacher_logprobs=teacher_collate,
        compute_teacher_kl_advantages=teacher_advantages,
    )
    rollouts = [
        SimpleNamespace(
            prompt=i,
            completion=[i + 1, i + 2],
            prompt_logprobs=[],
            completion_logprobs=[(-(i + 1) / 10.0,)] * 2,
            n_ignore_prefix_tokens=0,
            advantage=i + 10.0,
            teacher_logprobs=[i + 100.0],
            reward=1.0 if i % 3 == 1 else -1.0,
        )
        for i in range(8)
    ]
    if cache_case == "skipped":
        for rollout in rollouts:
            rollout.teacher_logprobs = None
        trainer.lr_schedulers.get_last_lr.return_value = []
        monkeypatch.setattr(grpo, "is_master_rank", lambda *args: True)
    report = grpo.GRPOTrainer.step_training(
        trainer,
        rollouts,
        current_step=1,
        total_steps=4,
        remain_samples_num=0,
        inter_policy_nccl=None,
        is_master_replica=False,
    )
    assert sorted(i for group in seen for i in group) == (
        [] if cache_case == "skipped" else sorted(list(range(8)) * 2)
    )
    expected_updates = []
    groups = iter(seen)
    optimize_size = trainer.batch_size_per_optimize or 8
    for _ in range(2):
        for start in range(0, 8, optimize_size):
            chunk_size = min(optimize_size, 8 - start)
            gradient = 0.0
            consumed = 0
            while consumed < chunk_size and cache_case != "skipped":
                group = next(groups)
                consumed += len(group)
                gradient -= sum(i + 10.0 for i in group) / chunk_size
                if positive_nll and any(i % 3 == 1 for i in group):
                    gradient -= 0.3 * len(group) / chunk_size
            expected_updates.append(gradient)
    assert updates == pytest.approx(expected_updates)
    assert weight.item() == pytest.approx(0.25 - 0.01 * sum(expected_updates))
    trainer.lr_schedulers.step.assert_called_once()
    if cache_case == "skipped":
        assert report["train/learning_rate"] == 0.0
        assert report["train/loss_avg"] == 0.0
        assert report["train/entropy"] == 0.0


@pytest.mark.parametrize("case", ["uneven", "fragmented", "skipped"])
def test_actual_minibatch_cache_and_skipped_chunk_boundaries(monkeypatch, device, case):
    test_grpo_chunks_keep_all_sample_fields_aligned(
        monkeypatch,
        device,
        dynamic=case == "fragmented",
        old_from_rollout=False,
        teacher=True,
        positive_nll=True,
        chunked=True,
        packing=False,
        cache_case=case,
    )
