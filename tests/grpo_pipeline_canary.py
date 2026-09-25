# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual GRPO trainer/PP updates versus non-PP optimizer continuations."""

import argparse
import copy
from contextlib import ExitStack, nullcontext
from datetime import timedelta
import os
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from cosmos_rl.policy.config import Config
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer, compute_loss
from cosmos_rl.utils.pipelining.pipelining_utils import build_pipeline_schedule


class Part(torch.nn.Module):
    def __init__(self, index, count):
        super().__init__()
        self.first, self.last = index == 0, index == count - 1
        # Distinct global names also exercise local reference-state assembly.
        self.key = f"layer_{index}"
        self.add_module(
            self.key,
            torch.nn.Embedding(7, 8)
            if self.first
            else torch.nn.Linear(8, 7 if self.last else 8),
        )

    def forward(self, value=None, *, input_ids=None, position_ids=None, **kwargs):
        value = input_ids if value is None else value
        result = getattr(self, self.key)(value)
        return result if self.last else result.tanh()


def run_case(
    device,
    schedule_name,
    reduction,
    variant,
    batch_size,
    optimize_size,
    rollout_old=False,
    decoupled=False,
):
    rank = dist.get_rank()
    count = 4 if schedule_name == "Interleaved1F1B" else 2
    torch.manual_seed(19)
    golden = torch.nn.Sequential(*[Part(i, count) for i in range(count)]).to(device)
    frozen = copy.deepcopy(golden)
    indices = list(range(rank, count, 2))
    parts = [copy.deepcopy(golden[i]) for i in indices]
    local_params = [p for part in parts for p in part.parameters()]
    golden_params = [p for i in indices for p in golden[i].parameters()]
    config = Config.from_dict(
        {
            "train": {
                "output_dir": "/tmp/grpo-pipeline-canary",
                "train_batch_per_replica": batch_size,
                "sequence_packing": False,
                "logprob_dtype": "float32",
                "train_policy": {
                    "mini_batch": 4,
                    "batch_size_per_optimize": optimize_size,
                    "mu_iterations": 2,
                    "loss_type": reduction,
                    "variant": variant,
                    "kl_beta": 0.05,
                    "entropy_coeff": 0.03,
                    "positive_nll_coef": 0.1,
                    "use_rollout_logprobs_for_loss": rollout_old,
                    "use_decoupled_loss": decoupled,
                },
            },
            "policy": {
                "model_max_length": 5,
                "parallelism": {"pp_micro_batch_size": 1 if count == 4 else 2},
            },
            "logging": {"logger": ["console"] if device.type == "cpu" else []},
        }
    )
    mesh = SimpleNamespace(
        get_local_rank=lambda: rank, size=lambda: 2, get_group=lambda: dist.group.WORLD
    )
    ids = torch.arange(batch_size * 5, device=device).reshape(batch_size, 5) % 7
    masks = torch.tensor([[False, True, True, True, False]] * batch_size, device=device)
    masks[1, 2:] = False
    masks[-1] = False
    advantages = torch.linspace(-1, 2, batch_size, device=device)
    positive = torch.arange(batch_size, device=device) % 2 == 1
    optimizer = torch.optim.SGD(local_params, lr=0.03, momentum=0.9)
    reference_optimizer = torch.optim.SGD(golden.parameters(), lr=0.03, momentum=0.9)

    def collate(samples, **kwargs):
        selected = [sample["index"] for sample in samples]
        return {"input_ids": ids[selected], "logprob_masks": masks[selected]}

    trainer = SimpleNamespace(
        config=config,
        device=device,
        parallel_dims=SimpleNamespace(
            pp_enabled=True,
            pp_coord=(rank, 2),
            pp=2,
            cp_enabled=False,
            dp_enabled=False,
            dp_shard_enabled=False,
            dp_replicate_enabled=False,
            dp_shard_coord=(0, 1),
            world_size=2,
            mesh={"pp": mesh},
        ),
        # Deliberately distinct from the scheduled parts: the real root can be meta.
        model=copy.deepcopy(golden),
        model_parts=parts,
        model_module_path=[f"stage_{index}" for index in indices],
        forward_model=SimpleNamespace(
            get_position_ids=lambda **batch: (
                torch.arange(5, device=device).expand_as(batch["input_ids"]),
                batch["input_ids"],
                1,
            )
        ),
        train_stream=torch.cuda.current_stream() if device.type == "cuda" else None,
        tokenizer=SimpleNamespace(pad_token_id=0),
        data_packer=SimpleNamespace(
            get_policy_input=lambda index, *a: {
                "index": index,
                "logprob_masks": masks[index].cpu().tolist(),
            },
            policy_collate_fn=collate,
            policy_compute_max_len=lambda samples: 5,
        ),
        mini_batch=4,
        batch_size_per_optimize=optimize_size,
        mu_iterations=2,
        seq_len_multiple=1,
        mini_step=0,
        global_rank=rank,
        optimizers=optimizer,
        set_model_eval=lambda: [p.eval() for p in parts],
        set_model_train=lambda: [p.train() for p in parts],
        lr_schedulers=SimpleNamespace(step=lambda: None, get_last_lr=lambda: [0.03]),
        reference_reset=lambda _: None,
        clear_teacher_result_cache=lambda: None,
        act_offloading_ctx_manager=nullcontext(),
    )
    for method in [
        "compute_logprobs",
        "_prepare_pp_loss",
        "_local_policy_state_dict",
        "_swap_model_state_dict",
    ]:
        setattr(trainer, method, MethodType(getattr(GRPOTrainer, method), trainer))
    trainer.reference_state_dict = {
        name: tensor.detach().clone()
        for name, tensor in trainer._local_policy_state_dict().items()
    }
    options = dict(
        pp_mesh=mesh,
        batch_size=4,
        num_stages=count,
        schedule_str=schedule_name,
        microbatch_size=config.policy.parallelism.pp_micro_batch_size,
        model_parts=parts,
        device=device,
    )
    trainer.pp_scheduler = build_pipeline_schedule(
        **options, loss_fn=GRPOTrainer.pp_loss_fn.fget(trainer), scale_grads=False
    )
    trainer.pp_scheduler_val = build_pipeline_schedule(**options, loss_fn=None)
    rollouts = [
        SimpleNamespace(
            prompt=i,
            completion=[1] * int(masks[i].sum()),
            completion_token_ids=[[1]] * int(masks[i].sum()),
            prompt_logprobs=[[-2.0 - i * 0.05]] * (4 - int(masks[i].sum())),
            completion_logprobs=[[-2.0 - i * 0.05]] * int(masks[i].sum()),
            n_ignore_prefix_tokens=0,
            advantage=advantages[i].item(),
            reward=1.0 if positive[i] else -1.0,
        )
        for i in range(batch_size)
    ]

    def logps(model, selected):
        batch = {"input_ids": ids[selected], "logprob_masks": masks[selected]}
        return trainer.compute_logprobs(
            batch, model(ids[selected]), is_full_logits=True
        )

    for update in range(2):
        expected_updates = []
        expected_metrics = {
            "loss": [],
            "kl_loss": [],
            "entropy": [],
            "effective_entropy": [],
        }
        chunks = [
            list(range(start, min(start + optimize_size, batch_size)))
            for start in range(0, batch_size, optimize_size)
        ]
        batches = [
            [chunk[start : start + 4] for start in range(0, len(chunk), 4)]
            for chunk in chunks
        ]
        with torch.no_grad():
            old = {
                tuple(selected): logps(golden, selected)[0].detach().clone()
                for group in batches
                for selected in group
            }
            ref = {
                tuple(selected): logps(frozen, selected)[0].detach().clone()
                for group in batches
                for selected in group
            }
            behavior = {
                tuple(selected): torch.cat(
                    [
                        torch.full(
                            (int(masks[i].sum()),), -2.0 - i * 0.05, device=device
                        )
                        for i in selected
                    ]
                )
                for group in batches
                for selected in group
            }
        for _ in range(2):
            for chunk, group in zip(chunks, batches):
                reference_optimizer.zero_grad()
                for selected in group:
                    current, cu, metrics = logps(golden, selected)
                    mask = masks[selected]
                    loss, policy_loss, kl_loss = compute_loss(
                        current,
                        behavior[tuple(selected)]
                        if rollout_old
                        else old[tuple(selected)],
                        ref[tuple(selected)],
                        advantages[selected, None].expand_as(mask),
                        cu,
                        config,
                        mask,
                        rollout_per_token_logps=behavior[tuple(selected)]
                        if decoupled
                        else None,
                    )
                    expected_metrics["loss"].append(policy_loss.detach())
                    expected_metrics["kl_loss"].append(kl_loss.detach())
                    for key in ("entropy", "effective_entropy"):
                        expected_metrics[key].append(metrics[key].detach())
                    loss = loss - 0.03 * metrics["effective_entropy"]
                    positive_mask = positive[selected].repeat_interleave(
                        cu[1:] - cu[:-1]
                    )
                    if positive_mask.any():
                        loss = loss - 0.1 * current[positive_mask].mean()
                    (loss * (len(selected) / len(chunk))).backward()
                gradients = [p.grad.detach().clone() for p in golden_params]
                reference_optimizer.step()
                expected_updates.append(
                    (
                        gradients,
                        [p.detach().clone() for p in golden_params],
                        [
                            reference_optimizer.state[p]["momentum_buffer"]
                            .detach()
                            .clone()
                            for p in golden_params
                        ],
                    )
                )
        expected = iter(expected_updates)

        def apply_update(_):
            gradients, parameters, momentum = next(expected)
            for actual, wanted in zip(local_params, gradients):
                torch.testing.assert_close(actual.grad, wanted, rtol=3e-5, atol=3e-6)
            optimizer.step()
            for actual, wanted, velocity in zip(local_params, parameters, momentum):
                torch.testing.assert_close(actual, wanted, rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(
                    optimizer.state[actual]["momentum_buffer"],
                    velocity,
                    rtol=3e-5,
                    atol=3e-6,
                )
            optimizer.zero_grad()
            return torch.zeros((), device=device)

        trainer.all_reduce_states = apply_update
        with ExitStack() as stack:
            if device.type == "cpu":
                stack.enter_context(
                    patch.object(torch.cuda, "stream", lambda _: nullcontext())
                )
                stack.enter_context(
                    patch.object(
                        torch.cuda,
                        "Event",
                        lambda **kw: Mock(
                            record=lambda: None,
                            query=lambda: True,
                            elapsed_time=lambda _: 1.0,
                        ),
                    )
                )
            report = GRPOTrainer.step_training(
                trainer,
                rollouts,
                current_step=update + 1,
                total_steps=4,
                remain_samples_num=0,
                inter_policy_nccl=None,
                is_master_replica=False,
            )
        if rank == 1 and device.type == "cpu":
            for key, values in expected_metrics.items():
                report_key = key + "_avg" if key in ("loss", "kl_loss") else key
                torch.testing.assert_close(
                    torch.tensor(report[f"train/{report_key}"]),
                    torch.stack(values).mean(),
                    rtol=3e-5,
                    atol=3e-6,
                )
        assert next(expected, None) is None
        print(
            f"GRPO_PIPELINE_PASS rank={rank} schedule={schedule_name} reduction={reduction} variant={variant} batch={batch_size} optimize={optimize_size} update={update} rollout_old={rollout_old} decoupled={decoupled}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    device = torch.device(
        "cpu" if args.cpu else f"cuda:{int(os.environ['LOCAL_RANK'])}"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo" if args.cpu else "nccl", timeout=timedelta(seconds=120)
    )
    assert dist.get_world_size() == 2
    try:
        if args.quick:
            run_case(device, "GPipe", "seq-mean-token-mean", "grpo", 8, 6)
        else:
            for schedule in ["GPipe", "1F1B", "Interleaved1F1B"]:
                for reduction in [
                    "seq-mean-token-mean",
                    "seq-mean-token-sum",
                    "token-mean",
                    "token-sum",
                ]:
                    for variant in ["grpo", "gspo"]:
                        for batch, optimize in (
                            [(8, 8)]
                            if schedule == "Interleaved1F1B"
                            else [(8, 8), (8, 6), (7, 7)]
                        ):
                            run_case(
                                device, schedule, reduction, variant, batch, optimize
                            )
                for rollout_old, decoupled in [
                    (True, False),
                    (False, True),
                ]:
                    run_case(
                        device,
                        schedule,
                        "seq-mean-token-mean",
                        "grpo",
                        8,
                        8,
                        rollout_old=rollout_old,
                        decoupled=decoupled,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
