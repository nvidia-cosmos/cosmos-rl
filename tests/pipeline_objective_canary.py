# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual two-rank PP SFT gradients/updates versus the unchanged non-PP loss."""

import argparse
import copy
from contextlib import ExitStack
from datetime import timedelta
import os
from types import SimpleNamespace
from unittest.mock import patch
from functools import partial

import torch
import torch.distributed as dist
from cosmos_rl.patch import PipelineStage, Schedule1F1B, ScheduleGPipe
from cosmos_rl.policy.kernel.loss import CrossEntropyLoss
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer, async_safe_ce
from cosmos_rl.utils.pipelining.pipelining_utils import build_pipeline_schedule


class Part(torch.nn.Module):
    def __init__(self, last=False):
        super().__init__()
        self.linear = torch.nn.Linear(4, 7 if last else 4)
        self.last = last

    def forward(self, value, position_ids=None, **kwargs):
        result = self.linear(value)
        return result if self.last else result.tanh()

    def step_hook(self, step):
        return None


def run_case(device, schedule_name, mini_batch, micro_batch, unequal):
    rank = dist.get_rank()
    torch.manual_seed(51)
    num_stages = 4 if schedule_name == "builder-interleaved" else 2
    reference = torch.nn.Sequential(
        *[Part(last=i == num_stages - 1) for i in range(num_stages)]
    ).to(device)
    local_indices = list(range(rank, num_stages, 2))
    parts = [copy.deepcopy(reference[i]) for i in local_indices]
    part = parts[0]
    local_parameters = [p for model in parts for p in model.parameters()]
    reference_parameters = [
        p for index in local_indices for p in reference[index].parameters()
    ]
    schedule_type = {
        "builder-1f1b": "1F1B",
        "builder-gpipe": "GPipe",
        "builder-interleaved": "Interleaved1F1B",
    }.get(schedule_name)
    stage = PipelineStage(
        part,
        rank,
        2,
        device,
        group=dist.group.WORLD,
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                train_batch_per_replica=8,
                sequence_packing=False,
                optm_grad_norm_clip=0.0,
                train_policy=SimpleNamespace(
                    mini_batch=mini_batch,
                    enable_dp_load_balancing=False,
                    balance_dp_token=False,
                ),
            ),
            policy=SimpleNamespace(
                model_max_length=5,
                parallelism=SimpleNamespace(pp_micro_batch_size=micro_batch),
            ),
            logging=SimpleNamespace(logger=[]),
        ),
        parallel_dims=SimpleNamespace(
            dp_shard_enabled=False,
            cp_enabled=False,
            dp_replicate_enabled=False,
            pp_enabled=True,
            pp_coord=(rank, 2),
            pp_dynamic_shape=False,
            pp_dynamic_shape_enabled=False,
            mesh={
                name: SimpleNamespace(get_group=lambda: dist.group.WORLD)
                for name in ("pp", "loss_parallel")
            },
        ),
        device=device,
        enable_dp_load_balancing=False,
        seq_len_multiple=1,
        model_parts=parts,
        set_model_train=lambda: [model.train() for model in parts],
        forward_model=SimpleNamespace(
            get_position_ids=lambda **batch: (
                torch.arange(5, device=device).expand(batch["input_ids"].shape[0], 5),
                batch["input_ids"],
                1,
            )
        ),
        data_packer=SimpleNamespace(
            batch_size=len,
            slice_batch=lambda batch, start, end: batch[start:end],
            sft_collate_fn=lambda batch, **kw: {
                "input_ids": torch.stack([sample[0] for sample in batch]),
                "label_ids": torch.stack([sample[1] for sample in batch]),
            },
        ),
    )
    trainer._prepare_pp_loss = partial(SFTTrainer._prepare_pp_loss, trainer)
    # Compilation is orthogonal to schedule scaling/denominators; use the same
    # actual callable without compilation startup in this small reference probe.
    with patch.object(torch, "compile", lambda function: function):
        loss_fn = SFTTrainer.pp_loss_fn.fget(trainer)
    if schedule_name.startswith("builder-"):
        schedule = build_pipeline_schedule(
            scale_grads=False,
            pp_mesh=SimpleNamespace(
                get_local_rank=lambda: rank,
                size=lambda: 2,
                get_group=lambda: dist.group.WORLD,
            ),
            batch_size=mini_batch,
            num_stages=num_stages,
            schedule_str=schedule_type,
            microbatch_size=micro_batch,
            model_parts=parts,
            device=device,
            loss_fn=loss_fn,
        )
    elif schedule_name == "1f1b":
        schedule = Schedule1F1B(stage, mini_batch // micro_batch, loss_fn=loss_fn)
    else:
        schedule = ScheduleGPipe(
            stage, mini_batch // micro_batch, loss_fn=loss_fn, scale_grads=False
        )
    optimizer = torch.optim.SGD(local_parameters, lr=0.05, momentum=0.9)
    golden_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9)
    trainer.optimizers = optimizer
    trainer.lr_schedulers = SimpleNamespace(step=lambda: None)
    captured_losses = []

    def step(*args, **kwargs):
        result = schedule.step(*args, **kwargs)
        if rank == 1:
            captured_losses.extend(loss.detach() for loss in kwargs["losses"])
        return result

    trainer.pp_scheduler = SimpleNamespace(step=step)
    data = torch.linspace(-2, 3, 160, device=device).reshape(8, 5, 4)
    labels = torch.arange(40, device=device).reshape(8, 5) % 7
    if unequal:
        labels[1, 2:] = -100
        labels[2, 3:] = -100
        labels[3, 1:] = -100
        labels[5, 3:] = -100
        labels[7, 2:] = -100
    for update in range(2):
        golden_optimizer.zero_grad(set_to_none=True)
        golden_loss = torch.zeros((), device=device)
        for start in range(0, 8, mini_batch):
            batch, targets = (
                data[start : start + mini_batch],
                labels[start : start + mini_batch],
            )
            scale = mini_batch / 8
            golden = async_safe_ce(
                reference(batch), targets, CrossEntropyLoss(), loss_scaling_factor=scale
            )
            golden.backward()
            golden_loss += golden.detach()
        captured_losses.clear()
        with ExitStack() as stack:
            if device.type == "cpu":
                # CPU compatibility for timing-only CUDA events and logging's
                # AVG reduction (Gloo has SUM but not AVG). The actual PP
                # forward/backward, norm collective and optimizer are unchanged.
                stack.enter_context(
                    patch.object(
                        torch.cuda,
                        "Event",
                        lambda **kw: SimpleNamespace(record=lambda: None),
                    )
                )
                original_reduce = dist.all_reduce

                def reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kw):
                    if op == dist.ReduceOp.AVG:
                        original_reduce(tensor, op=dist.ReduceOp.SUM, group=group, **kw)
                        tensor.div_(dist.get_world_size(group))
                    else:
                        original_reduce(tensor, op=op, group=group, **kw)

                stack.enter_context(patch.object(dist, "all_reduce", reduce))
            SFTTrainer.step_training(
                trainer,
                list(zip(data, labels, strict=True)),
                total_steps=2,
                train_step=update,
                save_freq=0,
            )
        for actual, expected in zip(
            local_parameters, reference_parameters, strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
        if rank == 1:
            torch.testing.assert_close(
                torch.stack(captured_losses).sum(), golden_loss, atol=2e-6, rtol=2e-5
            )
        golden_optimizer.step()
        for actual, expected in zip(
            local_parameters, reference_parameters, strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(
                optimizer.state[actual]["momentum_buffer"],
                golden_optimizer.state[expected]["momentum_buffer"],
                atol=2e-6,
                rtol=2e-5,
            )
        # Exercise the production train -> forward-only validation -> train
        # transition, including different validation/training microbatch counts.
        val_size = 2 * micro_batch
        if schedule_name.startswith("builder-"):
            validation_schedule = build_pipeline_schedule(
                pp_mesh=SimpleNamespace(
                    get_local_rank=lambda: rank,
                    size=lambda: 2,
                    get_group=lambda: dist.group.WORLD,
                ),
                batch_size=val_size,
                num_stages=num_stages,
                schedule_str=schedule_type,
                microbatch_size=micro_batch,
                model_parts=parts,
                device=device,
                loss_fn=None,
            )
        else:
            validation_schedule = ScheduleGPipe(schedule._stage, 2, loss_fn=None)
        grads_before = [p.grad.clone() for p in local_parameters]
        with torch.no_grad():
            output = validation_schedule.step(
                *([data[:val_size]] if rank == 0 else []),
                position_ids=torch.arange(5, device=device).expand(val_size, 5),
            )
            if rank == 1:
                torch.testing.assert_close(
                    output, reference(data[:val_size]), atol=2e-6, rtol=2e-5
                )
        for parameter, before in zip(local_parameters, grads_before, strict=True):
            torch.testing.assert_close(parameter.grad, before)
        dist.barrier()
        print(
            f"PP_OBJECTIVE_PASS rank={rank} schedule={schedule_name} mini={mini_batch} micro={micro_batch} unequal={unequal} update={update} device={device}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if device.type == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo" if args.cpu else "nccl", timeout=timedelta(seconds=90)
    )
    assert dist.get_world_size() == 2
    try:
        for schedule in (
            "1f1b",
            "gpipe",
            "builder-1f1b",
            "builder-gpipe",
            "builder-interleaved",
        ):
            for mini_batch, micro_batch in ((4, 1), (4, 2), (8, 2)):
                for unequal in (False, True):
                    run_case(device, schedule, mini_batch, micro_batch, unequal)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
