# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Settle distillation readiness before any model/optimizer work."""

from functools import wraps
from inspect import signature

import torch
import torch.distributed as dist

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils import constant
from cosmos_rl.utils.teacher_channel import deadline_after


def teacher_wait_budget(comm):
    # Healthy ranks enter the vote immediately. A missing rank must not wait
    # longer than their collectives can tolerate (the old teacher default is
    # 1800s, but both process-group and HA defaults are 600s).
    # Validate on every rank, not only the leader that waits for teacher bytes.
    # Invalid configuration must fail before peers enter the target broadcast.
    deadline_after(constant.COSMOS_TEACHER_RESULT_GET_TIMEOUT)
    budget = min(
        constant.COSMOS_TEACHER_RESULT_GET_TIMEOUT,
        constant.COSMOS_GLOO_TIMEOUT / 2,
        getattr(comm, "default_timeout_ms", constant.COSMOS_GLOO_TIMEOUT * 1000) / 2000,
    )
    deadline_after(budget)
    return budget


def agree_teacher_update(trainer, rollouts, comm):
    timeout = teacher_wait_budget(comm)
    missing = False
    try:
        trainer.fetch_teacher_logprobs(
            rollouts, list(range(len(rollouts))), timeout=timeout
        )
        missing = any(rollout.teacher_logprobs is None for rollout in rollouts)
    except (ValueError, TypeError, KeyError, IndexError, AssertionError) as error:
        # Bad teacher data, like absent data, cannot change local participation.
        # Transport/collective failures are not data errors and remain fatal.
        logger.warning("[Policy] Invalid teacher targets: %s", error)
        missing = True
    vote = torch.tensor([int(missing)], device=trainer.device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(vote, op=dist.ReduceOp.MAX)
    comm.allreduce(vote, vote, op=dist.ReduceOp.MAX)
    if dist.is_initialized():
        dist.all_reduce(vote, op=dist.ReduceOp.MAX)
    return not bool(vote.item())


def teacher_update_boundary(step):
    parameters = signature(step)

    @wraps(step)
    def wrapped(trainer, *args, **kwargs):
        if not trainer.config.distillation.enable:
            return step(trainer, *args, **kwargs)
        arguments = parameters.bind(trainer, *args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        comm = values["inter_policy_nccl"]
        # Membership is fixed before the vote and remains fixed through all
        # model/optimizer work. No additional distributed barrier is needed.
        with comm.operation_scope():
            try:
                if agree_teacher_update(trainer, values["rollouts"], comm):
                    return step(trainer, *args, **kwargs)
                logger.warning(
                    "[Policy] Skipping entire distillation update %s: teacher targets unavailable",
                    values["current_step"],
                )
                if values["is_master_replica"] and values["do_save_checkpoint"]:
                    trainer.save_checkpoint(
                        current_step=values["current_step"],
                        total_steps=values["total_steps"],
                        remain_samples_num=values["remain_samples_num"],
                        is_final=values["current_step"] == values["total_steps"],
                    )
                return {
                    "train_step": values["current_step"],
                    "train/teacher_update_skipped": 1,
                }
            finally:
                trainer.clear_teacher_result_cache()
                trainer.teacher_results.retire(
                    [rollout.teacher_result_uuid for rollout in values["rollouts"]]
                )

    return wrapped
