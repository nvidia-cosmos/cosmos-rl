# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import torch
import types
from functools import partial
import inspect
import numpy as np
import enum
from functools import cached_property
from typing import Optional, Dict, Any, Callable, List, Tuple, Set

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.trainer.llm_trainer.llm_trainer import LLMTrainer
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.optm import (
    build_lr_schedulers as common_build_lr_schedulers,
)
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import (
    compute_mfu,
    setup_tokenizer,
)
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.utils.balance_seqlen import rearrange_mini_batches
from cosmos_rl.utils.sequence_packing import (
    pack_sequences_for_inputs,
    pack_sequences_for_logprobs,
    pack_sequences_info_collect,
    pack_sequences_for_masks,
    pack_sequences_for_teacher_logprobs,
)
from cosmos_rl.utils.ulysses import (
    slice_inputs_for_ulysses,
)
from cosmos_rl.utils.util import is_master_rank, str2torch_dtype
from cosmos_rl.utils.util import compute_logprobs as logprobs_computing
import cosmos_rl.utils.distributed as dist_util
import torch.distributed as dist


class TrainerPhase(enum.Enum):
    REF_COMPUTE = "ref_compute"
    OLD_LOGP_COMPUTE = "old_logp_compute"
    TRAIN = "train"


def _apply_off_policy_mask(
    per_token_loss: torch.Tensor,
    rollout_per_token_logps: Optional[torch.Tensor],
    old_per_token_logps: torch.Tensor,
    current_token_logps: torch.Tensor,
    current_advantages: torch.Tensor,
    cu_seqlens: torch.Tensor,
    shifted_length: torch.Tensor,
    off_policy_masking_delta: float,
) -> torch.Tensor:
    """
    Off-Policy Sequence Masking.
    Reference:
    - DeepSeek-V3.2 Sec.3.1 Off-Policy Sequence Masking
      https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
    """
    masking_source_logps = (
        rollout_per_token_logps
        if rollout_per_token_logps is not None
        else old_per_token_logps
    )

    # Compute per-sequence mean of log π_old − log π_θ
    logprob_diff = masking_source_logps - current_token_logps
    prefix_sum_for_logprob_diff = torch.cat(
        [logprob_diff.new_zeros(1), logprob_diff]
    ).cumsum(dim=0)
    sum_diff = (
        prefix_sum_for_logprob_diff[cu_seqlens[1:]]
        - prefix_sum_for_logprob_diff[cu_seqlens[:-1]]
    )
    denom = shifted_length.to(prefix_sum_for_logprob_diff.dtype).clamp_min(1)
    seq_mean_logprob_diff = sum_diff / denom

    # Sequence-level advantage
    advantage_by_sequence = current_advantages[cu_seqlens[:-1]]

    # Apply the mask
    seq_mask = (
        (advantage_by_sequence >= 0)
        | (seq_mean_logprob_diff <= off_policy_masking_delta)
    ).to(per_token_loss.dtype)
    per_token_loss = per_token_loss * seq_mask.repeat_interleave(shifted_length)
    return per_token_loss


def compute_loss(
    current_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
    old_per_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
    ref_per_token_logps: Optional[
        torch.Tensor
    ],  # per-token logprobs of shape `[n_tokens_of_logprobs]`
    current_advantages: torch.Tensor,  # of shape `[batch_size, max_len]`
    cu_seqlens: torch.Tensor,  # of shape `[batch_size + 1]`
    config: CosmosConfig,
    logprob_masks: torch.Tensor,  # of shape `[batch_size, max_len]`
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
    ddp_comm: HighAvailabilitylNccl = None,
    rollout_per_token_logps: Optional[
        List[List[float]]
    ] = None,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Turn current_advantages from [batch_size, max_len] to [n_logprob_tokens]
    if current_advantages.shape == logprob_masks.shape:
        current_advantages = torch.masked_select(current_advantages, logprob_masks)
    else:
        assert (
            current_advantages.shape == current_token_logps.shape
        ), f"current_advantages.shape: {current_advantages.shape} != current_token_logps.shape: {current_token_logps.shape}"

    assert (
        current_token_logps.shape == current_advantages.shape
    ), "current_token_logps and current_advantages should have the same shape"
    assert (
        old_per_token_logps.shape == current_token_logps.shape
    ), "old_per_token_logps and current_token_logps should have the same shape"
    if ref_per_token_logps is not None:
        assert (
            ref_per_token_logps.shape == current_token_logps.shape
        ), "ref_per_token_logps and current_token_logps should have the same shape, but got {} and {}".format(
            ref_per_token_logps.shape, current_token_logps.shape
        )
    if rollout_per_token_logps is not None:
        rollout_per_token_logps = torch.tensor(
            np.concatenate(rollout_per_token_logps, axis=0),
            device=current_token_logps.device,
            dtype=current_token_logps.dtype,
        ).detach()
        assert (
            rollout_per_token_logps.shape == current_token_logps.shape
        ), "rollout_per_token_logps and current_token_logps should have the same shape, but got {} and {}".format(
            rollout_per_token_logps.shape, current_token_logps.shape
        )

    shifted_length = cu_seqlens[1:] - cu_seqlens[:-1]
    bsz = shifted_length.shape[0]
    negative_approx_kl = current_token_logps - old_per_token_logps

    # Compute importance ratio
    if config.train.train_policy.variant == "gspo":
        # For GSPO, we compute sequence-level importance ratios
        # but we need to maintain gradient flow through the token-level logprobs
        negative_approx_kl_seq = torch.zeros(
            bsz, device=negative_approx_kl.device, dtype=negative_approx_kl.dtype
        )

        # Compute sequence-level average KL divergence
        for i in range(bsz):
            seq_tokens = negative_approx_kl[cu_seqlens[i] : cu_seqlens[i + 1]]
            seq_length = shifted_length[i]
            assert (
                len(seq_tokens) == shifted_length[i]
            ), f"seq_length: {seq_length} != shifted_length: {shifted_length[i]}"
            if seq_length > 0:
                negative_approx_kl_seq[i] = seq_tokens.sum() / seq_length

        # Clamp for numerical stability
        negative_approx_kl_seq = torch.clamp(negative_approx_kl_seq, max=10.0)

        importance_ratio_per_token = torch.zeros_like(current_token_logps)
        for i in range(bsz):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_length = end_idx - start_idx
            if seq_length > 0:
                # Use expand to maintain gradient connection
                importance_ratio_per_token[start_idx:end_idx] = negative_approx_kl_seq[
                    i
                ].expand(seq_length)
    else:
        importance_ratio_per_token = torch.clamp(
            negative_approx_kl, min=-20.0, max=20.0
        )

    importance_ratio_per_token = torch.exp(importance_ratio_per_token)
    importance_ratio = importance_ratio_per_token

    if config.train.train_policy.aipo_rho is not None:
        # Due to the asynchronous update of the reference model, the rollout is not necessarily
        # the exact previous iterate of latest policy. So a more natural motivation is correct
        # for the off-policyness of samples generated under previous policy, to construct
        # approximate on-policy update to latest policy.
        # A difference from double-sided clipping of PPO, we use one-sided clipping.
        rho = config.train.train_policy.aipo_rho
        per_token_loss = -torch.clamp(importance_ratio, max=rho) * current_advantages
    else:
        # the standard grpo loss with dual-clip PPO: https://arxiv.org/pdf/1912.09729
        importance_ratio_clipped = torch.clamp(
            importance_ratio,
            1 - config.train.train_policy.epsilon_low,
            1 + config.train.train_policy.epsilon_high,
        )
        loss1 = importance_ratio * current_advantages
        loss2 = importance_ratio_clipped * current_advantages
        if config.train.train_policy.variant == "gspo":
            per_token_loss = -torch.min(loss1, loss2)
        else:
            loss3 = -config.train.train_policy.lower_bound_ratio * current_advantages
            clip_losses1 = -torch.min(loss1, loss2)
            clip_losses2 = torch.min(loss3, clip_losses1)
            per_token_loss = torch.where(
                current_advantages < 0, clip_losses2, clip_losses1
            )

    if rollout_per_token_logps is not None:
        # Compute behavior KL divergence and importance weight
        behav_kl = old_per_token_logps - rollout_per_token_logps
        behav_imp_weight = torch.exp(behav_kl)
        behav_mask = (
            (behav_imp_weight <= config.train.train_policy.behav_imp_weight_cap)
            if config.train.train_policy.behav_imp_weight_cap is not None
            else torch.ones_like(behav_imp_weight, dtype=torch.bool)
        )
        behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
        per_token_loss = per_token_loss * behav_imp_weight

    off_policy_masking_delta = config.train.train_policy.off_policy_masking_delta
    if off_policy_masking_delta is not None:
        per_token_loss = _apply_off_policy_mask(
            per_token_loss,
            rollout_per_token_logps,
            old_per_token_logps,
            current_token_logps,
            current_advantages,
            cu_seqlens,
            shifted_length,
            off_policy_masking_delta,
        )

    # Compute the KL divergence between the model and the reference model
    if config.train.train_policy.kl_beta != 0.0:
        assert (
            not ref_per_token_logps.requires_grad
        ), "ref_per_token_logps should not require gradient"
        """
            With reference model used for KL. The logic should be further reviewed to verify.
        """
        kl_ratio = ref_per_token_logps - current_token_logps
        # For numerical stability
        kl_ratio = torch.clamp(kl_ratio, min=-20, max=20)
        if config.train.train_policy.unbiased_kl_estimate:
            importance_sampling_ratio = torch.exp(
                current_token_logps - old_per_token_logps
            )
            kl_loss = (
                importance_sampling_ratio * (torch.exp(kl_ratio) - kl_ratio - 1)
            ).clamp(min=-10, max=10)
        else:
            kl_loss = (torch.exp(kl_ratio) - kl_ratio - 1).clamp(min=-10, max=10)

    else:
        kl_loss = torch.zeros_like(per_token_loss)

    bsz, _ = logprob_masks.shape
    per_token_loss_seq_sum = torch.zeros(
        bsz, device=per_token_loss.device, dtype=per_token_loss.dtype
    )  # [bsz,]
    kl_loss_seq_sum = torch.zeros(
        bsz, device=kl_loss.device, dtype=kl_loss.dtype
    )  # [bsz,]
    for i in range(bsz):
        per_token_loss_seq_sum[i] = per_token_loss[
            cu_seqlens[i] : cu_seqlens[i + 1]
        ].sum()
        kl_loss_seq_sum[i] = kl_loss[cu_seqlens[i] : cu_seqlens[i + 1]].sum()
    shifted_length = cu_seqlens[1:] - cu_seqlens[:-1]

    if config.train.train_policy.loss_type == "seq-mean-token-mean":
        # seq-mean-token-sum
        # If Dr.GRPO is used, we need to normalize the loss by the max tokens for unbiased loss
        if (
            config.train.train_policy.unbiased_loss_max_tokens is not None
            and config.train.train_policy.unbiased_loss_max_tokens > 0
        ):
            norm_factor = config.train.train_policy.unbiased_loss_max_tokens
        else:
            norm_factor = shifted_length

        per_token_loss = (per_token_loss_seq_sum / norm_factor).mean()
        kl_loss = (kl_loss_seq_sum / norm_factor).mean()
    elif config.train.train_policy.loss_type == "seq-mean-token-sum":
        # seq-mean-token-sum
        per_token_loss = per_token_loss_seq_sum.mean()
        kl_loss = kl_loss_seq_sum.mean()
    elif config.train.train_policy.loss_type == "token-mean":
        length_sum = shifted_length.sum()
        num_dp_workers = 1
        if config.train.train_policy.balance_dp_token:
            # Balance the number of tokens across data parallel ranks and replicas
            if dp_group is not None:
                # Take DP tokens into account
                num_dp_workers *= torch.distributed.get_world_size(group=dp_group)
                torch.distributed.all_reduce(length_sum, group=dp_group)
            if ddp_comm is not None:
                num_dp_workers *= ddp_comm.world_size()
                ddp_comm.allreduce(
                    length_sum, length_sum, op=torch.distributed.ReduceOp.SUM
                )
        per_token_loss = (
            per_token_loss_seq_sum.sum() / (length_sum + 1e-8) * (num_dp_workers)
        )
        kl_loss = kl_loss_seq_sum.sum() / (length_sum + 1e-8) * (num_dp_workers)
    else:
        raise ValueError(f"Invalid loss type: {config.train.train_policy.loss_type}")
    return (
        per_token_loss + kl_loss * config.train.train_policy.kl_beta,
        per_token_loss,
        kl_loss,
    )


# TODO: (lms) May be it's better to register this func as a hook to the last stage model.
# That way is more clean. I think it's feasible but need to be compatible with torch Pipelie schedule.
def _swizzle_pp_grpo_forward(
    trainer: "GRPOTrainer",
    ori_forward: Callable,
    config: CosmosConfig,
    inter_policy_nccl: HighAvailabilitylNccl,
    *args,
    **kwargs,
):
    args = args[1:]  # Skip self
    """
    Swizzle the forward function (only to last stage) to return the loss directly.
    """
    # [mini_batch_size]: the mini-batch index of the sample with respect to the whole batch
    # [micro_batch_size]: the micro-batch index of the sample with respect to the mini-batch
    mini_batch_ids = kwargs.pop("mini_batch_ids")
    micro_batch_ids = kwargs.pop("micro_batch_ids")
    loss_scaling = kwargs.pop("loss_scaling")
    is_computing_ref = kwargs.pop("is_computing_ref")
    is_computing_old_ahead = kwargs.pop("is_computing_old_ahead")
    advantages = kwargs.pop("advantages")
    positive_flags = kwargs.pop("positive_flags", None)

    micro_batch_id = micro_batch_ids[0].item()
    mini_batch_id = mini_batch_ids[0].item()
    loss_scaling = loss_scaling[0].item()
    is_computing_ref = is_computing_ref[0].item()
    is_computing_old_ahead = is_computing_old_ahead[0].item()

    # User defined input
    user_input = kwargs.copy()

    assert torch.all(
        micro_batch_ids == micro_batch_id
    ), f"micro_batch_ids are not all the same: {micro_batch_ids}"
    assert torch.all(
        mini_batch_ids == mini_batch_id
    ), f"mini_batch_ids are not all the same: {mini_batch_ids}"
    del micro_batch_ids, mini_batch_ids

    n_args = len(args)
    if n_args > 0:
        # remove the first `n_args` arguments from kwargs
        signature = list(inspect.signature(ori_forward).parameters.keys())[:n_args]
        for key in signature:
            if key in kwargs:
                kwargs.pop(key)

    raw_logits = ori_forward(*args, **kwargs)

    # recover the input ids and position ids
    if "input_ids_before_cp" in kwargs:
        user_input["input_ids"] = kwargs["input_ids_before_cp"]
    if "position_ids_before_cp" in kwargs:
        user_input["position_ids"] = kwargs["position_ids_before_cp"]

    if (
        config.train.train_policy.temperature > 1e-6
        and config.train.train_policy.temperature != 1.0
    ):
        raw_logits = raw_logits / config.train.train_policy.temperature
    # [n_tokens, n_vocab]
    current_per_token_logprobs, cu_seqlens, metrics = trainer.compute_logprobs(
        minibatch={
            **user_input,
        },
        logits=raw_logits,
        is_full_logits=True if raw_logits.ndim == 3 else False,
    )
    logprob_masks = user_input["logprob_masks"]
    current_advantages = logprob_masks * advantages

    if positive_flags is not None:
        pos_mask = positive_flags.bool().expand_as(logprob_masks)
        pos_token_mask = pos_mask & logprob_masks
    else:
        pos_token_mask = None

    if is_computing_ref:
        if trainer.ref_per_token_logps[mini_batch_id] is not None:
            assert isinstance(trainer.ref_per_token_logps[mini_batch_id], list)
            trainer.ref_per_token_logps[mini_batch_id].append(
                current_per_token_logprobs.detach()
            )
        else:
            trainer.ref_per_token_logps[mini_batch_id] = [
                current_per_token_logprobs.detach()
            ]
        # Skip the rest logic since we are computing ref
        return None
    if is_computing_old_ahead:
        if trainer.old_per_token_logps[mini_batch_id] is not None:
            assert isinstance(trainer.old_per_token_logps[mini_batch_id], list)
            trainer.old_per_token_logps[mini_batch_id].append(
                current_per_token_logprobs.detach()
            )
        else:
            trainer.old_per_token_logps[mini_batch_id] = [
                current_per_token_logprobs.detach()
            ]
        # Skip the rest logic since we are computing old ahead
        return None

    if (
        trainer.old_per_token_logps[mini_batch_id] is not None
        and len(trainer.old_per_token_logps[mini_batch_id]) > micro_batch_id
    ):
        old_per_token_logprobs = trainer.old_per_token_logps[mini_batch_id][
            micro_batch_id
        ]
        assert isinstance(old_per_token_logprobs, torch.Tensor)
        assert (
            old_per_token_logprobs.ndim == 1
        ), f"old_per_token_logprobs.ndim: {old_per_token_logprobs.ndim}, while it should be 1"
        assert (
            old_per_token_logprobs.shape == current_per_token_logprobs.shape
        ), f"old_per_token_logprobs.shape: {old_per_token_logprobs.shape}, while it should be {current_per_token_logprobs.shape}"
    else:
        old_per_token_logprobs = current_per_token_logprobs.detach()
        # Following should only happen in the first iteration
        if micro_batch_id == 0:
            # assert trainer.old_per_token_logps[mini_batch_id] is None, f"old_per_token_logps[mini_batch_id] should be None"
            # Due to the PP warmup, the first micro-batch could get processed multiple times
            trainer.old_per_token_logps[mini_batch_id] = [old_per_token_logprobs]
        else:
            assert isinstance(trainer.old_per_token_logps[mini_batch_id], list)
            trainer.old_per_token_logps[mini_batch_id].append(old_per_token_logprobs)

    ref_per_token_logprobs = None
    if trainer.ref_per_token_logps[mini_batch_id] is not None:
        ref_per_token_logprobs = trainer.ref_per_token_logps[mini_batch_id][
            micro_batch_id
        ]
        assert (
            ref_per_token_logprobs.ndim == 1
        ), f"ref_per_token_logprobs.ndim: {ref_per_token_logprobs.ndim}, while it should be 1"
        assert (
            ref_per_token_logprobs.shape == current_per_token_logprobs.shape
        ), f"ref_per_token_logprobs.shape: {ref_per_token_logprobs.shape}, while it should be {current_per_token_logprobs.shape}"

    compute_loss_fn = trainer.loss_fn if hasattr(trainer, "loss_fn") else compute_loss
    loss, _, _ = compute_loss_fn(
        current_per_token_logprobs,
        old_per_token_logprobs,
        ref_per_token_logprobs,
        current_advantages,
        cu_seqlens,
        config,
        logprob_masks,
        dp_group=trainer.parallel_dims.mesh["dp"].get_group()
        if trainer.parallel_dims.dp_enabled
        else None,
        ddp_comm=inter_policy_nccl,
        rollout_per_token_logps=user_input.get("rollout_logprobs", None),
    )
    if config.train.train_policy.entropy_coeff > 0.0:
        loss += (
            -config.train.train_policy.entropy_coeff * (metrics["effective_entropy"])
        )
    for key in metrics:
        trainer.metrics[key] += metrics[key]

    # Add Positive NLL if enabled and mask available
    pos_coef = config.train.train_policy.positive_nll_coef
    if (
        pos_coef is not None
        and pos_coef > 0.0
        and pos_token_mask is not None
        and pos_token_mask.any()
    ):
        flat_mask = pos_token_mask[logprob_masks]
        l_nll = -current_per_token_logprobs[flat_mask].mean()
        loss = loss + pos_coef * l_nll

    return loss.unsqueeze(0) * loss_scaling


@TrainerRegistry.register(trainer_type="grpo")
class GRPOTrainer(LLMTrainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        val_data_packer: BaseDataPacker,
        **kwargs,
    ):
        super(GRPOTrainer, self).__init__(
            config,
            parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            **kwargs,
        )

        self.reference_state_dict = {}

        self.lr_schedulers = self.build_lr_schedulers()
        self.lr_schedulers_updated = False
        if parallel_dims.dp_replicate > 1:
            raise ValueError(
                f"DP replicate size {parallel_dims.dp_replicate} is not supported for GRPO"
                "Please use elastic scaling feature instead."
            )

        self.grpo_config = self.config.train.train_policy

        # For iteration control
        self.mini_step = 0
        self.replica_batch_for_this_step = 0
        self.mini_batch = self.grpo_config.mini_batch
        self.batch_size_per_optimize = self.grpo_config.batch_size_per_optimize

        # For GRPO
        self.max_length = config.policy.model_max_length
        self.mu_iterations = self.config.train.train_policy.mu_iterations
        self.optimizers.zero_grad()

        if config.train.train_policy.variant == "gspo":
            logger.info("[Policy] Use GSPO loss in RL.")

        self.tokenizer = setup_tokenizer(self.config.policy.model_name_or_path)

        # For teacher model interaction
        self.teacher_interact_results: Dict[str, Any] = {}
        self.fetched_teacher_uuids: Set[str] = set()

    def collate_teacher_logprobs(
        self,
        rollouts: List[Rollout],
        processed_samples: List[Any],
        computed_max_len: int,
    ) -> List[List[float]]:
        teacher_logprobs_list = []
        assert (
            len(processed_samples) == len(rollouts)
        ), f"Length of processed_samples {len(processed_samples)} should be equal to length of rollouts {len(rollouts)}"
        for i in range(len(rollouts)):
            # get the teacher logprobs for current rollout
            teacher_logprobs = rollouts[i].teacher_logprobs
            if (
                self.config.train.train_policy.collect_rollout_logprobs
                and teacher_logprobs is not None
            ):
                sampled_completion_logprobs = rollouts[i].completion_logprobs
                sampled_prompt_logprobs = rollouts[i].prompt_logprobs
                sampled_logprobs = sampled_prompt_logprobs + sampled_completion_logprobs
                assert (
                    len(sampled_logprobs) == len(teacher_logprobs)
                ), f"sampled_logprobs: {len(sampled_logprobs)} != teacher_logprobs: {len(teacher_logprobs)}"
            if teacher_logprobs is None:
                logger.warning(
                    f"[Policy] Teacher logprobs is None for rollout {i}, using [0] * {computed_max_len} and set the logprob_masks to all 0 to avoid the loss calculation due to lack of teacher logprobs"
                )
                teacher_logprobs = [0] * computed_max_len
                if hasattr(processed_samples[i], "logprob_masks"):
                    # set the logprob_masks to all 0 to avoid the loss calculation due to lack of teacher logprobs
                    processed_samples[i].logprob_masks = [
                        0 for _ in processed_samples[i].logprob_masks
                    ]
                else:
                    processed_samples[i]["logprob_masks"] = [
                        0 for _ in processed_samples[i]["logprob_masks"]
                    ]
            teacher_logprobs = teacher_logprobs + [0]
            teacher_logprobs_list.append(teacher_logprobs)
            if self.config.distillation.include_prompt:
                if hasattr(processed_samples[i], "logprob_masks"):
                    processed_samples[i].logprob_masks = [
                        1 for _ in range(len(processed_samples[i].logprob_masks))
                    ]
                    processed_samples[i].logprob_masks[-1] = 0  # exclude the last token
                else:
                    processed_samples[i]["logprob_masks"] = [
                        1 for _ in range(len(processed_samples[i]["logprob_masks"]))
                    ]
                    processed_samples[i]["logprob_masks"][-1] = (
                        0  # exclude the last token
                    )

        return torch.tensor(
            [
                x[:computed_max_len] + [0] * (max(0, computed_max_len - len(x)))
                for x in teacher_logprobs_list
            ]
        )

    def compute_teacher_kl_advantages(
        self,
        current_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        current_advantages: torch.Tensor,
        logprob_masks: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        kl_penalty_coef = self.config.distillation.kl_penalty_coef
        kl_discount_factor = self.config.distillation.kl_discount_factor
        current_advantages = torch.masked_select(current_advantages, logprob_masks)
        teacher_logprobs = torch.masked_select(teacher_logprobs, logprob_masks)
        assert (
            current_logprobs.shape == teacher_logprobs.shape == current_advantages.shape
        ), f"current_logprobs.shape: {current_logprobs.shape} != teacher_logprobs.shape: {teacher_logprobs.shape} != current_advantages.shape: {current_advantages.shape}"
        reversed_kl = current_logprobs - teacher_logprobs
        kl_advantages = -kl_penalty_coef * reversed_kl  # [n_tokens]

        def discounted_future_sum_loop(x: list[float], gamma: float) -> list[float]:
            returns = [0] * len(x)
            cumulative = 0
            for i in range(len(x) - 1, -1, -1):
                cumulative = x[i] + gamma * cumulative
                returns[i] = cumulative
            return returns

        metrics = {
            "teacher_reversed_kl": reversed_kl.sum() / logprob_masks.sum(),
            "teacher_kl_advantages": kl_advantages.sum() / logprob_masks.sum(),
        }

        assert (
            cu_seqlens[-1] == kl_advantages.shape[0]
        ), f"cu_seqlens[-1]: {cu_seqlens[-1]} != kl_advantages.shape[0]: {kl_advantages.shape[0]}"
        if kl_discount_factor != 0.0:
            # Compute discounted future sum of kl_advantages for each sequence
            # Only needed when kl_discount_factor != 0.0
            for i in range(cu_seqlens.shape[0] - 1):
                start_idx = cu_seqlens[i]
                end_idx = cu_seqlens[i + 1]
                kl_advantages_discounted = discounted_future_sum_loop(
                    kl_advantages[start_idx:end_idx].tolist(), kl_discount_factor
                )
                kl_advantages[start_idx:end_idx] = torch.tensor(
                    kl_advantages_discounted
                ).to(self.device)
        advantages = current_advantages + kl_advantages
        metrics.update(
            {
                "teacher_kl_advantages_discounted": kl_advantages.sum()
                / logprob_masks.sum(),
            }
        )
        return advantages, metrics

    def fetch_teacher_logprobs(
        self,
        rollouts: List[Rollout],
        mini_batch_indices: List[int],
    ):
        all_uuids = [rollouts[i].teacher_result_uuid for i in mini_batch_indices]
        all_rank_uuids = dist_util.all_gather_object_cpu(all_uuids)
        scattered_teacher_logprobs = [[] for _ in range(self.world_size)]
        teacher_logprobs = [[]]
        if self.global_rank == 0:
            for index, rank_uuids in enumerate(all_rank_uuids):
                teacher_logprobs_needed = []
                for id in rank_uuids:
                    while id not in self.teacher_interact_results:
                        time.sleep(0.01)
                    self.fetched_teacher_uuids.add(id)
                    teacher_logprobs_needed.append(self.teacher_interact_results[id])
                scattered_teacher_logprobs[index] = teacher_logprobs_needed
        dist.scatter_object_list(
            teacher_logprobs,
            scattered_teacher_logprobs,
            src=0,
        )
        all_teacher_logprobs = teacher_logprobs[0]
        assert (
            len(all_teacher_logprobs) == len(mini_batch_indices)
        ), f"Length of all_teacher_logprobs {len(all_teacher_logprobs)} should be equal to length of mini_batch_indices {len(mini_batch_indices)}"
        for teacher_logprobs, idx in zip(all_teacher_logprobs, mini_batch_indices):
            rollouts[idx].teacher_logprobs = teacher_logprobs

    def clear_teacher_result_cache(self):
        # Clear the cached teacher interaction results to save memory
        for id in self.fetched_teacher_uuids:
            assert (
                id in self.teacher_interact_results
            ), f"Teacher result uuid {id} not found in teacher_interact_results"
            del self.teacher_interact_results[id]
        self.fetched_teacher_uuids.clear()

    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        pp_last_stage = (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )
        # Do it once
        if (
            pp_last_stage
            and self.parallel_dims.pp_enabled
            and not hasattr(self, "swizzled_forward")
        ):
            # Swizzle the forward function to return the current per-token logprobs.
            orig_forward = self.model.forward
            self.model.forward = types.MethodType(
                partial(
                    _swizzle_pp_grpo_forward,
                    self,
                    orig_forward,
                    self.config,
                    inter_policy_nccl,
                ),
                self.model,
            )
            self.swizzled_forward = True

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # For single-turn rollout, we use the prompt, for multi-turn rollout, we use the completed conversation
        if self.config.rollout.multi_turn_config.enable:
            samples = [rollout.completed_conversation for rollout in rollouts]
        else:
            samples = [rollout.prompt for rollout in rollouts]
        assert all(
            rollout.prompt is not None for rollout in rollouts
        ), "All rollouts should have a valid prompt"

        completions_list = [
            rollout.completion_token_ids
            if self.config.train.train_policy.rollout_as_token_ids
            else rollout.completion
            for rollout in rollouts
        ]

        # Optional Positive-NLL support: only compute flags when coefficient > 0
        pos_coef_global = self.config.train.train_policy.positive_nll_coef
        if pos_coef_global is not None and pos_coef_global > 0.0:
            rewards_list = [rollout.reward for rollout in rollouts]
            self._positive_flags_t = torch.tensor(
                [1 if r > 0 else 0 for r in rewards_list],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            self._positive_flags_t = None
        n_ignore_prefix_tokens_list = [
            rollout.n_ignore_prefix_tokens for rollout in rollouts
        ]
        assert all(
            samples[i] is not None for i in range(len(samples))
        ), "All samples should be not None"
        processed_samples: List[Any] = [
            self.data_packer.get_policy_input(
                samples[i],
                completions_list[i],
                n_ignore_prefix_tokens_list[i],
            )
            for i in range(len(samples))
        ]

        # On-policy Distillation related computations
        assert (
            len(processed_samples) == len(rollouts) and len(samples) == len(rollouts)
        ), f"Length of processed_samples {len(processed_samples)} should be equal to length of rollouts {len(rollouts)}"
        advantages_list = [rollout.advantage for rollout in rollouts]
        advantages_t = torch.tensor(advantages_list).to(self.device)

        self.metrics = {
            "entropy": 0.0,
            "effective_entropy": 0.0,
        }
        # user_info_keys = list(kwargs.keys())
        batch_size = len(rollouts)
        per_optimize_batch_size = (
            min(self.batch_size_per_optimize, batch_size)
            if self.batch_size_per_optimize is not None
            and self.batch_size_per_optimize > 0
            else batch_size
        )

        mini_batch_size = (
            min(self.mini_batch, per_optimize_batch_size)
            if self.mini_batch > 0
            else per_optimize_batch_size
        )
        num_mini_batch = batch_size // mini_batch_size

        # Initialize placeholder for old per-token logprobs
        self.old_per_token_logps = [None for _ in range(num_mini_batch)]
        self.ref_per_token_logps = [None for _ in range(num_mini_batch)]

        acc_n_tokens = 0
        # Validate the PP parallelism configuration
        if self.parallel_dims.pp_enabled:
            n_microbatches = (
                mini_batch_size // self.config.policy.parallelism.pp_micro_batch_size
            )
            assert (
                n_microbatches % self.parallel_dims.pp == 0
            ), f"n_microbatches {n_microbatches} should be divided evenly by pp size of {self.parallel_dims.pp}"

        need_compute_ref, kl_beta = self._swap_model_state_dict()
        need_compute_old_ahead = batch_size > per_optimize_batch_size
        loss_sum = torch.tensor(0.0, device=self.device)
        kl_loss_sum = torch.tensor(0.0, device=self.device)
        grad_norm_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0
        grad_norm_count = 0

        trainer_phases = []
        if need_compute_ref:
            trainer_phases.append(TrainerPhase.REF_COMPUTE)
        if need_compute_old_ahead:
            trainer_phases.append(TrainerPhase.OLD_LOGP_COMPUTE)
        trainer_phases.append(TrainerPhase.TRAIN)

        cached_minibatch_arrangements = []
        for phase in trainer_phases:
            is_computing_ref = phase == TrainerPhase.REF_COMPUTE
            is_computing_old_ahead = phase == TrainerPhase.OLD_LOGP_COMPUTE
            # Set model to eval mode if reference model is being used
            if is_computing_ref:
                self.model.eval()
            else:
                if need_compute_ref:
                    # Swap model state dict back to the original model
                    need_compute_ref = False
                    self._swap_model_state_dict()
                if is_computing_old_ahead:
                    self.model.eval()
                else:
                    self.model.train()

            with torch.set_grad_enabled(phase == TrainerPhase.TRAIN):
                for i_mu in range(
                    1
                    if (is_computing_ref or is_computing_old_ahead)
                    else self.mu_iterations
                ):
                    local_mini_step = 0
                    local_optimize_step = 0
                    with torch.cuda.stream(self.train_stream):
                        for i in range(0, batch_size, per_optimize_batch_size):
                            end = min(i + per_optimize_batch_size, batch_size)
                            # Convert advantages from [batch_size] -> [batch_size, max_len] via expanding
                            processed_samples_for_optimize = processed_samples[i:end]
                            if len(cached_minibatch_arrangements) > local_optimize_step:
                                (
                                    mini_batches,
                                    mini_batch_index,
                                ) = cached_minibatch_arrangements[local_optimize_step]
                            else:
                                if (
                                    self.config.train.train_policy.max_token_len_per_mini_batch
                                    is not None
                                    and self.config.train.train_policy.max_token_len_per_mini_batch
                                    > 0
                                ):
                                    minibatch_seq_len = [
                                        self.data_packer.policy_compute_max_len(
                                            [sample]
                                        )
                                        for sample in processed_samples_for_optimize
                                    ]
                                    # split batch into mini_batches with sequence parallelism
                                    if self.parallel_dims.cp_enabled:
                                        cp_size = self.parallel_dims.mesh["cp"].size()
                                    else:
                                        cp_size = 1
                                    max_token_len = (
                                        self.config.train.train_policy.max_token_len_per_mini_batch
                                        * cp_size
                                    )
                                    # dynamic rearrange mini batches
                                    mini_batches, mini_batch_index = (
                                        rearrange_mini_batches(
                                            batch=processed_samples_for_optimize,
                                            seq_len_effective=minibatch_seq_len,
                                            max_token_len=max_token_len,
                                            ddp_comm=inter_policy_nccl,
                                        )
                                    )
                                else:
                                    # split batch into mini_batches
                                    mini_batches = [
                                        processed_samples_for_optimize[
                                            i : i + self.mini_batch
                                        ]
                                        for i in range(
                                            0,
                                            len(processed_samples_for_optimize),
                                            self.mini_batch,
                                        )
                                    ]
                                    mini_batch_index = [
                                        list(
                                            range(
                                                i,
                                                min(
                                                    i + self.mini_batch,
                                                    len(processed_samples_for_optimize),
                                                ),
                                            )
                                        )
                                        for i in range(
                                            0,
                                            len(processed_samples_for_optimize),
                                            self.mini_batch,
                                        )
                                    ]
                                cached_minibatch_arrangements.append(
                                    (mini_batches, mini_batch_index)
                                )
                            for (
                                minibatched_processed_samples,
                                mini_batch_indices,
                            ) in zip(mini_batches, mini_batch_index):
                                loss_scaling_factor = len(
                                    minibatched_processed_samples
                                ) / len(processed_samples_for_optimize)
                                # TODO(jiaxin): support variable length in PP
                                computed_max_len = (
                                    self.config.policy.model_max_length
                                    if self.parallel_dims.pp_enabled
                                    else self.data_packer.policy_compute_max_len(
                                        minibatched_processed_samples
                                    )
                                )

                                computed_max_len = (
                                    (computed_max_len + self.seq_len_multiple - 1)
                                    // self.seq_len_multiple
                                    * self.seq_len_multiple
                                )
                                minibatched_advantages = (
                                    advantages_t[mini_batch_indices]
                                    .unsqueeze(1)
                                    .expand(-1, computed_max_len)
                                    .to(self.device)
                                )

                                if self.config.distillation.enable:
                                    self.fetch_teacher_logprobs(
                                        rollouts=rollouts,
                                        mini_batch_indices=mini_batch_indices,
                                    )
                                    if all(
                                        [
                                            rollouts[i].teacher_logprobs is None
                                            for i in mini_batch_indices
                                        ]
                                    ):
                                        logger.warning(
                                            "[Policy] All teacher logprobs are None for current mini-batch, skipping distillation loss calculation."
                                        )
                                        continue
                                    minibatched_teacher_logprobs = self.collate_teacher_logprobs(
                                        rollouts=[
                                            rollouts[i] for i in mini_batch_indices
                                        ],
                                        processed_samples=minibatched_processed_samples,
                                        computed_max_len=computed_max_len,
                                    )
                                user_mini_batch: Dict[str, Any] = (
                                    self.data_packer.policy_collate_fn(
                                        minibatched_processed_samples,
                                        computed_max_len=computed_max_len,
                                    )
                                )
                                if self.config.train.train_policy.use_decoupled_loss:
                                    rollout_logbprobs = []
                                    assert len(mini_batch_indices) == len(
                                        user_mini_batch["logprob_masks"]
                                    )
                                    for i in mini_batch_indices:
                                        assert (
                                            len(rollouts[i].completion_logprobs)
                                            == len(completions_list[i])
                                        ), f"Unexpected completion_logprobs length {len(rollouts[i].completion_logprobs)} vs completion length {len(completions_list[i])}"
                                        # Skip the last token logprob which is for <eos> if needed
                                        # Skip the n_ignore_prefix_tokens as they are not included in the loss calculation
                                        rollout_logbprobs.append(
                                            rollouts[i].completion_logprobs[
                                                n_ignore_prefix_tokens_list[i] :
                                            ]
                                        )
                                    user_mini_batch["rollout_logprobs"] = (
                                        rollout_logbprobs
                                    )
                                packing_seq = self.config.train.sequence_packing
                                if packing_seq:
                                    if self.parallel_dims.pp_enabled:
                                        packing_seq = False
                                        logger.debug(
                                            "[Policy] Packing sequence is disabled due to incompatible dimensions."
                                        )
                                    elif (
                                        hasattr(
                                            self.model,
                                            "check_sequence_packing_compatible",
                                        )
                                        and not self.model.check_sequence_packing_compatible()
                                    ):
                                        packing_seq = False
                                        logger.debug(
                                            "[Policy] Packing sequence is disabled due to unsupported model."
                                        )

                                # TP/CP will shard the sequence dimension into n-ranks.
                                # The interested_tokens will be unevenly distributed across ranks.
                                # So do not enable interested_tokens in TP.
                                if (
                                    self.parallel_dims.dp_shard_coord[1]
                                    == self.parallel_dims.world_size
                                ):
                                    user_mini_batch["interested_tokens"] = (
                                        user_mini_batch["logprob_masks"]
                                    )

                                # Move all tensor to device
                                for k in user_mini_batch.keys():
                                    v = user_mini_batch[k]
                                    if (
                                        isinstance(v, torch.Tensor)
                                        and v.device != self.device
                                    ):
                                        user_mini_batch[k] = v.to(self.device)

                                # input_ids are different across ranks in dp_shard_cp
                                position_ids, input_ids, pos_seq_dim = (
                                    self.model.get_position_ids(**user_mini_batch)
                                )

                                if packing_seq:
                                    # Prepare for the sequence packing information.
                                    packed_args = pack_sequences_info_collect(
                                        input_ids,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        seq_len_multiple=self.seq_len_multiple,
                                    )
                                    user_mini_batch.update(packed_args)
                                    packed_args = pack_sequences_for_masks(
                                        user_mini_batch["valid_input_len"],
                                        user_mini_batch["valid_input_len"],
                                    )
                                    user_mini_batch.update(packed_args)
                                    packed_args = pack_sequences_for_logprobs(
                                        user_mini_batch["logprob_masks"],
                                        user_mini_batch["valid_input_len"],
                                        advantages=advantages_t[mini_batch_indices],
                                    )
                                    user_mini_batch.update(packed_args)
                                    minibatched_advantages = user_mini_batch.pop(
                                        "advantages"
                                    )

                                acc_n_tokens += np.prod(input_ids.shape)
                                user_mini_batch["position_ids"] = position_ids
                                padding_mask = user_mini_batch.get("padding_mask", None)

                                input_ids_before_cp = user_mini_batch["input_ids"]
                                position_ids_before_cp = user_mini_batch["position_ids"]
                                padding_mask_before_cp = padding_mask
                                # For VLMs, we need to delay the slice of inputs for CP until after the embedding generation in the model forward.
                                delay_cp_slice_inputs = getattr(
                                    self.model, "delay_cp_slice_inputs", False
                                )
                                if (
                                    self.parallel_dims.cp_enabled
                                    and not packing_seq
                                    and not delay_cp_slice_inputs
                                ):
                                    [input_ids, position_ids, padding_mask] = (
                                        slice_inputs_for_ulysses(
                                            [input_ids, position_ids, padding_mask],
                                            self.parallel_dims.mesh["cp"],
                                            seq_dims=[1, pos_seq_dim, 1],
                                        )
                                    )
                                    user_mini_batch["position_ids"] = position_ids
                                    user_mini_batch["input_ids"] = input_ids
                                    if padding_mask is not None:
                                        user_mini_batch["padding_mask"] = padding_mask
                                if self.parallel_dims.cp_enabled:
                                    # Slice for cp after embedding generation and sequence packing in the model forward later.
                                    user_mini_batch["cp_mesh"] = (
                                        self.parallel_dims.mesh["cp"]
                                    )

                                if self.parallel_dims.pp_enabled:
                                    if pp_last_stage:
                                        if (
                                            self.old_per_token_logps[local_mini_step]
                                            is None
                                        ):
                                            assert (
                                                i_mu == 0
                                            ), "Only first `mu_iteration` should append `old_per_token_logps`"
                                        else:
                                            assert (
                                                i_mu > 0
                                            ), "Only `mu_iteration > 0` should reuse `old_per_token_logps`"
                                            assert (
                                                len(
                                                    self.old_per_token_logps[
                                                        local_mini_step
                                                    ]
                                                )
                                                == n_microbatches
                                            )

                                    # [mini_batch_size, 1]: indicating the index of mini-batch
                                    mini_batch_ids_cpu = torch.Tensor(
                                        [[local_mini_step]] * mini_batch_size
                                    ).int()
                                    micro_batch_ids_list = []
                                    for i in range(mini_batch_size):
                                        micro_batch_ids_list.append(
                                            [
                                                i
                                                // self.config.policy.parallelism.pp_micro_batch_size
                                            ]
                                        )
                                    micro_batch_ids_cpu = torch.Tensor(
                                        micro_batch_ids_list
                                    ).int()
                                    loss_scaling_cpu = torch.tensor(
                                        [
                                            [
                                                1.0
                                                * loss_scaling_factor
                                                / self.config.policy.parallelism.pp_micro_batch_size
                                            ]
                                        ]
                                        * mini_batch_size,
                                        dtype=torch.float32,
                                    )
                                    is_computing_ref_cpu = torch.tensor(
                                        [is_computing_ref] * mini_batch_size,
                                        dtype=torch.bool,
                                    )
                                    is_computing_old_ahead_cpu = torch.tensor(
                                        [is_computing_old_ahead] * mini_batch_size,
                                        dtype=torch.bool,
                                    )
                                    # Positive flags for Positive-NLL loss (only if coef >0)
                                    if self._positive_flags_t is not None:
                                        is_pos_cpu = (
                                            self._positive_flags_t[mini_batch_indices]
                                            .unsqueeze(1)
                                            .expand(-1, 1)
                                            .int()
                                        )
                                        user_mini_batch["positive_flags"] = is_pos_cpu

                                    pp_first_stage = self.parallel_dims.pp_coord[0] == 0
                                    # Pipeline Parallel forward / backward inside step() call
                                    losses = [] if pp_last_stage else None
                                    if pp_last_stage:
                                        # Inject the `mini-batch` and `micro-batch` ids to the input so that the last stage can know which microbatch it is processing
                                        user_mini_batch["mini_batch_ids"] = (
                                            mini_batch_ids_cpu
                                        )
                                        user_mini_batch["micro_batch_ids"] = (
                                            micro_batch_ids_cpu
                                        )
                                        user_mini_batch["loss_scaling"] = (
                                            loss_scaling_cpu
                                        )
                                        user_mini_batch["is_computing_ref"] = (
                                            is_computing_ref_cpu
                                        )
                                        user_mini_batch["is_computing_old_ahead"] = (
                                            is_computing_old_ahead_cpu
                                        )
                                        if self._positive_flags_t is not None:
                                            user_mini_batch["positive_flags"] = (
                                                is_pos_cpu
                                            )
                                    if pp_first_stage or pp_last_stage:
                                        # First/Last stage: pass all inputs
                                        kwargs = {}
                                        if self.parallel_dims.cp_enabled:
                                            # This is for recover these two tensors after ulysses
                                            kwargs["input_ids_before_cp"] = (
                                                input_ids_before_cp
                                            )
                                            kwargs["position_ids_before_cp"] = (
                                                position_ids_before_cp
                                            )

                                        self.pp_scheduler.step(
                                            **user_mini_batch,
                                            advantages=minibatched_advantages,
                                            losses=losses,
                                            target=torch.empty(
                                                [mini_batch_size, 1], device=self.device
                                            ),
                                            **kwargs,
                                        )
                                    else:
                                        # Middle stages: forward data from previous stage
                                        self.pp_scheduler.step(
                                            position_ids=position_ids
                                        )

                                    if is_computing_ref or is_computing_old_ahead:
                                        # Continue to next mini-batch since loss is not needed for reference model
                                        continue
                                    else:
                                        loss = (
                                            torch.mean(torch.stack(losses)).to(
                                                self.device
                                            )
                                            if pp_last_stage
                                            else torch.tensor(
                                                [-1.0], device=self.device
                                            )
                                        )
                                else:
                                    with self.act_offloading_ctx_manager:
                                        raw_logits = self.model(**user_mini_batch)

                                    if self.parallel_dims.cp_enabled:
                                        # reset the position ids and input ids
                                        user_mini_batch["position_ids"] = (
                                            position_ids_before_cp
                                        )
                                        user_mini_batch["input_ids"] = (
                                            input_ids_before_cp
                                        )
                                        if padding_mask_before_cp is not None:
                                            user_mini_batch["padding_mask"] = (
                                                padding_mask_before_cp
                                            )

                                    if (
                                        self.config.train.train_policy.temperature
                                        > 1e-6
                                        and self.config.train.train_policy.temperature
                                        != 1.0
                                    ):
                                        raw_logits = (
                                            raw_logits
                                            / self.config.train.train_policy.temperature
                                        )
                                    # returned shape:
                                    # current_per_token_logprobs: [n_tokens_of_logprobs]
                                    # cu_seqlens: [batch_size + 1]
                                    if packing_seq:
                                        # Pack sequences for inputs to match the logits from model forward.
                                        packed_args = pack_sequences_for_inputs(
                                            user_mini_batch["input_ids"],
                                            user_mini_batch["valid_input_len"],
                                        )
                                        user_mini_batch["input_ids"] = packed_args[
                                            "inputs"
                                        ]

                                    (
                                        current_per_token_logprobs,
                                        cu_seqlens,
                                        metrics,
                                    ) = self.compute_logprobs(
                                        user_mini_batch,
                                        logits=raw_logits,
                                        is_full_logits=True
                                        if raw_logits.ndim == 3
                                        else False,
                                    )
                                    logprob_masks = user_mini_batch["logprob_masks"]
                                    current_advantages = (
                                        logprob_masks * minibatched_advantages
                                    )
                                    if self.config.distillation.enable:
                                        if packing_seq:
                                            minibatched_teacher_logprobs = (
                                                pack_sequences_for_teacher_logprobs(
                                                    minibatched_teacher_logprobs.to(
                                                        self.device
                                                    ),
                                                    user_mini_batch["valid_input_len"],
                                                )
                                            )
                                        current_advantages, teacher_metrics = (
                                            self.compute_teacher_kl_advantages(
                                                current_logprobs=current_per_token_logprobs,
                                                teacher_logprobs=minibatched_teacher_logprobs.to(
                                                    self.device
                                                ),
                                                current_advantages=current_advantages,
                                                logprob_masks=logprob_masks,
                                                cu_seqlens=cu_seqlens,
                                            )
                                        )

                                    # Compute ref per-token logprobs if needed
                                    if is_computing_ref:
                                        assert (
                                            i_mu == 0
                                        ), "Only first iteration should compute ref"
                                        self.ref_per_token_logps[local_mini_step] = (
                                            current_per_token_logprobs.detach()
                                        )
                                        # Skip the rest of the loop
                                        local_mini_step += 1
                                        continue
                                    elif is_computing_old_ahead:
                                        assert (
                                            i_mu == 0
                                        ), "Only first iteration should compute old ahead"
                                        self.old_per_token_logps[local_mini_step] = (
                                            current_per_token_logprobs.detach()
                                        )
                                        local_mini_step += 1
                                        continue
                                    else:
                                        if (
                                            self.old_per_token_logps[local_mini_step]
                                            is None
                                        ):
                                            assert (
                                                i_mu == 0 and not need_compute_old_ahead
                                            ), "Only first iteration should append `old_per_token_logps`"
                                            self.old_per_token_logps[
                                                local_mini_step
                                            ] = current_per_token_logprobs.detach()
                                        else:
                                            assert (
                                                i_mu > 0 or need_compute_old_ahead
                                            ), "Only inner iteration should reuse `old_per_token_logps`"

                                        compute_loss_fn = (
                                            self.loss_fn
                                            if hasattr(self, "loss_fn")
                                            else compute_loss
                                        )
                                        loss, per_token_loss, kl_loss = compute_loss_fn(
                                            current_per_token_logprobs,
                                            self.old_per_token_logps[local_mini_step],
                                            self.ref_per_token_logps[local_mini_step],
                                            current_advantages,
                                            cu_seqlens,
                                            self.config,
                                            logprob_masks,
                                            dp_group=self.parallel_dims.mesh[
                                                "dp"
                                            ].get_group()
                                            if self.parallel_dims.dp_enabled
                                            else None,
                                            ddp_comm=inter_policy_nccl,
                                            rollout_per_token_logps=user_mini_batch.get(
                                                "rollout_logprobs", None
                                            ),
                                        )
                                        if (
                                            self.config.train.train_policy.entropy_coeff
                                            > 0.0
                                        ):
                                            loss += (
                                                -self.config.train.train_policy.entropy_coeff
                                                * (metrics["effective_entropy"])
                                            )

                                        # Positive Example LM Loss
                                        if (
                                            pos_coef_global is not None
                                            and pos_coef_global > 0.0
                                        ):
                                            pos_flag_batch = self._positive_flags_t[
                                                mini_batch_indices
                                            ]
                                            pos_mask = pos_flag_batch.unsqueeze(
                                                1
                                            ).expand_as(logprob_masks)
                                            pos_token_mask = pos_mask & logprob_masks
                                            if pos_token_mask.any():
                                                flat_mask = pos_token_mask[
                                                    logprob_masks
                                                ]
                                                l_nll = -current_per_token_logprobs[
                                                    flat_mask
                                                ].mean()
                                                loss = loss + pos_coef_global * l_nll

                                        for key in metrics:
                                            self.metrics[key] += metrics[key]
                                        loss = loss * loss_scaling_factor
                                        per_token_loss = (
                                            per_token_loss * loss_scaling_factor
                                        )
                                        kl_loss = kl_loss * loss_scaling_factor

                                        loss.backward()
                                        loss_sum += (
                                            per_token_loss.item() / loss_scaling_factor
                                        )
                                        kl_loss_sum += (
                                            kl_loss.item() / loss_scaling_factor
                                        )
                                        loss_count += 1
                                self.mini_step += 1
                                local_mini_step += 1

                                if (
                                    os.environ.get("COSMOS_GRPO_STEP_INTERVAL", None)
                                    is not None
                                    and local_mini_step
                                    % int(os.environ.get("COSMOS_GRPO_STEP_INTERVAL"))
                                    == 0
                                ) and local_mini_step > 1:
                                    all_reduced = True
                                    grad_norm_sum += self.all_reduce_states(
                                        inter_policy_nccl
                                    )
                                    grad_norm_count += 1
                                else:
                                    all_reduced = False

                            if (
                                not is_computing_ref
                                and not is_computing_old_ahead
                                and not all_reduced
                            ):
                                grad_norm_sum += self.all_reduce_states(
                                    inter_policy_nccl
                                )
                                grad_norm_count += 1
                            local_optimize_step += 1
        self.old_per_token_logps = []
        self.ref_per_token_logps = []
        end_event.record()

        loss = (loss_sum / loss_count) if loss_count > 0 else loss_sum
        kl_loss = (kl_loss_sum / loss_count) if loss_count > 0 else kl_loss_sum
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(loss, self.parallel_dims.mesh["dp_cp"]),
            )
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss, global_max_kl_loss = (  # noqa: F841
                    dist_util.dist_mean(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                    dist_util.dist_max(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                )
        else:
            global_avg_loss = global_max_loss = loss.item()  # noqa: F841
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if is_master_rank(self.parallel_dims, self.global_rank):
                report_data = {"train_step": current_step}
                # Calculate the iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds
                report_data["train/iteration_time"] = iter_time
                report_data["train/loss_avg"] = global_avg_loss
                report_data["train/loss_max"] = global_max_loss
                report_data["train/learning_rate"] = self.lr_schedulers.get_last_lr()[0]
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss
                report_data["train/grad_norm"] = (
                    grad_norm_sum.item() / grad_norm_count
                    if grad_norm_count > 0
                    else 0.0
                )
                if len(self.metrics) > 0:
                    for k, v in self.metrics.items():
                        report_data[f"train/{k}"] = (
                            v.item() if isinstance(v, torch.Tensor) else v
                        ) / loss_count
                if self.config.distillation.enable:
                    for k, v in teacher_metrics.items():
                        report_data[f"train/{k}"] = v.item()

                # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                # it will be inaccurate. Need a reduce for all the metrics.
                if self.config.logging.report_mfu:
                    mfu = compute_mfu(
                        model=self.model,
                        n_tokens=acc_n_tokens,
                        iter_time=iter_time,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v

        # Only step lr scheduler when all the mini-batches are processed
        self.lr_schedulers.step()

        # checkpointing
        if is_master_replica and (do_save_checkpoint):
            if self.config.train.ckpt.export_safetensors:
                logger.info(
                    f"[Policy] Saving huggingface checkpoint at step {current_step} to {self.config.train.output_dir}..."
                )
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=os.path.join(
                        "safetensors",
                        f"step_{current_step}",
                    ),
                    trainable_only=False,
                    is_final=current_step == total_steps,
                    dtype=str2torch_dtype(self.config.train.param_dtype),
                )
            logger.info(f"[Policy] Saving cosmos checkpoint at step {current_step}...")
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=current_step,
                total_steps=total_steps,
                **{
                    "remain_samples_num": remain_samples_num,
                    "is_final": current_step == total_steps,
                },
            )
            self.ckpt_manager.save_check(step=current_step)

        self.reference_reset(current_step)

        self.clear_teacher_result_cache()
        return report_data

    def reference_reset(self, current_step: int):
        if (
            self.config.train.train_policy.kl_beta != 0.0
            and self.config.train.train_policy.reference_reset_interval is not None
            and self.config.train.train_policy.reference_reset_interval > 0
        ):
            if (
                current_step % self.config.train.train_policy.reference_reset_interval
                == 0
            ):
                logger.info(
                    f"[Policy] Resetting reference model at step {current_step} with interval {self.config.train.train_policy.reference_reset_interval}"
                )
                # Update the state dict of hf model so that it can be used for KL-divergence calculation
                state_dict = self.model.state_dict()
                for key, value in state_dict.items():
                    assert (
                        key in self.reference_state_dict
                    ), f"Key {key} not found in reference state dict"
                    self.reference_state_dict[key] = value.detach().cpu()
                if self.config.train.train_policy.reset_optimizer_with_reference:
                    logger.info("[Policy] Resetting optimizer.")
                    self.build_optimizers()
                    for lr in self.lr_schedulers.schedulers:
                        lr.optimizer = self.optimizers

    @torch.no_grad()
    def _swap_model_state_dict(self):
        kl_beta = self.config.train.train_policy.kl_beta
        if kl_beta != 0.0:
            with torch.cuda.stream(self.train_stream):
                model_state_dict = self.model.state_dict()
                reference_state_dict = self.reference_state_dict
                for key, value in model_state_dict.items():
                    # clone the reference state dict to avoid inplace operation
                    ref_clone = reference_state_dict[key].clone()
                    # copy the current model state dict to the reference state dict
                    reference_state_dict[key].copy_(value)
                    # copy the reference state dict to the current model state dict
                    value.copy_(ref_clone)
            return True, kl_beta
        else:
            return False, 0.0

    def compute_logprobs(
        self,
        minibatch: Dict[str, Any],
        logits: torch.Tensor,
        is_full_logits: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the per-token log probabilities and advantages

        Args:
            minibatch: a dictionary containing the input_ids and logprob_masks
            logits: the logits of the model
            is_full_logits: whether the logits are full logits or have been index-selected for memory efficiency

        Returns:
            logps: the per-token log probabilities
            logprob_masks: the logprob_masks
            metrics: a dict of collected metrics, e.g. entropy
        """
        assert "input_ids" in minibatch, "input_ids is required for computing logprobs"
        assert (
            "logprob_masks" in minibatch
        ), "logprob_masks is required for computing logprobs"
        return logprobs_computing(
            minibatch["input_ids"],
            minibatch["logprob_masks"],
            logits.to(dtype=str2torch_dtype(self.config.train.logprob_dtype)),
            is_full_logits=is_full_logits,
            label_packing_mask=minibatch.get("label_packing_mask", None),
            input_packing_mask=minibatch.get("input_packing_mask", None),
            **kwargs,
        )

    @cached_property
    def map_w_from_policy_to_rollout(self):
        """
        Generate a mapping from local parameters into shape/layout that rollout requires.
        The mapping is created by iterating through the named parameters of both models
        and replacing certain substrings in the parameter names.
        """
        name_to_transform = {}
        assert len(self.model.weight_sync_transforms) > 0, "No sorted parameters found."
        for name, transform_block in self.model.weight_sync_transforms:
            assert isinstance(transform_block, Callable) or isinstance(
                transform_block, torch.Tensor
            )
            name_to_transform[name] = transform_block
        return name_to_transform

    @cached_property
    def weight_mapper(self):
        return self.model.weight_mapper

    def weight_resume(self):
        # If KL-divergence is enabled, hf model should always be loaded from checkpoint
        model_loaded = False
        if self.config.train.train_policy.kl_beta != 0.0:
            self.model_load_from_hf()
            model_loaded = True
            # Clone the state dict of hf model so that it can be used for KL-divergence calculation
            self.reference_state_dict = {}
            state_dict = self.model.state_dict()
            for key, value in state_dict.items():
                self.reference_state_dict[key] = value.detach().cpu()

        if self.config.train.resume:
            try:
                # Need to reload again from checkpoint to make sure the model is in the correct state
                self.model_resume_from_checkpoint()
                model_loaded = True
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    logger.info(
                        f"[Policy] Fail to resume from {self.config.train.resume} because the checkpoint file does not exist, trying to load from HuggingFace..."
                    )
                else:
                    logger.error(
                        f"[Policy] Cannot resume from {self.config.train.resume} {e}. Trying to load from HuggingFace..."
                    )
                if not model_loaded:
                    self.model_load_from_hf()
                    model_loaded = True
        elif not model_loaded:
            logger.info("[Policy] Resume not set. Trying to load from HuggingFace...")
            self.model_load_from_hf()
            model_loaded = True

        assert model_loaded, "Model weight must be populated before training starts."
        logger.info("[Policy] Model loaded from checkpoint.")
        assert (
            self.map_w_from_policy_to_rollout is not None
        ), "No parameters to sync found."

        self.model.train()

        return False

    def build_lr_schedulers(self):
        return common_build_lr_schedulers(self.optimizers, self.config, 1e6)

    def update_lr_schedulers(self, total_steps: Optional[int] = None):
        if not self.lr_schedulers_updated:
            assert (
                total_steps is not None and total_steps > 0
            ), "Total steps must be set for lr scheduler"
            logger.info(
                f"[Policy] Building lr schedulers for total steps {total_steps}"
            )

            # TODO(jiaxinc): This is a tricky part:
            # Rebuild lr schedulers for the very first step because
            # only until the first step, we can know the exact total steps from the controller
            new_lr_schedulers = self.build_lr_schedulers()
            with torch.no_grad():
                # Note: we need to load the state dict of the old lr schedulers
                # in case it is resumed from a checkpoint,
                # otherwise, the lr scheduler will be reset to the initial value
                new_lr_schedulers.load_state_dict(self.lr_schedulers.state_dict())
            self.lr_schedulers = new_lr_schedulers
            self.lr_schedulers_updated = True

    def all_reduce_states(self, inter_policy_nccl: HighAvailabilitylNccl) -> float:
        """
        # Add nccl allreduce operations for all parameters and necessary states.
        """
        with torch.cuda.stream(self.train_stream):
            for model_part in self.model_parts:
                # Model part may use same physical mesh for different logical mesh,
                # which is not supported by DTensor operands like `torch.nn.utils.get_total_norm`
                # So we need to do allreduce for each model part
                if model_part is not None:
                    dist_util.gradient_reduce_across_dp_replicas_(
                        [p for p in model_part.parameters()], inter_policy_nccl
                    )
            """
            Compute the global grad norm on all parameters and then apply
            gradient clipping using the global grad norm.
            """
            # Must pass empty list even if model_part is None,
            # GradNorm across pp stages will fail if some rank does not join the barrier
            all_params = [
                p
                for m in [model for model in self.model_parts if model is not None]
                for p in m.parameters()
            ]
            grad_norm = dist_util.gradient_norm_clipping(
                all_params,
                self.config.train.optm_grad_norm_clip,
                foreach=True,
                pp_mesh=self.parallel_dims.mesh["pp"]
                if self.parallel_dims.pp_enabled
                else None,
                return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
            )
            self.optimizers.step()
            self.optimizers.zero_grad()
        return grad_norm

    @property
    def pp_loss_fn(self):
        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """
            loss: the loss of shape `[n_tokens]`
            """
            return loss.mean()

        return fake_compute_loss
