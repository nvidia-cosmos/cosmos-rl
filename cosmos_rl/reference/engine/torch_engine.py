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

import torch
import types
from functools import partial
import inspect
from typing import Dict, Any, Callable, List, Tuple

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.trainer.llm_trainer.llm_trainer import LLMTrainer
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import (
    setup_tokenizer,
)
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.utils.sequence_packing import (
    pack_sequences_for_inputs,
    pack_sequences_for_logprobs,
    pack_sequences_info_collect,
    pack_sequences_for_masks,
)
from cosmos_rl.utils.ulysses import (
    slice_inputs_for_ulysses,
)
from cosmos_rl.utils.util import str2torch_dtype
from cosmos_rl.utils.util import compute_logprobs as logprobs_computing


# TODO: (lms) May be it's better to register this func as a hook to the last stage model.
# That way is more clean. I think it's feasible but need to be compatible with torch Pipelie schedule.
def _swizzle_pp_grpo_forward(
    trainer: "TorchEngine",
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

    # User defined input
    user_input = kwargs.copy()

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

    if config.train.train_policy.temperature > 1e-6:
        raw_logits = raw_logits / config.train.train_policy.temperature
    # [n_tokens, n_vocab]
    current_per_token_logprobs, cu_seqlens, metrics = trainer.compute_logprobs(
        minibatch={
            **user_input,
        },
        logits=raw_logits,
        is_full_logits=True if raw_logits.ndim == 3 else False,
    )
    assert (
        len(current_per_token_logprobs) == len(cu_seqlens) - 1
    ), f"current_per_token_logprobs.shape: {current_per_token_logprobs.shape}, cu_seqlens.shape: {cu_seqlens.shape}"
    current_per_token_logprobs = current_per_token_logprobs.cpu()
    cu_seqlens = cu_seqlens.cpu()
    for i in range(len(current_per_token_logprobs)):
        current_per_token_logprobs[i] = current_per_token_logprobs[i][
            : cu_seqlens[i + 1] - cu_seqlens[i]
        ]
    return current_per_token_logprobs


class TorchEngine(LLMTrainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        **kwargs,
    ):
        super(TorchEngine, self).__init__(
            config,
            parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=None,
            **kwargs,
        )

        if parallel_dims.dp_replicate > 1:
            raise ValueError(
                f"DP replicate size {parallel_dims.dp_replicate} is not supported for GRPO"
                "Please use elastic scaling feature instead."
            )
        # For iteration control
        self.batch_size = self.config.distillation.batch_size_per_replica
        self.max_length = self.config.distillation.model_max_length
        self.tokenizer = setup_tokenizer(self.config.distillation.model_name_or_path)

    def step_training(self):
        pass

    def build_lr_schedulers(self):
        pass

    def step_forward(
        self,
        rollouts: List[Rollout],
        inter_policy_nccl: HighAvailabilitylNccl = None,
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

        # For single-turn rollout, we use the prompt, for multi-turn rollout, we use the completed conversation
        samples = [rollout.prompt for rollout in rollouts]
        assert all(
            rollout.prompt is not None for rollout in rollouts
        ), "All rollouts should have a valid prompt"
        assert all(
            rollout.completion_token_ids is not None
            and len(rollout.completion_token_ids) > 0
            for rollout in rollouts
        ), "All rollouts should have a valid completion token ids"

        completions_list = [rollout.completion_token_ids for rollout in rollouts]
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

        batch_size = len(rollouts)
        mini_batch_size = batch_size
        # Validate the PP parallelism configuration
        if self.parallel_dims.pp_enabled:
            n_microbatches = (
                batch_size // self.config.policy.parallelism.pp_micro_batch_size
            )
            assert (
                n_microbatches % self.parallel_dims.pp == 0
            ), f"n_microbatches {n_microbatches} should be divided evenly by pp size of {self.parallel_dims.pp}"

        data = []
        with torch.set_grad_enabled(False):
            with torch.cuda.stream(self.train_stream):
                # TODO(jiaxin): support variable length in PP
                computed_max_len = (
                    self.config.policy.model_max_length
                    if self.parallel_dims.pp_enabled
                    else self.data_packer.policy_compute_max_len(processed_samples)
                )
                computed_max_len = (
                    (computed_max_len + self.seq_len_multiple - 1)
                    // self.seq_len_multiple
                    * self.seq_len_multiple
                )
                user_mini_batch: Dict[str, Any] = self.data_packer.policy_collate_fn(
                    processed_samples,
                    computed_max_len=computed_max_len,
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
                    user_mini_batch["interested_tokens"] = user_mini_batch[
                        "logprob_masks"
                    ]

                # Move all tensor to device
                for k in user_mini_batch.keys():
                    v = user_mini_batch[k]
                    if isinstance(v, torch.Tensor) and v.device != self.device:
                        user_mini_batch[k] = v.to(self.device)

                # input_ids are different across ranks in dp_shard_cp
                position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(
                    **user_mini_batch
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
                        advantages=None,
                    )
                    user_mini_batch.update(packed_args)
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
                    [input_ids, position_ids, padding_mask] = slice_inputs_for_ulysses(
                        [input_ids, position_ids, padding_mask],
                        self.parallel_dims.mesh["cp"],
                        seq_dims=[1, pos_seq_dim, 1],
                    )
                    user_mini_batch["position_ids"] = position_ids
                    user_mini_batch["input_ids"] = input_ids
                    if padding_mask is not None:
                        user_mini_batch["padding_mask"] = padding_mask
                if self.parallel_dims.cp_enabled:
                    # Slice for cp after embedding generation and sequence packing in the model forward later.
                    user_mini_batch["cp_mesh"] = self.parallel_dims.mesh["cp"]

                if self.parallel_dims.pp_enabled:
                    # [mini_batch_size, 1]: indicating the index of mini-batch
                    micro_batch_ids_list = []
                    for i in range(mini_batch_size):
                        micro_batch_ids_list.append(
                            [i // self.config.policy.parallelism.pp_micro_batch_size]
                        )
                    micro_batch_ids_cpu = torch.Tensor(micro_batch_ids_list).int()
                    pp_first_stage = self.parallel_dims.pp_coord[0] == 0
                    # Pipeline Parallel forward / backward inside step() call
                    losses = [] if pp_last_stage else None
                    if pp_last_stage:
                        # Inject the `mini-batch` and `micro-batch` ids to the input so that the last stage can know which microbatch it is processing
                        user_mini_batch["micro_batch_ids"] = micro_batch_ids_cpu
                    if pp_first_stage or pp_last_stage:
                        # First/Last stage: pass all inputs
                        kwargs = {}
                        if self.parallel_dims.cp_enabled:
                            # This is for recover these two tensors after ulysses
                            kwargs["input_ids_before_cp"] = input_ids_before_cp
                            kwargs["position_ids_before_cp"] = position_ids_before_cp

                        self.pp_scheduler.step(
                            **user_mini_batch,
                            advantages=None,
                            losses=losses,
                            target=torch.empty(
                                [mini_batch_size, 1], device=self.device
                            ),
                            **kwargs,
                        )
                    else:
                        # Middle stages: forward data from previous stage
                        self.pp_scheduler.step(position_ids=position_ids)
                    assert False, "Not implemented"
                else:
                    with self.act_offloading_ctx_manager:
                        raw_logits = self.model(**user_mini_batch)

                    if self.parallel_dims.cp_enabled:
                        # reset the position ids and input ids
                        user_mini_batch["position_ids"] = position_ids_before_cp
                        user_mini_batch["input_ids"] = input_ids_before_cp
                        if padding_mask_before_cp is not None:
                            user_mini_batch["padding_mask"] = padding_mask_before_cp

                    if self.config.train.train_policy.temperature > 1e-6:
                        raw_logits = (
                            raw_logits / self.config.train.train_policy.temperature
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
                        user_mini_batch["input_ids"] = packed_args["inputs"]

                    (
                        current_per_token_logprobs,
                        cu_seqlens,
                        metrics,
                    ) = self.compute_logprobs(
                        user_mini_batch,
                        logits=raw_logits,
                        is_full_logits=True if raw_logits.ndim == 3 else False,
                    )
                    data = []
                    if self.parallel_dims.mesh["dp"].rank == 0:
                        assert (
                            len(current_per_token_logprobs) == len(cu_seqlens) - 1
                        ), f"current_per_token_logprobs.shape: {current_per_token_logprobs.shape}, cu_seqlens.shape: {cu_seqlens.shape}"
                        current_per_token_logprobs = current_per_token_logprobs.cpu()
                        cu_seqlens = cu_seqlens.cpu()
                        for i in range(len(current_per_token_logprobs)):
                            current_per_token_logprobs[i] = current_per_token_logprobs[
                                i
                            ][: cu_seqlens[i + 1] - cu_seqlens[i]]
                            data.append(
                                {
                                    "logprob": current_per_token_logprobs[i].tolist(),
                                    "uuid": rollouts[i].uuid,
                                }
                            )
        return data

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

    @property
    def pp_loss_fn(self):
        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """
            loss: the loss of shape `[n_tokens]`
            """
            pass

        return fake_compute_loss
