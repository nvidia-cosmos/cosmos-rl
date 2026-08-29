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
import collections
import statistics
import math
import torch
import numpy as np
import torch.distributed as dist
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple
from cosmos_rl.utils.parallelism import (
    ParallelDims,
)
from cosmos_rl.policy.config import (
    Config as CosmosConfig,
)
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from cosmos_rl.utils.logging import logger

import cosmos_rl.utils.util as util
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.dispatcher.data.packer import BaseDataPacker

from cosmos_rl.utils.ulysses import (
    slice_inputs_for_ulysses,
)
from cosmos_rl.utils.sequence_packing import (
    pack_sequences_info_collect,
    pack_sequences_for_masks,
    pack_sequences_for_labels,
)
from cosmos_rl.policy.trainer.llm_trainer.llm_trainer import LLMTrainer
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.kernel.loss import CrossEntropyLoss


_VISUAL_BATCH_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
)


def _contains_visual_inputs(batch: Dict[str, object]) -> bool:
    """Return whether a collated mini-batch contains visual model inputs."""
    for key in _VISUAL_BATCH_KEYS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            if value.numel() > 0:
                return True
        elif value is not None:
            try:
                if len(value) > 0:
                    return True
            except TypeError:
                return True
    return False


def _module_parameters(module) -> List[torch.nn.Parameter]:
    if module is None:
        return []
    return list(module.parameters())


def _vlm_component_parameter_groups(model) -> OrderedDict:
    """Return non-overlapping parameter groups for an HF-style VLM.

    Qwen's multimodal projector (``visual.merger``) is nested under the
    vision module, and a tied language head can share an embedding parameter.
    Claim those specific modules first so component counts and norms do not
    double-count either case.
    """
    if not getattr(model, "is_vlm", False):
        return OrderedDict()

    projector_parameters = _module_parameters(model.multi_modal_projector)
    lm_head_parameters = _module_parameters(model.lm_head)
    projector_ids = {id(parameter) for parameter in projector_parameters}
    lm_head_ids = {id(parameter) for parameter in lm_head_parameters}

    vision_parameters = [
        parameter
        for parameter in _module_parameters(model.vision_model)
        if id(parameter) not in projector_ids
    ]
    language_parameters = [
        parameter
        for parameter in _module_parameters(model.language_model)
        if id(parameter) not in lm_head_ids
    ]

    groups = OrderedDict(
        (
            ("language_model", language_parameters),
            ("vision_encoder", vision_parameters),
            ("vision_projector", projector_parameters),
            ("lm_head", lm_head_parameters),
        )
    )
    claimed_ids = {
        id(parameter) for parameters in groups.values() for parameter in parameters
    }
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in claimed_ids
    ]
    if other_parameters:
        groups["other"] = other_parameters
    return groups


def _vlm_component_gradient_metrics(model) -> Dict[str, object]:
    """Measure component gradients before clipping or optimizer updates."""
    metrics: Dict[str, object] = {}
    for component, parameters in _vlm_component_parameter_groups(model).items():
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        with_grad = [parameter for parameter in trainable if parameter.grad is not None]
        grad_norm = 0.0
        if with_grad:
            grad_norm_tensor = dist_util.gradient_norm_clipping(
                with_grad,
                max_norm=0.0,
                foreach=True,
                return_norm_only=True,
            )
            grad_norm = float(grad_norm_tensor.detach().item())

        prefix = f"model/components/{component}"
        metrics[f"{prefix}/total_parameters"] = sum(
            parameter.numel() for parameter in parameters
        )
        metrics[f"{prefix}/trainable_parameters"] = sum(
            parameter.numel() for parameter in trainable
        )
        metrics[f"{prefix}/frozen_parameters"] = (
            metrics[f"{prefix}/total_parameters"]
            - metrics[f"{prefix}/trainable_parameters"]
        )
        metrics[f"{prefix}/trainable_parameter_tensors"] = len(trainable)
        metrics[f"{prefix}/parameter_tensors_with_grad"] = len(with_grad)
        metrics[f"{prefix}/grad_norm"] = grad_norm
    return metrics


def _enforce_visual_gradient_contract(metrics: Dict[str, object]) -> None:
    """Reject trainable visual components that received no usable gradient."""
    checked_components = []
    for component in ("vision_encoder", "vision_projector"):
        prefix = f"model/components/{component}"
        trainable = int(metrics.get(f"{prefix}/trainable_parameters", 0))
        if trainable == 0:
            continue
        checked_components.append(component)
        tensors_with_grad = int(metrics.get(f"{prefix}/parameter_tensors_with_grad", 0))
        grad_norm = float(metrics.get(f"{prefix}/grad_norm", 0.0))
        if tensors_with_grad == 0 or not np.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError(
                "Visual-gradient contract failed before the first optimizer "
                f"update: trainable component {component!r} has "
                f"parameter_tensors_with_grad={tensors_with_grad} and "
                f"grad_norm={grad_norm}. Verify that the VLM collator emits a "
                "padding-aware attention_mask and that supervised text tokens "
                "can attend to visual tokens."
            )
    metrics["model/components/visual_gradient_contract"] = (
        "passed" if checked_components else "not_applicable_frozen"
    )


def _distributed_any(value: bool, device: torch.device) -> bool:
    """Return a decision shared by every rank participating in training."""
    if not dist.is_available() or not dist.is_initialized():
        return value
    decision = torch.tensor(int(value), device=device, dtype=torch.int32)
    dist.all_reduce(decision, op=dist.ReduceOp.MAX)
    return bool(decision.item())


def async_safe_ce(
    output: torch.Tensor,
    target: torch.LongTensor,
    ce_impl: torch.nn.Module,
    ignore_index: int = -100,
    loss_scaling_factor: float = 1.0,
    output_packing_mask: Optional[torch.Tensor] = None,
    target_packing_mask: Optional[torch.Tensor] = None,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    return_stats: bool = False,
    **kwargs,
) -> torch.Tensor:
    if output_packing_mask is not None:
        output = output[output_packing_mask].contiguous().view(-1, output.size(-1))
    else:
        output = output[:, :-1].contiguous().view(-1, output.size(-1))

    lin_weight = kwargs.get(
        "lin_weight", None
    )  # For fused cross entropy, we need to pass the weight of lm_head to the loss function

    if target_packing_mask is not None:
        target = target[target_packing_mask].contiguous().view(-1)
    else:
        target = target[:, 1:].contiguous().view(-1)
    if cp_group is not None and cp_group.size() > 1:
        # Fallback to unbalance loss
        loss = (
            ce_impl(
                output,
                target,
                ignore_index=ignore_index,
                reduction="mean",
                lin_weight=lin_weight,
            )
            * loss_scaling_factor
        )
        # In case of all labels are ignored, loss will be nan.
        loss = torch.nan_to_num(loss, nan=0.0)
        if return_stats:
            raw = ce_impl(
                output,
                target,
                ignore_index=ignore_index,
                reduction="none",
                lin_weight=lin_weight,
            )
            valid = target != ignore_index
            return loss, raw[valid].sum().detach(), valid.sum().detach()
        return loss
    else:
        loss = ce_impl(
            output,
            target,
            ignore_index=ignore_index,
            reduction="none",
            lin_weight=lin_weight,
        )

        # Compute all token numbers across dp-world
        valid_mask = target != ignore_index
        local_numerator = loss[valid_mask].sum()
        local_denominator = valid_mask.sum()
        n_valid_tokens = local_denominator.detach().clone()
        num_dp_workers = 1
        if dp_group is not None:
            torch.distributed.all_reduce(n_valid_tokens, group=dp_group)
            num_dp_workers = torch.distributed.get_world_size(group=dp_group)

        loss = (
            local_numerator
            / (n_valid_tokens + 1e-8)
            * (num_dp_workers * loss_scaling_factor)
        )
        if return_stats:
            return loss, local_numerator.detach(), local_denominator.detach()
        return loss


@TrainerRegistry.register(trainer_type="sft")
class SFTTrainer(LLMTrainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: torch.cuda.Stream,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
        **kwargs,
    ):
        super(SFTTrainer, self).__init__(
            config,
            parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            **kwargs,
        )

        if self.parallel_dims.dp_shard_enabled:
            dp_group = self.parallel_dims.mesh["dp_shard"].get_group()
        else:
            dp_group = None

        if self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.mesh["cp"].get_group()
        else:
            cp_group = None

        self.loss_fn = partial(
            async_safe_ce,
            ce_impl=CrossEntropyLoss(self.config),
            dp_group=dp_group
            if self.config.train.train_policy.balance_dp_token
            else None,
            cp_group=cp_group,
        )
        self.enable_dp_load_balancing = (
            self.config.train.train_policy.enable_dp_load_balancing
        )
        self._visual_gradient_contract_checked = False

    # --------- loss-spike rollback ---------
    # Baseline is the MEDIAN of recent healthy step losses, not an exponential
    # mean: a median ignores the rising edge of a divergence, whereas an EMA
    # absorbs it and quietly raises its own alarm threshold. The arming window is
    # deliberately short because observed spikes can land inside the first ~30
    # steps, and the consecutive cap stops us fighting an unrecoverable state
    # forever.
    _LOSS_SPIKE_WINDOW = 50
    _LOSS_SPIKE_MIN_OBSERVATIONS = 12
    _LOSS_SPIKE_MAX_CONSECUTIVE = 8
    # The pre-clip gradient norm separates a damaging step from a merely noisy
    # one far better than the loss does. Measured on a healthy run: the loss
    # tops out at 5.8x its median while a spike reaches 16.2x (1.7x apart),
    # whereas the gradient norm tops out at 6.9x its median while the spike
    # reaches 72.5x (10x apart). Both are compared against a rolling median.
    _SPIKE_GRAD_NORM_FACTOR = 10.0
    # The loss only blows up one step AFTER the update that caused it, and the
    # gradient norm ramps for a few steps before the peak, so restoring a single
    # step is not enough -- keep a ring and rewind past the whole ramp.
    _SPIKE_ROLLBACK_DEPTH = 4
    # Restoring the pre-spike state is not enough on its own: the same learning
    # rate and optimizer state then re-enter the same unstable region and spike
    # again (observed 3-5 rollbacks per run). Back the step size off on each
    # rollback and let it climb back to the scheduled value over ~35 healthy
    # steps, so the nominal schedule is unchanged in steady state.
    _LOSS_SPIKE_LR_BACKOFF = 0.5
    _LOSS_SPIKE_LR_RECOVERY = 1.02
    _LOSS_SPIKE_LR_MIN_SCALE = 0.1

    def _loss_spike_lr_scale(self) -> float:
        return float(getattr(self, "_loss_spike_lr_scale_value", 1.0))

    def _apply_loss_spike_lr_scale(self) -> None:
        """Scale the scheduled LR for the coming step.

        The scheduler rewrites ``group["lr"]`` from its base value every step, so
        applying the factor after ``lr_schedulers.step()`` scales exactly one step
        and never compounds.
        """
        scale = self._loss_spike_lr_scale()
        if scale >= 1.0:
            return
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * scale

    def _loss_spike_trainable_params(self) -> List[torch.Tensor]:
        return [
            p
            for model_part in self.model_parts
            if model_part is not None
            for p in model_part.parameters()
            if p.requires_grad
        ]

    def _capture_loss_spike_snapshot(self) -> None:
        """Clone trainable parameters and optimizer moments before an update.

        LoRA keeps this cheap: the trainable set is a few tens of millions of
        parameters, so the clone costs a few milliseconds against a step time
        measured in seconds.
        """
        with torch.no_grad():
            params = [p.detach().clone() for p in self._loss_spike_trainable_params()]
            optimizer_snapshot = []
            for optimizer in self.optimizers:
                entries = []
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        state = optimizer.state.get(p)
                        if not state:
                            entries.append(None)
                            continue
                        entries.append(
                            {
                                key: (
                                    value.detach().clone()
                                    if torch.is_tensor(value)
                                    else value
                                )
                                for key, value in state.items()
                            }
                        )
                optimizer_snapshot.append(entries)
            ring = getattr(self, "_loss_spike_ring", None)
            if ring is None:
                ring = collections.deque(maxlen=self._SPIKE_ROLLBACK_DEPTH)
                self._loss_spike_ring = ring
            ring.append((params, optimizer_snapshot))

    def _restore_loss_spike_snapshot(self) -> bool:
        ring = getattr(self, "_loss_spike_ring", None)
        if not ring:
            return False
        # Oldest retained state: far enough back to precede the ramp that led in.
        params, optimizer_snapshot = ring[0]
        self._loss_spike_rewound_steps = len(ring)
        with torch.no_grad():
            for param, saved in zip(self._loss_spike_trainable_params(), params):
                param.copy_(saved)
            for optimizer, entries in zip(self.optimizers, optimizer_snapshot):
                index = 0
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        saved_state = entries[index] if index < len(entries) else None
                        index += 1
                        if saved_state is None:
                            optimizer.state.pop(p, None)
                            continue
                        state = optimizer.state.setdefault(p, {})
                        state.clear()
                        for key, value in saved_state.items():
                            state[key] = (
                                value.detach().clone()
                                if torch.is_tensor(value)
                                else value
                            )
        # The retained states all precede the spike; drop them so the next
        # rollback cannot rewind to a state we have already rejected.
        ring.clear()
        return True

    def _spike_median(self, attribute: str) -> Optional[float]:
        window = getattr(self, attribute, None)
        if not window or len(window) < self._LOSS_SPIKE_MIN_OBSERVATIONS:
            return None
        return statistics.median(window)

    def _loss_spike_baseline(self) -> Optional[float]:
        return self._spike_median("_loss_spike_history")

    def _grad_norm_baseline(self) -> Optional[float]:
        return self._spike_median("_grad_norm_history")

    def _loss_spike_should_rollback(
        self, step_loss: float, grad_norm_value: float, factor: float
    ) -> Tuple[bool, str]:
        if not math.isfinite(step_loss) or not math.isfinite(grad_norm_value):
            return True, "non-finite loss or gradient norm"
        grad_baseline = self._grad_norm_baseline()
        if grad_baseline is not None:
            limit = self._SPIKE_GRAD_NORM_FACTOR * max(grad_baseline, 1e-6)
            if grad_norm_value > limit:
                return True, (
                    f"gradient norm {grad_norm_value:.4f} exceeds "
                    f"{self._SPIKE_GRAD_NORM_FACTOR:.1f}x its rolling median "
                    f"{grad_baseline:.4f}"
                )
        loss_baseline = self._loss_spike_baseline()
        if loss_baseline is not None and step_loss > factor * max(loss_baseline, 1e-6):
            return True, (
                f"loss {step_loss:.4f} exceeds {factor:.1f}x its rolling median "
                f"{loss_baseline:.4f}"
            )
        return False, ""

    def _observe_loss_spike_baseline(
        self, step_loss: float, grad_norm_value: float
    ) -> None:
        for attribute, value in (
            ("_loss_spike_history", step_loss),
            ("_grad_norm_history", grad_norm_value),
        ):
            if not math.isfinite(value):
                continue
            window = getattr(self, attribute, None)
            if window is None:
                window = collections.deque(maxlen=self._LOSS_SPIKE_WINDOW)
                setattr(self, attribute, window)
            window.append(value)

    def step_training(
        self,
        global_batch,
        total_steps: int,
        train_step: int,
        save_freq: int,
        inter_policy_nccl: Optional[dist_util.HighAvailabilitylNccl] = None,
        data_arrival_event: Optional[torch.cuda.Event] = None,
    ):
        pp_last_stage = False
        if self.lr_schedulers is None:
            assert train_step == 0, (
                "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            )
            self.lr_schedulers = build_lr_schedulers(
                self.optimizers, self.config, total_steps
            )

        aux_loss_dict = OrderedDict()
        token_loss_numerator = torch.tensor(
            0.0, device=self.device, dtype=torch.float64
        )
        token_loss_denominator = torch.tensor(0, device=self.device, dtype=torch.long)
        visual_inputs_seen = False
        component_gradient_report: Dict[str, object] = {}

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        self.optimizers.zero_grad()

        # When DP load balancing is enabled, each element in global_batch is already a batch of samples,
        # so use mini_batch=1 to accumulate gradients correctly. Otherwise, use the configured mini_batch size.
        mini_batch = (
            self.config.train.train_policy.mini_batch
            if not self.enable_dp_load_balancing
            else 1
        )
        global_batch_size = self.data_packer.batch_size(global_batch)
        # split global_batch into mini_batches
        mini_batch_begin_idxs = list(
            range(
                0,
                global_batch_size,
                mini_batch,
            )
        )

        for i in mini_batch_begin_idxs:
            fixed_length = (
                self.config.policy.model_max_length
                if self.parallel_dims.pp_enabled
                and not self.parallel_dims.pp_dynamic_shape
                else None
            )
            raw_batch = (
                self.data_packer.slice_batch(
                    global_batch, i, i + self.config.train.train_policy.mini_batch
                )
                if not self.enable_dp_load_balancing
                else global_batch[i]
            )
            if fixed_length is None:
                max_len = min(
                    self.config.policy.model_max_length,
                    self.data_packer.sft_compute_max_len(raw_batch),
                )
                # logger.info(
                #     f"max_len: {max_len}, mini_batch_size: {len(raw_batch)}"
                # )
            else:
                max_len = fixed_length

            if self.seq_len_multiple > 1:
                max_len = (
                    (max_len + self.seq_len_multiple - 1)
                    // self.seq_len_multiple
                    * self.seq_len_multiple
                )

            packing_seq = self.config.train.sequence_packing
            if packing_seq:
                if self.parallel_dims.pp_enabled:
                    packing_seq = False
                    logger.debug(
                        "[Policy] Packing sequence is disabled due to incompatible dimensions."
                    )
                elif (
                    hasattr(self.forward_model, "check_sequence_packing_compatible")
                    and not self.forward_model.check_sequence_packing_compatible()
                ):
                    packing_seq = False
                    logger.debug(
                        "[Policy] Packing sequence is disabled due to unsupported model."
                    )

            batch = self.data_packer.sft_collate_fn(
                raw_batch,
                computed_max_len=max_len,
                ignore_label_id=-100,
            )
            visual_inputs_seen = visual_inputs_seen or _contains_visual_inputs(batch)
            self.set_model_train()
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            labels = batch.pop("label_ids")

            position_ids, input_ids, pos_seq_dim = self.forward_model.get_position_ids(
                **batch
            )

            batch["position_ids"] = position_ids
            padding_mask = batch.get("padding_mask", None)
            attention_mask = batch.get("attention_mask", None)

            if packing_seq:
                # Prepare for the sequence packing information.
                packed_args = pack_sequences_info_collect(
                    batch["input_ids"],
                    pad_token_id=self.data_packer.pad_token_id,
                    label_ids=labels,
                    ignore_label_id=-100,
                    seq_len_multiple=self.seq_len_multiple,
                )
                batch.update(packed_args)
                labels = pack_sequences_for_labels(labels, batch["valid_input_len"])
                packed_args = pack_sequences_for_masks(
                    batch["valid_input_len"], batch["valid_input_len"]
                )
                batch.update(packed_args)
            delay_cp_slice_inputs = getattr(
                self.forward_model, "delay_cp_slice_inputs", False
            )
            if (
                self.parallel_dims.cp_enabled
                and not packing_seq
                and not delay_cp_slice_inputs
            ):
                input_ids, position_ids, padding_mask, attention_mask = (
                    slice_inputs_for_ulysses(
                        [input_ids, position_ids, padding_mask, attention_mask],
                        self.parallel_dims.mesh["cp"],
                        seq_dims=[1, pos_seq_dim, 1, 1],
                    )
                )

                batch["input_ids"] = input_ids
                batch["position_ids"] = position_ids
                if padding_mask is not None:
                    batch["padding_mask"] = padding_mask
                if attention_mask is not None:
                    batch["attention_mask"] = attention_mask

            if self.parallel_dims.cp_enabled:
                # Slice for cp after embedding generation and sequence packing in the model forward later.
                batch["cp_mesh"] = self.parallel_dims.mesh["cp"]

            if self.parallel_dims.pp_enabled:
                pp_last_stage = (
                    self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
                )
                pp_first_stage = self.parallel_dims.pp_coord[0] == 0

                # PP case: pp_scheduler.step() handles forward/backward across all pipeline stages.
                # It internally manages all model_parts and coordinates communication between stages.
                # - First stage receives input_ids and starts the pipeline
                # - Last stage computes loss and initiates backward pass
                # - Intermediate stages only receive activations from previous stage
                targets, losses = (labels, []) if pp_last_stage else (None, None)

                pp_data_batch_args = []
                if pp_first_stage:
                    pp_data_batch_args.append(input_ids)
                self.pp_scheduler.step(
                    *pp_data_batch_args,
                    position_ids=batch["position_ids"],
                    target=targets,
                    losses=losses,
                    pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                    seq_len_multiple=self.seq_len_multiple,
                )
                ce_loss = (
                    torch.mean(torch.stack(losses)).to(self.device)
                    if pp_last_stage
                    else torch.tensor([-1.0], device=self.device)
                )
                aux_loss_dict["loss"] = (
                    ce_loss.detach()
                    if "loss" not in aux_loss_dict
                    else aux_loss_dict["loss"] + ce_loss.detach()
                )
            else:
                # This code is just for debugging purposes, where we can test whether the model can generate tokens correctly
                # last_token_ids = []
                # with torch.no_grad():
                #     N_NEW_TOKENS = 100
                #     for _ in range(N_NEW_TOKENS):
                #         if len(last_token_ids) > 0:
                #             batch["input_ids"] = torch.cat(
                #                 [batch["input_ids"], last_token_ids[-1]],
                #                 dim=-1,
                #             )
                #             position_ids, _, _ = (
                #                 self.model.get_position_ids(**batch)
                #             )
                #             batch["position_ids"] = position_ids

                #         logits = self.model(**batch)
                #         token_ids = torch.argmax(logits[:, -1:, :], dim=-1)
                #         last_token_ids.append(token_ids)
                #     if self.global_rank == 0:
                #         text = ''
                #         new_last_token_ids = torch.cat(last_token_ids, dim=-1).squeeze(0)
                #         logger.info(f'{new_last_token_ids=}')
                #         text = self.data_packer.tokenizer.decode(new_last_token_ids)
                #         logger.info(
                #             f"generated tokens at sample : {text}"
                #         )
                # return
                #########################################################################################

                loss = 0.0
                with self.act_offloading_ctx_manager:
                    output = self.forward_model(**batch)
                    logits = output.logits
                    # Enumerate the output to involve any `loss` like output
                    for k, v in output.items():
                        if "loss" in k.lower() and isinstance(v, torch.Tensor):
                            v = v / len(mini_batch_begin_idxs)
                            aux_loss_dict[k] = (
                                v.detach()
                                if k not in aux_loss_dict
                                else aux_loss_dict[k] + v.detach()
                            )
                            loss = loss + v
                kwargs = {
                    "output_packing_mask": batch.get("input_packing_mask", None),
                    "target_packing_mask": batch.get("label_packing_mask", None),
                    "loss_scaling_factor": 1.0 / len(mini_batch_begin_idxs),
                }

                if self.config.policy.enable_liger_fused_cross_entropy:
                    # In this case, `logits` in model output is not processed by lm_head. We have to pass weight
                    # of lm_head to the loss function to fuse the linear and cross entropy.
                    kwargs["lin_weight"] = self.model.lm_head.weight

                ce_loss, mini_numerator, mini_denominator = self.loss_fn(
                    logits,
                    labels,
                    return_stats=True,
                    **kwargs,
                )
                token_loss_numerator += mini_numerator.to(dtype=torch.float64)
                token_loss_denominator += mini_denominator.to(dtype=torch.long)
                aux_loss_dict["loss"] = (
                    ce_loss.detach()
                    if "loss" not in aux_loss_dict
                    else aux_loss_dict["loss"] + ce_loss.detach()
                )
                loss = loss + ce_loss
                loss.backward()

            loss_flat = torch.stack(list(aux_loss_dict.values()))
            loss_flat_keys = list(aux_loss_dict.keys())
        """
        Compute the global grad norm on all parameters and then apply
        gradient clipping using the global grad norm.
        """
        if inter_policy_nccl is not None:
            # Reduce gradients across all replicas for multiple replicas case
            for model_part in self.model_parts:
                # Model part may use same physical mesh for different logical mesh,
                # which is not supported by DTensor operands like `torch.nn.utils.get_total_norm`
                # So we need to do allreduce for each model part
                if model_part is not None:
                    dist_util.gradient_reduce_across_dp_replicas_(
                        [p for p in model_part.parameters()], inter_policy_nccl
                    )

        if not self._visual_gradient_contract_checked and getattr(
            self.forward_model, "is_vlm", False
        ):
            visual_inputs_seen = _distributed_any(visual_inputs_seen, self.device)
            require_visual_gradients = os.environ.get(
                "COSMOS_SFT_REQUIRE_VISUAL_GRADIENTS", "0"
            ).lower() in {"1", "true", "yes", "on"}
            if require_visual_gradients and not visual_inputs_seen:
                raise RuntimeError(
                    "Visual-gradient contract was required, but the first global "
                    "training batch contained no visual model inputs. Verify the "
                    "dataset adapter, media fields, and collator output."
                )
            if self.parallel_dims.pp_enabled:
                if require_visual_gradients:
                    raise RuntimeError(
                        "Visual-gradient contract cannot attest pipeline-parallel "
                        "component gradients. Set pipeline parallelism to 1 for "
                        "TAO Cosmos VLM training."
                    )
                component_gradient_report[
                    "model/components/visual_gradient_contract"
                ] = "not_checked_pipeline_parallel"
                logger.warning(
                    "The visual-gradient contract is not available with pipeline "
                    "parallelism; component ownership spans pipeline stages."
                )
            elif visual_inputs_seen:
                component_gradient_report = _vlm_component_gradient_metrics(
                    self.forward_model
                )
                _enforce_visual_gradient_contract(component_gradient_report)
                logger.info(
                    "Visual-gradient contract passed before the first optimizer "
                    f"update: {component_gradient_report}"
                )
            else:
                component_gradient_report[
                    "model/components/visual_gradient_contract"
                ] = "not_applicable_no_visual_inputs"
            self._visual_gradient_contract_checked = True

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

        rollback_factor = float(self.config.train.optm_loss_spike_rollback or 0.0)
        rolled_back = False
        step_loss = float("nan")
        if rollback_factor > 0.0:
            denominator = int(token_loss_denominator.item())
            if denominator > 0:
                step_loss = float(token_loss_numerator.item()) / denominator
            try:
                grad_norm_value = float(grad_norm)
            except (TypeError, ValueError):
                grad_norm_value = float("nan")
            rolled_back, spike_reason = self._loss_spike_should_rollback(
                step_loss, grad_norm_value, rollback_factor
            )

        if rolled_back:
            self._loss_spike_rollbacks = getattr(self, "_loss_spike_rollbacks", 0) + 1
            consecutive = getattr(self, "_loss_spike_consecutive", 0) + 1
            self._loss_spike_consecutive = consecutive
            restored = self._restore_loss_spike_snapshot()
            logger.warning(
                # ``train_step`` is pre-increment here while the worker's
                # "Step: N/M" line is post-increment, so report N to match it.
                f"[Policy] Training spike at step {train_step + 1}: {spike_reason}. "
                + (
                    f"Rewound {getattr(self, '_loss_spike_rewound_steps', 0)} step(s) "
                    f"of parameters and optimizer moments"
                    if restored
                    else "No snapshot available, update skipped"
                )
                + f" (rollbacks: {self._loss_spike_rollbacks}, consecutive: "
                f"{consecutive})."
            )
            self.optimizers.zero_grad()
            self._loss_spike_lr_scale_value = max(
                self._LOSS_SPIKE_LR_MIN_SCALE,
                self._loss_spike_lr_scale() * self._LOSS_SPIKE_LR_BACKOFF,
            )
            logger.warning(
                f"[Policy] Step-size backoff after rollback: scheduled LR will be "
                f"scaled by {self._loss_spike_lr_scale_value:.4f} and recover "
                f"toward 1.0 over subsequent healthy steps."
            )
            if consecutive >= self._LOSS_SPIKE_MAX_CONSECUTIVE:
                logger.warning(
                    f"[Policy] {consecutive} consecutive rollbacks at step "
                    f"{train_step + 1}; the run is not recovering from this state, so "
                    f"normal updates resume and the loss baseline is re-seeded "
                    f"rather than stalling training indefinitely."
                )
                self._loss_spike_consecutive = 0
                self._loss_spike_history = None
                self._grad_norm_history = None
        else:
            if rollback_factor > 0.0:
                self._loss_spike_consecutive = 0
                self._capture_loss_spike_snapshot()
                self._loss_spike_lr_scale_value = min(
                    1.0, self._loss_spike_lr_scale() * self._LOSS_SPIKE_LR_RECOVERY
                )
            self.optimizers.step()
        self.lr_schedulers.step()
        if rollback_factor > 0.0:
            self._apply_loss_spike_lr_scale()
        if rollback_factor > 0.0 and not rolled_back:
            self._observe_loss_spike_baseline(step_loss, grad_norm_value)

        if self.parallel_dims.pp_enabled:
            report_data = {}
            for model_part in self.model_parts:
                step_hook_report_data = model_part.step_hook(train_step)
                if step_hook_report_data is not None:
                    report_data.update(step_hook_report_data)
        else:
            step_hook_report_data = self.model.step_hook(train_step)
            report_data = (
                step_hook_report_data if step_hook_report_data is not None else {}
            )
        report_data.update(component_gradient_report)

        end_event.record()

        global_avg_loss = loss_flat.clone().to(self.device)
        global_max_loss = loss_flat.clone().to(self.device)
        torch.distributed.all_reduce(
            global_avg_loss,
            op=torch.distributed.ReduceOp.AVG,
            group=self.parallel_dims.mesh["loss_parallel"].get_group(),
        )
        torch.distributed.all_reduce(
            global_max_loss,
            op=torch.distributed.ReduceOp.MAX,
            group=self.parallel_dims.mesh["loss_parallel"].get_group(),
        )
        if self.parallel_dims.dp_replicate_enabled:
            torch.distributed.all_reduce(
                global_avg_loss,
                op=torch.distributed.ReduceOp.AVG,
                group=self.parallel_dims.mesh["dp_replicate"].get_group(),
            )
            torch.distributed.all_reduce(
                global_max_loss,
                op=torch.distributed.ReduceOp.MAX,
                group=self.parallel_dims.mesh["dp_replicate"].get_group(),
            )
        global_avg_loss = global_avg_loss.cpu()
        global_max_loss = global_max_loss.cpu()

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
        ):
            torch.distributed.all_reduce(
                token_loss_numerator,
                op=torch.distributed.ReduceOp.SUM,
                group=self.parallel_dims.mesh["dp"].get_group(),
            )
            torch.distributed.all_reduce(
                token_loss_denominator,
                op=torch.distributed.ReduceOp.SUM,
                group=self.parallel_dims.mesh["dp"].get_group(),
            )
        report_data["train/loss_numerator"] = float(token_loss_numerator.item())
        report_data["train/loss_denominator"] = int(token_loss_denominator.item())
        report_data["train/valid_label_count"] = int(token_loss_denominator.item())

        if self.config.logging.logger:
            assert end_event.query()
            fwd_bwd_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds
            batch_arrival_time = data_arrival_event.elapsed_time(start_event) / 1000.0
            if (
                self.parallel_dims.dp_replicate_enabled
                or self.parallel_dims.dp_shard_enabled
                or self.parallel_dims.cp_enabled
            ):
                time_metric_tensor_mean = torch.tensor(
                    [fwd_bwd_time, batch_arrival_time],
                    device=self.device,
                    dtype=torch.float32,
                )
                time_metric_tensor_max = time_metric_tensor_mean.clone()
                torch.distributed.all_reduce(
                    time_metric_tensor_mean,
                    op=torch.distributed.ReduceOp.AVG,
                    group=self.parallel_dims.mesh["dp_cp"].get_group(),
                )
                torch.distributed.all_reduce(
                    time_metric_tensor_max,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self.parallel_dims.mesh["dp_cp"].get_group(),
                )
                time_metric_tensor_mean_cpu = time_metric_tensor_mean.cpu()
                fwd_bwd_time_mean = time_metric_tensor_mean_cpu[0].item()
                batch_arrival_time_mean = time_metric_tensor_mean_cpu[1].item()
                # fwd_bwd_time_max = time_metric_tensor_max.cpu()[0].item()
                batch_arrival_time_max = time_metric_tensor_max.cpu()[1].item()
            else:
                # fwd_bwd_time_mean = fwd_bwd_time_max = fwd_bwd_time
                fwd_bwd_time_mean = fwd_bwd_time
                batch_arrival_time_mean = batch_arrival_time_max = batch_arrival_time

            if util.is_master_rank(self.parallel_dims, self.global_rank):
                loss_metrics = {
                    "train/iteration_time": fwd_bwd_time_mean,
                    "train/batch_arrival_time_mean": batch_arrival_time_mean,
                    "train/batch_arrival_time_max": batch_arrival_time_max,
                }
                learning_rates_metric = {
                    "optimizer/grad_norm": grad_norm if grad_norm is not None else -1,
                }
                for idx in range(len(self.model_parts)):
                    try:
                        learning_rates_metric[
                            f"optimizer/lr_{self.model_module_path[idx]}"
                        ] = self.lr_schedulers.get_last_lr(idx)[0]
                    except Exception:
                        # Maybe this model part is frozen, so no optimizer/scheduler for it, just skip.
                        # learning_rates_metric[f"optimizer/lr_{self.model_module_path[idx]}"] = -1.0
                        pass
                loss_metrics.update(learning_rates_metric)

                for idx, name in enumerate(loss_flat_keys):
                    loss_metrics[f"train/{name}_avg"] = global_avg_loss[idx]
                    loss_metrics[f"train/{name}_max"] = global_max_loss[idx]

                report_data.update(
                    loss_metrics,
                )

                # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                # it will be inaccurate. Need a reduce for all the metrics.
                if self.config.logging.report_mfu:
                    mfu = util.compute_mfu(
                        model=self.model,
                        n_tokens=np.prod(input_ids.shape),
                        iter_time=fwd_bwd_time_mean,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v
        return report_data

    def step_validation(self, val_global_batch, train_step: int, total_steps: int):
        if not self.config.validation.enable:
            return

        # ``Module.eval()`` recursively walks the complete model hierarchy.
        # Validation invokes this method once per batch, so repeating that
        # traversal is pure overhead after the first batch.  Training restores
        # train mode before its next forward, which makes this guard safe for
        # every later validation phase as well.
        if self.forward_model.training:
            self.set_model_eval()
        with torch.no_grad():
            fixed_length = (
                self.config.policy.model_max_length
                if self.parallel_dims.pp_enabled
                and not self.parallel_dims.pp_dynamic_shape
                else None
            )
            if fixed_length is None:
                max_len = min(
                    self.config.policy.model_max_length,
                    self.val_data_packer.sft_compute_max_len(val_global_batch),
                )
            else:
                max_len = fixed_length
            if self.seq_len_multiple > 1:
                max_len = (
                    (max_len + self.seq_len_multiple - 1)
                    // self.seq_len_multiple
                    * self.seq_len_multiple
                )

            val_batch = self.val_data_packer.sft_collate_fn(
                val_global_batch,
                computed_max_len=max_len,
                ignore_label_id=-100,
            )
            for k, v in val_batch.items():
                val_batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
            val_inputs = val_batch["input_ids"]
            val_labels = val_batch.pop("label_ids")
            val_position_ids, _, val_pos_seq_dim = self.forward_model.get_position_ids(
                **val_batch
            )

            val_batch["position_ids"] = val_position_ids
            val_padding_mask = val_batch.get("padding_mask", None)
            val_attention_mask = val_batch.get("attention_mask", None)

            delay_cp_slice_inputs = getattr(
                self.forward_model, "delay_cp_slice_inputs", False
            )
            if self.parallel_dims.cp_enabled and not delay_cp_slice_inputs:
                (
                    val_inputs,
                    val_position_ids,
                    val_padding_mask,
                    val_attention_mask,
                ) = slice_inputs_for_ulysses(
                    [
                        val_inputs,
                        val_position_ids,
                        val_padding_mask,
                        val_attention_mask,
                    ],
                    self.parallel_dims.mesh["cp"],
                    seq_dims=[1, val_pos_seq_dim, 1, 1],
                )

                val_batch["input_ids"] = val_inputs
                val_batch["position_ids"] = val_position_ids
                if val_padding_mask is not None:
                    val_batch["padding_mask"] = val_padding_mask
                if val_attention_mask is not None:
                    val_batch["attention_mask"] = val_attention_mask

            if self.parallel_dims.pp_enabled:
                pp_last_stage = (
                    self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
                )
                pp_first_stage = self.parallel_dims.pp_coord[0] == 0

                if pp_first_stage:
                    self.pp_scheduler_val.step(
                        **val_batch,
                        pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                        seq_len_multiple=self.seq_len_multiple,
                    )
                else:
                    pp_out = self.pp_scheduler_val.step(
                        position_ids=val_position_ids,
                        pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                        seq_len_multiple=self.seq_len_multiple,
                    ).logits

                if pp_last_stage:
                    val_loss, val_numerator, val_denominator = self.loss_fn(
                        pp_out, val_labels, return_stats=True
                    )
                else:
                    val_loss = torch.tensor([-1.0], device=self.device)
                    val_numerator = torch.tensor(
                        0.0, device=self.device, dtype=torch.float64
                    )
                    val_denominator = torch.tensor(
                        0, device=self.device, dtype=torch.long
                    )
            else:
                val_output = self.forward_model(**val_batch)
                val_logits = (
                    val_output.logits if hasattr(val_output, "logits") else val_output
                )

                val_loss, val_numerator, val_denominator = self.loss_fn(
                    val_logits, val_labels, return_stats=True
                )

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
        ):
            torch.distributed.all_reduce(
                val_numerator,
                op=torch.distributed.ReduceOp.SUM,
                group=self.parallel_dims.mesh["dp"].get_group(),
            )
            torch.distributed.all_reduce(
                val_denominator,
                op=torch.distributed.ReduceOp.SUM,
                group=self.parallel_dims.mesh["dp"].get_group(),
            )
        denominator = int(val_denominator.item())
        numerator = float(val_numerator.item())
        return {
            "loss_numerator": numerator,
            "loss_denominator": denominator,
            "avg_loss": numerator / max(denominator, 1),
        }

    def checkpointing(
        self,
        total_steps: int,
        train_step: int,
        save_freq: int,
        is_last_step: bool = False,
        pp_last_stage: bool = False,
        val_score: Optional[float] = None,
        steps_per_epoch: Optional[int] = None,
        do_save: bool = False,
        **kwargs,
    ):
        if (
            is_last_step or do_save or (train_step % save_freq == 0 and train_step > 0)
        ) and self.config.train.ckpt.enable_checkpoint:
            # When checkpointing is configured by epoch, use the completed epoch
            # consistently for checkpoint and safetensors directory names.
            completed_epoch = None
            if (
                self.config.train.ckpt.save_freq_in_epoch > 0
                and steps_per_epoch is not None
                and steps_per_epoch > 0
            ):
                completed_epoch = (train_step - 1) // steps_per_epoch + 1
                logger.debug(
                    f"[SFT] Epoch-based checkpoint: train_step={train_step}, steps_per_epoch={steps_per_epoch}, completed_epoch={completed_epoch}"
                )
            ckpt_identifier = (
                f"epoch_{completed_epoch}"
                if completed_epoch is not None
                else f"step_{train_step}"
            )

            if self.parallel_dims.dp_replicate_coord[0] == 0:
                # save safetensors
                if is_last_step or self.config.train.ckpt.export_safetensors:
                    logger.info(
                        f"Saving huggingface checkpoint {ckpt_identifier} at step {train_step} to {self.config.train.output_dir}..."
                    )

                    self.export_safetensors(
                        output_dir=self.config.train.output_dir,
                        rel_path=os.path.join(
                            "safetensors",
                            ckpt_identifier,
                        ),
                        trainable_only=False,
                        is_final=is_last_step,
                        dtype=util.str2torch_dtype(self.config.train.param_dtype),
                    )
                # save checkpoint
                logger.info(
                    f"Saving cosmos checkpoint {ckpt_identifier} at step {train_step}..."
                )

                if self.parallel_dims.pp_enabled:
                    pp_state_dict = {}
                    for i, mp in enumerate(self.model_parts):
                        prefix = self.model_module_path[i]
                        for k, v in mp.state_dict().items():
                            full_key = f"{prefix}.{k}" if prefix else k
                            pp_state_dict[full_key] = v
                    model_to_save = pp_state_dict
                else:
                    model_to_save = self.model
                self.ckpt_manager.save_checkpoint(
                    model=model_to_save,
                    optimizer=self.optimizers,
                    scheduler=self.lr_schedulers,
                    step=train_step,
                    total_steps=total_steps,
                    epoch=completed_epoch,
                    is_final=is_last_step,
                    **kwargs,
                )
                self.ckpt_manager.save_check(
                    step=train_step,
                    epoch=completed_epoch,
                    val_score=val_score,
                    pp_enabled=self.parallel_dims.pp_enabled,
                    pp_last_stage=pp_last_stage,
                    pp_master_rank=self.parallel_dims.world_size
                    - self.parallel_dims.world_size / self.parallel_dims.pp,
                )
            torch.distributed.barrier()
            return {
                "checkpoint/event": (
                    "complete"
                    if self.config.train.ckpt.save_mode == "sync"
                    else "submitted"
                ),
                "checkpoint/identifier": ckpt_identifier,
                "checkpoint/step": train_step,
                "checkpoint/epoch": completed_epoch,
                "checkpoint/output_dir": self.config.train.output_dir,
                "checkpoint/path": os.path.join(
                    self.config.train.output_dir,
                    "checkpoints",
                    ckpt_identifier,
                    "policy",
                ),
            }
        return None

    def load_model(self):
        """Load model weights from checkpoint if available."""
        ckpt_total_steps = 0
        train_step = 0
        ckpt_extra_vars = {}
        if (
            not self.parallel_dims.dp_replicate_enabled
        ) or self.parallel_dims.dp_replicate_coord[0] == 0:
            if self.config.train.resume:
                try:
                    # early init the lr_schedulers to avoid it is not initialized when loading the checkpoint
                    ckpt_extra_vars = self.model_resume_from_checkpoint()
                    ckpt_total_steps = ckpt_extra_vars.get("total_steps", 0)
                    train_step = ckpt_extra_vars.get("step", 0)
                except Exception as e:
                    logger.error(
                        f"Cannot resume due to error: {e}. Trying to load from HuggingFace..."
                    )
                    self.lr_schedulers = None
                    self.build_optimizers()
                    self.model_load_from_hf()
            else:
                self.model_load_from_hf()

        if self.parallel_dims.dp_replicate_enabled:
            if self.config.train.resume:
                ckpt_total_steps = dist_util.broadcast_object_cpu(
                    ckpt_total_steps,
                    group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                    group_src=0,
                )
                train_step = dist_util.broadcast_object_cpu(
                    train_step,
                    group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                    group_src=0,
                )

                if (
                    self.parallel_dims.dp_replicate_coord[0] != 0
                    and ckpt_total_steps > 0
                ):
                    # Initialize lr_schedulers on non-zero dp_replicate ranks when resuming training
                    # only when ckpt_total_steps > 0, means a checkpoint is loaded
                    self.lr_schedulers = build_lr_schedulers(
                        self.optimizers, self.config, ckpt_total_steps
                    )
                if ckpt_total_steps > 0:
                    assert self.lr_schedulers is not None, (
                        "lr_schedulers should not be None after broadcasting when resuming training with data parallel replication."
                    )

            send_recv_hook = partial(
                dist.broadcast,
                group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                group_src=0,
            )
            len_params = self.sync_all_states(
                is_send=self.parallel_dims.dp_replicate_coord[0] == 0,
                send_hook=send_recv_hook,
                recv_hook=send_recv_hook,
            )
            logger.info(
                f"Synchronized {len_params} parameters across data parallel replicas."
            )

        self.set_model_train()
        return ckpt_total_steps, train_step, ckpt_extra_vars

    @property
    def pp_loss_fn(self):
        # calculate the loss scaling factor
        mini_batch_size = max(self.config.train.train_policy.mini_batch or 1, 1)
        mini_batch_size = min(
            mini_batch_size, self.config.train.train_batch_per_replica
        )
        loss_scaling_factor = (
            mini_batch_size / self.config.train.train_batch_per_replica
        )
        if self.config.train.train_policy.enable_dp_load_balancing:
            loss_scaling_factor = (
                1.0
                / self.config.train.train_policy.load_balanced_batches_per_optimizer_step
            )
        if self.parallel_dims.dp_shard_enabled:
            dp_group = self.parallel_dims.mesh["dp_shard"].get_group()
        else:
            dp_group = None

        if self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.mesh["cp"].get_group()
        else:
            cp_group = None

        return torch.compile(
            partial(
                async_safe_ce,
                ce_impl=CrossEntropyLoss(),
                loss_scaling_factor=loss_scaling_factor,
                dp_group=dp_group
                if self.config.train.train_policy.balance_dp_token
                else None,
                cp_group=cp_group,
            )
        )

    def build_lr_schedulers(self):
        # just for instantiating this class.
        pass
