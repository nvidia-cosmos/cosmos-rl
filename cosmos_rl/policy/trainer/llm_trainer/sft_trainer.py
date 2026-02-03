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
import torch
import numpy as np

import torch.distributed as dist

from functools import partial

from typing import Optional


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


def async_safe_ce(
    output: torch.Tensor,
    target: torch.LongTensor,
    ignore_index: int = -100,
    loss_scaling_factor: float = 1.0,
    output_packing_mask: Optional[torch.Tensor] = None,
    target_packing_mask: Optional[torch.Tensor] = None,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    **kwargs,
) -> torch.Tensor:
    if output_packing_mask is not None:
        output = (
            output[output_packing_mask].contiguous().view(-1, output.size(-1)).float()
        )
    else:
        output = output[:, :-1].contiguous().view(-1, output.size(-1)).float()

    if target_packing_mask is not None:
        target = target[target_packing_mask].contiguous().view(-1)
    else:
        target = target[:, 1:].contiguous().view(-1)

    if cp_group is not None and cp_group.size() > 1:
        # Fallback to unbalance loss
        loss = (
            torch.nn.functional.cross_entropy(
                output,
                target,
                ignore_index=ignore_index,
                reduction="mean",
            )
            * loss_scaling_factor
        )
        # In case of all labels are ignored, loss will be nan.
        loss = torch.nan_to_num(loss, nan=0.0)
        return loss
    else:
        loss = torch.nn.functional.cross_entropy(
            output,
            target,
            ignore_index=ignore_index,
            reduction="none",
        )

        # Compute all token numbers across dp-world
        n_valid_tokens = (target != ignore_index).sum()
        num_dp_workers = 1
        if dp_group is not None:
            torch.distributed.all_reduce(n_valid_tokens, group=dp_group)
            num_dp_workers = torch.distributed.get_world_size(group=dp_group)

        loss = (
            loss.sum()
            / (n_valid_tokens + 1e-8)
            * (num_dp_workers * loss_scaling_factor)
        )
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
            dp_group=dp_group
            if self.config.train.train_policy.balance_dp_token
            else None,
            cp_group=cp_group,
        )

    def step_training(
        self,
        global_batch,
        total_steps: int,
        train_step: int,
        save_freq: int,
        inter_policy_nccl: Optional[dist_util.HighAvailabilitylNccl] = None,
    ):
        pp_last_stage = False
        if self.lr_schedulers is None:
            assert (
                train_step == 0
            ), "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            self.lr_schedulers = build_lr_schedulers(
                self.optimizers, self.config, total_steps
            )

        acc_loss = torch.zeros(1, device=self.device)
        self.optimizers.zero_grad()
        global_batch_size = self.data_packer.batch_size(global_batch)
        # split global_batch into mini_batches
        mini_batch_begin_idxs = list(
            range(
                0,
                global_batch_size,
                self.config.train.train_policy.mini_batch,
            )
        )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for i in mini_batch_begin_idxs:
            fixed_length = (
                self.config.policy.model_max_length
                if self.parallel_dims.pp_enabled
                and not self.parallel_dims.pp_dynamic_shape
                else None
            )
            raw_batch = self.data_packer.slice_batch(
                global_batch, i, i + self.config.train.train_policy.mini_batch
            )
            if fixed_length is None:
                max_len = min(
                    self.config.policy.model_max_length,
                    self.data_packer.sft_compute_max_len(raw_batch),
                )
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
                    hasattr(self.model, "check_sequence_packing_compatible")
                    and not self.model.check_sequence_packing_compatible()
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
            self.model.train()
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            labels = batch.pop("label_ids")

            position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(**batch)

            batch["position_ids"] = position_ids
            padding_mask = batch.get("padding_mask", None)

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
            # For VLMs, we need to delay the slice of inputs for CP until after the embedding generation in the model forward.
            delay_cp_slice_inputs = getattr(self.model, "delay_cp_slice_inputs", False)
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

                batch["input_ids"] = input_ids
                batch["position_ids"] = position_ids
                if padding_mask is not None:
                    batch["padding_mask"] = padding_mask

            if self.parallel_dims.cp_enabled:
                # Slice for cp after embedding generation and sequence packing in the model forward later.
                batch["cp_mesh"] = self.parallel_dims.mesh["cp"]

            if self.parallel_dims.pp_enabled:
                pp_last_stage = (
                    self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
                )
                pp_first_stage = self.parallel_dims.pp_coord[0] == 0

                # Pipeline Parallel forward / backward inside step() call
                targets, losses = (labels, []) if pp_last_stage else (None, None)
                if pp_first_stage:
                    self.pp_scheduler.step(
                        **batch,
                        pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                        seq_len_multiple=self.seq_len_multiple,
                    )
                else:
                    # FWD + BWD if it is 1F1B-like scheduler
                    self.pp_scheduler.step(
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

                aux_loss = None
                with self.act_offloading_ctx_manager:
                    output = self.model(**batch)
                    if isinstance(output, torch.Tensor):
                        logits = output
                    else:
                        logits = output.logits
                        # Enumerate the output to involve any `loss` like output
                        for k, v in output.items():
                            if "loss" in k.lower() and isinstance(v, torch.Tensor):
                                aux_loss = v if aux_loss is None else aux_loss + v

                ce_loss = self.loss_fn(
                    logits,
                    labels,
                    output_packing_mask=batch.get("input_packing_mask", None),
                    target_packing_mask=batch.get("label_packing_mask", None),
                    loss_scaling_factor=1.0 / len(mini_batch_begin_idxs),
                )
                if aux_loss is not None:
                    loss = ce_loss + aux_loss
                else:
                    loss = ce_loss
                # # Hint FSDP to do all-reduce on the last backward pass
                # if hasattr(self.model, "set_is_last_backward"):
                #     print(f"set_is_last_backward: {i == mini_batch_begin_idxs[-1]}")
                #     self.model.set_is_last_backward(i == mini_batch_begin_idxs[-1])
                loss.backward()
            acc_loss += ce_loss.detach()

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
        self.lr_schedulers.step()

        self.model.step_hook()

        end_event.record()

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(acc_loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(acc_loss, self.parallel_dims.mesh["dp_cp"]),
            )
        else:
            global_avg_loss = global_max_loss = acc_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if util.is_master_rank(self.parallel_dims, self.global_rank):
                # Calculate last iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds

                report_data = {
                    "train/iteration_time": iter_time,
                    "train/loss_avg": global_avg_loss,
                    "train/loss_max": global_max_loss,
                    "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                    "train/grad_norm": grad_norm if grad_norm is not None else -1,
                }

                # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                # it will be inaccurate. Need a reduce for all the metrics.
                if self.config.logging.report_mfu:
                    mfu = util.compute_mfu(
                        model=self.model,
                        n_tokens=np.prod(input_ids.shape),
                        iter_time=iter_time,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v

                report_data["train/learning_rate"] = self.lr_schedulers.get_last_lr()[0]

        return report_data

    def step_validation(self, val_global_batch, train_step: int, total_steps: int):
        if not self.config.validation.enable:
            return

        self.model.eval()
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
            val_position_ids, _, val_pos_seq_dim = self.model.get_position_ids(
                **val_batch
            )

            val_batch["position_ids"] = val_position_ids
            val_padding_mask = val_batch.get("padding_mask", None)

            delay_cp_slice_inputs = getattr(self.model, "delay_cp_slice_inputs", False)
            if self.parallel_dims.cp_enabled and not delay_cp_slice_inputs:
                [val_inputs, val_position_ids, val_padding_mask] = (
                    slice_inputs_for_ulysses(
                        [val_inputs, val_position_ids, val_padding_mask],
                        self.parallel_dims.mesh["cp"],
                        seq_dims=[1, val_pos_seq_dim, 1],
                    )
                )

                val_batch["input_ids"] = val_inputs
                val_batch["position_ids"] = val_position_ids
                if val_padding_mask is not None:
                    val_batch["padding_mask"] = val_padding_mask

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
                    )

                if pp_last_stage:
                    val_loss = self.loss_fn(pp_out, val_labels)
                else:
                    val_loss = torch.tensor([-1.0], device=self.device)
            else:
                val_logits = self.model(**val_batch)

                val_loss = self.loss_fn(val_logits, val_labels)
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
        ):
            val_loss = (  # noqa: F841
                dist_util.dist_mean(val_loss, self.parallel_dims.mesh["dp"])
            ) * self.parallel_dims.mesh["dp"].size()
        else:
            val_loss = val_loss.item()  # noqa: F841

        return val_loss * val_inputs.size(0)

    def checkpointing(
        self,
        total_steps: int,
        train_step: int,
        save_freq: int,
        is_last_step: bool = False,
        pp_last_stage: bool = False,
        val_score: Optional[float] = None,
        do_save: bool = False,
        **kwargs,
    ):
        if (
            is_last_step or do_save or (train_step % save_freq == 0 and train_step > 0)
        ) and self.parallel_dims.dp_replicate_coord[0] == 0:
            # save safetensors
            # TODO(dinghaoy): support export safetensors asynchronously.
            if is_last_step or (
                self.config.train.ckpt.export_safetensors
                and self.config.train.ckpt.enable_checkpoint
            ):
                logger.info(
                    f"Saving huggingface checkpoint at step {train_step} to {self.config.train.output_dir}..."
                )
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=os.path.join(
                        "safetensors",
                        f"step_{train_step}",
                    ),
                    trainable_only=False,
                    dtype=util.str2torch_dtype(self.config.train.param_dtype),
                )
            # save checkpoint
            if self.config.train.ckpt.enable_checkpoint:
                logger.info(f"Saving cosmos checkpoint at step {train_step}...")
                self.ckpt_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizers,
                    scheduler=self.lr_schedulers,
                    step=train_step,
                    total_steps=total_steps,
                    **kwargs,
                )
                self.ckpt_manager.save_check(
                    step=train_step,
                    val_score=val_score,
                    pp_enabled=self.parallel_dims.pp_enabled,
                    pp_last_stage=pp_last_stage,
                    pp_master_rank=self.parallel_dims.world_size
                    - self.parallel_dims.world_size / self.parallel_dims.pp,
                )

    def load_model(self):
        """Load model weights from checkpoint if available."""
        ckpt_total_steps = 0
        train_step = 0
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
                    self.model.load_hf_weights(
                        self.config.policy.model_name_or_path,
                        self.parallel_dims,
                        self.device,
                        revision=self.config.policy.model_revision,
                    )
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
                    and ckpt_total_steps is not None
                ):
                    # Initialize lr_schedulers on non-zero dp_replicate ranks when resuming training
                    # only when ckpt_total_steps > 0, means a checkpoint is loaded
                    self.lr_schedulers = build_lr_schedulers(
                        self.optimizers, self.config, ckpt_total_steps
                    )
                if ckpt_total_steps is not None:
                    assert (
                        self.lr_schedulers is not None
                    ), "lr_schedulers should not be None after broadcasting when resuming training with data parallel replication."

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

        self.model.train()
        return ckpt_total_steps, train_step

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
