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

from cosmos_rl.policy.trainer import Trainer
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.optm import build_lr_schedulers, LRSchedulersContainer
from cosmos_rl.utils.logging import logger
import torch
import torch.distributed as dist
import numpy as np
import cosmos_rl.utils.util as util
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.utils import constant, api_suffix
from cosmos_rl.utils.network_util import make_request_with_retry
import os
import time
import atexit
import threading
import requests
from typing import Optional, Dict, Any, Tuple, List
from cosmos_rl.utils.ulysses import slice_inputs_for_ulysses
from functools import partial
from cosmos_rl.utils.distributed import (
    HighAvailabilitylNccl,
    wrap_to_cuda_tensor,
    extract_from_cuda_tensor,
)
from cosmos_rl.dispatcher.command import Command, BuildMeshCommand


def async_safe_ce(
    output: torch.Tensor,
    target: torch.LongTensor,
    ignore_index: int = -100,
    loss_scaling_factor: float = 1.0,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    target = target[:, 1:].contiguous().view(-1)
    output = output[:, :-1].contiguous().view(-1, output.size(-1)).float()

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


class SFTTrainer(Trainer):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super(SFTTrainer, self).__init__(config, parallel_dims)

        # Enlarge the compile cache size for validation
        if config.train.compile and config.train.enable_validation:
            torch._dynamo.config.cache_size_limit = 64

        self.dp_rank, self.dp_world_size = 0, 1
        if parallel_dims.dp_enabled:
            self.dp_rank = parallel_dims.mesh["dp"].get_local_rank()
            self.dp_world_size = parallel_dims.mesh["dp"].size()

        # Init redis controller
        self.init_redis()

        # For mesh build
        self.inter_policy_nccl = HighAvailabilitylNccl(
            replica_name=self.replica_name,
            global_rank=self.global_rank,
            controller_hosts=self.remote_hosts,
        )
        self.kv_store = dist_util.DistKVStore(
            group=dist.distributed_c10d._get_default_group(),
            master_rank=0,
            shutdown_event=self.shutdown_signal,
        )
        self.fetch_command_thread = threading.Thread(
            target=self.fetch_command_worker,
            daemon=True,
            name="fetch_command_thread",
        )
        self.fetch_command_thread.start()

        self.lr_schedulers: LRSchedulersContainer = None
        self.start_epoch = 0
        # Load model
        train_step = 0
        if config.train.resume:
            try:
                # early init the lr_schedulers to avoid it is not initialized when loading the checkpoint
                ckpt_extra_vars, self.lr_schedulers = self.ckpt_manager.load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizers,
                    scheduler=partial(
                        build_lr_schedulers, self.optimizers, self.config
                    ),
                )
                # ckpt_total_steps = ckpt_extra_vars.get("total_steps", 0)
                train_step = ckpt_extra_vars.get("step", 0)
            except Exception as e:
                logger.error(
                    f"[SFTTrainer] Cannot resume due to error: {e}. Trying to load from HuggingFace..."
                )
                self.model.load_hf_weights(
                    config.policy.model_name_or_path,
                    parallel_dims,
                    self.device,
                    revision=config.policy.model_revision,
                )
        else:
            self.model.load_hf_weights(
                config.policy.model_name_or_path,
                parallel_dims,
                self.device,
                revision=config.policy.model_revision,
            )

        # Prepare dataset
        # build dataset at controller side, we don't need to prepare dataset here

        assert (
            self.tokenizer.pad_token_id is not None
        ), "Tokenizer must have a pad token id"

        if self.lr_schedulers is None:
            assert (
                train_step == 0
            ), "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            # This is a fake lr_schedulers, when fetch data, we update the lr_schedulers to real one.
            self.lr_schedulers = build_lr_schedulers(self.optimizers, self.config, 1e6)
            # use this to control the lr_scheduler update
            self.last_total_steps = -1

        if self.parallel_dims.dp_enabled:
            dp_group = self.parallel_dims.mesh["dp"].get_group()
        else:
            dp_group = None

        if self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.mesh["cp"].get_group()
        else:
            cp_group = None

        self.loss_fn = partial(
            async_safe_ce,
            dp_group=dp_group,
            cp_group=cp_group,
        )

        atexit.register(self.handle_shutdown)

    def handle_shutdown(self):
        """
        Handle shutdown of the trainer.
        """
        self.inter_policy_nccl.shutdown()
        self.shutdown_signal.set()

        if self.fetch_command_thread is not None:
            self.fetch_command_thread.join()
            self.fetch_command_thread = None

        # Another notice is that make sure the background threads detect the shutdown event in less than 15 seconds
        # Otherwise, the main thread may exit before the background threads detect the shutdown event
        time.sleep(15)

    def fetch_command_worker(self):
        """
        Fetch command from the controller.
        For SFT, we only need to process buildmesh command.
        """
        while not self.shutdown_signal.is_set():
            if self.global_rank == 0:
                commands = []
                try:
                    commands = self.redis_controller.subscribe_command(
                        self.replica_name
                    )
                except Exception as e:
                    logger.debug(
                        f"[SFTTrainer] Failed to get commands : {e} at replica {self.replica_name}, wait for next round"
                    )
                for x in commands:
                    command = Command.depack(x)
                    if isinstance(command, BuildMeshCommand):
                        """ directly push the buildmesh command to the nccl comm, will not block main thread """
                        logger.info(
                            "[SFTTrainer] Broadcast buildmesh command to all ranks"
                        )
                        # broadcast the buildmesh command to all ranks
                        cmd = self.kv_store.broadcast_command(command, src=0)
                        self.is_master_replica = (
                            cmd.replica_name_to_rank[self.replica_name] == 0
                        )
                        self.inter_policy_nccl.push_cmd(cmd)
                    else:
                        logger.debug(
                            f"[SFTTrainer] Fetch command drop command: {type(command)}"
                        )
            else:
                try:
                    bmcmd = self.kv_store.broadcast_command(None, src=0)
                    assert isinstance(
                        bmcmd, BuildMeshCommand
                    ), "Only buildmesh command is supported"
                    self.is_master_replica = (
                        bmcmd.replica_name_to_rank[self.replica_name] == 0
                    )
                    self.inter_policy_nccl.push_cmd(bmcmd)
                except Exception as e:
                    raise RuntimeError(f"Failed to broadcast on slave workers: {e}")

    def fetch_data_from_controller(
        self, validation_step: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any], int, int]:
        """
        Fetch data from the controller.
        """
        if util.is_master_rank(self.parallel_dims, self.global_rank):
            try:
                response = make_request_with_retry(
                    partial(
                        requests.get,
                        params={
                            "n": self.config.train.train_batch_per_replica,
                            "validation_step": validation_step,
                            "replica_name": self.replica_name,
                        },
                    ),
                    self.get_alternative_urls(api_suffix.COSMOS_API_NEXT_PROMPT_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Policy] Failed in fetch data from controller after retries {e}."
                )
        else:
            response = None

        response = response.json()
        # broadcast the response to all ranks
        response = dist_util.broadcast_object_cpu(
            response, src=0, device=torch.device("cpu")
        )

        # unpack the response
        is_end = response["is_end"]
        train_step = response["train_step"]
        total_steps = response["total_steps"]
        # TODO(zjx): package the payload into a global batch
        global_batch = response["global_batch"]

        # check if should update lr_scheduler
        if self.last_total_steps != total_steps:
            # Rebuild lr schedulers for the very first step because
            # 1. only until the first step, we can know the exact total steps from the controller
            # 2. need update lr_scheduler when total_steps is changed
            self.lr_schedulers = build_lr_schedulers(
                self.optimizers, self.config, total_steps
            )
            self.last_total_steps = total_steps

        return is_end, global_batch, train_step, total_steps

    def post_fetch_data_from_controller(
        self, report_data: Dict[str, Any], train_step: int, total_steps: int
    ):
        """
        Post fetch data from the controller.
        """
        if util.is_master_rank(self.parallel_dims, self.global_rank):
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "replica_name": self.replica_name,
                            "step": total_steps,
                            "total_steps": total_steps,
                            "profile_finished": self.profiler.check_finished(),
                            "report_data": util.sanitize(report_data),
                        },
                    ),
                    self.get_alternative_urls(
                        api_suffix.COSMOS_API_POLICY_TRAIN_ACK_SUFFIX
                    ),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Policy] Failed in send train ack to controller after retries {e}."
                )

        logger.debug(f"[Policy] Train ack sent for global step {total_steps}.")

    def _sync_weights_between_replicas(self):
        """
        Sync weights between replicas. all replicas will get the same weights from replica 0.
        """
        is_send = self.inter_policy_nccl.get_replica_rank(self.replica_name) == 0
        src_replica = self.replica_name
        for (
            replica_name,
            replica_rank,
        ) in self.inter_policy_nccl.replica_name_to_rank.items():
            if replica_rank == 0:
                src_replica = replica_name
                break

        len_params = 0
        model_state_dict = [self.model.state_dict()]

        # 1. Sync all model states
        for state_to_sync in model_state_dict:
            for dest_name in sorted(state_to_sync.keys()):
                obj = state_to_sync[dest_name]
                assert isinstance(obj, torch.Tensor)
                local_view = wrap_to_cuda_tensor(
                    dest_name, obj, current_device=self.device
                )
                self.inter_policy_nccl.broadcast(local_view, src_replica=src_replica)
                if not is_send:
                    if isinstance(obj, torch.distributed.tensor.DTensor):
                        to_write = obj.to_local()
                    else:
                        to_write = obj

                    # Copy again for offloaded tensor since it is not inplace received
                    if not to_write.is_cuda:
                        to_write.copy_(local_view)
                len_params += 1

        # 2. Sync optimizer states
        optimizer_state = self.optimizers.state_dict()
        for dest_name in sorted(optimizer_state.keys()):
            obj = optimizer_state[dest_name]
            local_view = wrap_to_cuda_tensor(dest_name, obj, current_device=self.device)
            if local_view.data_ptr() is None:
                # skip the optimizer state if the data pointer is None
                continue
            self.inter_policy_nccl.broadcast(local_view, src_replica=src_replica)
            if not is_send:
                optimizer_state[dest_name] = extract_from_cuda_tensor(
                    dest_name, obj, local_view, current_device=self.device
                )
            len_params += 1
        if not is_send:
            self.optimizers.load_state_dict(optimizer_state)

        # 3. Sync lr_scheduler states
        lr_sheduler_state = self.lr_schedulers.state_dict()
        for dest_name in sorted(lr_sheduler_state.keys()):
            obj = lr_sheduler_state[dest_name]
            local_view = wrap_to_cuda_tensor(dest_name, obj, current_device=self.device)
            self.inter_policy_nccl.broadcast(local_view, src_replica=src_replica)
            if not is_send:
                lr_sheduler_state[dest_name] = extract_from_cuda_tensor(
                    dest_name, obj, local_view, current_device=self.device
                )
            len_params += 1
        if not is_send:
            self.lr_schedulers.load_state_dict(lr_sheduler_state)

        # 4. Sync rng_state
        rng_state = self.ckpt_manager.get_rng_state()
        for dest_name in sorted(rng_state.keys()):
            obj = rng_state[dest_name]
            local_view = wrap_to_cuda_tensor(dest_name, obj, current_device=self.device)
            self.inter_policy_nccl.broadcast(local_view, src_replica=src_replica)
            if not is_send:
                rng_state[dest_name] = extract_from_cuda_tensor(
                    dest_name, obj, local_view, current_device=self.device
                )
            len_params += 1
        if not is_send:
            self.ckpt_manager.set_rng_state(rng_state)

        return len_params

    def _allreduce_gradients(self):
        """
        Allreduce gradients accross replicas for all parameters and necessary states.
        """
        for model_part in self.model_parts:
            if model_part is not None:
                dist_util.gradient_reduce_across_dp_replicas_(
                    [p for p in model_part.parameters()], self.inter_policy_nccl
                )

    def _clipping_gradients(self) -> torch.Tensor:
        """
        Compute the global grad norm on all parameters and then apply
        gradient clipping using the global grad norm.
        """
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
        return grad_norm

    def _try_save_ckpt(self, current_step: int, total_steps: int, val_score: float):
        """
        Try to save the checkpoint if the current step is the save frequency.
        """
        # save checkpoint
        if (
            self.config.train.ckpt.enable_checkpoint
            and current_step % self.config.train.ckpt.save_freq == 0
            and current_step > 0
        ):
            # TODO(dinghaoy): support export safetensors asynchronously.
            if self.config.train.ckpt.export_safetensors:
                logger.info(
                    f"[SFTTrainer] Saving huggingface checkpoint at step {current_step} to {self.config.train.output_dir}..."
                )
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=os.path.join(
                        "safetensors",
                        f"step_{current_step}",
                    ),
                    trainable_only=False,
                    dtype=util.str2torch_dtype(self.config.train.param_dtype),
                )
            logger.info(
                f"[SFTTrainer] Saving cosmos checkpoint at step {current_step}..."
            )
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=current_step,
                total_steps=self.total_steps,
            )
            self.ckpt_manager.save_check(
                step=current_step,
                val_score=val_score,
            )

    def _try_process_after_final_step(self, current_step: int, total_steps: int):
        """
        Do some process after the final step.
        """
        # process the final step
        if self.config.train.enable_validation:
            val_score = self.validate_step(current_step, total_steps)
        if self.config.train.ckpt.export_safetensors:
            logger.info(
                f"[SFTTrainer] Saving final huggingface checkpoint to {self.config.train.output_dir}..."
            )
            self.export_safetensors(
                output_dir=self.config.train.output_dir,
                rel_path=os.path.join(
                    "safetensors",
                    f"step_{current_step}",
                ),
                trainable_only=False,
                is_final=True,
                dtype=util.str2torch_dtype(self.config.train.param_dtype),
            )
        if self.config.train.ckpt.enable_checkpoint:
            logger.info(
                f"[SFTTrainer] Training finished at step {current_step}/{total_steps}, saving final cosmos checkpoint..."
            )
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=current_step,
                total_steps=total_steps,
                is_final=True,
            )
            self.ckpt_manager.save_check(
                step=current_step,
                val_score=val_score if self.config.train.enable_validation else -1,
                pp_enabled=self.parallel_dims.pp_enabled,
            )

    def main_loop(self):
        if self.config.profiler.enable_profiler:
            self.profiler.start()

        # Do training
        while True:
            # get batch data
            is_end, global_batch, current_step, total_steps = (
                self.fetch_data_from_controller()
            )
            if is_end:
                break

            report_data = self.train_step(global_batch, current_step, total_steps)
            self.post_fetch_data_from_controller(report_data, current_step, total_steps)

            # because we update the model weight, so it's the next step
            current_step += 1

            # try validation
            val_score = None
            if (
                self.config.train.enable_validation
                and current_step % self.config.train.validation_step == 0
            ):
                # for save ckpt
                val_score = self.validate_step(current_step, total_steps)

            # try save ckpt
            self._try_save_ckpt(current_step, total_steps, val_score)

        # try process after final step
        self._try_process_after_final_step(current_step, total_steps)

        if self.config.profiler.enable_profiler:
            self.profiler.stop()
        logger.info("[Policy] Main loop finished. Shutdown background task event set.")
        self.train_stream.synchronize()
        self.handle_shutdown()

    @torch.no_grad()
    def validate_step(self, current_step: int, total_steps: int):
        logger.info(f"[SFTTrainer] Validation at step {current_step}/{total_steps}...")
        self.model.eval()
        val_total_loss = 0.0
        val_total_samples = 0

        while True:
            is_end, val_global_batch, _, _ = self.fetch_data_from_controller(
                current_step
            )
            if is_end:
                break

            fixed_length = (
                self.config.policy.model_max_length
                if self.parallel_dims.pp_enabled
                and not self.parallel_dims.pp_dynamic_shape
                else None
            )
            if fixed_length is None:
                max_len = min(
                    self.config.policy.model_max_length,
                    self.data_packer.sft_compute_max_len(val_global_batch),
                )
            else:
                max_len = fixed_length
            if self.seq_len_multiple > 1:
                max_len = (
                    (max_len + self.seq_len_multiple - 1)
                    // self.seq_len_multiple
                    * self.seq_len_multiple
                )

            val_batch = self.data_packer.sft_collate_fn(
                val_global_batch,
                computed_max_len=max_len,
                pad_token_id=self.tokenizer.pad_token_id,
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

            if self.parallel_dims.cp_enabled:
                input_ids_before_cp = val_inputs
                position_ids_before_cp = val_position_ids
                padding_mask_before_cp = val_padding_mask

                [val_inputs, val_position_ids, val_padding_mask] = (
                    slice_inputs_for_ulysses(
                        [val_inputs, val_position_ids, val_padding_mask],
                        self.parallel_dims.mesh["cp"],
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

                # recover from ulysses if cp is enabled
                if self.parallel_dims.cp_enabled:
                    val_batch["input_ids"] = input_ids_before_cp
                    val_batch["position_ids"] = position_ids_before_cp
                    if padding_mask_before_cp is not None:
                        val_batch["padding_mask"] = padding_mask_before_cp

                val_loss = self.loss_fn(val_logits, val_labels)
            val_total_loss += val_loss.item() * val_inputs.size(0)
            val_total_samples += len(val_inputs)

        concat_avg_count = torch.tensor(
            [val_total_loss, val_total_samples], dtype=torch.float32, device=self.device
        )
        self.inter_policy_nccl.all_reduce(
            concat_avg_count, op=torch.distributed.ReduceOp.SUM
        )
        val_avg_loss = (concat_avg_count[0] / concat_avg_count[1]).item()
        logger.info(f"[SFTTrainer] Validation loss: {val_avg_loss}")
        return val_avg_loss

    def train_step(
        self, global_batch: List[Dict[str, Any]], current_step: int, total_steps: int
    ) -> Dict[str, Any]:
        report_data = {}
        self.model.train()

        # train a global batch data
        acc_loss = torch.zeros(1, device=self.device)
        self.optimizers.zero_grad()
        global_batch_size = len(global_batch)
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
            raw_batch = global_batch[i : i + self.config.train.train_policy.mini_batch]
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
            batch = self.data_packer.sft_collate_fn(
                raw_batch,
                computed_max_len=max_len,
                pad_token_id=self.tokenizer.pad_token_id,
                ignore_label_id=-100,
            )

            # if [profiler.enable_nsys] is true, cudaProfilerStart() / cudaProfilerStop() are used to trigger nsys capture
            # settings from [profiler.sub_profiler_config] are reused
            if (
                self.config.profiler.enable_nsys
                and self.profiler.global_rank in self.profiler.rank_filter
            ):
                if (
                    current_step
                    == self.profiler.wait_steps + self.profiler.warmup_steps
                ):
                    torch.cuda.cudart().cudaProfilerStart()
                elif (
                    current_step
                    == self.profiler.wait_steps
                    + self.profiler.warmup_steps
                    + self.profiler.active_steps
                ):
                    torch.cuda.cudart().cudaProfilerStop()

            self.model.train()
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            labels = batch.pop("label_ids")

            position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(**batch)

            batch["position_ids"] = position_ids
            padding_mask = batch.get("padding_mask", None)

            if self.parallel_dims.cp_enabled:
                input_ids_before_cp = input_ids
                position_ids_before_cp = position_ids
                padding_mask_before_cp = padding_mask

                [input_ids, position_ids, padding_mask] = slice_inputs_for_ulysses(
                    [input_ids, position_ids, padding_mask],
                    self.parallel_dims.mesh["cp"],
                )

                batch["input_ids"] = input_ids
                batch["position_ids"] = position_ids
                if padding_mask is not None:
                    batch["padding_mask"] = padding_mask

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
                loss = (
                    torch.mean(torch.stack(losses)).to(self.device)
                    if pp_last_stage
                    else torch.tensor([-1.0], device=self.device)
                )
            else:
                # # This code is just for debugging purposes, where we can test whether the model can generate tokens correctly
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
                #         for i in range(len(last_token_ids)):
                #             print(
                #                 f"generated tokens at sample {i}: {self.tokenizer.decode(torch.cat(last_token_ids, dim=-1)[i])}"
                #             )

                #     return
                # #########################################################################################

                logits = self.model(**batch)

                # recover from ulysses if cp is enabled
                if self.parallel_dims.cp_enabled:
                    batch["input_ids"] = input_ids_before_cp
                    batch["position_ids"] = position_ids_before_cp
                    if padding_mask_before_cp is not None:
                        batch["padding_mask"] = padding_mask_before_cp

                loss = self.loss_fn(
                    logits,
                    labels,
                    loss_scaling_factor=1.0 / len(mini_batch_begin_idxs),
                )

                # # Hint FSDP to do all-reduce on the last backward pass
                # if hasattr(self.model, "set_is_last_backward"):
                #     print(f"set_is_last_backward: {i == mini_batch_begin_idxs[-1]}")
                #     self.model.set_is_last_backward(i == mini_batch_begin_idxs[-1])
                loss.backward()
            acc_loss += loss.detach()

        self._allreduce_gradients()
        grad_norm = self._clipping_gradients()
        self.optimizers.step()
        self.lr_schedulers.step()

        if (
            self.config.train.sync_weight_interval > 0
            and current_step % self.config.train.sync_weight_interval == 0
        ):
            self._sync_weights_between_replicas()

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

        # INFO: will do statistic loss across replicas for logging in the controller

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

        # For profiling
        self.profiler.step()

        return report_data

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
        if self.parallel_dims.dp_enabled:
            dp_group = self.parallel_dims.mesh["dp"].get_group()
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
                dp_group=dp_group,
                cp_group=cp_group,
            )
        )
