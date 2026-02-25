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


import asyncio
import copy
from queue import Queue
import threading
import time
from typing import List, Tuple
from cosmos_rl.dispatcher.command import (
    PolicyToPolicyBroadcastCommand,
    PolicyToPolicyUnicastCommand,
    WeightResumeCommand,
)
from cosmos_rl.dispatcher.data.schema import RLPayload
import torch

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils import util
from cosmos_rl.utils.distributed import destroy_distributed

from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer
from cosmos_rl.policy.worker import RLPolicyWorker, SFTPolicyWorker
import cosmos_rl.utils.distributed as dist_util
import torch.distributed as dist
from cosmos_rl.policy.trainer.optm import build_lr_schedulers


class MultiReplicaSFTPolicyWorker(RLPolicyWorker):
    trainer: SFTTrainer

    policy_command_handler_registry = copy.deepcopy(
        RLPolicyWorker.policy_command_handler_registry
    )

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs):
        super(MultiReplicaSFTPolicyWorker, self).__init__(
            config=config, parallel_dims=parallel_dims, **kwargs
        )
        self.is_user_provided_train_set = kwargs.get("dataset", None) is not None
        self.is_user_provided_val_set = kwargs.get("val_dataset", None) is not None
        self.weight_sync_done = False
        # Calculate the step interval to save the checkpoint
        self._save_freq = self.config.train.ckpt.save_freq
        self.train_step = None
        self.current_epoch = None
        self.total_steps = None
        self.loaded_total_steps = None
        self.loaded_train_step = None
        self.do_save = False
        # Setup hooks for later use
        SFTPolicyWorker.setup_hooks(self)

    def execute_policy_to_policy_broadcast(
        self, command: PolicyToPolicyBroadcastCommand
    ):
        if self.total_steps is None:
            self.trainer.lr_schedulers = build_lr_schedulers(
                self.trainer.optimizers, self.config, command.total_steps
            )
            self.total_steps = command.total_steps
            if self.loaded_total_steps is not None and self.loaded_total_steps > 0:
                assert (
                    self.total_steps == self.loaded_total_steps
                ), f"total_steps {self.total_steps} should be equal to loaded_total_steps {self.loaded_total_steps}"
        else:
            assert (
                self.total_steps == command.total_steps
            ), f"total_steps {self.total_steps} should be equal to command.total_steps {command.total_steps}"
        ret = super().execute_policy_to_policy_broadcast(command)
        self.weight_sync_done = True
        logger.info("[SFT] Weight synchronization from broadcast command done.")
        return ret

    def execute_policy_to_policy_unicast(self, command: PolicyToPolicyUnicastCommand):
        if self.total_steps is None:
            self.trainer.lr_schedulers = build_lr_schedulers(
                self.trainer.optimizers, self.config, command.total_steps
            )
            self.total_steps = command.total_steps
            if self.loaded_total_steps is not None and self.loaded_total_steps > 0:
                assert (
                    self.total_steps == self.loaded_total_steps
                ), f"total_steps {self.total_steps} should be equal to loaded_total_steps {self.loaded_total_steps}"
        else:
            assert (
                self.total_steps == command.total_steps
            ), f"total_steps {self.total_steps} should be equal to command.total_steps {command.total_steps}"
        ret = super().execute_policy_to_policy_unicast(command)
        self.weight_sync_done = True
        logger.info("[SFT] Weight synchronization from unicast command done.")
        return ret

    def execute_weight_resume(self, command: WeightResumeCommand = None):
        self.loaded_total_steps, self.loaded_train_step, ckpt_extra_info = (
            self.trainer.load_model()
        )
        logger.info("[SFT] Weight resume command executed, model weights loaded.")
        if self.config.train.resume:
            self.api_client.post_resume_info(ckpt_extra_info)
            logger.info("[SFT] Posted resume info to controller after weight resume.")
        return False

    def prepare_shard_infos_for_weight_sync_insts(self):
        pass

    def validate(self, is_last_step: bool = False):
        """
        Perform validation and return the average validation loss.
        """
        # validation
        if not self.config.validation.enable:
            return None
        if (
            (
                (self.train_step is None or self.train_step == 0)
                and self.config.validation.val_before_train
            )
            or (
                self.train_step is not None
                and self.train_step != 0
                and self.train_step % self.config.validation.freq == 0
            )
            or is_last_step
        ):
            pass
        else:
            return None
        logger.info(f"Validation at step {self.train_step}/{self.total_steps}...")
        val_total_loss = 0.0
        val_batch_size = (
            self.config.validation.batch_size
            or self.config.train.train_batch_per_replica
        )
        batch_index = 0
        pre_validation_called = False
        while True:
            is_end, new_epoch = self.request_new_prompts(
                val_batch_size // self.dp_world_size,
                prompt_queue=self.data_queue,
                validation_step=self.train_step or 0,
            )
            assert (
                (val_batch_size % self.dp_world_size) == 0
            ), f"val_batch_size({val_batch_size}) must be divisible by dp_world_size({self.dp_world_size})"
            if self.data_queue.qsize() >= val_batch_size // self.dp_world_size:
                global_batch = [
                    self.data_queue.get()
                    for _ in range(val_batch_size // self.dp_world_size)
                ]
                # Call pre_validation_hook
                if self.pre_validation_hook is not None and not pre_validation_called:
                    report_data = {
                        "current_epoch": self.current_epoch,
                        "is_last_step": is_last_step,
                    }
                    self.pre_validation_hook(self, report_data=report_data)
                    pre_validation_called = True

                # Call pre_per_step_validation_hook
                if self.pre_per_step_validation_hook is not None:
                    report_data = {
                        "current_epoch": self.current_epoch,
                        "batch_index": batch_index,
                    }
                    self.pre_per_step_validation_hook(self, report_data=report_data)

                val_score = self.trainer.step_validation(
                    global_batch, self.train_step, self.total_steps
                )

                # Call post_per_step_validation_hook
                if self.post_per_step_validation_hook is not None:
                    report_data = {
                        "current_epoch": self.current_epoch,
                        "batch_index": batch_index,
                        "val_score": val_score,
                    }
                    self.post_per_step_validation_hook(self, report_data=report_data)

                val_total_loss += val_score
                batch_index += 1

            if is_end:
                break  # break outer epoch loop
        if val_total_loss == 0.0:
            logger.warning(
                f"[SFT] No validation data processed at train step {self.train_step}/{self.total_steps}, epoch {self.current_epoch}"
            )
        val_avg_loss = val_total_loss / (val_batch_size * batch_index or 1)
        logger.debug(
            f"[SFT] Validation loss: {val_avg_loss} for train step {self.train_step}/{self.total_steps}, epoch {self.current_epoch}"
        )
        # Clear the data queue for later use
        self.data_queue.queue.clear()

        # Call post_validation_hook
        if self.post_validation_hook is not None:
            report_data = {
                "current_epoch": self.current_epoch,
                "val_avg_loss": val_avg_loss,
            }
            self.post_validation_hook(self, report_data=report_data)

        # Call custom logger functions
        report_data = {
            "val/cur_epoch": self.current_epoch,
            "val/avg_loss": val_avg_loss,
            "val/train_epochs": self.config.train.epoch,
            "val/total_steps": self.total_steps,  # This total_steps is for training
            "val/train_step": self.train_step,
        }

        if util.is_master_rank(self.parallel_dims, self.global_rank):
            for key in report_data.keys():
                if isinstance(report_data[key], torch.Tensor):
                    report_data[key] = report_data[key].item()
            logger.debug(
                f"[SFT] Validation step {self.train_step}/{self.total_steps}, report data: {report_data}"
            )
            self.api_client.post_policy_train_ack(
                self.replica_name,
                self.train_step,
                self.total_steps,
                False,
                report_data,
            )

        return val_avg_loss

    def request_new_prompts(self, batch_size: int, prompt_queue: Queue, **kwargs):
        """
        Request new prompts from the controller for both training and validation.
        """
        assert (
            prompt_queue.empty()
        ), "Prompt queue should be empty before requesting new prompts."
        prompts_and_is_end: Tuple[List[RLPayload] | None, bool] = (None, False)
        is_validation = kwargs.get("validation_step", None) is not None
        if self.global_rank == 0:
            # blocking request to get prompts from controller
            # batch_size is per data parallel rank so we need to multiply it with data parallel size
            payloads, is_end = self.api_client.get_next_prompt(
                batch_size * self.parallel_dims.mesh["dp"].size(), **kwargs
            )

            assert all(
                payload["prompt_idx"] >= 0 for payload in payloads
            ), "All payloads should have a valid prompt index"

            if len(payloads) > 0:
                if self.config.train.local_dataset:
                    for payload in payloads:
                        payload["prompt"] = self.data_fetcher.get_payload_by_index(
                            payload["prompt_idx"],
                            is_validation=is_validation,
                        )
                payloads = [RLPayload.model_validate(payload) for payload in payloads]
            prompts_and_is_end = (
                payloads if len(payloads) > 0 else None,
                is_end,
            )

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end = dist_util.broadcast_object_cpu(prompts_and_is_end)
        if self.parallel_dims.mesh["dp"].size() > 1:
            # Scatter the prompts to all data parallel ranks
            prompts, is_end = prompts_and_is_end
            if (
                prompts is not None
                and self.parallel_dims.mesh["dp"].get_local_rank() == 0
            ):
                # assert (
                #     len(prompts) % self.parallel_dims.mesh["dp"].size() == 0
                # ), f"Number of prompts {len(prompts)} must be divisible by data parallel size {self.parallel_dims.mesh['dp'].size()}"
                ranks_to_scatter = self.parallel_dims.mesh["dp"].size()

                # Ignore extra prompts that cannot be evenly scattered
                prompts = prompts[: len(prompts) - len(prompts) % ranks_to_scatter]

                # Distribute prompts in an interleaved (round-robin) fashion
                # Rank 0 gets indices [0, N, 2N, ...], Rank 1 gets [1, N+1, 2N+1, ...], etc.
                scattered_prompts_and_is_end = []
                for rank in range(ranks_to_scatter):
                    rank_prompts = prompts[rank::ranks_to_scatter]
                    scattered_prompts_and_is_end.append(
                        (
                            rank_prompts,
                            is_end,
                        )
                    )
            else:
                scattered_prompts_and_is_end = [
                    (None, is_end) for _ in range(self.parallel_dims.mesh["dp"].size())
                ]
            recv_prompts_and_is_end = [(None, False)]
            dist.scatter_object_list(
                recv_prompts_and_is_end,
                scattered_prompts_and_is_end,
                group=self.parallel_dims.mesh["dp"].get_group(),
                group_src=0,
            )
            prompts_and_is_end = recv_prompts_and_is_end[0]
        prompts, is_end = prompts_and_is_end
        target_train_step = 0
        target_epoch = 0
        target_remain_samples_num = float("inf")
        new_epoch = False
        if prompts is not None:
            for fullpayload in prompts:
                payload = fullpayload.prompt
                if self.train_step is None:
                    self.train_step = fullpayload.weight_version
                    if self.loaded_train_step is not None:
                        assert (
                            self.train_step == self.loaded_train_step
                        ), f"train_step {self.train_step} should be equal to loaded_train_step {self.loaded_train_step}"
                target_train_step = max(target_train_step, fullpayload.weight_version)
                target_epoch = max(target_epoch, fullpayload.extra_info["epoch"])
                target_remain_samples_num = min(
                    target_remain_samples_num,
                    fullpayload.extra_info["remain_samples_num"],
                )
                if (
                    self.config.train.train_policy.conversation_column_name
                    and not self.is_user_provided_train_set
                ):
                    if isinstance(payload, list):
                        payload = [
                            self.data_packer.sft_process_sample(
                                x[
                                    self.config.train.train_policy.conversation_column_name
                                ]
                            )
                            for x in payload
                        ]
                    else:
                        payload = self.data_packer.sft_process_sample(
                            payload[
                                self.config.train.train_policy.conversation_column_name
                            ],
                        )
                else:
                    if isinstance(payload, list):
                        payload = [
                            self.data_packer.sft_process_sample(x) for x in payload
                        ]
                    else:
                        payload = self.data_packer.sft_process_sample(payload)
                prompt_queue.put(payload)
            assert (
                self.train_step == target_train_step or is_validation
            ), f"train_step {self.train_step} should be equal to target_train_step {target_train_step}"
            new_epoch = (
                self.current_epoch is not None and self.current_epoch != target_epoch
            )
            self.current_epoch = target_epoch
            self.remain_samples_num = target_remain_samples_num
        assert is_end or prompt_queue.qsize() >= batch_size, (
            f"Prompt queue size {prompt_queue.qsize()} should be at least batch_size {batch_size} "
            "unless is_end is True"
        )
        return is_end, new_epoch

    def main_loop(self):
        def fetch_command_helper(trainer: MultiReplicaSFTPolicyWorker):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_command())
            new_loop.stop()
            new_loop.close()
            return

        # Start the thread with daemon=True, so it will exit when the main program exits.
        # we need all ranks have fetch_command_thread, so that buildmesh command can be broadcasted to all ranks
        # TODO(zjx): we will only let rank 0 fetch and broadcast command
        self.fetch_command_thread = threading.Thread(
            target=fetch_command_helper,
            args=(self,),
            daemon=True,
            name="fetch_command_thread",
        ).start()

        self.profiler.start()
        pp_last_stage = False
        if self.parallel_dims.pp_enabled:
            pp_last_stage = (
                self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
            )

        data_arrival_event = torch.cuda.Event(enable_timing=True)
        data_arrival_event.record()
        while True:
            self.broadcast_command()
            while len(self.command_buffer.queue) > 0:
                cmd = self.command_buffer.get_nowait()
                logger.debug(f"[Policy] Executing command: {cmd}")
                assert (
                    isinstance(cmd, PolicyToPolicyUnicastCommand)
                    or isinstance(cmd, PolicyToPolicyBroadcastCommand)
                    or isinstance(cmd, WeightResumeCommand)
                ), f"Expected PolicyToPolicyUnicastCommand or PolicyToPolicyBroadcastCommand or WeightResumeCommand, but got {type(cmd)}"

                abort = self.execute_command(cmd)
                assert not abort, "Aborting main loop due to command execution."

            if not self.weight_sync_done:
                time.sleep(1)
                continue
            if self.train_step is None:
                val_avg_loss = self.validate(is_last_step=False)
            is_end, new_epoch = self.request_new_prompts(
                batch_size=self.config.train.train_batch_per_replica
                // self.dp_world_size,
                prompt_queue=self.data_queue,
            )

            if self.do_save and self.is_master_replica:
                self.trainer.checkpointing(
                    total_steps=self.total_steps,
                    train_step=self.train_step,
                    save_freq=self._save_freq,
                    pp_last_stage=pp_last_stage,
                    is_last_step=False,
                    val_score=val_avg_loss,
                    do_save=True,
                    **{
                        "remain_samples_num": self.remain_samples_num,
                    },
                )
            assert (
                (self.config.train.train_batch_per_replica % self.dp_world_size) == 0
            ), f"train_batch_per_replica({self.config.train.train_batch_per_replica}) must be divisible by dp_world_size({self.dp_world_size})"
            if (
                self.data_queue.qsize()
                >= self.config.train.train_batch_per_replica // self.dp_world_size
            ):
                global_batch = [
                    self.data_queue.get()
                    for _ in range(
                        self.config.train.train_batch_per_replica // self.dp_world_size
                    )
                ]
                if (
                    self.config.profiler.enable_nsys
                    and self.profiler.global_rank in self.profiler.rank_filter
                ):
                    if (
                        self.train_step
                        == self.profiler.wait_steps + self.profiler.warmup_steps
                    ):
                        torch.cuda.cudart().cudaProfilerStart()
                    elif (
                        self.train_step
                        == self.profiler.wait_steps
                        + self.profiler.warmup_steps
                        + self.profiler.active_steps
                    ):
                        torch.cuda.cudart().cudaProfilerStop()

                report_data = self.trainer.step_training(
                    global_batch=global_batch,
                    total_steps=self.total_steps,
                    train_step=self.train_step,
                    save_freq=self._save_freq,
                    inter_policy_nccl=self.inter_policy_nccl,
                    data_arrival_event=data_arrival_event,
                )

                self.train_step += 1

                if util.is_master_rank(self.parallel_dims, self.global_rank):
                    for key in report_data.keys():
                        if isinstance(report_data[key], torch.Tensor):
                            report_data[key] = report_data[key].item()
                    logger.debug(
                        f"[SFT] Train step {self.train_step}/{self.total_steps}, report data: {report_data}"
                    )
                    self.api_client.post_policy_train_ack(
                        self.replica_name,
                        self.train_step,
                        self.total_steps,
                        self.profiler.check_finished(),
                        report_data,
                    )
                data_arrival_event = torch.cuda.Event(enable_timing=True)
                data_arrival_event.record()

            if self.train_step >= self.total_steps or is_end:
                logger.info(
                    f"[SFT] Reached the end of training at step {self.train_step} with is_end={is_end}."
                )
                break  # break outer epoch loop

            if self.config.train.ckpt.save_freq_in_epoch > 0:
                # Checkpointing based on epoch if `save_freq_in_epoch` is set
                if new_epoch:
                    # New epoch begins and old epoch ends
                    # So check the epoch number against save_freq_in_epoch for saving checkpoint
                    do_save = (
                        self.current_epoch % self.config.train.ckpt.save_freq_in_epoch
                        == 0
                    )
                    if do_save:
                        logger.info(
                            f"[Controller] Epoch {self.current_epoch} ends, triggering checkpoint saving at step {self.train_step}"
                        )
            else:
                # Checkpointing based on step if `save_freq_in_epoch` is not set
                do_save = (
                    self.train_step % self.config.train.ckpt.save_freq == 0
                    and self.train_step > 0
                )
            self.do_save = do_save
            val_avg_loss = self.validate(is_last_step=False)

        # Finally: validation and save checkpoint
        val_avg_loss = self.validate(is_last_step=True)
        if self.is_master_replica:
            self.trainer.checkpointing(
                total_steps=self.total_steps,
                train_step=self.train_step,
                save_freq=self._save_freq,
                is_last_step=True,
                pp_last_stage=pp_last_stage,
                val_score=val_avg_loss,
                do_save=True,
            )

        self.train_stream.synchronize()
        self.handle_shutdown()

    def destroy_worker(self):
        destroy_distributed()
        logger.info("[Policy] Process group destroyed.")


MultiReplicaSFTPolicyWorker.register_policy_command_handler(
    PolicyToPolicyBroadcastCommand
)(MultiReplicaSFTPolicyWorker.execute_policy_to_policy_broadcast)
MultiReplicaSFTPolicyWorker.register_policy_command_handler(
    PolicyToPolicyUnicastCommand
)(MultiReplicaSFTPolicyWorker.execute_policy_to_policy_unicast)
MultiReplicaSFTPolicyWorker.register_policy_command_handler(WeightResumeCommand)(
    MultiReplicaSFTPolicyWorker.execute_weight_resume
)
