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
from cosmos_rl.dispatcher.command import (
    PolicyToPolicyBroadcastCommand,
    PolicyToPolicyUnicastCommand,
    WeightResumeCommand,
)
from cosmos_rl.dispatcher.data.schema import RLPayload
import torch
from tqdm import tqdm

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils import util
from cosmos_rl.utils.distributed import destroy_distributed
from cosmos_rl.utils.wandb_logger import (
    is_wandb_available,
    log_wandb,
)

from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer
from cosmos_rl.policy.worker import RLPolicyWorker
import cosmos_rl.utils.distributed as dist_util
import torch.distributed as dist


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
        if self.config.train.ckpt.save_freq_in_epoch > 0:
            # Use save_freq_in_epoch to calculate the save frequency in priority
            self._save_freq = (
                self.config.train.ckpt.save_freq_in_epoch * len(self.train_data_loader)
            ) // self.dp_world_size
            logger.info(
                f"Checkpoint will be saved every {self._save_freq} steps, which is approximately every `train.ckpt.save_freq_in_epoch` {self.config.train.ckpt.save_freq_in_epoch} epochs. `train.ckpt.save_freq` will be ignored."
            )
        else:
            self._save_freq = self.config.train.ckpt.save_freq
        self.train_step = None

    def execute_policy_to_policy_broadcast(
        self, command: PolicyToPolicyBroadcastCommand
    ):
        ret = super().execute_policy_to_policy_broadcast(command)
        self.weight_sync_done = True
        self.total_steps = command.total_steps
        logger.info("[SFT] Weight synchronization from broadcast command done.")
        return ret

    def execute_policy_to_policy_unicast(self, command: PolicyToPolicyUnicastCommand):
        ret = super().execute_policy_to_policy_unicast(command)
        self.weight_sync_done = True
        self.total_steps = command.total_steps
        logger.info("[SFT] Weight synchronization from unicast command done.")
        return ret

    def execute_weight_resume(self, command: WeightResumeCommand = None):
        self.trainer.load_model()
        logger.info("[SFT] Weight resume command executed, model weights loaded.")
        return False

    def prepare_shard_infos_for_weight_sync_insts(self):
        pass

    def validate(self, current_epoch: int, is_last_step: bool = False):
        if not self.config.validation.enable:
            return None
        if self.parallel_dims.dp_replicate_coord[0] != 0:
            return
        if (
            (self.train_step == 0 and self.config.validation.val_before_train)
            or (
                self.train_step != 0
                and self.train_step % self.config.validation.freq == 0
            )
            or is_last_step
        ):
            pass
        else:
            return None

        # Call pre_validation_hook
        if self.pre_validation_hook is not None:
            report_data = {
                "current_epoch": current_epoch,
                "is_last_step": is_last_step,
            }
            self.pre_validation_hook(self, report_data=report_data)

        # validation
        logger.info(f"Validation at step {self.train_step}/{self.total_steps}...")
        val_total_loss = 0.0

        for batch_index, val_global_batch in enumerate(
            tqdm(self.val_data_loader, desc="Validation")
        ):
            # Call pre_per_step_validation_hook
            if self.pre_per_step_validation_hook is not None:
                report_data = {
                    "current_epoch": current_epoch,
                    "batch_index": batch_index,
                }
                self.pre_per_step_validation_hook(self, report_data=report_data)

            val_score = self.trainer.step_validation(
                val_global_batch, self.train_step, self.total_steps
            )

            # Call post_per_step_validation_hook
            if self.post_per_step_validation_hook is not None:
                report_data = {
                    "current_epoch": current_epoch,
                    "batch_index": batch_index,
                    "val_score": val_score,
                }
                self.post_per_step_validation_hook(self, report_data=report_data)

            val_total_loss += val_score

        val_avg_loss = val_total_loss / len(self.val_data_loader.dataset)
        logger.info(
            f"[SFT] Validation loss: {val_avg_loss} for train step {self.train_step}/{self.total_steps}, epoch {current_epoch}"
        )

        # Call post_validation_hook
        if self.post_validation_hook is not None:
            report_data = {
                "current_epoch": current_epoch,
                "val_avg_loss": val_avg_loss,
            }
            self.post_validation_hook(self, report_data=report_data)

        # Call custom logger functions
        report_data = {
            "val/cur_epoch": current_epoch,
            "val/avg_loss": val_avg_loss,
            "val/train_epochs": self.epoch,
            "val/total_steps": self.total_steps,  # This total_steps is for training
            "val/train_step": self.train_step,
        }

        if util.is_master_rank(self.parallel_dims, self.global_rank):
            if "wandb" in self.config.logging.logger and is_wandb_available():
                log_wandb(
                    data=report_data,
                    step=self.train_step,
                )
            for custom_logger_fn in self.custom_logger_fns:
                try:
                    custom_logger_fn(report_data, self.train_step)
                except Exception as e:
                    logger.warning(f"[SFT] Error calling custom logger function: {e}")

        return val_avg_loss

    def request_new_prompts(self, batch_size: int, prompt_queue: Queue, **kwargs):
        """
        Request new prompts from the controller for both training and validation.
        """
        prompts_and_is_end = (None, False)
        if self.global_rank == 0:
            # blocking request to get prompts from controller
            # batch_size is per data parallel rank so we need to multiply it with data parallel size
            payloads, is_end = self.api_client.get_next_prompt(
                batch_size * self.parallel_dims.mesh["dp"].size(), **kwargs
            )

            assert all(
                payload["prompt_idx"] >= 0 for payload in payloads
            ), "All payloads should have a valid prompt index"

            is_validation = kwargs.get("validation_step", None) is not None

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
        if prompts is not None:
            for p in prompts:
                prompt_queue.put(p)
        return is_end

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
        # pp_last_stage = False

        # if self.parallel_dims.pp_enabled:
        #     pp_last_stage = (
        #         self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        #     )

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

            is_end = self.request_new_prompts(
                batch_size=self.config.train.train_batch_per_replica
                // self.dp_world_size,
                prompt_queue=self.data_queue,
            )
            assert (
                (self.config.train.train_batch_per_replica % self.dp_world_size) == 0
            ), f"train_batch_per_replica({self.config.train.train_batch_per_replica}) must be divisible by dp_world_size({self.dp_world_size})"
            if (
                self.data_queue.qsize()
                >= self.config.train.train_batch_per_replica // self.dp_world_size
            ):
                global_batch = []
                target_train_step = 0
                for _ in range(
                    self.config.train.train_batch_per_replica // self.dp_world_size
                ):
                    fullpayload: RLPayload = self.data_queue.get()
                    payload = fullpayload.prompt
                    self.train_step = (
                        fullpayload.weight_version
                        if self.train_step is None
                        else self.train_step
                    )
                    target_train_step = max(
                        target_train_step, fullpayload.weight_version
                    )
                    if (
                        self.config.train.train_policy.conversation_column_name
                        and not self.is_user_provided_train_set
                    ):
                        if isinstance(payload, list):
                            global_batch.append(
                                [
                                    self.data_packer.sft_process_sample(
                                        x[
                                            self.config.train.train_policy.conversation_column_name
                                        ]
                                    )
                                    for x in payload
                                ]
                            )
                        else:
                            global_batch.append(
                                self.data_packer.sft_process_sample(
                                    payload[
                                        self.config.train.train_policy.conversation_column_name
                                    ],
                                )
                            )
                    else:
                        if isinstance(payload, list):
                            global_batch.append(
                                [
                                    self.data_packer.sft_process_sample(x)
                                    for x in payload
                                ]
                            )
                        else:
                            global_batch.append(
                                self.data_packer.sft_process_sample(payload)
                            )
                assert (
                    self.train_step == target_train_step
                ), f"train_step {self.train_step} should be equal to target_train_step {target_train_step}"
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
                )

                self.train_step += 1

                if report_data and util.is_master_rank(
                    self.parallel_dims, self.global_rank
                ):
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

                if (
                    self.config.train.max_num_steps is not None
                    and self.train_step >= self.total_steps
                ):
                    break  # break outer epoch loop

                # val_avg_loss = self.validate(
                #     current_epoch=cur_epoch, is_last_step=False
                # )

                # self.trainer.checkpointing(
                #     total_steps=self.total_steps,
                #     train_step=self.train_step,
                #     save_freq=self._save_freq,
                #     pp_last_stage=False,
                #     is_last_step=False,
                #     val_score=val_avg_loss,
                # )
            if is_end:
                break

            # cur_epoch += 1

        # Finally: validation and save checkpoint
        # val_avg_loss = self.validate(current_epoch=cur_epoch, is_last_step=True)
        # self.trainer.checkpointing(
        #     total_steps=self.total_steps,
        #     train_step=self.train_step,
        #     save_freq=self._save_freq,
        #     is_last_step=True,
        #     pp_last_stage=pp_last_stage,
        #     val_score=val_avg_loss,
        # )

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
