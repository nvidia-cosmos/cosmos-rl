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

import copy
import subprocess
import atexit
import sys
import uuid
import asyncio
import time
import os
import math
import threading
import tempfile
from typing import List, Dict, Tuple, Optional, Callable
from cosmos_rl.dispatcher.replica import Atom, Rollout
from cosmos_rl.dispatcher.protocol import Role, MESH_NAMES
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.wandb_logger import (
    is_wandb_available,
    init_wandb,
)
import cosmos_rl.utils.util as util
import cosmos_rl.utils.network_util as network_util
import cosmos_rl.utils.constant as constant
from torch.utils.data import Dataset
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.dispatcher.status import (
    PolicyStatusManager,
    RolloutStatusManager,
)
from cosmos_rl.policy.config import Config, SubProfilerConfig
from cosmos_rl.dispatcher.protocol import SetProfileRequest
from cosmos_rl.utils.parallelism_map import ParallelizedShardMapper
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher


class Controller:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Controller, cls).__new__(cls)
            cls._instance._init_dist()
        return cls._instance

    def __init__(self):
        if not hasattr(self, "config"):
            self._init_dist()
        self._init_status()

    def _init_status(self):
        self.policy_status_manager = PolicyStatusManager()
        self.rollout_status_manager = RolloutStatusManager()
        self.teacher_result_manager = set()
        self.stat_prompt_tokens_count = 0
        self.stat_completion_tokens_count = 0
        self.stat_n_samples = 0
        self.begin_time = None
        # nccl error check
        self.post_ncclerror_policy_invoke_id = 0
        self.post_ncclerror_rollout_invoke_id = 0

    def _init_dist(self):
        self.config = None
        self.temp_kv_store = {}

        self.life_cycle_lock = asyncio.Lock()
        self.shut_down_event = threading.Event()

    def setup(
        self,
        config: Config,
        redis_port: int,
        redis_logfile_path: str,
        dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        custom_logger_fns: Optional[List[Callable]] = None,
        hook_fns: Optional[Dict[str, Callable]] = None,
        sampler: Optional[Callable] = None,
        batch_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        val_batch_sampler: Optional[Callable] = None,
    ):
        if self.config is not None:
            raise Exception(
                "[Controller] Config has been set. Please do not call setup again."
            )

        self.config = config
        task_type = config.train.train_policy.type
        self.policy_to_rollout_shard_mapper = ParallelizedShardMapper.get_instance(
            config
        )

        if "wandb" in config.logging.logger and is_wandb_available():
            init_wandb(config)
        else:
            logger.warning(
                "Wandb is not available. Please install it to use wandb logging features."
            )

        self.is_rl = task_type != "sft"
        self.weight_version_to_prompt_num = {}  # Only for on-policy.

        self.data_fetcher = ControllerDataFetcher(
            config=config,
            dataset=dataset,
            val_dataset=val_dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
            is_rl=self.is_rl,
        )

        redis_free_port = util.find_available_port(redis_port)
        self.config.redis = str(redis_free_port)

        ips = network_util.get_eth_ips()
        if len(ips) > 0:
            self.config.eth_ips = ";".join(ips)

        random_db_file_name = f"cosmos_rl_{str(uuid.uuid4())}.rdb"
        config_file_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".redis_config.conf"
        )

        custom_config = """
maxmemory 500G
maxmemory-policy allkeys-lfu
"""
        redis_cfg_path = util.write_redis_config(
            redis_free_port,
            redis_logfile_path,
            file_path=config_file_path.name,
            custom_config=custom_config,
        )
        redis_server_cmd = f'redis-server {redis_cfg_path} --dbfilename {random_db_file_name} --save ""'

        redis_server_proc = subprocess.Popen(
            redis_server_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr
        )

        # Check if the redis server started successfully
        redis_server_proc.wait()
        ret_code = redis_server_proc.returncode

        if ret_code is not None and ret_code != 0:
            raise RuntimeError(
                f"Failed to start redis server with command: {redis_server_cmd} with return code {ret_code}"
            )
        else:
            logger.info(
                f"[Controller] Redis server started on port {redis_free_port} with command {redis_server_cmd}"
            )

        self.redis_controller = RedisStreamHandler(
            ips=["0.0.0.0"], port=redis_free_port
        )

        self.policy_status_manager.setup(
            config,
            self.redis_controller,
            data_fetcher=self.data_fetcher,
            remain_samples_num=self.data_fetcher.remain_samples_num,
            samples_per_epoch=len(self.data_fetcher.dataset.train_set)
            * config.rollout.n_generation
            if self.is_rl
            else 0,
            tokenizer=util.setup_tokenizer(config.policy.model_name_or_path)
            if self.is_rl
            else None,
            current_step=self.data_fetcher.ckpt_extra_info.get("step", 0),
            max_num_steps=config.train.max_num_steps,
            custom_logger_fns=custom_logger_fns,
            hook_fns=hook_fns,
        )
        self.rollout_status_manager.setup(config, self.redis_controller)

        # Register the exit function to be called when the program exits
        def exit_server(redis_server_proc, redis_free_port):
            logger.info("Stopping redis server")
            redis_server_proc.terminate()
            redis_server_proc.wait()

            redis_terminate_cmd = f"redis-cli -p {redis_free_port} shutdown nosave"
            redis_terminate = subprocess.Popen(
                redis_terminate_cmd,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            redis_terminate.wait()
            try:
                os.unlink(config_file_path.name)
            except Exception:
                # best effort to remove the config file
                pass
            logger.info("Redis server stopped.")

        atexit.register(exit_server, redis_server_proc, redis_free_port)

    async def update_kv_store(self, key: str, value: str):
        self.temp_kv_store[key] = value

    async def clear_temp_kv_store(self, key: str):
        self.temp_kv_store.pop(key)

    async def get_kv_store(self, key: str) -> str:
        return self.temp_kv_store.get(key)

    """
    Rollout functionality
    """

    async def get_batched_prompt(
        self,
        n: int,
        validation_step: Optional[int] = None,
    ) -> Tuple[List[RLPayload], bool]:
        is_validation = validation_step is not None
        # Tag the prompt with specific weight-version for weight version control in on-policy training or outdated rollout control.
        rollouts_per_global_batch = self.config.train.train_batch_per_replica * len(
            self.policy_status_manager
        )
        global_batch_size = math.ceil(
            rollouts_per_global_batch / self.config.rollout.n_generation
        )  # global_batch_size: number of prompts needed for single policy step.
        rollouts_per_global_batch = rollouts_per_global_batch or 1
        weight_version_for_current_batch = (
            self.policy_status_manager.consumed_samples_num // rollouts_per_global_batch
        )

        if not is_validation and not self.config.mode == "colocated":
            # Throttle the generation speed:
            # 1. Detect the current left pending rollouts in all policy replicas.
            # 2. Check the config.train.train_policy.allowed_outdated_steps.
            # 3. If the current pending rollouts is larger than the allowed outdated version count, reduce the number of prompts to generate.
            current_pending_rollouts = self.policy_status_manager.samples_on_the_fly
            if (
                current_pending_rollouts
                >= (self.config.train.train_policy.allowed_outdated_steps + 1)
                * rollouts_per_global_batch
            ) and self.config.train.train_policy.variant != "dapo":
                # For non dapo variant, we only need to control the number of outdated weight versions when fetching new prompts.
                # Since the number of fetched prompts is directly related to the number of rollouts to be trained.
                n = min(
                    n,
                    self.config.train.train_policy.outdated_rollout_fetch_batch_size,
                )
                if n > 0:
                    # Log only when n is reduced but not when set to 0 since 0 is logged too frequently
                    logger.warning(
                        f"[Controller] Current pending rollouts {current_pending_rollouts} is larger than the allowed outdated version count {self.config.train.train_policy.allowed_outdated_steps * len(self.policy_status_manager)}. Generate with batch {n}"
                    )
            if (
                self.config.train.train_policy.variant == "dapo"
                and self.config.train.train_policy.max_retry_for_on_policy > 0
            ):
                # In DAPO, we also need to control the number of outdated weight versions when fetching new prompts.
                # Estimating the number of outdated weight versions when the generation results of these fetched prompts start training based on the total pending rollouts
                estimated_delta_weight_version = (
                    self.policy_status_manager.total_pending_rollouts()
                    // rollouts_per_global_batch
                )
                allowed_unfinished_weight_versions = (
                    self.config.train.train_policy.allowed_outdated_steps
                    - estimated_delta_weight_version
                )
                # Estimating the number of unfinished rollouts based on the samples on the fly and the pending rollouts
                estimated_unfinished_rollouts = max(
                    self.policy_status_manager.samples_on_the_fly
                    - self.policy_status_manager.total_pending_rollouts(),
                    0,
                )
                if (
                    estimated_unfinished_rollouts
                    >= (1 + allowed_unfinished_weight_versions)
                    * self.config.train.train_policy.max_retry_for_on_policy
                    * rollouts_per_global_batch
                ):
                    n = min(
                        n,
                        self.config.train.train_policy.outdated_rollout_fetch_batch_size,
                    )
                    if n > 0:
                        # Log only when n is reduced but not when set to 0 since 0 is logged too frequently
                        logger.warning(
                            f"[Controller] Current pending rollouts {current_pending_rollouts} is larger than the allowed outdated version count {self.config.train.train_policy.allowed_outdated_steps * len(self.policy_status_manager)}. Generate with batch {n}"
                        )

        if (
            (not is_validation)
            and len(self.rollout_status_manager.replica_scaling_log) == 0
            and not self.config.mode == "colocated"
        ):
            if self.config.train.train_policy.variant != "dapo":
                # Fully Synchronized mode is enabled and no dapo variant, we need to ensure that for each weight version, we fetch exactly global_batch_size prompts.
                if (
                    weight_version_for_current_batch
                    not in self.weight_version_to_prompt_num
                ):
                    n = min(n, global_batch_size)
                else:
                    n = min(
                        n,
                        global_batch_size
                        - self.weight_version_to_prompt_num[
                            weight_version_for_current_batch
                        ],
                    )

            payloads_list, is_end = self.data_fetcher.get_batched_prompt(
                n, validation_step
            )
            current_fetch_count = len(payloads_list)
            # record the number of valid prompts for current weight version
            if (
                weight_version_for_current_batch
                not in self.weight_version_to_prompt_num
            ):
                self.weight_version_to_prompt_num[weight_version_for_current_batch] = (
                    current_fetch_count
                )
            else:
                self.weight_version_to_prompt_num[weight_version_for_current_batch] += (
                    current_fetch_count
                )

            # check if for current weight version, we have reached the upper limit of retries to generate enough samples.
            if self.config.train.train_policy.max_retry_for_on_policy > 0:
                already_retried_times = math.ceil(
                    self.weight_version_to_prompt_num[weight_version_for_current_batch]
                    / global_batch_size
                )
                if (
                    already_retried_times
                    > self.config.train.train_policy.max_retry_for_on_policy
                ):
                    raise RuntimeError(
                        f"[Controller] After {self.config.train.train_policy.max_retry_for_on_policy} retries, samples for weight version {weight_version_for_current_batch} are still not enough. May be the dataset is too difficult for current model? Or you could also set the `max_retry_for_on_policy` to 0 or negative to always retry."
                    )

            for i in range(current_fetch_count):
                # get_batched_prompt is called in single thread, so we use `consumed_samples_num` to calculate the weight version.
                # This could ensure that each step of policy will get enough prompts to generae rollouts needed.
                payloads_list[i].weight_version = weight_version_for_current_batch
            # logger.info(f"[Controller] Fully Synchronized mode is enabled, weight_versions: {weight_versions}, train_batch_per_replica: {self.config.train.train_batch_per_replica}, policy_replicas: {len(self.policy_status_manager)}")
        else:
            payloads_list, is_end = self.data_fetcher.get_batched_prompt(
                n, validation_step
            )
            current_fetch_count = len(payloads_list)
            for i in range(current_fetch_count):
                payloads_list[i].weight_version = 0
        if not is_validation:
            self.policy_status_manager.samples_on_the_fly += (
                current_fetch_count * self.config.rollout.n_generation
            )

        return payloads_list, is_end

    async def set_profile(self, request: SetProfileRequest):
        replica = self.policy_status_manager[request.replica_name]
        if replica is None:
            logger.warning(
                f"[Controller] Replica {request.replica_name} not found in policy replicas. The profile request takes no effect."
            )
            return {
                "message": "Replica not found in policy replicas. The profile request takes no effect."
            }
        if replica.sub_profiler_config.do_profile:
            logger.warning(
                f"[Controller] Replica {request.replica_name} is already in profile mode. The profile request takes no effect."
            )
            return {
                "message": "Replica is already in profile mode. The profile request takes no effect."
            }
        else:
            kwargs_dict = request.model_dump()
            # remove the replica_name from the kwargs_dict
            kwargs_dict.pop("replica_name")
            # add do_profile to the kwargs_dict
            kwargs_dict["do_profile"] = True
            replica.sub_profiler_config = SubProfilerConfig(**kwargs_dict)
            logger.info(
                f"[Controller] Set profile mode for replica {request.replica_name}."
            )
            return {"message": f"Set replica {request.replica_name} to profile mode."}

    async def set_trace_path(
        self, replica_name: str, trace_path: str, global_rank: int
    ):
        replica = self.policy_status_manager[replica_name]
        if replica is None:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in policy replicas. The trace path request takes no effect."
            )
            return None
        return await replica.set_trace_path(trace_path, global_rank)

    async def put_rollouts(self, rollouts: List[Rollout]):
        """
        Dispatch the rollouts to the policy replicas in a round-robin manner.
        rollouts: List[Rollout]: The rollouts to be dispatched
        """
        completion_tokens_count, n_samples = self.policy_status_manager.put_rollouts(
            rollouts
        )

        self.stat_completion_tokens_count += completion_tokens_count
        self.stat_n_samples += n_samples

        # Statistic
        if self.begin_time is None:
            self.begin_time = time.time()

        # Print pending rollouts inside all policy replicas
        pending_count = self.policy_status_manager.total_pending_rollouts()

        elapsed_time_in_seconds = time.time() - self.begin_time
        logger.info(
            f"[Controller] Stat: {self.stat_n_samples} samples, {self.stat_completion_tokens_count} completion tokens, {pending_count} pending rollouts, {elapsed_time_in_seconds} seconds elapsed"
        )

    """
    State of controller
    """

    def policy_mesh_and_group_size(self) -> tuple[List[str], List[int]]:
        mesh_names = copy.deepcopy(MESH_NAMES)
        group_sizes = []
        for replica in self.policy_status_manager:
            group_sizes.append(replica.group_size)
            break

        return mesh_names, group_sizes

    def rollout_mesh_and_group_size(self) -> tuple[List[str], List[int]]:
        mesh_names = copy.deepcopy(MESH_NAMES)
        group_sizes = []
        for replica in self.rollout_status_manager:
            group_sizes.append(replica.group_size)
            break

        return mesh_names, group_sizes

    def replica_heartbeat(self, replica_name: str):
        if replica_name in self.policy_status_manager:
            self.policy_status_manager.heartbeat(replica_name)
        elif replica_name in self.rollout_status_manager:
            self.rollout_status_manager.heartbeat(replica_name)
        elif replica_name in self.teacher_result_manager:
            pass
        else:
            logger.error(f"[Controller] Replica {replica_name} not found")

    """
    Life-cycle of controller
    """

    async def register(self, atom: Atom, role: Role):
        async with self.life_cycle_lock:
            if role == Role.POLICY:
                self.policy_status_manager.register(
                    atom, self.config, self.rollout_status_manager
                )
            elif role == Role.ROLLOUT:
                self.rollout_status_manager.register(
                    atom, self.config, self.policy_status_manager
                )
            elif role == Role.REFERENCE:
                self.teacher_result_manager.add(atom.replica_name)
                logger.info(
                    f"[Controller] Registering reference replica {atom.replica_name}"
                )
            else:
                raise Exception(f"[Controller] Unknown role: {role}")

    async def unregister(self, replica_name: str):
        logger.info(f"[Controller] Unregistering replica {replica_name}")
        async with self.life_cycle_lock:
            if replica_name in self.policy_status_manager:
                self.policy_status_manager.unregister(replica_name)
            elif replica_name in self.rollout_status_manager:
                self.rollout_status_manager.unregister(
                    replica_name, self.policy_status_manager
                )
            elif replica_name in self.teacher_result_manager:
                self.teacher_result_manager.remove(replica_name)
                if len(self.teacher_result_manager) > 0:
                    await self.end_reference_replica()
            else:
                raise Exception(f"[Controller] Replica {replica_name} not found")

    async def end_reference_replica(self):
        self.redis_controller.publish_teacher_request(
            {"is_end": True, "prompt_idx": -1, "completion_token_ids": []}, "controller"
        )

    async def set_replica_ncclerror(self, replica_name: str, error: str):
        if replica_name in self.policy_status_manager:
            self.policy_status_manager.set_ncclerror(replica_name, int(time.time()))

            # we use a time window to check nccl report, the last report will invoke post_ncclerror
            self.post_ncclerror_policy_invoke_id += 1
            current_invoke_id = self.post_ncclerror_policy_invoke_id
            await asyncio.sleep(constant.COSMOS_NCCL_ERROR_CLEAN_REPLICA_DELAY)
            if current_invoke_id == self.post_ncclerror_policy_invoke_id:
                # only the latest invoke will trigger the nccl error check
                await self.post_ncclerror(
                    self.policy_status_manager.get_all_policy_report_ncclerror(),
                    Role.POLICY,
                )
                self.policy_status_manager.clear_ncclerror()
        elif replica_name in self.rollout_status_manager:
            raise NotImplementedError(
                f"[Controller] Rollout replica {replica_name} set timeout ack not supported"
            )
        else:
            logger.error(
                f"[Controller] Replica {replica_name} not found in both policy and rollout."
            )

    async def post_ncclerror(
        self, replicas_report_ncclerror: Dict[str, int], role: Role
    ):
        """
        This function is used to clean the hang replicas and trigger the buildmesh command
        """
        all_replicas_ = (
            self.policy_status_manager.policy_replicas
            if role == Role.POLICY
            else self.rollout_status_manager.rollout_replicas
        )
        live_replicas = {rn: all_replicas_[rn] for rn in replicas_report_ncclerror}
        hang_replicas = [
            replica_name
            for replica_name in all_replicas_
            if replica_name not in live_replicas
        ]

        logger.info(f"[Controller] will clean hang replicas: {hang_replicas}")

        if len(live_replicas) == 1:
            # if there is only one replica, it's critical status, we should warning user to scale up the replica
            logger.warning(
                "[Controller] Only one replica is live, it's critical status, user should scale up the replica ASAP!"
            )

        # step 1, manual unregister the hang replicas, we only trigger buildmesh command after update the status
        if role == Role.POLICY:
            for hang_replica in hang_replicas:
                self.policy_status_manager.unregister(hang_replica)
        elif role == Role.ROLLOUT:
            raise NotImplementedError(
                f"[Controller] Rollout replica {hang_replica} set timeout ack not supported"
            )
        else:
            raise Exception(f"[Controller] Unknown role during post_ncclerror: {role}")
