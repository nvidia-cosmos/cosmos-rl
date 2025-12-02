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
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress warning when forking with tokenizers

import threading
import traceback
import torch
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from cosmos_rl.rollout import RolloutWorkerBase, State
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data import RLPayload, IdxAndRLPayload
from cosmos_rl.dispatcher.command import Command, PolicyToRolloutUnicastCommand, RolloutToRolloutBroadcastCommand, BuildMeshCommand
from cosmos_rl.dispatcher.protocol import RolloutRequest, ValidationReportRequest
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.reward.reward_calculator import RewardDispatcher
from cosmos_rl.utils.constant import COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.utils.pynccl import create_nccl_uid, create_nccl_comm, nccl_broadcast

from .vla_rollout import VLARollout
from cosmos_rl.utils.trajectory_buffer import save_trajectory_to_buffer, get_trajectory_buffer


class VLARolloutWorker(RolloutWorkerBase):
    """
    VLA Rollout Worker that integrates with cosmos-rl's distributed training infrastructure.
    
    Extends the base rollout worker to support VLA-specific environment interaction,
    multi-environment parallel execution, and robotic task management.
    """
    
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims) -> None:
        super(VLARolloutWorker, self).__init__(config, parallel_dims)
        
        self.state = State()
        
        # VLA-specific configuration
        self.vla_config = config.vla
        self.task_suite = self.vla_config.task_suite
        self.num_parallel_envs = self.vla_config.num_parallel_envs
        
        # VLA-specific sampling parameters from rollout config
        self.temperature = config.rollout.sampling_config.temperature
        self.n_generation = config.rollout.n_generation
        
        # Command and prompt queues (borrowed from vLLM worker pattern)
        self._command_queue: Queue[Command] = Queue()
        self._prompt_queue: Queue[List[IdxAndRLPayload]] = Queue()
        self.current_weight_version = 0
        
        # Trajectory buffer for filesystem-based storage
        self._trajectory_buffer = get_trajectory_buffer()
        self._rollout_count = 0  # Track rollouts for periodic cleanup
        
        # NCCL communicator infrastructure for r2r broadcast (borrowed from vLLM worker)
        self.global_commnicator_idex = -1  # Index for the global NCCL communicator
        self.rank_in_rollout_repicas = -1  # Rank within rollout replicas
        self.replica_name_to_rank: Dict[str, int] = {}  # Mapping from replica name to rank
        
        # Initialize VLA rollout engine
        self.rollout: VLARollout = VLARollout(self.config, self.tokenizer)
        
        # Environment management
        self.env_pool: Dict[str, Any] = {}
        self.env_thread_pool = ThreadPoolExecutor(max_workers=self.num_parallel_envs * 2)
        
        # Batch processing
        self.batch_size = self.config.rollout.batch_size
        if self.config.validation.enable:
            self.val_batch_size = self.config.validation.batch_size or self.batch_size
        else:
            self.val_batch_size = None
            
        self.background_thread: Optional[threading.Thread] = None
        
        # Validation support
        self.validation_flag = threading.Event()
        self.current_step = 0
        
        # Reward dispatcher with VLA-specific payload processing
        self.reward_dispatcher = RewardDispatcher(
            payload_per_task=COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK
        )
        
        logger.info(f"Initialized VLA rollout worker for task suite: {self.task_suite}")
        logger.info(f"Parallel environments: {self.num_parallel_envs}")
        logger.info(f"VLA sampling: temperature={self.temperature}, n_generation={self.n_generation}")
        
        # Note: Shard info posting will be done after model initialization in lazy_initialize_rollout_engine
        # This is deferred because VLA model is initialized lazily on first weight sync command
    
    def setup(self, dataset=None, reward_fns=None, filter_reward_fns=None, 
              val_dataset=None, val_reward_fns=None, num_workers=8):
        """Setup VLA rollout worker with datasets and reward functions"""
        logger.info("Setting up VLA rollout worker")
        
        # Setup reward dispatcher with VLA-specific configuration
        self.reward_dispatcher.setup(
            config=self.config,
            dataset=dataset,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_dataset=val_dataset,
            val_reward_fns=val_reward_fns,
            data_packer=None,  # VLA doesn't use text data packer
            val_data_packer=None,
            num_workers=num_workers
            if self.parallel_dims.tp_coord[0] == 0
            and (self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1)
            else 0,
        )
        
        logger.info("VLA rollout worker setup completed")
    
    def work(self):
        """Main work method - borrowed from vLLM worker communication pattern"""
        # Start the background thread with daemon=True, so it will exit when the main program exits.
        if self.global_rank == 0:
            # create a thread to query command as a producer (borrowed from vLLM worker)
            self.background_thread = threading.Thread(
                target=self.query_command_from_controller, daemon=True
            )
            self.background_thread.start()

        self.main_loop()
        # Note: VLA doesn't have inference_stream like vLLM
        self.handle_shutdown()
    
    # ==================== Communication Methods (borrowed from vLLM worker) ====================
    
    def query_command_from_controller(self):
        """Background task to check commands from the controller (borrowed from vLLM worker)"""
        while not self.shutdown_signal.is_set():
            commands = []
            try:
                # blocking request
                commands = self.redis_controller.subscribe_command(self.replica_name)
            except Exception as e:
                logger.error(
                    f"[VLA Rollout] Failed in query commands from controller for replica {self.replica_name}\n: {str(e)}"
                )

            for instruction in commands:
                command = Command.depack(instruction)
                logger.debug(f"[VLA Rollout] Received command: {command.command_type}")
                self._command_queue.put(command)

    def request_new_prompts(self, batch_size: int, prompt_queue: Queue, **kwargs):
        """
        Request new prompts from the controller for both training and validation (borrowed from vLLM worker).
        """
        prompts_and_is_end = (None, False)
        if self.global_rank == 0:
            if prompt_queue.empty():
                # blocking request
                payloads, is_end = self.api_client.get_next_prompt(batch_size, **kwargs)
                prompts_and_is_end = (
                    payloads if len(payloads) > 0 else None,
                    is_end,
                )

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end = dist_utils.broadcast_object_cpu(prompts_and_is_end)
        prompts, is_end = prompts_and_is_end
        if prompts is not None:
            prompts = [
                (prompt[0], RLPayload.model_validate(prompt[1])) for prompt in prompts
            ]
            prompt_queue.put(prompts)
        return is_end

    def consume_one_command(self, cmd_pred: Optional[Callable[[Command], bool]] = None):
        """Consume one command from the queue (borrowed from vLLM worker)"""
        current_command = None
        if self.global_rank == 0:
            if not self._command_queue.empty():
                if cmd_pred is None:
                    current_command = self._command_queue.get()
                else:
                    if cmd_pred(self._command_queue.queue[0]):
                        current_command = self._command_queue.get()
                    else:
                        # Do not go on if the command is not expected
                        current_command = None

        current_command = dist_utils.broadcast_object_cpu(current_command)

        if current_command is not None:
            handler = self.get_rollout_command_handler(type(current_command))
            if handler is None:
                raise Exception(
                    f"No such command supported in VLA rollout {current_command}"
                )
            try:
                handler(self, current_command)
                logger.debug(
                    f"[VLA Rollout] Command executed: {current_command._serialize()} for rank: {self.global_rank}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[VLA Rollout] Command execution failed for {current_command._serialize()}"
                ) from e
        return current_command

    def consume_command(
        self,
        cmd_pred: Optional[Callable[[Command], bool]] = None,
        timeout=30.0,  # COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT
    ):
        """Consume all pending commands (borrowed from vLLM worker)"""
        # Consume all pending commands for weight sync.
        # To ensure the weight update is using the up-to-date commands.
        last_cmd = None
        none_cnt = 0
        start_time = time.time()
        while time.time() - start_time < float(timeout):
            cmd = self.consume_one_command(cmd_pred=cmd_pred)
            if cmd is not None:
                last_cmd = cmd
                none_cnt = 0
                start_time = time.time()
            else:
                none_cnt += 1
            if none_cnt >= 5 and (  # COSMOS_ROLLOUT_CMD_WAIT_TIMES
                (
                    last_cmd is not None
                    and not isinstance(last_cmd, PolicyToRolloutUnicastCommand)
                )
                or last_cmd is None
            ):
                # If continuously get None for 5 times, and the last command is not P2R command, we break.
                # Since P2R must be followed by another R2R broadcast command, we need wait.
                break
            time.sleep(0.1)  # COSMOS_ROLLOUT_CMD_WAIT_INTERVAL

    def send_end_signal(self):
        """
        Send end signal to the controller (borrowed from vLLM worker).
        This is used to notify the controller that the rollout worker has finished processing all prompts.
        """
        payloads, is_validation, _, empty = self.report_rollouts(block=True)
        assert (
            not is_validation and payloads is None and empty
        ), f"Payloads must be empty and not for validation when sending end signal {is_validation}, {payloads}, {empty}"
        from cosmos_rl.dispatcher.protocol import RolloutRequest
        response = RolloutRequest(
            src_replica_name=self.replica_name,
            prompt_idxs=[],
            payloads=[],
            completions=[],
            is_end=True,
        )
        logger.info(f"[VLA Rollout] Posting rollout end signal to controller: {response}")
        self.api_client.post_rollout_completion(response)

    def report_rollouts(self, block=False):
        """
        Report rollouts to controller (borrowed from vLLM worker).
        
        This method:
        1. Dequeues processed payloads from reward dispatcher
        2. Sends normal rollout results to controller via RolloutRequest
        3. Returns validation payloads for special handling in do_validation
        4. Supports blocking mode for validation to wait for all batches
        
        Returns:
            tuple: (payloads, is_validation, step, empty)
        """
        while True:
            payloads, is_validation, step, empty = (
                self.reward_dispatcher.dequeue_rewards_cal()
            )
            if payloads is not None:
                if is_validation:
                    # Don't send validation results here - let do_validation handle them
                    break
                response = RolloutRequest(
                    src_replica_name=self.replica_name,
                    prompt_idxs=[],
                    payloads=payloads,
                    is_end=False,
                )
                self.api_client.post_rollout_completion(response)
            elif not block or empty:
                break
        return payloads, is_validation, step, empty
    
    def do_validation(self):
        """
        Perform validation rollouts (adapted from vLLM worker for VLA tasks).
        
        This method orchestrates the validation process:
        1. Request validation prompts in batches
        2. Run VLA rollout generation on validation data
        3. Collect results and send to reward dispatcher
        4. Gather all batched results via report_rollouts
        5. Send ValidationReportRequest to controller with final metrics
        
        The validation uses a separate queue to avoid mixing with training prompts.
        """
        logger.info(f"[VLA Rollout] Starting validation at step {self.current_step}")
        
        validation_queue = Queue()
        prompt_idxs: List[int] = []
        validation_payloads: List[RLPayload] = []
        
        # Fetch and process validation prompts in batches
        while True:
            is_end = self.request_new_prompts(
                self.val_batch_size,
                validation_queue,
                validation_step=self.current_step,
            )
            
            if not validation_queue.empty():
                prompt_id_and_payload_list: List[IdxAndRLPayload] = (
                    validation_queue.get()
                )
                payloads = [p for _, p in prompt_id_and_payload_list]
                
                logger.info(f"[VLA Rollout] Running validation on {len(payloads)} prompts")
                
                # Run VLA rollout generation for validation (greedy, no replication, save videos)
                rollout_results: List[RolloutResult] = self.rollout.rollout_generation(
                    payloads=payloads,
                    n_generation=1,  # No replication for validation
                    is_validation=True,  # Save videos for validation
                    global_steps=self.current_step,
                )
                
                if rollout_results:
                    prompt_idxs.extend([idx for idx, _ in prompt_id_and_payload_list])
                    # Attach completions and metadata to payloads (VLA generates action sequences)
                    for p, rr in zip(payloads, rollout_results):
                        p.completions = rr.completions
                        # Attach VLA-specific metadata from environment_info
                        # This is needed for the validation workaround to compute rewards
                        if hasattr(rr, 'environment_info') and rr.environment_info:
                            if not hasattr(p, 'metadata') or p.metadata is None:
                                p.metadata = {}
                            # Extract key environment results
                            p.metadata.update({
                                'success': rr.environment_info.get('success', False),
                                'episode_length': rr.environment_info.get('episode_length', 0),
                                'task_suite': rr.environment_info.get('task_suite', ''),
                                'total_reward': rr.environment_info.get('total_reward', 0.0),
                            })
                        # Note: VLA doesn't use multi-turn conversations like text LLMs
                    validation_payloads.extend(payloads)

            if is_end:
                break

        # Clear the flag to indicate validation is done
        self.validation_flag.clear()
        logger.info(f"[VLA Rollout] Validation complete. Processed {len(validation_payloads)} payloads")
        
        # Only the last rank in pipeline parallelism should report
        should_report = self.parallel_dims.tp_coord[0] == 0 and (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )

        if should_report:
            # VLA WORKAROUND: Bypass reward_calculator entirely for validation
            # reward_calculator.py has too many LLM/VLM assumptions that don't apply to VLA:
            # - Expects text completions with reference answers
            # - GRPO advantage computation not needed for validation  
            # - Token-level prefix detection irrelevant for environment tasks
            # - Dynamic sampling based on filter_reward not applicable
            #
            # For VLA validation, we just need: environment success → binary reward (1.0/0.0)
            logger.info(f"[VLA Workaround] Computing validation rewards directly (bypassing reward_calculator)")
            
            # Compute rewards directly from environment success
            total_success = 0
            total_episodes = 0
            
            for payload in validation_payloads:
                # Extract success from metadata (if available)
                success = False
                if hasattr(payload, 'metadata') and payload.metadata:
                    success = payload.metadata.get('success', False)
                
                # Compute binary reward: 1.0 for success, 0.0 for failure
                reward = 1.0 if success else 0.0
                
                # Attach rewards and advantages to payload
                # advantages = 0.0 since validation doesn't train
                num_completions = len(payload.completions) if payload.completions else 1
                payload.rewards = [reward] * num_completions
                payload.advantages = [0.0] * num_completions
                payload.valid = True
                
                # Initialize fields required by extract_rollouts
                if not hasattr(payload, 'n_ignore_prefix_tokens') or payload.n_ignore_prefix_tokens is None:
                    payload.n_ignore_prefix_tokens = [0] * num_completions
                if not hasattr(payload, 'completed_conversations') or payload.completed_conversations is None:
                    payload.completed_conversations = [[]] * num_completions
                
                total_success += int(success)
                total_episodes += 1
            
            success_rate = (total_success / total_episodes * 100.0) if total_episodes > 0 else 0.0
            logger.info(
                f"[VLA Workaround] Computed validation rewards: "
                f"{total_success}/{total_episodes} success ({success_rate:.1f}%)"
            )
            
            # Send validation report directly to controller
            # This bypasses reward_dispatcher.dequeue_rewards_cal() loop
            response = ValidationReportRequest(
                src_replica_name=self.replica_name,
                validation_step=self.current_step,
                prompt_idxs=[],
                payloads=validation_payloads,
                is_end=True,
            )
            logger.info(f"[VLA Workaround] Posting validation report: step={self.current_step}, payloads={len(validation_payloads)}")
            self.api_client.post_validation_report(response)
        
        logger.info(f"[VLA Rollout] Validation reporting complete for step {self.current_step}")

    @torch.no_grad()
    def main_loop(self):
        """Main processing loop (adapted from vLLM worker for VLA rollouts)"""
        while not self.shutdown_signal.is_set():
            self.consume_command(cmd_pred=None)
            
            # If weight is not ready, nothing else to do.
            if not self.state.weight_synced():
                continue

            # Check if validation should be performed
            logger.debug(f"[VLA Rollout] Main loop: validation_flag.is_set()={self.validation_flag.is_set()}")
            if self.validation_flag.is_set():
                logger.info("[VLA Rollout] 🎯 Validation flag is set, starting validation")
                self.do_validation()
                # Continue to next iteration after validation
                continue



            # try fetching new prompts if no ending signal is set
            if not self.state.prompt_fetch_end():
                no_more_prompts = self.request_new_prompts(
                    self.batch_size, self._prompt_queue
                )
                if no_more_prompts:
                    logger.info(
                        f"[VLA Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation"
                    )
                    self.state.set_prompt_fetch_end()
                    # Further make sure to set `prompt_consume_end` if no more prompts to be consumed
                    if self._prompt_queue.empty():
                        self.state.set_prompt_consume_end()
                        if self.global_rank == 0:
                            self.send_end_signal()
            _, is_validation, _, _ = self.report_rollouts()
            assert not is_validation, "Validation report should be handled in the broadcast command rather than main loop."
            if self.state.prompt_consume_end():
                assert (
                    self._prompt_queue.empty() and self.state.prompt_fetch_end()
                ), "[VLA Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                continue
            elif self._prompt_queue.empty():
                continue
            else:
                logger.debug(f"[VLA Rollout] generate start for rank {self.global_rank}")

                # Check if the prompt is valid for the current weight version
                first_payload: RLPayload = self._prompt_queue.queue[0][0][1]
                is_valid_prompt_for_current_weight_version = (
                    first_payload.weight_version <= self.current_weight_version
                )

                if not is_valid_prompt_for_current_weight_version:
                    # Fully Synchronized mode is enabled, we need to wait until the weight version is updated
                    continue

                new_prompt_id_and_payload_list: List[IdxAndRLPayload] = (
                    self._prompt_queue.get()
                )
                new_payloads: List[RLPayload] = [
                    payload for _, payload in new_prompt_id_and_payload_list
                ]
                
                # Use VLA rollout generation for training (with GRPO-style replication)
                rollout_results: List[RolloutResult] = self.rollout.rollout_generation(
                    payloads=new_payloads,
                    n_generation=self.config.rollout.n_generation,  # GRPO replication
                    is_validation=False,  # No video saving for training
                    global_steps=self.current_step,
                )

                if len(rollout_results) == 0:
                    continue

                # GRPO filtering happens inside rollout_generation
                # Results only include valid (accepted) groups
                n_gen = self.config.rollout.n_generation
                num_valid_payloads = len(rollout_results) // n_gen if n_gen > 0 else 0
                
                logger.info(
                    f"[GRPO Filter] Got {len(rollout_results)} results from {len(new_payloads)} payloads "
                    f"({num_valid_payloads} valid groups accepted)"
                )

                # Replicate prompt_id_and_payload_list to match actual results
                # Each valid payload contributes n_generation results in sequence
                if n_gen > 1:
                    replicated_prompt_payload_list = []
                    for i in range(num_valid_payloads):
                        idx_and_payload = new_prompt_id_and_payload_list[i]
                        for _ in range(n_gen):
                            replicated_prompt_payload_list.append(idx_and_payload)
                else:
                    replicated_prompt_payload_list = new_prompt_id_and_payload_list[:num_valid_payloads]

                # Process and send results back to controller
                self._send_rollout_results(rollout_results, replicated_prompt_payload_list)
                
                # Periodic cleanup of old trajectory files (every 100 rollouts)
                self._rollout_count += len(rollout_results)
                if self._rollout_count >= 1024:
                    self._trajectory_buffer.cleanup_old_trajectories(max_age_seconds=3600)  # Remove files older than 1 hour
                    stats = self._trajectory_buffer.get_buffer_stats()
                    logger.info(f"[TrajectoryBuffer] Stats: {stats['num_files']} files, {stats['total_size_mb']:.1f} MB")
                    self._rollout_count = 0

            if self.state.prompt_fetch_end() and self._prompt_queue.empty():
                self.state.set_prompt_consume_end()
                if self.global_rank == 0:
                    self.send_end_signal()
        logger.info(f"[VLA Rollout] Main loop of {self.replica_name} finished")
    
    # ==================== Command Handlers (simplified for VLA) ====================
    
    @RolloutWorkerBase.register_rollout_command_handler(BuildMeshCommand)
    def build_global_mesh(self, build_mesh_command: BuildMeshCommand):
        """
        Build global NCCL mesh for rollout-to-rollout communication
        
        This is called when multiple rollout replicas are active and need to sync weights.
        Creates NCCL communicator groups for efficient broadcast between replicas.
        """
        logger.info(f"[VLA Rollout] Building global mesh for {self.replica_name}")
        
        replica_name_to_rank = build_mesh_command.replica_name_to_rank
        if self.replica_name not in replica_name_to_rank:
            raise RuntimeError(
                f"[VLA Rollout] Replica {self.replica_name} not found in registered replicas."
            )
        
        self.rank_in_rollout_repicas = replica_name_to_rank[self.replica_name]
        logger.info(f"[VLA Rollout] My rank in rollout replicas: {self.rank_in_rollout_repicas}")
        
        if len(replica_name_to_rank) == 1:
            logger.info(f"[VLA Rollout] Only one rollout replica, no need to build mesh")
            return
        
        # Generate unique key for NCCL group
        # Creates separate NCCL groups per rank across replicas:
        # group_0: [rank 0 in replica 0, rank 0 in replica 1, ...]
        # group_1: [rank 1 in replica 0, rank 1 in replica 1, ...]
        unique_rollout_group_key = self.get_group_unique_key(replica_name_to_rank)
        logger.debug(f"[VLA Rollout] NCCL group key: {unique_rollout_group_key}")
        
        nccl_group_id = None
        if self.rank_in_rollout_repicas == 0:
            # Only replica_rank == 0 generates the NCCL ID and posts to controller
            logger.info(f"[VLA Rollout] Rank 0 creating NCCL UID and posting to controller...")
            nccl_group_id = create_nccl_uid()
            self.api_client.post_nccl_comm_initiator(
                unique_rollout_group_key, nccl_group_id
            )
            logger.info(f"[VLA Rollout] ✅ NCCL UID posted to controller")
        else:
            # Other replicas query the NCCL group ID from controller
            logger.info(f"[VLA Rollout] Rank {self.rank_in_rollout_repicas} querying NCCL UID from controller...")
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                unique_rollout_group_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[VLA Rollout] Failed to query nccl group_id from controller!"
                )
            logger.info(f"[VLA Rollout] ✅ Got NCCL UID from controller")
        
        # Create NCCL communicator for global mesh
        logger.info(f"[VLA Rollout] Creating NCCL communicator: rank={self.rank_in_rollout_repicas}, world_size={len(replica_name_to_rank)}")
        self.global_commnicator_idex = create_nccl_comm(
            nccl_group_id, self.rank_in_rollout_repicas, len(replica_name_to_rank)
        )
        
        # Cache the replica name to rank mapping
        self.replica_name_to_rank = replica_name_to_rank
        
        logger.info(f"[VLA Rollout] ✅ Global mesh built successfully (communicator index={self.global_commnicator_idex})")
    
    @RolloutWorkerBase.register_rollout_command_handler(PolicyToRolloutUnicastCommand)
    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync VLA model weights from policy to rollout worker via NCCL
        
        Flow:
        1. Initialize VLA model structure (if first sync)
        2. Setup NCCL communicator with policy worker
        3. Receive weights from policy worker via NCCL P2P
        4. Load weights into rollout model
        """
        logger.info(f"[VLA Rollout] Received weight sync command from {command.src_replica_name} -> {command.dst_replica_name}")
        logger.info(f"[VLA Rollout] My replica: {self.replica_name}, Global rank: {self.global_rank}, World size: {self.world_size}")
        logger.info(f"[VLA Rollout] Command src_size: {command.src_replica_size}, dst_size: {command.dst_replica_size}")
        
        # Lazy initialization of the VLA engine
        # Always use "dummy" mode for weight sync - weights will come from policy worker via NCCL
        # This is much faster than loading full weights from HF checkpoint
        if not self.rollout.is_engine_initialized():
            logger.info(f"[VLA Rollout] Initializing VLA engine (dummy mode for weight sync)...")
            import time
            start_time = time.time()
            self.rollout.init_engine(
                quantization="none",  # VLA models don't use quantization
                seed=self.config.rollout.seed,
                load_format="dummy"  # Structure only, weights from policy worker
            )
            elapsed = time.time() - start_time
            logger.info(f"[VLA Rollout] ✅ Engine initialization completed in {elapsed:.2f}s")
        
        # Only process if command is for this replica
        if command.dst_replica_name != self.replica_name:
            logger.info(f"[VLA Rollout] Command not for this replica, returning")
            return
        
        logger.info(f"[VLA Rollout] Processing weight sync for this replica")
        
        # Setup NCCL communicator for policy-to-rollout communication
        nccl_unique_id_key = command.src_replica_name + "_" + command.dst_replica_name
        
        if not hasattr(self, "policy_to_rollout_nccl_communicators"):
            self.policy_to_rollout_nccl_communicators = {}
        
        if nccl_unique_id_key in self.policy_to_rollout_nccl_communicators:
            logger.debug(f"[VLA Rollout] Reusing cached communicator for {nccl_unique_id_key}")
            communicator_index = self.policy_to_rollout_nccl_communicators[nccl_unique_id_key]
        else:
            logger.info(f"[VLA Rollout] Setting up NEW NCCL communicator for {nccl_unique_id_key}")
            # Query NCCL group ID from controller
            logger.info(f"[VLA Rollout] Querying NCCL UID from controller...")
            nccl_group_id = self.query_nccl_unique_id_from_controller(nccl_unique_id_key)
            if nccl_group_id is None:
                raise RuntimeError("[VLA Rollout] Failed to query nccl group_id from controller!")
            logger.info(f"[VLA Rollout] ✅ Got NCCL UID from controller: {nccl_group_id[:4]}...")
            
            # Create NCCL communicator
            from cosmos_rl.utils.pynccl import create_nccl_comm
            my_nccl_rank = self.global_rank + command.src_replica_size
            nccl_world_size = command.src_replica_size + self.world_size
            logger.info(f"[VLA Rollout] Creating NCCL comm: rank={my_nccl_rank}, world_size={nccl_world_size}")
            logger.info(f"[VLA Rollout] Waiting for all {nccl_world_size} ranks to join...")
            
            communicator_index = create_nccl_comm(
                nccl_group_id,
                my_nccl_rank,  # Rollout rank in combined group
                nccl_world_size,  # Total size (policy + rollout)
            )
            self.policy_to_rollout_nccl_communicators[nccl_unique_id_key] = communicator_index
            logger.info(f"[VLA Rollout] ✅ NCCL communicator created (index={communicator_index})")
        
        # Post shard info to controller (if first sync)
        if not hasattr(self, "_shard_info_posted"):
            logger.info(f"[VLA Rollout] Posting shard info to controller...")
            self._post_shard_info_to_controller()
            self._shard_info_posted = True
            logger.info(f"[VLA Rollout] ✅ Shard info posted")
        
        # Fetch receive instructions from controller (if first sync)
        if not hasattr(self, "policy_to_rollout_recv_insts"):
            assert not command.trainable_only, "all params must be transferred at the first time P2R"
            logger.info(f"[VLA Rollout] Fetching receive instructions from controller...")
            self.policy_to_rollout_recv_insts = self.api_client.post_rollout_shard_recv_insts(self.global_rank)
            logger.info(f"[VLA Rollout] ✅ Got {len(self.policy_to_rollout_recv_insts)} instruction groups")
        else:
            assert command.trainable_only, "only trainable params should be transferred at not first time P2R"
        
        # Receive weights from policy worker using instructions
        # ALWAYS verify first sync to ensure NCCL transfer is working correctly
        do_verification = (self.current_weight_version == 0)
        if do_verification:
            logger.info(f"[VLA Rollout] First weight sync - verification ENABLED (compulsory)")
        self._receive_weights_from_policy(communicator_index, command.trainable_only, do_verification)
        
        # Mark weight as synced
        self.state.set_weight_synced()
        
        # Update weight version and check for validation
        current_step = command.weight_step if hasattr(command, 'weight_step') else self.current_weight_version + 1
        
        if hasattr(command, 'weight_step') and command.weight_step is not None:
            self.current_weight_version = command.weight_step
            logger.info(f"[VLA Rollout] Updated weight version to {current_step} (from command)")
        else:
            self.current_weight_version += 1
            logger.info(f"[VLA Rollout] Updated weight version to {self.current_weight_version} (incremental)")
            current_step = self.current_weight_version
        
        # Handle validation flag (if enabled) - IMPORTANT: Must check here too!
        # If there's only 1 rollout replica, R2R broadcast won't happen, so we need to check validation here
        total_steps = command.total_steps if hasattr(command, 'total_steps') else None
        logger.info(f"[VLA Rollout] Validation check in P2R: current_step={current_step}, "
                   f"validation.enable={self.config.validation.enable}, "
                   f"validation.val_before_train={self.config.validation.val_before_train}, "
                   f"validation.freq={self.config.validation.freq}, "
                   f"total_steps={total_steps}")
        
        if current_step is not None and current_step > 0:
            # Check if validation should be triggered
            is_initial_validation = (current_step == 1 and self.config.validation.val_before_train)
            is_periodic_validation = (current_step > 1 and current_step % self.config.validation.freq == 0)
            is_final_validation = (total_steps is not None and current_step == total_steps)
            
            should_do_validation = self.config.validation.enable and (
                is_initial_validation 
                or is_periodic_validation
                or is_final_validation
            )
            
            logger.info(f"[VLA Rollout] should_do_validation={should_do_validation} "
                       f"(val_before_train={is_initial_validation}, periodic={is_periodic_validation}, final={is_final_validation})")
            
            if should_do_validation:
                self.current_step = current_step
                # Set validation flag for main loop
                if hasattr(self, 'validation_flag'):
                    self.validation_flag.set()
                    logger.info(f"[VLA Rollout] ✅ Validation flag SET for step {current_step} (via P2R)")
                else:
                    logger.warning(f"[VLA Rollout] ⚠️ validation_flag attribute not found!")
        else:
            logger.info(f"[VLA Rollout] Skipping validation check (current_step={current_step})")
        
        logger.info(f"[VLA Rollout] ✅ Weight sync completed, version: {self.current_weight_version}")
    
    def _receive_weights_from_policy(self, communicator_index: int, trainable_only: bool, do_verification: bool = False):
        """
        Receive model weights from policy worker via NCCL using instruction-based coordination
        
        Args:
            communicator_index: NCCL communicator index
            trainable_only: If True, only sync trainable parameters (incremental update)
            do_verification: If True, verify received weights against HF checkpoint
        """
        import time
        from cosmos_rl.utils.pynccl import nccl_recv
        
        logger.info(f"[VLA Rollout] Starting instruction-based weight receive (trainable_only={trainable_only}, verification={do_verification})...")
        
        # Load reference weights if verification is enabled
        reference_weights = {}
        if do_verification:
            logger.info(f"[VLA Rollout] Loading reference weights from HF checkpoint for verification...")
            reference_weights = self._load_reference_weights()
            logger.info(f"[VLA Rollout] ✅ Loaded {len(reference_weights)} reference parameters")
        
        # Get model parameters dict
        model = self.rollout.module
        model_params = dict(model.named_parameters())
        
        # Prepare trainable params list if needed
        if trainable_only:
            if not hasattr(self, 'trainable_params'):
                logger.info(f"[VLA Rollout] Fetching trainable params from controller...")
                self.trainable_params = set(self.api_client.get_trainable_params() if self.global_rank == 0 else [])
                # Broadcast to all ranks
                from cosmos_rl.utils import distributed as dist_utils
                self.trainable_params = dist_utils.broadcast_object_cpu(self.trainable_params)
                logger.info(f"[VLA Rollout] Got {len(self.trainable_params)} trainable params")
        
        st = time.time()
        total_bytes_received = 0
        transferred_params = 0
        skipped_params = 0

        def recv_tensor_creator(vllm_tensor_view: torch.Tensor):
            recv_tensor = None
            inplace = True

            if vllm_tensor_view.is_contiguous():
                recv_tensor = vllm_tensor_view
            else:
                # new a temp tensor
                recv_tensor = torch.empty_like(vllm_tensor_view).contiguous()
                inplace = False

            # if vllm_tensor_view.dtype != target_dtype:
            #     recv_tensor = recv_tensor.to(target_dtype)
            #     inplace = False
            # Hold these recv_tensor, in case of buffer reusing by torch
            # self.total_temp_tensor_pool.append(recv_tensor)

            return recv_tensor, inplace
        
        # Create inference stream for async communication
        if not hasattr(self, 'inference_stream'):
            self.inference_stream = torch.cuda.Stream()
        
        with torch.cuda.stream(self.inference_stream):
            # Process each instruction group
            for insts_group in self.policy_to_rollout_recv_insts:
                # Process each parameter in this group
                for insts_for_per_param in insts_group.param_instructions:
                    param_name = insts_for_per_param.param_name
                    
                    # Skip if trainable_only and param is not trainable
                    if trainable_only and param_name not in self.trainable_params:
                        logger.debug(f"[VLA Rollout] Skip {param_name} (not trainable)")
                        skipped_params += 1
                        continue
                    
                    # Get the parameter tensor
                    if param_name not in model_params:
                        logger.warning(f"[VLA Rollout] Parameter {param_name} not found in model, skipping")
                        continue
                    
                    target_param = model_params[param_name]
                    transferred_params += 1
                    
                    # Process each instruction for this parameter
                    for inst in insts_for_per_param.instructions:
                        p_rank = inst.policy_rank  # Policy rank to receive from
                        r_rank = inst.rollout_rank  # Rollout rank (should match self.global_rank)
                        
                        if r_rank != self.global_rank:
                            continue  # This instruction is for a different rollout rank

                        # For VLA models (no model parallelism), we expect full tensors
                        # The slice strategy should be empty or indicate full tensor
                        recv_tensor = target_param.data if target_param.data.is_contiguous() else target_param.data.contiguous()
                        
                        tensor_split_strategys = inst.slice_strategy
                        # assert r_rank == global_rank_of_rollout
                        vllm_tensor_view = recv_tensor.cosmos_slice(tensor_split_strategys)
                        recv_tensor, inplace = recv_tensor_creator(vllm_tensor_view)

                        # TODO: need none-inplace support
                        assert inplace, "recv_tensor_creator should return inplace tensor"
                        logger.debug(
                            f"[Rollout] Recving tensor {param_name} from policy rank {p_rank} to rollout rank {r_rank}, shape {vllm_tensor_view.shape} of {recv_tensor.shape} with dtype {vllm_tensor_view.dtype}."
                        )
                        nccl_recv(recv_tensor, p_rank, communicator_index)
                        # inplace copy
                        # if not inplace:
                        #     all_tensor_views_to_copy.append(
                        #         (vllm_tensor_view, recv_tensor, inst_dest_name)
                        #     )
                        total_bytes_received += recv_tensor.numel() * recv_tensor.element_size()
            
            # Synchronize to ensure all receives complete
            self.inference_stream.synchronize()
        
        elapsed = time.time() - st
        throughput_gbps = (total_bytes_received / elapsed) / (1024 ** 3) if elapsed > 0 else 0
        
        logger.info(
            f"[VLA Rollout] ✅ Received {transferred_params} parameters (skipped {skipped_params}) "
            f"({total_bytes_received / (1024**2):.2f} MB) in {elapsed:.2f}s "
            f"({throughput_gbps:.2f} GB/s)"
        )
        
        # Verify weights if enabled
        if do_verification and reference_weights:
            logger.info(f"[VLA Rollout] Starting weight verification...")
            self._verify_weights(model_params, reference_weights, trainable_only)
            logger.info(f"[VLA Rollout] ✅ Weight verification passed!")
            del reference_weights
            torch.cuda.empty_cache()
    
    def _post_shard_info_to_controller(self):
        """
        Post shard information to controller for VLA models
        
        Uses the same parameter collection method as policy (weight_sync_transforms)
        to ensure proper matching during weight synchronization.
        """
        from cosmos_rl.utils import distributed as dist_utils
        from cosmos_rl.utils.parallelism_map import ParallelTopoMapperGroup
        
        # Use weight_sync_transforms to match policy worker's parameter collection
        vla_model = self.rollout.vla_model
        keys_n_ranks = []
        
        for name, tensor_or_callable in vla_model.weight_sync_transforms:
            if isinstance(tensor_or_callable, torch.Tensor):
                keys_n_ranks.append((name, tensor_or_callable.ndim))
            else:
                from collections.abc import Callable
                assert isinstance(tensor_or_callable, Callable)
                tensor = tensor_or_callable()
                keys_n_ranks.append((name, tensor.ndim))
        
        # Get hf_config and weight_mapper from VLA model
        hf_config = self.rollout.hf_config
        weight_mapper = vla_model.weight_mapper
        
        # Prepare local shard infos using ParallelTopoMapperGroup
        local_shard_infos = ParallelTopoMapperGroup(
            self.parallel_dims,
            hf_config,
            is_policy=False,
            underlying_model=vla_model.model,  # Use the inner OpenVLAForActionPrediction model
            backend="vllm",  # Use vLLM backend conventions
            weight_mapper=weight_mapper,
        ).prepare_local_shard_infos(keys_n_ranks, self.global_rank)
        
        # Gather shard info from all ranks
        all_rank_shard_infos = dist_utils.all_gather_object_cpu(local_shard_infos)
        
        # Create sorted param list (for matching with policy)
        sorted_params = sorted([x[0] for x in keys_n_ranks])
        sorted_params_all_rank = dist_utils.all_gather_object_cpu(sorted_params)
        
        # Only rank 0 posts to controller
        if self.global_rank == 0:
            logger.info(f"[VLA Rollout] Rank 0 posting shard info for {len(keys_n_ranks)} parameters")
            self.api_client.post_rollout_shard_info(
                shard_infos=all_rank_shard_infos,
                param_groups=[],  # No grouped parameters for VLA
                sorted_params=sorted_params_all_rank,
            )
            logger.info(f"[VLA Rollout] ✅ Shard info posted to controller")
    
    def query_nccl_unique_id_from_controller(self, unique_id_key: str):
        """
        Query NCCL unique ID from controller for P2P communication setup
        
        This is needed for establishing NCCL communicators between policy and rollout workers.
        """
        return self.api_client.post_nccl_comm_acceptor(unique_id_key)
    
    def _load_reference_weights(self) -> Dict[str, torch.Tensor]:
        """
        Load reference weights from HuggingFace checkpoint for verification
        
        Returns:
            Dict mapping parameter names to reference weight tensors
        """
        from cosmos_rl.policy.model.vla import VLAModel, VLAArgs
        from cosmos_rl.policy.model.vla_utils import create_vla_config
        
        model_path = self.config.policy.model_name_or_path
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        logger.info(f"[VLA Rollout] Loading reference model from {model_path}...")
        
        # Create VLA config
        vla_config, _, _ = create_vla_config(
            model_path,
            cosmos_config=self.config,
            model=self.config.vla.vla_type
        )
        
        # Create VLA args
        vla_args = VLAArgs(
            vla_type=self.config.vla.vla_type,
            use_proprio=self.config.vla.use_proprio,
            proprio_dim=self.config.vla.action_dim,
            num_images_in_input=self.config.vla.num_images_in_input,
            hf_config=vla_config
        )
        
        # Load full model with weights from HF
        reference_model = VLAModel.from_model_args(vla_args)
        reference_model.load_from_checkpoint(
            model_name_or_path=model_path,
            parallel_dims=None,
            device=device
        )
        
        # Extract weights
        reference_weights = {
            name: param.data.clone()
            for name, param in reference_model.model.named_parameters()
        }
        
        # Clean up reference model
        del reference_model
        torch.cuda.empty_cache()
        
        logger.info(f"[VLA Rollout] ✅ Loaded {len(reference_weights)} reference weights")
        return reference_weights
    
    def _verify_weights(
        self, 
        model_params: Dict[str, torch.nn.Parameter], 
        reference_weights: Dict[str, torch.Tensor],
        trainable_only: bool
    ):
        """
        Verify that received weights match reference weights from HF checkpoint
        
        Note: Vision backbone parameters are loaded directly from checkpoint (not synced),
        so they are excluded from this verification.
        
        Args:
            model_params: Current model parameters (after NCCL sync)
            reference_weights: Reference weights from HF checkpoint
            trainable_only: Whether only trainable params were synced
        
        Raises:
            ValueError: If weights don't match
        """
        mismatches = []
        total_verified = 0
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        
        # Get trainable params if needed
        if trainable_only:
            trainable_params = self.trainable_params
        else:
            trainable_params = set(model_params.keys())
        
        for param_name in trainable_params:
            if param_name not in model_params:
                continue
            
            if param_name not in reference_weights:
                logger.warning(f"[VLA Rollout] Parameter {param_name} not in reference weights, skipping")
                continue
            
            received_weight = model_params[param_name].data
            reference_weight = reference_weights[param_name]
            
            # Compare shapes
            if received_weight.shape != reference_weight.shape:
                mismatches.append(
                    f"{param_name}: shape mismatch (received {received_weight.shape} vs reference {reference_weight.shape})"
                )
                continue
            
            # Compare values
            try:
                # Use allclose with reasonable tolerances
                if not torch.allclose(received_weight, reference_weight, rtol=1e-5, atol=1e-7):
                    abs_diff = (received_weight - reference_weight).abs().max().item()
                    rel_diff = ((received_weight - reference_weight).abs() / (reference_weight.abs() + 1e-8)).max().item()
                    max_abs_diff = max(max_abs_diff, abs_diff)
                    max_rel_diff = max(max_rel_diff, rel_diff)
                    
                    mismatches.append(
                        f"{param_name}: value mismatch (max_abs_diff={abs_diff:.2e}, max_rel_diff={rel_diff:.2e})"
                    )
                else:
                    total_verified += 1
            except Exception as e:
                mismatches.append(f"{param_name}: comparison failed ({e})")
        
        # Report results
        logger.info(f"[VLA Rollout] Verification results:")
        logger.info(f"  ✅ Matched: {total_verified} parameters")
        if len(mismatches) > 0:
            logger.info(f"  ❌ Mismatched: {len(mismatches)} parameters")
        if mismatches:
            logger.info(f"  Max absolute difference: {max_abs_diff:.2e}")
            logger.info(f"  Max relative difference: {max_rel_diff:.2e}")
            logger.info(f"  First 10 mismatches:")
            for mismatch in mismatches[:10]:
                logger.info(f"    - {mismatch}")
        
        # Raise error if there are mismatches
        if mismatches:
            raise ValueError(
                f"Weight verification failed: {len(mismatches)} parameters don't match reference weights. "
                f"This indicates an issue with NCCL weight synchronization."
            )

    @RolloutWorkerBase.register_rollout_command_handler(RolloutToRolloutBroadcastCommand)
    def broadcast_to_all_rollout_replica(self, broadcast_command: RolloutToRolloutBroadcastCommand):
        """
        Broadcast VLA model weights to all other rollout replicas via NCCL
        
        This happens after PolicyToRolloutUnicast updates weights in replica 0.
        Replica 0 broadcasts to all other replicas for parallel rollout.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        dst_replica_names: List[str] = broadcast_command.dst_replica_names
        
        logger.info(f"[VLA Rollout] Received R2R broadcast command from {src_replica_name} to {len(dst_replica_names)} replicas")
        logger.info(f"[VLA Rollout] My replica: {self.replica_name}, trainable_only={broadcast_command.trainable_only}")
        
        # Lazy initialize engine for non-src replicas  
        # They don't receive P2R, so initialize with dummy weights for r2r broadcast
        if self.replica_name != src_replica_name:
            if not self.rollout.is_engine_initialized():
                logger.info(f"[VLA Rollout] Non-src replica initializing engine with dummy weights...")
                self.rollout.init_engine(
                    quantization="none",
                    seed=self.config.rollout.seed,
                    load_format="dummy"
                )
                logger.info(f"[VLA Rollout] ✅ Engine initialized")
        
        # Only do broadcast if there are multiple replicas
        if len(dst_replica_names) > 1:
            self._prepare_trainable_params()
            skipped_params_cnt = 0
            transferred_params_cnt = 0
            logger.info("[VLA Rollout] Starting broadcasting of parameters to all replicas...")
            
            # Create inference stream if not exists
            if not hasattr(self, 'inference_stream'):
                self.inference_stream = torch.cuda.Stream()
            
            with torch.cuda.stream(self.inference_stream):
                assert (
                    self.rank_in_rollout_repicas >= 0
                ), "[VLA Rollout] rank_in_rollout_repicas should be set before broadcast (build_global_mesh not called?)"
                assert (
                    len(dst_replica_names) == len(self.replica_name_to_rank)
                ), f"[VLA Rollout] dst replicas count {len(dst_replica_names)} must match replica_name_to_rank count {len(self.replica_name_to_rank)}"
                
                src_rank = self.replica_name_to_rank[src_replica_name]
                logger.info(f"[VLA Rollout] Broadcasting from rank {src_rank} using communicator {self.global_commnicator_idex}")
                
                with torch.inference_mode():
                    # Get all model parameters
                    model_params = dict(self.rollout.module.named_parameters())
                    
                    for name, parameter in model_params.items():
                        # Skip non-trainable params if trainable_only is set
                        if (
                            name not in self.trainable_params
                            and broadcast_command.trainable_only
                        ):
                            logger.debug(f"[VLA Rollout] Skip {name} in R2R due to non-trainable")
                            skipped_params_cnt += 1
                            continue
                        
                        transferred_params_cnt += 1
                        
                        # Ensure parameter is contiguous for NCCL
                        recv_tensor = parameter
                        if not parameter.is_contiguous():
                            recv_tensor = parameter.contiguous()
                        
                        # Broadcast parameter across all replicas
                        nccl_broadcast(
                            recv_tensor, src_rank, self.global_commnicator_idex
                        )
                        
                        # Copy back if we made it contiguous
                        if not parameter.is_contiguous():
                            parameter.copy_(recv_tensor)
                    
                    # Mark weight as synced on first broadcast
                    if not self.state.weight_synced():
                        assert not broadcast_command.trainable_only, "[VLA Rollout] Trainable only must be False for first broadcast"
                        self.state.set_weight_synced()
            
            logger.info(
                f"[VLA Rollout] ✅ Finished broadcasting. Skipped {skipped_params_cnt} non-trainable params, transferred {transferred_params_cnt} params"
            )
        
        # Update weight version
        current_step = broadcast_command.weight_step
        if current_step is not None:
            assert (
                current_step >= self.current_weight_version
            ), f"current_step: {current_step} must be >= self.current_weight_version: {self.current_weight_version}"
            self.current_weight_version = current_step
            logger.info(f"[VLA Rollout] Updated weight version to {current_step}")
        elif self.current_weight_version == 0:
            # Initial validation, set weight version to 1
            self.current_weight_version = 1
            current_step = 1

        # Handle validation flag (if enabled)
        logger.info(f"[VLA Rollout] Validation check: current_step={current_step}, "
                   f"validation.enable={self.config.validation.enable}, "
                   f"validation.val_before_train={self.config.validation.val_before_train}, "
                   f"validation.freq={self.config.validation.freq}, "
                   f"total_steps={broadcast_command.total_steps}")
        
        if current_step is not None and current_step > 0:
            # Check if validation should be triggered
            is_initial_validation = (current_step == 1 and self.config.validation.val_before_train)
            is_periodic_validation = (current_step > 1 and current_step % self.config.validation.freq == 0)
            is_final_validation = (current_step == broadcast_command.total_steps)
            
            should_do_validation = self.config.validation.enable and (
                is_initial_validation 
                or is_periodic_validation
                or is_final_validation
            )
            
            logger.info(f"[VLA Rollout] should_do_validation={should_do_validation} "
                       f"(val_before_train={is_initial_validation}, periodic={is_periodic_validation}, final={is_final_validation})")
            
            if should_do_validation:
                self.current_step = current_step
                # Set validation flag for main loop
                if hasattr(self, 'validation_flag'):
                    self.validation_flag.set()
                    logger.info(f"[VLA Rollout] ✅ Validation flag SET for step {current_step}")
                else:
                    logger.warning(f"[VLA Rollout] ⚠️ validation_flag attribute not found!")
        else:
            logger.info(f"[VLA Rollout] Skipping validation check (current_step={current_step})")
        
        # Handle shutdown signal
        if broadcast_command.replica_should_stop():
            logger.info(f"[VLA Rollout] Shutdown signal received")
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()
    
    def _prepare_trainable_params(self):
        """
        Prepare the list of trainable parameters for R2R broadcast
        
        Queries from controller on rank 0 and broadcasts to all ranks
        """
        if not hasattr(self, "trainable_params"):
            if self.global_rank == 0:
                logger.info(f"[VLA Rollout] Rank 0 fetching trainable params from controller...")
                self.trainable_params = set(self.api_client.get_trainable_params())
                logger.info(f"[VLA Rollout] Got {len(self.trainable_params)} trainable params")
            else:
                self.trainable_params = set()
            
            # Broadcast trainable params list to all ranks
            self.trainable_params = dist_utils.broadcast_object_cpu(self.trainable_params)
            logger.debug(f"[VLA Rollout] Rank {self.global_rank} has {len(self.trainable_params)} trainable params")
    
    def _send_rollout_results(self, results: List[RolloutResult], prompt_id_and_payload_list: List[IdxAndRLPayload]):
        """
        Send rollout results back to controller.
        
        VLA WORKAROUND: Bypasses reward_calculator entirely and computes rewards directly.
        reward_calculator.py has too many LLM/VLM assumptions that don't apply to VLA.
        For VLA training, we compute: environment success → binary reward (1.0/0.0)
        
        **GRPO Group Normalization**: Following SimpleVLA-RL's approach, advantages are normalized
        within each task group (prompt_idx), not globally. This ensures fair learning signals
        across tasks of varying difficulty.
        """
        logger.debug(f"Sending {len(results)} VLA rollout results to controller")
        
        try:
            # Step 1: Extract rewards and group by prompt_idx (task)
            from collections import defaultdict
            import numpy as np
            
            rewards_by_task = defaultdict(list)  # prompt_idx -> list of rewards
            result_info = []  # [(result, idx_and_payload, reward, result_idx), ...]
            
            for result_idx, (result, idx_and_payload) in enumerate(zip(results, prompt_id_and_payload_list)):
                # Unpack tuple: IdxAndRLPayload = Tuple[int, RLPayload]
                prompt_idx, payload = idx_and_payload
                
                # Extract success directly from environment_info
                env_info = result.environment_info or {}
                success = env_info.get('success', False)
                reward = 1.0 if success else 0.0
                
                rewards_by_task[prompt_idx].append(reward)
                result_info.append((result, idx_and_payload, reward, result_idx))
            
            # Step 2: Compute group-based advantages (SimpleVLA-RL style)
            # For each task: advantage = (reward - group_mean) / (group_std + epsilon)
            advantages_flat = []
            epsilon = 1e-6
            
            for result, idx_and_payload, reward, result_idx in result_info:
                prompt_idx, payload = idx_and_payload
                task_rewards = np.array(rewards_by_task[prompt_idx], dtype=np.float32)
                
                # Group normalization (per-task)
                if len(task_rewards) > 1:
                    group_mean = task_rewards.mean()
                    group_std = task_rewards.std()
                    advantage = (reward - group_mean) / (group_std + epsilon)
                else:
                    # Single rollout per task: advantage = 0 (matches SimpleVLA-RL)
                    advantage = 0.0
                
                advantages_flat.append(advantage)
            
            # Step 3: Build result payloads with normalized advantages
            result_payloads = []
            
            for (result, idx_and_payload, reward, result_idx), advantage in zip(result_info, advantages_flat):
                prompt_idx, payload = idx_and_payload
                
                # Extract environment info for this result
                env_info = result.environment_info or {}
                success = env_info.get('success', False)
                
                # Build metadata dict directly (avoid intermediate VLAEpisodeMetadata object)
                metadata_dict = {
                    'success': bool(success),  # Convert numpy.bool_ to Python bool
                    'task_suite': str(env_info.get('task_suite', '')),
                    'num_actions': int(env_info.get('num_actions', 0)),  # Convert numpy int to Python int
                    'prompt_id': int(prompt_idx),
                    'temperature': float(self.temperature),
                    'n_generation': int(self.n_generation),
                    'episode_length': result.episode_length,
                    'task_id': env_info.get('task_id', 0),
                    'trial_id': env_info.get('trial_id', 0),
                    'gen_idx': env_info.get('gen_idx', 0),
                    'weight_version': self.current_weight_version if self.current_weight_version > 1 else 0
                }
                
                # Add trajectory data if available (from dedicated vla_trajectory field)
                if hasattr(result, 'vla_trajectory') and result.vla_trajectory:
                    # Save full trajectory (including pixel_values) to filesystem
                    trajectory_id = save_trajectory_to_buffer(result.vla_trajectory)
                    
                    # Only send the small trajectory_id via HTTP (not the actual data!)
                    metadata_dict['trajectory_id'] = trajectory_id
                    
                    # Optionally include lightweight metadata for debugging
                    # Ensure all values are JSON-serializable Python native types
                    input_ids = result.vla_trajectory.get('input_ids', [])
                    num_steps = len(input_ids) if hasattr(input_ids, '__len__') else 0
                    metadata_dict['trajectory_stats'] = {
                        'num_steps': int(num_steps),
                        'has_pixel_values': bool('pixel_values' in result.vla_trajectory),
                        'has_responses': bool('responses' in result.vla_trajectory),
                    }
                
                # Create result payload with pre-computed rewards and NORMALIZED advantages
                result_payload = RLPayload(
                    prompt=result.prompt,
                    conversation=getattr(payload, 'conversation', None),
                    weight_version=payload.weight_version,
                    temperature=self.temperature,
                    
                    # VLA-specific results (completions for n_generation support)
                    completions=result.completions if result.completions else [""],
                    
                    # VLA metadata (includes trajectory data for training)
                    metadata=metadata_dict,
                    
                    # Pre-computed rewards and advantages (bypass reward_calculator)
                    rewards=[float(reward)],  # Ensure Python float
                    advantages=[float(advantage)],  # **GROUP-NORMALIZED** advantage (SimpleVLA-RL style)
                    valid=True,
                    
                    # Fields required by extract_rollouts
                    n_ignore_prefix_tokens=[0],
                    completed_conversations=[[]],
                    
                    # VLA doesn't have reference answers
                    reference_answer=None,
                )
                result_payloads.append(result_payload)
            
            # Send training results directly to controller (bypass reward_dispatcher)
            prompt_idxs = [int(payload.metadata.get('prompt_id', 0)) for payload in result_payloads]
            
            response = RolloutRequest(
                src_replica_name=self.replica_name,
                prompt_idxs=prompt_idxs,
                payloads=result_payloads,
                is_end=False,
            )
            self.api_client.post_rollout_completion(response)
            logger.info(f"[VLA Rollout] Sent {len(result_payloads)} training rollout results directly to controller")
            
        except Exception as e:
            logger.error(f"Error sending rollout results: {e}")
            traceback.print_exc()
    
