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
import threading
import traceback
import torch
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from cosmos_rl.rollout import RolloutWorkerBase, State
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data import RLPayload, IdxAndRLPayload
from cosmos_rl.dispatcher.command import Command, PolicyToRolloutUnicastCommand, RolloutToRolloutBroadcastCommand
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.reward.reward_calculator import RewardDispatcher
from cosmos_rl.utils.constant import COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK
import cosmos_rl.utils.distributed as dist_utils

from .vla_rollout import VLARollout
from .environment_wrappers import LiberoEnvWrapper, RobotwinEnvWrapper


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
        """Report rollouts to controller (borrowed from vLLM worker)"""
        while True:
            payloads, is_validation, step, empty = (
                self.reward_dispatcher.dequeue_rewards_cal()
            )
            if payloads is not None:
                if is_validation:
                    break
                from cosmos_rl.dispatcher.protocol import RolloutRequest
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

    @torch.no_grad()
    def main_loop(self):
        """Main processing loop (adapted from vLLM worker for VLA rollouts)"""
        while not self.shutdown_signal.is_set():
            self.consume_command(cmd_pred=None)
            # Note: VLA doesn't have validation flag like vLLM for now
            # if self.validation_flag.is_set():
            #     self.do_validation()

            # If weight is not ready, nothing else to do.
            if not self.state.weight_synced():
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

                prompt_id_and_payload_list: List[IdxAndRLPayload] = (
                    self._prompt_queue.get()
                )
                payloads: List[RLPayload] = [
                    payload for _, payload in prompt_id_and_payload_list
                ]

                # Use VLA rollout generation instead of vLLM
                rollout_results: List[RolloutResult] = self.rollout.rollout_generation(
                    payloads=payloads
                )

                if len(rollout_results) == 0:
                    continue

                assert len(rollout_results) == len(
                    payloads
                ), f"Error: VLA returned {len(rollout_results)} for {len(payloads)}"

                # Process and send results back to controller
                self._send_rollout_results(rollout_results, prompt_id_and_payload_list)

            if self.state.prompt_fetch_end() and self._prompt_queue.empty():
                self.state.set_prompt_consume_end()
                if self.global_rank == 0:
                    self.send_end_signal()
        logger.info(f"[VLA Rollout] Main loop of {self.replica_name} finished")
    
    # ==================== Command Handlers (simplified for VLA) ====================
    
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
        self.current_weight_version += 1
        
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
                        
                        logger.debug(
                            f"[VLA Rollout] Receiving {param_name} from policy rank {p_rank}, "
                            f"shape {recv_tensor.shape}, dtype {recv_tensor.dtype}"
                        )
                        
                        # Receive from the correct policy rank
                        nccl_recv(
                            recv_tensor,
                            p_rank,  # Receive from this policy rank
                            communicator_index,
                            stream=self.inference_stream
                        )
                        
                        # Copy back if we created a contiguous copy
                        if recv_tensor is not target_param.data:
                            target_param.data.copy_(recv_tensor)
                        
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
            
            # Skip vision backbone parameters - they're loaded directly from checkpoint, not synced via NCCL
            if param_name.startswith("vision_backbone"):
                logger.debug(f"[VLA Rollout] Skipping vision_backbone parameter {param_name} (loaded from checkpoint)")
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
        Simplified rollout-to-rollout broadcast for VLA models (adapted from vLLM worker)
        """
        logger.info(f"[VLA Rollout] Received broadcast command from {broadcast_command.src_replica_name}")
        
        # For VLA models, we'll implement simplified broadcast handling
        # TODO: Implement actual VLA model weight broadcasting between replicas
        
        current_step = broadcast_command.weight_step
        if current_step is not None:
            assert (
                current_step >= self.current_weight_version
            ), f"current_step: {current_step} must be greater than or equal to self.current_weight_version: {self.current_weight_version}"
            self.current_weight_version = current_step

        if not self.state.weight_synced():
            self.state.set_weight_synced()

        logger.info(f"[VLA Rollout] Broadcast handling completed for step {current_step}")

        if broadcast_command.replica_should_stop():
            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()
    
    def _send_rollout_results(self, results: List[RolloutResult], prompt_id_and_payload_list: List[IdxAndRLPayload]):
        """Send rollout results back to controller via reward dispatcher"""
        logger.debug(f"Sending {len(results)} VLA rollout results to controller")
        
        try:
            # Convert RolloutResult objects to RLPayload objects for reward calculation
            result_payloads = []
            
            for result, idx_and_payload in zip(results, prompt_id_and_payload_list):
                # Create modified payload with VLA results
                result_payload = RLPayload(
                    prompt=result.prompt,
                    conversation=getattr(idx_and_payload.payload, 'conversation', None),
                    weight_version=idx_and_payload.payload.weight_version,
                    temperature=self.temperature,
                    
                    # VLA-specific results
                    completion=result.completions[0] if result.completions else "",
                    log_prob=float(result.log_probs[0][0]) if result.log_probs and result.log_probs[0] else 0.0,
                    
                    # Additional VLA metadata  
                    metadata={
                        'vla_episode_length': result.environment_info.get('episode_length', 0),
                        'vla_success': result.environment_info.get('success', False),
                        'vla_reward': result.environment_info.get('total_reward', 0.0),
                        'task_suite': result.environment_info.get('task_suite', self.task_suite),
                        'task_id': result.environment_info.get('task_id', 0),
                        'trial_id': result.environment_info.get('trial_id', 0),
                        'prompt_id': idx_and_payload.idx,
                        'temperature': self.temperature,
                        'n_generation': self.n_generation
                    }
                )
                result_payloads.append(result_payload)
            
            # Send to reward dispatcher for processing
            # The reward dispatcher will compute final rewards and send to controller
            prompt_idxs = [int(payload.metadata.get('prompt_id', 0)) for payload in result_payloads]
            
            self.reward_dispatcher.enqueue_rewards_cal(
                payloads=result_payloads, 
                is_validation=False, 
                step=0,  # TODO: Get actual step from training
                prompt_idxs=prompt_idxs
            )
            
            logger.info(f"Successfully queued {len(result_payloads)} VLA rollout results for reward calculation")
            
        except Exception as e:
            logger.error(f"Error sending rollout results: {e}")
            traceback.print_exc()
    
    def _extract_libero_task_info(self, payload: RLPayload) -> Dict[str, Any]:
        """
        Extract LIBERO task information from RLPayload
        
        The payload should contain LIBERO dataset format:
        - task_suite_name: str (e.g., "libero_10")
        - task_id: tensor or int (e.g., 0-9 for libero_10)  
        - trial_id: tensor or int (e.g., 0-49)
        - trial_seed: tensor or int (default -1)
        
        Args:
            payload: RLPayload containing task information
            
        Returns:
            Dict with task configuration
        """
        try:
            # Try to extract from payload metadata first
            if hasattr(payload, 'metadata') and payload.metadata:
                # Check if task info is in metadata
                if 'task_suite_name' in payload.metadata:
                    task_suite_name = payload.metadata['task_suite_name']
                    task_id = payload.metadata.get('task_id', 0)
                    trial_id = payload.metadata.get('trial_id', 0)
                    trial_seed = payload.metadata.get('trial_seed', -1)
                else:
                    # Fallback: parse from prompt or conversation
                    task_suite_name, task_id, trial_id, trial_seed = self._parse_task_from_prompt(payload)
            else:
                # Fallback: parse from prompt or conversation
                task_suite_name, task_id, trial_id, trial_seed = self._parse_task_from_prompt(payload)
            
            # Convert tensors to Python integers if needed
            if hasattr(task_id, 'item'):
                task_id = task_id.item()
            if hasattr(trial_id, 'item'):
                trial_id = trial_id.item()
            if hasattr(trial_seed, 'item'):
                trial_seed = trial_seed.item()
            
            # Create environment configuration
            task_config = {
                'task_suite_name': task_suite_name,
                'task_name': task_suite_name,  # For compatibility
                'task_id': int(task_id),
                'trial_id': int(trial_id),
                'trial_seed': int(trial_seed),
                'max_steps': 400,  # Default max steps
                **self.vla_config.env_config
            }
            
            return task_config
            
        except Exception as e:
            logger.warning(f"Failed to extract LIBERO task info from payload: {e}")
            # Return default configuration
            return {
                'task_suite_name': self.task_suite,
                'task_name': self.task_suite,
                'task_id': 0,
                'trial_id': 0,
                'trial_seed': -1,
                'max_steps': 400,
                **self.vla_config.env_config
            }
    
    def _parse_task_from_prompt(self, payload: RLPayload) -> tuple:
        """
        Parse task information from prompt or conversation as fallback
        
        Returns:
            Tuple of (task_suite_name, task_id, trial_id, trial_seed)
        """
        # Default values
        task_suite_name = self.task_suite
        task_id = 0
        trial_id = 0
        trial_seed = -1
        
        try:
            # Try to parse from prompt
            if payload.prompt:
                prompt_text = payload.prompt.lower()
                # Look for task suite indicators
                if 'libero_10' in prompt_text:
                    task_suite_name = 'libero_10'
                elif 'libero_90' in prompt_text:
                    task_suite_name = 'libero_90'
                elif 'robotwin' in prompt_text:
                    task_suite_name = 'robotwin2'
                
                # Try to extract numbers as task_id and trial_id
                import re
                numbers = re.findall(r'\d+', prompt_text)
                if len(numbers) >= 2:
                    task_id = int(numbers[0])
                    trial_id = int(numbers[1])
                elif len(numbers) >= 1:
                    task_id = int(numbers[0])
            
            # Try to parse from conversation
            if payload.conversation:
                for message in payload.conversation:
                    if message.get('role') == 'user' and message.get('content'):
                        content = message.get('content', '').lower()
                        if 'task' in content and 'trial' in content:
                            # Extract task information from conversation
                            import re
                            task_match = re.search(r'task[_\s]*(\d+)', content)
                            trial_match = re.search(r'trial[_\s]*(\d+)', content)
                            if task_match:
                                task_id = int(task_match.group(1))
                            if trial_match:
                                trial_id = int(trial_match.group(1))
        
        except Exception as e:
            logger.debug(f"Error parsing task from prompt: {e}")
        
        return task_suite_name, task_id, trial_id, trial_seed
