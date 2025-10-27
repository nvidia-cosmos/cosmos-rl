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
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from cosmos_rl.rollout import RolloutWorkerBase, State
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data import RLPayload, IdxAndRLPayload
from cosmos_rl.dispatcher.command import Command
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.dispatcher.replica import Rollout

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
        
        # Command and prompt queues (inherited from base class)
        self._command_queue: Queue[Command] = Queue()
        self._prompt_queue: Queue[List[IdxAndRLPayload]] = Queue()
        self.current_weight_version = 0
        
        # Communication setup (similar to vLLM rollout worker)
        self.rollout_status_manager = None  # Will be set up during setup()
        
        # Data handling
        self.data_packer = None
        
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
        
        logger.info(f"Initialized VLA rollout worker for task suite: {self.task_suite}")
        logger.info(f"Parallel environments: {self.num_parallel_envs}")
    
    def setup(self, dataset=None, reward_fns=None, filter_reward_fns=None, 
              val_dataset=None, val_reward_fns=None):
        """Setup VLA rollout worker with datasets and reward functions"""
        logger.info("Setting up VLA rollout worker")
        
        # Initialize the VLA rollout engine
        self.rollout.init_engine(
            quantization=self.config.rollout.quantization,
            seed=42,
            load_format="dummy"
        )
        
        # Setup communication (following vLLM rollout worker pattern)
        from cosmos_rl.reward.reward_calculator import RewardDispatcher
        from cosmos_rl.dispatcher.data.packer.base import DataPacker
        
        # Setup reward dispatcher (similar to vLLM rollout worker)
        self.reward_dispatcher = RewardDispatcher()
        self.reward_dispatcher.setup(
            config=self.config,
            dataset=dataset,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_dataset=val_dataset,
            val_reward_fns=val_reward_fns,
            data_packer=None,  # VLA doesn't use text data packer
            val_data_packer=None
        )
        
        logger.info("VLA rollout worker setup completed")
    
    def work(self):
        """Main work loop for VLA rollout worker"""
        logger.info("Starting VLA rollout worker")
        
        try:
            # Start background threads for command processing
            self.background_thread = threading.Thread(
                target=self._background_work_loop,
                daemon=True
            )
            self.background_thread.start()
            
            # Main work loop - follow vLLM rollout worker pattern
            self.main_loop()
                    
        except KeyboardInterrupt:
            logger.info("VLA rollout worker interrupted")
        except Exception as e:
            logger.error(f"Fatal error in VLA rollout worker: {e}")
            raise
        finally:
            self._cleanup()
    
    @torch.no_grad()
    def main_loop(self):
        """Main processing loop following vLLM rollout worker pattern"""
        while True:
            try:
                # Process commands from controller
                if not self._command_queue.empty():
                    command = self._command_queue.get_nowait()
                    self._process_command(command)
                
                # Check if we can process rollouts
                if self.state.prompt_consume_end():
                    assert (
                        self._prompt_queue.empty() and self.state.prompt_fetch_end()
                    ), "[VLA Rollout] If prompts are all consumed, prompt queue should be empty and prompt end event should be set."
                    continue
                elif self._prompt_queue.empty():
                    continue
                else:
                    logger.debug(f"[VLA Rollout] Generate start for rank {self.global_rank}")

                    # Check if the prompt is valid for the current weight version
                    first_payload: RLPayload = self._prompt_queue.queue[0][0].payload
                    is_valid_prompt_for_current_weight_version = (
                        first_payload.weight_version <= self.current_weight_version
                    )

                    if not is_valid_prompt_for_current_weight_version:
                        # Wait until the weight version is updated
                        continue

                    # Get batch of prompts to process
                    prompt_id_and_payload_list: List[IdxAndRLPayload] = (
                        self._prompt_queue.get()
                    )
                    payloads: List[RLPayload] = [
                        idx_and_payload.payload for idx_and_payload in prompt_id_and_payload_list
                    ]

                    # Generate VLA rollouts
                    rollout_results: List[RolloutResult] = self.rollout.rollout_generation(
                        payloads=payloads
                    )

                    if len(rollout_results) == 0:
                        logger.warning("[VLA Rollout] No rollout results generated")
                        continue

                    assert len(rollout_results) == len(
                        payloads
                    ), f"Error: VLA rollout returned {len(rollout_results)} results for {len(payloads)} payloads"

                    # Process and send results back to controller
                    self._send_rollout_results(rollout_results, prompt_id_and_payload_list)
                    
            except Exception as e:
                logger.error(f"Error in VLA rollout main loop: {e}")
                traceback.print_exc()
                continue
    
    def _background_work_loop(self):
        """Background thread for continuous processing"""
        logger.info("Starting VLA rollout background work loop")
        
        while True:
            try:
                # Background tasks like environment maintenance
                self._maintain_environments()
                
                # Sleep to avoid busy waiting
                threading.Event().wait(0.1)
                
            except Exception as e:
                logger.error(f"Error in background work loop: {e}")
                continue
    
    def _process_command(self, command: Command):
        """Process command from controller"""
        logger.debug(f"Processing command: {command.command_type}")
        
        if command.command_type == "SYNC_WEIGHT":
            self._sync_weights(command)
        elif command.command_type == "UPDATE_CONFIG":
            self._update_config(command)
        else:
            logger.warning(f"Unknown command type: {command.command_type}")
    
    def _sync_weights(self, command: Command):
        """Sync model weights from policy workers"""
        logger.info("Syncing weights for VLA rollout")
        
        try:
            # Weight synchronization logic
            # TODO: Implement weight sync with VLA model
            self.current_weight_version += 1
            self.state.set_weight_synced()
            
            logger.info(f"Weight sync completed, version: {self.current_weight_version}")
            
        except Exception as e:
            logger.error(f"Failed to sync weights: {e}")
    
    def _update_config(self, command: Command):
        """Update configuration from controller"""
        logger.info("Updating VLA rollout configuration")
        
        # TODO: Implement config updates
        pass
    
    def _process_rollout_batch(self, prompts_batch: List[IdxAndRLPayload]):
        """Process a batch of rollout requests"""
        logger.info(f"Processing VLA rollout batch with {len(prompts_batch)} prompts")
        
        try:
            # Extract prompts and metadata
            prompts = []
            indices = []
            
            for idx_and_payload in prompts_batch:
                prompts.append(idx_and_payload.payload.prompt)
                indices.append(idx_and_payload.idx)
            
            # Generate VLA rollouts
            rollout_results = self.rollout.rollout_generation(prompts)
            
            # Process results and send back to controller
            self._send_rollout_results(rollout_results, indices)
            
            logger.info(f"Completed VLA rollout batch processing")
            
        except Exception as e:
            logger.error(f"Error processing rollout batch: {e}")
            raise
    
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
                    'max_steps': self._get_max_steps_for_task(task_suite_name),
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
                'max_steps': self.max_steps,
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
    
    def _maintain_environments(self):
        """Maintain environment pool health"""
        # Check environment status and restart if needed
        for env_id, env_wrapper in self.env_pool.items():
            if not env_wrapper.is_active() and env_wrapper.env is not None:
                # Environment may need reset or cleanup
                try:
                    env_wrapper.reset()
                except Exception as e:
                    logger.warning(f"Failed to reset environment {env_id}: {e}")
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up VLA rollout worker")
        
        # Close all environments
        for env_id, env_wrapper in self.env_pool.items():
            try:
                env_wrapper.close()
            except Exception as e:
                logger.error(f"Error closing environment {env_id}: {e}")
        
        # Shutdown thread pool
        if hasattr(self, 'env_thread_pool'):
            self.env_thread_pool.shutdown(wait=True)
        
        # Cleanup rollout engine
        if hasattr(self, 'rollout'):
            self.rollout.cleanup()
        
        logger.info("VLA rollout worker cleanup completed")
    
    def __del__(self):
        """Destructor"""
        self._cleanup()
