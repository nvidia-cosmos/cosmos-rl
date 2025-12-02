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
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import defaultdict
import traceback
from multiprocessing import Process, Queue

from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload
from transformers import AutoTokenizer

from .action_processing import VLAActionProcessor
from .vla_model_inference import VLAModelInference
from .libero_utils import save_rollout_video, obs_to_vla_input
from .env_worker import libero_env_worker, robotwin_env_worker

# Import VLA constants (will be set based on robot platform at module load time)
try:
    from cosmos_rl.policy.model.vla.openvla_oft.constants import NUM_ACTIONS_CHUNK
except ImportError:
    NUM_ACTIONS_CHUNK = 8  # Default for LIBERO
    logger.warning(f"Could not import NUM_ACTIONS_CHUNK, using default: {NUM_ACTIONS_CHUNK}")


class VLARollout(RolloutBase):
    """
    VLA (Vision-Language-Action) Rollout implementation for embodied AI tasks.
    
    This class handles the actual VLA rollout generation by:
    1. Processing RLPayload objects from the dispatcher
    2. Setting up robotic environments 
    3. Running VLA model inference
    4. Executing actions in environments
    5. Computing rewards and returning RolloutResult objects
    """
    
    def __init__(self, config: Config, tokenizer: AutoTokenizer):
        super().__init__(config, tokenizer)
        
        self.config = config
        self.tokenizer = tokenizer
        
        # VLA-specific configuration
        self.vla_config = config.vla
        
        # Initialize VLA processor and action processor
        self.processor = None
        self.action_processor = VLAActionProcessor(config)
        self.model_inference = None  # Will be initialized in init_engine
        
        # Environment configuration
        self.task_suite = self.vla_config.task_suite
        self.max_steps = self._get_max_steps_for_task(self.task_suite)
        self.num_envs = self.vla_config.num_parallel_envs
        
        # VLA inference configuration (stored as instance variables)
        self.task_suite_name = self.task_suite
        self.vla_type = self.vla_config.vla_type
        self.do_sample = True  # Enable sampling for stochastic GRPO rollouts
        self.temperature = config.rollout.sampling_config.temperature
        self.center_crop = self.vla_config.center_crop  # Default center crop
        self.use_proprio = False  # Default proprioception usage
        self.use_wrist_camera = self.vla_config.use_wrist_camera
        self.unnorm_key = self.task_suite
        
        # VLA model module (placeholder)
        self.module = None
        
        # Thread pool for parallel environment execution
        self.env_thread_pool = ThreadPoolExecutor(max_workers=self.num_envs * 2)
        self.env_lock = Lock()
        
        # Simulation worker process pool (reused across rollouts to avoid frequent process creation/destruction)
        self.sim_processes = []
        self.sim_input_queues = []
        self.sim_output_queues = []
        self.sim_pool_size = 0
        self.sim_pool_lock = Lock()
        
        # Success rate thresholds for GRPO filtering (default member variables)
        # NOTE: These are different from PPO epsilon_low/high which are clipping ratios!
        self.success_rate_threshold_low = 0.1
        self.success_rate_threshold_high = 0.9
        
        logger.info(f"Initialized VLA rollout for task suite: {self.task_suite}")
        logger.info(f"GRPO filtering: success_rate ∈ [{self.success_rate_threshold_low:.2f}, {self.success_rate_threshold_high:.2f}]")
        logger.info(f"Max steps per episode: {self.max_steps}")
        logger.info(f"Sampling config: do_sample={self.do_sample}, temperature={self.temperature}")
        logger.info(f"Parallel environments: {self.num_envs}")
    
    
    def _get_max_steps_for_task(self, task_suite: str) -> int:
        """Get maximum steps for different task suites"""
        max_steps_map = {
            # LIBERO tasks
            "libero_spatial": 512,
            "libero_object": 512,
            "libero_goal": 512,
            "libero_10": 512,
            "libero_90": 512,
            
            # RoboTwin 2.0 tasks
            "robotwin2_click_bell": 200,
            "robotwin2_move_can_pot": 200,
            "robotwin2_place_phone_stand": 200,
            "robotwin2_place_a2b_left": 200,
            "robotwin2_place_a2b_right": 200,
            "robotwin2_handover_mic": 200,
            "robotwin2_pick_dual_bottles": 100,
            "robotwin2_lift_pot": 200,
            "robotwin2_put_bottles_dustbin": 800,
            "robotwin2_stack_blocks_two": 400,
            "robotwin2_stack_bowls_two": 400,
            "robotwin2_handover_block": 400,
            "robotwin2_place_empty_cup": 200,
            "robotwin2_shake_bottle": 75,
            "robotwin2_move_stapler_pad": 200,
            "robotwin2_place_container_plate": 150,
            "robotwin2_blocks_ranking_rgb": 600,
            "robotwin2_beat_block_hammer": 200,
            "robotwin2_place_mouse_pad": 200,
            "robotwin2_place_shoe": 250,
            "robotwin2_move_pillbottle_pad": 200,
        }
        
        return max_steps_map.get(task_suite, self.vla_config.max_episode_length)
    
    def init_engine(self, quantization: str, seed: int, load_format: str):
        """
        Initialize the VLA processing engine with actual model loading
        
        Args:
            quantization: Quantization mode (unused for VLA, kept for API compatibility)
            seed: Random seed
            load_format: "dummy" for structure only, "auto" for full weight loading
        """
        try:
            logger.info(f"Initializing VLA engine with load_format={load_format}")
            
            model_path = self.config.policy.model_name_or_path
            
            # Initialize the processor for VLA models using PrismaticProcessor
            # IMPORTANT: Must use PrismaticProcessor, not generic AutoProcessor!
            logger.info(f"Loading VLA processor from {model_path} (type: {self.vla_type})")
            
            if self.vla_type == "openvla-oft":
                from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import PrismaticProcessor
            else:  # openvla
                from cosmos_rl.policy.model.vla.openvla.processing_prismatic import PrismaticProcessor
            
            self.processor = PrismaticProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            logger.info("✅ VLA processor loaded")
            
            # Initialize VLA model
            logger.info(f"Initializing VLA model structure for {self.vla_type}")
            
            if load_format == "dummy":
                # Load model structure only (weights will come from policy worker)
                self._init_dummy_vla_model(model_path)
            else:  # "auto"
                # Load full model with weights (for checkpointing/resume)
                self._init_full_vla_model(model_path)
            
            # Initialize VLA model inference
            self.model_inference = VLAModelInference(self.module, self.processor, self)
            logger.info("✅ VLA model inference initialized")
            
            logger.info("✅ VLA engine initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize VLA engine: {e}")
            traceback.print_exc()
            raise
    
    def _init_dummy_vla_model(self, model_path: str):
        """
        Initialize VLA model structure without loading weights (dummy mode)
        Weights will be synchronized from policy worker via NCCL
        
        This mode is for distributed training where policy workers send weights
        to rollout workers for data parallelism.
        """
        from cosmos_rl.policy.model.vla import VLAModel, VLAArgs
        from cosmos_rl.policy.model.vla_utils import create_vla_config
        
        logger.info("Creating VLA model structure (dummy weights, for weight sync)...")
        
        # Create VLA config with norm_stats
        vla_config, processor, tokenizer = create_vla_config(
            model_path,
            cosmos_config=self.config,
            model=self.vla_type
        )
        
        # Create VLA args
        vla_args = VLAArgs(
            vla_type=self.vla_type,
            use_proprio=self.config.vla.use_proprio,
            proprio_dim=self.config.vla.action_dim,
            num_images_in_input=self.config.vla.num_images_in_input,
            hf_config=vla_config
        )
        
        # Initialize model structure on CUDA directly (no meta device)
        # Note: VLA models use TIMM vision backbone which is incompatible with meta tensors
        logger.info("Initializing VLA model on CUDA (TIMM meta tensor incompatibility)")
        vla_model = VLAModel(vla_args, init_device="cuda")
        
        # NOTE: Vision backbone weights will be synced via P2R NCCL
        # The vision backbone IS trainable (not frozen) and will receive updates from policy workers
        # All weights (vision_backbone + projector + language_model + action_head) initialized randomly
        # and will be synced from policy worker via NCCL on first P2R weight sync
        logger.info("✅ VLA model structure created with random weights (P2R sync will provide actual weights)")
        
        # Save both the wrapper and inner module
        self.vla_model = vla_model  # Keep wrapper for weight_sync_transforms
        self.module = vla_model.model  # Inner OpenVLAForActionPrediction for inference
        self.module.eval()
        
        # Save processor, tokenizer, and config for later use
        self.processor = processor
        self.tokenizer = tokenizer
        self.hf_config = vla_config  # Save for shard info generation
        
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        logger.info(f"✅ VLA model structure initialized on {device}")
        logger.info(f"   ALL weights (vision_backbone + projector + LLM + action_head) will be synced via P2R NCCL")
        logger.info(f"   Vision backbone IS trainable (not frozen)")
        logger.info(f"   Model has norm_stats: {hasattr(self.module, 'norm_stats')}")
        if hasattr(self.module, 'norm_stats'):
            logger.info(f"   norm_stats keys: {list(self.module.norm_stats.keys())}")
    
    def _init_full_vla_model(self, model_path: str):
        """
        Initialize VLA model with full weight loading (auto mode)
        Used for standalone rollout or verification at init
        
        Uses HF's from_pretrained interface (like RLinf) - simple and fast
        for single-GPU rollout with data parallelism.
        """
        from cosmos_rl.policy.model.vla import VLAModel, VLAArgs
        from cosmos_rl.policy.model.vla_utils import create_vla_config
        
        logger.info("Loading full VLA model with weights (single-GPU, like RLinf)...")
        
        # Create VLA config (includes norm_stats from dataset_statistics.json)
        vla_config, processor, tokenizer = create_vla_config(
            model_path,
            cosmos_config=self.config,
            model=self.vla_type
        )
        
        # Create VLA args
        vla_args = VLAArgs(
            vla_type=self.vla_type,
            use_proprio=self.config.vla.use_proprio,
            proprio_dim=self.config.vla.action_dim,
            num_images_in_input=self.config.vla.num_images_in_input,
            hf_config=vla_config
        )
        
        # Initialize model structure
        vla_model = VLAModel.from_model_args(vla_args)
        
        # Load weights using HF interface (simple, like RLinf)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        vla_model.load_from_checkpoint(
            model_name_or_path=model_path,
            parallel_dims=None,  # Single GPU rollout (data parallelism only)
            device=device
        )
        
        # Save both the wrapper and inner module
        self.vla_model = vla_model  # Keep wrapper for weight_sync_transforms
        self.module = vla_model.model  # Inner OpenVLAForActionPrediction for inference
        self.module.eval()
        self.processor = vla_model.processor  # Save processor from load_from_checkpoint
        self.tokenizer = tokenizer
        self.hf_config = vla_config  # Save for shard info generation
        
        logger.info("✅ Full VLA model loaded with weights (single-GPU)")
        logger.info(f"   Device: {device}")
        logger.info(f"   Model has norm_stats: {hasattr(self.module, 'norm_stats')}")
        if hasattr(self.module, 'norm_stats'):
            logger.info(f"   norm_stats keys: {list(self.module.norm_stats.keys())}")
    
    # DEPRECATED: No longer needed - vision backbone weights are synced via P2R NCCL
    def _load_frozen_vision_backbone_weights(self, model, model_path: str):
        """
        [DEPRECATED] Previously loaded vision backbone weights assuming they were frozen.
        
        This method is no longer used because:
        1. Vision backbone IS trainable (not frozen)
        2. All VLA weights (vision_backbone + projector + LLM + action_head) are synced via P2R NCCL
        3. The VLAWeightMapper now defines parallelism strategies for complete P2R weight sync
        
        Historical context:
        - Originally assumed vision backbone was frozen during training
        - Loaded vision backbone weights once from checkpoint to avoid NCCL sync
        - This was incorrect - vision backbone should be trained and synced like other components
        """
        logger.warning("[DEPRECATED] _load_frozen_vision_backbone_weights called but no longer used")
        logger.warning("   Vision backbone weights will be synced via P2R NCCL instead")
        return  # Early return - don't load anything
    
    def is_engine_initialized(self) -> bool:
        """Check if the VLA engine has been initialized"""
        return self.module is not None and self.processor is not None
    
    def _extract_task_config_from_payload(self, payload: RLPayload) -> Dict[str, Any]:
        """Extract task configuration from a single payload"""
        # Try to extract task info from payload metadata or prompt
        task_suite_name = self.task_suite  # Default
        task_id = 0  # Default
        trial_id = 0  # Default
        trial_seed = -1  # Default
        
        # Check if task info is in metadata
        if hasattr(payload, 'metadata') and payload.metadata:
            task_suite_name = payload.metadata.get('task_suite_name', task_suite_name)
            task_id = payload.metadata.get('task_id', task_id)
            trial_id = payload.metadata.get('trial_id', trial_id)
            trial_seed = payload.metadata.get('trial_seed', trial_seed)
        
        # Convert tensors to Python values if needed
        if hasattr(task_id, 'item'):
            task_id = task_id.item()
        if hasattr(trial_id, 'item'):
            trial_id = trial_id.item()
        if hasattr(trial_seed, 'item'):
            trial_seed = trial_seed.item()
        
        return {
            'task_suite_name': task_suite_name,
            'task_name': task_suite_name,  # For compatibility
            'task_id': int(task_id),
            'trial_id': int(trial_id),
            'trial_seed': int(trial_seed),
            'max_steps': self._get_max_steps_for_task(task_suite_name),
            **self.vla_config.env_config
        }
    
    def _process_grpo_group(self, 
                           payloads: List[RLPayload],
                           n_generation: int,
                           is_validation: bool,
                           global_steps: int,
                           group_idx: int) -> List[RolloutResult]:
        """
        Process one GRPO group (parallel episodes)
        
        Two modes:
        - Training (GRPO): Replicates payloads[0] n_generation times for group normalization
        - Validation: Processes multiple different payloads in parallel (no replication)
        
        Args:
            payloads: List of payloads (len=1 for training, len>1 for validation batching)
            n_generation: Number of parallel episodes (used for training replication)
            is_validation: Whether this is validation (controls video saving and filtering)
            global_steps: Current training step
            group_idx: Index of this GRPO group (for logging)
            
        Returns:
            List of RolloutResult objects (n_generation for training, len(payloads) for validation)
        """
        if len(payloads) == 1 and not is_validation:
            # Training mode: replicate single payload for GRPO group normalization
            payload = payloads[0]
            instruction = self._extract_instruction_from_payload(payload)
            
            payloads_expanded = [payload] * n_generation
            gen_indices = list(range(n_generation))
            instructions = [instruction] * n_generation
            group_size = n_generation
        else:
            # Validation mode: process multiple different payloads in parallel
            payloads_expanded = payloads
            gen_indices = list(range(len(payloads)))
            instructions = [self._extract_instruction_from_payload(p) for p in payloads]
            group_size = len(payloads)
            
            # For filtering, we'll use the first instruction (all should be similar tasks)
            instruction = instructions[0]
        
        # Run parallel inference and simulation
        vla_history, task_records = self._parallel_inference_and_sim(
            payloads=payloads_expanded,
            gen_indices=gen_indices,
            instructions=instructions,
            temperature=self.temperature,
            is_valid=is_validation,
            global_steps=global_steps
        )
        
        # Pack GRPO results: Apply filtering, compute old_log_probs, create RolloutResults
        # Returns None if group doesn't meet GRPO criteria (training only)
        results = self._pack_grpo_results(
            vla_history=vla_history,
            task_records=task_records,
            instruction=instruction,
            group_size=group_size,
            enable_filtering=not is_validation,
        )
        
        return results
    
    def _get_payload_task_info(self, payload: RLPayload) -> str:
        """Extract task_id and trial_id from payload for logging"""
        # Task info is stored directly in metadata, not nested under 'task_config'
        if not hasattr(payload, 'metadata') or not payload.metadata:
            return "task_id=?, trial_id=?"
        
        task_id = payload.metadata.get('task_id', '?')
        trial_id = payload.metadata.get('trial_id', '?')
        
        # Convert tensors to Python values if needed
        if hasattr(task_id, 'item'):
            task_id = task_id.item()
        if hasattr(trial_id, 'item'):
            trial_id = trial_id.item()
        
        return f"task_id={task_id}, trial_id={trial_id}"
    
    
    def rollout_generation(self, payloads: List[RLPayload], 
                          n_generation: int = 1,
                          is_validation: bool = True,
                          global_steps: int = 0) -> List[RolloutResult]:
        """
        Generate VLA rollouts by interacting with robotic environments
        
        This is the main entry point called by the rollout worker.
        It processes RLPayload objects from the dispatcher and returns RolloutResult objects.
        
        For GRPO-style training (n_generation > 1):
        - Roll each payload n_generation times
        - Apply GRPO filtering: accept groups with success rate in [threshold_low, threshold_high]
        - Discard groups that don't meet the criteria
        - No retry, no leftover carrying, no replacement logic
        
        Uses self.temperature for sampling control (configured at initialization).
        
        Args:
            payloads: List of RLPayload objects containing task instructions and prompts
            n_generation: Number of rollouts per task (GRPO replication). Default 1 for validation.
            is_validation: If True, save videos. If False (training), don't save videos.
            global_steps: Current training step for logging/video naming.
            
        Returns:
            List of RolloutResult containing trajectories and outcomes.
            For training with n_generation > 1, returns only valid results (filtered by GRPO criteria).
        """
        if not self.processor:
            raise RuntimeError("VLA processor not initialized. Call init_engine() first.")
        
        logger.info(
            f"Starting VLA rollout generation: {len(payloads)} payloads, "
            f"n_generation={n_generation}, temperature={self.temperature}, "
            f"is_validation={is_validation}"
        )
        
        # Get success rate thresholds (only for training)
        enable_grpo_filter = (not is_validation and n_generation > 1)
        success_rate_low = self.success_rate_threshold_low if enable_grpo_filter else 0.0
        success_rate_high = self.success_rate_threshold_high if enable_grpo_filter else 1.0
        
        if enable_grpo_filter:
            logger.info(
                f"[GRPO Filter] Enabled: success_rate ∈ [{success_rate_low:.2f}, {success_rate_high:.2f}]"
            )
        
        # Determine batching strategy
        if is_validation:
            # Validation: batch multiple different payloads together for efficiency
            # (n_generation=1 typically, so we batch to avoid process churn)
            batch_size = max(1, self.config.rollout.n_generation)  # Use n_generation as batch size
            num_batches = (len(payloads) + batch_size - 1) // batch_size
            logger.info(
                f"[Rollout] Validation mode: {len(payloads)} payloads in {num_batches} batches "
                f"(batch_size={batch_size}, parallel envs per batch)"
            )
        else:
            # Training (GRPO): each payload is replicated n_generation times for group normalization
            batch_size = 1
            num_batches = len(payloads)
            logger.info(
                f"[Rollout] Training mode: {len(payloads)} GRPO groups "
                f"(each payload × {n_generation} episodes for group normalization)"
            )
        
        valid_results = []
        num_accepted = 0
        num_discarded = 0
        
        for group_idx in range(num_batches):
            try:
                # Extract batch of payloads
                start_idx = group_idx * batch_size
                end_idx = min(start_idx + batch_size, len(payloads))
                payload_batch = payloads[start_idx:end_idx]
                
                if is_validation:
                    # Validation: process multiple different payloads in parallel
                    group_results = self._process_grpo_group(
                        payloads=payload_batch,
                        n_generation=len(payload_batch),  # Group size = number of payloads in batch
                        is_validation=is_validation,
                        global_steps=global_steps,
                        group_idx=group_idx
                    )
                else:
                    # Training: replicate single payload n_generation times (GRPO)
                    group_results = self._process_grpo_group(
                        payloads=payload_batch,  # Single payload list
                        n_generation=n_generation,
                        is_validation=is_validation,
                        global_steps=global_steps,
                        group_idx=group_idx
                    )
                
                if group_results is not None:
                    # Group passed filtering or filtering disabled
                    valid_results.extend(group_results)
                    num_accepted += 1
                else:
                    # Group filtered out (only happens when enable_grpo_filter=True)
                    num_discarded += 1
                    
            except Exception as e:
                import traceback
                batch_info = f"batch={group_idx}, size={len(payload_batch)}"
                logger.error(f"[GRPO Group {group_idx}] Failed to process group [{batch_info}]: {e}")
                logger.error(f"[GRPO Group {group_idx}] Traceback:\n{traceback.format_exc()}")
                logger.error(f"[GRPO Group {group_idx}] Skipping this entire group")
                num_discarded += 1
            finally:
                self._destroy_parallel_envs()
        
        # Summary
        if enable_grpo_filter:
            logger.info(
                f"[GRPO Filter] Complete: {num_accepted} accepted, {num_discarded} discarded "
                f"({len(valid_results)} total results from {len(payloads)} payloads)"
            )
        
        logger.info(f"Generated {len(valid_results)} VLA rollout results")
        return valid_results
    
    def _extract_instruction_from_payload(self, payload: RLPayload) -> str:
        """Extract task instruction from RLPayload"""
        
        # For VLA tasks, the instruction should be in the prompt or conversation
        if hasattr(payload, 'prompt') and payload.prompt:
            return payload.prompt
        
        if hasattr(payload, 'conversation') and payload.conversation:
            # Extract from conversation messages
            for message in payload.conversation:
                if message.get('role') == 'user':
                    return message.get('content', '')
        
        # Fallback to task-specific default instruction
        return f"Complete the {self.task_suite} task"
    
    def _setup_parallel_envs(self, payloads: List[RLPayload], gen_indices: List[int], 
                             is_valid: bool, global_steps: int):
        """
        Setup parallel environment workers
        
        Args:
            payloads: List of RLPayload objects containing task information
            gen_indices: List of generation indices (for varying random seeds)
            is_valid: Whether to save validation videos
            global_steps: Current training step
            
        Returns:
            Dict with initial_data containing: task_descriptions, inputs, task_records, valid_video
        
        Side Effects:
            Populates self.sim_processes, self.sim_input_queues, self.sim_output_queues
        """
        batch_size = len(payloads)
        
        # Extract task information from payloads and construct env configs
        task_suite_names = []
        task_ids = []
        trial_ids = []
        max_steps_list = []
        
        for payload in payloads:
            task_config = self._extract_task_config_from_payload(payload)
            task_suite_names.append(task_config['task_suite_name'])
            task_ids.append(task_config['task_id'])
            trial_ids.append(task_config.get('trial_id', 0))
            max_steps_list.append(task_config['max_steps'])
        
        # Spawn worker processes for each environment (stored in member variables)
        for idx in range(batch_size):
            task_name = task_suite_names[idx]
            t_id = task_ids[idx]
            tr_id = trial_ids[idx]
            max_steps = max_steps_list[idx]
            input_q = Queue()
            output_q = Queue()
            
            # Determine worker function based on task type
            if 'libero' in task_name.lower():
                worker_fn = libero_env_worker
            elif 'robotwin' in task_name.lower():
                worker_fn = robotwin_env_worker
            else:
                logger.warning(f"Unknown task type {task_name}, defaulting to LIBERO")
                worker_fn = libero_env_worker
            
            args = (task_name, t_id, tr_id, input_q, output_q, is_valid, global_steps, max_steps)
            p = Process(target=worker_fn, args=args)
            p.start()
            self.sim_processes.append(p)
            self.sim_input_queues.append(input_q)
            self.sim_output_queues.append(output_q)
        
        logger.debug(f"Spawned {len(self.sim_processes)} worker processes (total sim pool: {len(self.sim_processes)})")
        
        # Collect initial observations from workers
        task_descriptions = []
        inputs = []
        task_records = []
        valid_video = defaultdict(list)
        
        for idx in range(batch_size):
            init_data = self.sim_output_queues[idx].get(timeout=120)
            assert init_data['type'] == 'init', f"Expected 'init', got '{init_data['type']}'"
            
            task_descriptions.append(init_data["task_description"])
            inputs.append(obs_to_vla_input(init_data['obs'], is_robotwin='robotwin' in task_suite_names[idx].lower()))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name'],
                "task_id": task_ids[idx],
                "trial_id": trial_ids[idx],
                "gen_idx": gen_indices[idx],
            })
            
            # Collect initial video frames
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
        
        initial_data = {
            'task_descriptions': task_descriptions,
            'inputs': inputs,
            'task_records': task_records,
            'valid_video': valid_video,
        }
        
        return initial_data
    
    def _destroy_parallel_envs(self):
        """
        Cleanly destroy parallel environment workers and reset simulation pool
        
        Side Effects:
            - Terminates processes in self.sim_processes
            - Resets all self.sim_* member variables to empty
        """
        with self.sim_pool_lock:
            if not self.sim_processes:
                return  # Nothing to clean up
            
            logger.debug(f"Destroying {len(self.sim_processes)} simulation processes...")
            
            # Send termination signal to all workers
            for q in self.sim_input_queues:
                try:
                    q.put(None, timeout=1)
                except Exception:
                    pass  # Ignore errors, process termination will handle stuck workers
            
            # Wait for processes to finish, terminate if hung
            for p in self.sim_processes:
                p.join(timeout=20)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)  # Wait again after terminate
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=2)  # Final wait after kill
            
            # Reset simulation pool state
            num_destroyed = len(self.sim_processes)
            self.sim_processes = []
            self.sim_input_queues = []
            self.sim_output_queues = []
            self.sim_pool_size = 0
            
            logger.debug(f"Destroyed {num_destroyed} simulation processes, pool reset")
    
    def _parallel_inference_and_sim(self, payloads: List[RLPayload], gen_indices: List[int], 
                                    instructions: List[str], temperature: float = 0.0, 
                                    is_valid: bool = False, global_steps: int = 0):
        """
        Run parallel VLA inference and simulation for a GRPO group
        
        Uses separate processes for each environment to avoid shared OpenGL/MuJoCo state.
        
        Args:
            payloads: List of RLPayload objects containing task information
            gen_indices: List of generation indices (for varying random seeds)
            instructions: List of task instructions  
            temperature: Sampling temperature for action generation
            is_valid: Whether to save validation videos
            global_steps: Current training step (for video naming)
            
        Returns:
            Tuple of (vla_history, task_records)
        """
        batch_size = len(payloads)
        logger.debug(f"Starting {batch_size} parallel VLA episodes: temperature={temperature}, save_videos={is_valid}")
        
        # Extract task info from payloads (already available, no need to pass through initial_data)
        task_suite_names = []
        task_ids = []
        trial_ids = []
        for payload in payloads:
            task_config = self._extract_task_config_from_payload(payload)
            task_suite_names.append(task_config['task_suite_name'])
            task_ids.append(task_config['task_id'])
            trial_ids.append(task_config.get('trial_id', 0))
        gen_idxs = gen_indices
        
        # CRITICAL: Release GPU resources before spawning sim workers
        # After NCCL broadcast, GPU SM resources may still be occupied by:
        # - Lingering CUDA kernels from weight sync
        # - VLA model occupying GPU memory
        # - CUDA context fragmentation
        # Without this, sim workers timeout trying to initialize rendering contexts
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        torch.cuda.empty_cache()  # Free up cached GPU memory
        logger.debug(f"Released GPU resources before spawning {len(payloads)} sim workers")
        
        # Setup parallel environments (populates self.sim_processes, self.sim_input_queues, self.sim_output_queues)
        initial_data = self._setup_parallel_envs(
            payloads, gen_indices, is_valid, global_steps
        )
        
        # Unpack initial data
        task_descriptions = initial_data['task_descriptions']
        inputs = initial_data['inputs']
        task_records = initial_data['task_records']
        valid_video = initial_data['valid_video']
        
        # Use member variables for process/queue management
        batch_size = len(self.sim_processes)

        # Episode execution loop
        step = 0
        vla_history = []
        
        while step < self.max_steps:
            # Find active environments
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            if not active_indices:
                break
            
            # VLA model inference on all inputs
            current_inputs = inputs
            current_task_descriptions = task_descriptions
            
            vla_input = self.model_inference.process_input(current_inputs, current_task_descriptions)
            vla_output = self.model_inference.generate_one_step(vla_input)
            
            actions = vla_output.get("action", np.random.randn(batch_size, NUM_ACTIONS_CHUNK, 7))
            
            # Store step data
            step_data = {
                "responses": vla_output.get("responses", [f"Step {step}"] * batch_size),
                "input_ids": vla_output.get("input_ids", torch.randint(0, 1000, (batch_size, 10))),
                "attention_mask": vla_output.get("attention_mask", torch.ones(batch_size, 10)),
                "pixel_values": vla_output.get("pixel_values", torch.randn(batch_size, 3, 224, 224)),
                "action": actions,
                "step": step
            }
            vla_history.append(step_data)
            
            # Send actions to active workers
            for idx in active_indices:
                self.sim_input_queues[idx].put(actions[idx])
            
            # Collect results from active workers
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = self.sim_output_queues[idx].get(timeout=30)
                assert result['type'] == 'step', f"Expected 'step', got '{result['type']}'"
                
                new_inputs[idx] = obs_to_vla_input(result['obs'], is_robotwin='robotwin' in task_suite_names[idx].lower())
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                
                # Collect video frames
                if is_valid and len(result['valid_images']) > 0:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
                
                if not result['active']:
                    status = "✅ SUCCESS" if result['complete'] else "❌ FAILED"
                    logger.debug(f"Task {idx} [task_id={task_ids[idx]}, trial_id={trial_ids[idx]}, gen={gen_idxs[idx]}]: {status} (steps={result['finish_step']})")
            
            inputs = new_inputs
            step += NUM_ACTIONS_CHUNK
        
        # Save rollout videos
        if valid_video:
            experiment_name = getattr(self.config, 'experiment_name', 'vla_rollout')
            
            for task_file, images in valid_video.items():
                if len(images) > 0:
                    complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                    
                    try:
                        video_path = save_rollout_video(
                            images,
                            experiment_name,
                            task_file,
                            global_steps,
                            complete
                        )
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to save {task_file}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Clear memory
        torch.cuda.empty_cache()
        
        # Return raw VLA history and task records for post-processing
        return vla_history, task_records
    
    def _pack_grpo_results(self, vla_history: List[Dict], task_records: List[Dict],
                           instruction: str, group_size: int, enable_filtering: bool):
        """
        Pack GRPO results: apply filtering, compute old_log_probs, create RolloutResults
        
        This function:
        1. Checks GRPO filtering criteria (if enabled)
        2. If group is invalid, returns None (skips expensive log prob computation)
        3. If group is valid, extracts trajectories and computes old_log_probs
        4. Creates and returns RolloutResult objects
        
        Args:
            vla_history: List of step_data dicts from _parallel_inference_and_sim
            task_records: List of task metadata dicts
            instruction: Task instruction for this group
            group_size: Number of episodes in the group (n_generation)
            enable_filtering: Whether to apply GRPO filtering
            
        Returns:
            List of RolloutResult objects if valid, None if filtered out
        """
        # Check GRPO filtering criteria first (before expensive log prob computation)
        if enable_filtering:
            successes = sum(1 for r in task_records if r['complete'])
            success_rate = successes / group_size
            
            # Check if success rate is in valid range
            if not (self.success_rate_threshold_low <= success_rate <= self.success_rate_threshold_high):
                logger.info(
                    f"[GRPO Filter] ❌ DISCARDED task {task_records[0]['task_id']}, trial {task_records[0]['trial_id']} "
                    f"(success={successes}/{group_size}, rate={success_rate:.2f})"
                )
                return None
            else:
                logger.info(
                    f"[GRPO Filter] ✅ ACCEPTED task {task_records[0]['task_id']}, trial {task_records[0]['trial_id']} "
                    f"(success={successes}/{group_size}, rate={success_rate:.2f})"
                )
        
        # Group passed filtering or filtering disabled - proceed with expensive computation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize per-episode trajectories
        trajectories = [{'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'responses': []} 
                        for _ in range(group_size)]
        
        # Extract trajectory data for each episode up to their finish_step
        import math
        num_steps = len(vla_history)
        
        for episode_idx in range(group_size):
            finish_step = task_records[episode_idx]['finish_step']
            # Each step produces 8 actions, so num_steps_needed = ceil(finish_step / 8)
            num_steps_needed = math.ceil(finish_step / 8) if finish_step > 0 else 0
            
            for step_idx in range(min(num_steps_needed, num_steps)):
                step_data = vla_history[step_idx]
                trajectories[episode_idx]['input_ids'].append(step_data['input_ids'][episode_idx])
                trajectories[episode_idx]['attention_mask'].append(step_data['attention_mask'][episode_idx])
                trajectories[episode_idx]['pixel_values'].append(step_data['pixel_values'][episode_idx])
                trajectories[episode_idx]['responses'].append(step_data['responses'][episode_idx])
        
        # Compute old_log_probs for each episode by replaying trajectory
        for episode_idx in range(group_size):
            traj = trajectories[episode_idx]
            
            # Stack into batch tensors
            input_ids_batch = torch.stack(traj['input_ids']).to(device)
            attention_mask_batch = torch.stack(traj['attention_mask']).to(device)
            pixel_values_batch = torch.stack(traj['pixel_values']).to(device)
            responses_batch = torch.stack(traj['responses']).to(device)
            
            # Compute old_log_probs
            with torch.no_grad():
                outputs = self.vla_model.forward_with_trajectory_structure(
                    input_ids=input_ids_batch,
                    pixel_values=pixel_values_batch,
                    attention_mask=attention_mask_batch,
                    labels=responses_batch,
                    temperature=self.temperature,
                    proprio=None
                )
                logits, _, old_log_probs = outputs.logits, outputs.entropy, outputs.logprobs
            # if episode_idx == 0:
            #     logger.info(f"input_ids {input_ids_batch.shape}, logits {logits.shape}, log_probs {old_log_probs.shape}")
            #     logger.info(f"logits chunk 0: {logits[0]}")
            #     logger.info(f"old_log_probs chunk 0: {old_log_probs.reshape(-1, 56)[0]}")
            #     logger.info(f"logits chunk 31: {logits[31]}")
            #     logger.info(f"old_log_probs chunk 31: {old_log_probs.reshape(-1, 56)[31]}")
            #     # logger.info(f"responses {responses_batch.shape}")
            #     # logger.info(f"{responses_batch}")
            #     logger.info(f"pixel_values {pixel_values_batch.shape}")
            #     logger.info(f"pixel_values chunk 31: {pixel_values_batch[31]}")
            #     logger.info(f"attention_mask {attention_mask_batch.shape}")
            #     logger.info(f"{attention_mask_batch[31]}")
            
            # OPTIMIZATION: Store as STACKED tensor instead of list to reduce pickle overhead
            # (num_steps, 56) instead of list of (56,) tensors
            # This reduces pickle metadata from num_steps entries to 1 entry
            traj['old_log_prob'] = old_log_probs.cpu()
            
            # Also stack other fields to reduce pickle overhead (8x improvement!)
            traj['input_ids'] = input_ids_batch.cpu()  # (num_steps, seq_len)
            # Convert attention_mask to int for JSON serialization (bool tensors cause issues)
            traj['attention_mask'] = attention_mask_batch.cpu().to(torch.int64)  # (num_steps, seq_len)
            traj['pixel_values'] = pixel_values_batch.cpu()  # (num_steps, 6, 224, 224)
            traj['responses'] = responses_batch.cpu()  # (num_steps, 56)
        
        # Create RolloutResults for each episode in the group
        results = []
        for env_idx in range(group_size):
            success = task_records[env_idx]['complete']
            episode_length = task_records[env_idx]['finish_step']
            trajectory_data = trajectories[env_idx]
            
            status = 'completed' if success else 'failed'
            result = RolloutResult(
                prompt=instruction,
                completions=[f"Task {status} in {episode_length} steps"],
                log_probs=[[np.log(0.5)]],
                input_tokens=100,
                output_tokens=int(episode_length),  # Ensure Python int
                rewards=[1.0 if success else 0.0],
                episode_length=int(episode_length),  # Ensure Python int
                environment_info={
                    'task_suite': str(self.task_suite),  # Ensure Python str
                    'success': bool(success),  # Convert numpy.bool_ to Python bool
                    'num_actions': int(episode_length),  # Add num_actions field expected by worker
                    'task_id': task_records[env_idx]['task_id'],
                    'trial_id': task_records[env_idx]['trial_id'],
                    'gen_idx': task_records[env_idx]['gen_idx'],
                },
                vla_trajectory=trajectory_data
            )
            results.append(result)
        
        return results
    