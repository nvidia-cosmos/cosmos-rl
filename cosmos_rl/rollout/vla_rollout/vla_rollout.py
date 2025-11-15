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
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import defaultdict
import gc
import traceback
from multiprocessing import Process, Queue

from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload
from transformers import AutoTokenizer
import traceback

from .action_processing import VLAActionProcessor
from .vla_model_inference import VLAModelInference
from .libero_utils import save_rollout_video
from .env_worker import libero_env_worker, robotwin_env_worker, EnvConfig

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
        
        # GRPO Streaming Queue: Leftover state persists across batches
        self.leftover_payloads = []  # Payloads that didn't meet success rate criteria
        
        # Success rate thresholds for GRPO filtering (default member variables)
        # NOTE: These are different from PPO epsilon_low/high which are clipping ratios!
        self.success_rate_threshold_low = 0.1
        self.success_rate_threshold_high = 0.9
        self.leftover_metadata = {}  # Track leftover_count, success_rate per payload_id
        self.replacement_mapping = {}  # Maps exhausted payload_id -> replacement payload
        self.MAX_LEFTOVER_ATTEMPTS = 2  # Max attempts before replacement
        
        # Sorted candidate pool for replacement selection
        # Each entry: (hardness_score, leftover_count, success_rate, payload_id, payload)
        # Sorted by hardness descending (hardest first)
        self.candidate_pool = []  # List of successfully validated payloads
        self.candidate_pool_lock = Lock()  # Thread-safe access
        
        # GRPO filtering mode: retry or discard
        self.grpo_discard_mode = True #getattr(config.train.train_policy, 'grpo_discard_instead_of_retry', False)
        
        logger.info(f"Initialized VLA rollout for task suite: {self.task_suite}")
        logger.info(f"GRPO filtering mode: {'discard' if self.grpo_discard_mode else 'retry (leftover)'}")
        logger.info(f"Max steps per episode: {self.max_steps}")
        logger.info(f"Sampling config: do_sample={self.do_sample}, temperature={self.temperature}")
        logger.info(f"Parallel environments: {self.num_envs}")
        logger.info(f"GRPO Streaming Queue: enabled with max_leftover_attempts={self.MAX_LEFTOVER_ATTEMPTS}")
    
    
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
    
    def _process_rollout_chunk(self, 
                               rollout_tasks: List[Tuple[RLPayload, int]],
                               is_validation: bool,
                               global_steps: int,
                               chunk_idx: int) -> List[RolloutResult]:
        """
        Process a chunk of rollout tasks (payload, gen_idx pairs)
        
        Each task creates exactly ONE environment config, ensuring we never exceed MAX_CHUNK_SIZE parallel envs.
        Uses self.temperature for sampling control.
        
        Args:
            rollout_tasks: List of (payload, gen_idx) tuples to process
            is_validation: Whether this is validation (controls video saving)
            global_steps: Current training step
            chunk_idx: Index of this chunk (for logging)
            
        Returns:
            List of RolloutResult objects (len = len(rollout_tasks))
        """
        chunk_size = len(rollout_tasks)
        
        # Create exactly one environment config per rollout task
        env_configs = []
        instructions = []
        
        for payload, gen_idx in rollout_tasks:
            instruction = self._extract_instruction_from_payload(payload)
            task_config = self._extract_task_config_from_payload(payload)
            
            # Create environment config for this rollout
            # Use gen_idx to vary the trial_id for different random seeds
            trial_id = task_config.get('trial_id', 0)
            
            env_config = EnvConfig(
                task_suite=task_config['task_suite_name'],
                task_id=task_config['task_id'],
                trial_id=trial_id,
                max_steps=task_config['max_steps'],
                gen_idx=gen_idx,  # Track generation index for logging
                resolution=256,
                save_video=is_validation,
                global_steps=global_steps,
                extra_config=task_config.get('extra_config', None)
            )
            
            env_configs.append(env_config)
            instructions.append(instruction)
        
        logger.debug(
            f"[Chunk {chunk_idx}] Created {chunk_size} environment configs"
        )
        
        try:
            # Run parallel episodes for all environment configs in this chunk
            # Note: Retry/failover is handled by GRPO filter at the group level
            batch_result = self._run_parallel_episodes(
                env_configs, 
                instructions,
                len(env_configs),
                temperature=self.temperature,  # Use instance variable
                is_valid=is_validation,
                global_steps=global_steps
            )
            for key, value in batch_result.items():
                if isinstance(value, list) and isinstance(value[0], torch.Tensor):
                    logger.warning(f"batch_result[{key}] = {len(value)} x {value[0].shape}")
                elif isinstance(value, torch.Tensor):
                    logger.warning(f"batch_result[{key}] = {value.shape}")
                else:
                    logger.warning(f"batch_result[{key}] = {value}")
            
            # Extract results - one per rollout task
            results = []
            for env_idx, (payload, gen_idx) in enumerate(rollout_tasks):
                # Handle both tensor and scalar extraction
                if 'complete' in batch_result:
                    complete_val = batch_result['complete'][env_idx]
                    success = complete_val.item() if hasattr(complete_val, 'item') else bool(complete_val)
                else:
                    success = False
                
                if 'finish_step' in batch_result:
                    step_val = batch_result['finish_step'][env_idx]
                    episode_length = step_val.item() if hasattr(step_val, 'item') else int(step_val)
                else:
                    episode_length = 0
                
                # Extract trajectory data for this environment
                # IMPORTANT: episode_length is in INDIVIDUAL ACTION STEPS (e.g., 236 actions)
                # But trajectory data is organized in ACTION CHUNKS (8 actions per chunk)
                # So: num_chunks_needed = ceil(episode_length / 8)
                import math
                action_chunk_size = 8  # Each chunk generates 8 actions
                num_chunks_needed = math.ceil(episode_length / action_chunk_size) if episode_length > 0 else 0
                
                trajectory_data = {
                    'input_ids': [],
                    'attention_mask': [],
                    'pixel_values': [],
                    'responses': [],  # Action token IDs
                }
                
                # Collect trajectory from batch_result, but ONLY up to num_chunks_needed
                # Keep tensors as-is for efficient filesystem serialization (will be pickled)
                if 'input_ids' in batch_result and batch_result['input_ids'] is not None:
                    # Only iterate up to num_chunks_needed (not episode_length!)
                    for chunk_idx, step_input_ids in enumerate(batch_result['input_ids']):
                        if chunk_idx >= num_chunks_needed:
                            break  # Stop at actual number of chunks needed
                        if isinstance(step_input_ids, torch.Tensor):
                            ids = step_input_ids[env_idx].clone()  # Clone to avoid shared storage
                        elif isinstance(step_input_ids, list):
                            ids = step_input_ids[env_idx]
                        else:
                            ids = step_input_ids
                        trajectory_data['input_ids'].append(ids)
                
                if 'attention_mask' in batch_result and batch_result['attention_mask'] is not None:
                    for chunk_idx, step_attention_mask in enumerate(batch_result['attention_mask']):
                        if chunk_idx >= num_chunks_needed:
                            break
                        if isinstance(step_attention_mask, torch.Tensor):
                            mask = step_attention_mask[env_idx].clone()  # Clone to avoid shared storage
                        elif isinstance(step_attention_mask, list):
                            mask = step_attention_mask[env_idx]
                        else:
                            mask = step_attention_mask
                        trajectory_data['attention_mask'].append(mask)
                
                # Collect pixel_values - will be stored to filesystem buffer (not sent via HTTP)
                if 'pixel_values' in batch_result and batch_result['pixel_values'] is not None:
                    for chunk_idx, step_pixel_values in enumerate(batch_result['pixel_values']):
                        if chunk_idx >= num_chunks_needed:
                            break
                        # CRITICAL: Must extract ONLY this environment's pixel_values, not all 8!
                        # step_pixel_values shape: (batch_size=8, channels, H, W)
                        if isinstance(step_pixel_values, torch.Tensor):
                            # Debug: log shape BEFORE extraction
                            if env_idx == 0 and chunk_idx == 0:
                                logger.warning(f"[DEBUG] BEFORE extraction: step_pixel_values.shape={step_pixel_values.shape}, dtype={step_pixel_values.dtype}, storage_size={step_pixel_values.storage().size()}")
                            
                            # CRITICAL FIX: Clone the tensor to avoid sharing storage with the full batch!
                            # When you slice a tensor (step_pixel_values[env_idx]), PyTorch creates a view
                            # that shares the underlying storage with the original tensor. This means
                            # pickle will serialize the ENTIRE underlying storage (all 8 envs) even though
                            # we only reference 1/8 of it!
                            pix_vals = step_pixel_values[env_idx].clone()  # Clone to get independent storage
                            
                            # Debug: log shape AFTER extraction and memory size
                            if env_idx == 0 and chunk_idx == 0:
                                mem_mb = pix_vals.element_size() * pix_vals.nelement() / (1024 * 1024)
                                storage_mb = pix_vals.storage().size() * pix_vals.element_size() / (1024 * 1024)
                                logger.warning(f"[DEBUG] AFTER extraction: pix_vals.shape={pix_vals.shape}, dtype={pix_vals.dtype}, mem={mem_mb:.2f} MB, storage={storage_mb:.2f} MB")
                            
                            trajectory_data['pixel_values'].append(pix_vals)
                        elif isinstance(step_pixel_values, list):
                            pix_vals = step_pixel_values[env_idx]  # Extract one env from list
                            trajectory_data['pixel_values'].append(pix_vals)
                        else:
                            # Fallback: log warning if unexpected type
                            logger.warning(f"Unexpected pixel_values type: {type(step_pixel_values)}, storing as-is")
                            trajectory_data['pixel_values'].append(step_pixel_values)
                
                if 'responses' in batch_result and batch_result['responses'] is not None:
                    for chunk_idx, step_responses in enumerate(batch_result['responses']):
                        if chunk_idx >= num_chunks_needed:
                            break
                        if isinstance(step_responses, torch.Tensor):
                            resp = step_responses[env_idx].clone()  # Clone to avoid shared storage
                        elif isinstance(step_responses, list):
                            resp = step_responses[env_idx]
                        else:
                            resp = step_responses
                        trajectory_data['responses'].append(resp)
                
                # Compute old_log_probs for the generated trajectory (matching SimpleVLA-RL)
                # This replays the trajectory through the model to get log probabilities
                if (len(trajectory_data['input_ids']) > 0 and 
                    len(trajectory_data['responses']) > 0 and
                    self.model_inference is not None):
                    try:
                        # Stack all steps into batch tensors
                        input_ids_batch = torch.stack(trajectory_data['input_ids'])  # (num_steps, seq_len)
                        attention_mask_batch = torch.stack(trajectory_data['attention_mask'])  # (num_steps, seq_len)
                        pixel_values_batch = torch.stack(trajectory_data['pixel_values'])  # (num_steps, C, H, W)
                        responses_batch = torch.stack(trajectory_data['responses'])  # (num_steps, response_len)
                        
                        # Move to device
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        input_ids_batch = input_ids_batch.to(device)
                        attention_mask_batch = attention_mask_batch.to(device)
                        pixel_values_batch = pixel_values_batch.to(device)
                        responses_batch = responses_batch.to(device)
                        
                        # Compute log probabilities by replaying trajectory
                        with torch.no_grad():
                            old_log_probs = self.model_inference.compute_old_log_probs(
                                input_ids=input_ids_batch,
                                attention_mask=attention_mask_batch,
                                pixel_values=pixel_values_batch,
                                responses=responses_batch,
                                temperature=self.temperature,
                                proprio=None  # TODO: Add proprio support if needed
                            )
                        
                        # Store old_log_probs in trajectory data
                        # Convert to list of tensors for consistency with other trajectory fields
                        trajectory_data['old_log_prob'] = [old_log_probs[i].cpu() for i in range(old_log_probs.shape[0])]
                        
                        logger.debug(f"Computed old_log_probs for trajectory with {len(trajectory_data['old_log_prob'])} steps")
                    
                    except Exception as e:
                        logger.warning(f"Failed to compute old_log_probs for trajectory: {e}")
                        # Don't fail the entire rollout, just skip old_log_prob computation
                        import traceback
                        traceback.print_exc()
                
                # Create RolloutResult directly (avoid intermediate episode_data dict)
                num_actions = len(trajectory_data['responses'])
                completion_text = f"Task {'completed' if success else 'failed'} in {episode_length} steps ({num_actions} actions)"
                
                result = RolloutResult(
                    prompt=instructions[env_idx],
                    completions=[completion_text],
                    log_probs=[[np.log(0.5)]],  # Single log prob for single completion
                    input_tokens=100,  # Approximate
                    output_tokens=len(completion_text.split()),
                    
                    # VLA-specific fields
                    rewards=[1.0 if success else 0.0],  # Binary reward based on success
                    episode_length=episode_length,
                    environment_info={
                        'task_suite': self.task_suite,
                        'success': success,
                        'num_actions': num_actions,
                    },
                    
                    # Trajectory data for training (stored as-is, no conversion)
                    vla_trajectory=trajectory_data
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"[Chunk {chunk_idx}] Error processing chunk: {e}")
            raise e
    
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
    
    def _get_payload_id(self, payload: RLPayload) -> str:
        """Get unique identifier for payload (for tracking across batches)"""
        # Task info is stored directly in metadata, not nested under 'task_config'
        if not hasattr(payload, 'metadata') or not payload.metadata:
            return "unknown_unknown"
        
        task_id = payload.metadata.get('task_id', 'unknown')
        trial_id = payload.metadata.get('trial_id', 'unknown')
        
        # Convert tensors to Python values if needed
        if hasattr(task_id, 'item'):
            task_id = task_id.item()
        if hasattr(trial_id, 'item'):
            trial_id = trial_id.item()
        
        # Use task_id + trial_id as unique identifier
        return f"{task_id}_{trial_id}"
    
    def _add_to_candidate_pool(self, payload: RLPayload, payload_id: str, avg_success_rate: float, 
                                leftover_count: int, replacement_usage_count: int, success_rate_history: list):
        """Add or update a payload in the candidate pool for replacement selection
        
        The candidate pool is sorted by hardness score (descending).
        Hardness = (1 - avg_success_rate) * 100 + leftover_count - (replacement_usage_count * 20)
        
        Key insight:
        - Lower avg success rate = harder task (more valuable for training)
        - Higher replacement_usage_count = already used many times (less valuable)
        
        Example scores:
        - Task with 10% success, used 0 times: hardness = 90 + 0 - 0 = 90 (very hard, fresh)
        - Task with 10% success, used 2 times: hardness = 90 + 0 - 40 = 50 (hard, but overused)
        - Task with 50% success, used 0 times: hardness = 50 + 0 - 0 = 50 (medium, fresh)
        
        Within-batch diversity example (4 exhausted payloads in same batch):
        1. Payload 0 exhausted → picks task_8_37 (hardness=90, usage=0)
           → Increments usage to 1, re-adds to pool with hardness=70
        2. Payload 1 exhausted → picks task_9_42 (hardness=85, usage=0) instead
           → Increments usage to 1, re-adds to pool with hardness=65
        3. Payload 2 exhausted → picks task_7_25 (hardness=80, usage=0)
        4. Result: 3 different hard tasks used, not just one!
        
        This ensures diversity both across batches AND within the same batch.
        """
        with self.candidate_pool_lock:
            # Remove existing entry for this payload_id if present
            self.candidate_pool = [entry for entry in self.candidate_pool if entry[4] != payload_id]
            
            # Calculate hardness score
            # Penalize replacement_usage heavily (×20) to diversify replacements
            hardness = (1.0 - avg_success_rate) * 100 + leftover_count - (replacement_usage_count * 20)
            
            # Add new entry: (hardness, avg_success_rate, leftover_count, replacement_usage_count, payload_id, payload)
            entry = (hardness, avg_success_rate, leftover_count, replacement_usage_count, payload_id, payload)
            self.candidate_pool.append(entry)
            
            # Sort by hardness descending (hardest first)
            self.candidate_pool.sort(key=lambda x: x[0], reverse=True)
            
            logger.debug(
                f"[Candidate Pool] Added {payload_id} (hardness={hardness:.2f}, "
                f"avg_rate={avg_success_rate:.2f} (history={[f'{r:.2f}' for r in success_rate_history]}), "
                f"leftover={leftover_count}, replacement_usage={replacement_usage_count}), "
                f"pool size={len(self.candidate_pool)}"
            )
    
    def _get_hardest_candidate(self, exclude_id: str = None) -> tuple:
        """Get the hardest candidate from the pool (excluding specified payload_id)
        
        Returns:
            (hardness, avg_success_rate, leftover_count, replacement_usage_count, payload_id, payload) 
            or None if pool empty
        """
        with self.candidate_pool_lock:
            # Find first candidate not in exclusion list
            for entry in self.candidate_pool:
                payload_id = entry[4]  # Updated index after adding replacement_usage_count
                if payload_id != exclude_id and payload_id not in self.replacement_mapping:
                    return entry
            return None
    
    def _check_grpo_group_valid(self, group_results: List[RolloutResult], 
                                success_rate_low: float, success_rate_high: float) -> bool:
        """Check if a GRPO group meets success rate criteria.
        
        Args:
            group_results: List of RolloutResult for same task (n_generation rollouts)
            success_rate_low: Minimum acceptable success rate (e.g., 0.2 = 20%)
            success_rate_high: Maximum acceptable success rate (e.g., 0.8 = 80%)
            
        Returns:
            True if group is valid (success rate in [success_rate_low, success_rate_high])
        """
        successes = sum(1 for r in group_results if r.environment_info.get('success', False))
        success_rate = successes / len(group_results)
        return success_rate_low <= success_rate <= success_rate_high
    
    def rollout_generation(self, payloads: List[RLPayload], 
                          n_generation: int = 1,
                          is_validation: bool = True,
                          global_steps: int = 0,
                          flush_leftovers: bool = False) -> List[RolloutResult]:
        """
        Generate VLA rollouts by interacting with robotic environments
        
        This is the main entry point called by the rollout worker.
        It processes RLPayload objects from the dispatcher and returns RolloutResult objects.
        
        For GRPO-style training, this method uses a STREAMING QUEUE approach:
        - Rollout each payload group once
        - Payloads with success rate in [success_rate_threshold_low, success_rate_threshold_high] are accepted immediately
        - Payloads outside range become "leftovers" and are carried to next iteration
        - After MAX_LEFTOVER_ATTEMPTS (default 3), exhausted leftovers are replaced with HARDEST tasks
        - Hardness = (1 - avg_success_rate) * 100 + leftover_count - replacement_usage_count * 20
        - This avoids repeatedly using the same hard task and diversifies training data
        - Creates a natural curriculum: easy tasks graduate fast, hard tasks get more attempts
        
        Uses self.temperature for sampling control (configured at initialization).
        
        Args:
            payloads: List of RLPayload objects containing task instructions and prompts
            n_generation: Number of rollouts per task (GRPO replication). Default 1 for validation.
            is_validation: If True, save videos. If False (training), don't save videos.
            global_steps: Current training step for logging/video naming.
            
        Returns:
            List of RolloutResult containing trajectories and outcomes.
            For training with n_generation > 1, returns n_generation results per input payload.
        """
        if not self.processor:
            raise RuntimeError("VLA processor not initialized. Call init_engine() first.")
        
        # STREAMING QUEUE: Merge new payloads with leftovers from previous batch
        if enable_grpo_filter := (not is_validation and n_generation > 1):
            num_leftovers = len(self.leftover_payloads)
            if num_leftovers > 0:
                logger.info(
                    f"[GRPO Streaming Queue] Merging {len(payloads)} new payloads "
                    f"with {num_leftovers} leftovers from previous batch"
                )
                # Prepend leftovers so they get processed first
                all_payloads = self.leftover_payloads + list(payloads)
            else:
                all_payloads = list(payloads)
        else:
            all_payloads = list(payloads)
            num_leftovers = 0
        
        logger.info(
            f"Starting VLA rollout generation: {len(all_payloads)} total payloads "
            f"({len(payloads)} new, {num_leftovers} leftovers), "
            f"n_generation={n_generation}, temperature={self.temperature}, "
            f"is_validation={is_validation}"
        )
        
        try:
            # STREAMING QUEUE strategy with adaptive GRPO filtering
            # For training (n_generation > 1):
            #   1. Roll all payloads once
            #   2. Accept valid ones (success rate in epsilon range)
            #   3. Mark invalid ones as "leftovers" 
            #   4. Carry leftovers to next iteration
            #   5. After MAX attempts, replace with hardest tasks from valid set
            #   6. Hardness = leftover_count * 10 + (1 - success_rate)
            MAX_CHUNK_SIZE = 8  # Maximum parallel environments per chunk
            
            # Get success rate thresholds (only for training)
            success_rate_low = self.success_rate_threshold_low if enable_grpo_filter else 0.0
            success_rate_high = self.success_rate_threshold_high if enable_grpo_filter else 1.0
            
            if enable_grpo_filter:
                logger.info(
                    f"[GRPO Streaming Queue] Filtering enabled: success_rate ∈ [{success_rate_low:.2f}, {success_rate_high:.2f}], "
                    f"max leftover attempts={self.MAX_LEFTOVER_ATTEMPTS}, "
                    f"hardness metric=(1-avg_success_rate)*100 + leftover_count - replacement_usage*20"
                )
            
            # Initialize metadata for new payloads
            for idx, payload in enumerate(all_payloads):
                payload_id = self._get_payload_id(payload)
                if payload_id not in self.leftover_metadata:
                    self.leftover_metadata[payload_id] = {
                        'leftover_count': 0,
                        'success_rate_history': [],  # Track all success rates
                        'avg_success_rate': 0.0,  # Average success rate across all attempts
                        'replacement_usage_count': 0,  # How many times used as replacement
                        'payload_idx': idx
                    }
            
            # STREAMING QUEUE: One uniform generation per call, carry leftovers to next batch
            # Create rollout tasks: all_payloads × n_generation
            rollout_tasks = []
            for payload_idx, payload in enumerate(all_payloads):
                # Use replacement payload if this one was exhausted
                payload_id = self._get_payload_id(payload)
                actual_payload = self.replacement_mapping.get(payload_id, payload)
                
                for gen_idx in range(n_generation):
                    rollout_tasks.append((payload_idx, actual_payload, gen_idx))
            
            total_tasks = len(rollout_tasks)
            num_chunks = (total_tasks + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
            
            logger.info(
                f"[GRPO Rollout] Rolling {len(all_payloads)} payloads × {n_generation} = {total_tasks} tasks "
                f"→ {num_chunks} chunks"
            )
            
            # Process chunks and collect results by payload
            results_per_payload = {idx: [] for idx in range(len(all_payloads))}
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * MAX_CHUNK_SIZE
                chunk_end = min(chunk_start + MAX_CHUNK_SIZE, total_tasks)
                chunk_tasks_with_idx = rollout_tasks[chunk_start:chunk_end]
                
                # Extract (payload, gen_idx) for processing
                chunk_tasks = [(payload, gen_idx) for _, payload, gen_idx in chunk_tasks_with_idx]
                
                # Process this chunk
                chunk_results = self._process_rollout_chunk(
                    rollout_tasks=chunk_tasks,
                    is_validation=is_validation,
                    global_steps=global_steps,
                    chunk_idx=chunk_idx
                )
                
                # Group results by payload_idx
                for (payload_idx, _, _), result in zip(chunk_tasks_with_idx, chunk_results):
                    results_per_payload[payload_idx].append(result)
            
            # Check each payload's group and apply GRPO filtering
            new_leftovers = []
            num_exhausted_replaced = 0
            valid_results = []  # Results to return (only for new payloads, not leftovers)
            
            for payload_idx, payload in enumerate(all_payloads):
                group_results = results_per_payload[payload_idx]
                task_info = self._get_payload_task_info(payload)
                
                # DEBUG: Check group size
                if len(group_results) != n_generation:
                    logger.error(
                        f"[GRPO Filter BUG] Payload {payload_idx} [{task_info}]: Expected {n_generation} results, "
                        f"got {len(group_results)}! This will cause incorrect filtering."
                    )
                
                if enable_grpo_filter:
                    payload_id = self._get_payload_id(payload)
                    metadata = self.leftover_metadata[payload_id]
                    
                    is_valid = self._check_grpo_group_valid(group_results, success_rate_low, success_rate_high)
                    successes = sum(1 for r in group_results if r.environment_info.get('success', False))
                    success_rate = successes / len(group_results) if len(group_results) > 0 else 0.0
                    
                    # Add current success rate to history and calculate average
                    metadata['success_rate_history'].append(success_rate)
                    metadata['avg_success_rate'] = sum(metadata['success_rate_history']) / len(metadata['success_rate_history'])
                    
                    if is_valid:
                        # Valid group - always accept and send results immediately
                        # (Whether it's a new payload or a leftover that finally passed)
                        valid_results.extend(group_results)
                        
                        # If this payload was replaced and now succeeded, clear the replacement
                        if payload_id in self.replacement_mapping:
                            replacement_id = self._get_payload_id(self.replacement_mapping[payload_id])
                            logger.info(
                                f"[GRPO Filter] Payload {payload_idx} [{task_info}, id={payload_id}]: "
                                f"✅ REPLACEMENT {replacement_id} SUCCEEDED - clearing replacement mapping"
                            )
                            del self.replacement_mapping[payload_id]
                        
                        # Add to candidate pool for potential replacement use
                        self._add_to_candidate_pool(
                            payload, payload_id, 
                            metadata['avg_success_rate'], 
                            metadata['leftover_count'],
                            metadata['replacement_usage_count'],
                            metadata['success_rate_history']
                        )
                        
                        logger.info(
                            f"[GRPO Filter] Payload {payload_idx} [{task_info}, id={payload_id}]: ✅ VALID "
                            f"(success={successes}/{len(group_results)}, rate={success_rate:.2f}, "
                            f"avg_rate={metadata['avg_success_rate']:.2f}, leftover={metadata['leftover_count']})"
                        )
                    else:
                        # Not valid - discard or mark as leftover based on mode
                        if self.grpo_discard_mode:
                            # Discard mode: skip this payload entirely, don't retry
                            logger.info(
                                f"[GRPO Filter] Payload {payload_idx} [{task_info}, id={payload_id}]: ❌ DISCARDED "
                                f"(success={successes}/{len(group_results)}, rate={success_rate:.2f}, "
                                f"avg_rate={metadata['avg_success_rate']:.2f}) - discard mode active"
                            )
                            # Don't add to new_leftovers, just skip
                            continue
                        
                        # Retry mode (default): mark as leftover for next batch
                        metadata['leftover_count'] += 1
                        
                        if metadata['leftover_count'] >= self.MAX_LEFTOVER_ATTEMPTS:
                            # Exceeded max attempts - replace with hardest task for NEXT batch
                            hardest_candidate = self._get_hardest_candidate(exclude_id=payload_id)
                            
                            if hardest_candidate:
                                (hardness_score, replacement_avg_rate, repl_leftover_count, repl_usage_count,
                                 replacement_id, replacement_payload) = hardest_candidate
                                replacement_task_info = self._get_payload_task_info(replacement_payload)
                                
                                # Store replacement mapping (will be used in next batch)
                                self.replacement_mapping[payload_id] = replacement_payload
                                num_exhausted_replaced += 1
                                
                                # Increment replacement usage count for the selected task
                                if replacement_id in self.leftover_metadata:
                                    repl_metadata = self.leftover_metadata[replacement_id]
                                    repl_metadata['replacement_usage_count'] += 1
                                    
                                    # CRITICAL: Update candidate pool with new usage count
                                    # This recalculates hardness and re-sorts, so next exhausted payload
                                    # won't pick the same task (lower hardness due to higher usage)
                                    self._add_to_candidate_pool(
                                        replacement_payload, 
                                        replacement_id,
                                        repl_metadata['avg_success_rate'],
                                        repl_metadata['leftover_count'],
                                        repl_metadata['replacement_usage_count'],  # Now incremented
                                        repl_metadata['success_rate_history']
                                    )
                                
                                logger.warning(
                                    f"[GRPO Filter] Payload {payload_idx} [{task_info}, id={payload_id}]: "
                                    f"⚠️ EXHAUSTED (attempts={metadata['leftover_count']}, "
                                    f"rate={success_rate:.2f}, avg_rate={metadata['avg_success_rate']:.2f}) - "
                                    f"will replace with HARDEST task {replacement_id} [{replacement_task_info}] "
                                    f"(avg_rate={replacement_avg_rate:.2f}, leftover={repl_leftover_count}, "
                                    f"usage={repl_usage_count}→{repl_usage_count+1}, hardness={hardness_score:.2f}→lower) in NEXT batch"
                                )
                                
                                # Still add to leftovers (but will use replacement payload next time)
                                new_leftovers.append(payload)
                            else:
                                logger.error(
                                    f"[GRPO Filter] Payload {payload_idx} [{task_info}, id={payload_id}]: "
                                    f"EXHAUSTED but candidate pool empty! Accepting results."
                                )
                                # Accept current results
                                # In flush mode, return leftover results; otherwise skip them
                                if flush_leftovers or payload_idx >= num_leftovers:
                                    valid_results.extend(group_results)
                        else:
                            # Not exhausted yet - add to candidate pool and leftovers
                            self._add_to_candidate_pool(
                                payload, payload_id,
                                metadata['avg_success_rate'],
                                metadata['leftover_count'],
                                metadata['replacement_usage_count'],
                                metadata['success_rate_history']
                            )
                            new_leftovers.append(payload)
                            
                            logger.info(
                                f"[GRPO Filter] Payload {payload_idx} [{task_info}, id={payload_id}]: ↻ LEFTOVER "
                                f"(attempt {metadata['leftover_count']}/{self.MAX_LEFTOVER_ATTEMPTS}, "
                                f"success={successes}/{len(group_results)}, rate={success_rate:.2f}, "
                                f"avg_rate={metadata['avg_success_rate']:.2f}) - carry to next batch"
                            )
                else:
                    # No filtering for validation - accept all results
                    # In flush mode, return leftover results; otherwise skip them
                    if flush_leftovers or payload_idx >= num_leftovers:
                        valid_results.extend(group_results)
            
            # Use collected valid results
            results = valid_results
            
            # Update leftover state for next batch
            if enable_grpo_filter:
                self.leftover_payloads = new_leftovers
                
                # Calculate stats
                total_leftover_attempts = sum(meta['leftover_count'] for meta in self.leftover_metadata.values())
                max_leftover_count = max((meta['leftover_count'] for meta in self.leftover_metadata.values()), default=0)
                pool_size = len(self.candidate_pool)
                
                logger.info(
                    f"[GRPO Filter] Batch complete: {len(all_payloads)} payloads "
                    f"({len(payloads)} new, {num_leftovers} from prev batch), "
                    f"{total_leftover_attempts} total leftover attempts, "
                    f"max leftover count={max_leftover_count}, "
                    f"{num_exhausted_replaced} exhausted to be replaced, "
                    f"{len(new_leftovers)} leftovers → next batch, "
                    f"candidate pool size={pool_size}"
                )
        except Exception as e:
            logger.error(f"Error in batch rollout generation: {e}")
            traceback.print_exc()
            # Return failure results: n_generation failures per payload
            results = []
            for payload in payloads:
                for _ in range(n_generation):
                    results.append(self._create_failure_result(payload))
        
        logger.info(f"Generated {len(results)} VLA rollout results")
        return results
    
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
    
    def _run_episodes_with_retry(self, env_configs: List[EnvConfig], instructions: List[str],
                                  temperature: float = 0.0, is_valid: bool = False, 
                                  global_steps: int = 0, max_retries: int = 3) -> Dict:
        """
        Run episodes with automatic retry for failed workers
        
        Calls _run_parallel_episodes and retries any failed tasks.
        Each retry runs a complete new episode from step 0.
        
        Args:
            env_configs: List of environment configurations
            instructions: List of task instructions
            temperature: Sampling temperature
            is_valid: Whether to save validation videos
            global_steps: Current training step
            max_retries: Maximum number of retry attempts per task
        
        Returns:
            Batch result dictionary with all episodes completed
        """
        batch_size = len(env_configs)
        logger.info(f"Running {batch_size} episodes with retry (max_retries={max_retries})")
        
        # Track which indices still need to be run
        pending_configs = list(env_configs)
        pending_instructions = list(instructions)
        pending_indices = list(range(batch_size))  # Original indices
        
        # Store completed results
        final_results = [None] * batch_size
        retry_counts = [0] * batch_size
        
        attempt = 0
        while pending_configs and attempt < max_retries:
            attempt += 1
            logger.info(f"Attempt {attempt}/{max_retries}: Running {len(pending_configs)} episodes")
            
            # Run pending episodes
            batch_result = self._run_parallel_episodes(
                pending_configs,
                pending_instructions,
                len(pending_configs),
                temperature=temperature,
                is_valid=is_valid,
                global_steps=global_steps
            )
            
            # Check results and separate success/failures
            new_pending_configs = []
            new_pending_instructions = []
            new_pending_indices = []
            
            for i, orig_idx in enumerate(pending_indices):
                # Extract result for this task (stored in batch_result)
                # The result should indicate if it failed
                if self._is_episode_failed(batch_result, i):
                    # Retry this one
                    retry_counts[orig_idx] += 1
                    logger.warning(
                        f"Task {orig_idx} [task_id={pending_configs[i].task_id}, "
                        f"trial_id={pending_configs[i].trial_id}, gen={pending_configs[i].gen_idx}] "
                        f"failed, retry {retry_counts[orig_idx]}/{max_retries}"
                    )
                    new_pending_configs.append(pending_configs[i])
                    new_pending_instructions.append(pending_instructions[i])
                    new_pending_indices.append(orig_idx)
                else:
                    # Success, store result
                    final_results[orig_idx] = self._extract_single_result(batch_result, i)
                    logger.debug(f"Task {orig_idx} completed successfully")
            
            pending_configs = new_pending_configs
            pending_instructions = new_pending_instructions
            pending_indices = new_pending_indices
        
        # Any remaining pending tasks have exhausted retries
        if pending_configs:
            logger.error(f"{len(pending_configs)} tasks failed after {max_retries} retries")
            # Create failure results for these
            for i, orig_idx in enumerate(pending_indices):
                final_results[orig_idx] = self._create_failure_result_from_config(pending_configs[i])
        
        # Merge all results into a single batch result
        merged_result = self._merge_episode_results(final_results, batch_size)
        
        total_retries = sum(retry_counts)
        logger.info(f"Episode batch completed: {batch_size} tasks, {total_retries} total retries")
        
        return merged_result
    
    def _is_episode_failed(self, batch_result: Dict, index: int) -> bool:
        """Check if episode at given index failed"""
        # Check task_records in batch_result
        if 'task_records' in batch_result and index < len(batch_result['task_records']):
            return batch_result['task_records'][index].get('failed', False)
        return False
    
    def _extract_single_result(self, batch_result: Dict, index: int) -> Dict:
        """Extract result for a single episode from batch result"""
        # Extract data for index from all lists in batch_result
        single_result = {}
        for key, value in batch_result.items():
            if isinstance(value, list) and len(value) > index:
                single_result[key] = value[index]
            else:
                single_result[key] = value
        return single_result
    
    def _merge_episode_results(self, results: List[Dict], batch_size: int) -> Dict:
        """Merge individual episode results into a batch result"""
        # Reconstruct batch_result format from individual results
        complete_list = []
        finish_step_list = []
        responses_list = []
        task_records_list = []
        
        for result in results:
            if result:
                # Extract data from single result and convert to scalar
                if 'complete' in result:
                    val = result['complete']
                    # Extract scalar from tensor if needed
                    if torch.is_tensor(val):
                        if val.numel() == 1:
                            complete_list.append(val.item())
                        else:
                            # If it's already a batch, take first element
                            complete_list.append(val[0].item() if val.numel() > 0 else False)
                    else:
                        complete_list.append(bool(val))
                elif 'task_records' in result:
                    complete_list.append(result['task_records'].get('complete', False))
                else:
                    complete_list.append(False)
                    
                if 'finish_step' in result:
                    val = result['finish_step']
                    if torch.is_tensor(val):
                        if val.numel() == 1:
                            finish_step_list.append(val.item())
                        else:
                            finish_step_list.append(val[0].item() if val.numel() > 0 else 0)
                    else:
                        finish_step_list.append(int(val))
                elif 'task_records' in result:
                    finish_step_list.append(result['task_records'].get('finish_step', 0))
                else:
                    finish_step_list.append(0)
                    
                responses_list.append(result.get('responses', [[]]))
                if 'task_records' in result:
                    task_records_list.append(result['task_records'])
        
        # Create tensors from lists
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        merged = {
            'complete': torch.tensor(complete_list, dtype=torch.bool, device=device),
            'finish_step': torch.tensor(finish_step_list, dtype=torch.long, device=device),
            'responses': responses_list,
            'task_records': task_records_list
        }
        
        return merged
    
    def _create_failure_result_from_config(self, env_config: EnvConfig) -> Dict:
        """Create a failure result from an environment config"""
        return {
            'complete': torch.tensor(False),
            'finish_step': torch.tensor(0),
            'responses': [[]],
            'task_records': {
                'active': False,
                'complete': False,
                'finish_step': 0,
                'task_file_name': f'task_{env_config.task_id}_trial_{env_config.trial_id}_gen_{env_config.gen_idx}',
                'failed': True
            }
        }
    
    def _run_parallel_episodes(self, env_configs: List[EnvConfig], instructions: List[str], batch_size: int, 
                               temperature: float = 0.0, is_valid: bool = False, global_steps: int = 0) -> Dict:
        """
        Run multiple VLA episodes in parallel using multiprocessing
        
        This executes a chunk of parallel environments (not a full "batch").
        Uses separate processes for each environment to avoid shared OpenGL/MuJoCo state.
        
        Args:
            env_configs: List of environment configurations
            instructions: List of task instructions  
            batch_size: Number of parallel environments in this chunk
            temperature: Sampling temperature for action generation
            is_valid: Whether to save validation videos
            global_steps: Current training step (for video naming)
        """

        logger.info(f"Starting {batch_size} parallel VLA episodes: max_steps={self.max_steps}, temperature={temperature}, save_videos={is_valid}")
        
        # Extract task information from env_configs
        task_suite_names = []
        task_ids = []
        trial_ids = []
        gen_idxs = []
        
        for env_config in env_configs:
            task_suite_names.append(env_config.task_suite)
            task_ids.append(env_config.task_id)
            trial_ids.append(env_config.trial_id)
            gen_idxs.append(env_config.gen_idx)
        
        # Spawn worker processes for each environment
        processes = []
        input_queues = []
        output_queues = []
        
        for idx in range(batch_size):
            task_name = task_suite_names[idx]
            t_id = task_ids[idx]
            tr_id = trial_ids[idx]
            input_q = Queue()
            output_q = Queue()
            
            # Determine worker function based on task type
            if 'libero' in task_name.lower():
                worker_fn = libero_env_worker
                args = (task_name, t_id, tr_id, input_q, output_q, is_valid, global_steps, self.max_steps)
            elif 'robotwin' in task_name.lower():
                worker_fn = robotwin_env_worker
                args = (task_name, t_id, tr_id, input_q, output_q, is_valid, global_steps, self.max_steps)
            else:
                logger.warning(f"Unknown task type {task_name}, defaulting to LIBERO")
                worker_fn = libero_env_worker
                args = (task_name, t_id, tr_id, input_q, output_q, is_valid, global_steps, self.max_steps)
            
            p = Process(target=worker_fn, args=args)
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)
        
        logger.info(f"Spawned {len(processes)} worker processes")
        
        # Collect initial observations from workers
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        
        for idx in range(batch_size):
            try:
                init_data = output_queues[idx].get(timeout=360)
                assert init_data['type'] == 'init', f"Expected 'init', got '{init_data['type']}'"
                
                task_descriptions.append(init_data["task_description"])
                inputs.append(self._obs_to_input(init_data['obs'], is_robotwin='robotwin' in task_suite_names[idx].lower()))
                task_records.append({
                    "active": init_data['active'],
                    "complete": init_data['complete'],
                    "finish_step": init_data['finish_step'],
                    "task_file_name": init_data['task_file_name'],
                    "failed": False  # Track if worker failed
                })
                
                # Collect initial video frames
                if is_valid:
                    valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
                    
            except Exception as e:
                logger.error(f"Failed to get initial data for task {idx}: {e}")
                raise
        
        # Episode execution loop
        step = 0
        vla_history = []
        
        logger.info(f"Starting episode execution loop (max_steps={self.max_steps})")
        
        while step < self.max_steps:
            # Find active environments
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            if not active_indices:
                logger.info(f"[Step {step}] All environments completed")
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
                input_queues[idx].put(actions[idx])
            
            # Collect results from active workers
            new_inputs = inputs.copy()
            for idx in active_indices:
                try:
                    result = output_queues[idx].get(timeout=30)
                    assert result['type'] == 'step', f"Expected 'step', got '{result['type']}'"
                    
                    new_inputs[idx] = self._obs_to_input(result['obs'], is_robotwin='robotwin' in task_suite_names[idx].lower())
                    task_records[idx]['active'] = result['active']
                    task_records[idx]['complete'] = result['complete']
                    task_records[idx]['finish_step'] = result['finish_step']
                    
                    # Collect video frames
                    if is_valid and len(result['valid_images']) > 0:
                        valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
                        logger.debug(f"Task {idx} collected {len(result['valid_images'])} step frames (total now: {len(valid_video[task_records[idx]['task_file_name']])})")
                    
                    if not result['active']:
                        status = "✅ SUCCESS" if result['complete'] else "❌ FAILED"
                        logger.info(f"Task {idx} [task_id={task_ids[idx]}, trial_id={trial_ids[idx]}, gen={gen_idxs[idx]}]: {status} (steps={result['finish_step']})")
                        
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e) if str(e) else "No error message"
                    
                    # Check if worker process is still alive
                    process_alive = processes[idx].is_alive()
                    exit_code = processes[idx].exitcode
                    
                    logger.error(
                        f"[Task {idx}, task_id={task_ids[idx]}, trial_id={trial_ids[idx]}, gen={gen_idxs[idx]}] "
                        f"Worker failure: {error_type}: {error_msg} "
                        f"(worker_alive={process_alive}, exit_code={exit_code})"
                    )
                    
                    # Mark as failed
                    task_records[idx]['active'] = False
                    task_records[idx]['complete'] = False
                    task_records[idx]['failed'] = True
            
            inputs = new_inputs
            step += NUM_ACTIONS_CHUNK
        
        # Terminate workers
        logger.info("Terminating worker processes...")
        for q in input_queues:
            q.put(None)  # Send termination signal
        
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                logger.warning(f"Process {p.pid} didn't terminate, killing...")
                p.terminate()
                p.join(timeout=5)
        
        # Log final statistics
        successes = sum(1 for r in task_records if r['complete'])
        total_steps = sum(r['finish_step'] for r in task_records)
        avg_steps = total_steps / batch_size if batch_size > 0 else 0
        
        logger.info(f"Episode batch completed: Success={successes}/{batch_size} ({100*successes/batch_size if batch_size > 0 else 0:.1f}%), Avg steps={avg_steps:.1f}")
        
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
        
        # Prepare output batch
        return self._prepare_output_batch(vla_history, task_records, batch_size)
    
    def _obs_to_input(self, obs: Dict, is_robotwin: bool = False) -> Dict:
        """Convert environment observation to VLA model input format"""
        if is_robotwin:
            # RoboTwin format
            return {
                'full_image': obs.get('agentview_image', obs.get('image', np.zeros((256, 256, 3)))),
                'state': obs.get('robot0_proprio', obs.get('state', np.zeros(32)))
            }
        else:
            # LIBERO format
            # IMPORTANT: Resize to 224x224 to match SimpleVLA-RL's get_libero_image()
            img = obs.get('agentview_image', np.zeros((256, 256, 3)))
            
            # Resize using TensorFlow (EXACT match to SimpleVLA-RL's resize_image function)
            if img.shape[0] != 224 or img.shape[1] != 224:
                import tensorflow as tf
                # Exactly match SimpleVLA-RL's resize_image implementation:
                # - Encode as JPEG (as done in RLDS dataset builder)
                # - Decode back
                # - Resize with lanczos3 and antialias
                # - Clip and round
                img = tf.image.encode_jpeg(img)
                img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
                img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
                img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
                img = img.numpy()
            
            return {
                'full_image': img
            }
    
    def _prepare_output_batch(self, vla_history: List[Dict], task_records: List[Dict], batch_size: int) -> Dict:
        """Prepare output batch matching SimpleVLA-RL format"""
        if not vla_history:
            # Return empty batch if no history
            return {
                'responses': torch.empty(batch_size, 0),
                'input_ids': torch.empty(batch_size, 0, dtype=torch.long),
                'attention_mask': torch.empty(batch_size, 0, dtype=torch.bool),
                'pixel_values': torch.empty(batch_size, 0, 3, 224, 224),
                'complete': torch.tensor([r['complete'] for r in task_records], dtype=torch.bool),
                'finish_step': torch.tensor([r['finish_step'] for r in task_records], dtype=torch.long),
                'task_records': task_records  # Include for retry wrapper
            }
        
        # Stack history data
        batch = {
            'responses': [],
            'input_ids': [],
            'attention_mask': [], 
            'pixel_values': []
        }
        
        for step_data in vla_history:
            batch['responses'].append(step_data['responses'])
            batch['input_ids'].append(step_data['input_ids'])
            batch['attention_mask'].append(step_data['attention_mask'])
            batch['pixel_values'].append(step_data['pixel_values'])
        
        # Convert to tensors and transpose (steps, batch) -> (batch, steps)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        batch['complete'] = torch.tensor([r['complete'] for r in task_records], dtype=torch.bool, device=device)
        batch['finish_step'] = torch.tensor([r['finish_step'] for r in task_records], dtype=torch.long, device=device)
        batch['task_records'] = task_records  # Include for retry wrapper
        
        return batch
    
    def _create_failure_result(self, payload: RLPayload) -> RolloutResult:
        """Create failure result for error cases"""
        
        return RolloutResult(
            prompt=getattr(payload, 'prompt', 'VLA task failed'),
            completions=['Task execution failed'],
            log_probs=[[np.log(0.01)]],  # Very low probability for failure
            input_tokens=10,
            output_tokens=3,
            
            # VLA failure data
            rewards=[0.0],
            episode_length=0,
            environment_info={
                'task_suite': self.task_suite,
                'success': False,
                'total_reward': 0.0,
                'error': True
            }
        )
    