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
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from threading import Lock
from collections import defaultdict
import gc
import traceback
import time
from multiprocessing import Process, Queue

from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.schema import RLPayload
from transformers import AutoTokenizer, AutoProcessor

from .environment_wrappers import LiberoEnvWrapper, RobotwinEnvWrapper
from .action_processing import VLAActionProcessor
from .utils import encode_observation, create_vla_prompt, compute_vla_reward
from .vla_model_inference import VLAModelInference
from .libero_utils import save_rollout_video
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
        self.do_sample = False  # Default VLA sampling behavior
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
        
        # Environment pool - maintain persistent environments for efficiency
        self.env_pool = {}
        self.env_pool_lock = Lock()
        
        logger.info(f"Initialized VLA rollout for task suite: {self.task_suite}")
        logger.info(f"Max steps per episode: {self.max_steps}")
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
            import traceback
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
        
        # Load frozen vision backbone weights directly from checkpoint
        # These weights are never updated during training, so no need to sync via NCCL
        logger.info(f"Loading frozen vision backbone weights from {model_path}...")
        self._load_frozen_vision_backbone_weights(vla_model.model, model_path)
        logger.info("✅ Frozen vision backbone weights loaded")
        
        # Save both the wrapper and inner module
        self.vla_model = vla_model  # Keep wrapper for weight_sync_transforms
        self.module = vla_model.model  # Inner OpenVLAForActionPrediction for inference
        self.module.eval()
        
        # Save processor, tokenizer, and config for later use
        self.processor = processor
        self.tokenizer = tokenizer
        self.hf_config = vla_config  # Save for shard info generation
        
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        logger.info(f"✅ VLA model structure created (device: {device})")
        logger.info(f"   Trainable weights initialized randomly (will be synced from policy via NCCL)")
        logger.info(f"   Frozen vision backbone loaded from checkpoint")
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
    
    def _load_frozen_vision_backbone_weights(self, model, model_path: str):
        """
        Load frozen vision backbone weights directly from HuggingFace checkpoint
        
        These weights are frozen during training, so we load them once from the checkpoint
        instead of syncing them via NCCL from the policy worker.
        
        Args:
            model: The VLA model (OpenVLAForActionPrediction)
            model_path: Path to HuggingFace checkpoint
        """
        import torch
        from safetensors import safe_open
        from pathlib import Path
        from huggingface_hub import snapshot_download
        
        try:
            # Resolve model path
            path = Path(model_path)
            if not path.exists():
                logger.info(f"Downloading model from HuggingFace Hub: {model_path}")
                path = Path(snapshot_download(repo_id=model_path))
            
            # Load vision backbone weights from checkpoint
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            vision_backbone_weights = {}
            
            # Try safetensors first
            safetensor_files = list(path.glob("*.safetensors"))
            if safetensor_files:
                for st_file in safetensor_files:
                    with safe_open(st_file, framework="pt", device=str(device)) as f:
                        for key in f.keys():
                            if key.startswith("vision_backbone"):
                                vision_backbone_weights[key] = f.get_tensor(key)
            else:
                # Fallback to pytorch checkpoints
                pt_files = list(path.glob("pytorch_model*.bin"))
                if pt_files:
                    for pt_file in pt_files:
                        state_dict = torch.load(pt_file, map_location=device)
                        for key, value in state_dict.items():
                            if key.startswith("vision_backbone"):
                                vision_backbone_weights[key] = value
                else:
                    raise FileNotFoundError(f"No safetensors or pytorch_model.bin found in {path}")
            
            if not vision_backbone_weights:
                logger.error("❌ CRITICAL: No vision_backbone weights found in checkpoint!")
                logger.error(f"   This means vision backbone has RANDOM TIMM weights!")
                logger.error(f"   Checkpoint path: {path}")
                logger.error(f"   Files found: safetensors={len(safetensor_files)}, pytorch={len(pt_files) if 'pt_files' in locals() else 0}")
                logger.error(f"   Model will have INCORRECT vision features!")
                # Don't return - raise an error to fail fast
                raise RuntimeError(f"Vision backbone weights not found in checkpoint {model_path}")
            
            # Load the weights into model
            missing_keys, unexpected_keys = model.load_state_dict(vision_backbone_weights, strict=False)
            
            logger.info(f"✅ Loaded {len(vision_backbone_weights)} vision backbone parameters")
            
            # Warn if any vision_backbone keys are missing
            if missing_keys:
                missing_vision = [k for k in missing_keys if k.startswith("vision_backbone")]
                if missing_vision:
                    logger.warning(f"⚠️  Missing {len(missing_vision)} vision backbone keys: {missing_vision[:5]}")
            
        except Exception as e:
            logger.error(f"Failed to load frozen vision backbone weights: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def is_engine_initialized(self) -> bool:
        """Check if the VLA engine has been initialized"""
        return self.module is not None and self.processor is not None
    
    def _initialize_environments_for_payloads(self, payloads: List[RLPayload]) -> List[Any]:
        """Initialize environments based on task information from payloads
        
        Returns:
            List of environment wrappers, one per payload in the same order
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Initializing environments for batch")
        logger.info(f"  Number of payloads: {len(payloads)}")
        logger.info(f"{'=' * 80}\n")
        
        # Create environments as a list to maintain 1-1 correspondence with payloads
        environments = []
        
        for i, payload in enumerate(payloads):
            task_config = self._extract_task_config_from_payload(payload)
            
            logger.info(f"[Payload {i}] Task config:")
            logger.info(f"  - Task suite: {task_config['task_suite_name']}")
            logger.info(f"  - Task ID: {task_config['task_id']}")
            logger.info(f"  - Trial ID: {task_config['trial_id']}")
            logger.info(f"  - Trial seed: {task_config['trial_seed']}")
            logger.info(f"  - Max steps: {task_config['max_steps']}")
            
            # Create environment wrapper based on task suite
            trial_id = task_config.get('trial_id', 0)
            if 'libero' in task_config['task_suite_name'].lower():
                env_wrapper = LiberoEnvWrapper(task_config, trial_id=trial_id)
            elif 'robotwin' in task_config['task_suite_name'].lower():
                env_wrapper = RobotwinEnvWrapper(task_config, trial_id=trial_id)
            else:
                logger.warning(f"Unknown task suite {task_config['task_suite_name']}, using default LIBERO")
                env_wrapper = LiberoEnvWrapper(task_config, trial_id=trial_id)
            
            # Note: Environment will be initialized later in _run_vla_episode_batch
            # (we only create the wrapper here, not initialize yet)
            
            # Add to list - each payload gets its own environment instance
            environments.append(env_wrapper)
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ Created {len(environments)} environment wrappers")
        logger.info(f"{'=' * 80}\n")
        return environments
    
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
    
    def rollout_generation(self, payloads: List[RLPayload], *args, **kwargs) -> List[RolloutResult]:
        """
        Generate VLA rollouts by interacting with robotic environments
        
        This is the main entry point called by the rollout worker.
        It processes RLPayload objects from the dispatcher and returns RolloutResult objects.
        
        Args:
            payloads: List of RLPayload objects containing task instructions and prompts
            
        Returns:
            List of RolloutResult containing trajectories and outcomes
        """
        if not self.processor:
            raise RuntimeError("VLA processor not initialized. Call init_engine() first.")
        
        logger.info(f"Starting VLA rollout generation for {len(payloads)} payloads")
        
        try:
            # Initialize environments based on task information from payloads
            environments = self._initialize_environments_for_payloads(payloads)
            
            # Process payloads in batch (matching SimpleVLA-RL approach)
            results = self._process_payload_batch(payloads, environments, **kwargs)
        except Exception as e:
            logger.error(f"Error in batch rollout generation: {e}")
            traceback.print_exc()
            # Return failure results for all payloads
            results = [self._create_failure_result(payload) for payload in payloads]
        
        logger.info(f"Generated {len(results)} VLA rollout results")
        return results
    
    def _process_payload_batch(self, payloads: List[RLPayload], environments: List[Any], **kwargs) -> List[RolloutResult]:
        """Process batch of payloads using parallel environments (matching SimpleVLA-RL)"""
        batch_size = len(payloads)
        logger.info(f"Processing VLA payload batch of size {batch_size}")
        
        # Extract video generation flags from kwargs (matching SimpleVLA-RL pattern)
        # is_valid is True for validation runs (when we want to save videos)
        is_valid = kwargs.get('save_videos', False)  # Can be set to True to enable video saving
        global_steps = kwargs.get('global_steps', 0)
        
        # Create environment wrappers for batch
        env_wrappers = []
        instructions = []
        
        for i, payload in enumerate(payloads):
            # Get the corresponding environment for this payload (1-1 mapping)
            env_wrapper = environments[i]
            
            env_wrappers.append(env_wrapper)
            instructions.append(self._extract_instruction_from_payload(payload))
        
        try:
            # Run batch episode (matching SimpleVLA-RL pattern)
            batch_result = self._run_vla_episode_batch(
                env_wrappers, instructions, batch_size,
                is_valid=is_valid, global_steps=global_steps
            )
            
            # Convert batch result to individual RolloutResults
            results = []
            for i, payload in enumerate(payloads):
                episode_data = {
                    'instruction': instructions[i],
                    'success': batch_result['complete'][i].item() if 'complete' in batch_result else False,
                    'episode_length': batch_result['finish_step'][i].item() if 'finish_step' in batch_result else 0,
                    'responses': [batch_result.get('responses', [[""]*batch_size])[j][i] for j in range(len(batch_result.get('responses', [[""]*batch_size])))]
                }
                
                result = self._create_rollout_result(payload, episode_data)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing payload batch: {e}")
            traceback.print_exc()
            # Return failure results
            return [self._create_failure_result(payload) for payload in payloads]
        
        finally:
            # Return environments to pool
            for env_wrapper in env_wrappers:
                self._return_environment_to_pool(env_wrapper)
    
    def _get_available_environment(self) -> Optional[Any]:
        """Get an available environment from the pool"""
        with self.env_pool_lock:
            for env_id, env_info in self.env_pool.items():
                if not env_info['in_use']:
                    env_info['in_use'] = True
                    env_info['last_used'] = time.time()
                    return env_info['wrapper']
        
        # No available environment - this shouldn't happen with proper sizing
        logger.warning("No available environment in pool")
        return None
    
    def _return_environment_to_pool(self, env_wrapper):
        """Return environment to pool"""
        with self.env_pool_lock:
            for env_id, env_info in self.env_pool.items():
                if env_info['wrapper'] == env_wrapper:
                    env_info['in_use'] = False
                    break
    
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
    
    def _run_vla_episode_batch(self, env_wrappers: List, instructions: List[str], batch_size: int, 
                               is_valid: bool = False, global_steps: int = 0) -> Dict:
        """
        Run VLA episode batch with multiprocessing (matching SimpleVLA-RL pattern)
        
        Uses separate processes for each environment to avoid shared OpenGL/MuJoCo state.
        
        Args:
            env_wrappers: List of environment wrappers (contains task config info)
            instructions: List of task instructions  
            batch_size: Number of environments
            is_valid: Whether to save validation videos
            global_steps: Current training step (for video naming)
        """

        logger.info(f"Starting VLA episode batch with multiprocessing: {batch_size} tasks, max_steps={self.max_steps}, save_videos={is_valid}")
        
        # Extract task information from env_wrappers
        task_suite_names = []
        task_ids = []
        trial_ids = []
        
        for env_wrapper in env_wrappers:
            if isinstance(env_wrapper, LiberoEnvWrapper):
                task_suite_names.append(env_wrapper.benchmark_name)
                task_ids.append(env_wrapper.task_id)
                trial_ids.append(env_wrapper.trial_id)
            elif isinstance(env_wrapper, RobotwinEnvWrapper):
                task_suite_names.append(env_wrapper.task_name)
                task_ids.append(env_wrapper.task_id)
                trial_ids.append(env_wrapper.trial_id)
            else:
                # Fallback
                task_suite_names.append(self.task_suite)
                task_ids.append(0)
                trial_ids.append(0)
        
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
                init_data = output_queues[idx].get(timeout=120)
                assert init_data['type'] == 'init', f"Expected 'init', got '{init_data['type']}'"
                
                task_descriptions.append(init_data["task_description"])
                inputs.append(self._obs_to_input(init_data['obs'], is_robotwin='robotwin' in task_suite_names[idx].lower()))
                task_records.append({
                    "active": init_data['active'],
                    "complete": init_data['complete'],
                    "finish_step": init_data['finish_step'],
                    "task_file_name": init_data['task_file_name']
                })
                
                # Collect initial video frames
                if is_valid:
                    valid_video[init_data['task_file_name']].extend(init_data['valid_images'])
                    logger.info(f"Task {idx} collected {len(init_data['valid_images'])} initial frames for video '{init_data['task_file_name']}'")
                    
                logger.info(f"Task {idx} initialized: {task_descriptions[idx][:60]}")
            except Exception as e:
                logger.error(f"Failed to initialize task {idx}: {e}")
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
            
            logger.info(f"[Step {step}] Active: {len(active_indices)}/{batch_size}")
            
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
                        logger.info(f"Task {idx}: {status} (steps={result['finish_step']})")
                        
                except Exception as e:
                    logger.error(f"[Task {idx}] Error receiving result: {e}")
                    task_records[idx]['active'] = False
            
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
            logger.info(f"Saving {len(valid_video)} rollout videos...")
            for task_file, images in valid_video.items():
                logger.info(f"  Video '{task_file}': {len(images)} frames")
            
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
                        logger.info(f"  ✅ Saved {len(images)} frames to: {video_path}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to save {task_file}: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            logger.warning(f"⚠️  No video frames collected! is_valid={is_valid}")
        
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
    
    def _generate_dummy_batch_actions(self, inputs: List[Dict], task_descriptions: List[str]) -> Dict:
        """Generate dummy batch actions when model inference is not available"""
        batch_size = len(inputs)
        
        actions = []
        responses = []
        
        for i in range(batch_size):
            task_desc = task_descriptions[i].lower()
            
            # Generate task-specific actions
            if "pick" in task_desc or "grasp" in task_desc:
                action = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, -1.0])  # Move down and close
            elif "place" in task_desc or "put" in task_desc:
                action = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0])   # Move up and open
            else:
                action = np.array([0.02, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])  # Small movement
            
            # Add noise
            noise = np.random.normal(0, 0.01, size=6)
            action[:6] += noise
            action = np.clip(action, -1.0, 1.0)
            
            actions.append(action)
            responses.append(f"<action>{action.tolist()}</action>")
        
        return {
            "action": np.array(actions),
            "responses": responses,
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
            "pixel_values": torch.randn(batch_size, 3, 224, 224)
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
                'finish_step': torch.tensor([r['finish_step'] for r in task_records], dtype=torch.long)
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
        
        return batch
    
    def _generate_vla_action(self, prompt: str, observation: Dict) -> Dict:
        """Generate VLA action using actual model inference (matching SimpleVLA-RL)"""
        
        # This will be implemented with actual VLA model inference
        # For now, return more realistic dummy actions based on SimpleVLA-RL patterns
        
        # Simulate the VLA model output format from SimpleVLA-RL
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
        
        # More realistic action patterns based on task
        if "pick" in prompt.lower() or "grasp" in prompt.lower():
            # Picking motion: move down and close gripper
            base_action = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0])
        elif "place" in prompt.lower() or "put" in prompt.lower():
            # Placing motion: move up and open gripper  
            base_action = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
        else:
            # General motion: small movements
            base_action = np.array([0.05, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0])
        
        # Add some controlled randomness
        noise = np.random.normal(0, 0.02, size=6)
        base_action[:6] += noise
        
        # Ensure action is in valid range [-1, 1]
        base_action = np.clip(base_action, -1.0, 1.0)
        
        return {
            'actions': base_action,
            'response': f"<action>{base_action.tolist()}</action>",  # SimpleVLA-RL format
            'logits': np.random.randn(7),
            'input_ids': torch.randint(0, 1000, (1, 10)),  # Dummy tokens
            'attention_mask': torch.ones(1, 10),
            'pixel_values': torch.randn(1, 3, 224, 224),  # Dummy pixel values
        }
    
    def _create_rollout_result(self, payload: RLPayload, episode_data: Dict) -> RolloutResult:
        """Create RolloutResult from episode data"""
        
        # Extract required fields for RolloutResult
        completions = episode_data.get('responses', [''])
        rewards = episode_data.get('rewards', [0.0])
        
        # Create mock log probabilities (would come from actual model)
        log_probs = [np.log(0.5)] * len(completions)  # Dummy log probs
        
        result = RolloutResult(
            prompt=episode_data.get('instruction', ''),
            completions=completions,
            log_probs=[log_probs],
            input_tokens=100,  # Approximate
            output_tokens=len(' '.join(completions).split()),
            
            # VLA-specific additional data
            rewards=rewards,
            episode_length=episode_data.get('episode_length', 0),
            environment_info={
                'task_suite': self.task_suite,
                'success': episode_data.get('success', False),
                'total_reward': episode_data.get('total_reward', 0.0),
                'num_actions': len(episode_data.get('actions', [])),
                'final_observation': episode_data.get('observations', [])[-1] if episode_data.get('observations') else None
            }
        )
        
        return result
    
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
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up VLA rollout resources")
        
        # Close all environments in pool
        with self.env_pool_lock:
            for env_id, env_info in self.env_pool.items():
                try:
                    env_info['wrapper'].close()
                except Exception as e:
                    logger.error(f"Error closing environment {env_id}: {e}")
        
        # Shutdown thread pool
        if hasattr(self, 'env_thread_pool'):
            self.env_thread_pool.shutdown(wait=True)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("VLA rollout cleanup completed")