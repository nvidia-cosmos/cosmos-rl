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
import gc
import threading
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

from cosmos_rl.utils.logging import logger


class BaseEnvWrapper(ABC):
    """
    Base environment wrapper for VLA rollout environments.
    Provides thread-safe interface for robotic environment interaction.
    """
    
    def __init__(self, task_config: Dict[str, Any], trial_id: int = 0):
        self.task_config = task_config
        self.trial_id = trial_id
        self.lock = threading.Lock()
        
        # Environment state
        self.env = None
        self.active = False
        self.complete = False
        self.finish_step = 0
        self.instruction = ""
        
        # Task info
        self.task_name = task_config.get('task_name', 'unknown')
        self.task_id = task_config.get('task_id', 0)
        self.max_steps = task_config.get('max_steps', 512)
        
    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """Initialize the environment and return initial observation"""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool]:
        """Execute action and return (observation, done)"""
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation"""
        pass
    
    @abstractmethod
    def close(self):
        """Close the environment"""
        pass
    
    def get_instruction(self) -> str:
        """Get task instruction"""
        return self.instruction
    
    def is_active(self) -> bool:
        """Check if environment is still active"""
        return self.active
    
    def is_complete(self) -> bool:
        """Check if task is completed successfully"""
        return self.complete


class LiberoEnvWrapper(BaseEnvWrapper):
    """
    Environment wrapper for LIBERO benchmark tasks.
    Provides thread-safe interface for LIBERO environment interaction.
    """
    
    def __init__(self, task_config: Dict[str, Any], trial_id: int = 0):
        super().__init__(task_config, trial_id)
        
        # LIBERO-specific configuration
        self.benchmark_name = task_config.get('benchmark_name', 'libero_10')
        self.model_family = task_config.get('model_family', 'openvla')
        self.resolution = task_config.get('resolution', 256)
        
        logger.info(f"Initializing LIBERO environment: {self.benchmark_name}, task {self.task_id}")
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize LIBERO environment"""
        with self.lock:
            try:
                # Use the dedicated LIBERO integration module
                from .libero_integration import setup_libero_environment
                
                self.env, self.instruction, obs = setup_libero_environment(
                    task_suite=self.benchmark_name,
                    task_id=self.task_id,
                    trial_id=self.trial_id
                )
                
                self.active = True
                self.complete = False
                self.finish_step = 0
                
                logger.info(f"LIBERO environment initialized successfully: {self.task_name}")
                return {
                    'obs': obs,
                    'instruction': self.instruction,
                    'active': self.active,
                    'complete': self.complete
                }
                
            except ImportError as e:
                logger.error(f"LIBERO not available: {e}")
                # Create a dummy environment for testing
                self.env = None
                self.instruction = f"Complete {self.task_name} task"
                self.active = True
                self.complete = False
                self.finish_step = 0
                
                # Return dummy observation
                dummy_obs = {
                    'agentview_image': np.zeros((256, 256, 3), dtype=np.uint8),
                    'robot0_proprio': np.zeros(32)
                }
                
                return {
                    'obs': dummy_obs,
                    'instruction': self.instruction,
                    'active': self.active,
                    'complete': self.complete
                }
                
            except Exception as e:
                logger.error(f"Failed to initialize LIBERO environment: {e}")
                traceback.print_exc()
                self.active = False
                raise
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool]:
        """Execute action in LIBERO environment"""
        with self.lock:
            try:
                # Process action for LIBERO
                processed_action = self._process_action_for_libero(action)
                
                # Execute action
                obs, reward, done, info = self.env.step(processed_action)
                
                self.finish_step += 1
                
                # Check termination conditions
                if done or self.finish_step >= self.max_steps:
                    self.active = False
                    self.complete = done
                
                return obs, done
                
            except Exception as e:
                logger.error(f"Error in LIBERO environment step: {e}")
                self.active = False
                return {}, True
    
    def reset(self) -> Dict[str, Any]:
        """Reset LIBERO environment"""
        with self.lock:
            if self.env is not None:
                obs = self.env.reset()
                self.active = True
                self.complete = False
                self.finish_step = 0
                return obs
            else:
                raise RuntimeError("Environment not initialized")
    
    def close(self):
        """Close LIBERO environment"""
        with self.lock:
            if self.env is not None:
                try:
                    self.env.close()
                    logger.info("LIBERO environment closed")
                except Exception as e:
                    logger.error(f"Error closing LIBERO environment: {e}")
                finally:
                    self.env = None
                    torch.cuda.empty_cache()
                    gc.collect()
    
    def _create_libero_env(self, task, model_family: str, resolution: int):
        """Create LIBERO environment"""
        try:
            # Import local LIBERO utilities
            from cosmos_rl.rollout.vla_rollout.libero_utils import get_libero_env
            return get_libero_env(task, model_family, resolution)
        except ImportError:
            logger.error("LIBERO utilities not available. Please install LIBERO.")
            raise
    
    def _get_dummy_action(self) -> List[float]:
        """Get dummy action for LIBERO"""
        try:
            from cosmos_rl.rollout.vla_rollout.libero_utils import get_libero_dummy_action
            return get_libero_dummy_action(self.model_family)
        except ImportError:
            # Fallback dummy action
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    
    def _process_action_for_libero(self, action: np.ndarray) -> List[float]:
        """Process action for LIBERO environment"""
        try:
            from cosmos_rl.rollout.vla_rollout.libero_utils import normalize_gripper_action, invert_gripper_action
            
            # Ensure action is the right shape
            if action.ndim > 1:
                action = action[0]  # Take first action if batch
            
            # Process gripper action
            normalized_action = normalize_gripper_action(action, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            
            return inverted_action.tolist()
            
        except ImportError:
            logger.warning("LIBERO action processing utilities not available, using raw action")
            return action.tolist()


class RobotwinEnvWrapper(BaseEnvWrapper):
    """
    Environment wrapper for RoboTwin benchmark tasks.
    Provides thread-safe interface for RoboTwin environment interaction.
    """
    
    def __init__(self, task_config: Dict[str, Any], trial_id: int = 0):
        super().__init__(task_config, trial_id)
        
        # RoboTwin-specific configuration
        self.robotwin_version = task_config.get('robotwin_version', '2.0')
        self.instruction_type = task_config.get('instruction_type', 'language')
        
        logger.info(f"Initializing RoboTwin {self.robotwin_version} environment: {self.task_name}")
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize RoboTwin environment"""
        with self.lock:
            try:
                # Import RoboTwin modules based on version
                if self.robotwin_version == '2.0':
                    self.env = self._create_robotwin2_env()
                else:
                    self.env = self._create_robotwin1_env()
                
                # Initialize environment
                self.env.set_task_name(self.task_name)
                obs = self.env.get_obs()
                
                # Generate instruction
                self.instruction = self._generate_instruction()
                if self.instruction:
                    self.env.set_instruction(instruction=self.instruction)
                
                self.active = True
                self.complete = False
                self.finish_step = 0
                
                logger.info(f"RoboTwin environment initialized: {self.task_name}")
                return {
                    'obs': obs,
                    'instruction': self.instruction,
                    'active': self.active,
                    'complete': self.complete
                }
                
            except Exception as e:
                logger.error(f"Failed to initialize RoboTwin environment: {e}")
                traceback.print_exc()
                self.active = False
                raise
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool]:
        """Execute action in RoboTwin environment"""
        with self.lock:
            try:
                # Execute action
                self.env.take_action(action)
                done = self.env.eval_success
                
                # Get new observation
                obs = self.env.get_obs()
                
                self.finish_step += action.shape[0] if action.ndim > 1 else 1
                
                # Check termination conditions
                if done or self.finish_step >= self.max_steps:
                    self.active = False
                    self.complete = done
                
                return obs, done
                
            except Exception as e:
                logger.error(f"Error in RoboTwin environment step: {e}")
                traceback.print_exc()
                self.active = False
                return {}, True
    
    def reset(self) -> Dict[str, Any]:
        """Reset RoboTwin environment"""
        with self.lock:
            if self.env is not None:
                # RoboTwin reset implementation
                obs = self.env.get_obs()
                self.active = True
                self.complete = False
                self.finish_step = 0
                return obs
            else:
                raise RuntimeError("Environment not initialized")
    
    def close(self):
        """Close RoboTwin environment"""
        with self.lock:
            if self.env is not None:
                try:
                    self.env.close()
                    logger.info("RoboTwin environment closed")
                except Exception as e:
                    logger.error(f"Error closing RoboTwin environment: {e}")
                finally:
                    self.env = None
                    torch.cuda.empty_cache()
                    gc.collect()
    
    def _create_robotwin2_env(self):
        """Create RoboTwin 2.0 environment"""
        try:
            # Import RoboTwin 2.0 modules
            from robotwin2.envs.env import RoboTwinEnv
            
            env_config = {
                'task_name': self.task_name,
                'trial_id': self.trial_id,
                'headless': True,  # Headless for distributed training
                **self.task_config.get('env_config', {})
            }
            
            return RoboTwinEnv(**env_config)
            
        except ImportError:
            logger.error("RoboTwin 2.0 not available. Please install RoboTwin.")
            raise
    
    def _create_robotwin1_env(self):
        """Create RoboTwin 1.0 environment"""
        try:
            # Import RoboTwin 1.0 modules
            from robotwin.envs.env import RoboTwinEnv
            
            env_config = {
                'task_name': self.task_name,
                'trial_id': self.trial_id,
                **self.task_config.get('env_config', {})
            }
            
            return RoboTwinEnv(**env_config)
            
        except ImportError:
            logger.error("RoboTwin 1.0 not available. Please install RoboTwin.")
            raise
    
    def _generate_instruction(self) -> str:
        """Generate instruction for the task"""
        try:
            # Try to generate dynamic instruction
            from robotwin2.description.instruction_generation import generate_episode_descriptions
            
            episode_info_list = [{'task_name': self.task_name}]
            results = generate_episode_descriptions(self.task_name, episode_info_list, 1, seed=self.trial_id)
            
            if results and len(results) > 0 and self.instruction_type in results[0]:
                return np.random.choice(results[0][self.instruction_type])
            
        except ImportError:
            logger.warning("RoboTwin instruction generation not available")
        
        # Fallback to default instruction
        return f"Complete the {self.task_name} task"
