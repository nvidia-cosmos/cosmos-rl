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
from typing import Dict, List, Union, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms

from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger


class VLAActionProcessor:
    """
    VLA Action Processing utilities for converting between model outputs and environment actions.
    
    Handles action normalization, denormalization, and format conversion for different
    robotic environments (LIBERO, RoboTwin).
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.vla_config = config.vla
        
        # Action space configuration
        self.action_dim = self.vla_config.action_dim
        self.action_normalization = self.vla_config.action_normalization
        
        # Action bounds for normalization
        self.action_bounds = self._get_action_bounds()
        
        # Image processing
        self.image_size = self.vla_config.image_size
        self.image_transform = self._get_image_transform()
        
        logger.info(f"Initialized VLA action processor with action_dim={self.action_dim}")
    
    def _get_action_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get action bounds for different action dimensions"""
        return {
            'position': (-1.0, 1.0),  # x, y, z
            'rotation': (-np.pi, np.pi),  # rx, ry, rz
            'gripper': (0.0, 1.0)  # open/close
        }
    
    def _get_image_transform(self) -> transforms.Compose:
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Process raw environment observation into VLA model input format
        
        Args:
            obs: Raw observation dictionary containing images and proprioception
            
        Returns:
            Processed observation ready for VLA model
        """
        processed_obs = {}
        
        # Process images
        if 'agentview_image' in obs:
            image = self._process_image(obs['agentview_image'])
            processed_obs['pixel_values'] = image
        
        if 'wrist_image' in obs and self.vla_config.use_wrist_camera:
            wrist_image = self._process_image(obs['wrist_image'])
            processed_obs['wrist_pixel_values'] = wrist_image
        
        # Process proprioception if available
        if 'robot0_proprio' in obs:
            proprio = torch.from_numpy(obs['robot0_proprio']).float()
            processed_obs['proprioception'] = proprio
        
        return processed_obs
    
    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """Process a single image"""
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                # Normalize to [0, 255] if needed
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.image_transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def process_vla_output(self, model_output: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Process VLA model output into environment-ready actions
        
        Args:
            model_output: Raw output from VLA model
            
        Returns:
            Processed actions ready for environment execution
        """
        processed_output = {}
        
        # Extract action predictions
        if 'action_logits' in model_output:
            actions = self._extract_actions_from_logits(model_output['action_logits'])
        elif 'actions' in model_output:
            actions = model_output['actions'].detach().cpu().numpy()
        else:
            raise ValueError("No action predictions found in model output")
        
        processed_output['actions'] = actions
        
        # Extract text responses if available
        if 'responses' in model_output:
            processed_output['responses'] = model_output['responses']
        
        return processed_output
    
    def _extract_actions_from_logits(self, action_logits: torch.Tensor) -> np.ndarray:
        """Extract continuous actions from model logits"""
        # For now, assume direct regression output
        # TODO: Implement proper action extraction based on model architecture
        actions = action_logits.detach().cpu().numpy()
        
        # Apply denormalization if needed
        actions = self.denormalize_actions(actions)
        
        return actions
    
    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions to model input range"""
        if self.action_normalization == 'minmax':
            # Normalize to [-1, 1] range
            normalized_actions = np.zeros_like(actions)
            
            # Position (first 3 dims)
            pos_min, pos_max = self.action_bounds['position']
            normalized_actions[:, :3] = 2 * (actions[:, :3] - pos_min) / (pos_max - pos_min) - 1
            
            # Rotation (next 3 dims)
            rot_min, rot_max = self.action_bounds['rotation']
            normalized_actions[:, 3:6] = 2 * (actions[:, 3:6] - rot_min) / (rot_max - rot_min) - 1
            
            # Gripper (last dim)
            if actions.shape[1] > 6:
                grip_min, grip_max = self.action_bounds['gripper']
                normalized_actions[:, 6] = 2 * (actions[:, 6] - grip_min) / (grip_max - grip_min) - 1
            
            return normalized_actions
        
        return actions
    
    def denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Denormalize actions from model output range to environment range"""
        if self.action_normalization == 'minmax':
            # Denormalize from [-1, 1] range
            actions = np.zeros_like(normalized_actions)
            
            # Position (first 3 dims)
            pos_min, pos_max = self.action_bounds['position']
            actions[:, :3] = (normalized_actions[:, :3] + 1) / 2 * (pos_max - pos_min) + pos_min
            
            # Rotation (next 3 dims)
            rot_min, rot_max = self.action_bounds['rotation']
            actions[:, 3:6] = (normalized_actions[:, 3:6] + 1) / 2 * (rot_max - rot_min) + rot_min
            
            # Gripper (last dim)
            if normalized_actions.shape[1] > 6:
                grip_min, grip_max = self.action_bounds['gripper']
                actions[:, 6] = (normalized_actions[:, 6] + 1) / 2 * (grip_max - grip_min) + grip_min
            
            return actions
        
        return normalized_actions
    
    def convert_action_format(self, actions: np.ndarray, source_format: str, target_format: str) -> np.ndarray:
        """
        Convert action format between different representations
        
        Args:
            actions: Input actions
            source_format: Source format ('euler', 'quaternion', 'axis_angle')
            target_format: Target format ('euler', 'quaternion', 'axis_angle')
            
        Returns:
            Converted actions
        """
        if source_format == target_format:
            return actions
        
        # TODO: Implement rotation format conversions
        logger.warning(f"Action format conversion from {source_format} to {target_format} not implemented")
        return actions
    
    def postprocess_actions_for_env(self, actions: np.ndarray, env_type: str) -> np.ndarray:
        """
        Post-process actions for specific environment requirements
        
        Args:
            actions: Raw actions from model
            env_type: Environment type ('libero', 'robotwin')
            
        Returns:
            Environment-specific actions
        """
        if env_type == 'libero':
            return self._postprocess_libero_actions(actions)
        elif env_type == 'robotwin':
            return self._postprocess_robotwin_actions(actions)
        else:
            return actions
    
    def _postprocess_libero_actions(self, actions: np.ndarray) -> np.ndarray:
        """Post-process actions for LIBERO environments"""
        # Apply LIBERO-specific action processing
        processed_actions = actions.copy()
        
        # Ensure gripper action is binary for LIBERO
        if processed_actions.shape[1] > 6:
            processed_actions[:, 6] = (processed_actions[:, 6] > 0.5).astype(float)
        
        return processed_actions
    
    def _postprocess_robotwin_actions(self, actions: np.ndarray) -> np.ndarray:
        """Post-process actions for RoboTwin environments"""
        # Apply RoboTwin-specific action processing
        processed_actions = actions.copy()
        
        # RoboTwin may have different action space requirements
        # TODO: Implement RoboTwin-specific processing
        
        return processed_actions
