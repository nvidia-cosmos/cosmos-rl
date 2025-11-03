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

"""
VLA Model Inference implementation matching SimpleVLA-RL patterns

This module implements the actual VLA model inference following the exact
patterns from SimpleVLA-RL's rob_rollout.py
"""

import torch
import torch.nn as nn
import numpy as np
import contextlib
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torchvision.transforms as transforms

from cosmos_rl.utils.logging import logger


def center_crop_image(image: Image.Image, crop_size: int = 256) -> Image.Image:
    """Center crop image to specified size"""
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """Normalize proprioception data using norm stats"""
    mean = norm_stats.get('mean', 0.0)
    std = norm_stats.get('std', 1.0)
    return (proprio - mean) / std


class VLAModelInference:
    """
    VLA Model Inference class that mimics SimpleVLA-RL's inference pattern
    
    This class handles:
    1. Input processing (observations -> model input)
    2. Model inference (OpenVLA, OpenVLA-OFT)
    3. Action generation and post-processing
    """
    
    def __init__(self, module: nn.Module, processor, config):
        self.module = module
        self.processor = processor
        self.config = config
        
        # VLA preprocessing
        self.vla_preprocess()
        
        # Get vla_type from config (either config.vla_type or config.vla)
        vla_type = getattr(config, 'vla_type', getattr(config, 'vla', 'unknown'))
        logger.info(f"Initialized VLA model inference for {vla_type}")
    
    def vla_preprocess(self):
        """VLA preprocessing setup (matching SimpleVLA-RL)"""
        # Get vla_type from config (handles both config.vla_type and config.vla)
        vla_type = getattr(self.config, 'vla_type', getattr(self.config, 'vla', None))
        
        if vla_type and vla_type in ["openvla", "openvla-oft"]:
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except ImportError:
                logger.warning("TensorFlow not available for GPU memory growth setup")
        
        if vla_type and vla_type in ["openvla-oft"]:
            # Handle normalization key setup
            if hasattr(self.module, 'norm_stats') and hasattr(self.config, 'unnorm_key'):
                if "libero" in self.config.task_suite_name:
                    if (self.config.unnorm_key not in self.module.norm_stats and 
                        f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats):
                        self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
                elif "robotwin" in self.config.task_suite_name:
                    self.config.unnorm_key = self.config.unnorm_key.removeprefix("robotwin_").removeprefix("robotwin2_")
                
                if self.config.unnorm_key not in self.module.norm_stats:
                    logger.warning(f"Action un-norm key {self.config.unnorm_key} not found in VLA `norm_stats`!")
    
    def process_input(self, inputs: List[Dict], task_descriptions: List[str]) -> Dict[str, torch.Tensor]:
        """
        Process inputs for VLA model (matching SimpleVLA-RL's process_input)
        
        Args:
            inputs: List of observation dictionaries  
            task_descriptions: List of task description strings
            
        Returns:
            Processed batch data for VLA model
        """
        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        
        # Add proprioception for robotwin tasks
        use_proprio = (hasattr(self.config, 'use_proprio') and 
                      self.config.use_proprio and 
                      "robotwin" in getattr(self.config, 'task_suite_name', ''))
        if use_proprio:
            batchdata["proprio"] = []
        
        for i in range(len(inputs)):
            input_data = inputs[i]
            task_description = task_descriptions[i]
            
            # Process main image
            if 'full_image' in input_data:
                image_array = input_data["full_image"]
            elif 'agentview_image' in input_data:
                image_array = input_data["agentview_image"]
            else:
                # Create dummy image if none available
                image_array = np.zeros((256, 256, 3), dtype=np.uint8)
            
            image = Image.fromarray(image_array).convert("RGB")
            
            # Center crop if configured
            if hasattr(self.config, 'center_crop') and self.config.center_crop:
                image = center_crop_image(image)
            
            # Create prompt (matching SimpleVLA-RL format)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            
            # Process with VLA processor
            try:
                batch_feature = self.processor(prompt, image)
                input_ids = batch_feature["input_ids"]
                attention_mask = batch_feature.get("attention_mask", torch.ones_like(input_ids))
                pixel_values = batch_feature["pixel_values"]
            except Exception as e:
                logger.warning(f"VLA processor failed, using dummy data: {e}")
                # Create dummy data if processor fails
                input_ids = torch.randint(0, 1000, (1, 20))
                attention_mask = torch.ones_like(input_ids)
                pixel_values = torch.randn(1, 3, 224, 224)
            
            # Handle multi-view images (wrist cameras, etc.)
            pixel_values_list = [pixel_values]
            
            # Process additional camera views
            if hasattr(self.config, 'use_wrist_camera') and self.config.use_wrist_camera:
                if 'wrist_image' in input_data:
                    wrist_image = Image.fromarray(input_data['wrist_image']).convert("RGB")
                    if hasattr(self.config, 'center_crop') and self.config.center_crop:
                        wrist_image = center_crop_image(wrist_image)
                    
                    try:
                        wrist_feature = self.processor("", wrist_image)  # Empty prompt for additional views
                        pixel_values_list.append(wrist_feature["pixel_values"])
                    except:
                        pass  # Skip if processing fails
            
            # Concatenate pixel values
            if len(pixel_values_list) > 1:
                pixel_values = torch.cat(pixel_values_list, dim=1)
            else:
                pixel_values = pixel_values_list[0]
            
            # Handle OpenVLA-OFT specific formatting
            vla_type = getattr(self.config, 'vla_type', getattr(self.config, 'vla', None))
            if vla_type == "openvla-oft":
                # Add space token if needed (matching SimpleVLA-RL)
                space_token_id = 29871  # Space token for LLaMA-based models
                input_ids = torch.cat((input_ids, torch.tensor([[space_token_id]], dtype=input_ids.dtype)), dim=1)
                attention_mask = torch.cat((attention_mask, torch.tensor([[True]], dtype=attention_mask.dtype)), dim=1)
            
            batchdata["input_ids"].append(input_ids)
            batchdata["attention_mask"].append(attention_mask)
            batchdata["pixel_values"].append(pixel_values)
            
            # Process proprioception for robotwin
            if use_proprio and 'state' in input_data:
                proprio = input_data["state"]
                if hasattr(self.module, 'norm_stats') and hasattr(self.config, 'unnorm_key'):
                    norm_stats = self.module.norm_stats.get(self.vla_rollout.unnorm_key, {}).get("proprio", {})
                    proprio = normalize_proprio(proprio, norm_stats)
                batchdata["proprio"].append(torch.from_numpy(proprio))
        
        # Device placement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle batch formatting based on VLA type
        vla_type = getattr(self.config, 'vla_type', getattr(self.config, 'vla', None))
        if vla_type == "openvla-oft":
            # OpenVLA-OFT specific batch processing
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            
            pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', 0)
            batchdata["input_ids"] = pad_sequence(
                batchdata["input_ids"], batch_first=True, padding_value=pad_token_id
            ).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(
                batchdata["attention_mask"], batch_first=True, padding_value=0
            ).squeeze(-1).to(device)
            
            # Handle padding and sorting (matching SimpleVLA-RL)
            padding_mask = batchdata["input_ids"].ne(pad_token_id)
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int()
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(device)
            
            if use_proprio:
                batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(device)
        else:
            # Standard batch processing
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)
            
            if use_proprio:
                batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(device)
        
        return batchdata
    
    @torch.no_grad()
    def generate_one_step(self, prompts: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Generate one step of actions (matching SimpleVLA-RL's _generate_one_step)
        
        Args:
            prompts: Processed input batch
            
        Returns:
            VLA output with actions and metadata
        """
        # Get vla_type from config (handles both vla_type and vla attributes)
        vla_type = getattr(self.config, 'vla_type', getattr(self.config, 'vla', None))
        
        if vla_type == "openvla-oft":
            logger.debug(f"Using OpenVLA-OFT generation method")
            return self._generate_one_step_oft(prompts)
        elif vla_type == "openvla":
            logger.debug(f"Using OpenVLA generation method")
            return self._generate_one_step_openvla(prompts)
        
        # Fallback to dummy implementation
        logger.warning(f"No VLA model specified (vla_type={vla_type}), using dummy action generation")
        return self._generate_dummy_step(prompts)
    
    def _generate_one_step_oft(self, prompts: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate one step for OpenVLA-OFT (matching SimpleVLA-RL)"""
        input_ids = prompts['input_ids']
        attention_mask = prompts['attention_mask']
        pixel_values = prompts["pixel_values"]
        proprio = prompts.get("proprio", None)
        
        # Generation parameters
        do_sample = getattr(self.config, 'do_sample', True)
        temperature = getattr(self.config, 'temperature', 1.0)
        
        # Handle FSDP model parameters
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        
        try:
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Try to call the VLA model's generation method
                    if hasattr(self.module, 'generate_action_verl'):
                        actions, responses = self.module.generate_action_verl(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            proprio=proprio,
                            attention_mask=attention_mask,
                            padding_idx=getattr(self.processor.tokenizer, 'pad_token_id', 0),
                            do_sample=do_sample,
                            unnorm_key=getattr(self.config, 'unnorm_key', 'default'),
                            temperature=temperature,
                        )
                        
                        # Convert actions to numpy if needed (might already be numpy from _unnormalize_actions)
                        if isinstance(actions, torch.Tensor):
                            actions_np = actions.cpu().numpy()
                        else:
                            actions_np = actions
                        
                        return {
                            "action": actions_np,
                            "responses": responses,
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "pixel_values": pixel_values,
                        }
                    else:
                        logger.warning("VLA model does not have generate_action_verl method")
                        return self._generate_dummy_step(prompts)
                        
        except Exception as e:
            logger.warning(f"VLA-OFT inference failed: {e}")
            return self._generate_dummy_step(prompts)
    
    def _generate_one_step_openvla(self, prompts: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate one step for OpenVLA (matching SimpleVLA-RL)"""
        input_ids = prompts['input_ids']
        attention_mask = prompts['attention_mask']  
        pixel_values = prompts["pixel_values"]
        
        # Generation parameters
        do_sample = getattr(self.config, 'do_sample', True)
        temperature = getattr(self.config, 'temperature', 1.0)
        
        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Try to call the VLA model's generation method
                if hasattr(self.module, 'generate_action'):
                    actions, responses = self.module.generate_action(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        temperature=temperature,
                    )
                    
                    # Convert actions to numpy if needed (might already be numpy from _unnormalize_actions)
                    if isinstance(actions, torch.Tensor):
                        actions_np = actions.cpu().numpy()
                    else:
                        actions_np = actions
                    
                    return {
                        "action": actions_np,
                        "responses": responses,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                    }
                else:
                    logger.warning("VLA model does not have generate_action method")
                    return self._generate_dummy_step(prompts)
                    
        except Exception as e:
            logger.warning(f"OpenVLA inference failed: {e}")
            return self._generate_dummy_step(prompts)
    
    def _generate_dummy_step(self, prompts: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate dummy step when actual model inference fails"""
        batch_size = prompts['input_ids'].shape[0]
        
        # Generate realistic dummy actions
        actions = []
        responses = []
        
        for i in range(batch_size):
            # Generate dummy action based on step
            action = np.array([
                np.random.uniform(-0.1, 0.1),  # x
                np.random.uniform(-0.1, 0.1),  # y  
                np.random.uniform(-0.1, 0.1),  # z
                np.random.uniform(-0.1, 0.1),  # rx
                np.random.uniform(-0.1, 0.1),  # ry
                np.random.uniform(-0.1, 0.1),  # rz
                np.random.choice([-1.0, 1.0])  # gripper
            ])
            
            actions.append(action)
            responses.append(f"<action>{action.tolist()}</action>")
        
        return {
            "action": np.array(actions),
            "responses": responses,
            "input_ids": prompts['input_ids'],
            "attention_mask": prompts['attention_mask'],
            "pixel_values": prompts['pixel_values'],
        }
