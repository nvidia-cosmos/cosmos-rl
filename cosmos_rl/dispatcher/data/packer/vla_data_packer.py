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

from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload
from typing import Any, Dict, List
import torch
from cosmos_rl.utils.logging import logger


class VLADataPacker(DataPacker):
    """
    Data packer for Vision-Language-Action (VLA) models.
    
    Converts plain dict with VLA task information into RLPayload objects
    with metadata field populated for the rollout worker.
    
    Expected input dict format:
    {
        "task_suite_name": str,  # e.g., "libero_10", "robotwin2"
        "task_id": int,          # Task ID within the suite
        "trial_id": int,         # Trial ID for this task
        "trial_seed": int,       # Random seed for environment
        ... (other optional fields)
    }
    """
    
    def get_rollout_input(self, item: Any) -> RLPayload:
        """
        Convert VLA task dict to RLPayload with metadata
        
        Args:
            item: Either a dict with VLA task info, or already an RLPayload
            
        Returns:
            RLPayload with task info in metadata field
        """
        # If already an RLPayload, return as-is
        if isinstance(item, RLPayload):
            return item
        
        # Extract VLA task information from dict
        if not isinstance(item, dict):
            raise ValueError(f"VLA data packer expects dict or RLPayload, got {type(item)}")
        
        # Build metadata dict from task info
        metadata = {}
        
        # Required fields
        metadata['task_suite_name'] = item.get('task_suite_name', 'libero_10')
        metadata['task_id'] = item.get('task_id', 0)
        metadata['trial_id'] = item.get('trial_id', 0)
        metadata['trial_seed'] = item.get('trial_seed', -1)
        
        # Convert tensors to Python values if needed
        for key in ['task_id', 'trial_id', 'trial_seed']:
            if hasattr(metadata[key], 'item'):
                metadata[key] = metadata[key].item()
        
        # Optional fields
        if 'instruction' in item:
            metadata['instruction'] = item['instruction']
        if 'max_steps' in item:
            metadata['max_steps'] = item['max_steps']
        
        # Copy any additional metadata
        for key, value in item.items():
            if key not in metadata and key not in ['task_suite_name', 'task_id', 'trial_id', 'trial_seed']:
                # Skip large data like images
                if not isinstance(value, (torch.Tensor, list, dict)):
                    metadata[key] = value
        
        # Create RLPayload with metadata
        # For VLA, we don't use text prompt, so set it to empty string
        # The actual task instruction is in metadata
        payload = RLPayload(
            prompt="",  # VLA uses environment interactions, not text prompts
            metadata=metadata,
            weight_version=0  # Will be set by dispatcher
        )
        
        logger.debug(f"VLA data packer created payload with metadata: {metadata}")
        return payload
    
    def get_policy_input(
        self,
        sample: Any,
        rollout_output: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """
        Process VLA sample and rollout output for policy training
        
        For VLA, the rollout output contains trajectory data (observations, actions, rewards)
        rather than text completions.
        """
        # For VLA, we typically don't use text-based policy input
        # The actual training data comes from the trajectory
        # This method might need customization based on your VLA training approach
        raise NotImplementedError(
            "VLA policy training uses trajectory data, not text-based get_policy_input. "
            "Override this method if you need custom VLA policy input processing."
        )
    
    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute max sequence length for VLA policy training
        
        For VLA, this might be the max trajectory length
        """
        # VLA doesn't use variable-length text sequences in the same way
        # Return a fixed value or compute based on trajectory lengths
        if hasattr(self.config, 'vla') and hasattr(self.config.vla, 'max_episode_length'):
            return self.config.vla.max_episode_length
        return 512  # Default
    
    def policy_collate_fn(
        self,
        processed_samples: List[Any],
        computed_max_len: int,
    ) -> Dict[str, Any]:
        """
        Collate VLA samples for policy training
        
        For VLA, this should batch trajectory data (observations, actions, etc.)
        """
        # VLA uses different data format than text LLMs
        # This should be customized based on your VLA policy training needs
        raise NotImplementedError(
            "VLA policy training uses trajectory-based collation. "
            "Override this method for your specific VLA training approach."
        )

