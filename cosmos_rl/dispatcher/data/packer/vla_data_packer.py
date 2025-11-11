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
        Process VLA sample and rollout output for policy training.
        
        For VLA training (similar to SimpleVLA-RL approach):
        - Vision tokens and prompt tokens are treated as context (mask = 0, no gradient)
        - Action tokens are trained (mask = 1, compute loss)
        - This allows training only the LLM's action prediction while vision features flow through
        
        Args:
            sample: The prompt (empty string for VLA) or metadata
            rollout_output: Text summary like "Task completed in 10 steps"
            n_ignore_prefix_tokens: Number of prefix tokens to ignore
            
        Returns:
            RLPolicyInput object with input_ids and logprob_masks
        """
        # TODO: Get actual trajectory data (action tokens, observations) from rollout
        # Currently, VLA rollout only stores metadata, not the full trajectory
        # This needs to be implemented in vla_rollout.py to store and transfer trajectory data
        
        # For now, create a plausible structure that allows GRPO training to run
        # This demonstrates the correct format even with dummy data
        
        # VLA sequence structure (like SimpleVLA-RL):
        # [vision_placeholder_tokens] + [text_prompt_tokens] + [action_tokens] + [stop_token]
        #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^
        #              Context (mask=0)                        Trained (mask=1)
        
        # Typical VLA dimensions (OpenVLA):
        # - Vision: ~256 patches × 2 (DinoSiglip) = ~512 tokens
        # - Prompt: ~50-100 tokens for task description  
        # - Actions: 7 dims × 8 chunks = 56 action tokens per episode
        
        # Create dummy sequence (TODO: replace with real trajectory data)
        num_vision_tokens = 512  # Vision encoder output tokens
        num_prompt_tokens = 50   # Task description tokens
        num_action_tokens = 56   # Action tokens to train on (7 dims × 8 chunks)
        
        total_length = num_vision_tokens + num_prompt_tokens + num_action_tokens + 1  # +1 for stop token
        
        # Create input_ids (dummy for now - TODO: get from trajectory)
        input_ids = [1] * total_length  # All set to 1 as placeholder
        
        # Create logprob_masks: 1 for action tokens only, 0 for everything else
        logprob_masks = [0] * (num_vision_tokens + num_prompt_tokens)  # Context tokens
        logprob_masks += [1] * num_action_tokens  # Action tokens (TRAIN ON THESE)
        logprob_masks += [0]  # Stop token (don't train on this)
        
        assert len(input_ids) == len(logprob_masks), f"Length mismatch: {len(input_ids)} vs {len(logprob_masks)}"
        
        logger.debug(
            f"[VLA Policy Input] Created sequence: "
            f"total={total_length}, vision={num_vision_tokens}, "
            f"prompt={num_prompt_tokens}, actions={num_action_tokens}, "
            f"trainable_tokens={sum(logprob_masks)}"
        )
        
        class RLPolicyInput:
            """Mimics the structure expected by GRPO trainer"""
            def __init__(self, input_ids, logprob_masks):
                self.input_ids = input_ids
                self.logprob_masks = logprob_masks
        
        return RLPolicyInput(input_ids, logprob_masks)
    
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
        Collate VLA samples for policy training.
        
        Similar to DecoderOnlyLLMDataPacker, but adapted for VLA sequences:
        - Pads sequences to computed_max_len
        - Returns input_ids and logprob_masks tensors
        
        Args:
            processed_samples: List of RLPolicyInput objects from get_policy_input
            computed_max_len: Maximum sequence length for padding
            
        Returns:
            Dict with 'input_ids' and 'logprob_masks' tensors
        """
        input_ids = [x.input_ids for x in processed_samples]
        logprob_masks = [x.logprob_masks for x in processed_samples]
        
        assert len(input_ids) == len(logprob_masks), \
            f"Mismatch: {len(input_ids)} input_ids vs {len(logprob_masks)} logprob_masks"
        
        device = torch.cuda.current_device()
        pad_token_id = 0  # VLA models typically use 0 for padding
        
        # Pad sequences to computed_max_len
        collated_dict = {}
        collated_dict["input_ids"] = torch.tensor(
            [
                x[:computed_max_len] + [pad_token_id] * max(0, computed_max_len - len(x))
                for x in input_ids
            ],
            dtype=torch.long,
        ).to(device)
        
        collated_dict["logprob_masks"] = torch.tensor(
            [
                x[:computed_max_len] + [0] * max(0, computed_max_len - len(x))
                for x in logprob_masks
            ],
            dtype=torch.long,
        ).to(device)
        
        logger.debug(
            f"[VLA Collate] Batched {len(processed_samples)} samples, "
            f"max_len={computed_max_len}, "
            f"input_ids shape={collated_dict['input_ids'].shape}, "
            f"trainable_ratio={collated_dict['logprob_masks'].float().mean():.2%}"
        )
        
        return collated_dict

