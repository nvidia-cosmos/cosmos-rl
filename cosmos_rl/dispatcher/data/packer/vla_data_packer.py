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
from cosmos_rl.utils.trajectory_buffer import load_trajectory_from_buffer


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
            sample: For VLA, this is the full Rollout object with metadata containing trajectory
            rollout_output: Text summary like "Task completed in 10 steps"
            n_ignore_prefix_tokens: Number of prefix tokens to ignore
            
        Returns:
            RLPolicyInput object with input_ids and logprob_masks
        """
        # Extract trajectory data from Rollout metadata (if available)
        from cosmos_rl.dispatcher.data.schema import Rollout
        
        trajectory = None
        if isinstance(sample, Rollout) and sample.metadata:
            # Check if we have a trajectory_id (new filesystem buffer approach)
            trajectory_id = sample.metadata.get('trajectory_id')
            if trajectory_id:
                # Load full trajectory data (including pixel_values) from filesystem
                try:
                    trajectory = load_trajectory_from_buffer(
                        trajectory_id, 
                        remove_after_load=False  # Don't remove yet, training uses it multiple times
                    )
                    logger.info(f"[VLA Policy Input] Loaded trajectory {trajectory_id} from filesystem buffer")
                except Exception as e:
                    logger.error(f"[VLA Policy Input] Failed to load trajectory {trajectory_id}: {e}")
                    trajectory = None
            else:
                # Fallback: old approach where trajectory was embedded in metadata
                trajectory = sample.metadata.get('trajectory')
        
        if trajectory and trajectory.get('input_ids') and trajectory.get('responses'):
            # *** REAL TRAJECTORY DATA AVAILABLE ***
            logger.info(f"[VLA Policy Input] Using REAL trajectory data from rollout")
            
            # VLA structure: Keep per-step organization for proper batching
            # Each step has: input_ids (prompt), responses (action tokens), pixel_values (image)
            input_ids_list = trajectory['input_ids']
            responses_list = trajectory['responses']
            pixel_values_list = trajectory.get('pixel_values', [])
            
            # Build per-step data structure (matches SimpleVLA-RL)
            per_step_data = []
            
            for step_idx, (step_ids, step_responses) in enumerate(zip(input_ids_list, responses_list)):
                # Convert prompt tokens to tensor
                if isinstance(step_ids, torch.Tensor):
                    prompt_tokens = step_ids
                elif isinstance(step_ids, list):
                    prompt_tokens = torch.tensor(step_ids, dtype=torch.long)
                else:
                    continue
                
                # Convert response tokens to tensor
                if isinstance(step_responses, torch.Tensor):
                    # Flatten if multi-dimensional: (8 actions, 7 tokens) → [56 tokens]
                    response_tokens = step_responses.flatten()
                    if step_idx == 0:  # Log first step for debugging
                        logger.info(f"[VLA Policy Input] Step 0: responses shape {step_responses.shape} → flattened to {response_tokens.shape[0]} tokens")
                elif isinstance(step_responses, list):
                    # Flatten nested lists
                    flat_responses = []
                    if step_responses and isinstance(step_responses[0], (list, torch.Tensor)):
                        for r in step_responses:
                            if isinstance(r, torch.Tensor):
                                flat_responses.extend(r.flatten().tolist())
                            elif isinstance(r, list):
                                flat_responses.extend(r)
                    else:
                        flat_responses = step_responses
                    response_tokens = torch.tensor(flat_responses, dtype=torch.long)
                else:
                    response_tokens = torch.tensor([], dtype=torch.long)
                
                # CRITICAL: Concatenate prompt + responses for THIS STEP ONLY
                # This creates the full sequence for this timestep
                full_sequence = torch.cat([prompt_tokens, response_tokens])
                
                # Create attention mask (all 1s for actual tokens)
                attention_mask = torch.ones_like(full_sequence)
                
                # Create logprob mask: 0 for prompt (don't train), 1 for actions (train)
                logprob_mask = torch.cat([
                    torch.zeros(len(prompt_tokens), dtype=torch.long),
                    torch.ones(len(response_tokens), dtype=torch.long)
                ])
                
                per_step_data.append({
                    'input_ids': full_sequence,
                    'attention_mask': attention_mask,
                    'logprob_mask': logprob_mask,
                })
                
                if step_idx == 0:  # Log first step
                    logger.info(
                        f"[VLA Policy Input] Step 0: prompt={len(prompt_tokens)}, "
                        f"actions={len(response_tokens)}, total={len(full_sequence)}"
                    )
            
            num_steps = len(per_step_data)
            total_tokens = sum(len(s['input_ids']) for s in per_step_data)
            total_action_tokens = sum(s['logprob_mask'].sum().item() for s in per_step_data)
            
            logger.info(
                f"[VLA Policy Input] Trajectory: {num_steps} steps, "
                f"total_tokens={total_tokens}, action_tokens={total_action_tokens}, "
                f"trainable_ratio={total_action_tokens/total_tokens:.2%}"
            )
        
        class RLPolicyInput:
            """Per-step structured input for VLA training"""
            def __init__(self, per_step_data, pixel_values):
                self.per_step_data = per_step_data  # List of dicts, one per step
                self.pixel_values = pixel_values  # List of tensors, one per step
                self.num_steps = len(per_step_data)
        
        return RLPolicyInput(per_step_data, pixel_values_list)
    
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
        Collate VLA samples for policy training with per-step structure.
        
        This creates (batch, num_steps, seq_len) tensors to match SimpleVLA-RL format.
        
        Args:
            processed_samples: List of RLPolicyInput objects with per_step_data
            computed_max_len: Maximum sequence length per step for padding
            
        Returns:
            Dict with tensors shaped (batch, num_steps, ...) for per-step processing
        """
        batch_size = len(processed_samples)
        
        # VLA: With mini_batch=1, process actual episode length (no cross-episode padding)
        # For mini_batch>1, pad to max_steps (but sorting minimizes this)
        max_steps = max(s.num_steps for s in processed_samples)
        
        device = torch.cuda.current_device()
        pad_token_id = 0
        
        if batch_size == 1:
            logger.debug(
                f"[VLA Collate] Processing single episode: "
                f"num_steps={max_steps}, max_len_per_step={computed_max_len} "
                f"(no cross-episode padding)"
            )
        else:
            logger.debug(
                f"[VLA Collate] Batching {batch_size} samples, "
                f"max_steps={max_steps}, max_len_per_step={computed_max_len}"
            )
        
        # Initialize batch lists
        batch_input_ids = []
        batch_attention_masks = []
        batch_logprob_masks = []
        batch_pixel_values = []
        actual_num_steps = []
        
        for sample in processed_samples:
            # Stack this sample's steps: (num_steps, seq_len)
            step_input_ids = []
            step_attention_masks = []
            step_logprob_masks = []
            
            for step_data in sample.per_step_data:
                # Pad/truncate each step's sequence to computed_max_len
                ids = step_data['input_ids'][:computed_max_len].cpu()  # Ensure CPU
                mask = step_data['attention_mask'][:computed_max_len].cpu()
                logprob_mask = step_data['logprob_mask'][:computed_max_len].cpu()
                
                # Pad if shorter than computed_max_len (keep on CPU)
                if len(ids) < computed_max_len:
                    pad_len = computed_max_len - len(ids)
                    ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long, device='cpu')])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long, device='cpu')])
                    logprob_mask = torch.cat([logprob_mask, torch.zeros(pad_len, dtype=torch.long, device='cpu')])
                
                step_input_ids.append(ids)
                step_attention_masks.append(mask)
                step_logprob_masks.append(logprob_mask)
            
            # Pad to max_steps if this sample has fewer steps (keep on CPU)
            actual_num_steps.append(len(sample.per_step_data))
            while len(step_input_ids) < max_steps:
                step_input_ids.append(torch.full((computed_max_len,), pad_token_id, dtype=torch.long, device='cpu'))
                step_attention_masks.append(torch.zeros(computed_max_len, dtype=torch.long, device='cpu'))
                step_logprob_masks.append(torch.zeros(computed_max_len, dtype=torch.long, device='cpu'))
            
            # Stack: (num_steps, seq_len)
            batch_input_ids.append(torch.stack(step_input_ids))
            batch_attention_masks.append(torch.stack(step_attention_masks))
            batch_logprob_masks.append(torch.stack(step_logprob_masks))
            
            # Stack pixel_values: (num_steps, C, H, W) - keep on CPU for now
            step_pixel_values = []
            for pv in sample.pixel_values[:max_steps]:
                if isinstance(pv, torch.Tensor):
                    step_pixel_values.append(pv.cpu())  # Ensure CPU
                else:
                    # Fallback: create dummy pixel values on CPU
                    step_pixel_values.append(torch.zeros(6, 224, 224, device='cpu'))
            
            # Pad pixel_values to max_steps (keep on CPU)
            while len(step_pixel_values) < max_steps:
                step_pixel_values.append(torch.zeros(6, 224, 224, device='cpu'))
            
            batch_pixel_values.append(torch.stack(step_pixel_values))
        
        # Final batch shapes: (batch, num_steps, seq_len) and (batch, num_steps, C, H, W)
        collated_dict = {
            'input_ids': torch.stack(batch_input_ids).to(device),  # (batch, steps, seq_len)
            'attention_mask': torch.stack(batch_attention_masks).to(device),
            'logprob_masks': torch.stack(batch_logprob_masks).to(device),
            'pixel_values': torch.stack(batch_pixel_values).to(device),  # (batch, steps, C, H, W)
            'num_steps': torch.tensor(actual_num_steps, dtype=torch.long).to(device),
        }
        
        # Calculate trainable ratio
        total_tokens = collated_dict['logprob_masks'].numel()
        trainable_tokens = collated_dict['logprob_masks'].sum().item()
        
        logger.info(
            f"[VLA Collate] Batch created: "
            f"input_ids={collated_dict['input_ids'].shape}, "
            f"pixel_values={collated_dict['pixel_values'].shape}, "
            f"trainable_ratio={trainable_tokens/total_tokens:.2%}"
        )
        
        return collated_dict

