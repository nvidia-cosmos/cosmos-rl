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
        
        episode_length = sample.metadata.get('episode_length', 0)
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
                    logger.debug(f"[VLA Policy Input] Loaded trajectory {trajectory_id} from filesystem buffer")
                except Exception as e:
                    logger.error(f"[VLA Policy Input] Failed to load trajectory {trajectory_id}: {e}")
                    trajectory = None
            else:
                # Fallback: old approach where trajectory was embedded in metadata
                trajectory = sample.metadata.get('trajectory')
        
        input_ids_data = trajectory['input_ids']
        responses_data = trajectory['responses']
        pixel_values_data = trajectory.get('pixel_values', [])
        old_log_prob_data = trajectory.get('old_log_prob', [])
        task_id = sample.metadata.get('task_id', 0)
        trial_id = sample.metadata.get('trial_id', 0)
        gen_idx = sample.metadata.get('gen_idx', 0)
        
        # Handle both stacked tensors and lists
        if isinstance(input_ids_data, torch.Tensor) and input_ids_data.dim() >= 2:
            # NEW FORMAT: Stacked tensors
            num_steps = input_ids_data.shape[0]
            input_ids_list = [input_ids_data[i] for i in range(num_steps)]
            responses_list = [responses_data[i] for i in range(num_steps)]
            pixel_values_list = [pixel_values_data[i] for i in range(num_steps)] if isinstance(pixel_values_data, torch.Tensor) else pixel_values_data
            old_log_prob_list = [old_log_prob_data[i] for i in range(num_steps)] if isinstance(old_log_prob_data, torch.Tensor) and old_log_prob_data.numel() > 0 else []
        else:
            # OLD FORMAT: Lists
            input_ids_list = input_ids_data if isinstance(input_ids_data, list) else [input_ids_data]
            responses_list = responses_data if isinstance(responses_data, list) else [responses_data]
            pixel_values_list = pixel_values_data if isinstance(pixel_values_data, list) else [pixel_values_data]
            old_log_prob_list = old_log_prob_data if isinstance(old_log_prob_data, list) else [old_log_prob_data]
        
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
            
            # Extract old_log_prob for this step (if available)
            old_log_prob = None
            if old_log_prob_list and step_idx < len(old_log_prob_list):
                if isinstance(old_log_prob_list[step_idx], torch.Tensor):
                    old_log_prob = old_log_prob_list[step_idx].flatten()
                elif isinstance(old_log_prob_list[step_idx], list):
                    old_log_prob = torch.tensor(old_log_prob_list[step_idx], dtype=torch.float32).flatten()
            
            # If no old_log_prob available, create dummy zeros
            if old_log_prob is None:
                old_log_prob = torch.zeros(len(response_tokens), dtype=torch.float32)
            
            # CRITICAL: Only pass prompt to model (NOT concatenated with actions)
            # The openvla-oft model internally:
            #   1. Adds 56 placeholder action tokens
            #   2. Adds 1 stop token
            #   3. Returns logits ONLY for the 56 action positions
            # So input is just the prompt, model handles the rest
            
            # Create attention mask for prompt only
            attention_mask = torch.ones_like(prompt_tokens)
            
            # Create logprob mask for action tokens only (56 values)
            # This will be aligned with the model's output logits (56 tokens)
            ACTION_CHUNK_SIZE = 8
            ACTION_DIM_SIZE = 7
            mini_step_start = step_idx * ACTION_CHUNK_SIZE
            mini_step_mask = (mini_step_start + torch.arange(ACTION_CHUNK_SIZE)) < episode_length  # (8,)
            logprob_mask = mini_step_mask.unsqueeze(-1).repeat(1, ACTION_DIM_SIZE).flatten()  # (56,) = repeat 7 times
            
            per_step_data.append({
                'input_ids': prompt_tokens,  # Prompt only (31 tokens)
                'attention_mask': attention_mask,  # Mask for prompt (31 values)
                'logprob_mask': logprob_mask,
                'responses': response_tokens,  # Action labels for loss computation (56 tokens)
                'old_log_prob': old_log_prob,  # Old log probs from rollout (56 values)
            })
            
        # Return full episode without chunking
        # Chunking will be handled in train_vla() for gradient accumulation
        class RLPolicyInput:
            """Per-step structured input for VLA training"""
            def __init__(self, per_step_data, pixel_values, task_id, trial_id, gen_idx):
                self.per_step_data = per_step_data  # List of dicts, one per step
                self.pixel_values = pixel_values  # List of tensors, one per step
                self.num_steps = len(per_step_data)
                self.task_id = task_id
                self.trial_id = trial_id
                self.gen_idx = gen_idx
        
        return RLPolicyInput(per_step_data, pixel_values_list, task_id, trial_id, gen_idx)

    
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
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Collate VLA samples for policy training with per-step structure.
        
        This creates (batch, num_steps, seq_len) tensors to match SimpleVLA-RL format.
        
        Args:
            processed_samples: List of RLPolicyInput objects with per_step_data
            max_steps: Maximum sequence length per step for padding
            
        Returns:
            Dict with tensors shaped (batch, num_steps, ...) for per-step processing
        """
        batch_size = len(processed_samples)
        
        # VLA: With mini_batch=1, process actual episode length (no cross-episode padding)
        # For mini_batch>1, pad to max_steps (but sorting minimizes this)
        max_prompt_len = max(len(s.per_step_data[0]['input_ids']) for s in processed_samples)
        
        device = torch.cuda.current_device()
        # Get pad_token_id from tokenizer (must match what VLA model uses for unpadding)
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        
        # Initialize batch lists
        batch_input_ids = []
        batch_attention_masks = []
        batch_logprob_masks = []
        batch_responses = []
        batch_pixel_values = []
        batch_old_log_probs = []
        actual_num_steps = []
        
        for sample in processed_samples:
            # Stack this sample's steps: (num_steps, seq_len)
            step_input_ids = []
            step_attention_masks = []
            step_logprob_masks = []
            step_responses = []
            step_old_log_probs = []
            
            for step_data in sample.per_step_data:
                # Pad/truncate each step's sequence to computed_max_len
                ids = step_data['input_ids'][:max_prompt_len].cpu()  # Ensure CPU (prompt only)
                mask = step_data['attention_mask'][:max_prompt_len].cpu()
                logprob_mask = step_data['logprob_mask'].cpu()  # No truncation - these match action tokens
                responses = step_data['responses'].cpu()  # Action tokens for loss computation
                old_log_prob = step_data['old_log_prob'].cpu()  # Old log probs from rollout
                
                # Pad prompt if shorter than computed_max_len (keep on CPU)
                if len(ids) < max_prompt_len:
                    pad_len = max_prompt_len - len(ids)
                    ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long, device='cpu')])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long, device='cpu')])

                step_input_ids.append(ids)
                step_attention_masks.append(mask)
                step_logprob_masks.append(logprob_mask)
                step_responses.append(responses)
                step_old_log_probs.append(old_log_prob)
            
            # Pad to max_steps if this sample has fewer steps (keep on CPU)
            actual_num_steps.append(len(sample.per_step_data))
            action_chunk_size = len(step_responses[0])
            
            # For VLA: Pad responses with ACTION_TOKEN_BEGIN_IDX (31744) instead of 0
            # After remapping (response - 31744), padding becomes 0, which is in valid range [0, 256)
            ACTION_TOKEN_BEGIN_IDX = 31744  # From openvla-oft constants
            
            while len(step_input_ids) < max_steps:
                step_input_ids.append(torch.full((max_prompt_len,), pad_token_id, dtype=torch.long, device='cpu'))
                step_attention_masks.append(torch.zeros(max_prompt_len, dtype=torch.long, device='cpu'))
                step_logprob_masks.append(torch.zeros(action_chunk_size, dtype=torch.long, device='cpu'))
                # Pad with ACTION_TOKEN_BEGIN_IDX so that after remapping they become 0 (valid index)
                step_responses.append(torch.full((action_chunk_size,), ACTION_TOKEN_BEGIN_IDX, dtype=torch.long, device='cpu'))
                # Pad old_log_probs with zeros (will be masked out anyway)
                step_old_log_probs.append(torch.zeros(action_chunk_size, dtype=torch.float32, device='cpu'))
            
            # Stack: (num_steps, seq_len) for prompts, (num_steps, action_seq_len) for actions
            batch_input_ids.append(torch.stack(step_input_ids))
            batch_attention_masks.append(torch.stack(step_attention_masks))
            batch_logprob_masks.append(torch.stack(step_logprob_masks))
            batch_responses.append(torch.stack(step_responses))
            batch_old_log_probs.append(torch.stack(step_old_log_probs))
            
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
        
        
        # Final batch shapes:
        # - Prompts: (batch, num_steps, prompt_seq_len)
        # - Actions: (batch, num_steps, action_seq_len)  # action_seq_len = 56
        # - Pixel values: (batch, num_steps, C, H, W)
        # - Old log probs: (batch, num_steps, action_seq_len)  # action_seq_len = 56
        collated_dict = {
            'input_ids': torch.stack(batch_input_ids).to(device),  # (batch, steps, prompt_len) - prompt only
            'attention_mask': torch.stack(batch_attention_masks).to(device),
            'logprob_masks': torch.stack(batch_logprob_masks).to(device),  # (batch, steps, action_len)
            'responses': torch.stack(batch_responses).to(device),  # (batch, steps, action_len) - action tokens
            'pixel_values': torch.stack(batch_pixel_values).to(device),  # (batch, steps, C, H, W)
            'old_log_prob': torch.stack(batch_old_log_probs).to(device),  # (batch, steps, action_len)
            'num_steps': torch.tensor(actual_num_steps, dtype=torch.long).to(device),
        }
        return collated_dict

