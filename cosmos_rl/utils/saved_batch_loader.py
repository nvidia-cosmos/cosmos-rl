"""
Loader for pre-saved training batches from SimpleVLA-RL.
Allows Cosmos-RL to train on identical data for fair comparison.
"""
import os
import pickle
import glob
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from cosmos_rl.utils.logging import logger

class RLPolicyInput:
    """
    Container for policy input data matching Cosmos-RL's RLPolicyInput format.
    Must be at module level to be picklable for distributed training.
    """
    def __init__(self, per_step_data, pixel_values, num_steps, task_id, trial_id):
        self.per_step_data = per_step_data
        self.pixel_values = pixel_values
        self.num_steps = num_steps
        self.task_id = task_id
        self.trial_id = trial_id

class SavedBatchLoader:
    """
    Load pre-saved training batches from SimpleVLA-RL.
    
    Key conversions:
    - 1 SimpleVLA batch (512 episodes) = 4 Cosmos-RL steps (128 episodes each)
    - Unpads input_ids and attention_mask from 512 to actual sequence length
    - Maintains microbatch distribution from SimpleVLA-RL
    """
    
    def __init__(
        self, 
        batch_dir: str,
        episodes_per_step: int = 128,  # Cosmos-RL processes 128 episodes per step
        device: str = 'cuda'
    ):
        self.batch_dir = batch_dir
        self.episodes_per_step = episodes_per_step
        self.device = device
        
        # Find all batch files
        self.batch_files = sorted(glob.glob(os.path.join(batch_dir, 'batch_step_*.pkl')))
        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found in {batch_dir}")
        
        logger.info(f"[SavedBatchLoader] Found {len(self.batch_files)} batch files")
        logger.info(f"[SavedBatchLoader] First batch: {os.path.basename(self.batch_files[0])}")
        logger.info(f"[SavedBatchLoader] Last batch: {os.path.basename(self.batch_files[-1])}")
        
        self.current_file_idx = 0
        self.current_batch = None
        self.current_episode_idx = 0  # Track position within current batch
        
    def load_next_batch_file(self):
        """Load the next full batch file from disk"""
        if self.current_file_idx >= len(self.batch_files):
            raise StopIteration("No more batch files available")
        
        batch_path = self.batch_files[self.current_file_idx]
        logger.info(f"[SavedBatchLoader] Loading batch file: {batch_path}")
        
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f)
        
        # Log batch info
        batch_size = len(batch['tensors']['responses'])
        logger.info(f"[SavedBatchLoader]   Batch size: {batch_size} episodes")
        logger.info(f"[SavedBatchLoader]   Step: {batch['step']}")
        logger.info(f"[SavedBatchLoader]   Tensor keys: {list(batch['tensors'].keys())}")
        
        self.current_batch = batch
        self.current_episode_idx = 0
        self.current_file_idx += 1
        
        return batch
    
    def get_next_episodes(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get next N episodes from the current batch.
        
        Args:
            num_episodes: Number of episodes to retrieve. If None, uses self.episodes_per_step
            
        Returns:
            Dictionary with sliced tensors for the requested episodes
        """
        if num_episodes is None:
            num_episodes = self.episodes_per_step
        
        # Load new batch file if needed
        if self.current_batch is None or self.current_episode_idx >= len(self.current_batch['tensors']['responses']):
            self.load_next_batch_file()
        
        # Calculate slice indices
        start_idx = self.current_episode_idx
        end_idx = min(start_idx + num_episodes, len(self.current_batch['tensors']['responses']))
        actual_episodes = end_idx - start_idx
        
        logger.info(f"[SavedBatchLoader] Extracting episodes {start_idx}:{end_idx} ({actual_episodes} episodes)")
        
        # Extract episode slice
        episode_data = {
            'tensors': {},
            'meta_info': self.current_batch['meta_info'].copy(),
            'original_step': self.current_batch['step'],
            'episode_range': (start_idx, end_idx)
        }
        
        # Slice all tensors
        for key, tensor in self.current_batch['tensors'].items():
            episode_data['tensors'][key] = tensor[start_idx:end_idx]
        
        # Also copy non_tensors if present
        if 'non_tensors' in self.current_batch:
            episode_data['non_tensors'] = {}
            for key, val in self.current_batch['non_tensors'].items():
                episode_data['non_tensors'][key] = val[start_idx:end_idx]
        
        self.current_episode_idx = end_idx
        
        return episode_data
    
    def convert_to_cosmos_format(self, episode_data: Dict[str, Any]) -> List[Any]:
        """
        Convert SimpleVLA-RL batch format to Cosmos-RL policy_inputs format.
        
        Key operations:
        1. Unpad input_ids and attention_mask to actual sequence length
        2. Convert from (batch, traj_len, seq_len) to list of per-episode structures
        3. Match Cosmos-RL's RLPolicyInput format
        
        Returns:
            List of policy_input objects (one per episode)
        """
        tensors = episode_data['tensors']
        
        # Get dimensions
        batch_size = tensors['responses'].shape[0]
        traj_len = tensors['responses'].shape[1]  # e.g., 64
        per_step_action_len = tensors['responses'].shape[2] # 56
        action_dim = 7
        action_chunk_size = 8

        logger.info(f"[SavedBatchLoader] Converting {batch_size} episodes, {traj_len} trajectory steps")
        
        policy_inputs = []
        advantages_list = []
        
        for episode_idx in range(batch_size):
            # Get finish_step (number of valid trajectory steps)
            finish_step = tensors['finish_step'][episode_idx].item()
            old_log_probs = tensors['old_log_probs'][episode_idx].reshape(traj_len, -1)
            task_id = tensors['task_id'][episode_idx][0].item()
            trial_id = tensors['trial_id'][episode_idx][0].item()
            
            # Extract data for this episode (only up to finish_step)
            per_step_data = []
            
            for step_idx in range(traj_len):
                # Get input_ids and attention_mask for this step
                input_ids_padded = tensors['input_ids'][episode_idx, step_idx]  # (512,)
                attention_mask_padded = tensors['attention_mask'][episode_idx, step_idx]  # (512,)
                
                # Unpad to actual sequence length
                actual_len = attention_mask_padded.sum().item()
                input_ids = input_ids_padded[-actual_len:]
                attention_mask = attention_mask_padded[-actual_len:]

                # Get responses and old_log_prob for this step
                responses = tensors['responses'][episode_idx, step_idx]  # (action_len,)
                old_log_prob = old_log_probs[step_idx]  # (action_len,)

                # Create logprob_mask based on finish_step
                # Each step_idx has 8 mini-steps: [step_idx*8, step_idx*8+1, ..., step_idx*8+7]
                # Mask out mini-steps >= finish_step, repeated for each of 7 action dimensions
                mini_step_start = step_idx * action_chunk_size
                mini_step_mask = (mini_step_start + torch.arange(action_chunk_size)) < finish_step  # (8,)
                logprob_mask = mini_step_mask.unsqueeze(-1).repeat(1, action_dim).flatten()  # (56,) = repeat 7 times
                
                step_data = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'responses': responses,
                    'old_log_prob': old_log_prob,
                    'logprob_mask': logprob_mask,
                }
                
                per_step_data.append(step_data)
            
            # Extract pixel_values (all steps including padding for now)
            pixel_values_list = []
            for step_idx in range(traj_len):
                pixel_values_list.append(tensors['pixel_values'][episode_idx, step_idx])
            
            # Create RLPolicyInput structure
            policy_input = RLPolicyInput(per_step_data, pixel_values_list, finish_step, task_id, trial_id)
            policy_inputs.append(policy_input)
            
            # Extract advantage value (convert tensor to Python scalar if needed)
            advantage_value = tensors['advantages'][episode_idx][0]
            if isinstance(advantage_value, torch.Tensor):
                advantage_value = advantage_value.item()
            advantages_list.append(advantage_value)
        
        logger.info(f"[SavedBatchLoader] Converted {len(policy_inputs)} policy inputs")
        logger.info(f"[SavedBatchLoader] Advantages range: [{min(advantages_list):.4f}, {max(advantages_list):.4f}]")
        
        return policy_inputs, advantages_list
    
    def __len__(self):
        """Total number of Cosmos-RL steps available"""
        total_episodes = len(self.batch_files) * 512  # Assuming 512 episodes per batch
        return total_episodes // self.episodes_per_step
    
    def reset(self):
        """Reset to beginning"""
        self.current_file_idx = 0
        self.current_batch = None
        self.current_episode_idx = 0


class SavedBatchIterator:
    """
    Iterator interface for training loop integration.
    Yields (policy_inputs, advantages_list) tuples for each training step.
    """
    
    def __init__(self, batch_loader: SavedBatchLoader):
        self.loader = batch_loader
        self.step_count = 0
        
    def __iter__(self):
        self.loader.reset()
        self.step_count = 0
        return self
    
    def __next__(self):
        try:
            # Get next batch of episodes
            episode_data = self.loader.get_next_episodes()
            
            # Convert to Cosmos-RL format
            policy_inputs, advantages_list = self.loader.convert_to_cosmos_format(episode_data)
            
            self.step_count += 1
            logger.info(f"[SavedBatchIterator] Step {self.step_count}: {len(policy_inputs)} episodes")
            
            return policy_inputs, advantages_list, episode_data['meta_info']
            
        except StopIteration:
            raise StopIteration

