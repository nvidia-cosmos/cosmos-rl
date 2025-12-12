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
Filesystem-based trajectory buffer for VLA rollout data.

This module provides utilities to store large VLA trajectory data (especially pixel_values)
to the filesystem instead of sending them over HTTP, avoiding multi-GB payload issues.

Usage:
    # Rollout worker: Save trajectory data
    trajectory_id = save_trajectory_to_buffer(trajectory_data, buffer_dir)
    # Send only the small trajectory_id via HTTP
    
    # Policy worker: Load trajectory data
    trajectory_data = load_trajectory_from_buffer(trajectory_id, buffer_dir)
"""

import os
import pickle
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from cosmos_rl.utils.logging import logger


class ReplayBuffer:
    """
    Filesystem-based buffer for storing large VLA trajectory data.
    
    Avoids sending multi-GB pixel_values over HTTP by storing them on disk
    and only passing a small trajectory ID via HTTP/Redis.
    """
    
    def __init__(self, buffer_dir: str):
        """
        Initialize replay buffer.
        
        Args:
            buffer_dir: Directory to store replay files (should be on a shared filesystem
                       accessible by both rollout and policy workers)
        """
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[ReplayBuffer] Initialized at {self.buffer_dir}")
    
    def save_trajectory(self, trajectory_data: Dict[str, Any], 
                       trajectory_id: Optional[str] = None) -> str:
        """
        Save trajectory data to filesystem.
        
        Args:
            replay_data: Dictionary containing replay data (input_ids, pixel_values, etc.)
            trajectory_id: Optional custom ID; if None, generates a unique UUID
            
        Returns:
            trajectory_id: Unique identifier for this trajectory
        """
        if trajectory_id is None:
            trajectory_id = f"traj_{uuid.uuid4().hex}_{int(time.time())}"
        
        filepath = self.buffer_dir / f"{trajectory_id}.pkl"
        
        try:
            # Use pickle protocol 4 for better performance with large data
            with open(filepath, 'wb') as f:
                pickle.dump(trajectory_data, f, protocol=4)
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            num_chunks = len(trajectory_data.get('input_ids', []))
            
            logger.debug(f"[ReplayBuffer] Saved trajectory {trajectory_id}: {num_chunks} chunks, {file_size_mb:.1f} MB")
            
            return trajectory_id
        except Exception as e:
            logger.error(f"[ReplayBuffer] Failed to save trajectory {trajectory_id}: {e}")
            raise
    
    def load_trajectory(self, trajectory_id: str, remove_after_load: bool = False) -> Dict[str, Any]:
        """
        Load trajectory data from filesystem.
        
        Args:
            trajectory_id: Unique identifier of the trajectory to load
            remove_after_load: If True, delete the file after loading (cleanup)
            
        Returns:
            trajectory_data: Dictionary containing trajectory data
        """
        filepath = self.buffer_dir / f"{trajectory_id}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"[ReplayBuffer] Trajectory file not found: {filepath}. "
                f"Make sure rollout and policy workers share the same filesystem."
            )
        
        try:
            with open(filepath, 'rb') as f:
                trajectory_data = pickle.load(f)
            
            # Debug logging for loaded trajectory structure
            # Handle both stacked tensors and lists
            input_ids_data = trajectory_data.get('input_ids')
            if isinstance(input_ids_data, torch.Tensor):
                num_chunks = input_ids_data.shape[0] if input_ids_data.dim() > 0 else 0
            elif isinstance(input_ids_data, list):
                num_chunks = len(input_ids_data)
            else:
                num_chunks = 0
            
            #logger.info(f"[ReplayBuffer] Loaded trajectory {trajectory_id}: {num_chunks} chunks")
            
            if remove_after_load:
                filepath.unlink()
                logger.debug(f"[ReplayBuffer] Removed trajectory file {trajectory_id}")
            
            return trajectory_data
        except Exception as e:
            logger.error(f"[ReplayBuffer] Failed to load trajectory {trajectory_id}: {e}")
            raise
    
    def cleanup_old_trajectories(self, max_age_seconds: int = 3600):
        """
        Remove trajectory files older than max_age_seconds.
        
        Args:
            max_age_seconds: Maximum age of files to keep (default: 1 hour)
        """
        current_time = time.time()
        removed_count = 0
        
        for filepath in self.buffer_dir.glob("traj_*.pkl"):
            try:
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    filepath.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"[ReplayBuffer] Failed to remove old file {filepath}: {e}")
        
        if removed_count > 0:
            logger.info(f"[ReplayBuffer] Cleaned up {removed_count} old trajectory files")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the replay buffer.
        
        Returns:
            Dictionary with buffer statistics (file count, total size, etc.)
        """
        files = list(self.buffer_dir.glob("traj_*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            'num_files': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'buffer_dir': str(self.buffer_dir),
        }


# Global instance for convenience
_global_buffer: Optional[ReplayBuffer] = None


def get_replay_buffer(buffer_dir: Optional[str] = None) -> ReplayBuffer:
    """
    Get or create a global trajectory buffer instance.
    
    Args:
        buffer_dir: Optional custom buffer directory; if None, uses default
        
    Returns:
        ReplayBuffer instance
    """
    global _global_buffer
    
    if _global_buffer is None or (buffer_dir is not None):
        buffer_dir = buffer_dir or os.environ.get('COSMOS_REPLAY_BUFFER_DIR', './replay_buffer')
        _global_buffer = ReplayBuffer(buffer_dir)
    
    return _global_buffer


# Convenience functions
def save_trajectory_to_buffer(trajectory_data: Dict[str, Any], 
                              trajectory_id: Optional[str] = None,
                              buffer_dir: Optional[str] = None) -> str:
    """Save trajectory data to buffer (convenience function)."""
    buffer = get_replay_buffer(buffer_dir)
    return buffer.save_trajectory(trajectory_data, trajectory_id)


def load_trajectory_from_buffer(trajectory_id: str, 
                                buffer_dir: Optional[str] = None,
                                remove_after_load: bool = False) -> Dict[str, Any]:
    """Load trajectory data from buffer (convenience function)."""
    buffer = get_replay_buffer(buffer_dir)
    return buffer.load_trajectory(trajectory_id, remove_after_load)

