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
import re
from typing import List, Tuple, Dict, Any
from functools import cached_property
from transformers import AutoConfig

from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils import util
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism_registry import (
    ParallelismStrategyRole,
    register_parallelism_strategy,
    get_policy_parallelism_strategy as get_policy_strategy,
    get_rollout_parallelism_strategy as get_rollout_strategy,
)


class VLAWeightMapper(WeightMapper):
    """
    Weight mapper for VLA (Vision-Language-Action) models
    
    Handles weight mapping and tensor parallelism for VLA models including:
    - OpenVLA: Standard OpenVLA models  
    - OpenVLA-OFT: OpenVLA with Orthogonal Fine-Tuning
    
    The mapper handles the complex architecture of VLA models which typically include:
    - Vision backbone (e.g., DINOv2, SigLIP) 
    - Vision-language projector
    - Language model (e.g., Llama, Vicuna)
    - Action head for robotic control
    """
    
    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        
        # VLA models are multimodal
        self.is_vlm = True
        
        # Get language model configuration
        if hasattr(hf_config, 'text_config'):
            text_config = hf_config.text_config
        elif hasattr(hf_config, 'language_config'):  
            text_config = hf_config.language_config
        else:
            # Fallback to main config for some VLA models
            text_config = hf_config
            
        # Attention configuration for tensor parallelism
        self.num_attention_heads = getattr(text_config, 'num_attention_heads', 32)
        self.num_key_value_heads = getattr(text_config, 'num_key_value_heads', self.num_attention_heads)
        self.hidden_size = getattr(text_config, 'hidden_size', 4096)
        
        self.kv_head_ratio = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Vision configuration
        if hasattr(hf_config, 'vision_config'):
            self.vision_config = hf_config.vision_config
        else:
            self.vision_config = None
            
        # Action configuration
        if hasattr(hf_config, 'action_dim'):
            self.action_dim = hf_config.action_dim
        else:
            # Default for most robotic tasks
            self.action_dim = 7
            
        logger.info(f"VLA WeightMapper initialized:")
        logger.info(f"  - Attention heads: {self.num_attention_heads}")
        logger.info(f"  - KV heads: {self.num_key_value_heads}")  
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Action dim: {self.action_dim}")
        
    def _rollout_vllm_name_to_hf(self, rollout_weight_name: str) -> str:
        """
        Map rollout weight names to HuggingFace weight names for VLA models
        
        VLA models have a complex architecture with multiple components:
        - vision_backbone: Vision encoder weights
        - projector: Vision-to-language projection  
        - language_model: Language model weights
        - action_head: Action prediction head
        """
        
        # Handle vision backbone weights
        if rollout_weight_name.startswith("vision_backbone"):
            return rollout_weight_name
            
        # Handle projector weights  
        if rollout_weight_name.startswith("projector"):
            return rollout_weight_name
            
        # Handle action head weights
        if rollout_weight_name.startswith("action_head"):
            return rollout_weight_name
            
        # Handle language model weights - these follow standard transformer patterns
        if rollout_weight_name.startswith("language_model"):
            # Remove language_model prefix for some mappings
            base_name = rollout_weight_name.replace("language_model.", "")
            
            # Standard transformer weight mappings
            if "self_attn" in base_name:
                return rollout_weight_name
            elif "mlp" in base_name:
                return rollout_weight_name  
            elif "embed_tokens" in base_name:
                return rollout_weight_name
            elif "norm" in base_name:
                return rollout_weight_name
            else:
                return rollout_weight_name
                
        # For weights that don't match patterns above, return as-is
        return rollout_weight_name
        
    def policy_map_local_key_to_hf_key(self, policy_weight_name: str) -> str:
        """
        Map policy weight names to HuggingFace weight names
        
        This handles the mapping from cosmos-rl policy model names
        to the actual HuggingFace model weight names.
        """
        
        # VLA models typically have prefixed components
        if policy_weight_name.startswith("model."):
            # Remove model prefix
            base_name = policy_weight_name[6:]  # len("model.") = 6
            
            # Vision backbone mapping
            if base_name.startswith("vision_backbone"):
                return base_name
                
            # Projector mapping  
            elif base_name.startswith("projector"):
                return base_name
                
            # Language model mapping
            elif base_name.startswith("language_model"):
                return base_name
                
            # Action head mapping
            elif base_name.startswith("action_head"):
                return base_name
                
            else:
                return base_name
        else:
            return policy_weight_name
            
    def rollout_prepare_recv(
        self,
        vllm_model: Any,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[str, int]]]]:
        """
        Rollout prepare recv list for P2R weight sync:
            - vllm_weight_inplace_view_map: Dict[str, torch.Tensor]: the map of vllm weight inplace view to be written by P2R weight sync
            - recv_key_n_rank_list: List[List[Tuple[str, int]]]: the list of grouped recv key and its tensor rank
        
        For VLA models, we need to handle the multimodal components properly.
        """
        
        vllm_weight_inplace_view_map = {}
        recv_key_n_rank_list = []
        
        # Get model state dict from vLLM model
        if hasattr(vllm_model, 'state_dict'):
            model_state_dict = vllm_model.state_dict()
        elif hasattr(vllm_model, 'model') and hasattr(vllm_model.model, 'state_dict'):
            model_state_dict = vllm_model.model.state_dict()
        else:
            logger.warning("Could not find state_dict in vLLM model for VLA weight sync")
            return vllm_weight_inplace_view_map, recv_key_n_rank_list
            
        # Group parameters by component for efficient syncing
        language_model_params = []
        vision_backbone_params = []
        projector_params = []
        action_head_params = []
        
        for param_name, param_tensor in model_state_dict.items():
            # Map rollout name to HF name for consistency
            hf_name = self._rollout_vllm_name_to_hf(param_name)
            
            # Add to inplace view map
            vllm_weight_inplace_view_map[hf_name] = param_tensor
            
            # Group by component type
            if "language_model" in param_name:
                language_model_params.append((hf_name, param_tensor.dim()))
            elif "vision_backbone" in param_name:
                vision_backbone_params.append((hf_name, param_tensor.dim()))
            elif "projector" in param_name:
                projector_params.append((hf_name, param_tensor.dim()))
            elif "action_head" in param_name:
                action_head_params.append((hf_name, param_tensor.dim()))
            else:
                # Default group for other parameters
                language_model_params.append((hf_name, param_tensor.dim()))
        
        # Create recv key rank lists - group similar parameters together for efficiency
        if language_model_params:
            recv_key_n_rank_list.append(language_model_params)
        if vision_backbone_params:
            recv_key_n_rank_list.append(vision_backbone_params)
        if projector_params:
            recv_key_n_rank_list.append(projector_params)
        if action_head_params:
            recv_key_n_rank_list.append(action_head_params)
        logger.info(f"VLA rollout_prepare_recv: prepared {len(vllm_weight_inplace_view_map)} parameters "
                   f"in {len(recv_key_n_rank_list)} groups")
        
        return vllm_weight_inplace_view_map, recv_key_n_rank_list
    
    @property
    def packed_modules_mapping(self) -> Dict[str, List[str]]:
        return {}
            
    def get_unsplited_weight_name(self, weight_name: str) -> str:
        """
        Get the unsplit weight name for tensor parallel weights
        
        This is used to identify which weights belong to the same
        logical parameter when split across multiple devices.
        """
        
        # For attention weights, remove TP suffixes
        if ".q_proj" in weight_name or ".k_proj" in weight_name or ".v_proj" in weight_name:
            # These are split along head dimension
            base_name = weight_name.replace(".q_proj", ".qkv_proj")
            base_name = base_name.replace(".k_proj", ".qkv_proj") 
            base_name = base_name.replace(".v_proj", ".qkv_proj")
            return base_name
            
        elif ".gate_proj" in weight_name or ".up_proj" in weight_name:
            # These are split along hidden dimension
            base_name = weight_name.replace(".gate_proj", ".gate_up_proj")
            base_name = base_name.replace(".up_proj", ".gate_up_proj")
            return base_name
            
        # For other weights, return as-is
        return weight_name
        
    @cached_property  
    def attention_weight_specs(self) -> List[Tuple[str, int, str]]:
        """
        Specification for attention weight tensor parallelism
        
        Returns:
            List of (weight_name_pattern, split_dim, split_type) tuples
        """
        return [
            # Query, Key, Value projections split along head dimension  
            (r".*\.q_proj\.weight", 0, "head"),
            (r".*\.k_proj\.weight", 0, "head"), 
            (r".*\.v_proj\.weight", 0, "head"),
            
            # Output projection splits along input dimension
            (r".*\.o_proj\.weight", 1, "head"),
            
            # MLP gate and up projections split along output dimension
            (r".*\.gate_proj\.weight", 0, "ffn"),
            (r".*\.up_proj\.weight", 0, "ffn"),
            
            # MLP down projection splits along input dimension  
            (r".*\.down_proj\.weight", 1, "ffn"),
        ]
        
    @cached_property
    def embedding_weight_specs(self) -> List[Tuple[str, int, str]]:
        """
        Specification for embedding weight tensor parallelism
        """
        return [
            # Language model embeddings
            (r".*embed_tokens\.weight", 1, "vocab"),
            (r".*lm_head\.weight", 0, "vocab"),
            
            # Action head (if vocab parallel is desired)
            (r".*action_head\.weight", 0, "action"),
        ]
        
    def get_tensor_parallel_split_spec(self, weight_name: str) -> Tuple[int, str]:
        """
        Get tensor parallel split specification for a weight
        
        Args:
            weight_name: Name of the weight tensor
            
        Returns:
            Tuple of (split_dimension, split_type)
            split_dimension: Which dimension to split (-1 for no split)
            split_type: Type of split ("head", "ffn", "vocab", "action", etc.)
        """
        
        # Check attention weights
        for pattern, split_dim, split_type in self.attention_weight_specs:
            if re.match(pattern, weight_name):
                return split_dim, split_type
                
        # Check embedding weights  
        for pattern, split_dim, split_type in self.embedding_weight_specs:
            if re.match(pattern, weight_name):
                return split_dim, split_type
                
        # Vision backbone weights - typically not split
        if "vision_backbone" in weight_name:
            return -1, "none"
            
        # Projector weights - typically not split  
        if "projector" in weight_name:
            return -1, "none"
            
        # Layer norms and biases - typically not split
        if "norm" in weight_name or "bias" in weight_name:
            return -1, "none"
            
        # Default: no split
        return -1, "none"
        
    def get_split_size(self, split_type: str, tp_size: int) -> int:
        """
        Get the size after tensor parallel split
        
        Args:
            split_type: Type of split 
            tp_size: Tensor parallel size
            
        Returns:
            Size of tensor after splitting
        """
        
        if split_type == "head":
            # Attention heads split
            return self.num_attention_heads // tp_size
        elif split_type == "ffn":
            # FFN intermediate size split  
            if hasattr(self.config, 'intermediate_size'):
                return self.config.intermediate_size // tp_size
            else:
                # Default FFN ratio
                return (self.hidden_size * 4) // tp_size
        elif split_type == "vocab":
            # Vocabulary split
            if hasattr(self.config, 'vocab_size'):
                return self.config.vocab_size // tp_size  
            else:
                return 32000 // tp_size  # Default vocab size
        elif split_type == "action":
            # Action dimension split
            return self.action_dim // tp_size
        else:
            # No split
            return -1
    
    def get_policy_parallelism_strategy(self):
        """
        Define parallelism strategies for VLA model components.
        
        All VLA components (vision_backbone, projector, language_model, action_head)
        are now sharded uniformly by FSDP, so they're all in parallelism_info_for_params.
        
        We use automatic inference for all parameters - no special handling needed!
        
        Returns:
            List containing the registered "openvla" strategy function
        """
        
        # Register VLA policy parallelism strategy
        @register_parallelism_strategy(
            "openvla",
            role=ParallelismStrategyRole.POLICY,
            allow_override=True
        )
        def vla_policy_strategy(shape, dest_name, parallelism, hf_config):
            """
            Use automatic inference for all VLA parameters.
            
            All components are FSDP-sharded uniformly, so automatic inference
            will correctly detect the sharding from parallelism_info_for_params.
            
            Args:
                shape: Tensor shape (tuple of ints)
                dest_name: Parameter name (str)
                parallelism: ParallelDims configuration
                hf_config: HuggingFace model config
                
            Returns:
                Tuple of (None, None, None) to trigger automatic inference
            """
            # All parameters: use automatic inference
            return None, None, None
        
        logger.info("[VLAWeightMapper] Registered policy parallelism strategy for P2R weight sync")
        return [get_policy_strategy("openvla")]
    
    def get_rollout_parallelism_strategy(self):
        """
        Define parallelism strategies for VLA rollout workers.
        
        Rollout workers receive weights from policy workers via P2R sync.
        Since all parameters are now uniformly sharded on policy side,
        we use automatic inference for all parameters.
        
        Returns:
            List of strategy functions for rollout recv instructions
        """
        
        # Register VLA rollout parallelism strategy
        @register_parallelism_strategy(
            "openvla",
            role=ParallelismStrategyRole.ROLLOUT,
            allow_override=True
        )
        def vla_rollout_strategy(shape, dest_name, parallelism, hf_config):
            """
            Use automatic inference for all VLA parameters.
            
            Args:
                shape: Tensor shape (tuple of ints)
                dest_name: Parameter name (str)
                parallelism: ParallelDims configuration
                hf_config: HuggingFace model config
                
            Returns:
                Tuple of (None, None, None) to trigger automatic inference
            """
            # All parameters: use automatic inference
            return None, None, None
        
        logger.info("[VLAWeightMapper] Registered rollout parallelism strategy for P2R weight sync")
        return [get_rollout_strategy("openvla")]
    
    def policy_decompose_param_1_to_n_for_sync(self, name):
        """
        Override to prevent parameter decomposition for weight sync.
        
        VLA models do NOT decompose parameters (like qkv -> q,k,v) for P2R weight sync.
        All parameters are synced as complete tensors.
        
        Without this override, the base WeightMapper might try to decompose parameters
        like 'vision_backbone.*.attn.qkv.weight', causing them to be skipped from
        weight_sync_transforms and never synced to rollout workers.
        
        Args:
            name: Parameter name
            
        Returns:
            Empty list [] means no decomposition
        """
        # VLA models: no parameter decomposition for weight sync
        return []
