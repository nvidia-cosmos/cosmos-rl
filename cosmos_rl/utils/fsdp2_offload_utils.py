# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Manual CPU offloading utilities for FSDP2 models.

Provides granular control over offloading parameters, gradients, and optimizer states,
similar to SimpleVLA-RL's approach but compatible with FSDP2's composable API.

Key benefits over FSDP2's CPUOffloadPolicy:
1. Separate control over parameters, gradients, and optimizer states
2. Can offload/load at specific points in training pipeline
3. Can store parameters as FP32 on CPU for numerical stability while using BF16 on GPU
"""

import torch
import torch.nn as nn
from typing import Optional
from torch.optim import Optimizer


def offload_fsdp2_parameters(
    module: nn.Module,
    store_as_fp32: bool = False,
    offload_grad: bool = False,
):
    """
    Offload FSDP2-sharded parameters (and optionally gradients) to CPU.
    
    Args:
        module: The FSDP2-wrapped module
        store_as_fp32: If True, convert BF16 parameters to FP32 on CPU for numerical stability
        offload_grad: If True, also offload gradients to CPU
    """
    for name, param in module.named_parameters():
        # Handle FSDP2's DTensor structure
        if hasattr(param, '_local_tensor'):
            # FSDP2 uses DTensor - offload the local shard
            local_tensor = param._local_tensor
            if local_tensor.is_cuda:
                if store_as_fp32 and local_tensor.dtype == torch.bfloat16:
                    param._local_tensor = local_tensor.to('cpu', dtype=torch.float32, non_blocking=True)
                else:
                    param._local_tensor = local_tensor.to('cpu', non_blocking=True)
        elif param.data.is_cuda:
            # Regular tensor (not DTensor)
            if store_as_fp32 and param.data.dtype == torch.bfloat16:
                param.data = param.data.to('cpu', dtype=torch.float32, non_blocking=True)
            else:
                param.data = param.data.to('cpu', non_blocking=True)
        
        # Offload gradient if requested
        if offload_grad and param.grad is not None and param.grad.is_cuda:
            # Gradients should always be FP32 for numerical stability
            param.grad = param.grad.to('cpu', dtype=torch.float32, non_blocking=True)
    
    torch.cuda.empty_cache()


def load_fsdp2_parameters(
    module: nn.Module,
    device_id: torch.device,
    target_param_dtype: torch.dtype = torch.bfloat16,
    load_grad: bool = False,
):
    """
    Load FSDP2-sharded parameters (and optionally gradients) back to GPU.
    
    Args:
        module: The FSDP2-wrapped module
        device_id: Target GPU device
        target_param_dtype: Target dtype for parameters (typically BF16 for training)
        load_grad: If True, also load gradients back to GPU
    """
    for name, param in module.named_parameters():
        # Handle FSDP2's DTensor structure
        if hasattr(param, '_local_tensor'):
            local_tensor = param._local_tensor
            if not local_tensor.is_cuda:
                # Convert back to target dtype if needed
                if local_tensor.dtype != target_param_dtype:
                    param._local_tensor = local_tensor.to(device_id, dtype=target_param_dtype, non_blocking=True)
                else:
                    param._local_tensor = local_tensor.to(device_id, non_blocking=True)
        elif not param.data.is_cuda:
            # Regular tensor
            if param.data.dtype != target_param_dtype:
                param.data = param.data.to(device_id, dtype=target_param_dtype, non_blocking=True)
            else:
                param.data = param.data.to(device_id, non_blocking=True)
        
        # Load gradient if requested
        if load_grad and param.grad is not None and not param.grad.is_cuda:
            param.grad = param.grad.to(device_id, non_blocking=True)
    
    torch.cuda.empty_cache()


def offload_fsdp2_gradients(module: nn.Module):
    """
    Offload only gradients to CPU (keep parameters on GPU).
    Useful after backward pass before optimizer step.
    """
    for param in module.parameters():
        if param.grad is not None and param.grad.is_cuda:
            param.grad = param.grad.to('cpu', dtype=torch.float32, non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp2_gradients(module: nn.Module, device_id: torch.device):
    """Load only gradients back to GPU."""
    for param in module.parameters():
        if param.grad is not None and not param.grad.is_cuda:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def offload_optimizer_states(optimizer: Optimizer):
    """
    Offload optimizer states (momentum, variance, etc.) to CPU.
    
    This is the most memory-intensive part and provides the biggest savings.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    state[key] = value.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()


def load_optimizer_states(optimizer: Optimizer, device_id: torch.device):
    """Load optimizer states back to GPU."""
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and not value.is_cuda:
                    state[key] = value.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


class FSDP2OffloadManager:
    """
    High-level manager for FSDP2 offloading operations.
    
    Provides a convenient interface similar to SimpleVLA-RL's pattern.
    
    Example usage:
        manager = FSDP2OffloadManager(
            offload_params=True,
            offload_grads=True,
            offload_optimizer=True,
            store_params_as_fp32=True,  # For numerical stability
        )
        
        # After initialization
        manager.offload_all(model, optimizer)
        
        # Before training step
        manager.load_for_training(model, optimizer)
        
        # After training step
        manager.offload_after_training(model, optimizer)
        
        # Before rollout/inference
        manager.load_for_inference(model)
        
        # After rollout/inference
        manager.offload_after_inference(model)
    """
    
    def __init__(
        self,
        offload_params: bool = False,
        offload_grads: bool = False,
        offload_optimizer: bool = False,
        store_params_as_fp32: bool = False,
        target_param_dtype: torch.dtype = torch.bfloat16,
    ):
        self.offload_params = offload_params
        self.offload_grads = offload_grads
        self.offload_optimizer = offload_optimizer
        self.store_params_as_fp32 = store_params_as_fp32
        self.target_param_dtype = target_param_dtype
    
    def offload_all(self, module: nn.Module, optimizer: Optional[Optimizer] = None):
        """Offload everything configured to be offloaded."""
        if self.offload_params:
            offload_fsdp2_parameters(
                module, 
                store_as_fp32=self.store_params_as_fp32,
                offload_grad=self.offload_grads
            )
        elif self.offload_grads:
            offload_fsdp2_gradients(module)
        
        if self.offload_optimizer and optimizer is not None:
            offload_optimizer_states(optimizer)
    
    def load_for_training(self, module: nn.Module, optimizer: Optional[Optimizer] = None):
        """Load everything needed for training (forward + backward + optimizer step)."""
        device_id = torch.cuda.current_device()
        
        if self.offload_params:
            load_fsdp2_parameters(
                module,
                device_id,
                target_param_dtype=self.target_param_dtype,
                load_grad=self.offload_grads
            )
        elif self.offload_grads:
            load_fsdp2_gradients(module, device_id)
        
        if self.offload_optimizer and optimizer is not None:
            load_optimizer_states(optimizer, device_id)
    
    def offload_after_training(self, module: nn.Module, optimizer: Optional[Optimizer] = None):
        """Offload after a training step completes."""
        self.offload_all(module, optimizer)
    
    def load_for_inference(self, module: nn.Module):
        """Load only parameters needed for inference (no grads, no optimizer)."""
        if self.offload_params:
            device_id = torch.cuda.current_device()
            load_fsdp2_parameters(
                module,
                device_id,
                target_param_dtype=self.target_param_dtype,
                load_grad=False
            )
    
    def offload_after_inference(self, module: nn.Module):
        """Offload after inference completes."""
        if self.offload_params:
            offload_fsdp2_parameters(
                module,
                store_as_fp32=self.store_params_as_fp32,
                offload_grad=False
            )

