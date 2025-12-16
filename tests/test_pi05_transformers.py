# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test PI05 forward pass and compare with official openpi implementation.

Usage:
    python tests/test_pi05_forward.py
"""

import sys

sys.path.insert(0, "/workspace/test_cosmos")
from cosmos_rl.policy.model.pi05 import PI05
# from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

from openpi.models.pi0_config import Pi0Config as OpenPiConfig

import os
import torch
import safetensors.torch
from huggingface_hub import hf_hub_download, snapshot_download


def create_dummy_input(
    batch_size: int = 1,
    action_dim: int = 7,
    action_horizon: int = 50,
    device: str = "cuda",
):
    """Create dummy observation and actions for PI05 model."""

    class Observation:
        pass

    obs = Observation()

    # Images: [B, C, H, W] in range [-1, 1]
    obs.images = {
        "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device) * 0.5,
        "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device) * 0.5,
        "right_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device) * 0.5,
    }

    obs.image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
    }

    obs.state = torch.randn(batch_size, action_dim, device=device)
    obs.tokenized_prompt = torch.randint(0, 1000, (batch_size, 64), device=device)
    obs.tokenized_prompt_mask = torch.ones(
        batch_size, 64, dtype=torch.bool, device=device
    )
    obs.token_ar_mask = torch.zeros(batch_size, 64, dtype=torch.bool, device=device)
    obs.token_loss_mask = torch.zeros(batch_size, 64, dtype=torch.bool, device=device)

    actions = torch.randn(batch_size, action_horizon, action_dim, device=device)

    return obs, actions


def check_output(name: str, loss: torch.Tensor):
    """Print output info."""
    print(
        f"[{name}] Shape: {loss.shape}, Mean: {loss.mean().item():.6f}, Finite: {torch.isfinite(loss).all().item()}"
    )


def save_inputs(obs, actions, noise, time, save_dir: str = "/workspace"):
    """Save inputs to local directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save images and masks
    images_data = {k: v for k, v in obs.images.items()}
    masks_data = {k: v for k, v in obs.image_masks.items()}
    
    save_dict = {
        "state": obs.state,
        "tokenized_prompt": obs.tokenized_prompt,
        "tokenized_prompt_mask": obs.tokenized_prompt_mask,
        "token_ar_mask": obs.token_ar_mask,
        "token_loss_mask": obs.token_loss_mask,
        "actions": actions,
        "noise": noise,
        "time": time,
        **{f"image_{k}": v for k, v in images_data.items()},
        **{f"mask_{k}": v for k, v in masks_data.items()},
    }
    
    save_path = os.path.join(save_dir, "inputs.safetensors")
    safetensors.torch.save_file(save_dict, save_path)
    print(f"Inputs saved to: {save_path}")


def load_inputs(save_dir: str = "/workspace", device: str = "cuda"):
    """Load inputs from local directory."""
    load_path = os.path.join(save_dir, "inputs.safetensors")
    data = safetensors.torch.load_file(load_path, device=device)
    
    class Observation:
        pass
    
    obs = Observation()
    
    # Reconstruct images and masks
    obs.images = {}
    obs.image_masks = {}
    for key, value in data.items():
        if key.startswith("image_"):
            obs.images[key[6:]] = value
        elif key.startswith("mask_"):
            obs.image_masks[key[5:]] = value
    
    obs.state = data["state"]
    obs.tokenized_prompt = data["tokenized_prompt"]
    obs.tokenized_prompt_mask = data["tokenized_prompt_mask"]
    obs.token_ar_mask = data["token_ar_mask"]
    obs.token_loss_mask = data["token_loss_mask"]
    
    actions = data["actions"]
    noise = data["noise"]
    time = data["time"]
    
    print(f"Inputs loaded from: {load_path}")
    return obs, actions, noise, time


if __name__ == "__main__":
    device = "cuda"
    model_id = "sunshk/pi05_libero_pytorch"

    # Model config for LIBERO (from openpi pi05_libero config)
    action_dim = 32
    action_horizon = 10

    # Download model weights
    print(f"Downloading model from {model_id}...")
    model_dir = snapshot_download(repo_id=model_id)
    weight_path = os.path.join(model_dir, "model.safetensors")
    print(f"Model downloaded to: {model_dir}")

    # Create openpi config (this is what openpi uses internally)
    openpi_config = OpenPiConfig(
        pi05=True,
        action_dim=action_dim,
        action_horizon=action_horizon,
        discrete_state_input=False,
    )

    # Create shared dummy input (use same random seed for fair comparison)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    obs, actions = create_dummy_input(
        batch_size=1,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=device,
    )
    noise = torch.randn_like(actions)
    time = torch.rand(1, device=device) * 0.999 + 0.001

    # Load inputs from /workspace
    # obs, actions, noise, time = load_inputs(save_dir="/workspace", device=device)

    # ========== Cosmos-RL PI05 ==========
    print("\n[Cosmos-RL PI05]")
    cosmos_model = PI05(model_id, openpi_config).to(device)
    safetensors.torch.load_model(cosmos_model, weight_path)
    cosmos_model.eval()

    # Reset seed again to match the same augmentations
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cosmos_loss = cosmos_model.forward(obs, actions, noise=noise, time=time)
    check_output("Cosmos-RL", cosmos_loss)
