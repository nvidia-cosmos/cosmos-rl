# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test PI05 forward pass with cosmos-rl implementation.

Usage:
    python tests/test_pi05_transformers.py
"""

import os

import torch
import safetensors.torch
import sys
sys.path.append("/workspace/test_cosmos")  # Ensure openpi is in the path
from cosmos_rl.policy.model.pi05 import PI05
from transformers import AutoConfig

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

    # ========== Load config via HF ==========
    # This will error if the repo/folder does not contain a valid config.json (as desired).
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)

    # ========== Build model + load weights ==========
    print(f"\n[Loading PI05 from {model_id}]")
    
    cosmos_model = PI05(model_id, hf_config).to(device)
    
    # Load weights
    cosmos_model.load_hf_weights(model_id)
    cosmos_model.eval()

    # ========== Create test input ==========
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    obs, actions = create_dummy_input(
        batch_size=1,
        action_dim=cosmos_model.action_dim,
        action_horizon=cosmos_model.action_horizon,
        device=device,
    )
    noise = torch.randn_like(actions)
    time = torch.rand(1, device=device) * 0.999 + 0.001

    # ========== Forward pass ==========
    print("\n[Cosmos-RL PI05]")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cosmos_loss = cosmos_model.forward(obs, actions, noise=noise, time=time)
    check_output("Cosmos-RL", cosmos_loss)

    print("\n✅ Test completed successfully!")
