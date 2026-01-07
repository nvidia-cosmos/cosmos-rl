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

from contextlib import nullcontext


def _inject_pi05_runtime_defaults(hf_config):
    """
    Pi05Config (HF) only stores core model fields. Cosmos-RL PI05 also expects
    several runtime-only knobs (GRPO/OpenPI style). In training, these are injected
    by `PI05.preprocess_hf_config`; for this standalone test we add safe defaults.
    """
    defaults = {
        "num_steps": 10,
        "action_chunk": 5,
        "action_env_dim": 7,
        "noise_method": "flow_sde",
        "noise_level": 0.5,
        "noise_anneal": False,
        "noise_params": [0.7, 0.3, 400],
        "noise_logvar_range": [0.08, 0.16],
        "joint_logprob": False,
        "safe_get_logprob": False,
        "ignore_last": False,
        "train_expert_only": True,
        "discrete_state_input": False,
        "max_token_len": 200,
        "cosmos_compile": False,
    }
    for k, v in defaults.items():
        if not hasattr(hf_config, k):
            setattr(hf_config, k, v)
    return hf_config


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
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hf_config = _inject_pi05_runtime_defaults(hf_config)

    # ========== Build model + load weights ==========
    print(f"\n[Loading PI05 from {model_id}]")

    cosmos_model = PI05(model_id, hf_config).to(device)

    # Load weights
    cosmos_model.load_hf_weights(model_id)
    # Some checkpoints/loaders may leave a subset of params on CPU; force-move after load.
    cosmos_model = cosmos_model.to(device)
    cosmos_model.eval()

    # Pick a safe autocast setting based on actual weight dtype to avoid
    # "Input type (CUDABFloat16Type) and weight type (torch.FloatTensor) should be the same".
    model_weight_dtype = next(cosmos_model.parameters()).dtype
    use_cuda_amp = (device == "cuda") and (
        model_weight_dtype in (torch.float16, torch.bfloat16)
    )
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=model_weight_dtype)
        if use_cuda_amp
        else nullcontext()
    )

    # Sanity check: all parameters should be on the same device as inputs.
    param_devices = {p.device.type for p in cosmos_model.parameters()}
    if len(param_devices) != 1 or (device == "cuda" and "cuda" not in param_devices):
        raise RuntimeError(f"Model parameter devices mismatch: {sorted(param_devices)}")

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

    # ========== Forward pass (training loss) ==========
    print("\n[Cosmos-RL PI05 Forward Pass]")
    with torch.no_grad():
        with amp_ctx:
            cosmos_loss = cosmos_model.forward(obs, actions, noise=noise, time=time)
    check_output("Forward Loss", cosmos_loss)

    # # ========== Sample actions (inference) ==========
    # print("\n[Cosmos-RL PI05 Sample Actions]")
    # with torch.no_grad():
    #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #         result = cosmos_model.sample_actions(device, obs, noise=None, mode="eval")

    # print(f"  Actions shape: {result['actions'].shape}")
    # print(f"  Chains shape: {result['chains'].shape}")
    # print(f"  Denoise inds shape: {result['denoise_inds'].shape}")
    # print(f"  Old log probs shape: {result['old_log_probs'].shape}")
    # print(f"  Actions finite: {torch.isfinite(result['actions']).all().item()}")

    print("\n✅ Test completed successfully!")
