# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test PI05 forward pass and compare with official openpi implementation.

Usage:
    python tests/test_pi05_forward.py
"""

import logging
import sys

sys.path.insert(0, "/workspace/test_cosmos/openpi/src/openpi")
from cosmos_rl.policy.model.pi05 import PI05
from transformers import AutoConfig
from contextlib import nullcontext

import os
import torch
import safetensors.torch
from huggingface_hub import snapshot_download


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
    state_dim: int = 7,
    action_dim: int = 32,
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

    # Proprio/state (env dimension, e.g. 7 for LIBERO)
    obs.state = torch.randn(batch_size, state_dim, device=device)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = "cuda"
    model_id = "sunshk/pi05_libero_pytorch"

    # Download model weights
    print(f"Downloading model from {model_id}...")
    model_dir = snapshot_download(repo_id=model_id)
    weight_path = os.path.join(model_dir, "model.safetensors")
    print(f"Model downloaded to: {model_dir}")

    # ========== Load config via HF (Cosmos-RL PI05) ==========
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hf_config = _inject_pi05_runtime_defaults(hf_config)

    # ========== Cosmos-RL PI05 ==========
    print("\n[Cosmos-RL PI05]")
    cosmos_model = PI05(model_id, hf_config).to(device)
    cosmos_model.load_hf_weights(model_id)
    cosmos_model = cosmos_model.to(device)
    cosmos_model.eval()

    # Create shared dummy input (use same random seed for fair comparison)
    # IMPORTANT: PI05 expects `actions.shape[-1] == action_dim` (e.g. 32), not env dim (e.g. 7).
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    obs, actions = create_dummy_input(
        batch_size=1,
        state_dim=cosmos_model.action_env_dim,
        action_dim=cosmos_model.action_dim,
        action_horizon=cosmos_model.action_horizon,
        device=device,
    )
    noise = torch.randn_like(actions)
    time = torch.rand(1, device=device) * 0.999 + 0.001

    # Safe autocast based on actual weight dtype to avoid dtype mismatch errors.
    cosmos_weight_dtype = next(cosmos_model.parameters()).dtype
    cosmos_use_amp = (device == "cuda") and (
        cosmos_weight_dtype in (torch.float16, torch.bfloat16)
    )
    cosmos_amp_ctx = (
        torch.autocast(device_type="cuda", dtype=cosmos_weight_dtype)
        if cosmos_use_amp
        else nullcontext()
    )

    # Reset seed again to match the same augmentations in preprocessing
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    with torch.no_grad(), cosmos_amp_ctx:
        cosmos_loss = cosmos_model.forward(obs, actions, noise=noise, time=time)
    check_output("Cosmos-RL", cosmos_loss)

    # ========== Official OpenPI ==========
    print("\n[Official OpenPI]")
    try:
        # OpenPI (official) reference implementation
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models.pi0_config import Pi0Config as OpenPiConfig
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenPI is not importable. Please ensure `openpi` is installed and on PYTHONPATH.\n"
            f"Import error: {e}"
        ) from e

    # Create OpenPI config consistent with the HF config used above.
    openpi_config = OpenPiConfig(
        pi05=True,
        action_dim=cosmos_model.action_dim,
        action_horizon=cosmos_model.action_horizon,
        discrete_state_input=bool(getattr(hf_config, "discrete_state_input", False)),
    )

    openpi_model = PI0Pytorch(config=openpi_config).to(device)
    safetensors.torch.load_model(openpi_model, weight_path)
    openpi_model.eval()

    openpi_weight_dtype = next(openpi_model.parameters()).dtype
    openpi_use_amp = (device == "cuda") and (
        openpi_weight_dtype in (torch.float16, torch.bfloat16)
    )
    openpi_amp_ctx = (
        torch.autocast(device_type="cuda", dtype=openpi_weight_dtype)
        if openpi_use_amp
        else nullcontext()
    )

    # Reset seed before forward to ensure deterministic preprocessing augmentations
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    with torch.no_grad(), openpi_amp_ctx:
        openpi_loss = openpi_model.forward(obs, actions, noise=noise, time=time)
    check_output("OpenPI", openpi_loss)

    # ========== Compare ==========
    print("\n[Comparison]")
    diff = (cosmos_loss.float() - openpi_loss.float()).abs()
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")

    # Tolerance: allow small numerical differences (AMP / kernel-level nondeterminism).
    tol = 1e-3
    if diff.max().item() < tol:
        print("\n✅ Outputs match!")
    else:
        raise AssertionError(
            f"Outputs differ: max diff {diff.max().item():.6f} >= tol {tol}"
        )
