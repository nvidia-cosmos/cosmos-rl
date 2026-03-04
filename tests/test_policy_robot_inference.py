# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script for Cosmos Policy robot inference using cosmos-rl WFM modules.
# Mirrors the logic of cosmos-policy/test.py: loads the same checkpoint,
# builds the same observation data batch, runs inference, and prints results.
#
# Usage:
#   python tests/test_policy_robot_inference.py \
#       --ckpt-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
#       --observation-pkl cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl \
#       --t5-embeddings-pkl nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
#       --dataset-stats-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json

import argparse
import io
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Prevent transitive imports through cosmos_rl.__init__ (needs timm, etc.)
if "cosmos_rl" not in sys.modules:
    _stub = type(sys)("cosmos_rl")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "cosmos_rl")]
    sys.modules["cosmos_rl"] = _stub

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ---------------------------------------------------------------------------
# Config (mirrors cosmos-policy PolicyEvalConfig fields used in test.py)
# ---------------------------------------------------------------------------

@dataclass
class PolicyTestConfig:
    ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    dataset_stats_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    t5_text_embeddings_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"
    observation_pkl: str = ""
    suite: str = "libero"
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    flip_images: bool = False
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    use_variance_scale: bool = False
    chunk_size: int = 16
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    seed: int = 1


# ---------------------------------------------------------------------------
# Utility functions (self-contained, no cosmos-policy imports)
# ---------------------------------------------------------------------------

ACTION_DIM = 7
COSMOS_IMAGE_SIZE = 224
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4


def resolve_hf_path(path: str) -> str:
    """Download from HuggingFace if path looks like org/repo/file."""
    if path is None or path == "":
        return path
    if "/" in path and not path.startswith("/") and not path.startswith("./"):
        parts = path.split("/")
        if len(parts) == 2:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(repo_id=path, resume_download=True)
            model_dir = os.path.join(local_dir, "model")
            if os.path.exists(model_dir):
                return model_dir
            pt_files = [f for f in os.listdir(local_dir) if f.endswith(".pt")]
            if pt_files:
                return os.path.join(local_dir, pt_files[0])
            return local_dir
        elif len(parts) >= 3:
            from huggingface_hub import hf_hub_download
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            return hf_hub_download(repo_id=repo_id, filename=filename, resume_download=True)
    return path


def load_dataset_stats(path: str) -> dict:
    path = resolve_hf_path(path)
    with open(path, "r") as f:
        stats = json.load(f)
    return {k: np.array(v) for k, v in stats.items()}


def load_t5_embeddings(path: str) -> dict:
    path = resolve_hf_path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def rescale_proprio(proprio: np.ndarray, dataset_stats: dict) -> np.ndarray:
    curr_min = dataset_stats["proprio_min"]
    curr_max = dataset_stats["proprio_max"]
    return 2 * ((proprio - curr_min) / (curr_max - curr_min)) - 1


def unnormalize_actions(actions: np.ndarray, dataset_stats: dict) -> np.ndarray:
    a_min = dataset_stats["actions_min"]
    a_max = dataset_stats["actions_max"]
    orig = actions.shape
    actions = actions.reshape(-1, a_min.shape[0])
    actions = 0.5 * (actions + 1) * (a_max - a_min) + a_min
    return actions.reshape(orig)


def jpeg_compress(images: np.ndarray, quality: int = 95) -> np.ndarray:
    out = []
    for img in images:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(np.array(Image.open(buf)))
    return np.stack(out)


def resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    if images.shape[-3:] == (target_size, target_size, 3):
        return images.copy()
    out = []
    for img in images:
        out.append(np.array(Image.fromarray(img).resize((target_size, target_size))))
    return np.stack(out)


def apply_image_transforms(images: np.ndarray) -> np.ndarray:
    """90%-area center crop + resize back (matches cosmos-policy train-time augmentation)."""
    _, H, W, C = images.shape
    crop_size = int(H * 0.9 ** 0.5)
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    results = [TF.resize(TF.center_crop(img, crop_size), [H, W], antialias=True) for img in images_tensor]
    return torch.stack(results).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def prepare_images(images: List[np.ndarray], cfg: PolicyTestConfig) -> np.ndarray:
    images = np.stack(images, axis=0)
    if cfg.flip_images:
        images = np.flipud(images)
    if cfg.use_jpeg_compression:
        images = jpeg_compress(images, quality=95)
    images = resize_images(images, COSMOS_IMAGE_SIZE)
    if cfg.trained_with_image_aug:
        images = apply_image_transforms(images)
    return images


def duplicate_array(arr: np.ndarray, total_num_copies: int) -> np.ndarray:
    return np.stack([arr] * total_num_copies)


# ---------------------------------------------------------------------------
# get_action – builds data batch and runs inference (mirrors cosmos_utils.get_action)
# ---------------------------------------------------------------------------

def get_action(
    cfg: PolicyTestConfig,
    model,
    dataset_stats: dict,
    obs: Dict[str, Any],
    task_label: str,
    t5_cache: dict,
    seed: int = 42,
    num_denoising_steps_action: int = 5,
    generate_future_state_and_value_in_parallel: bool = True,
    batch_size: int = 1,
) -> Dict[str, Any]:
    from cosmos_rl.policy.model.wfm.cosmos_policy import (
        _extract_action_chunk as extract_action_chunk,
        _extract_value as extract_value,
        ACTION_DIM,
    )

    with torch.inference_mode():
        text_embedding = t5_cache[task_label]

        all_camera_images = [obs["wrist_image"], obs["primary_image"]]
        all_camera_images = prepare_images(all_camera_images, cfg)

        proprio = None
        if cfg.use_proprio:
            proprio = obs["proprio"]
            if cfg.normalize_proprio:
                proprio = rescale_proprio(proprio, dataset_stats)

        primary_image = all_camera_images[1]
        blank_image = np.zeros_like(primary_image)
        blank_dup = duplicate_array(blank_image.copy(), COSMOS_TEMPORAL_COMPRESSION_FACTOR)

        image_sequence = []
        seq_idx = 0

        # 0: blank
        image_sequence.append(np.expand_dims(blank_image, axis=0))
        seq_idx += 1

        # 1: current proprio (blank placeholder)
        if cfg.use_proprio:
            image_sequence.append(blank_dup.copy())
            current_proprio_latent_idx = seq_idx
            seq_idx += 1

        # 2: current wrist image
        wrist_image = all_camera_images[0]
        wrist_dup = duplicate_array(wrist_image, COSMOS_TEMPORAL_COMPRESSION_FACTOR)
        image_sequence.append(wrist_dup)
        current_wrist_image_latent_idx = seq_idx
        seq_idx += 1

        # 3: current primary image
        primary_dup = duplicate_array(primary_image, COSMOS_TEMPORAL_COMPRESSION_FACTOR)
        image_sequence.append(primary_dup)
        current_image_latent_idx = seq_idx
        seq_idx += 1

        # 4: action (blank placeholder)
        image_sequence.append(blank_dup.copy())
        action_latent_idx = seq_idx
        seq_idx += 1

        # 5: future proprio (blank placeholder)
        if cfg.use_proprio:
            image_sequence.append(blank_dup.copy())
            future_proprio_latent_idx = seq_idx
            seq_idx += 1

        # 6: future wrist image
        image_sequence.append(wrist_dup.copy())
        future_wrist_image_latent_idx = seq_idx
        seq_idx += 1

        # 7: future primary image
        image_sequence.append(primary_dup.copy())
        future_image_latent_idx = seq_idx
        seq_idx += 1

        # 8: value (blank placeholder)
        image_sequence.append(blank_dup.copy())
        value_latent_idx = seq_idx
        seq_idx += 1

        for i, t in enumerate(image_sequence):
            print(f"Image sequence {i} shape: {t.shape}")

        raw_image_sequence = np.concatenate(image_sequence, axis=0)
        raw_image_sequence = np.expand_dims(raw_image_sequence, axis=0)
        raw_image_sequence = np.tile(raw_image_sequence, (batch_size, 1, 1, 1, 1))
        raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))
        raw_image_sequence = torch.from_numpy(raw_image_sequence).to(dtype=torch.uint8).cuda()

        proprio_tensor = None
        if cfg.use_proprio:
            proprio_tensor = torch.from_numpy(proprio).reshape(batch_size, -1).to(dtype=torch.bfloat16).cuda()

        data_batch = {
            "dataset_name": "video_data",
            "video": raw_image_sequence,
            "t5_text_embeddings": text_embedding.repeat(batch_size, 1, 1).to(dtype=torch.bfloat16).cuda(),
            "fps": torch.tensor([16] * batch_size, dtype=torch.bfloat16).cuda(),
            "padding_mask": torch.zeros((batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16).cuda(),
            "num_conditional_frames": model.config.min_num_conditional_frames,
            "proprio": proprio_tensor,
            "current_proprio_latent_idx": torch.tensor([current_proprio_latent_idx] * batch_size, dtype=torch.int64).cuda() if cfg.use_proprio else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda(),
            "current_wrist_image_latent_idx": torch.tensor([current_wrist_image_latent_idx] * batch_size, dtype=torch.int64).cuda(),
            "current_wrist_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64).cuda(),
            "current_image_latent_idx": torch.tensor([current_image_latent_idx] * batch_size, dtype=torch.int64).cuda(),
            "current_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64).cuda(),
            "action_latent_idx": torch.tensor([action_latent_idx] * batch_size, dtype=torch.int64).cuda(),
            "future_proprio_latent_idx": torch.tensor([future_proprio_latent_idx] * batch_size, dtype=torch.int64).cuda() if cfg.use_proprio else torch.tensor([-1] * batch_size, dtype=torch.int64).cuda(),
            "future_wrist_image_latent_idx": torch.tensor([future_wrist_image_latent_idx] * batch_size, dtype=torch.int64).cuda(),
            "future_wrist_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64).cuda(),
            "future_image_latent_idx": torch.tensor([future_image_latent_idx] * batch_size, dtype=torch.int64).cuda(),
            "future_image2_latent_idx": torch.tensor([-1] * batch_size, dtype=torch.int64).cuda(),
            "value_latent_idx": torch.tensor([value_latent_idx] * batch_size, dtype=torch.int64).cuda(),
        }

        generated_latent, orig_clean_latent_frames = model.generate_samples_from_batch(
            data_batch,
            n_sample=batch_size,
            num_steps=num_denoising_steps_action,
            seed=seed,
            is_negative_prompt=False,
            use_variance_scale=cfg.use_variance_scale,
            return_orig_clean_latent_frames=True,
        )

        action_indices = torch.full(
            (batch_size,), action_latent_idx, dtype=torch.int64, device=generated_latent.device
        )
        actions = (
            extract_action_chunk(generated_latent, (cfg.chunk_size, ACTION_DIM), action_indices)
            .to(torch.float32).cpu().numpy()
        )
        if cfg.unnormalize_actions:
            actions = unnormalize_actions(actions, dataset_stats)

        future_image_predictions = {}
        value_prediction = None
        if generate_future_state_and_value_in_parallel:
            INDICES_TO_REPLACE = [0, 1, 4, 5]
            sample = generated_latent.clone()
            for idx in INDICES_TO_REPLACE:
                sample[:, :, idx, :, :] = orig_clean_latent_frames[:, :, idx, :, :]
            decoded = ((model.decode(sample) + 1.0) * 127.5).clamp(0, 255)
            decoded = decoded.permute(0, 2, 3, 4, 1).to(torch.uint8).cpu().numpy()

            fw_raw = (future_wrist_image_latent_idx - 1) * COSMOS_TEMPORAL_COMPRESSION_FACTOR + 1
            fi_raw = (future_image_latent_idx - 1) * COSMOS_TEMPORAL_COMPRESSION_FACTOR + 1
            future_image_predictions["future_wrist_image"] = decoded[:, fw_raw]
            future_image_predictions["future_image"] = decoded[:, fi_raw]

            v_indices = torch.full((batch_size,), -1, dtype=torch.int64, device=generated_latent.device)
            value_prediction = extract_value(generated_latent, v_indices)
            value_prediction = torch.clamp((value_prediction + 1) / 2, 0, 1)

        actions = actions[0]
        actions = [actions[i] for i in range(len(actions))]
        future_image_predictions = {k: v[0] for k, v in future_image_predictions.items() if v is not None}
        value_prediction = value_prediction[0].item() if value_prediction is not None else None

        return dict(
            actions=actions,
            future_image_predictions=future_image_predictions,
            value_prediction=value_prediction,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(cfg: PolicyTestConfig):
    """Load model, run inference on sample observation, print results."""
    from cosmos_rl.policy.model.wfm.cosmos_policy import CosmosPolicy, CosmosPolicyConfig

    print("Loading dataset stats...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    print("Loading T5 text embeddings cache...")
    t5_cache = load_t5_embeddings(cfg.t5_text_embeddings_path)

    print("Loading model...")
    cosmos_policy = CosmosPolicy.from_pretrained(None, cfg.ckpt_path)
    model = cosmos_policy.generative_model
    model_config = model.config

    print("Loading observation...")
    with open(cfg.observation_pkl, "rb") as f:
        observation = pickle.load(f)
    task_description = "put both the alphabet soup and the tomato sauce in the basket"

    for k, v in observation.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(f"Observation {k}: {v.shape}")
        else:
            print(f"Observation {k}: {v}")

    print("\nRunning inference...")
    result = get_action(
        cfg, model, dataset_stats, observation, task_description, t5_cache,
        seed=cfg.seed,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )

    print(f"\nGenerated action chunk: {result['actions']}")
    print(f"Generated value: {result['value_prediction']}")

    if result["future_image_predictions"]:
        for name, img in result["future_image_predictions"].items():
            path = f"{name}.png"
            Image.fromarray(img).save(path)
            print(f"Saved {name} to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmos Policy robot inference test (cosmos-rl)")
    parser.add_argument("--ckpt-path", type=str, default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    parser.add_argument("--observation-pkl", type=str, default="../cosmos-policy/cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl")
    parser.add_argument("--t5-embeddings-pkl", type=str, default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl")
    parser.add_argument("--dataset-stats-path", type=str, default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-denoising-steps", type=int, default=5)
    args = parser.parse_args()

    cfg = PolicyTestConfig(
        ckpt_path=args.ckpt_path,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_embeddings_pkl,
        observation_pkl=args.observation_pkl,
        seed=args.seed,
        num_denoising_steps_action=args.num_denoising_steps,
    )
    run_inference(cfg)
