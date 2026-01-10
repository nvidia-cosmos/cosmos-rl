from __future__ import annotations

from typing import Any, Dict, Optional

import argparse
import toml
import torch
import json
from types import SimpleNamespace
from torch.utils.data import Dataset
import os
import einops
import numpy as np
from transformers import AutoTokenizer
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher

from PIL import Image


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Resize a single PIL image with padding to target size."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.array(zero_image)


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


class InjectDefaultPrompt:
    def __init__(self, prompt=None):
        self.prompt = prompt

    def __call__(self, data):
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data

class ResizeImages:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, data):
        data["image"] = {k: resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


class TokenizePrompt:
    def __init__(self, tokenizer, discrete_state_input=False):
        self.tokenizer = tokenizer
        self.discrete_state_input = discrete_state_input

    def __call__(self, data):
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize_openpi(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x

class PadStatesAndActions:
    def __init__(self, model_action_dim):
        self.model_action_dim = model_action_dim

    def __call__(self, data):
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def _convert_norm_stats_tree(obj: Any) -> Any:
    """
    Convert norm_stats json subtree into a tree whose leaves are *objects* (not dicts),
    so `flatten_dict` / `apply_tree` treat them as leaves.
    """
    if isinstance(obj, dict) and "mean" in obj and "std" in obj:
        return SimpleNamespace(
            mean=np.asarray(obj["mean"]),
            std=np.asarray(obj["std"]),
            q01=np.asarray(obj["q01"]) if "q01" in obj else None,
            q99=np.asarray(obj["q99"]) if "q99" in obj else None,
        )
    if isinstance(obj, dict):
        return {k: _convert_norm_stats_tree(v) for k, v in obj.items()}
    return obj


def load_norm_stats(norm_stats_path: str) -> dict[str, Any]:
    """Load norm_stats.json and convert leaf stats dicts into `NormStats`."""
    with open(norm_stats_path) as f:
        data = json.load(f)["norm_stats"]
    converted = _convert_norm_stats_tree(data)
    return converted


class PromptFromLeRobotTask:
    """
    Minimal transform: inject a string prompt based on `task_index`.

    This mirrors the simple behavior used in openpi, without extra Protocol/typing
    scaffolding.
    """

    def __init__(self, tasks: Dict[int, str]):
        self.tasks = tasks

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


class RepackTransform:
    # filter out keys that are not in the structure, and rearrange the keys
    def __init__(self, structure: dict[str, str] | None = None):
        if structure is None:
            structure = {
            "observation/egocentric_camera": "observation.images.rgb.head",
            "observation/wrist_image_left": "observation.images.rgb.left_wrist",
            "observation/wrist_image_right": "observation.images.rgb.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        self.structure = structure
    
    def __call__(self, data):
        flat = self._flatten_dict(data)
        return {new_key: flat[old_key] for new_key, old_key in self.structure.items()}
    
    def _flatten_dict(self, d, parent_key="", sep="/"):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep))
            else:
                items[new_key] = v
        return items


def flatten_dict(tree, parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    """Flatten a nested dict. Uses '/' as the separator."""
    items: dict[str, Any] = {}
    for k, v in tree.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def unflatten_dict(flat: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    """Unflatten a dict that was flattened with `flatten_dict`."""
    root: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(sep) if sep in key else [key]
        cur = root
        for p in parts[:-1]:
            nxt = cur.get(p)
            if nxt is None:
                nxt = {}
                cur[p] = nxt
            elif not isinstance(nxt, dict):
                raise ValueError(f"Cannot unflatten: leaf {p!r} aliases a node for key={key!r}")
            cur = nxt
        last = parts[-1]
        if last in cur and isinstance(cur[last], dict):
            raise ValueError(f"Cannot unflatten: node {last!r} aliases a leaf for key={key!r}")
        cur[last] = value
    return root


def apply_tree(tree, selector, fn, *, strict: bool = False):
    """
    Apply `fn(value, selector_value)` to each leaf in `tree` whose flattened key exists in `selector`.

    Ported from openpi `transforms.apply_tree`.
    """
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v):
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def _assert_quantile_stats(norm_stats) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantiles is True. Key {k} is missing q01 or q99."
            )

class NormalizeTransform:
    def __init__(self, norm_stats=None, use_quantiles=False, strict=False):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict

        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: dict) -> dict:
        if self.norm_stats is None:
            return data
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    left_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)  # 1
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        arm_right_qpos,
        left_gripper_width,
        right_gripper_width,
    ], axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


class B1kInputsTransform:
    def __init__(self, action_dim, model_type="PI05"):
        self.model_type = model_type
        self.action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action = data["actions"]

        # LeRobot stores images as float32 (C,H,W); normalize to uint8 (H,W,C)
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        if self.model_type in ("PI0", "PI05"):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image_left, wrist_image_right)
            image_masks = (np.True_, np.True_, np.True_)
        elif self.model_type == "PI0_FAST":
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, wrist_image_left, wrist_image_right)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class BehaviorSFTDataset(Dataset):
    """
    Thin wrapper around `BehaviorLeRobotDataset` used for SFT.

    We intentionally keep this minimal:
    - dataset construction lives here
    - optional per-sample transform happens in __getitem__
    """

    def __init__(self, config: CosmosConfig):
        self.config = config

        # Load dataset
        self.dataset = BehaviorLeRobotDataset(
            repo_id=config.train.train_policy.dataset.repo_id,
            root=config.train.train_policy.dataset.root,
            tasks=config.train.train_policy.dataset.tasks,
            modalities=config.train.train_policy.dataset.modalities,
            local_only=True,
            delta_timestamps={
                key: [t / 30.0 for t in range(config.custom['action_horizon'])]
                for key in config.custom['action_sequence_keys']
            },
            episodes=config.custom['episodes_index'],
            chunk_streaming_using_keyframe=True,
            shuffle=True,
        )

        # Load norm stats
        self.init_norm_stats()


        # build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy.model_name_or_path, max_len=config.custom['max_token_len'], trust_remote_code=True)

        # Load transforms
        self._transforms = []
        
        # Add prompt transform if enabled
        if getattr(config.train.train_policy.dataset, "prompt_from_task", False):
            self._transforms.append(PromptFromLeRobotTask(self.dataset.meta.tasks))
        
        # Add repack transform
        self._transforms.append(RepackTransform())
        # Convert into B1K-style model inputs (ported from openpi)
        self._transforms.append(B1kInputsTransform(config.custom['action_dim'], config.train.train_policy.dataset.model_type))
        # openpi/src/openpi/transforms.py
        self._transforms.append(NormalizeTransform(self.norm_stats, use_quantiles=True))
        # openpi/src/openpi/training/config.py ModelTransformFactory.inputs
        self._transforms.append(InjectDefaultPrompt())
        self._transforms.append(ResizeImages(config.custom['image_size'][0], config.custom['image_size'][1]))
        self._transforms.append(TokenizePrompt(self.tokenizer, discrete_state_input=config.train.train_policy.dataset.discrete_state_input))
        self._transforms.append(PadStatesAndActions(config.custom['action_dim']))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        # Apply all transforms in sequence
        for transform in self._transforms:
            item = transform(item)
        return item
    
    def init_norm_stats(self):
        self.skip_norm_stats = self.config.train.train_policy.dataset.skip_norm_stats

        if self.skip_norm_stats:
            logger.info("Skipping norm stats")
            self.norm_stats = None
        else:
            norm_stats_path = self.config.train.train_policy.dataset.norm_stats
            if norm_stats_path is None:
                norm_stats_path = os.path.join(self.config.policy.model_name_or_path, "assets", self.config.train.train_policy.dataset.repo_id, "norm_stats.json")
            if os.path.exists(norm_stats_path):
                logger.info(f"Loading norm stats from {norm_stats_path}")
                self.norm_stats = load_norm_stats(norm_stats_path)
            else:
                logger.warning(f"Norm stats file not found at {norm_stats_path}")
                self.norm_stats = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    
    with open(args.config, "r") as f:
        config_dict = toml.load(f)
    config = CosmosConfig.from_dict(config_dict)
    
    dataset = BehaviorSFTDataset(config)
    launch_dispatcher(dataset=dataset)
