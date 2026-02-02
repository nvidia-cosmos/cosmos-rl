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

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
import numpy as np

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.replay_buffer import load_trajectory_from_buffer

class RLPolicyInput:
    def __init__(
        self,
        weight_version,
        task_id,
        trial_id,
        finish_step,
        complete,
        advantage,
        chains,
        denoise_inds,
        images,
        image_masks,
        states,
        tokenized_prompt,
        tokenized_prompt_mask,
        old_log_probs,
    ):
        self.weight_version = weight_version
        self.task_id = task_id
        self.trial_id = trial_id
        self.finish_step = finish_step
        self.complete = complete
        self.advantage = advantage
        self.trajectory_kind = "pi05"
        self.chains = chains
        self.denoise_inds = denoise_inds
        self.images = images
        self.image_masks = image_masks
        self.states = states
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.old_log_probs = old_log_probs

class Observation:
    """Observation class matching OpenPI's Observation structure."""
    
    def __init__(self, images, image_masks, state, tokenized_prompt, tokenized_prompt_mask, token_ar_mask=None, token_loss_mask=None):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Create Observation from dict, matching OpenPI's Observation.from_dict()."""
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = (
                    data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                )
        
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

class PI05DataPacker(BaseDataPacker):
    """Data packer for PI0/PI05 models."""

    def sft_compute_max_len(self, processed_samples):
        return 512

    def sft_process_sample(self, sample):
        return sample

    def get_rollout_input(self, item: Any) -> RLPayload:
        if isinstance(item, RLPayload): # TODO: check if this is needed
            return item

        if not isinstance(item, dict):
            raise ValueError(f"PI05 data packer expects dict or RLPayload, got {type(item)}")

        metadata = {}
        metadata["task_suite_name"] = item.get("task_suite_name", "libero_10")
        metadata["task_id"] = item.get("task_id", 0)
        metadata["trial_id"] = item.get("trial_id", 0)
        metadata["trial_seed"] = item.get("trial_seed", -1)

        for key in ["task_id", "trial_id", "trial_seed"]:
            if hasattr(metadata[key], "item"):
                metadata[key] = metadata[key].item()

        if "instruction" in item:
            metadata["instruction"] = item["instruction"]
        if "max_steps" in item:
            metadata["max_steps"] = item["max_steps"]

        for key, value in item.items():
            if key not in metadata and key not in [
                "task_suite_name",
                "task_id",
                "trial_id",
                "trial_seed",
            ]:
                if not isinstance(value, (torch.Tensor, list, dict)):
                    metadata[key] = value

        payload = RLPayload(
            prompt="",
            metadata=metadata,
            weight_version=0,
        )
        logger.debug(f"PI05 data packer created payload with metadata: {metadata}")
        return payload

    def get_policy_input(self, sample: Rollout, device: torch.device) -> Any:
        weight_version = sample.weight_version
        prompt = sample.prompt
        task_id = prompt["task_id"]
        trial_id = prompt["trial_id"]
        advantage = sample.advantage
        finish_step = sample.completion["finish_step"]
        complete = sample.completion["complete"]

        trajectory_id = sample.completion["trajectory_id"]
        try:
            trajectory = load_trajectory_from_buffer(
                trajectory_id,
                buffer_dir=os.path.join(self.config.train.output_dir, "replay_buffer"),
                remove_after_load=False,
            )
        except Exception as e:
            logger.error(
                f"[PI05 Policy Input] Failed to load trajectory {trajectory_id}: {e}"
            )
            raise
        return RLPolicyInput(
            weight_version,
            task_id,
            trial_id,
            finish_step,
            complete,
            advantage,
            chains=trajectory["chains"].to(device),
            denoise_inds=trajectory["denoise_inds"].to(device),
            images=trajectory["images"].to(device),
            image_masks=trajectory["image_masks"].to(device),
            states=trajectory["states"].to(device),
            tokenized_prompt=trajectory["input_ids"].to(device),
            tokenized_prompt_mask=trajectory["attention_mask"].to(device),
            old_log_probs=trajectory["old_log_probs"].to(device),
        )
    def policy_compute_max_len(self, *args, **kwargs):
        return 512

    def policy_collate_fn(self, policy_input: Any, max_chunks: int) -> Dict[str, Any]:
        chains = policy_input.chains
        denoise_inds = policy_input.denoise_inds
        images = policy_input.images
        image_masks = policy_input.image_masks
        states = policy_input.states
        tokenized_prompt = policy_input.tokenized_prompt
        tokenized_prompt_mask = policy_input.tokenized_prompt_mask
        old_log_probs = policy_input.old_log_probs
        finish_step = int(policy_input.finish_step)

        num_chunks = int(chains.shape[0])
        pad_chunks = int(max_chunks - num_chunks)

        chains = torch.cat(
            (
                chains,
                torch.zeros(
                    (pad_chunks, *chains.shape[1:]),
                    dtype=chains.dtype,
                    device=chains.device,
                ),
            ),
            dim=0,
        )
        denoise_inds = torch.cat(
            (
                denoise_inds,
                torch.zeros(
                    (pad_chunks, *denoise_inds.shape[1:]),
                    dtype=denoise_inds.dtype,
                    device=denoise_inds.device,
                ),
            ),
            dim=0,
        )
        images = torch.cat(
            (
                images,
                torch.zeros(
                    (pad_chunks, *images.shape[1:]),
                    dtype=images.dtype,
                    device=images.device,
                ),
            ),
            dim=0,
        )
        image_masks = torch.cat(
            (
                image_masks,
                torch.zeros(
                    (pad_chunks, *image_masks.shape[1:]),
                    dtype=image_masks.dtype,
                    device=image_masks.device,
                ),
            ),
            dim=0,
        )
        states = torch.cat(
            (
                states,
                torch.zeros(
                    (pad_chunks, *states.shape[1:]),
                    dtype=states.dtype,
                    device=states.device,
                ),
            ),
            dim=0,
        )
        tokenized_prompt = torch.cat(
            (
                tokenized_prompt,
                torch.zeros(
                    (pad_chunks, *tokenized_prompt.shape[1:]),
                    dtype=tokenized_prompt.dtype,
                    device=tokenized_prompt.device,
                ),
            ),
            dim=0,
        )
        tokenized_prompt_mask = torch.cat(
            (
                tokenized_prompt_mask,
                torch.zeros(
                    (pad_chunks, *tokenized_prompt_mask.shape[1:]),
                    dtype=tokenized_prompt_mask.dtype,
                    device=tokenized_prompt_mask.device,
                ),
            ),
            dim=0,
        )
        old_log_probs = torch.cat(
            (
                old_log_probs,
                torch.zeros(
                    (pad_chunks, *old_log_probs.shape[1:]),
                    dtype=old_log_probs.dtype,
                    device=old_log_probs.device,
                ),
            ),
            dim=0,
        )

        action_chunk = int(old_log_probs.shape[1])
        action_env_dim = int(old_log_probs.shape[2])
        max_steps = int(max_chunks * action_chunk)
        valid_steps = int(max(0, min(finish_step, max_steps)))
        pad_steps = int(max_steps - valid_steps)

        logprob_masks = torch.cat(
            (
                torch.ones(
                    (valid_steps, action_env_dim),
                    dtype=torch.float32,
                    device=chains.device,
                ),
                torch.zeros(
                    (pad_steps, action_env_dim),
                    dtype=torch.float32,
                    device=chains.device,
                ),
            ),
            dim=0,
        ).reshape(max_chunks, action_chunk * action_env_dim)

        return {
            "chains": chains,
            "denoise_inds": denoise_inds,
            "images": images,
            "image_masks": image_masks,
            "states": states,
            "input_ids": tokenized_prompt,
            "attention_mask": tokenized_prompt_mask,
            "old_log_probs": old_log_probs,
            "logprob_masks": logprob_masks,
        }

    def sft_collate_fn(self, batch, *args, **kwargs) -> Dict[str, Any]:
        """
        Collate samples into batched format, matching OpenPI's data loader behavior.
        """
        # Stack each field across the batch
        image_keys = list(batch[0]["image"].keys())
        
        images = {
            key: torch.from_numpy(np.stack([s["image"][key] for s in batch]))
            for key in image_keys
        }
        image_masks = {
            key: torch.from_numpy(
                np.asarray([bool(s["image_mask"][key]) for s in batch], dtype=np.bool_)
            )
            for key in image_keys
        }
        
        state = torch.from_numpy(np.stack([s["state"] for s in batch]))
        tokenized_prompt = torch.from_numpy(np.stack([s["tokenized_prompt"] for s in batch]))
        tokenized_prompt_mask = torch.from_numpy(np.stack([s["tokenized_prompt_mask"] for s in batch]))
        actions = torch.from_numpy(np.stack([s["actions"] for s in batch]))


        observation = Observation.from_dict(
            {
                "image": images,
                "image_mask": image_masks,
                "state": state,
                "tokenized_prompt": tokenized_prompt,
                "tokenized_prompt_mask": tokenized_prompt_mask,
            }
        )

        return {"observation": observation, "actions": actions}


