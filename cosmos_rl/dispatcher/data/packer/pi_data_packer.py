import torch
import numpy as np
from typing import Dict, List, Any
from cosmos_rl.dispatcher.data.packer import BaseDataPacker


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

class PIDataPacker(BaseDataPacker):
    """Data packer for PI0/PI05 models."""

    def sft_compute_max_len(self, processed_samples):
        return 512

    def sft_process_sample(self, sample):
        return sample
    
    def get_policy_input(self, *args, **kwargs):
        pass

    def get_rollout_input(self, *args, **kwargs):
        pass

    def policy_compute_max_len(self, *args, **kwargs):
        pass

    def policy_collate_fn(self, *args, **kwargs):
        pass

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


