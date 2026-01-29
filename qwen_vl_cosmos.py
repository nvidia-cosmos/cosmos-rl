# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import logging
logging.getLogger("cosmos").propagate = False
import importlib
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig

from imaginaire.lazy_config import instantiate
# from imaginaire.utils import distributed, log, misc
from imaginaire.utils.config_helper import get_config_module, override
from projects.cosmos3.vlm.processors import build_processor
from projects.cosmos3.vlm.train import init_cosmos_rl_model
from projects.cosmos3.vlm.trainer.sft_trainer_cosmos_rl import init_optimizer_scheduler
from projects.cosmos3.vlm.utils.broadcast_databatch import broadcast_databatch
from projects.cosmos3.vlm.utils.create_position_ids import get_position_ids

        # # Instantiate i4 model
        # config_file = "projects/cosmos3/vlm/configs/base/config.py"
        # config_module = get_config_module(config_file)
        # config = importlib.import_module(config_module).make_config()
        # experiment = "pre_exp020_000_qwen3_vl_30b_a3b_thinking"
        # config = override(
        #     config,
        #     [
        #         "--",
        #         f"experiment={experiment}",
        #         # "data_train=09_eagle_sft_full_mul_repeat_debug_s3",
        #         # "data_train=debug_image_data_qwen",
        #     ],
        # )
        # self.dataloader = instantiate(config.dataloader_train)

"""
Usage:
PYTHONPATH=.:cosmos-rl torchrun --nproc_per_node=8 projects/cosmos3/vlm/scripts/compute_flop_qwen3/profile_qwen3vl.py --skip_train_loop=1 --experiment=pre_exp020_010_qwen3_vl_30b_a3b_thinking_flop3s
"""
    
# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import copy
from typing import Any, Dict, List, Optional

import torch
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import retry
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoProcessor, AutoTokenizer


from imaginaire.datasets.webdataset.dataloader import DataLoader as WebDataLoader
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import instantiate
from projects.cosmos3.vlm.datasets.augmentors.nvlm_data_to_conversation import NVLMImageDataConversation
from projects.cosmos3.vlm.datasets.augmentors.nvlm_data_unify import NVLMImageDataUnify
from projects.cosmos3.vlm.datasets.data_sources.nvlm import get_data_key, get_data_weight_dict
from projects.cosmos3.vlm.datasets.dataset_provider_nvlm import get_nvlm_dataset
import multiprocessing as mp


MAX_PIXELS = 81920
IGNORE_LABEL_ID = -100


class EagleVLMDataPacker(DataPacker):
    """
    Data protocol & processing logic for EagleVLM SFT.
    """

    Payload = List[Dict[str, Any]]

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        super().setup(config, tokenizer, *args, **kwargs)
        self.hf_processor = retry(AutoProcessor.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )

        hf_config = retry(AutoConfig.from_pretrained)(config.policy.model_name_or_path, trust_remote_code=True)

        image_token_id = getattr(hf_config, "image_token_id", None) or getattr(
            hf_config.vision_config, "image_token_id", None
        )
        if image_token_id is None:
            image_token_id = getattr(hf_config, "image_token_index", None) or getattr(
                hf_config.vision_config, "image_token_index", None
            )
        assert image_token_id is not None, f"Cannot find image token id in {hf_config=}"
        self.image_token_id = image_token_id
        self.image_token = getattr(self.hf_processor, "image_token", None)

        video_token_id = getattr(hf_config, "video_token_id", None) or getattr(
            hf_config.vision_config, "video_token_id", None
        )
        if video_token_id is None:
            video_token_id = getattr(hf_config, "video_token_index", None) or getattr(
                hf_config.vision_config, "video_token_index", None
            )
        if video_token_id is None:
            self.video_token = None
            self.video_token_id = None
        else:
            self.video_token = self.tokenizer.decode([video_token_id])
            self.video_token_id = video_token_id
        self.vision_ids = [self.image_token_id, self.video_token_id]
        self.hf_config = hf_config

    def get_rollout_input(self, sample: Payload) -> Any:
        return sample

    def _replace_assistant_content(
        self,
        token_ids: List[int],
        label_ids: List[int],
        pad_token_id: int,
        eos_token_id: int,
        replacement_ids: List[int],
        pad_run_length: int = 10,
    ) -> List[int]:
        """
        Find the first run of exactly `pad_run_length` pad_token_id's in token_ids,
        replace that run with replacement_ids, and return the new list.
        If no such run is found, returns the original list unchanged.
        """
        n = len(token_ids)
        target_run = [pad_token_id] * pad_run_length

        # find the start index of the first matching run
        for i in range(n - pad_run_length + 1):
            if token_ids[i : i + pad_run_length] == target_run:
                # splice in the replacement
                if len(token_ids) > i + pad_run_length and token_ids[i + pad_run_length] == eos_token_id:
                    label_ids = label_ids[:i] + replacement_ids + [eos_token_id] + label_ids[i + pad_run_length + 1 :]
                else:
                    label_ids = label_ids[:i] + replacement_ids + label_ids[i + pad_run_length :]
                return (
                    True,
                    token_ids[:i] + replacement_ids + token_ids[i + pad_run_length :],
                    label_ids,
                )
        # no match found
        return False, token_ids, label_ids

    def _process_single_sample(
        self,
        conversation: "EagleVLMDataPacker.Payload",
        add_generation_prompt: bool,
    ) -> Dict[str, Any]:
        try:
            # Replace all the assistant content with consecutive `pad_token` * 10
            pad_token = self.tokenizer.pad_token
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_run_length = 10
            assistant_content = []
            assert "messages" in conversation, f"messages not in conversation: {conversation}"
            assert "images" in conversation, f"images not in conversation: {conversation}"
            messages = conversation["messages"]
            image_inputs = conversation["images"]
            for message in messages:
                if message["role"] == "assistant":
                    content = message["content"]
                    new_content = copy.deepcopy(content)
                    if isinstance(new_content, str):
                        assistant_content.append(new_content)
                        new_content = pad_token * pad_run_length
                    elif isinstance(new_content, dict):
                        assert "text" in new_content, f"text not in content: {content}"
                        assistant_content.append(new_content["text"])
                        new_content["text"] = pad_token * pad_run_length
                    elif isinstance(content, list):
                        for i, item in enumerate(content):
                            if isinstance(item, dict):
                                assert "text" in item, f"text not in content: {item}"
                                assistant_content.append(item["text"])
                                new_content[i]["text"] = pad_token * pad_run_length
                            else:
                                raise ValueError(f"Unsupported content type: {type(item)}")
                    else:
                        raise ValueError(f"Unsupported content type: {type(content)}")
                    message["content"] = new_content
                elif message["role"] == "user":
                    content = message["content"]
                    new_content = copy.deepcopy(content)
                    if isinstance(content, list):
                        for i, item in enumerate(content):
                            if isinstance(item, dict):
                                if "type" in item and item["type"] == "image":
                                    new_content[i]["image"] = image_inputs[0]
                                # elif "video" in item:
                                #     new_content[i]["video"] = video_inputs
                                else:
                                    continue
                            else:
                                raise ValueError(f"Unsupported content type: {type(item)}")
                    message["content"] = new_content

            text = self.hf_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

            kwarg = {
                "return_tensors": "pt",
                "images": image_inputs,
            }

            image_inputs, video_inputs = self.hf_processor.process_vision_info(messages)
            kwarg["images"] = image_inputs
            kwarg["videos"] = video_inputs
            inputs = self.hf_processor(
                text=[text],
                **kwarg,
            )

            input_ids = inputs["input_ids"][0].tolist()
            label_ids = [IGNORE_LABEL_ID] * len(input_ids)

            for assistant_content in assistant_content:
                replacement_ids = self.tokenizer.encode(assistant_content, add_special_tokens=False)

                replaced, input_ids, label_ids = self._replace_assistant_content(
                    input_ids,
                    label_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    replacement_ids=replacement_ids,
                    pad_run_length=pad_run_length,
                )
                if not replaced:
                    raise ValueError("No assistant content to replace")
                if len(input_ids) != len(label_ids):
                    raise ValueError(
                        f"input_ids and label_ids should have the same length, but got {len(input_ids)} and {len(label_ids)}"
                    )
        except Exception as e:
            logger.error(f"Error processing sample: {e}, please fix to ensure SFT works")
            raise e

        result_dict = {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }

        result_dict["pixel_values"] = inputs["pixel_values"] if "pixel_values" in inputs else None
        result_dict["image_sizes"] = inputs["image_sizes"] if "image_sizes" in inputs else None

        return result_dict

    def _collate_fn(self, processed_samples: List[Dict[str, Any]], computed_max_len: int) -> Dict[str, Any]:
        pixel_values = [x["pixel_values"] for x in processed_samples]
        image_sizes = [x["image_sizes"] for x in processed_samples]

        if all([x is not None for x in pixel_values]):
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            assert all([x is None for x in pixel_values]), "pixel_values should be None"
            pixel_values = None

        if all([x is not None for x in image_sizes]):
            image_sizes = torch.cat(image_sizes, dim=0)
        else:
            assert all([x is None for x in image_sizes]), "image_sizes should be None"
            image_sizes = None

        batch = {}
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        if image_sizes is not None:
            batch["image_sizes"] = image_sizes

        # Pad the input_ids, logprob_masks
        batch["input_ids"] = torch.tensor(
            [
                x["input_ids"][:computed_max_len]
                + [self.tokenizer.pad_token_id] * (max(0, computed_max_len - len(x["input_ids"])))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        if "label_ids" in processed_samples[0]:
            batch["label_ids"] = torch.tensor(
                [
                    x["label_ids"][:computed_max_len]
                    + [IGNORE_LABEL_ID] * (max(0, computed_max_len - len(x["label_ids"])))
                    for x in processed_samples
                ],
                dtype=torch.long,
            )
        batch["logprob_masks"] = torch.tensor(
            [
                x["logprob_masks"][:computed_max_len] + [0] * (max(0, computed_max_len - len(x["logprob_masks"])))
                for x in processed_samples
            ],
            dtype=torch.bool,
        )

        assert len(batch["input_ids"]) == len(batch["logprob_masks"]), (
            "The length of input_ids, logprob_masks should be the same"
        )

        return batch

    def get_policy_input(
        self,
        sample: "EagleVLMDataPacker.Payload",
        rollout_output: Optional[str] = None,
        n_ignore_prefix_tokens: int = 0,
        add_generation_prompt: bool = True,
    ) -> Any:
        # assert all(
        #     isinstance(x, dict) and "role" in x and "content" in x for x in sample
        # ), "All samples should be in conversation format, but got: {}".format(sample)
        x = self._process_single_sample(
            sample,
            add_generation_prompt=add_generation_prompt,
        )

        return_dict = {}
        return_dict["pixel_values"] = x["pixel_values"] if "pixel_values" in x else None
        return_dict["image_sizes"] = x["image_sizes"] if "image_sizes" in x else None

        # Common fields
        input_ids = x["input_ids"]

        return_dict["input_ids"] = input_ids

        return_dict["logprob_masks"] = (
            [0] * (len(input_ids) - 1 + n_ignore_prefix_tokens) + [1] * (-n_ignore_prefix_tokens) + [0]
        )

        return_dict["label_ids"] = x["label_ids"]
        return return_dict

    def policy_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        return max([len(x["input_ids"]) for x in processed_samples])

    def policy_collate_fn(self, processed_samples: List[Dict[str, Any]], computed_max_len: int) -> Dict[str, Any]:
        for x in processed_samples:
            if "label_ids" in x:
                del x["label_ids"]
        return self._collate_fn(processed_samples, computed_max_len)

    def sft_process_sample(self, sample: "EagleVLMDataPacker.Payload") -> Dict[str, Any]:
        """
        Accepts either raw text or conversation format.
        """
        return self.get_policy_input(sample, add_generation_prompt=False)

    def sft_compute_max_len(self, processed_samples: Dict[str, Any]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        return processed_samples['input_ids'].shape[1]

    def sft_collate_fn(
        self,
        processed_samples: Dict[str, Any],
        computed_max_len: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        # Reuse the RL collate minibatch function
        # model_inputs: Dict[str, Any] = self._collate_fn(processed_samples, computed_max_len)
        # del model_inputs["logprob_masks"]
        # # Mask the loss on vision padding tokens
        # if self.vision_ids is not None:
        #     assert isinstance(self.vision_ids, list)
        #     for vision_id in self.vision_ids:
        #         if vision_id is not None:
        #             model_inputs["label_ids"][model_inputs["label_ids"] == vision_id] = ignore_label_id
        model_inputs: Dict[str, Any] = processed_samples
        model_inputs["label_ids"] = processed_samples["labels"]
        return model_inputs

    def batch_size(self, batch: Dict[str, Any]) -> int:
        return batch["input_ids"].size(0)
    
    def slice_batch(
        self,
        batch: Dict[str, Any],
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        sliced_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                # logger.info(f"====== Slicing tensor/ndarray for key: {key}, shape: {value.shape}")
                sliced_batch[key] = value[start:end]
            elif isinstance(value, list):
                # logger.info(f"====== Slicing list for key: {key}, length: {len(value)}")
                sliced_batch[key] = value[start:end]
            else:
                # logger.info(f"====== Keeping original value for key: {key}, type: {type(value)}")
                sliced_batch[key] = value
        return sliced_batch


class CustomSFTDataPacker(EagleVLMDataPacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.IGNORE_LABEL_ID = IGNORE_LABEL_ID


class CosmosSFTDataset(Dataset):    
    def setup(self, config: Config, tokenizer: AutoTokenizer=None, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # Instantiate i4 model
        config_file = "projects/cosmos3/vlm/configs/base/config.py"
        config_module = get_config_module(config_file)
        config = importlib.import_module(config_module).make_config()
        experiment = "pre_exp020_000_qwen3_vl_30b_a3b_thinking"
        config = override(
            config,
            [
                "--",
                f"experiment={experiment}",
                # "data_train=09_eagle_sft_full_mul_repeat_debug_s3",
                # "data_train=debug_image_data_qwen",
            ],
        )
        self.dataloader = instantiate(config.dataloader_train)
        self.iterator = iter(self.dataloader)
        self.data_loader = self.dataloader

    def __len__(self):
        return len(self.dataloader)



if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True) 
    def get_dataset(config: CosmosConfig) -> Dataset:
        return CosmosSFTDataset()

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
        data_packer=CustomSFTDataPacker(),
    )