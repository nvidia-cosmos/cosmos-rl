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

import io
import os
import base64
import copy
import torch
import numpy as np
from PIL import Image
from typing import (
    List,
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
)  # Tuple used in dpo_collate_fn
from transformers import AutoProcessor, AutoConfig
from glob import glob

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import retry
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.data.schema import ChatMessage
from cosmos_rl.dispatcher.data.packer.base import DataPacker

from qwen_vl_utils import fetch_image, fetch_video
from qwen_vl_utils.vision_process import smart_nframes

IGNORE_LABEL_ID = -100


def fetch_video_frames(vision_info, image_patch_size, return_video_metadata):
    FRAME_FACTOR = 2
    frame_dir = vision_info.get("frame_dir", None)
    frames = sorted(
        filter(lambda x: "json" not in x, glob(os.path.join(frame_dir, "*")))
    )
    max_pixels = vision_info.get("max_pixels", 196 * (16 * 2) ** 2)
    extracted_fps = (
        vision_info.get("extracted_fps")
        if "metadata" not in vision_info
        else vision_info["metadata"].get("extracted_fps")
    )
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"No frames found in {frame_dir}")
    if total_frames < FRAME_FACTOR:
        nframes = FRAME_FACTOR
        idx = list(range(total_frames)) + [total_frames - 1] * (
            FRAME_FACTOR - total_frames
        )
        sample_fps = extracted_fps
    else:
        nframes = smart_nframes(
            vision_info, total_frames=total_frames, video_fps=extracted_fps
        )
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        sample_fps = nframes / max(total_frames, 1e-6) * extracted_fps
    sampled_frames = [
        fetch_image(
            {"image": frames[i], "max_pixels": max_pixels},
            image_patch_size=image_patch_size,
        )
        for i in idx
    ]
    video = torch.stack(
        [torch.from_numpy(np.array(img).transpose(2, 0, 1)) for img in sampled_frames]
    ).float()
    if return_video_metadata:
        video_metadata = dict(
            fps=extracted_fps,
            frames_indices=idx,
            total_num_frames=len(frames),
        )
        return (video, video_metadata), sample_fps
    return video, sample_fps


def extract_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or "frame_dir" in ele
                        or ele.get("type", "text")
                        in ("image", "image_url", "video", "video_frames")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def qwen_vl_process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[
    Optional[List[Image.Image]],
    Optional[List[Union[torch.Tensor, List[Image.Image]]]],
    Optional[Dict[str, Any]],
]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(
                fetch_image(vision_info, image_patch_size=image_patch_size)
            )
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        elif "frame_dir" in vision_info:
            video_input, video_sample_fps = fetch_video_frames(
                vision_info,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url, frame_dir or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {"do_sample_frames": False}
    if not return_video_metadata:  # BC for qwen2.5vl
        video_kwargs.update({"fps": video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs


def process_vision_info(sample: List[Dict[str, Any]]) -> Tuple[Any, Any]:
    image_inputs = []
    video_inputs = []
    for x in sample:
        if x["role"] == "user":
            for item in x["content"]:
                if item["type"] == "image":
                    image_inputs.append(item["image"])
    return image_inputs, video_inputs


def decode_base64_to_image(
    image_inputs: List[Union[str, Image.Image]],
) -> List[Union[str, Image.Image]]:
    new_image_inputs = []
    for image_input in image_inputs:
        # TODO: hardcode
        if isinstance(image_input, Image.Image):
            new_image_inputs.append(image_input)
            continue
        else:
            assert isinstance(image_input, str), (
                f"image_input should be a string, but got {type(image_input)}"
            )
            if os.path.isfile(image_input):
                continue
            else:
                img_bytes = base64.b64decode(image_input)
                img_buffer = io.BytesIO(img_bytes)
                image = Image.open(img_buffer)
                new_image_inputs.append(image)
    return new_image_inputs


def retrieve_not_none_values(input_list):
    input_list = [x for x in input_list if x is not None]
    if len(input_list) > 0:
        input_list = torch.cat(input_list, dim=0)
    else:
        input_list = None
    return input_list


class HFVLMDataPacker(DataPacker):
    """
    Data protocol & processing logic for the HF VLMs for SFT and RL training.
    """

    Payload = List[Dict[str, Any]]

    class RLPolicyInput:
        input_ids: List[int]
        logprob_masks: List[int]

        def __init__(self, input_ids: List[int], logprob_masks: List[int]):
            self.input_ids = input_ids
            self.logprob_masks = logprob_masks

    def setup(self, config: Config, *args, **kwargs):
        super().setup(config, *args, **kwargs)
        self.hf_processor = retry(AutoProcessor.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )

        hf_config = retry(AutoConfig.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )

        image_token_id = (
            getattr(hf_config, "image_token_id", None)
            or getattr(hf_config.vision_config, "image_token_id", None)
            or getattr(hf_config, "img_context_token_id", None)
        )
        if image_token_id is None:
            image_token_id = getattr(hf_config, "image_token_index", None) or getattr(
                hf_config.vision_config, "image_token_index", None
            )
        assert image_token_id is not None, f"Cannot find image token id in {hf_config=}"
        self.image_token_id = image_token_id
        self.image_token = getattr(self.hf_processor, "image_token", None) or getattr(
            hf_config, "img_context_token", None
        )

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
        self.model_type = hf_config.model_type
        self.use_qwen_vl_process = self.model_type in [
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
        ] or os.environ.get("USE_QWEN_VL_PROCESS", "0") in ["1", "true", "True"]
        self.use_siglip2_process = os.environ.get("USE_SIGLIP2_PROCESS", "0") in [
            "1",
            "true",
            "True",
        ]
        logger.info(
            f"Initialized HFVLMDataPacker with image_token_id={self.image_token_id} "
            f"and video_token_id={self.video_token_id}, model_type={self.model_type}, "
            f"use_qwen_vl_process={self.use_qwen_vl_process}, use_siglip2_process={self.use_siglip2_process}"
        )

    def get_rollout_input(self, sample: Payload) -> Any:
        """
        This VL data packer only accepts the conversation data format.
        check https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat for more details.

        example:
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        It is user's responsibility to ensure the conversation format is correct
          and multi-media files involved in conversation are accessible.
        """
        if isinstance(sample, list):
            sample = [
                x.model_dump() if isinstance(x, ChatMessage) else x for x in sample
            ]
            assert all(
                isinstance(x, dict) and "role" in x and "content" in x for x in sample
            ), "All samples should be in conversation format, but got: {}".format(
                sample
            )

        if self.image_token is not None:
            for x in sample:
                if x["role"] == "user":
                    contents = x["content"]
                    for idx, content in enumerate(contents):
                        if (
                            content["type"] == "text"
                            and self.image_token in content["text"]
                        ):
                            new_content = content.copy()
                            contents[idx]["text"] = new_content["text"].replace(
                                self.image_token, ""
                            )

        # Here we need to convert the conversation format to the format required by vllm
        prompt = self.hf_processor.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=True
        )
        video_kwargs = {}
        image_inputs, video_inputs = process_vision_info(sample)
        if (
            (self.use_qwen_vl_process or self.use_siglip2_process)
            and len(image_inputs) == 0
            and len(video_inputs) == 0
        ):
            image_inputs, video_inputs, video_kwargs = qwen_vl_process_vision_info(
                sample,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        if len(video_inputs) > 0:
            return {
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs},
                "mm_processor_kwargs": video_kwargs,
            }
        elif len(image_inputs) > 0:
            return {
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            }
        else:
            return {
                "prompt": prompt,
            }

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
                if (
                    len(token_ids) > i + pad_run_length
                    and token_ids[i + pad_run_length] == eos_token_id
                ):
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + [eos_token_id]
                        + label_ids[i + pad_run_length + 1 :]
                    )
                else:
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + label_ids[i + pad_run_length :]
                    )
                return (
                    True,
                    token_ids[:i] + replacement_ids + token_ids[i + pad_run_length :],
                    label_ids,
                )
        # no match found
        return False, token_ids, label_ids

    def _process_single_sample(
        self,
        conversation: "HFVLMDataPacker.Payload",
        add_generation_prompt: bool,
    ) -> Dict[str, Any]:
        try:
            conversation = copy.deepcopy(conversation)
            # Replace all the assistant content with consecutive `pad_token` * 10
            pad_token = self.tokenizer.pad_token
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_run_length = 10
            assistant_contents = []
            messages = None
            if "messages" in conversation:
                messages = conversation["messages"]
                for message in messages:
                    if message["role"] == "assistant":
                        content = message["content"]
                        new_content = content.copy()
                        if isinstance(new_content, str):
                            assistant_contents.append(new_content)
                            new_content = pad_token * pad_run_length
                        elif isinstance(new_content, dict):
                            assert "text" in new_content, (
                                f"text not in content: {content}"
                            )
                            assistant_contents.append(new_content["text"])
                            new_content["text"] = pad_token * pad_run_length
                        elif isinstance(content, list):
                            for i, item in enumerate(content):
                                if isinstance(item, dict):
                                    assert "text" in item, (
                                        f"text not in content: {item}"
                                    )
                                    assistant_contents.append(item["text"])
                                    new_content[i]["text"] = pad_token * pad_run_length
                                else:
                                    raise ValueError(
                                        f"Unsupported content type: {type(item)}"
                                    )
                        else:
                            raise ValueError(
                                f"Unsupported content type: {type(content)}"
                            )
                        message["content"] = new_content
            else:
                messages = conversation
                if self.image_token is not None:
                    for x in messages:
                        if x["role"] == "user":
                            contents = x["content"]
                            for idx, content in enumerate(contents):
                                if (
                                    content["type"] == "text"
                                    and self.image_token in content["text"]
                                ):
                                    new_content = content.copy()
                                    contents[idx]["text"] = new_content["text"].replace(
                                        self.image_token, ""
                                    )
                for x in messages:
                    if x["role"] == "assistant":
                        content = x["content"]
                        if isinstance(content, str):
                            assistant_contents.append(content)
                        elif isinstance(content, dict):
                            assert "text" in content, f"text not in content: {content}"
                            assistant_contents.append(content["text"])
                        elif isinstance(content, list):
                            for _, item in enumerate(content):
                                assert "text" in item, (
                                    f"text not in content of assistant: {item}"
                                )
                                assistant_contents.append(item["text"])
                        else:
                            raise ValueError(
                                f"Unsupported content type: {type(content)}"
                            )
                        x["content"] = pad_token * pad_run_length

            text = self.hf_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

            video_kwargs = {}
            image_inputs = []
            video_inputs = []
            video_metadatas = None

            kwarg = {
                "return_tensors": "pt",
                "images": image_inputs,
            }

            if (self.use_qwen_vl_process or self.use_siglip2_process) and isinstance(
                messages, list
            ):
                image_inputs, video_inputs, video_kwargs = qwen_vl_process_vision_info(
                    messages,
                    image_patch_size=16,  # TODO: hardcode
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                if video_inputs is not None:
                    video_inputs, video_metadatas = zip(*video_inputs)
                    video_inputs, video_metadatas = (
                        list(video_inputs),
                        list(video_metadatas),
                    )
                else:
                    video_metadatas = None

                kwarg["images"] = image_inputs
                kwarg["videos"] = video_inputs
                kwarg["video_metadata"] = video_metadatas
                kwarg["do_resize"] = False

                if self.use_siglip2_process:
                    # Read max_pixels from content dicts (set by modify_messages
                    # in launcher_wds.py); the processor converts to max_num_patches.
                    for msg in conversation:
                        if isinstance(msg.get("content"), list):
                            for c in msg["content"]:
                                if isinstance(c, dict) and "max_pixels" in c:
                                    kwarg["max_pixels"] = int(c["max_pixels"])
                                    break
                            if "max_pixels" in kwarg:
                                break

            inputs = self.hf_processor(
                text=[text],
                **kwarg,
                **video_kwargs,
            )
            input_ids = inputs["input_ids"][0].tolist()
            label_ids = [IGNORE_LABEL_ID] * len(input_ids)

            for assistant_content in assistant_contents:
                replacement_ids = self.tokenizer.encode(
                    assistant_content, add_special_tokens=False
                )

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
            print(f"Error processing sample: {e}, please fix to ensure SFT works")
            raise e

        result_dict = {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }
        if "pixel_values_videos" in inputs:
            result_dict["pixel_values_videos"] = inputs["pixel_values_videos"]

            if "video_grid_thw" in inputs:
                result_dict["video_grid_thw"] = inputs["video_grid_thw"]
            else:
                result_dict["video_grid_thw"] = None

            if "second_per_grid_ts" in inputs:
                result_dict["second_per_grid_ts"] = torch.tensor(
                    inputs["second_per_grid_ts"], dtype=torch.float
                )
            else:
                result_dict["second_per_grid_ts"] = None

            if "pixel_values_videos_lengths_per_sample" in inputs:
                result_dict["pixel_values_videos_lengths_per_sample"] = inputs[
                    "pixel_values_videos"
                ].shape[0]
            else:
                result_dict["pixel_values_videos_lengths_per_sample"] = None

        if "pixel_values" in inputs:
            result_dict["pixel_values"] = inputs["pixel_values"]

            if "image_grid_thw" in inputs:
                result_dict["image_grid_thw"] = inputs["image_grid_thw"]
            else:
                result_dict["image_grid_thw"] = None

            if "pixel_values_lengths_per_sample" in inputs:
                result_dict["pixel_values_lengths_per_sample"] = inputs[
                    "pixel_values"
                ].shape[0]
            else:
                result_dict["pixel_values_lengths_per_sample"] = None

        if "aspect_ratio_ids" in inputs:
            result_dict["aspect_ratio_ids"] = inputs["aspect_ratio_ids"]
        else:
            result_dict["aspect_ratio_ids"] = None

        if "aspect_ratio_mask" in inputs:
            result_dict["aspect_ratio_mask"] = inputs["aspect_ratio_mask"]
        else:
            result_dict["aspect_ratio_mask"] = None

        if "image_sizes" in inputs:
            result_dict["image_sizes"] = inputs["image_sizes"]
        else:
            result_dict["image_sizes"] = None

        if "batch_num_images" in inputs:
            result_dict["batch_num_images"] = inputs["batch_num_images"]
        else:
            result_dict["batch_num_images"] = None

        if "mm_token_type_ids" in inputs:
            result_dict["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        else:
            result_dict["mm_token_type_ids"] = None

        return result_dict

    def _collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        pixel_values_videos = [x["pixel_values_videos"] for x in processed_samples]
        video_grid_thw = [x["video_grid_thw"] for x in processed_samples]
        second_per_grid_ts = [x["second_per_grid_ts"] for x in processed_samples]
        pixel_values = [x["pixel_values"] for x in processed_samples]
        image_grid_thw = [x["image_grid_thw"] for x in processed_samples]
        pixel_values_videos_lengths_per_sample = [
            x["pixel_values_videos_lengths_per_sample"] for x in processed_samples
        ]
        pixel_values_lengths_per_sample = [
            x["pixel_values_lengths_per_sample"] for x in processed_samples
        ]
        aspect_ratio_ids = [x["aspect_ratio_ids"] for x in processed_samples]
        aspect_ratio_mask = [x["aspect_ratio_mask"] for x in processed_samples]
        image_sizes = [x["image_sizes"] for x in processed_samples]
        batch_num_images = [x["batch_num_images"] for x in processed_samples]

        pixel_values_videos = retrieve_not_none_values(pixel_values_videos)

        pixel_values_videos_lengths_per_sample = [
            x for x in pixel_values_videos_lengths_per_sample if x is not None
        ]
        pixel_values_videos_lengths_per_sample = (
            pixel_values_videos_lengths_per_sample
            if len(pixel_values_videos_lengths_per_sample) > 0
            else None
        )

        video_grid_thw = retrieve_not_none_values(video_grid_thw)

        second_per_grid_ts = retrieve_not_none_values(second_per_grid_ts)

        pixel_values = retrieve_not_none_values(pixel_values)

        pixel_values_lengths_per_sample = [
            x for x in pixel_values_lengths_per_sample if x is not None
        ]
        pixel_values_lengths_per_sample = (
            pixel_values_lengths_per_sample
            if len(pixel_values_lengths_per_sample) > 0
            else None
        )

        image_grid_thw = retrieve_not_none_values(image_grid_thw)

        aspect_ratio_ids = retrieve_not_none_values(aspect_ratio_ids)

        aspect_ratio_mask = retrieve_not_none_values(aspect_ratio_mask)

        image_sizes = retrieve_not_none_values(image_sizes)

        batch_num_images = retrieve_not_none_values(batch_num_images)

        # Shape description:
        #
        # pixel_values_[videos/images]: (BATCH_SIZE, N_PATCH, HIDDEN_SIZE)
        # [video/image]_grid_thw: (BATCH_SIZE, 3)
        # second_per_grid_ts: (BATCH_SIZE, 1)
        # pixel_values_[videos/images]_lengths_per_sample: (BATCH_SIZE, 1)
        batch = {}
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos

        if video_grid_thw is not None:
            batch["video_grid_thw"] = video_grid_thw

        if second_per_grid_ts is not None:
            batch["second_per_grid_ts"] = second_per_grid_ts

        if pixel_values_videos_lengths_per_sample is not None:
            batch["pixel_values_videos_lengths_per_sample"] = torch.tensor(
                pixel_values_videos_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)

        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw

        if pixel_values_lengths_per_sample is not None:
            batch["pixel_values_lengths_per_sample"] = torch.tensor(
                pixel_values_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)

        if aspect_ratio_ids is not None:
            batch["aspect_ratio_ids"] = aspect_ratio_ids

        if aspect_ratio_mask is not None:
            batch["aspect_ratio_mask"] = aspect_ratio_mask

        if image_sizes is not None:
            batch["image_sizes"] = image_sizes

        if batch_num_images is not None:
            batch["batch_num_images"] = batch_num_images

        # Pad the input_ids, logprob_masks
        batch["input_ids"] = torch.tensor(
            [
                x["input_ids"][:computed_max_len]
                + [self.tokenizer.pad_token_id]
                * (max(0, computed_max_len - len(x["input_ids"])))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        if "mm_token_type_ids" in processed_samples[0]:

            def _to_padded_mm_ids(x):
                ids = x.get("mm_token_type_ids")
                if ids is None:
                    ids = []
                elif isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                # Flatten 2D arrays (e.g., shape (1, seq_len)) to 1D
                if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
                    ids = (
                        [item for sublist in ids for item in sublist]
                        if len(ids) > 1
                        else list(ids[0])
                    )
                truncated = ids[:computed_max_len]
                pad_len = computed_max_len - len(truncated)
                return truncated + [0] * max(0, pad_len)

            batch["mm_token_type_ids"] = torch.tensor(
                [_to_padded_mm_ids(x) for x in processed_samples],
                dtype=torch.long,
            )

        if "label_ids" in processed_samples[0]:
            batch["label_ids"] = torch.tensor(
                [
                    x["label_ids"][:computed_max_len]
                    + [IGNORE_LABEL_ID]
                    * (max(0, computed_max_len - len(x["label_ids"])))
                    for x in processed_samples
                ],
                dtype=torch.long,
            )

        batch["logprob_masks"] = torch.tensor(
            [
                x["logprob_masks"][:computed_max_len]
                + [0] * (max(0, computed_max_len - len(x["logprob_masks"])))
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
        sample: "HFVLMDataPacker.Payload",
        rollout_output: Optional[Union[str, List[int]]] = None,
        n_ignore_prefix_tokens: int = 0,
        add_generation_prompt: bool = True,
    ) -> Any:
        if isinstance(sample, list):
            sample = [
                x.model_dump() if isinstance(x, ChatMessage) else x for x in sample
            ]
            assert all(
                isinstance(x, dict) and "role" in x and "content" in x for x in sample
            ), "All samples should be in conversation format, but got: {}".format(
                sample
            )

        x = self._process_single_sample(
            sample,
            add_generation_prompt=add_generation_prompt,
        )

        return_dict = {}
        if "pixel_values_videos" in x:
            return_dict["pixel_values_videos"] = x["pixel_values_videos"]
        else:
            return_dict["pixel_values_videos"] = None

        if "video_grid_thw" in x:
            return_dict["video_grid_thw"] = x["video_grid_thw"]
        else:
            return_dict["video_grid_thw"] = None

        if "second_per_grid_ts" in x:
            return_dict["second_per_grid_ts"] = x["second_per_grid_ts"]
        else:
            return_dict["second_per_grid_ts"] = None

        if "pixel_values_videos_lengths_per_sample" in x:
            return_dict["pixel_values_videos_lengths_per_sample"] = x[
                "pixel_values_videos_lengths_per_sample"
            ]
        else:
            return_dict["pixel_values_videos_lengths_per_sample"] = None

        if "pixel_values" in x:
            return_dict["pixel_values"] = x["pixel_values"]
        else:
            return_dict["pixel_values"] = None

        if "image_grid_thw" in x:
            return_dict["image_grid_thw"] = x["image_grid_thw"]
        else:
            return_dict["image_grid_thw"] = None

        if "pixel_values_lengths_per_sample" in x:
            return_dict["pixel_values_lengths_per_sample"] = x[
                "pixel_values_lengths_per_sample"
            ]
        else:
            return_dict["pixel_values_lengths_per_sample"] = None

        if "aspect_ratio_ids" in x:
            return_dict["aspect_ratio_ids"] = x["aspect_ratio_ids"]
        else:
            return_dict["aspect_ratio_ids"] = None

        if "aspect_ratio_mask" in x:
            return_dict["aspect_ratio_mask"] = x["aspect_ratio_mask"]
        else:
            return_dict["aspect_ratio_mask"] = None

        if "image_sizes" in x:
            return_dict["image_sizes"] = x["image_sizes"]
        else:
            return_dict["image_sizes"] = None

        if "batch_num_images" in x:
            return_dict["batch_num_images"] = x["batch_num_images"]
        else:
            return_dict["batch_num_images"] = None

        if "mm_token_type_ids" in x:
            return_dict["mm_token_type_ids"] = x["mm_token_type_ids"]
        else:
            return_dict["mm_token_type_ids"] = None

        # Common fields
        input_ids = x["input_ids"]
        completion_ids = []
        if rollout_output:
            rollout_as_token_ids = isinstance(rollout_output, list) and all(
                isinstance(i, int) for i in rollout_output
            )
            if rollout_as_token_ids:
                completion_ids = rollout_output
            else:
                completion_ids = self.tokenizer(
                    rollout_output
                ).input_ids  # Don't pad yet
        return_dict["input_ids"] = input_ids + completion_ids

        return_dict["logprob_masks"] = (
            [0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0]
        )

        return_dict["label_ids"] = x["label_ids"]
        return return_dict

    def policy_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        return max([len(x["input_ids"]) for x in processed_samples])

    def policy_collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        for x in processed_samples:
            if "label_ids" in x:
                del x["label_ids"]
        return self._collate_fn(processed_samples, computed_max_len)

    @staticmethod
    def _detect_media_types(sample: "HFVLMDataPacker.Payload") -> str:
        """Classify a conversation sample as 'text', 'image', 'video', or 'mixed'."""
        types_found: set = set()
        if isinstance(sample, list):
            for msg in sample:
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            t = c.get("type", "text")
                            if t in ("image", "video"):
                                types_found.add(t)
                elif isinstance(content, str):
                    types_found.add("text")
        if not types_found:
            return "unknown"
        if len(types_found) > 1:
            return "mixed(" + "+".join(sorted(types_found)) + ")"
        return next(iter(types_found))

    def sft_process_sample(self, sample: "HFVLMDataPacker.Payload") -> Dict[str, Any]:
        """
        Accepts either raw text or conversation format.
        """
        result = self.get_policy_input(sample, add_generation_prompt=False)

        max_len = getattr(self.config.policy, "model_max_length", None)
        if max_len is not None and len(result["input_ids"]) > max_len:
            media_type = self._detect_media_types(sample)
            has_vision = (
                result.get("pixel_values") is not None
                or result.get("pixel_values_videos") is not None
            )
            if has_vision:
                # Truncation is safe only if every vision placeholder token
                # falls within the kept prefix (indices 0..max_len-1).  If so,
                # only trailing text tokens are removed and pixel tensor
                # alignment is preserved.
                input_ids = result["input_ids"]
                vision_ids = {v for v in self.vision_ids if v is not None}
                last_vision_pos = -1
                for i, tok in enumerate(input_ids):
                    if tok in vision_ids:
                        last_vision_pos = i
                        if last_vision_pos >= max_len:
                            break
                if last_vision_pos >= max_len:
                    raise ValueError(
                        f"[{media_type}] Sample exceeds model_max_length after tokenization "
                        f"({len(result['input_ids'])} > {max_len}) and truncation would "
                        f"break vision token alignment, skipping."
                    )
            orig_len = len(result["input_ids"])
            result["input_ids"] = result["input_ids"][:max_len]
            if "label_ids" in result:
                result["label_ids"] = result["label_ids"][:max_len]
            if "logprob_masks" in result:
                result["logprob_masks"] = result["logprob_masks"][:max_len]
            if has_vision:
                logger.warning(
                    f"[{media_type}] Truncated vision sample from {orig_len} to "
                    f"{max_len} tokens (trailing text only, vision tokens preserved)."
                )
            else:
                logger.warning(
                    f"[{media_type}] Truncated sample from {orig_len} to "
                    f"{max_len} tokens."
                )

        return result

    def sft_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        return max([len(x["input_ids"]) for x in processed_samples])

    def dpo_process_sample(
        self, sample: Dict[str, "HFVLMDataPacker.Payload"]
    ) -> Dict[str, Dict[str, Any]]:
        """
        DPO: Process a sample with chosen/rejected conversation pairs.
        sample: {"chosen": messages_list, "rejected": messages_list}
        Returns: {"chosen": model_input_dict, "rejected": model_input_dict}
        """
        chosen = self._process_single_sample(
            sample["chosen"], add_generation_prompt=False
        )
        rejected = self._process_single_sample(
            sample["rejected"], add_generation_prompt=False
        )
        # DPO needs logprob_masks (1 = response tokens for log prob)
        for d in [chosen, rejected]:
            label_ids = d.get("label_ids")
            if label_ids is not None:
                if isinstance(label_ids, list):
                    d["logprob_masks"] = [
                        1 if x != IGNORE_LABEL_ID else 0 for x in label_ids
                    ]
                else:
                    d["logprob_masks"] = (label_ids != IGNORE_LABEL_ID).long().tolist()
        max_len = getattr(self.config.policy, "model_max_length", None)
        if max_len is not None:
            for d in [chosen, rejected]:
                if len(d["input_ids"]) > max_len:
                    raise ValueError(
                        f"DPO sample exceeds model_max_length ({len(d['input_ids'])} > {max_len})"
                    )
        return {"chosen": chosen, "rejected": rejected}

    def dpo_collate_fn(
        self, batch: List[Dict[str, Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Collate a list of {"chosen": dict, "rejected": dict} into batched chosen and rejected.
        """
        chosen_list = [x["chosen"] for x in batch]
        rejected_list = [x["rejected"] for x in batch]

        # _collate_fn expects all vision keys; image-only samples from _process_single_sample
        # may lack pixel_values_videos etc. Fill missing keys with None before collating.
        _OPTIONAL_COLLATE_KEYS = [
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos_lengths_per_sample",
            "pixel_values_lengths_per_sample",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "image_sizes",
            "batch_num_images",
        ]

        def _ensure_collate_keys(samples: List[Dict]) -> None:
            for s in samples:
                for k in _OPTIONAL_COLLATE_KEYS:
                    if k not in s:
                        s[k] = None

        _ensure_collate_keys(chosen_list)
        _ensure_collate_keys(rejected_list)

        computed_max_len_chosen = self.policy_compute_max_len(chosen_list)
        computed_max_len_rejected = self.policy_compute_max_len(rejected_list)
        # Use max of both for padding
        computed_max_len = max(computed_max_len_chosen, computed_max_len_rejected)
        chosen_batch = self._collate_fn(chosen_list, computed_max_len)
        rejected_batch = self._collate_fn(rejected_list, computed_max_len)
        return chosen_batch, rejected_batch

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        computed_max_len: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        # Reuse the RL collate minibatch function
        model_inputs: Dict[str, Any] = self._collate_fn(
            processed_samples, computed_max_len
        )
        del model_inputs["logprob_masks"]
        # Mask the loss on vision padding tokens
        if self.vision_ids is not None:
            assert isinstance(self.vision_ids, list)
            for vision_id in self.vision_ids:
                if vision_id is not None:
                    model_inputs["label_ids"][
                        model_inputs["label_ids"] == vision_id
                    ] = ignore_label_id

        return model_inputs
