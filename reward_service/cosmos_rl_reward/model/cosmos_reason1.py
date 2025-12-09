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

import time
import torch
import numpy as np
from cosmos_rl_reward.utils.logging import logger
from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from typing import Dict, Any
import qwen_vl_utils
from PIL import Image as PILImage
from torchvision import transforms
from qwen_vl_utils.vision_process import smart_resize
from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor
from cosmos_rl_reward.handler.registry import RewardRegistry

# System prompt used during training
SYSTEM_PROMPT = """You are a helpful video analyzer. The goal is to identify artifacts and anomalies in the video. Watch carefully and focus on the following aspects:

* Gravity (e.g. a ball cannot fly in the air)
* Collision (e.g. two objects cannot penetrate each other)
* Object interaction (e.g. an object cannot move without any apparent reason)
* Fluid dynamics (e.g. a liquid cannot flow through a solid object)
* Object permanence (e.g. an object cannot suddenly appear, disappear or change its shape)
* Common sense (e.g. an object should be functional and useful)
* Cause-and-effect (e.g. a door cannot open without any apparent reason)
* Human motion (e.g. a person's body cannot morph and the joints cannot move in impossible ways)

Here are some examples of non-artifacts you should not include in your analysis:

* Being an animated video, such as a cartoon, does not automatically make it artifacts.
* The video has no sound. Do not make any conclusions based on sound.
* Ignore any lighting, shadows, blurring, and camera effects.
* Avoid judging based on overall impression, artistic style, or background elements.

Begin your response with a single word: "Yes" or "No"."""

USER_PROMPT = "Does the video contain any anomalies or artifacts?"


@RewardRegistry.register()
class CosmosReason1Reward(BaseRewardHandler):
    """
    CosmosReason1Reward is a reward handler for evaluating video quality based on the Cosmos-Reason1-7B-Reward model.
    It processes video inputs, reformats videos, prepares them for the model, and computes reward scores based on the model's inference.
    """

    reward_name = "cosmos_reason1"

    def __init__(
        self,
        model_path="nvidia/Cosmos-Reason1-7B-Reward",
        dtype=torch.float16,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.dtype = dtype
        self.device = device

    def set_up(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
            use_cache=False,
        )
        self.model.eval()

    def _prepare_batch(
        self,
        video_input,
        video_info,
        prompt,
    ):
        # Process video input
        video_frames, video_fps = self._process_video_inputs(video_input, video_info)
        # Prepare message format (same as training)
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_frames, "fps": video_fps},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        ]
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages[0], tokenize=False, add_generation_prompt=True
        )
        logger.debug(f"Processed text input: {text}")
        # Process inputs
        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
            messages, return_video_kwargs=True
        )

        logger.debug(
            f"[{self.reward_name}] Video inputs: {len(video_inputs)}, Video kwargs: {video_kwargs}"
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        # Move to device
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        return inputs

    def _process_video_inputs(
        self,
        video: torch.Tensor,
        video_info: Dict[str, Any],
        temporal_patch_size: int = 2,
        target_num_tokens: int = 9216,
        frame_count_range: tuple[int, int] = (40, 160),
    ) -> tuple[list[PILImage.Image], float]:
        """Decode video file for inference, similar to training preprocessing."""
        video_fps = video_info["video_fps"]
        patch_size = 14
        # min_height_width = 56
        min_pixels = 16 * (patch_size * 2) ** 2 * temporal_patch_size
        max_pixels = target_num_tokens * (patch_size * 2) ** 2 * temporal_patch_size
        # change CTHW to THWC
        video = video.permute(1, 2, 3, 0)
        logger.debug(
            f"[{self.reward_name}] Decoded video: shape: {video.shape}, dtype: {video.dtype}, max: {video.max()}, min: {video.min()}"
        )
        total_frames = video.shape[0]
        # Calculate downsampling interval to get frames within target range
        min_frames_range, max_frames_range = frame_count_range
        interval = max(1, (total_frames - 1) // max_frames_range + 1)
        if interval != 1:
            logger.debug(
                f"[{self.reward_name}] Video downsampled from {total_frames} to {total_frames // interval} frames"
            )

        # Downsample frames
        idx = np.arange(0, total_frames, interval)
        video_frames = video[idx, ...]
        nframes = len(idx)
        sample_fps = video_fps / interval

        # Make frame count divisible by temporal_patch_size
        frames_to_remove = nframes % temporal_patch_size
        if frames_to_remove != 0:
            video_frames = video_frames[:-frames_to_remove, ...]

        # Resize frames
        nframes, height, width, _ = video_frames.shape
        max_pixels_per_frame = max_pixels // nframes
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,  # Ensure compatibility with patch size of 14
            min_pixels=min_pixels,
            max_pixels=max_pixels_per_frame,
        )
        to_pil = transforms.ToPILImage()
        list_of_pil_images = [
            to_pil(x.permute(2, 0, 1)).resize(
                (resized_width, resized_height), resample=PILImage.Resampling.BICUBIC
            )
            for x in video_frames
        ]
        logger.debug(
            f"[{self.reward_name}] Processed video frames: {len(list_of_pil_images)} frames, resized to {resized_width}x{resized_height}"
        )
        return list_of_pil_images, sample_fps

    def _reward(
        self,
        video_inputs,
        video_infos,
        prompts,
    ):
        assert len(video_infos) == 1, "Only one video input is supported."
        assert len(prompts) == 1, "Only one prompt is supported."
        video_input = (
            video_inputs.squeeze(0) if len(video_inputs.shape) == 5 else video_inputs
        )
        prompt = prompts[0]
        video_info = video_infos[0]
        inputs = self._prepare_batch(video_input, video_info, prompt)
        # Generate prediction
        with torch.no_grad():
            # Get logits for next token prediction
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get tokens for "Yes" and "No"
            yes_token_id = self.processor.tokenizer.encode(
                "Yes", add_special_tokens=False
            )[0]
            no_token_id = self.processor.tokenizer.encode(
                "No", add_special_tokens=False
            )[0]
            # Compare logits for Yes vs No
            yes_logit = logits[yes_token_id].item()
            no_logit = logits[no_token_id].item()
            # Compute softmax score - probability of "No" (video has no anomalies)
            no_score = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[
                1
            ].item()
            prediction = "Bad" if yes_logit > no_logit else "Good"
            return prediction, no_score, yes_logit, no_logit

    def calculate_reward(self, images, metadata):
        st = time.time()
        batch_size = 1
        prompts = metadata["prompts"]
        images_batched = torch.chunk(
            images, int(np.ceil(len(images) / batch_size)), dim=0
        )
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))
        all_prediction = []
        all_no_score = []
        all_yes_logit = []
        all_no_logit = []
        if "video_infos" not in metadata:
            infos = [{}] * len(images)
        else:
            infos = metadata["video_infos"]
        info_batched = np.array_split(infos, np.ceil(len(infos) / batch_size))

        if "input_info" not in metadata:
            metadata["input_info"] = {}
        metadata["input_info"].update(
            {
                "video_infos": metadata.get("video_infos", []),
            }
        )
        for image_batch, prompts_batch, info_batch in zip(
            images_batched, prompts_batched, info_batched
        ):
            logger.debug(f"Encoded image latent shape: {image_batch.shape}")
            try:
                prediction, no_score, yes_logit, no_logit = self._reward(
                    image_batch, info_batch, prompts_batch
                )
                all_prediction.append(prediction)
                all_no_score.append(no_score)
                all_yes_logit.append(yes_logit)
                all_no_logit.append(no_logit)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                all_prediction.append("Error")
                all_no_score.append(-1.0)
                all_yes_logit.append(-1.0)
                all_no_logit.append(-1.0)
        scores = {
            "prediction": all_prediction,
            "no_score": all_no_score,
            "yes_logit": all_yes_logit,
            "no_logit": all_no_logit,
        }
        duration = time.time() - st
        logger.info(
            f"[{self.reward_name}] Scores calculated {scores} in {duration:.2f} seconds"
        )
        return {
            "scores": scores,
            "input_info": metadata.get("input_info", {}),
            "duration": f"{duration:.2f}",
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }
