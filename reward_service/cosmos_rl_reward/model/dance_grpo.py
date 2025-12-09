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
import os
import torch
from fastvideo.models.videoalign.inference import VideoVLMRewardInference
from fastvideo.models.videoalign.prompt_template import build_prompt
from fastvideo.models.videoalign.vision_process import (
    extract_vision_info,
    IMAGE_FACTOR,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    VIDEO_TOTAL_PIXELS,
    FRAME_FACTOR,
    smart_resize,
)
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from cosmos_rl_reward.utils.logging import logger
from fastvideo.models.videoalign.vision_process import smart_nframes
from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry


@RewardRegistry.register()
class DanceGRPOVideoReward(BaseRewardHandler):
    """
    DanceGRPOVideoReward is a reward handler for evaluating rewards of VQ, MQ, TA adopted from DanceGRPO.
    It processes video inputs, reformat videos, extracts features, and computes reward scores based on the model's inference.
    """

    reward_name = "dance_grpo"

    def __init__(
        self,
        model_path="/workspace/VideoReward",
        dtype=torch.float16,
        device="cuda",
        download_path="",
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.download_path = download_path

    def set_up(self):
        # Inside fastvideo, a "./Qwen2-VL-2B-Instruct" folder is expected.
        source = os.path.join(self.download_path, "Qwen2-VL-2B-Instruct")
        assert os.path.exists(source), f"Source path does not exist: {source}"
        target = "./Qwen2-VL-2B-Instruct"
        if not os.path.exists(target):
            os.symlink(source, target)
        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained=self.model_path,
            dtype=self.dtype,
            device=self.device,
        )

    def _process_video_inputs(self, chat_data, video, video_infos):
        # import pdb; pdb.set_trace()
        vision_infos = extract_vision_info(chat_data)
        ## Read images or videos
        video_inputs = []
        idx = 0
        for vision_info in vision_infos:
            if "video" in vision_info:
                video_inputs.append(
                    self._process_video(video, video_infos[idx], vision_info)
                )
            else:
                raise ValueError("image, image_url or video should in content.")
            idx += 1
        if len(video_inputs) == 0:
            video_inputs = None
        return video_inputs

    def _process_video(self, video, video_info, ele, image_factor: int = IMAGE_FACTOR):
        # video_reader_backend = get_video_reader_backend()
        # video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        # if video.shape[0] % 2 != 0:
        #     # Ensure even number of frames otherwise the model will crash
        #     video = torch.cat([video, video[-1:]], dim=0)  # Ensure even number of frames
        total_frames, video_fps = video.size(0), video_info["video_fps"]
        if ele["sample_type"] == "uniform":
            nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
            idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        elif ele["sample_type"] == "multi_pts":
            frames_each_pts = 6
            num_pts = 4
            fps = 8
            nframes = int(total_frames * fps // video_fps)
            frames_idx = (
                torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            )

            start_pt = int(frames_each_pts // 2)
            end_pt = int(nframes - frames_each_pts // 2 - 1)
            pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
            idx = []
            for pt in pts:
                idx.extend(
                    frames_idx[pt - frames_each_pts // 2 : pt + frames_each_pts // 2]
                )
        video = video[idx]

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video

    def _prepare_batch(
        self,
        video_inputs,
        video_infos,
        prompts,
        fps=None,
        num_frames=None,
        max_pixels=None,
    ):
        fps = self.inferencer.data_config.fps if fps is None else fps
        num_frames = (
            self.inferencer.data_config.num_frames if num_frames is None else num_frames
        )
        max_pixels = (
            self.inferencer.data_config.max_frame_pixels
            if max_pixels is None
            else max_pixels
        )

        video_path = "video_path"
        video_path = "/work/VideoAlign/datasets/train/videos/example_1_A.mp4"
        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}",
                                "max_pixels": max_pixels,
                                "fps": fps,
                                "sample_type": self.inferencer.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt,
                                    self.inferencer.data_config.eval_dim,
                                    self.inferencer.data_config.prompt_template_type,
                                ),
                            },
                        ],
                    },
                ]
                for prompt in prompts
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}",
                                "max_pixels": max_pixels,
                                "nframes": num_frames,
                                "sample_type": self.inferencer.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt,
                                    self.inferencer.data_config.eval_dim,
                                    self.inferencer.data_config.prompt_template_type,
                                ),
                            },
                        ],
                    },
                ]
                for prompt in prompts
            ]
        # image_inputs, video_inputs = process_vision_info(chat_data)
        logger.debug(f"chat data: {chat_data}")
        image_inputs = None
        video_inputs = self._process_video_inputs(chat_data, video_inputs, video_infos)
        batch = self.inferencer.processor(
            text=self.inferencer.processor.apply_chat_template(
                chat_data, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self.inferencer._prepare_inputs(batch)
        return batch

    def _reward(
        self,
        video_inputs,
        video_infos,
        prompts,
        fps=None,
        num_frames=None,
        max_pixels=None,
        use_norm=True,
    ):
        """
        Inputs:
            video_inputs: List[torch.Tensor], B videos in TCHW format.
            video_infos: List[dict], B video infos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert (
            fps is None or num_frames is None
        ), "fps and num_frames cannot be set at the same time."

        batch_size = 1
        images_batched = torch.chunk(
            video_inputs, int(np.ceil(len(video_inputs) / batch_size)), dim=0
        )
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))
        for video_input, prompt in zip(images_batched, prompts_batched):
            video_input = (
                video_input.squeeze(0) if len(video_input.shape) == 5 else video_input
            )
            video_input = video_input.permute(1, 0, 2, 3)  # Convert to TCHW format
            batch = self._prepare_batch(
                video_input, video_infos, prompt, fps, num_frames, max_pixels
            )
            rewards = self.inferencer.model(return_dict=True, **batch)["logits"]
            rewards = [
                {"VQ": reward[0].item(), "MQ": reward[1].item(), "TA": reward[2].item()}
                for reward in rewards
            ]
            for i in range(len(rewards)):
                if use_norm:
                    rewards[i] = self.inferencer._norm(rewards[i])
                rewards[i]["Overall"] = (
                    rewards[i]["VQ"] + rewards[i]["MQ"] + rewards[i]["TA"]
                )
        return rewards

    def calculate_reward(self, images, metadata):
        st = time.time()
        batch_size = 1
        prompts = metadata["prompts"]
        images_batched = torch.chunk(
            images, int(np.ceil(len(images) / batch_size)), dim=0
        )
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))
        all_vq_rewards = []
        all_mq_rewards = []
        all_ta_rewards = []
        all_overall_rewards = []

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
                with torch.no_grad():
                    reward = self._reward(image_batch, info_batch, prompts_batch)
                    vq_reward = torch.tensor(reward[0]["VQ"]).cpu()
                    all_vq_rewards.append(vq_reward.unsqueeze(0))
                    mq_reward = torch.tensor(reward[0]["MQ"]).cpu()
                    all_mq_rewards.append(mq_reward.unsqueeze(0))
                    ta_reward = torch.tensor(reward[0]["TA"]).cpu()
                    all_ta_rewards.append(ta_reward.unsqueeze(0))
                    overall_reward = torch.tensor(reward[0]["Overall"]).cpu()
                    all_overall_rewards.append(overall_reward.unsqueeze(0))
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                vq_reward = torch.tensor(-1.0).cpu()
                all_vq_rewards.append(vq_reward.unsqueeze(0))
                mq_reward = torch.tensor(-1.0).cpu()
                all_mq_rewards.append(mq_reward.unsqueeze(0))
                ta_reward = torch.tensor(-1.0).cpu()
                all_ta_rewards.append(ta_reward.unsqueeze(0))
                overall_reward = torch.tensor(-1.0).cpu()
                all_overall_rewards.append(overall_reward.unsqueeze(0))

        all_vq_rewards = torch.cat(all_vq_rewards, dim=0).cpu().numpy().tolist()
        all_mq_rewards = torch.cat(all_mq_rewards, dim=0).cpu().numpy().tolist()
        all_ta_rewards = torch.cat(all_ta_rewards, dim=0).cpu().numpy().tolist()
        all_overall_rewards = (
            torch.cat(all_overall_rewards, dim=0).cpu().numpy().tolist()
        )

        scores = {
            "vq_reward": all_vq_rewards,
            "mq_reward": all_mq_rewards,
            "ta_reward": all_ta_rewards,
            "overall_reward": all_overall_rewards,
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
