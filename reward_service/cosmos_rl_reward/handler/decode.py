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

import os
import time
import json
import torch
import numpy as np
from cosmos_rl_reward.utils.logging import logger
import multiprocessing as mp
import queue
import io
from cosmos_rl_reward.utils.shmem import CrossProcessHandler
from typing import Dict


class DecodeHandler:
    name = "decoder"

    def __init__(self):
        """
        Inititialize the DecodeHandler with necessary attributes.
        """
        self.device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )

        # For multithreading and multiprocessing
        self.threading_pool = None
        self.set_pool = None

        # For communication with reward processes
        self.reward_dispatcher: Dict[str, CrossProcessHandler] = {}

        # Initialize multiprocessing manager and queues
        self.manager = mp.Manager()
        self.task_queue = self.manager.Queue()
        self.loop = self.manager.Event()

        # Control the maximum pending tasks in the queue
        self.max_pending_tasks = int(
            os.getenv("COSMOS_RL_REWARD_MAX_PENDING_TASKS", 60)
        )

    @classmethod
    def get_instance(cls):
        """
        Singleton pattern to get the instance of the decode handler.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def init_reward_process(self, reward_name: str, shmem_name: str):
        """
        Initialize the cross process communicator for each reward process for a given reward name and shared memory name.
        It will create a CrossProcessHandler for the reward process using the given shared memory name.
        The CrossProcessHandler will be used to communicate the decode process with the reward process with the given shared memory.
        The decoded videos with information metadata will be sent to the reward process via the shared memory in CrossProcessHandler.

        Args:
            reward_name (str): The name of the reward process.
            shmem_name (str): The name of the shared memory segment.

        """
        if reward_name not in self.reward_dispatcher:
            self.reward_dispatcher[reward_name] = CrossProcessHandler(
                name=reward_name,
                shm_name=shmem_name,
                index=0,
                total_handler=2,
            )
            self.reward_dispatcher[reward_name].init_memory()
            logger.info(
                f"[{self.name}] Initialized reward process for {reward_name} with shared memory {shmem_name}."
            )

    def set_latent_decoder(
        self,
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        model_path="/workspace/tokenizer_ckpt.pth",
        device="cuda",
        **kwargs,
    ):
        """
        Initialize the Wan2pt1Tokenizer based latent decoder for decoding video latents to videos.
        Args:
            chunk_duration (int): The chunk duration for the tokenizer.
            temporal_window (int): The temporal window for the tokenizer.
            load_mean_std (bool): Whether to load mean and std for the tokenizer.
            model_path (str): The path to the tokenizer model for decoding.
            device (str): The device to run the tokenizer on.
        """
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt1 import Wan2pt1TokenizerHelper
        self.latent_decoder = Wan2pt1TokenizerHelper(
            chunk_duration=chunk_duration,
            load_mean_std=load_mean_std,
            vae_pth=model_path,
            temporal_window=temporal_window,
            device=device,
        )

    def decode_video(self, latents: torch.Tensor):
        """
        Decode the video latents to videos using the latent decoder.
        Args:
            latents (torch.Tensor): The video latents to decode.
        Returns:
            torch.Tensor: The decoded videos in (batch_size, C, frame, H, W) format.
        """

        # Decode the latents
        logger.debug("\n--- DECODING ---")
        reconstructed_video = self.latent_decoder.decode_latents(latents)
        logger.debug(
            f"Reconstructed video range: [{reconstructed_video.min():.3f}, {reconstructed_video.max():.3f}]"
        )
        # save the first video and reconstructed video
        video = ((1 + reconstructed_video[0].clamp(-1, 1)) / 2).unsqueeze(
            0
        ).float() * 255
        video = video.to(torch.uint8)
        logger.info(
            f"[{self.name}] Decoded video: shape: {video.shape}, min: {video.min()}, max: {video.max()}"
        )
        return video

    def export_mp4(self, video, fps, path="/tmp/output.mp4"):
        """
        Export the video tensor to an mp4 file. For debugging purposes.
        Args:
            video (torch.Tensor): The video tensor in (batch=1, C, frame, H, W) format.
            fps (int): Frames per second for the output video.
            path (str): The file path to save the mp4 video.
        """

        import torchvision

        video = video.squeeze(0)  # Remove batch dimension for saving
        video = video.permute(1, 2, 3, 0)
        torchvision.io.write_video(
            path,
            video,
            int(fps),
            video_codec="h264",  # requires ffmpeg with libx264
        )
        logger.info(f"[{self.name}] Exported video to {path} with fps {fps}")

    @classmethod
    def set_latent_attr(cls, fields: str):
        """
        Reset the attributes for the latent decoder based on the given JSON string of fields.
        Args:
            fields (str): The JSON string of the latent attributes including chunk_duration, load_mean_std, vae_pth, temporal_window, device, fps, num_frames, height, and width.
        """
        fields = json.loads(fields)
        logger.info(f"[{cls.name}] Setting latent attributes: {fields}")
        controller = cls.get_instance()
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt1 import Wan2pt1TokenizerHelper
        controller.latent_decoder = Wan2pt1TokenizerHelper(
            chunk_duration=fields.get("chunk_duration", 81),
            load_mean_std=fields.get("load_mean_std", False),
            vae_pth=fields.get("vae_pth", "/workspace/tokenizer_ckpt.pth"),
            temporal_window=fields.get("temporal_window", 16),
            device=fields.get("device", "cuda"),
        )
        controller.fps = fields.get("fps", 16)
        controller.num_frames = fields.get("num_frames", 93)
        controller.height = fields.get("height", 432)
        controller.width = fields.get("width", 768)

    def set_info(self, info={}):
        """
        Set the reward processes related information for the decode handler to initialize the communication with the processes for each reward.
        For each reward name and shared memory name in the info dictionary, it will call init_reward_process to initialize the communication.
        Args:
            info (dict): The information dictionary including reward names and their shared memory names.
        """
        rewards = info.get("reward", {})
        for r, shm in rewards.items():
            if r not in self.reward_dispatcher:
                self.init_reward_process(r, shm)

    @classmethod
    def initialize(cls, info: dict = {}, requires_latent_decode: bool = True, **kwargs):
        """
        Initialize the decode handler singleton instance in the process by setting up
        the latent decoder (if required) and initializing the communication with the
        reward processes.
        Args:
            info (dict): The information dictionary including reward names and their shared memory names.
            requires_latent_decode (bool): Whether the latent decoder is required by any enabled reward.
        """
        controller = cls.get_instance()
        if requires_latent_decode and not hasattr(controller, "latent_decoder"):
            controller.set_latent_decoder(**kwargs)
        controller.set_info(info)
        logger.info(f"[{cls.name}] Set up complete.")

    @classmethod
    def extract_video_in_queue(cls, tasks, max_size, loop):
        """
        This method is intended to be run in a separate thread or process to continuously check the queue for new tasks.
        It continuously reads tasks from the queue, extracts video latents and metadata, decodes the videos, and sends them to the reward processes.
        Args:
            tasks (mp.Queue): The multiprocessing queue to get the tasks.
            max_size (int): The maximum size of the task queue to control pending tasks.
            loop (mp.Event): The multiprocessing event to control the loop.
        """
        controller = cls.get_instance()
        while loop.is_set():
            try:
                while tasks.qsize() > max_size:
                    tasks.get(block=False)  # Ignore tasks that exceed max_size
                uuid, raw_body = tasks.get(block=False)  # Wait for a task
                logger.info(f"[{cls.name}] Start extracting video for {uuid}")
                controller.extract_video(uuid, raw_body)
                logger.info(
                    f"[{cls.name}] Complete extracting video for {uuid}. Remaining tasks in queue: {tasks.qsize()}."
                )
            except queue.Empty:
                time.sleep(0.001)
                continue

    def extract_video(
        self, uuid: str, raw_body: bytes, use_latent=True, export_video=False
    ):
        """
        Extract video latents and metadata from the raw body, decode the videos, and send them to the reward processes.
        It is called by the extract_video_in_queue method to process one task for extracting and decoding one batch of videos.
        Args:
            uuid (str): The unique identifier for the task.
            raw_body (bytes): The raw bytes containing the video latents and metadata.
            use_latent (bool): Whether to decode the video latents using the latent decoder.
            export_video (bool): Whether to export the decoded video to an mp4 file for debugging.
        """
        try:
            # Assume first line is JSON length or delimiter
            # Let's use a simple format: `{JSON}\n<BINARY DATA>`
            delimiter = b"\n"
            sep_index = raw_body.find(delimiter)
            if sep_index == -1:
                return {"status": "error", "message": "Delimiter \\n not found"}
            # Parse JSON part
            json_bytes = raw_body[:sep_index]
            file = raw_body[sep_index + 1 :]  # Skip delimiter
            info_data = json_bytes.decode("utf-8")

            metadata = json.loads(info_data)
            reward_fn = metadata["reward_fn"]
            for key in list(reward_fn.keys()):
                if reward_fn[key] <= 0:
                    logger.warning(
                        f"[{self.name}] Reward function {key} has weight {reward_fn[key]}, which is less than or equal to 0. It will be ignored."
                    )
                    del reward_fn[key]


            media_type = metadata.get("media_type", None)
            is_image_payload = (
                media_type is not None and str(media_type).lower() == "image"
            )

            if is_image_payload:
                # Parse image tensor payload; prefer NPY [B,C,H,W] uint8
                try:
                    buffer = io.BytesIO(file)
                    npy = np.load(buffer, allow_pickle=False)
                except Exception:
                    buffer = io.BytesIO(file)
                    npy = torch.load(buffer, map_location=torch.device("cpu")).cpu().numpy()

                images_tensor = torch.from_numpy(npy)

                decoded_images = images_tensor

                logger.info(
                    f"[{self.name}] Received image tensor with shape: {decoded_images.shape}, dtype: {decoded_images.dtype}, range: [{images_tensor.min():.3f}, {images_tensor.max():.3f}]"
                )

                decoded_info = {
                    "shape": decoded_images.shape,
                    "dtype": str(decoded_images.dtype),
                }
                metadata["decoded_info"] = decoded_info
                metadata["decode_duration"] = "0.00"
                metadata.setdefault("input_info", {})
                metadata["input_info"].update({
                    "shape": images_tensor.shape,
                    "dtype": str(images_tensor.dtype),
                    "min": f"{images_tensor.min():.3f}",
                    "max": f"{images_tensor.max():.3f}",
                })
                metadata["uuid"] = uuid
                logger.info(
                    f"[{self.name}] Prepared image batch for {uuid}"
                )
            else:
                try:
                    buffer = io.BytesIO(file)
                    images = np.load(buffer)
                    images = torch.from_numpy(images)
                    if images.dtype == torch.uint8:
                        images = images.view(torch.bfloat16)
                    else:
                        images = images.to(torch.bfloat16)
                except Exception:
                    buffer = io.BytesIO(file)
                    images = torch.load(buffer, map_location=torch.device("cpu"))
                    images = images.to(torch.bfloat16)
                logger.info(
                    f"[{self.name}] Received raw tensor with shape: {images.shape} dtype: {images.dtype}, range: [{images.min():.3f}, {images.max():.3f}]"
                )
                logger.info(f"[{self.name}] Received metadata: {metadata}")
                input_info = {
                    "shape": images.shape,
                    "dtype": str(images.dtype),
                    "min": f"{images.min():.3f}",
                    "max": f"{images.max():.3f}",
                }
                st = time.time()
                batch_size = 1
                images_batched = torch.chunk(
                    images, int(np.ceil(len(images) / batch_size)), dim=0
                )
                if "video_infos" not in metadata:
                    infos = [{} for _ in range(len(images))]
                else:
                    infos = metadata["video_infos"]
                info_batched = np.array_split(infos, np.ceil(len(infos) / batch_size))

                decoded_images = []
                for image_latent, info_sample in zip(images_batched, info_batched):
                    if use_latent:
                        image_latent = image_latent.to(self.device)
                        logger.debug(
                            f"Image latent shape: {image_latent.shape} {image_latent.dtype}"
                        )
                        decoded_images.append(self.decode_video(image_latent))
                        if export_video:
                            self.export_mp4(
                                decoded_images[-1],
                                info_sample[0].get("video_fps", 16),
                                path=f"/tmp/decoded_video_{uuid}.mp4",
                            )
                    else:
                        decoded_images.append(image_latent)
                decoded_images = torch.cat(decoded_images, dim=0)
                logger.debug(
                    f"Decoded images shape: {decoded_images.shape} {decoded_images.dtype}"
                )
                duration = time.time() - st
                decoded_info = {
                    "shape": decoded_images.shape,
                    "dtype": str(decoded_images.dtype),
                }
                metadata["decoded_info"] = decoded_info
                metadata["decode_duration"] = f"{duration:.2f}"
                metadata["input_info"] = input_info
                metadata["uuid"] = uuid
                logger.info(f"[{self.name}] Decoded images in {duration:.2f} seconds")

            for key in reward_fn:
                if key not in self.reward_dispatcher:
                    raise ValueError(
                        f"Reward process for {key} not initialized. Call init_reward_process('{key}') first."
                    )
                self.reward_dispatcher[key].wait_start_in_decode()
                self.reward_dispatcher[key].set_tensor_in_decode(
                    decoded_images, metadata
                )
        except Exception as e:
            import traceback

            logger.error(
                f"[{self.name}] Error in extracting video: {str(e)}\n{traceback.format_exc()}"
            )
