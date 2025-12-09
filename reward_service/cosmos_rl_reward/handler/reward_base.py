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
from cosmos_rl_reward.utils.logging import logger
import queue
from cosmos_rl_reward.utils.client import KVClientManager
from cosmos_rl_reward.utils.shmem import CrossProcessHandler
import multiprocessing as mp


class BaseRewardHandler:
    reward_name = "base"

    def __init__(self, **kwargs):
        """
        Initialize the BaseRewardHandler with necessary attributes.
        Override this method in subclasses to add more attributes if needed.
        Usually the added attributes in subclasses include:
            model_path: the path to the main model used for rewarding.
            dtype: the data type for model.
            device: the device to run the model on, e.g., "cuda" or "cpu".
            download_path: the path to download or load related files including models.
        """
        # For multithreading and multiprocessing
        self.score_pool = None
        self.threading_pool = None
        self.redis_manager = None

        # Initialize multiprocessing manager and queues
        self.manager = mp.Manager()
        self.score_queue = self.manager.Queue()
        self.loop = self.manager.Event()

    @classmethod
    def get_instance(cls, **kwargs):
        """
        Singleton pattern to get the instance of the reward handler.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls(**kwargs)
        return cls._instance

    def calculate_reward(self, images, metadata):
        """
        Calculate the reward given bunches of images as videos with metadata for additional information.

        Args:
            images (Tensor): bunches of images in tensor format (batch_size, C, frame, H, W) to present videos.
            metadata (Dict): additional information for the reward calculation such as prompts, fps, etc.

        Returns:
            A dictionary including the calculated reward scores:
                "scores": the calculated reward scores in dict of list format, reward specific.
                "input_info": the information about the input video latents including value range, shape, etc.
                "duration": the duration in seconds taken for the reward calculation.
                "decoded_duration": the duration in seconds taken for decoding the input latents to videos.
                "type": the type of the reward calculated.
        """

        raise NotImplementedError(
            "This method should be overridden in subclasses to calculate the reward."
        )

    def set_up(self):
        """
        Set up the inference engine and models for the reward handler.
        This method should be overridden in subclasses to set up specific inference engine.
        """
        raise NotImplementedError(
            "This method should be overridden in subclasses to set up the inference engine."
        )

    @classmethod
    def initialize(cls, **kwargs):
        """
        Initialize the reward handler singleton instance in the process by setting up the inference engine and models.
        """
        controller = cls.get_instance(**kwargs)
        controller.set_up()
        logger.info(f"[{cls.reward_name}] Set up complete.")

    @classmethod
    def calculate_scores_in_shmem(cls, shmem_name, scores_q, loop):
        """
        This method is intended to be run in a separate thread or process to continuously check the queue for new tasks.
        It continuously reads tensors from shared memory, calculates rewards, and puts the results into a queue for later recording the scores.
        Args:
            shmem_name (str): The name of the shared memory segment.
            scores_q (mp.Queue): The multiprocessing queue to put the calculated scores.
            loop (mp.Event): The multiprocessing event to control the loop.
        """
        controller = cls.get_instance()
        shmem_controller = CrossProcessHandler(
            name=controller.reward_name,
            shm_name=shmem_name,
            index=1,
            total_handler=2,
        )
        try:
            shmem_controller.init_memory()
        except Exception as e:
            import traceback

            logger.error(
                f"[{cls.reward_name}] Failed to initialize shared memory: {str(e)} \n{traceback.format_exc()}\nPlease check the torch version compatibility."
            )
        while loop.is_set():
            try:
                try:
                    tensor, metadata = shmem_controller.wait_tensor_in_inference()
                    tensor = tensor.detach().clone()
                    shmem_controller.set_finish_in_inference()
                    uuid = metadata.get("uuid", "unknown")
                    if uuid == "unknown":
                        logger.warning(
                            "UUID not found in metadata. Using 'unknown' as UUID."
                        )
                        continue
                    logger.info(
                        f"[{cls.reward_name}] Start calculating scores for {uuid} with tensor shape: {tensor.shape}, dtype: {tensor.dtype}"
                    )
                    scores = controller.calculate_reward(tensor, metadata)
                    scores_q.put((uuid, scores))
                    logger.info(
                        f"[{cls.reward_name}] Complete calculating scores for {uuid}."
                    )
                except queue.Empty:
                    time.sleep(0.001)
                    continue
            except Exception as e:
                import traceback

                logger.error(
                    f"[{cls.reward_name}] Error in calculate_scores_in_shmem: {str(e)}\n{traceback.format_exc()}"
                )
                time.sleep(0.001)

    @classmethod
    def upload_scores_in_queue(cls, scores_q, loop):
        """
        This method is intended to be run in a separate thread or process to continuously check the queue for new tasks.
        It continuously reads calculated scores from the queue and uploads them to the KV store such as Redis.
        Args:
            scores_q (mp.Queue): The multiprocessing queue to get the calculated scores.
            loop (mp.Event): The multiprocessing event to control the loop.
        """
        inst = cls.get_instance()
        if inst.redis_manager is None:
            inst.redis_manager = KVClientManager(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                os.getenv("REDIS_TOKEN", None),
            )
        while loop.is_set():
            try:
                uuid, scores = scores_q.get(block=False)  # Wait for a task
                logger.info(f"[{cls.reward_name}] Start pushing scores for {uuid}")
                inst.redis_manager.push_scores(uuid, scores)
                logger.info(
                    f"[{cls.reward_name}] Complete pushing scores for {uuid}. Remaining scores in queue: {scores_q.qsize()}."
                )
            except queue.Empty:
                time.sleep(0.001)
                continue
