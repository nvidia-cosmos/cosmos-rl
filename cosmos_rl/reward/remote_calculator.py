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
import json
from queue import Queue
import numpy as np
import os
import requests
from functools import partial
from typing import List, Optional, Callable, Tuple

import torch

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.model.wfm.tokenizer.wan2pt1 import Wan2pt1VAEInterface
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.network_util import make_request_with_retry


class RemoteRewardCalculator:
    """
    RemoteRewardCalculator is responsible for calculating the rewards for the rollouts remotely.
    It adds rewards and advantages to the RLPayload.
    It supports dynamic sampling to filter out rollouts that have the same filter rewards with valid=False.
    It also supports finding shared prefix among rollouts and ignore the prefix tokens during training.
    """

    def setup(
        self,
        config: Config,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
    ) -> None:
        """
        Setup the RemoteRewardCalculator with the given configuration and data packers.
        Args:
            config (Config): The configuration for the reward calculator.
            reward_fns (Optional[List[Callable]]): The list of reward functions for training.
            filter_reward_fns (Optional[List[Callable]]): The list of filter reward functions for dynamic sampling.
            val_reward_fns (Optional[List[Callable]]): The list of reward functions for validation.
            data_packer (Optional[BaseDataPacker]): The data packer for processing the payloads.
            val_data_packer (Optional[BaseDataPacker]): The data packer for processing the validation payloads.
        """
        if hasattr(self, "rl_algo"):
            logger.warning(
                "[RemoteRewardCalculator] RemoteRewardCalculator is already setup, returning directly."
            )
            return
        self.config = config.train.train_policy.remote_reward
        # We use wan2pt1 VAE tokenizer to encode the images/videos into latents.
        self.tokenizer = Wan2pt1VAEInterface(
            **config.policy.diffusers.tokenizer.model_dump()
        )
        self.enqueue_url = os.environ.get("REMOTE_REWARD_ENQUEUE_URL", "")
        self.fetch_url = os.environ.get("REMOTE_REWARD_FETCH_URL", "")
        self.token = os.environ.get("REMOTE_REWARD_TOKEN", "")
        self.uuid2payload = dict()

    @classmethod
    def get_instance(cls) -> "RemoteRewardCalculator":
        """
        Get the singleton instance of the RemoteRewardCalculator.
        Returns:
            RemoteRewardCalculator: The singleton instance of the RemoteRewardCalculator.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def compute_validation_rewards(
        self,
        payloads: List[RLPayload],
        step: int,
    ) -> Tuple[List[RLPayload], bool, int]:
        """
        Compute rewards and advantages for the given payloads using validation reward function.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            step (int): The weight step where the payloads are generated.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set (always True)
                step: the weight step where the payloads are generated
        """
        # TODO(dinghaoy): support remote validation reward computation
        raise NotImplementedError(
            "[RemoteRewardCalculator] Remote validation reward computation is not implemented yet."
        )

    def enqueue_request(self, tensor, data):
        """Enqueue the request and return UUID."""

        buffer = io.BytesIO()
        np.save(buffer, tensor)
        buffer.seek(0)

        # Combine JSON + binary data
        payload = json.dumps(data).encode("utf-8") + b"\n" + buffer.getvalue()

        response = make_request_with_retry(
            partial(
                requests.post,
                data=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Authorization": f"Bearer {self.token}",
                },
                timeout=10.0,
            ),
            [self.enqueue_url],
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Enqueue failed with status {response.status_code}: {response.text}"
            )

        uuid = response.json()["uuid"]
        # logger.info(f"[RemoteReward] Enqueued request with UUID: {uuid}, latents.max(): {tensor.max()}")
        return uuid

    def compute_rewards(
        self,
        payloads: List[RLPayload],
        is_validation: bool,
        step: int,
    ) -> Tuple[List[RLPayload], bool, int]:
        """
        Send reward calculation request to remote server and get UUID.
        If is_validation is True, use the validation reward function and return all rollouts.
        If is_validation is False, use the training reward function and apply dynamic sampling.
        Args:
            payloads (List[RLPayload]): List of RLPayload to compute rewards for.
            is_validation (bool): Whether the payloads are from validation set.
            step (int): The weight step where the payloads are generated.
        Returns:
            uuid (str): The UUID of the enqueued reward calculation request.
        """

        if is_validation:
            return self.compute_validation_rewards(payloads, step)

        payload = payloads[0]
        mm_datas = payload.completions
        prompts = [payload.prompt] * len(mm_datas)
        latents = self.tokenizer.encode(mm_datas)

        tensor = latents.cpu().numpy()

        # Create video info for entire batch (assuming 16 FPS as default)
        video_infos = []
        for _ in range(latents.shape[0]):
            video_infos.append({"video_fps": 16.0})

        data = {
            "prompts": prompts,
            "reward_fn": {
                f"{self.reward_fn}": 1.0,
            },
            "video_infos": video_infos,
        }

        # Enqueue request (single call for entire batch)
        uuid = self.enqueue_request(tensor, data)
        self.uuid2payload[uuid] = payload

        return uuid

    def fetch_reward(self, uuid, return_all: bool = False):
        """Poll for reward until ready."""

        response = make_request_with_retry(
            partial(
                requests.post,
                data={"uuid": uuid, "type": self.config.reward_fn},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=5.0,
            ),
            [self.fetch_url],
        )

        response_json = response.json()

        # Extract overall reward
        if return_all:
            return response_json["scores"]
        if self.config.score_key not in response_json["scores"]:
            rewards = [
                torch.tensor(response_json["scores"][w])
                for w in self.config.score_key.split("+")
            ]
            reward = sum(rewards)
        else:
            reward = torch.tensor(response_json["scores"][self.config.score_key])
        return (
            torch.clamp(
                reward, min=self.config.reward_clip_min, max=self.config.reward_clip_max
            )
            * self.config.scale
        )

    def get_results(
        self,
        uuids: Queue[str],
    ) -> Tuple[List[RLPayload], bool, int]:
        """
        Get the results from remote server using the UUIDs.
        Args:
            uuids (Queue[str]): Queue of UUIDs to fetch results for.
        Returns:
            Tuple[List[RLPayload], bool, int]: (payloads, is_validation, step)
                payloads: List of RLPayload with rewards and advantages
                is_validation: whether the payloads are from validation set (always False)
                step: the weight step where the payloads are generated
        """
        # Try to fetch results from the first uuid until failed.
        valid_results = []
        valid_payloads = []
        while not uuids.empty():
            uuid = uuids.queue[0]
            try:
                rewards = self.fetch_reward(uuid)
                uuids.get()  # remove the uuid from the queue
                valid_results.append(rewards)
                valid_payloads.append(self.uuid2payload[uuid])
                # Remove the payload from the dict to save memory
                del self.uuid2payload[uuid]
            except Exception as e:
                logger.debug(
                    f"[RemoteRewardCalculator] Failed to fetch reward for UUID {uuid}: {e}"
                )
                break

        # Convert the rewards results to RLPayloads
        payload_list: List[RLPayload] = []
        for i, payload in enumerate(valid_payloads):
            rewards = valid_results[i]
            # Compute advantages (normalize rewards)
            avg_rewards_all = rewards.mean(dim=1)
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (
                avg_rewards_all.std() + 1e-4
            )
            # Create a new RLPayload with the reward
            new_payload = RLPayload(
                prompt=payload.prompt,
                completions=payload.completions,
                rewards=rewards.tolist(),
                advantages=advantages.tolist(),
                extra_info=payload.extra_info,
            )
            payload_list.append(new_payload)

        return payload_list, False, 0  # the step is not used for remote reward
