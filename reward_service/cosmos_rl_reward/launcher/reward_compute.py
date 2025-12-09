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
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sys
import os

# Ensure the root directory is in sys.path so that cosmos_rl_reward can be imported
root_directory = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if root_directory not in sys.path:
    sys.path.append(root_directory)

from cosmos_rl_reward.utils.logging import logger
from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
import cosmos_rl_reward.model  # noqa: F401 to register all reward models


def calculate_reward_loop(reward_name, shm_name, **kwargs):
    """
    Calculate reward scores in a loop using shared memory to communicate for getting the decoded videos with metadata.
    Two processes are used:
        - for initializing the reward model and calculating scores using the model.
        - for uploading the calculated scores to the key-value store such as Redis.
        - These two processes run concurrently and communicate via a shared queue for passing scores.
    Args:
        reward_name (str): The name of the reward handler to use.
        shm_name (str): The name of the shared memory segment.
    """

    reward_class: BaseRewardHandler = RewardRegistry.get_reward_class(reward_name)
    if not reward_class:
        raise ValueError(f"Unknown reward name: {reward_name}")
    controller: BaseRewardHandler = reward_class.get_instance(**kwargs)
    if controller.threading_pool is None:
        controller.threading_pool = ProcessPoolExecutor(max_workers=1)
    future = controller.threading_pool.submit(reward_class.initialize, **kwargs)
    if controller.score_pool is None:
        controller.score_pool = ProcessPoolExecutor(max_workers=1)
    # Get result
    future.result()
    logger.debug("[Reward compute] Start subprocesses.")
    if not controller.loop.is_set():
        controller.loop.set()
        controller.threading_pool.submit(
            reward_class.calculate_scores_in_shmem,
            shm_name,
            controller.score_queue,
            controller.loop,
        )
        controller.score_pool.submit(
            reward_class.upload_scores_in_queue,
            controller.score_queue,
            controller.loop,
        )
    while controller.loop.is_set():
        time.sleep(1)


def main():
    """
    Main function to parse arguments and start the reward calculation loop.
    """

    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Run the web panel for the dispatcher."
    )
    parser.add_argument(
        "--shm",
        type=str,
        required=True,
        help="Shared memory name for the dispatcher.",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="dance_grpo",
        help="Reward to run, e.g., dance_grpo",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the reward model.",
    )

    parser.add_argument(
        "--download_path",
        type=str,
        default="",
        help="Root folder path of the downloaded models.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the reward model on.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type for model inference.",
    )

    args = parser.parse_args()
    calculate_reward_loop(
        args.reward,
        args.shm,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        download_path=args.download_path,
    )


if __name__ == "__main__":
    main()
