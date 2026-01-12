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
from cosmos_rl_reward.utils.logging import logger
import argparse
import multiprocessing as mp
from cosmos_rl_reward.launcher.config import Config
import toml


def main():
    """
    Prepare the environments for running each reward model by setting up necessary dependencies and configurations.
    Each reward model may have its own setup script and virtual environment.
    This script executes the setup scripts for each reward model as specified in the configuration file.
    Each script will download required model files to the specified download path and install dependencies in the specified virtual environment.
    """

    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Run the web panel for the dispatcher."
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration file."
    )

    args = parser.parse_args()

    if args.config is not None:
        logger.info(f"Attempting to load configuration from {args.config}")
        with open(args.config, "r") as f:
            config_dict = toml.load(f)
        loaded_config = Config.from_dict(config_dict)
    else:
        logger.info("No configuration file provided. Using default configuration file.")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(cur_dir, "../configs/rewards.toml")
        if os.path.exists(config_file):
            logger.info(f"Loading default configuration from {config_file}")
            with open(config_file, "r") as f:
                config_dict = toml.load(f)
            loaded_config = Config.from_dict(config_dict)
        else:
            loaded_config = Config()
            logger.info(
                "No default configuration file found. Using default configuration."
            )

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    for reward_arg in loaded_config.reward_args:
        # Respect the enable flag; skip disabled rewards during environment preparation
        if hasattr(reward_arg, "enable") and not reward_arg.enable:
            logger.info(f"Skipping setup for disabled reward: {reward_arg.reward_type}")
            continue
        logger.info(f"Setting up reward: {reward_arg.reward_type}")
        key = reward_arg.reward_type
        download_path = reward_arg.download_path
        venv_path = reward_arg.venv_python.rsplit("/", 2)[0]
        script = (
            reward_arg.setup_script
            if reward_arg.setup_script
            else os.path.join(cur_dir, f"../setup/{key}.sh")
        )
        logger.info(
            f"Running setup script: {script} with download path: {download_path} and venv path: {venv_path}"
        )
        os.system(f"bash {script} {download_path} {venv_path}")
        logger.info(f"Setup script {script} completed.")
    logger.info("Environment preparation completed.")


if __name__ == "__main__":
    main()
