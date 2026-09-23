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
from typing import Union

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig
from cosmos_rl.utils.logging import logger

try:
    import wandb
except ImportError:
    wandb = None
    logger.warning(
        "wandb is not installed. Please install it to use wandb logging features."
    )


def is_wandb_available() -> bool:
    """
    Check if wandb is available in the current environment.

    Returns:
        bool: True if wandb is available, False otherwise.
    """
    try:
        import wandb  # noqa: F401

        return wandb.api.api_key is not None
    except ImportError:
        return False


wandb_run = None


def init_wandb(config: Union[CosmosConfig, CosmosVisionGenConfig]):
    """Create or borrow the active SDK run; never finish an application run.

    An existing run owns its identity/configuration. Call this again explicitly
    to adopt a replacement run. Failed initialization must not retain an old
    cached handle.
    """
    global wandb_run
    wandb_run = None
    if wandb is None:
        logger.warning("Wandb is not installed; logging is disabled.")
        return None
    if wandb.run is not None:
        wandb_run = wandb.run
        logger.info("Using the existing W&B run without changing its configuration.")
        return wandb_run

    resume = "allow"
    if isinstance(config, CosmosConfig):
        output_dir = config.train.output_dir
        project_name = config.logging.project_name
        group_name = config.logging.group_name
        wandb_id = config.logging.wandb_run_id or config.train.timestamp
        resume = config.logging.wandb_resume
        os.makedirs(output_dir, exist_ok=True)
        if (
            config.logging.experiment_name is None
            or config.logging.experiment_name == "None"
            or config.logging.experiment_name == ""
        ):
            experiment_name = output_dir
        else:
            experiment_name = os.path.join(
                config.logging.experiment_name, config.train.timestamp
            )
        if config.logging.wandb_run_name is not None:
            experiment_name = config.logging.wandb_run_name
    elif isinstance(config, CosmosVisionGenConfig):
        output_dir = config.job.path_local
        experiment_name = config.job.name
        project_name = config.job.project
        group_name = config.job.group
        wandb_id = config.job.timestamp
    else:
        logger.error("Unsupported config type for wandb initialization.")
        return None

    logger.info(
        f"Initialize wandb, project: {project_name}, experiment: {experiment_name}. Saved to {output_dir}"
    )
    try:
        run = wandb.init(
            project=project_name,
            group=group_name,
            name=experiment_name,
            config=config.model_dump(),
            dir=output_dir,
            id=wandb_id,
            resume=resume,
        )
        wandb_run = run
        return run
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return None


def log_wandb(data: dict, step: int):
    global wandb_run
    # Finishing/replacing a run in the host application invalidates our borrowed
    # handle. Do not log to a finished run or silently adopt an unrelated run.
    if wandb is None or wandb_run is not wandb.run:
        wandb_run = None
    if wandb_run is not None:
        wandb_run.log(data, step=step)
    else:
        logger.warning("Wandb is not initialized. Please check the configuration.")
