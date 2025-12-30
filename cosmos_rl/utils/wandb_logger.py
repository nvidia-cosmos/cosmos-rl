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
    # In this repo we can have multiple processes per job (policy/rollout/controller, plus
    # multiple replicas). If more than one process calls `wandb.init()` with the same
    # (dir, id), W&B can race on creating the same `run-*.wandb` sync file and crash with
    # `FileExistsError`.
    #
    # Our launch scripts set `COSMOS_ROLE` to one of: Controller/Policy/Rollout/Reference.
    # Only the controller should initialize W&B; other processes should just emit logs to
    # stdout and report metrics to the controller.
    role = os.getenv("COSMOS_ROLE", "").strip()
    if role and role.lower() != "controller":
        logger.info(f"Skip wandb init on COSMOS_ROLE={role} (controller-only).")
        return

    # Avoid duplicate initialization of wandb
    if wandb.run is not None:
        logger.warning("Wandb is already initialized. Skipping initialization.")
        return

    if isinstance(config, CosmosConfig):
        output_dir = config.train.output_dir
        project_name = config.logging.project_name
        group_name = config.logging.group_name
        wandb_id = config.train.timestamp
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
            id=wandb_id,  # Use timestamp as the run ID
            resume="allow",
        )
        global wandb_run
        wandb_run = run
        return run
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return None


def log_wandb(data: dict, step: int):
    global wandb_run
    if wandb_run is not None:
        wandb_run.log(data, step=step)
    else:
        logger.warning("Wandb is not initialized. Please check the configuration.")
