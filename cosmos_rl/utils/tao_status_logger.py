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

"""TAO-compatible status logger for Cosmos-RL training metrics."""

import os
from typing import Dict, Any
from nvidia_tao_core.loggers.logging import (
    StatusLogger,
    Status,
    Verbosity
)
from cosmos_rl.utils.logging import logger


def get_tao_status_file_path() -> str:
    """Get the TAO status file path based on TAO_API_JOB_ID environment variable.

    Returns:
        Path to status.json file: /results/{TAO_API_JOB_ID}/status.json
    """
    job_id = os.getenv('TAO_API_JOB_ID')
    if not job_id:
        raise ValueError("TAO_API_JOB_ID environment variable is required for TAO status logging")

    results_dir = f"/results/{job_id}"
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, "status.json")


def log_tao_status(data: Dict[str, Any], step: int = None, component_name: str = "Cosmos-RL Training", max_steps: int = None, current_epoch: int = None, max_epochs: int = None):
    """Log training/validation metrics directly to TAO status file.

    This function works like wandb logging - called directly when needed,
    no initialization required. Uses the exact same format as TAO PyTorch.

    Args:
        data: Training or validation metrics data
        step: Current step number (deprecated, use current_epoch instead)
        component_name: Name of the component logging the data
        max_steps: Maximum number of training steps (deprecated, use max_epochs instead)
        current_epoch: Current epoch number
        max_epochs: Maximum number of training epochs
    """
    try:
        # Get TAO status file path
        status_file = get_tao_status_file_path()

        # Create status logger on-demand
        status_logger = StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=Verbosity.INFO,
            append=True
        )

        # Put metrics in KPI dictionary as-is
        status_logger.kpi = data

        # Follow TAO PyTorch loggers.py format with step, max_step, time_per_step, eta
        from datetime import timedelta

        # Use epoch-based logging if provided, otherwise fall back to step-based
        if current_epoch is not None and max_epochs is not None:
            # Epoch-based logging
            current_value = current_epoch
            max_value = max_epochs
            remaining_units = max(max_epochs - current_epoch, 0)
            # Estimate time per epoch based on iteration time and steps per epoch (if available)
            estimated_time_per_unit = data.get('train/iteration_time', 1.0) * data.get('steps_per_epoch', 100)
        else:
            # Fall back to step-based logging for backward compatibility
            current_value = step if step is not None else 0
            max_value = max_steps if max_steps is not None else current_value + 1000
            remaining_units = max(max_value - current_value, 0)
            estimated_time_per_unit = data.get('train/iteration_time', 1.0)

        eta_seconds = remaining_units * estimated_time_per_unit

        # Create summary message based on available metrics
        log_key = "epoch" if current_epoch is not None else "step"
        tao_data = {
            'component': component_name,
            log_key: current_value,
            f'max_{log_key}': max_value,
            'time_per_epoch': str(timedelta(seconds=estimated_time_per_unit)),
            'eta': str(timedelta(seconds=eta_seconds))
        }

        # Use epoch-based messaging
        if 'train/loss_avg' in data:
            message = f"Training {log_key} {current_value}/{max_value} - Loss: {data['train/loss_avg']:.6f}"
        elif 'val/loss' in data:
            message = f"Validation {log_key} {current_value}/{max_value} - Loss: {data['val/loss']:.6f}"
        else:
            message = "Training loop in progress"

        # Write to TAO status log (same format as TAO PyTorch loggers.py)
        status_logger.write(
            data=tao_data,  # TAO PyTorch format with step, max_step, time_per_step, eta
            status_level=Status.RUNNING,
            verbosity_level=Verbosity.INFO,
            message=message
        )

        if current_epoch is not None:
            logger.debug(f"TAO status logged for {component_name} epoch {current_epoch}")
        else:
            logger.debug(f"TAO status logged for {component_name} step {current_value}")

    except Exception as e:
        import traceback
        logger.warning(traceback.format_exc())
        if current_epoch is not None:
            logger.warning(f"TAO status logging failed for epoch {current_epoch} with exception: {e}")
        else:
            logger.warning(f"TAO status logging failed for step {current_value if 'current_value' in locals() else step} with exception: {e}")
