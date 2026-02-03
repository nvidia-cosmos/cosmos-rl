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

"""
Custom Logger Functions and Hooks Example

This module demonstrates how to use custom_logger_fns and hook_fns
to integrate external logging systems (e.g., TAO, MLflow, etc.)
with Cosmos-RL training workflows.

Usage:
    Pass these functions to cosmos_rl.launcher.worker_entry.main():

    ```python
    from cosmos_rl.launcher.worker_entry import main
    from cosmos_rl.tools.custom_example.custom_loggers_and_hooks import (
        create_status_logger,
        create_validation_hooks,
        create_training_hooks,
    )

    # Create custom logger function
    status_logger = create_status_logger(
        endpoint="http://localhost:8080/status",
        component_name="My Training Job"
    )

    # Create validation hooks
    validation_hooks = create_validation_hooks(
        status_endpoint="http://localhost:8080/status"
    )

    # Create training hooks
    training_hooks = create_training_hooks(
        status_endpoint="http://localhost:8080/status"
    )

    # Merge all hooks
    all_hooks = {**validation_hooks, **training_hooks}

    # Launch the worker
    main(
        custom_logger_fns=[status_logger],
        hook_fns=all_hooks,
    )
    ```
"""

import json
import os
import time
from typing import Any, Callable, Dict, Optional
from cosmos_rl.utils.logging import logger


def create_status_logger(
    endpoint: Optional[str] = None,
    component_name: str = "Cosmos-RL Training",
    log_to_console: bool = True,
) -> Callable[[Dict[str, Any], int], None]:
    """Create a custom logger function for status reporting.

    This function creates a logger that can send training status to external
    systems (e.g., TAO, MLflow, custom monitoring systems).

    Args:
        endpoint: Optional HTTP endpoint to send status updates.
        component_name: Name of the training component for identification.
        log_to_console: Whether to also log to console.

    Returns:
        A callable that takes (report_data, step) and logs the status.

    Example:
        ```python
        status_logger = create_status_logger(
            endpoint="http://monitoring-server/api/status",
            component_name="SFT Training"
        )

        # Pass to worker entry
        main(custom_logger_fns=[status_logger])
        ```
    """

    def status_logger(report_data: Dict[str, Any], step: int) -> None:
        """Log training status to external system.

        Args:
            report_data: Dictionary containing training metrics like:
                - train/loss_avg: Average training loss
                - train/learning_rate: Current learning rate
                - train/iteration_time: Time per iteration
                - val/avg_loss: Validation loss (if validation step)
            step: Current training step
        """
        timestamp = time.time()
        status_payload = {
            "component": component_name,
            "step": step,
            "timestamp": timestamp,
            "metrics": report_data,
        }

        if log_to_console:
            logger.info(
                f"[{component_name}] Step {step}: {json.dumps(report_data, default=str)}"
            )

        if endpoint:
            try:
                import requests

                response = requests.post(endpoint, json=status_payload, timeout=5)
                if response.status_code != 200:
                    logger.warning(
                        f"[{component_name}] Failed to send status: {response.status_code}"
                    )
            except Exception as e:
                logger.warning(f"[{component_name}] Error sending status: {e}")

    return status_logger


def create_validation_hooks(
    status_endpoint: Optional[str] = None,
    component_name: str = "Cosmos-RL Validation",
) -> Dict[str, Callable]:
    """Create validation lifecycle hooks for status monitoring.

    These hooks can be used to send heartbeat/status updates during
    long-running validation to prevent timeout issues in external systems.

    Args:
        status_endpoint: Optional HTTP endpoint to send status updates.
        component_name: Name of the component for identification.

    Returns:
        Dictionary with validation hook functions:
            - pre_validation_hook
            - pre_per_step_validation_hook
            - post_per_step_validation_hook
            - post_validation_hook
    """

    def send_status(phase: str, data: Dict[str, Any]) -> None:
        """Send status update to external system."""
        if status_endpoint:
            try:
                import requests

                payload = {
                    "component": component_name,
                    "phase": phase,
                    "timestamp": time.time(),
                    "data": data,
                }
                requests.post(status_endpoint, json=payload, timeout=5)
            except Exception as e:
                logger.debug(f"Failed to send validation status: {e}")

    def pre_validation_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called before validation starts.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains current_epoch, is_last_step
        """
        logger.info(
            f"[{component_name}] Starting validation at epoch {report_data.get('current_epoch')}"
        )
        send_status(
            "validation_starting",
            {
                "validation_dataset_size": len(worker.val_data_loader.dataset),
                **report_data,
            },
        )

    def pre_per_step_validation_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called before each validation batch.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains current_epoch, batch_index
        """
        batch_idx = report_data.get("batch_index", 0)
        total_batches = len(worker.val_data_loader)
        progress = (batch_idx / total_batches) * 100 if total_batches > 0 else 0
        send_status(
            "validation_batch_start",
            {
                "batch_index": batch_idx,
                "total_batches": total_batches,
                "progress_percent": progress,
            },
        )

    def post_per_step_validation_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called after each validation batch.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains current_epoch, batch_index, val_score
        """
        batch_idx = report_data.get("batch_index", 0)
        total_batches = len(worker.val_data_loader)
        progress = ((batch_idx + 1) / total_batches) * 100 if total_batches > 0 else 0
        send_status(
            "validation_batch_complete",
            {
                "batch_index": batch_idx,
                "total_batches": total_batches,
                "progress_percent": progress,
                "batch_loss": report_data.get("val_score"),
            },
        )

    def post_validation_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called after validation completes.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains current_epoch, val_avg_loss
        """
        logger.info(
            f"[{component_name}] Validation complete. Avg loss: {report_data.get('val_avg_loss')}"
        )
        send_status("validation_complete", report_data)

    return {
        "pre_validation_hook": pre_validation_hook,
        "pre_per_step_validation_hook": pre_per_step_validation_hook,
        "post_per_step_validation_hook": post_per_step_validation_hook,
        "post_validation_hook": post_validation_hook,
    }


def create_training_hooks(
    status_endpoint: Optional[str] = None,
    component_name: str = "Cosmos-RL Training",
    heartbeat_interval: int = 10,
) -> Dict[str, Callable]:
    """Create training lifecycle hooks for status monitoring.

    These hooks can be used to send heartbeat/status updates during
    training to external monitoring systems.

    Args:
        status_endpoint: Optional HTTP endpoint to send status updates.
        component_name: Name of the component for identification.
        heartbeat_interval: Send heartbeat every N steps (default: 10).

    Returns:
        Dictionary with training hook functions:
            - pre_training_hook
            - pre_training_step_hook
            - post_training_step_hook
            - post_training_hook
    """
    last_heartbeat_step = {"value": -heartbeat_interval}

    def send_status(phase: str, data: Dict[str, Any]) -> None:
        """Send status update to external system."""
        if status_endpoint:
            try:
                import requests

                payload = {
                    "component": component_name,
                    "phase": phase,
                    "timestamp": time.time(),
                    "data": data,
                }
                requests.post(status_endpoint, json=payload, timeout=5)
            except Exception as e:
                logger.debug(f"Failed to send training status: {e}")

    def pre_training_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called before training loop starts.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains total_epochs, total_steps, start_epoch, start_step
        """
        logger.info(
            f"[{component_name}] Training starting. "
            f"Total steps: {report_data.get('total_steps')}, "
            f"Total epochs: {report_data.get('total_epochs')}"
        )
        send_status("training_starting", report_data)

    def pre_training_step_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called before each training step.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains current_epoch, current_step, total_steps
        """
        current_step = report_data.get("current_step", 0)
        # Only send periodic heartbeats to avoid flooding
        if current_step - last_heartbeat_step["value"] >= heartbeat_interval:
            send_status("training_heartbeat", report_data)
            last_heartbeat_step["value"] = current_step

    def post_training_step_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called after each training step.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains current_epoch, current_step, total_steps,
                        plus all training metrics from the step
        """
        # This hook can be used for step-level monitoring
        # Metrics include: train/loss_avg, train/learning_rate, etc.
        pass

    def post_training_hook(worker, report_data: Dict[str, Any]) -> None:
        """Called after training completes.

        Args:
            worker: The SFTPolicyWorker instance
            report_data: Contains final_epoch, final_step, total_steps, final_val_loss
        """
        logger.info(
            f"[{component_name}] Training complete. "
            f"Final step: {report_data.get('final_step')}, "
            f"Final validation loss: {report_data.get('final_val_loss')}"
        )
        send_status("training_complete", report_data)

    return {
        "pre_training_hook": pre_training_hook,
        "pre_training_step_hook": pre_training_step_hook,
        "post_training_step_hook": post_training_step_hook,
        "post_training_hook": post_training_hook,
    }


def create_all_hooks(
    status_endpoint: Optional[str] = None,
    component_name: str = "Cosmos-RL",
    heartbeat_interval: int = 10,
) -> Dict[str, Callable]:
    """Create all training and validation hooks combined.

    Convenience function to create all hooks at once.

    Args:
        status_endpoint: Optional HTTP endpoint to send status updates.
        component_name: Name of the component for identification.
        heartbeat_interval: Send heartbeat every N steps (default: 10).

    Returns:
        Dictionary with all hook functions (training + validation).
    """
    validation_hooks = create_validation_hooks(
        status_endpoint=status_endpoint,
        component_name=f"{component_name} Validation",
    )
    training_hooks = create_training_hooks(
        status_endpoint=status_endpoint,
        component_name=f"{component_name} Training",
        heartbeat_interval=heartbeat_interval,
    )
    return {**validation_hooks, **training_hooks}


# Example usage for TAO-like status logging
class TAOStatusLogger:
    """TAO-compatible status logger that writes to status.json file.

    This class replicates the TAO status logging functionality, writing
    status updates to a JSON file in the format expected by TAO/NVAIE.

    The status file is written to:
        {TAO_API_RESULTS_DIR}/{TAO_API_JOB_ID}/status.json

    Usage:
        ```python
        tao_logger = TAOStatusLogger(
            experiment_name="sft-training"
        )

        main(
            custom_logger_fns=[tao_logger.log_status],
            hook_fns=tao_logger.get_hooks(),
        )
        ```
    """

    def __init__(
        self,
        experiment_name: str = "cosmos-rl-experiment",
        status_file_path: Optional[str] = None,
    ):
        """Initialize TAO status logger.

        Args:
            experiment_name: Name of the experiment/component for logging.
            status_file_path: Optional explicit path to status.json file.
                If not provided, uses TAO_API_RESULTS_DIR/TAO_API_JOB_ID/status.json
        """
        self.experiment_name = experiment_name
        self._status_file_path = status_file_path
        self._status_logger = None

    def _get_status_file_path(self) -> Optional[str]:
        """Get the TAO status file path based on environment variables.

        Returns:
            Path to status.json file, or None if TAO_API_JOB_ID not set.
        """
        if self._status_file_path:
            return self._status_file_path

        job_id = os.environ.get("TAO_API_JOB_ID")
        if not job_id:
            logger.debug("TAO_API_JOB_ID not set, skipping status.json logging")
            return None

        # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
        results_base = os.environ.get("TAO_API_RESULTS_DIR", "/results")
        results_dir = os.path.join(results_base, job_id)
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, "status.json")

    def _get_status_logger(self):
        """Get or create the TAO StatusLogger instance."""
        if self._status_logger is None:
            status_file = self._get_status_file_path()
            if status_file is None:
                return None

            try:
                from nvidia_tao_core.loggers.logging import StatusLogger, Verbosity

                is_master = int(os.environ.get("NODE_RANK", 0)) == 0
                self._status_logger = StatusLogger(
                    filename=status_file,
                    is_master=is_master,
                    verbosity=Verbosity.INFO,
                    append=True,
                )
            except ImportError:
                logger.warning(
                    "nvidia_tao_core not installed. Using fallback JSON logging."
                )
                # Use fallback logger if TAO core not available
                self._status_logger = _FallbackStatusLogger(status_file)

        return self._status_logger

    @staticmethod
    def _convert_tensors_to_scalars(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any PyTorch tensors in a dict to Python scalars for JSON serialization."""
        result = {}
        for k, v in data.items():
            if hasattr(v, "item"):  # PyTorch tensor
                result[k] = v.item()
            elif hasattr(v, "tolist"):  # NumPy array
                result[k] = v.tolist()
            else:
                result[k] = v
        return result

    def log_status(self, report_data: Dict[str, Any], step: int) -> None:
        """Custom logger function for TAO status updates.

        This replaces the hardcoded log_tao_status calls with a pluggable
        custom logger function that writes to status.json.

        Args:
            report_data: Dictionary containing training/validation metrics.
            step: Current training step.
        """
        status_logger = self._get_status_logger()
        if status_logger is None:
            return

        try:
            from datetime import timedelta

            # Convert any tensors to Python scalars for JSON serialization
            report_data = self._convert_tensors_to_scalars(report_data)

            # Get epoch info from either training or validation report_data
            current_epoch = report_data.get(
                "train/cur_epoch", report_data.get("val/cur_epoch")
            )
            max_epochs = report_data.get(
                "train/total_epochs", report_data.get("val/train_epochs")
            )
            total_steps = report_data.get(
                "train/total_steps", report_data.get("val/total_steps", 0)
            )
            steps_per_epoch = report_data.get("steps_per_epoch", 1)

            # Use epoch-based logging if provided, otherwise fall back to step-based
            # But calculate ETA based on remaining STEPS for more accurate progress
            iteration_time = report_data.get("train/iteration_time", 1.0)

            if current_epoch is not None and max_epochs is not None:
                current_value = current_epoch
                max_value = max_epochs
                # ETA based on remaining steps (more accurate than remaining epochs)
                remaining_steps = max(total_steps - step, 0)
                eta_seconds = remaining_steps * iteration_time
                estimated_time_per_unit = iteration_time * steps_per_epoch
                log_key = "epoch"
            else:
                current_value = step
                max_value = total_steps if total_steps > 0 else step
                remaining_steps = max(max_value - current_value, 0)
                eta_seconds = remaining_steps * iteration_time
                estimated_time_per_unit = iteration_time
                log_key = "step"

            # Create TAO-compatible status data
            tao_data = {
                "component": self.experiment_name,
                log_key: current_value,
                f"max_{log_key}": max_value,
                "time_per_epoch": str(timedelta(seconds=estimated_time_per_unit)),
                "eta": str(timedelta(seconds=eta_seconds)),
            }

            # Create summary message based on available metrics
            if "train/loss_avg" in report_data:
                message = f"Training {log_key} {current_value}/{max_value} - Loss: {report_data['train/loss_avg']:.6f}"
            elif "val/loss" in report_data or "val/avg_loss" in report_data:
                val_loss = report_data.get("val/loss", report_data.get("val/avg_loss"))
                message = f"Validation {log_key} {current_value}/{max_value} - Loss: {val_loss:.6f}"
            else:
                message = f"{self.experiment_name} in progress"

            # Write to status file
            # Check if using fallback logger (supports kpi as argument) or TAO core logger
            if isinstance(status_logger, _FallbackStatusLogger):
                status_logger.write(data=tao_data, kpi=report_data, message=message)
            else:
                # TAO core StatusLogger: set kpi as attribute, then call write()
                try:
                    from nvidia_tao_core.loggers.logging import Status, Verbosity

                    status_logger.kpi = report_data
                    status_logger.write(
                        data=tao_data,
                        status_level=Status.RUNNING,
                        verbosity_level=Verbosity.INFO,
                        message=message,
                    )
                except Exception as write_err:
                    logger.warning(f"TAO core write failed: {write_err}")

            logger.debug(
                f"TAO status logged for {self.experiment_name} {log_key} {current_value}"
            )

        except Exception as e:
            logger.warning(f"TAO status logging failed: {e}")

    def _write_status(
        self, phase: str, data: Dict[str, Any], message: str = ""
    ) -> None:
        """Write status update to TAO status file."""
        # Only write from master rank (rank 0)
        node_rank = int(os.environ.get("NODE_RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        if node_rank != 0 or local_rank != 0:
            return

        status_logger = self._get_status_logger()
        if status_logger is None:
            return

        try:
            # Convert tensors to scalars
            data = self._convert_tensors_to_scalars(data)

            # Only put component and phase at outer level, metrics go in kpi only
            tao_data = {
                "component": self.experiment_name,
                "phase": phase,
            }

            if not message:
                message = f"{self.experiment_name} - {phase}"

            if isinstance(status_logger, _FallbackStatusLogger):
                status_logger.write(data=tao_data, kpi=data, message=message)
            else:
                try:
                    from nvidia_tao_core.loggers.logging import Status, Verbosity

                    status_logger.kpi = data
                    status_logger.write(
                        data=tao_data,
                        status_level=Status.RUNNING,
                        verbosity_level=Verbosity.INFO,
                        message=message,
                    )
                except Exception as write_err:
                    logger.debug(f"TAO hook write failed: {write_err}")
        except Exception as e:
            logger.debug(f"TAO hook status failed: {e}")

    def get_hooks(self) -> Dict[str, Callable]:
        """Get all hooks for TAO status updates during training/validation.

        Returns hooks that write to the TAO status file for per-batch progress.
        """

        def pre_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            self._write_status(
                "validation_starting",
                {
                    "validation_dataset_size": len(worker.val_data_loader.dataset),
                    **report_data,
                },
                f"Starting validation at epoch {report_data.get('current_epoch', 0) + 1}",
            )

        def pre_per_step_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            batch_idx = report_data.get("batch_index", 0)
            total_batches = len(worker.val_data_loader)
            progress = (batch_idx / total_batches) * 100 if total_batches > 0 else 0
            self._write_status(
                "validation_batch_start",
                {
                    "batch_index": batch_idx,
                    "total_batches": total_batches,
                    "progress_percent": progress,
                },
                f"Validation batch {batch_idx + 1}/{total_batches}",
            )

        def post_per_step_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            batch_idx = report_data.get("batch_index", 0)
            total_batches = len(worker.val_data_loader)
            progress = (
                ((batch_idx + 1) / total_batches) * 100 if total_batches > 0 else 0
            )
            self._write_status(
                "validation_batch_complete",
                {
                    "batch_index": batch_idx,
                    "total_batches": total_batches,
                    "progress_percent": progress,
                    "batch_loss": report_data.get("val_score"),
                },
                f"Validation batch {batch_idx + 1}/{total_batches} complete",
            )

        def post_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            self._write_status(
                "validation_complete",
                report_data,
                f"Validation complete. Avg loss: {report_data.get('val_avg_loss', 'N/A')}",
            )

        return {
            "pre_validation_hook": pre_validation_hook,
            "pre_per_step_validation_hook": pre_per_step_validation_hook,
            "post_per_step_validation_hook": post_per_step_validation_hook,
            "post_validation_hook": post_validation_hook,
        }


class _FallbackStatusLogger:
    """Fallback status logger when nvidia_tao_core is not available.

    Writes status to JSON file in a TAO-compatible format.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.is_master = int(os.environ.get("NODE_RANK", 0)) == 0

    def write(
        self,
        data: Dict[str, Any],
        kpi: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> None:
        """Write status to JSON file."""
        if not self.is_master:
            return

        import json
        from datetime import datetime

        status_entry = {
            "date": datetime.now().isoformat(),
            "status": "RUNNING",
            "message": message,
            "data": data,
        }
        if kpi:
            status_entry["kpi"] = kpi

        try:
            # Read existing entries
            entries = []
            if os.path.exists(self.filename):
                with open(self.filename, "r") as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, list):
                            entries = content
                        else:
                            entries = [content]
                    except json.JSONDecodeError:
                        entries = []

            # Append new entry
            entries.append(status_entry)

            # Write back
            with open(self.filename, "w") as f:
                json.dump(entries, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to write status to {self.filename}: {e}")
