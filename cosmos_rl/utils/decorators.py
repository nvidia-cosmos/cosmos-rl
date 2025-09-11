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

"""Common decorators for Cosmos-RL with status monitoring."""

from functools import wraps
import os
from typing import Optional, Dict, Any, Callable

from nvidia_tao_core.loggers.logging import (
    set_status_logger,
    get_status_logger,
    StatusLogger,
    Status,
    Verbosity
)
from cosmos_rl.utils.logging import logger


def monitor_status(name: str = 'Cosmos-RL',
                   mode: str = 'train',
                   results_dir: Optional[str] = None,
                   verbosity: int = None):
    """Status monitoring decorator for Cosmos-RL functions.

    This decorator provides comprehensive status logging for training and other
    operations, similar to the TAO Deploy and TAO PyTorch implementations.

    Args:
        name: Name of the operation being monitored
        mode: Mode of operation (e.g., 'train', 'rollout', 'evaluate')
        results_dir: Directory to save status logs (auto-detects from config/TAO_API_JOB_ID)
        verbosity: Logging verbosity level
    """
    def inner(runner: Callable) -> Callable:
        @wraps(runner)
        def _func(*args, **kwargs):
            # Setup results directory - use job results dir if available
            if results_dir is None:
                # Try to get results_dir from config if available
                config = None
                if args and hasattr(args[0], 'config'):
                    config = args[0].config
                elif 'config' in kwargs:
                    config = kwargs['config']
                elif 'cfg' in kwargs:
                    config = kwargs['cfg']
                
                if config and hasattr(config, 'results_dir'):
                    default_results_dir = config.results_dir
                else:
                    # Use TAO job environment if available
                    job_id = os.getenv('TAO_API_JOB_ID')
                    if job_id:
                        default_results_dir = f'/results/{job_id}'
                        logger.info(f"Using TAO job results dir: {default_results_dir}")
                    else:
                        default_results_dir = './results'
                        logger.warning(f"TAO_API_JOB_ID not found, using default: {default_results_dir}")
            else:
                default_results_dir = results_dir

            # Create results directory
            os.makedirs(default_results_dir, exist_ok=True)

            # Setup status logging - single consolidated status.json file
            status_file = os.path.join(default_results_dir, "status.json")
            log_verbosity = verbosity if verbosity is not None else Verbosity.INFO
            status_logger = StatusLogger(
                filename=status_file,
                is_master=True,
                verbosity=log_verbosity,
                append=True  # Append to consolidated status.json file
            )
            set_status_logger(status_logger)
            s_logger = get_status_logger()

            # Extract config if it's the first argument
            config = None
            if args and hasattr(args[0], 'logging'):
                config = args[0]
            elif 'config' in kwargs:
                config = kwargs['config']
            elif 'cfg' in kwargs:
                config = kwargs['cfg']

            try:
                # Log operation start 
                s_logger.kpi = {
                    'operation': mode,
                    'component': name
                }
                s_logger.write(
                    status_level=Status.STARTED,
                    message=f"Starting {name} {mode}"
                )
                logger.info(f"Starting {name} {mode} with status logging to {status_file}")

                # Execute the wrapped function
                result = runner(*args, **kwargs)

                # Log successful completion
                s_logger.kpi = {
                    'operation': mode,
                    'component': name,
                    'status': 'completed'
                }
                s_logger.write(
                    status_level=Status.SUCCESS,
                    message=f"{name} {mode} completed successfully"
                )
                logger.info(f"{name} {mode} completed successfully")

                # Check for cloud upload
                if os.getenv("CLOUD_BASED") == "True":
                    s_logger.kpi = {'cloud_upload': True}
                    s_logger.write(
                        status_level=Status.RUNNING,
                        message="Job artifacts are being uploaded to the cloud"
                    )

                return result

            except (KeyboardInterrupt, SystemError) as e:
                s_logger.kpi = {
                    'operation': mode,
                    'component': name,
                    'error_type': type(e).__name__,
                    'interrupted': True
                }
                s_logger.write(
                    message=f"{name} {mode} was interrupted: {str(e)}",
                    verbosity_level=Verbosity.WARNING,
                    status_level=Status.FAILURE
                )
                logger.warning(f"{name} {mode} was interrupted")
                raise

            except Exception as e:
                s_logger.kpi = {
                    'operation': mode,
                    'component': name,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                s_logger.write(
                    message=f"{name} {mode} failed: {str(e)}",
                    verbosity_level=Verbosity.ERROR,
                    status_level=Status.FAILURE
                )
                logger.error(f"{name} {mode} failed: {str(e)}")
                raise

        return _func
    return inner


def log_step_progress(step_name: str = "step"):
    """Decorator to log progress of individual steps within a larger operation.

    Args:
        step_name: Name of the step being executed
    """
    def inner(func: Callable) -> Callable:
        @wraps(func)
        def _func(*args, **kwargs):
            s_logger = get_status_logger()

            # Extract step number if available
            step_num = kwargs.get('step', kwargs.get('current_step', None))

            s_logger.kpi = {'step_name': step_name}
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Executing {step_name}"
            )

            try:
                result = func(*args, **kwargs)

                s_logger.kpi = {'step_name': step_name, 'status': 'completed'}
                s_logger.write(
                    status_level=Status.SUCCESS,
                    message=f"{step_name} completed"
                )

                return result

            except Exception as e:
                s_logger.kpi = {
                    'step_name': step_name,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                s_logger.write(
                    status_level=Status.FAILURE,
                    message=f"{step_name} failed: {str(e)}"
                )
                raise

        return _func
    return inner


def monitor_replica_lifecycle(replica_type: str = "replica"):
    """Decorator to monitor replica lifecycle events.

    Args:
        replica_type: Type of replica (e.g., 'policy', 'rollout')
    """
    def inner(func: Callable) -> Callable:
        @wraps(func)
        def _func(*args, **kwargs):
            s_logger = get_status_logger()

            # Extract replica name if available
            replica_name = kwargs.get('replica_name')
            if not replica_name and args:
                # Try to extract from first argument if it has a name attribute
                if hasattr(args[0], 'name'):
                    replica_name = args[0].name
                elif hasattr(args[0], 'replica_name'):
                    replica_name = args[0].replica_name

            s_logger.kpi = {
                'replica_type': replica_type,
                'event': func.__name__
            }
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"{replica_type} lifecycle event: {func.__name__}"
            )

            return func(*args, **kwargs)

        return _func
    return inner


def monitor_training(name: str = 'Cosmos-RL Training',
                    results_dir: Optional[str] = None,
                    verbosity: int = None,
                    track_timing: bool = True):
    """Training-specific monitoring decorator for Cosmos-RL.

    This decorator provides detailed training progress monitoring using
    the integrated status logging in PolicyStatusManager. No separate 
    training_logger.py needed - all functionality is built into status.py.

    Args:
        name: Name of the training operation
        results_dir: Directory to save training logs
        verbosity: Logging verbosity level
        track_timing: Whether to track detailed timing information
    """
    def inner(training_func: Callable) -> Callable:
        @wraps(training_func)
        def _func(*args, **kwargs):
            # Setup results directory - use job results dir if available
            if results_dir is None:
                # Try to get results_dir from config if available
                config = None
                if args and hasattr(args[0], 'config'):
                    config = args[0].config
                elif 'config' in kwargs:
                    config = kwargs['config']
                elif 'cfg' in kwargs:
                    config = kwargs['cfg']
                
                if config and hasattr(config, 'results_dir'):
                    default_results_dir = config.results_dir
                else:
                    # Use TAO job environment if available
                    job_id = os.getenv('TAO_API_JOB_ID')
                    if job_id:
                        default_results_dir = f'/results/{job_id}'
                        logger.info(f"Using TAO job results dir for training: {default_results_dir}")
                    else:
                        default_results_dir = './training_results'
                        logger.warning(f"TAO_API_JOB_ID not found for training, using default: {default_results_dir}")
            else:
                default_results_dir = results_dir

            os.makedirs(default_results_dir, exist_ok=True)

            # Setup status logger - single consolidated status.json file
            log_verbosity = verbosity if verbosity is not None else Verbosity.INFO
            status_logger = StatusLogger(
                filename=os.path.join(default_results_dir, 'status.json'),
                is_master=True,
                verbosity=log_verbosity,
                append=True  # Append to consolidated status.json file
            )
            set_status_logger(status_logger)
            s_logger = get_status_logger()

            try:
                # Extract training parameters if available
                total_steps = kwargs.get('total_steps') or kwargs.get('max_steps', 1000)
                max_epochs = kwargs.get('max_epochs') or kwargs.get('epochs')
                current_step = kwargs.get('current_step', 0)

                # Log training start
                s_logger.kpi = {
                    'total_steps': total_steps,
                    'max_epochs': max_epochs,
                    'current_step': current_step,
                    'training_phase': 'training_start'
                }
                s_logger.write(
                    status_level=Status.STARTED,
                    message=f"{name} started - Total steps: {total_steps}"
                )

                # Execute the training function
                result = training_func(*args, **kwargs)

                # Extract final metrics if returned as dict
                final_metrics = {}
                if isinstance(result, dict) and 'metrics' in result:
                    final_metrics = result['metrics']
                elif isinstance(result, dict):
                    final_metrics = result

                # Log training completion
                training_kpis = {
                    'training_completed': True,
                    'training_phase': 'training_end'
                }
                if final_metrics:
                    training_kpis.update(final_metrics)
                s_logger.kpi = training_kpis
                s_logger.write(
                    status_level=Status.RUNNING,
                    message=f"{name} completed successfully"
                )

                logger.info(f"{name} completed successfully")
                return result

            except (KeyboardInterrupt, SystemError) as e:
                s_logger.kpi = {'error_type': 'interruption', 'training_phase': 'training_error'}
                s_logger.write(
                    status_level=Status.FAILURE,
                    message=f"{name} was interrupted: {str(e)}"
                )
                logger.warning(f"{name} was interrupted: {str(e)}")
                raise

            except Exception as e:
                s_logger.kpi = {'error_type': 'training_error', 'training_phase': 'training_error'}
                s_logger.write(
                    status_level=Status.FAILURE,
                    message=f"{name} failed: {str(e)}"
                )
                logger.error(f"{name} failed: {str(e)}")
                raise

        return _func
    return inner
