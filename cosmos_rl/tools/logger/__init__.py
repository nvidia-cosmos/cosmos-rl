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


def BaseLoggerFunc(data: dict, step: int) -> None:
    """
    Example of custom logger functions.
    Please mimic this to implement your own custom logger function.
    Your custom logger function is in the same format with this BaseLoggerFunc.
    It shoule be with two arguments: `data` and `step`.

    Arguments:
        The data argument will always be the follwing format:
            data = {
                "train/loss_avg": total_loss_avg,
                "train/loss_max": total_loss_max,
                "train/learning_rate": total_learning_rate,
                "train/iteration_time": total_iter_time_avg,
                "train/kl_loss_avg": total_kl_loss_avg,
                "train/kl_loss_max": total_kl_loss_max,
                "train/grad_norm": total_grad_norm,
            }
        The data argument includes the infomation could be logged.
        The step argument is the number for the current training step.

    Usage:
        After implementing this `BaseLoggerFunc` function, you can use it as a custom logger for your training process:
        In the config .toml file, modify the `logging.logger` field as following:
            ```
            [logging]
            logger = ['console', 'wandb', 'tools/logger/__init__.py:BaseLoggerFunc']

            ```
        The `console` and `wandb` items are the intrinsic logger to log on the console and the wandb.
        The `tools/logger/__init__.py:BaseLoggerFunc` item is the custom logger function you implemented.
        The custom logger is specified as the format `{.py file path}:{function name}`.
        Then, the custom logger function will be called with the appropriate arguments during the training process.
    """
    total_loss_avg = data.get("train/loss_avg", None)
    total_loss_max = data.get("train/loss_max", None)
    total_learning_rate = data.get("train/learning_rate", None)
    total_iter_time_avg = data.get("train/iteration_time", None)
    total_kl_loss_avg = data.get("train/kl_loss_avg", None)
    total_kl_loss_max = data.get("train/kl_loss_max", None)
    total_grad_norm = data.get("train/grad_norm", None)
    print(f"Step {step}:")
    print(f"  Loss: {total_loss_avg} (max: {total_loss_max})")
    print(f"  Learning Rate: {total_learning_rate}")
    print(f"  Iteration Time: {total_iter_time_avg}")
    print(f"  KL Loss: {total_kl_loss_avg} (max: {total_kl_loss_max})")
    print(f"  Grad Norm: {total_grad_norm}")
