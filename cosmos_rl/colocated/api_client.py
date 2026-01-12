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
API client for the dispatcher.
"""

from cosmos_rl.dispatcher.api.client import APIClient
from typing import Dict, Any, List, Tuple, Optional

from cosmos_rl.dispatcher.protocol import (
    RolloutRequest,
)


class ColocatedAPIClient(APIClient):
    """
    API client for the dispatcher in colocated mode.
    In colocated mode, the policy and rollout controllers are in the same process.
    This client directly interacts with the controller instance.
    """

    def set_controller(self, controller):
        """
        Set the controller instance.
        Args:
            controller: The controller instance.
        """
        self.controller = controller

    def get_next_prompt(
        self,
        batch_size: int,
        validation_step: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Get the next batch of prompts from the controller.
        Args:
            batch_size: The number of prompts to fetch.
            validation_step: The current validation step, if any.
            local_control: Whether to use local communication only.
        Returns:
            A tuple of (list of prompts as dicts, is_end flag).
        """
        return super().get_next_prompt(batch_size, validation_step)

    def post_rollout_completion(self, response: RolloutRequest):
        """
        Post the rollout completion response to the controller.
        Args:
            response: The rollout completion response.
        """
        self.controller.put_rollouts(response)

    def post_policy_train_ack(
        self,
        replica_name: str,
        weight_step: int,
        total_steps: int,
        profile_finished: bool,
        report_data: Dict[str, Any],
    ):
        """
        Post the policy training acknowledgment to the controller.
        Args:
            replica_name: The name of the replica.
            weight_step: The current weight step.
            total_steps: The total number of steps.
            profile_finished: Whether profiling is finished.
            report_data: The report data.
        """
        report_data.update(self.controller.train_report_data.get(weight_step, {}))
        super().post_policy_train_ack(
            replica_name, weight_step, total_steps, profile_finished, report_data
        )

    def get_policy_model(self) -> Any:
        """
        Get the policy model parameters from the controller.
        Returns:
            torch.nn.Module: the policy model instance.
        """
        return self.controller.get_policy_model()
