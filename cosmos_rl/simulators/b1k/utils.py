# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import numpy as np
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
)


def get_b1k_task_descriptions():
    task_description_path = os.path.join(
        os.path.dirname(__file__), "behavior_task.jsonl"
    )
    with open(task_description_path, "r") as f:
        text = f.read()
        task_description = [json.loads(x) for x in text.strip().split("\n") if x]

    task_description_map = {
        task_description[i]["task_name"]: task_description[i]["task"]
        for i in range(len(task_description))
    }
    return task_description_map


def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[
        ..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]
    ]  #  7
    arm_right_qpos = proprio_data[
        ..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]
    ]  #  7
    left_gripper_width = proprio_data[
        ..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]
    ].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[
        ..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]
    ].sum(axis=-1, keepdims=True)  # 1
    return np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            # left_gripper_width,
            arm_right_qpos,
            left_gripper_width,  # NOTE: we rearrange the gripper from 21 to 14 to match the action space
            right_gripper_width,
        ],
        axis=-1,
    )
