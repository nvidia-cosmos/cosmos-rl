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
