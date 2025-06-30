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
from cosmos_rl.dispatcher.run_web_panel import main as launch_dispatcher

module_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "deepseek_v3"
)

if __name__ == "__main__":
    # Ensure that the module path is accessible on all nodes
    launch_dispatcher(model_module=module_path)
