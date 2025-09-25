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

from transformers import AutoConfig
from typing import Any


def pre_hf_models_patch(hf_config: AutoConfig):
    if (
        hf_config.model_type == "internvl_chat"
        and hasattr(hf_config, "llm_config")
        and hf_config.llm_config.model_type == "gpt_oss"
    ):
        hf_config.vision_config.drop_path_rate = 0.0
        print("Set drop_path_rate to 0.0")


def post_hf_models_patch(hf_config: AutoConfig, model: Any):
    if (
        hf_config.model_type == "internvl_chat"
        and hasattr(hf_config, "llm_config")
        and hf_config.llm_config.model_type == "gpt_oss"
    ):
        model.img_context_token_id = 200021
        print("Set img_context_token_id to 200021")
