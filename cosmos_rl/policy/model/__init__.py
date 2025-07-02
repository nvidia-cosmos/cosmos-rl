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

from cosmos_rl.policy.model.gpt import GPT
from cosmos_rl.policy.model.qwen2_5_vl import Qwen2_5_VLConditionalModel
from cosmos_rl.policy.model.qwen3_moe import Qwen3MoE
from cosmos_rl.policy.model.deepseek_v3 import DeepseekV3MoEModel
from cosmos_rl.policy.model.hf_llm import HFLLMModel
from cosmos_rl.policy.config import Config as CosmosConfig
import cosmos_rl.utils.util as util
from cosmos_rl.utils.constant import COSMOS_HF_MODEL_TYPES
from cosmos_rl.utils.logging import logger
from transformers import AutoConfig
import torch


supported_cls_list = [
    GPT,
    Qwen2_5_VLConditionalModel,
    Qwen3MoE,
    DeepseekV3MoEModel,
    HFLLMModel,  # Make HFLLMModel the last one to be used
]


def build_model(config: CosmosConfig):
    model_name_or_path = config.policy.model_name_or_path
    model = None
    hf_config = util.retry(AutoConfig.from_pretrained)(
        model_name_or_path, trust_remote_code=True
    )

    with torch.device("meta"):
        with util.cosmos_default_dtype(config.train.param_torch_dtype):
            for model_cls in supported_cls_list:
                supported_model_types = model_cls.supported_model_types()
                use_hfllm_type = supported_model_types[0] == COSMOS_HF_MODEL_TYPES
                if use_hfllm_type:
                    logger.info(
                        f"Using {COSMOS_HF_MODEL_TYPES} for {model_name_or_path}"
                    )
                if hf_config.model_type in supported_model_types or use_hfllm_type:
                    try:
                        model = model_cls.from_pretrained(
                            hf_config,
                            model_name_or_path,
                            max_position_embeddings=config.policy.model_max_length,
                        )
                        model.use_hfllm_type = use_hfllm_type
                    except Exception as e:
                        logger.error(
                            f"Failed to load model {model_name_or_path} with error: {e}"
                        )
                        raise e
                    break
    if model is None:
        raise ValueError(f"Model {model_name_or_path} not supported.")
    return model
