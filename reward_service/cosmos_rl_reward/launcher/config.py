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

from pydantic import BaseModel, Field, model_validator
from typing import Any


class RewardArgs(BaseModel):
    """
    Configuration arguments for running one type of reward model.
    """

    reward_type: str = Field(
        default="cosmos_reason1", description="Type of the reward model"
    )
    venv_python: str = Field(
        default="python",
        description="Path to the Python interpreter in the virtual environment",
    )
    model_path: str = Field(
        default="nvidia/Cosmos-Reason1-7B-Reward",
        description="Path to the reward model, it will be a subpath inside `download_path` for the case where model files need to be downloaded explicitly",
    )
    download_path: str = Field(
        default="/workspace",
        description="Local path to download the related model files if applicable",
    )
    setup_script: str = Field(
        default="",
        description="Path to the setup script for the reward model environment",
    )

    dtype: str = Field(default="float16", description="Data type for model inference")
    work_dir: str = Field(
        default="evaluation_results",
        description="Working directory for reward processing",
    )
    tmp_dir: str = Field(
        default="/tmp", description="Temporary directory for intermediate files"
    )
    device: str = Field(
        default="cuda",
        description="Device to run the reward model on, e.g., 'cuda' or 'cpu'",
    )
    enable: bool = Field(default=True, description="Whether to enable this reward")


class DecodeArgs(BaseModel):
    """
    Configuration arguments for latent decoding using a VAE model.
    """

    model_path: str = Field(
        default="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth",
        description="Path to the VAE model for latent decoding",
    )
    dtype: str = Field(default="float16", description="Data type for VAE inference")
    device: str = Field(
        default="cuda",
        description="Device to run the VAE model on, e.g., 'cuda' or 'cpu'",
    )
    chunk_duration: int = Field(
        default=81, description="Duration of each video chunk in seconds"
    )
    temporal_window: int = Field(
        default=16, description="Temporal window size for decoding"
    )
    load_mean_std: bool = Field(
        default=False, description="Whether to load mean and std for normalization"
    )


class Config(BaseModel):
    """
    Configuration for the whole reward service.
    """

    no_latent_decoder: bool = Field(
        default=False,
        description="If true, do not initialize the latent decoder (image-only mode).",
    )
    
    host: str = Field(default="localhost", description="Host address for the launcher")
    port: int = Field(default=8080, description="Port number for the launcher")

    redis_host: str = Field(
        default="localhost", description="Redis server host for the launcher"
    )
    redis_port: int = Field(
        default=6379, description="Redis server port for the launcher"
    )

    reward_args: list[RewardArgs] = Field(
        default_factory=lambda: [RewardArgs()],
        description="List of reward model arguments.",
    )

    decode_args: DecodeArgs = Field(
        default_factory=DecodeArgs,
        description="Arguments for latent decoding.",
    )

    @classmethod
    def from_dict(cls, config_data: dict[str, Any]) -> "Config":
        config = cls.model_validate(config_data)
        return config

    @model_validator(mode="before")
    def preprocess(cls, data: dict) -> dict:
        return data

    @model_validator(mode="after")
    def check_params_value(self):
        return self
