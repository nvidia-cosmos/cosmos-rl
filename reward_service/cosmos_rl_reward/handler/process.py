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
from cosmos_rl_reward.utils.logging import logger
from cosmos_rl_reward.utils.shmem import CrossProcessMaster
import subprocess
import sys
from cosmos_rl_reward.handler.registry import RewardRegistry
import cosmos_rl_reward.model  # noqa: F401 to register all reward models


class RewardProcessHandler:
    def __init__(self, reward_name, model_path, dtype, device, download_path):
        """
        Initialize the RewardProcessHandler with necessary attributes for launching the reward process.
        Args:
            reward_name (str): The name of the reward handler.
            model_path (str): The path to the main model used for rewarding.
            dtype (str): The data type for model.
            device (str): The device to run the model on, e.g., "cuda"
            download_path (str): The path to download or load related files including models.
        """
        self.reward_name = reward_name
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.download_path = download_path

    def init_process(
        self, cuda_maxbytes=1024 * 1024 * 512, cpu_maxbytes=1024 * 1024, envs={}
    ):
        """
        Initialize the reward process with shared memory.
        Launch the reward process as a subprocess with the appropriate environment and command.
        Use the CrossProcessMaster to set up shared memory for communication with the decoding process.

        Args:
            cuda_maxbytes (int): Maximum bytes for CUDA memory. Default is 512MB.
            cpu_maxbytes (int): Maximum bytes for CPU memory. Default is 1MB.
            envs (dict): Dictionary of environment variables for the process.
        Raises:
            ValueError: If the reward name is not recognized.
        """
        logger.info(f"Initializing reward process for {self.reward_name}...")
        self.process_handler = CrossProcessMaster(
            name=self.reward_name,
            cuda_maxbytes=cuda_maxbytes,
            cpu_maxbytes=cpu_maxbytes,
        )
        self.process_handler.init_memory()
        cur_envs = os.environ.copy()
        cur_envs.update(envs)
        venv_python_path = RewardRegistry.get_reward_venv(self.reward_name)
        check_cmd = [
            venv_python_path,
            "-c",
            'import cosmos_rl_reward; print(f\'cosmos_rl_reward found, version: {getattr(cosmos_rl_reward, "__version__", "unknown")}\')',
        ]
        try:
            subprocess.run(
                check_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            cmd = [
                venv_python_path,
                "-m",
                "cosmos_rl_reward.launcher.reward_compute",
            ]
        except subprocess.CalledProcessError as e:
            logger.debug(
                f"Error checking 'cosmos_rl_reward' in venv {venv_python_path}: {e.stderr}"
            )
            cmd = [
                venv_python_path,
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../launcher/reward_compute.py",
                ),
            ]
        cmd.extend(
            [
                "--shm",
                self.process_handler.shm_cpu.name,
                "--reward",
                self.reward_name,
                "--model_path",
                self.model_path,
                "--download_path",
                self.download_path,
                "--dtype",
                str(self.dtype),
                "--device",
                str(self.device),
            ]
        )
        logger.info(f"Starting reward process with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            env=cur_envs,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        logger.info(f"Reward process {self.reward_name} initialized.")
