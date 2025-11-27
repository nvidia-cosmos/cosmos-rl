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


from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.distributed import init_distributed
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.policy.worker.sft_worker import SFTPolicyWorker
from cosmos_rl.policy.config import Config as CosmosConfig
import torch
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.comm.base import WorkerBase


class LLMPolicyWorker(WorkerBase):
    """
    For LLM traing, we support SFT and GRPO. They are both implemented in disaggregated way.
    There are three main componets for LLM training:
    - Policy/Trainer
    - Rollout
    - Controller
    This class will delegate the trainer.
    """

    def __init__(self, **kwargs):
        # For LLM, config is retrieved from controller, so we pass None to the base class.
        # self.config will be set in worker_init.
        super().__init__(None, **kwargs)

    def worker_init(self, **kwargs):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        api_client = APIClient(role="POLICY")
        metadata = api_client.get_controller_metadata()

        if metadata["config"] is None:
            raise RuntimeError(
                f"[Policy] Please first go to http://{api_client.remote_ips}:{api_client.remote_port} to configure training parameters."
            )
        cosmos_config = CosmosConfig.from_dict(metadata["config"])
        self.config = cosmos_config

        logger.info(f"[Policy] Loaded configuration: {cosmos_config.model_dump()}")

        # Init distribution and build device mesh
        self.parallel_dims = ParallelDims.from_config(
            parallesim_config=cosmos_config.policy.parallelism
        )
        init_distributed()
        self.parallel_dims.build_mesh(device_type="cuda")

        # Build the trainer
        self.build_runner(**kwargs)

    def build_runner(self, **kwargs):
        policy_type = self.config.train.train_policy.type

        if policy_type == "grpo":
            self.runner = RLPolicyWorker(
                config=self.config,
                parallel_dims=self.parallel_dims,
                dataset=kwargs.get("dataset", None),
                data_packer=kwargs.get("data_packer", None),
                val_dataset=kwargs.get("val_dataset", None),
                val_data_packer=kwargs.get("val_data_packer", None),
            )
        elif policy_type == "sft":
            custom_sft_dataset = kwargs.get("dataset")
            custom_sft_data_packer = kwargs.get("data_packer")
            self.runner = SFTPolicyWorker(
                config=self.config,
                parallel_dims=self.parallel_dims,
                dataset=custom_sft_dataset,
                data_packer=custom_sft_data_packer,
                val_dataset=kwargs.get("val_dataset", None),
                val_data_packer=kwargs.get("val_data_packer", None),
                sampler=kwargs.get("sampler", None),
                batch_sampler=kwargs.get("batch_sampler", None),
                val_sampler=kwargs.get("val_sampler", None),
                val_batch_sampler=kwargs.get("val_batch_sampler", None),
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    def execute(self):
        """
        Execute the training.
        """
        # assert self.trainer is not None, "[Policy] Trainer has not been built."
        try:
            self.runner.execute()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.destroy_worker()

    def destroy_worker(self):
        if self.runner is not None:
            self.runner.destroy_worker()
            self.runner = None
        logger.info("[Policy] Process group destroyed.")
