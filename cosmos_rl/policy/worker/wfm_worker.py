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


from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig
from cosmos_rl.comm.base import WorkerBase
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.constant import CACHE_DIR
from cosmos_rl.utils.s3_utils import set_s3_available
from cosmos_rl.utils.util import create_cached_dir_if_needed
from cosmos_rl.policy.config.wfm import CosmosVisionGenConfig as Config
from cosmos_rl.policy.trainer.wfm_trainer import CosmosVisionGenTrainer
from cosmos_rl.utils.wfm.io import dump_dict_to_file
from cosmos_rl.utils.wfm.io.cred_env_parser import CRED_ENVS, CRED_ENVS_DICT
from cosmos_rl.utils.wfm.io.easy_io import easy_io
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed


def init_s3(config: Config):
    try:
        # Check whether set the credential env vars, if so, dump them to a file.
        if str(CRED_ENVS):
            logger.info("Detect credential env vars, dumping them to files...")
            # Get the credentials from environment variables
            s3_checkpoint_cred = CRED_ENVS_DICT["PROD_S3_CHECKPOINT"]
            team_dir_cred = CRED_ENVS_DICT["PROD_TEAM_DIR"]
            # Create the credential files
            create_cached_dir_if_needed()
            os.makedirs(os.path.join(CACHE_DIR, "credentials"), exist_ok=True)
            s3_checkpoint_path = os.path.join(
                CACHE_DIR, "credentials", "s3_checkpoint.secret"
            )
            s3_training_path = os.path.join(
                CACHE_DIR, "credentials", "s3_training.secret"
            )
            pbss_dir_path = os.path.join(CACHE_DIR, "credentials", "pbss_dir.secret")
            # Set the env vars to point to the credential files
            os.environ["S3_CHECKPOINT_CREDENTIAL_PATH"] = s3_checkpoint_path
            os.environ["S3_TRAINING_CREDENTIAL_PATH"] = s3_training_path
            os.environ["PBSS_DIR_CREDENTIAL_PATH"] = pbss_dir_path
            # Dump the credentials to the files
            if not os.path.exists(s3_checkpoint_path):
                dump_dict_to_file(s3_checkpoint_path, s3_checkpoint_cred)
            if not os.path.exists(s3_training_path):
                dump_dict_to_file(s3_training_path, s3_checkpoint_cred)
            if not os.path.exists(pbss_dir_path):
                dump_dict_to_file(pbss_dir_path, team_dir_cred)
            logger.info(
                f"Dumped S3 credentials to {os.path.join(CACHE_DIR, 'credentials')}"
            )
            # Update the config to point to the credential files
            config.checkpoint.save_to_object_store.credentials = s3_checkpoint_path
            config.checkpoint.load_from_object_store.credentials = s3_checkpoint_path
            if config.model.text_encoder_config is not None:
                config.model.text_encoder_config.encoder_model_config.s3_credential_path = pbss_dir_path

        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": {
                    "s3://timestamps_rundir/": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/job_runs/{config.job.timestamp}/",
                    "s3://rundir/": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/",
                },
                "s3_credential_path": config.checkpoint.save_to_object_store.credentials,
            }
        )
        set_s3_available(True)
        logger.info("S3 backend initialized successfully.")
    except Exception as e:
        set_s3_available(False)
        logger.error(f"Failed to initialize S3 backend: {e}")


class WFMPolicyWorker(WorkerBase):
    def __init__(self, config: CosmosVisionGenConfig, **kwargs):
        super().__init__(config)
        # Override remote reward config with environment variables if provided
        if os.environ.get("WFM_REWARD_TOKEN", ""):
            self.config.model.rl.reward_config.remote_reward.token = os.environ.get(
                "WFM_REWARD_TOKEN", ""
            )
            logger.info("Using token from environment variable")
        if os.environ.get("WFM_REWARD_ENQUEUE_URL", ""):
            self.config.model.rl.reward_config.remote_reward.enqueue_url = (
                os.environ.get("WFM_REWARD_ENQUEUE_URL", "")
            )
            logger.info("Using enqueue_url from environment variable")
        if os.environ.get("WFM_REWARD_FETCH_URL", ""):
            self.config.model.rl.reward_config.remote_reward.fetch_url = os.environ.get(
                "WFM_REWARD_FETCH_URL", ""
            )
            logger.info("Using fetch_url from environment variable")

        logger.info(f"Successfully loaded configuration: {self.config.model_dump()}")

        # init dist
        init_distributed()

        logger.info(f"Local rank: {os.environ.get('LOCAL_RANK')}")

        # init s3 if needed
        if (
            config.checkpoint.load_from_object_store.enabled
            or config.checkpoint.save_to_object_store.enabled
            or config.trainer.use_s3
        ):
            init_s3(self.config)

        # build runner
        self.build_runner(**kwargs)

    def build_runner(self, **kwargs):
        # For wfm, parallel_dims is instantiated in the trainer now.
        self.trainer = CosmosVisionGenTrainer(self.config, None, **kwargs)

    def execute(self):
        assert self.trainer is not None, "[Policy] Trainer has not been built."
        try:
            self.trainer.main_loop()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.destroy_worker()

    def destroy_worker(self):
        destroy_distributed()
        logger.info("[Policy] Process group destroyed.")
