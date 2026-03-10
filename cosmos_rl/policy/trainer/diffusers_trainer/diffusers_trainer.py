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

import torch
import os
import json
import random
import threading
import numpy as np
from typing import Optional, Dict
from safetensors.torch import save_file
from huggingface_hub import create_repo, upload_folder, whoami
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from peft import get_peft_model_state_dict

from cosmos_rl.policy.trainer.base import Trainer
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model import ModelRegistry
from cosmos_rl.policy.trainer.optm import build_optimizers
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.ema import EMAModuleWrapper
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.s3_utils import upload_folder_to_s3
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker


class DiffusersTrainer(Trainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: Optional[torch.cuda.Stream] = None,
        data_packer: BaseDataPacker = None,
        val_data_packer: BaseDataPacker = None,
        **kwargs,
    ):
        super(DiffusersTrainer, self).__init__(
            config=config,
            parallel_dims=parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            **kwargs,
        )

        if config.train.seed:
            torch.manual_seed(config.train.seed)
            torch.cuda.manual_seed(config.train.seed)
            torch.cuda.manual_seed_all(config.train.seed)
            random.seed(config.train.seed)
            np.random.seed(config.train.seed)

        if config.train.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(mode=True, warn_only=True)

        # This model contains all part for a diffusers pipeline (transformers, vae, text_encoder)
        model = ModelRegistry.build_model(config)

        # Add low precision support

        try:
            # Apply parallelism to the model
            parallelize_fn, _ = model.parallelize_fn
            # `pp_scheduler` is used for both `sft` and `RLHF`
            # `pp_scheduler_val` is used only for `sft`, since `RLHF` does not require policy model via validation
            self.pp_scheduler, self.pp_scheduler_val = parallelize_fn(
                model, parallel_dims, config, pp_loss_fn=None
            )
            # Enable gradient checkpointing for the model
            model.set_gradient_checkpointing_enabled(
                config.policy.model_gradient_checkpointing
            )

            torch.cuda.empty_cache()
            self.model_parts = model.separate_model_parts()
            self.model = model
            # util.add_nan_checks(model)
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e

        self.is_video = config.policy.diffusers.is_video
        self.is_lora = config.policy.lora is not None

        # Create ema if needed
        if self.config.train.ema_enable:
            if self.is_lora:
                adapter_name = self.model.transformer.active_adapters()
                logger.info(
                    f"Creating EMA for adapter {adapter_name} with decay {self.config.train.ema_decay} and update step interval {self.config.train.ema_update_step_interval}"
                )
            else:
                logger.info(
                    f"Creating EMA for model with decay {self.config.train.ema_decay} and update step interval {self.config.train.ema_update_step_interval}"
                )
            self.ema = EMAModuleWrapper(
                parameters=self.model.trainable_params,
                decay=self.config.train.ema_decay,
                update_step_interval=self.config.train.ema_update_step_interval,
                device=self.device,
            )

        self.ckpt_manager = CheckpointMananger(
            self.config, self.parallel_dims, self.global_rank
        )

        self.build_optimizers()
        self.lr_schedulers = None

    def build_optimizers(self):
        # TODO (yy): Add low precision support
        self.optimizers = build_optimizers(self.model.trained_model, self.config)

    def build_lr_schedulers(self):
        pass

    def step_training(self):
        pass

    def step_validation(self):
        pass

    def export_safetensors(
        self,
        output_dir: str,
        rel_path: str,
        trainable_only: bool = False,
        is_final=False,
        dtype: Optional[torch.dtype] = None,
    ):
        """Export safetensors to the local directory/huggingface/s3.
        Args:
            output_dir (str): The directory to save the safetensors.
            rel_path (str): The relative path to the output directory.
            trainable_only (bool): Whether to export only the trainable parameters.
            is_final (bool): Whether this is the final export.
            dtype (torch.dtype): The dtype of the parameters.
        """
        if self.is_lora and not trainable_only:
            trainable_only = True
            logger.info(
                "Exporting safetensors with param `trainable_only` is overridden to `True` for LoRA."
            )

        max_file_size_gb = 4 if not self.is_lora else float("inf")
        max_size_bytes = max_file_size_gb * 1024**3  # 4 GB in bytes
        transformer_state_to_save = {}

        def _materialize_tensor_for_export(
            tensor: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            is_dtensor = isinstance(tensor, torch.distributed.tensor.DTensor)
            tensor = tensor.full_tensor() if is_dtensor else tensor
            if self.global_rank != 0:
                return None
            tensor = tensor.detach()
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            return tensor.cpu()

        if self.is_lora:
            lora_state_dict = get_peft_model_state_dict(self.model.transformer)
            for name, param in lora_state_dict.items():
                tensor_to_save = _materialize_tensor_for_export(param)
                if self.global_rank == 0:
                    transformer_state_to_save[name] = tensor_to_save
        else:
            if trainable_only:
                transformer_items = (
                    (name, param)
                    for name, param in self.model.transformer.named_parameters()
                    if param.requires_grad
                )
            else:
                transformer_items = self.model.transformer.state_dict().items()

            for name, tensor in transformer_items:
                tensor_to_save = _materialize_tensor_for_export(tensor)
                if self.global_rank == 0:
                    transformer_state_to_save[name] = tensor_to_save

        def save_and_upload_handler(
            output_dir: str,
            rel_path: str,
            trainable_only: bool,
            is_final: bool,
            config: CosmosConfig,
            is_lora: bool,
            transformer_state_to_save: Dict[str, torch.Tensor],
            max_size_bytes: int,
            max_retries: int = 3,
        ):
            path = os.path.join(output_dir, rel_path)
            # Save the weights to local
            if is_lora:
                # Save lora weight
                save_lora_path = os.path.join(path, "lora")
                os.makedirs(save_lora_path, exist_ok=True)
                save_file(
                    transformer_state_to_save,
                    os.path.join(save_lora_path, "model.safetensors"),
                )
                # Save the LoRA config
                lora_config = self.config.policy.lora.model_dump(mode="json")
                lora_config["base_model_name_or_path"] = (
                    self.config.policy.model_name_or_path
                )
                lora_config["peft_type"] = "LORA"
                with open(
                    os.path.join(save_lora_path, "adapter_config.json"), "w"
                ) as f:
                    json.dump(lora_config, f, indent=4)
                logger.info(f"Exported LoRA adapter to {save_lora_path}")
            else:
                if trainable_only:
                    # Save trainable transformer weights
                    save_transformer_path = os.path.join(path, "transformer")
                    os.makedirs(save_transformer_path, exist_ok=True)
                    self.model.transformer.save_pretrained(
                        save_transformer_path,
                        state_dict=transformer_state_to_save,
                        safe_serialization=True,
                        max_shard_size=max_size_bytes,
                    )
                    logger.info(
                        f"Exported trainable transformer weights to {save_transformer_path}"
                    )
                else:
                    # Save complete diffusers pipeline
                    save_pipeline_path = path
                    os.makedirs(save_pipeline_path, exist_ok=True)
                    self.model.pipeline.save_pretrained(
                        save_pipeline_path,
                        safe_serialization=True,
                        max_shard_size=max_size_bytes,
                    )
                    self.model.transformer.save_pretrained(
                        os.path.join(save_pipeline_path, "transformer"),
                        state_dict=transformer_state_to_save,
                        safe_serialization=True,
                        max_shard_size=max_size_bytes,
                    )
                    logger.info(f"Exported full pipeline to {save_pipeline_path}")

            # Upload the weights to huggingface
            if config.train.ckpt.upload_hf and is_final:
                username = whoami()["name"]
                repo_id = (
                    username
                    + "/"
                    + config.train.ckpt.hf_repo_name
                    + "-"
                    + config.train.timestamp
                )
                logger.info(f"Uploading the final model to huggingface: {repo_id}...")
                retry = 0
                success = False
                while retry < max_retries:
                    try:
                        create_repo(repo_id, exist_ok=True)
                        # hide redundant logs of huggingface
                        disable_progress_bars()
                        upload_folder(
                            folder_path=path,
                            path_in_repo=".",
                            repo_id=repo_id,
                            commit_message="Upload model",
                        )
                        enable_progress_bars()
                        logger.info(f"Model uploaded to huggingface: {repo_id}")
                        success = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to upload model to huggingface: {e}")
                        retry += 1
                if not success:
                    logger.error(
                        "All retry attempts to upload model to huggingface failed."
                    )
                    raise RuntimeError(
                        f"Failed to upload model to huggingface after {max_retries} attempts."
                    )

            # Upload the weights to s3
            if config.train.ckpt.upload_s3:
                if is_final:
                    # syncronizely upload the final model to s3
                    upload_folder_to_s3(
                        path,
                        config.train.ckpt.s3_bucket,
                        os.path.join(config.train.ckpt.s3_prefix, rel_path),
                    )
                elif config.train.ckpt.upload_s3 == "all":
                    # asynchronously upload the model to s3
                    upload_folder_to_s3(
                        path,
                        config.train.ckpt.s3_bucket,
                        os.path.join(config.train.ckpt.s3_prefix, rel_path),
                    )
            logger.info(f"\n\nExported safetensors to {path}\n\n")

        if self.global_rank == 0:
            # If the upload thread is already running, wait for it to finish
            if self.upload_thread is not None:
                self.upload_thread.join()
            self.upload_thread = threading.Thread(
                target=save_and_upload_handler,
                args=(
                    output_dir,
                    rel_path,
                    trainable_only,
                    is_final,
                    self.config,
                    self.is_lora,
                    transformer_state_to_save,
                    max_size_bytes,
                ),
                name="save_and_upload_safetensors",
                daemon=True,
            )
            self.upload_thread.start()

    def model_load_from_hf(self):
        # TODO (yy): meta init not support now
        logger.critical("[Policy] Meta init not supported for diffusers trainer.")
