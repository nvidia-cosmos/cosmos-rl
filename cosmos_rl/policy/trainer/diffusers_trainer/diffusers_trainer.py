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
import random
import numpy as np
from typing import Optional
from safetensors.torch import save_file
from peft import get_peft_model_state_dict

from cosmos_rl.policy.trainer.base import Trainer
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model import ModelRegistry
from cosmos_rl.policy.trainer.optm import build_optimizers
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.utils.ema import EMAModuleWrapper
from cosmos_rl.utils.logging import logger
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

        def _materialize_tensor_for_export(
            tensor: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            is_dtensor = isinstance(tensor, torch.distributed.tensor.DTensor)
            tensor = tensor.full_tensor() if is_dtensor else tensor
            if self.global_rank != 0:
                return None
            tensor = tensor.detach()
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            return tensor.cpu()

        if self.is_lora:
            # Save lora weight & config for peft model
            save_lora_path = os.path.join(output_dir, rel_path, "lora")
            lora_state_dict = get_peft_model_state_dict(self.model.transformer)
            if self.global_rank == 0:
                lora_state_dict_to_save = {}
            for name, param in lora_state_dict.items():
                tensor_to_save = _materialize_tensor_for_export(param)
                if self.global_rank == 0:
                    lora_state_dict_to_save[name] = tensor_to_save
            if self.global_rank == 0:
                os.makedirs(save_lora_path, exist_ok=True)
                save_file(
                    lora_state_dict_to_save,
                    os.path.join(save_lora_path, "model.safetensors"),
                )
                logger.info(f"Exported LoRA adapter to {save_lora_path}")
        else:
            if trainable_only:
                transformer_items = (
                    (name, param)
                    for name, param in self.model.transformer.named_parameters()
                    if param.requires_grad
                )
            else:
                transformer_items = self.model.transformer.state_dict().items()

            transformer_state_to_save = {}
            for name, tensor in transformer_items:
                tensor_to_save = _materialize_tensor_for_export(tensor)
                if self.global_rank == 0:
                    transformer_state_to_save[name] = tensor_to_save

            if self.global_rank == 0:
                if trainable_only:
                    # Save trainable transformer weights
                    save_transformer_path = os.path.join(
                        output_dir, rel_path, "transformer"
                    )
                    os.makedirs(save_transformer_path, exist_ok=True)
                    save_file(
                        transformer_state_to_save,
                        os.path.join(save_transformer_path, "model.safetensors"),
                    )
                    logger.info(
                        f"Exported trainable transformer weights to {save_transformer_path}"
                    )
                else:
                    # Save complete diffusers pipeline
                    save_pipeline_path = os.path.join(output_dir, rel_path)
                    os.makedirs(save_pipeline_path, exist_ok=True)
                    self.model.pipeline.save_pretrained(
                        save_pipeline_path, safe_serialization=True
                    )
                    self.model.transformer.save_pretrained(
                        os.path.join(save_pipeline_path, "transformer"),
                        state_dict=transformer_state_to_save,
                        safe_serialization=True,
                    )
                    logger.info(f"Exported full pipeline to {save_pipeline_path}")

    def model_load_from_hf(self):
        # TODO (yy): meta init not support now
        logger.critical("[Policy] Meta init not supported for diffusers trainer.")
