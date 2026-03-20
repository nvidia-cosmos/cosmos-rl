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
import shutil
import threading
import numpy as np
from typing import Optional, Dict
from safetensors.torch import save_file
from huggingface_hub import (
    create_repo,
    upload_folder,
    whoami,
    split_torch_state_dict_into_shards,
)
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

        self.is_video = config.policy.diffusers.is_video
        self.is_lora = config.policy.lora is not None

        max_file_size_gb = 4 if not self.is_lora else float("inf")
        self.max_size_bytes = max_file_size_gb * 1024**3  # 4 GB in bytes

        # This model contains all part for a diffusers pipeline (transformers, vae, text_encoder)
        model = ModelRegistry.build_model(config)

        # Pre-save the pipeline for full pipeline export
        # The trainer only need to replace the transformer weights during training
        # TODO(dinghaoy): current diffusion RL only adopts FSDP, thus global_rank can be used to determine whether to save the pipeline
        # In the future, we will support other parallelism strategies, and need to modify this logic
        if (
            self.global_rank == 0
            and not self.is_lora
            and self.config.train.ckpt.enable_checkpoint
        ):
            model.pipeline.save_pretrained(
                os.path.join(
                    self.config.train.output_dir,
                    "safetensors",
                    "step_0",
                ),
                safe_serialization=True,
                max_shard_size=self.max_size_bytes,
            )

        # TODO(dinghaoy): Support low precision training
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
            self.model.ema = EMAModuleWrapper(
                parameters=self.model.trainable_params,
                decay=self.config.train.ema_decay,
                update_step_interval=self.config.train.ema_update_step_interval,
                device=self.device,
            )

        self.ckpt_manager = CheckpointMananger(
            self.config,
            self.parallel_dims,
            self.global_rank,
            hook_fns=kwargs.get("hook_fns", {}),
        )

        self.build_optimizers()
        self.lr_schedulers = None
        self.upload_thread = None

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

        def _save_state_dict_with_sharding(
            state_dict: Dict[str, torch.Tensor],
            save_path: str,
            max_shard_size: int,
        ):
            weights_name_pattern = "diffusion_pytorch_model{suffix}.safetensors"
            state_dict_split = split_torch_state_dict_into_shards(
                state_dict,
                max_shard_size=max_shard_size,
                filename_pattern=weights_name_pattern,
            )

            for filename, tensors in state_dict_split.filename_to_tensors.items():
                shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
                filepath = os.path.join(save_path, filename)
                save_file(shard, filepath, metadata={"format": "pt"})

            if state_dict_split.is_sharded:
                index = {
                    "metadata": state_dict_split.metadata,
                    "weight_map": state_dict_split.tensor_to_filename,
                }
                save_index_file = "diffusion_pytorch_model.safetensors.index.json"
                save_index_file = os.path.join(save_path, save_index_file)
                # Save the index as well
                with open(save_index_file, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
                logger.info(
                    f"[Policy] The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                    f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                    f"index located at {save_index_file}."
                )

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
                logger.info(f"[Policy] Exported LoRA adapter to {save_lora_path}")
            else:
                if trainable_only:
                    # Save trainable transformer weights
                    save_transformer_path = os.path.join(path, "transformer")
                    os.makedirs(save_transformer_path, exist_ok=True)
                    _save_state_dict_with_sharding(
                        transformer_state_to_save,
                        save_transformer_path,
                        max_size_bytes,
                    )
                    logger.info(
                        f"[Policy] Exported trainable transformer weights to {save_transformer_path}"
                    )
                else:
                    # Save complete diffusers pipeline
                    save_pipeline_path = path
                    os.makedirs(save_pipeline_path, exist_ok=True)
                    step_0_path = os.path.join(output_dir, "safetensors", "step_0")
                    if os.path.exists(step_0_path):
                        shutil.copytree(
                            step_0_path, save_pipeline_path, dirs_exist_ok=True
                        )
                        # Remove the `transformer` folder
                        shutil.rmtree(os.path.join(save_pipeline_path, "transformer"))
                    else:
                        logger.warning(
                            f"[Policy] Pipeline not found in the output directory: {step_0_path}. "
                            "Only saving the transformer weights."
                        )
                    os.makedirs(
                        os.path.join(save_pipeline_path, "transformer"), exist_ok=True
                    )
                    _save_state_dict_with_sharding(
                        transformer_state_to_save,
                        os.path.join(save_pipeline_path, "transformer"),
                        max_size_bytes,
                    )
                    # Copy the config of transformer
                    shutil.copy(
                        os.path.join(step_0_path, "transformer", "config.json"),
                        os.path.join(save_pipeline_path, "transformer", "config.json"),
                    )
                    logger.info(
                        f"[Policy] Exported full pipeline to {save_pipeline_path}"
                    )

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
                logger.info(
                    f"[Policy] Uploading the final model to huggingface: {repo_id}..."
                )
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
                        logger.info(
                            f"[Policy] Model uploaded to huggingface: {repo_id}"
                        )
                        success = True
                        break
                    except Exception as e:
                        logger.error(
                            f"[Policy] Failed to upload model to huggingface: {e}"
                        )
                        retry += 1
                if not success:
                    logger.error(
                        "[Policy] All retry attempts to upload model to huggingface failed."
                    )
                    raise RuntimeError(
                        f"[Policy] Failed to upload model to huggingface after {max_retries} attempts."
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
            logger.info(f"\n\n[Policy] Exported safetensors to {path}\n\n")

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
                    self.max_size_bytes,
                ),
                name="save_and_upload_safetensors",
                daemon=True,
            )
            self.upload_thread.start()

    def model_load_from_hf(self):
        # TODO (yy): meta init not support now
        logger.critical("[Policy] Meta init not supported for diffusers trainer.")
