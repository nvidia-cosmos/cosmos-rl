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
import json
import threading
from typing import Optional, Dict
from transformers import AutoConfig, GenerationConfig

from safetensors.torch import save_file
from huggingface_hub import create_repo, upload_folder, whoami
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from cosmos_rl.utils.checkpoint import upload_folder_to_s3

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.trainer.optm import build_optimizers
from cosmos_rl.policy.model import ModelRegistry
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
import cosmos_rl.utils.util as util
from cosmos_rl.utils.fp8.fp8_util import FP8ModelConverter
from cosmos_rl.policy.kernel.modeling_utils import init_flash_attn_meta
from cosmos_rl.utils.activation_offloading import get_act_offloading_ctx_manager
from cosmos_rl.policy.trainer.base import Trainer
from cosmos_rl.policy.trainer.base import extract_from_cuda_tensor, wrap_to_cuda_tensor
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from functools import partial


class LLMTrainer(Trainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        val_data_packer: BaseDataPacker,
        **kwargs,
    ):
        super(LLMTrainer, self).__init__(
            config,
            parallel_dims,
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

        init_flash_attn_meta(
            config.train.deterministic, config.train.compile, config.train.fa_version
        )

        self.hf_config = util.retry(AutoConfig.from_pretrained)(
            config.policy.model_name_or_path,
            trust_remote_code=True,
        )

        model = ModelRegistry.build_model(config)

        # FP8 settings
        with torch.device("meta"):
            if config.train.fp8.enable_fp8:
                self.model_converter = FP8ModelConverter(config, parallel_dims)
                self.model_converter.convert_model(model)
            elif config.train.fp4.enable_fp4:
                from cosmos_rl.utils.fp4.fp4_util import FP4ModelConverter

                self.model_converter = FP4ModelConverter(config, parallel_dims)
                self.model_converter.convert_model(model)

        if config.train.fsdp_offload:
            model._apply(
                lambda t: torch.empty_like(t, device="cpu")
                if t.device.type == "meta"
                else t.to("cpu"),
                recurse=True,
            )

        try:
            # Apply parallelism to the model
            parallelize_fn, _ = model.parallelize_fn
            # `pp_scheduler` is used for both `sft` and `RLHF`
            # `pp_scheduler_val` is used only for `sft`, since `RLHF` does not require policy model via validation
            self.pp_scheduler, self.pp_scheduler_val = parallelize_fn(
                model, parallel_dims, config, pp_loss_fn=self.pp_loss_fn
            )
            if not config.train.fsdp_offload:
                model._apply(
                    lambda t: torch.empty_like(t, device=self.device)
                    if t.device.type == "meta"
                    else t.to("cuda"),
                    recurse=True,
                )
            model.post_to_empty_hook(config)
            if config.policy.lora is not None:
                from cosmos_rl.policy.lora.plugin import reinitialize_lora_params

                reinitialize_lora_params(model)
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

        self.ckpt_manager = CheckpointMananger(
            self.config, self.parallel_dims, self.global_rank
        )
        # FIXME: (lms) use_streams=True could cause NaN in backward. Fix this later.
        self.act_offloading_ctx_manager = get_act_offloading_ctx_manager(
            self.model, config.train.activation_offload, use_streams=False
        )

        self.build_optimizers()

        self.seq_len_multiple = parallel_dims.cp * parallel_dims.tp
        self.lr_schedulers = None
        if self.config.train.fp8.enable_fp8 or self.config.train.fp4.enable_fp4:
            # Constraint of FP8/FP4 kernel(torch._scaled_mm): it requires K in MNK is mutiple of 16. In backward of Linear, to
            # calculate the gradient of weight, we have to round the seq_len_multiple to mutiple of 16.
            # See: https://github.com/pytorch/pytorch/blob/851a6fa82df251fbc368f0284d941ce78f68e7b1/aten/src/ATen/native/cuda/Blas.cpp#L1252
            self.seq_len_multiple = (self.seq_len_multiple + 16 - 1) // 16 * 16
            logger.info(
                "FP8/FP4 Training enabled, round seq_len_multiple to mutiple of 16."
            )
        logger.info(
            f"Trainer initialized at local rank {self.local_rank}, with seq_len_multiple: {self.seq_len_multiple}"
        )

    def build_optimizers(self):
        # TODO(cjx): add `CompiledAutograd` support
        self.optimizers = build_optimizers(self.model_parts, self.config)
        if self.config.train.fp8.enable_fp8 or self.config.train.fp4.enable_fp4:
            self.optimizers.register_step_post_hook(
                lambda *args, **kwargs: self.model_converter.post_optimizer_hook(
                    self.model_parts
                )
            )

    def sync_all_states(
        self,
        is_send: bool,
        send_hook: callable,
        recv_hook: callable,
        has_reference_model: bool = False,
    ) -> int:
        """
        Sync all states of the model and optimizer.
        Args:
            is_send (bool): Whether to send or receive the states.
            send_hook (callable): The hook function to send the states.
            recv_hook (callable): The hook function to receive the states.
            reference_model (bool): Whether to sync the reference model state dict.
        Returns:
            len_params (int): The number of parameters synced.
        """
        len_params = 0
        # It's a HFModel, we need to sync the named buffers
        state_dict = self.model.state_dict()
        model_state_dict = [state_dict]

        if has_reference_model:
            if len(self.reference_state_dict) == 0:
                assert (
                    not is_send
                ), "Reference model state dict should be populated before sending"
                for key, value in model_state_dict[0].items():
                    self.reference_state_dict[key] = torch.empty_like(
                        value, device="cpu"
                    )
            model_state_dict.append(self.reference_state_dict)
        model_state_dict[0].update(dict(self.model.named_buffers()))

        # 1. Sync all model states
        for state_to_sync in model_state_dict:
            for dest_name in sorted(state_to_sync.keys()):
                obj = state_to_sync[dest_name]
                assert isinstance(obj, torch.Tensor)
                local_view = wrap_to_cuda_tensor(
                    self.device, dest_name, obj, in_place=obj.is_cuda
                )
                if is_send:
                    send_hook(local_view)
                else:
                    recv_hook(local_view)
                    if isinstance(obj, torch.distributed.tensor.DTensor):
                        to_write = obj.to_local()
                    else:
                        to_write = obj

                    # Copy again for offloaded tensor since it is not inplace received
                    if not to_write.is_cuda:
                        to_write.copy_(local_view)
                len_params += 1

        # 2. Sync optimizer states
        optimizer_state = self.optimizers.state_dict()
        for dest_name in sorted(optimizer_state.keys()):
            obj = optimizer_state[dest_name]
            local_view = wrap_to_cuda_tensor(self.device, dest_name, obj)
            if local_view.data_ptr() is None:
                # skip the optimizer state if the data pointer is None
                continue
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                optimizer_state[dest_name] = extract_from_cuda_tensor(
                    self.device,
                    dest_name,
                    obj,
                    local_view,
                )
            len_params += 1

        if not is_send:
            self.optimizers.load_state_dict(optimizer_state)

        # 3. Sync lr_scheduler states
        if self.lr_schedulers is not None:
            lr_sheduler_state = self.lr_schedulers.state_dict()
            for dest_name in sorted(lr_sheduler_state.keys()):
                obj = lr_sheduler_state[dest_name]
                local_view = wrap_to_cuda_tensor(self.device, dest_name, obj)
                if is_send:
                    # nccl send
                    send_hook(local_view)
                else:
                    # nccl recv
                    recv_hook(local_view)
                    lr_sheduler_state[dest_name] = extract_from_cuda_tensor(
                        self.device,
                        dest_name,
                        obj,
                        local_view,
                    )
                len_params += 1
            if not is_send:
                self.lr_schedulers.load_state_dict(lr_sheduler_state)

        # 4. Sync rng_state
        rng_state = self.ckpt_manager.get_rng_state()
        for dest_name in sorted(rng_state.keys()):
            obj = rng_state[dest_name]
            local_view = wrap_to_cuda_tensor(self.device, dest_name, obj)
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                rng_state[dest_name] = extract_from_cuda_tensor(
                    self.device, dest_name, obj, local_view
                )
            len_params += 1
        if not is_send:
            self.ckpt_manager.set_rng_state(rng_state)
        return len_params

    def export_safetensors(
        self,
        output_dir: str,
        rel_path: str,
        trainable_only: bool = False,
        is_final=False,
        dtype: Optional[torch.dtype] = None,
    ):
        if self.config.policy.lora is not None and not trainable_only:
            trainable_only = True
            logger.info(
                "Exporting safetensors with param `trainable_only` is overridden to `True` for LoRA."
            )

        save_hf_config = self.config.policy.lora is None
        save_lora_config = self.config.policy.lora is not None
        save_generation_config = True
        export_weight_index_json = self.config.policy.lora is None

        path = os.path.join(output_dir, rel_path)
        if self.parallel_dims.dp_replicate_coord[0] > 0:
            return

        if self.global_rank == 0:
            logger.info(
                f"Prepare to exporting safetensors to {path} at rank {self.global_rank}"
            )

        if not self.parallel_dims.dp_replicate_enabled:
            torch.distributed.barrier()

        def get_tensor_size(tensor):
            """Get the size of the tensor in bytes."""
            return tensor.element_size() * tensor.numel()

        max_file_size_gb = 4 if not save_lora_config else float("inf")
        max_size_bytes = max_file_size_gb * 1024**3  # 4 GB in bytes
        current_chunk = {}
        total_chunk_size = 0
        current_chunk_size = 0
        file_idx = 0
        manifest = {}  # Record the weight->file name mapping

        def create_file_name(save_lora_config, pp_rank, pp_size, file_idx):
            if save_lora_config:
                name = "adapter_model.safetensors"
            else:
                if pp_size == 1:
                    name = f"{file_idx:05d}.safetensors"
                else:
                    name = f"model-{pp_rank}-of-{pp_size}-{file_idx:05d}.safetensors"
            return os.path.join(path, name)

        def save_chunked_tensors(
            chunk: Dict[str, torch.Tensor], chunk_size: int, file_path: str
        ):
            """
            Save a dictionary of tensors into a safetensors file.
            Only the rank 0 of dp_shard, cp, tp will save the file.
            """
            nonlocal total_chunk_size
            if (
                self.parallel_dims.dp_shard_coord[0] == 0
                and self.parallel_dims.cp_coord[0] == 0
                and self.parallel_dims.tp_coord[0] == 0
            ):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                for name, param in chunk.items():
                    manifest[name] = os.path.basename(file_path)
                total_chunk_size += chunk_size
                save_file(chunk, file_path)
                logger.info(f"Saved chunk {file_idx} to {os.path.basename(file_path)}")

        for name, param in self.model.named_parameters():
            # First map the key from local to hf naming convention
            name = self.model.weight_mapper.policy_map_local_key_to_hf_key(name)
            if trainable_only and not param.requires_grad:
                continue
            is_dtensor = isinstance(param, torch.distributed.tensor.DTensor)
            param = param.full_tensor() if is_dtensor else param
            param = param.detach().data

            pp_rank, pp_size = self.parallel_dims.pp_coord

            for (
                _name,
                _param,
            ) in self.model.weight_mapper.policy_maybe_decompose_weights_to_hf_naming(
                name, param
            ):
                if _param is None:
                    logger.debug(
                        f"[Policy] Skipping None parameter for {name} in safetensors export."
                    )
                    continue
                elif save_lora_config and not _name.startswith("base_model"):
                    # LoRA model needs to add a prefix to the weight name to be consistent with the HF naming convention
                    _name = f"base_model.model.{_name}"

                _param = _param.to(dtype=dtype) if dtype is not None else _param
                tensor_size = get_tensor_size(_param)
                # If adding the current tensor exceeds the size limit, save the current chunk
                if current_chunk_size + tensor_size > max_size_bytes:
                    # Save the current chunk as a safetensor file
                    file_name = create_file_name(
                        save_lora_config, pp_rank, pp_size, file_idx
                    )
                    save_chunked_tensors(current_chunk, current_chunk_size, file_name)

                    # Reset for the next chunk
                    current_chunk = {_name: _param}
                    current_chunk_size = tensor_size
                    file_idx += 1
                else:
                    # Add tensor to the current chunk
                    current_chunk[_name] = _param
                    current_chunk_size += tensor_size

        # Save any remaining tensors in the last chunk
        if current_chunk:
            file_name = create_file_name(save_lora_config, pp_rank, pp_size, file_idx)
            save_chunked_tensors(current_chunk, current_chunk_size, file_name)

        # Allgather the manifest from all pipeline stages
        if self.parallel_dims.pp_enabled:
            pp_group = self.parallel_dims.mesh["pp"].get_group()
            pp_size = self.parallel_dims.pp
            output = [None for _ in range(pp_size)]
            tensor_sizes = [0 for _ in range(pp_size)]
            torch.distributed.all_gather_object(output, manifest, group=pp_group)
            torch.distributed.all_gather_object(
                tensor_sizes, total_chunk_size, group=pp_group
            )
            merged_manifest = {}
            for m in output:
                merged_manifest.update(m)
            total_tensor_size = sum(tensor_sizes)
        else:
            merged_manifest = manifest
            total_tensor_size = total_chunk_size

        if not self.parallel_dims.dp_replicate_enabled:
            torch.distributed.barrier()

        def upload_handler(config, is_final, path, rel_path, max_retries=3):
            """Handle the upload of the model to huggingface and s3."""
            # upload the final model to huggingface
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
            # upload the model to s3
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
            if export_weight_index_json:
                with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
                    json.dump(
                        {
                            "metadata": {
                                "total_size": total_tensor_size,
                            },
                            "weight_map": merged_manifest,
                        },
                        f,
                        indent=4,
                    )

            # save hf_config
            if save_hf_config:
                self.hf_config.save_pretrained(path)
            if save_lora_config:
                # Save the LoRA config
                lora_config = self.config.policy.lora.model_dump(mode="json")
                lora_config["base_model_name_or_path"] = (
                    self.config.policy.model_name_or_path
                )
                lora_config["peft_type"] = "LORA"
                with open(os.path.join(path, "adapter_config.json"), "w") as f:
                    json.dump(lora_config, f, indent=4)

            self.data_packer.save_state(path)

            # save the generation config to get the generation aligned with HF.
            try:
                if save_generation_config:
                    generation_config = util.retry(GenerationConfig.from_pretrained)(
                        self.config.policy.model_name_or_path
                    )
                    generation_config.save_pretrained(path)
            except Exception:
                logger.warning("[Policy] No generation config found, do not save it.")

            need_upload = (
                self.config.train.ckpt.upload_hf and is_final
            ) or self.config.train.ckpt.upload_s3
            if need_upload:
                # If the upload thread is already running, wait for it to finish
                if self.upload_thread is not None:
                    self.upload_thread.join()
                self.upload_thread = threading.Thread(
                    target=upload_handler,
                    args=(self.config, is_final, path, rel_path),
                    name="upload_safetensors",
                    daemon=True,
                )
                self.upload_thread.start()

    def model_load_from_hf(self):
        self.model.load_hf_weights(
            self.config.policy.model_name_or_path,
            self.parallel_dims,
            self.device,
            revision=self.config.policy.model_revision,
        )

    def model_resume_from_checkpoint(self):
        ckpt_extra_vars, self.lr_schedulers = self.ckpt_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=partial(build_lr_schedulers, self.optimizers, self.config),
            model_name_or_path=self.config.policy.model_name_or_path,
            revision=self.config.policy.model_revision,
        )
        return ckpt_extra_vars

    def step_validation(self):
        # By default, add an empty step_validation for LLM trainer.
        # For GRPO, validation is handled in rollout side, so this method is not needed.
        # For SFT trainer, we will override this method in sft_trainer.py.
        pass
