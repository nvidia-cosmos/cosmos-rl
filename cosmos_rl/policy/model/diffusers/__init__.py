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

from typing import List, Tuple, Optional

import torch
from torch import nn

from abc import ABC, abstractmethod
from diffusers import training_utils
from diffusers.loaders.peft import PeftAdapterMixin
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import DiffusersConfig
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.config import LoraConfig as CosmosLoraConfig
from cosmos_rl.utils.util import str2torch_dtype
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.diffusers_utils import DiffusionPipeline


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


@ModelRegistry.register(DiffuserModelWeightMapper)
class DiffuserModel(BaseModel, ABC):
    @staticmethod
    def supported_model_types():
        return ["diffusers"]

    def __init__(
        self,
        config: DiffusersConfig,
        lora_config: CosmosLoraConfig = None,
        model_str: str = "",
        model_revision: str = "main",
    ):
        super().__init__()
        self.config = config
        self.offload = self.config.offload
        self.dtype = str2torch_dtype(config.dtype)
        self.load_models_from_hf(model_str, model_revision)
        if lora_config is not None:
            self.is_lora = True
            self.apply_lora(lora_config)
        else:
            self.is_lora = False
        # Decide timesampling method
        self.weighting_scheme = self.config.weighting_scheme
        self.train_sampling_steps = self.scheduler.config.num_train_timesteps
        self.init_output_process()

    def register_models(self):
        self.valid_models = [
            k
            for k, v in self.pipeline._internal_dict.items()
            if isinstance(v, tuple) and v[0] is not None
        ]
        self.model_parts = []
        self.offloaded_models = []
        for valid_model in self.valid_models:
            model_part = getattr(self.pipeline, valid_model)
            if isinstance(model_part, nn.Module) and valid_model != "transformer":
                # Offload all torch.nn.Modules to cpu except transformers
                model_part.to(dtype=torch.bfloat16)
                if self.offload:
                    model_part.to("cpu")
                    self.offloaded_models.append(model_part)
            setattr(self, valid_model, model_part)
            self.model_parts.append((valid_model, model_part))

    def current_device(self):
        return next(self.transformer.parameters()).device

    def set_gradient_checkpointing_enabled(self, enabled: bool):
        """
        Diffusers's transformer model support gradient checkpointing, just need to set it to True
        """
        super().set_gradient_checkpointing_enabled(enabled)
        if (
            hasattr(self.transformer, "_supports_gradient_checkpointing")
            and self.transformer._supports_gradient_checkpointing
            and enabled
        ):
            self.transformer.enable_gradient_checkpointing()

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pass

    def apply_pipeline_split(self, pp_rank, pp_size):
        """
        Apply pipeline split to the model.
        This typically involves splitting the model into multiple stages,
        and moving each stage to a different device.
        """
        assert False, "Pipeline split is not supported for DiffusersModel"

    def separate_model_parts(self):
        """
        Return different model parts, diffusers usually contain transformer, vae_model and text_encoder
        """
        return self.model_parts

    @property
    def trainable_params(self):
        # Get all trainable parameters
        return [
            params for params in self.transformer.parameters() if params.requires_grad
        ]

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_name_or_path (str): The name or path of the model.
            parallel_dims (ParallelDims): The parallel dimensions.
            device (torch.device): The device to load the weights.
        """
        pass

    def init_output_process(self):
        """
        For inference, output processer is needed to transfer latents back to visual output
        """
        # For diffusers, pipeline will have video_processor for video and image_processor for image model
        if hasattr(self.pipeline, "image_processor"):
            self.is_video = False
            self.visual_processor_fn = self.pipeline.image_processor.preprocess
        elif hasattr(self.pipeline, "video_processor"):
            self.is_video = True
            self.visual_processor_fn = self.pipeline.video_processor.preprocess_video
        else:
            raise ValueError(
                f"{self.model_str} have neither video_processor or image_processor, may not be a valid pipeline"
            )

    def load_models_from_hf(self, model_str: str, revision: str = "main"):
        """
        Load all models

        Args:
            model_str (str): The name or path of the diffusers pipeline.
        """
        # Init from pipeline
        self.model_str = model_str
        # Always init on cuda now
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_str, revision=revision, torch_dtype=self.dtype, device_map="cuda"
        )
        if self.pipeline._execution_device.type != "cuda":
            logger.warning(
                f"{model_str} pipeline does not support cuda device map. Manually move it to cuda."
            )
            self.pipeline.reset_device_map()
            self.pipeline.to("cuda")
        # Register all model parts to self
        # self.transformer will point to self.pipeline.transformer
        self.register_models()

    @classmethod
    def from_pretrained(
        cls, config: CosmosConfig, model_str: str, model_revision: str = "main"
    ):
        """
        Model initialize entrypoint.
        """
        return cls(
            config.policy.diffusers,
            lora_config=config.policy.lora,
            model_str=model_str,
            model_revision=model_revision,
        )

    @classmethod
    def get_nparams_and_flops(self):
        # TODO (yy): Support nparams and flops calculation
        pass

    @abstractmethod
    def text_embedding(self, prompt_list: List[str], device="cuda", **kwargs):
        """
        Text embedding of list of prompts
        """
        raise NotImplementedError

    @abstractmethod
    def set_scheduler_timestep(self, timestep: int):
        """
        Set scheduler's timestep for nosie addition and noise removal process
        """
        raise NotImplementedError

    @abstractmethod
    def visual_embedding(
        self, input_visual_list, height=None, width=None, device="cuda", **kwargs
    ):
        """
        Text embedding of list of preprocessed image tensor
        """
        # whether usiong resolution bins is a pipeline specific feature, findout how to solve it after
        # Ignore resolution bin now
        raise NotImplementedError

    def sample_timestep_indice(self, bsz, device):
        """
        Sample timestep for noise addition with given sampling method
        """
        u = training_utils.compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.config.logit_mean,
            logit_std=self.config.logit_std,
            mode_scale=None,
        )
        timesteps_indices = (u * self.train_sampling_steps).long().to(device)
        return timesteps_indices

    @abstractmethod
    def add_noise(self, clean_latents, timestep=None, noise=None):
        """
        Add random noise by random sampling timestep index
        """
        raise NotImplementedError

    def get_trained_model_state_dict(self):
        if self.is_lora:
            model_state_dict = get_peft_model_state_dict(self.transformer)
        else:
            model_state_dict = self.transformer.state_dict()
        return model_state_dict

    def training_sft_step(
        self,
        clean_image,
        prompt_list,
        loss_only=True,
        x_t=None,
        timestep=None,
        noise=None,
    ):
        """
        Main training_step, do visual/text embedding on the fly
        Only support MSE loss now
        """
        latents = self.visual_embedding(clean_image)
        # Different model may have different kind of text embedding output
        # Key of this dict will name of the corresponding args' names
        text_embedding_dict = self.text_embedding(prompt_list)
        noised_latents, noise, timesteps = self.add_noise(
            latents, timestep=timestep, noise=noise
        )

        if x_t is not None:
            noised_latents = x_t

        self.transformer.train()
        model_output = self.transformer(
            noised_latents.to(self.transformer.dtype),
            timestep=timesteps,
            return_dict=False,
            **text_embedding_dict,
        )[0]

        # TODO (yy): Only support flow-matching now, expand later
        target = noise - latents
        loss = mean_flat((target - model_output) ** 2)
        if loss_only:
            return {"loss": loss}
        else:
            return {
                "loss": loss,
                "x_t": noised_latents,
                "text_embedding_dict": text_embedding_dict,
                "visual_embedding": latents,
                "output": model_output,
            }

    def inference(
        self,
        inference_step,
        height,
        width,
        prompt_list,
        guidance_scale,
        save_dir="",
        frames=None,
        negative_prompt="",
    ):
        """
        Main inference, do diffusers generation with given sampling parameters
        """
        # Denoise loop
        self.transformer.eval()
        kwargs = {}
        if self.is_video:
            kwargs["frames"] = frames

        if self.offload:
            for model in self.offloaded_models:
                model.to("cuda")
        with torch.no_grad():
            visual_output = self.pipeline(
                prompt=prompt_list,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=inference_step,
                **kwargs,
            )[0]
        self.transformer.train()

        # After inference, set scheduler's timesteps back to train_sampling_steps for following train steps
        self.set_scheduler_timestep(self.train_sampling_steps)
        if self.offload:
            for model in self.offloaded_models:
                model.to("cpu")

        return visual_output

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.diffusers.parallelize import parallelize

        return parallelize, self

    # Lora Supports
    def apply_lora(self, lora_config):
        self.transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            init_lora_weights=lora_config.init_lora_weights,
            target_modules=lora_config.target_modules,
        )
        for lora_name in lora_config.lora_names:
            if not hasattr(self.transformer, "add_adapter"):
                self.transformer = get_peft_model(
                    self.transformer,
                    peft_config=transformer_lora_config,
                    adapter_name=lora_name,
                )
            else:
                # The add_adapter API of diffusers and peft are different
                if isinstance(self.transformer, PeftAdapterMixin):
                    self.transformer.add_adapter(
                        adapter_config=transformer_lora_config, adapter_name=lora_name
                    )
                else:
                    self.transformer.add_adapter(
                        peft_config=transformer_lora_config, adapter_name=lora_name
                    )

    @property
    def trained_model(self):
        return [self.transformer]

    def check_tp_compatible(self, tp_size):
        assert tp_size == 1, "tp is not supported for DiffuserModel"

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        assert cp_size == 1, "cp is not supported for DiffuserModel"
