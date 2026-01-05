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

from diffusers import DiffusionPipeline
from diffusers import training_utils

from abc import ABC, abstractmethod

from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import DiffusersConfig
from cosmos_rl.policy.config import LoraConfig as cosmos_lora_config

from peft import LoraConfig, get_peft_model_state_dict


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
        lora_config: cosmos_lora_config = None,
        model_str: str = "",
    ):
        super().__init__()
        self.config = config
        self.offload = self.config.offload
        self.onload_multistream = self.config.onload_multistream
        if self.onload_multistream:
            self.onload_stream = torch.Stream(device="cuda")
        self.load_models_from_hf(model_str)
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
        """
        Register all parts to be used for diffusion pipeline
        """
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
                model_part.to(torch.bfloat16)
                if self.offload:
                    self.move_model_to_cpu(model_part)
                    if self.onload_multistream:
                        self.apply_onload_hook(model_part)
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
    def trainable_parameters(self):
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

    def load_models_from_hf(self, model_str: str):
        """
        Load all models

        Args:
            model_str (str): The name or path of the diffusers pipeline.
        """
        # Init from pipeline
        self.model_str = model_str
        # Always init on cuda now
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_str, torch_dtype=torch.get_default_dtype(), device_map="cuda"
        )

        # Register all model parts to self
        # self.transformer will point to self.pipeline.transformer
        self.register_models()

    @classmethod
    def from_pretrained(cls, config, diffusers_config_args):
        """
        Model initialize entrypoiny
        """
        return cls(
            config.policy.diffusers_config,
            lora_config=config.policy.lora,
            model_str=config.policy.model_name_or_path,
        )

    @classmethod
    def get_nparams_and_flops(self):
        # TODO (yy): Support nparams and flops calculation
        pass

    def apply_onload_hook(self, model):
        """
        Apply multistream onload forward_pre_hook to trigger onload/compute overlapping
        """

        def onload_hook(module, input):
            for name, parameter in module.named_parameters(recurse=False):
                with torch.cuda.stream(module.onload_stream):
                    if parameter.data.untyped_storage().size() == 0:
                        assert hasattr(
                            parameter, "cpu_data"
                        ), "To use this onload hook, make sure call move_model_to_cpu first to create cpu copy for each parameters"
                        parameter.data = parameter.cpu_data.to(
                            "cuda", non_blocking=True
                        )

            copy_event = module.onload_stream.record_event()
            torch.cuda.current_stream().wait_event(copy_event)

        # Setup onload stream and forward_prehook
        for name, module in model.named_modules():
            module.onload_stream = self.onload_stream
            module.register_forward_pre_hook(onload_hook)

    def move_model_to_cpu(self, model):
        """
        Instead of directly moving parameters to cpu, init pinned cpu_memory for each parameter to hold data on cpu.
        Clean underlying GPU memory by Tensor.untyped_storage().resize_(0)
        """
        for name, parameters in model.named_parameters():
            if hasattr(parameters, "cpu_data"):
                # cpu_data created, just zero out original storage on gpu
                parameters.data.untyped_storage().resize_(0)
            else:
                # cpu_data not created, create here
                parameters.cpu_data = torch.empty_like(
                    parameters.data, device="cpu", pin_memory=True
                )
                parameters.cpu_data.copy_(parameters.data)
                # zero out original gpu data
                parameters.data.untyped_storage().resize_(0)

    def move_model_to_cuda(self, model):
        """
        Onload parameters when onload_multistream is not used
        """
        for name, parameters in model.named_parameters():
            if hasattr(parameters, "cpu_data"):
                # cpu_data created, just zero out original storage on gpu
                parameters.data = parameters.cpu_data.to("cuda", non_blocking=True)
            else:
                raise ValueError(
                    "To use this onload function, make sure call move_model_to_cpu first to create cpu copy for each parameters"
                )

        # Call synchronize to wait onloading finish, maybe not needed
        torch.cuda.synchronize()

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

    def text_embedding(self, prompt_list: List[str], device="cuda"):
        """
        Text embedding of list of prompts
        """
        if self.offload and (not self.onload_multistream):
            for model_tuple in self.separate_model_parts():
                if "text" in model_tuple[0]:
                    self.move_model_to_cuda(model_tuple[1])
        output = self._text_embedding(prompt_list, device)

        if self.offload:
            for model_tuple in self.separate_model_parts():
                if "text" in model_tuple[0]:
                    self.move_model_to_cpu(model_tuple[1])
        return output

    def set_scheduler_timestep(self, timestep: int):
        """
        Set scheduler's timestep for nosie addition and noise removal process
        """
        self._set_scheduler_timestep(timestep)

    def add_noise(self, clean_latents, timestep=None, noise=None):
        """
        Add random noise by random sampling timestep index
        """
        return self._add_noise(clean_latents, timestep, noise)

    def visual_embedding(
        self, input_visual_list, height=None, width=None, device="cuda"
    ):
        """
        Text embedding of list of preprocessed image tensor
        """
        if self.offload and (not self.onload_multistream):
            self.move_model_to_cuda(self.vae)
            torch.cuda.synchronize()

        output = self._visual_embedding(input_visual_list, height, width, device)

        if self.offload:
            self.move_model_to_cpu(self.vae)
        return output

    @abstractmethod
    def _add_noise(self, clean_latents, timestep=None, noise=None):
        """
        (Inner function) Add random noise by random sampling timestep index
        """
        raise NotImplementedError

    @abstractmethod
    def _text_embedding(self, prompt_list: List[str], device="cuda"):
        """
        (Inner function) Text embedding of list of prompts
        """
        raise NotImplementedError

    @abstractmethod
    def _set_scheduler_timestep(self, timestep: int):
        """
        (Inner function) Set scheduler's timestep for nosie addition and noise removal process
        """
        raise NotImplementedError

    @abstractmethod
    def _visual_embedding(
        self, input_visual_list, height=None, width=None, device="cuda"
    ):
        """
        (Inner function) Text embedding of list of preprocessed image tensor
        """
        # whether usiong resolution bins is a pipeline specific feature, findout how to solve it after
        # Ignore resolution bin now
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

        if self.offload and (not self.onload_multistream):
            for model in self.offloaded_models:
                self.move_model_to_cuda(model)
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
                self.move_model_to_cpu(model)

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
        self.transformer.add_adapter(transformer_lora_config)

    @property
    def trained_model(self):
        return [self.transformer]

    def check_tp_compatible(self, tp_size):
        assert tp_size == 1, "tp is not supported for DiffuserModel"

    def check_cp_compatible(self, cp_size: int, tp_size: int):
        assert cp_size == 1, "cp is not supported for DiffuserModel"
