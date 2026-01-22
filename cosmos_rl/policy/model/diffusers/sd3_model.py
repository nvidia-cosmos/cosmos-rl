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
import inspect
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from typing import Any, Dict, List, Optional, Union

from cosmos_rl.policy.model.base import ModelRegistry
from cosmos_rl.policy.model.diffusers import DiffuserModel
from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
from cosmos_rl.policy.config import DiffusersConfig
from cosmos_rl.utils.diffusers.solver import run_sampling
from cosmos_rl.utils.logging import logger


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


@ModelRegistry.register(DiffuserModelWeightMapper)
class SD3Model(DiffuserModel):
    @staticmethod
    def supported_model_types():
        return ["StableDiffusion3Pipeline"]

    def __init__(self, config: DiffusersConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.set_scheduler_timestep(timestep=self.train_sampling_steps)
        # Combine text_encoders and tokenizers for easier access
        self.text_encoders = [
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
        ]
        self.tokenizers = [self.tokenizer, self.tokenizer_2, self.tokenizer_3]

    def text_embedding(self, prompt_list: List[str], device="cuda", **kwargs):
        """
        Text embedding of list of prompts
        """
        # Move all text encoder to cuda if offload is enabled
        if self.offload:
            for model_tuple in self.separate_model_parts():
                if "text" in model_tuple[0]:
                    model_tuple[1].to(device)

        with torch.no_grad():
            # Fetch all default value of pipeline.__call__ that used by encode_prompt
            ignore_args = ["prompt", "do_classifier_free_guidance"]
            default_kwargs = {}
            sig_encode_pompt = inspect.signature(self.pipeline.encode_prompt)
            sig_call = inspect.signature(self.pipeline.__call__)
            for name, params in sig_encode_pompt.parameters.items():
                if name not in ignore_args and name in sig_call.parameters:
                    default_kwargs[name] = sig_call.parameters[name].default
            default_kwargs.update(kwargs)
            logger.debug(f"Default kwargs for encode_prompt: {default_kwargs}")
            # Training doesn't need to do cfg, only prompt embedding is needed
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = self.pipeline.encode_prompt(
                prompt_list,
                do_classifier_free_guidance=False,
                device=device,
                **default_kwargs,
            )

        if self.offload:
            for model_tuple in self.separate_model_parts():
                if "text" in model_tuple[0]:
                    model_tuple[1].to("cpu")
        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
        }

    def visual_embedding(
        self, input_visual_list, height=None, width=None, device="cuda"
    ):
        """
        Text embedding of list of preprocessed image tensor.
            input_visual_list: Only support List[torch.Tensor] now. Each tensor is [c,h,w] for image and [c,t,h,w] for video
        """
        if self.offload:
            self.vae.to("cuda")

        input_samples = torch.stack(input_visual_list).to(self.vae.device)

        with torch.no_grad():
            vae_config_shift_factor = self.vae.config.shift_factor
            vae_config_scaling_factor = self.vae.config.scaling_factor
            visual_embedding = (
                self.vae.encode(
                    input_samples.to(self.vae.dtype), return_dict=True
                ).latent_dist.sample()
                - vae_config_shift_factor
            ) * vae_config_scaling_factor

        if self.offload:
            self.vae.to("cpu")
            torch.cuda.empty_cache()
        return visual_embedding

    def set_scheduler_timestep(self, timestep: int):
        """
        Set scheduler's timestep for nosie addition and noise removal process
        """
        self.scheduler.set_timesteps(num_inference_steps=timestep)
        self.timestep_map = torch.flip(self.scheduler.timesteps, dims=(0,)).to("cuda")

    def add_noise(self, clean_latents, timestep=None, noise=None):
        """
        Add random noise by random sampling timestep index
        """

        # random timestep
        if timestep is None:
            bsz = clean_latents.shape[0]
            timesteps_indices = self.sample_timestep_indice(bsz, clean_latents.device)
        else:
            timesteps_indices = timestep

        timesteps = self.timestep_map[timesteps_indices]

        # random noise
        if noise is None:
            noise = torch.randn_like(clean_latents).to(clean_latents.device)

        noised_latent = self.scheduler.scale_noise(
            sample=clean_latents, noise=noise, timestep=timesteps
        )
        return noised_latent, noise, timesteps

    @torch.no_grad()
    def pipeline_with_logprob(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        noise_level: float = 0.7,
        deterministic: bool = False,
        solver: str = "flow",
        **kwargs,
    ):
        height = (
            height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        )
        width = (
            width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        )

        # 1. Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._joint_attention_kwargs = joint_attention_kwargs
        self.pipeline._current_timestep = None
        self.pipeline._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        lora_scale = (
            self.pipeline.joint_attention_kwargs.get("scale", None)
            if self.pipeline.joint_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.pipeline.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.pipeline.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=None,
        )
        self.pipeline._num_timesteps = len(timesteps)

        sigmas = self.pipeline.scheduler.sigmas.float()

        def v_pred_fn(z, sigma):
            latent_model_input = (
                torch.cat([z] * 2) if self.pipeline.do_classifier_free_guidance else z
            )
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = torch.full(
                [latent_model_input.shape[0]],
                sigma * 1000,
                device=z.device,
                dtype=torch.long,
            )
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.pipeline.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            return noise_pred

        # 6. Prepare image embeddings
        all_latents = [latents]
        all_log_probs = []

        # 7. Denoising loop
        latents, all_latents, all_log_probs = run_sampling(
            v_pred_fn, latents, sigmas, solver, deterministic, noise_level
        )

        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        latents = latents.to(dtype=self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.pipeline.image_processor.postprocess(
            image, output_type=output_type
        )  # (B, C, H, W)

        # Offload all models
        self.pipeline.maybe_free_model_hooks()

        return image, all_latents, all_log_probs
