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

import math
import torch
import inspect
from diffusers.pipelines.sana.pipeline_sana import (
    ASPECT_RATIO_4096_BIN,
    ASPECT_RATIO_2048_BIN,
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_512_BIN,
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from typing import Any, Dict, List, Optional, Union


from cosmos_rl.policy.model.base import ModelRegistry
from cosmos_rl.policy.model.diffusers import DiffuserModel
from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
from cosmos_rl.policy.config import DiffusersConfig
from cosmos_rl.utils.logging import logger


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


@ModelRegistry.register(DiffuserModelWeightMapper)
class SanaModel(DiffuserModel):
    @staticmethod
    def supported_model_types():
        return ["SanaVideoPipeline", "SanaPipeline"]

    def __init__(self, config: DiffusersConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.set_scheduler_timestep(timestep=self.train_sampling_steps)
        self.text_encoders = [self.text_encoder]
        self.tokenizers = [self.tokenizer]

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
                prompt_attention_mask,
                _,
                _,
            ) = self.pipeline.encode_prompt(
                prompt_list,
                do_classifier_free_guidance=False,
                device=device,
                **kwargs,
            )

        if self.offload:
            for model_tuple in self.separate_model_parts():
                if "text" in model_tuple[0]:
                    model_tuple[1].to("cpu")
        return {
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_attention_mask,
            "pooled_projections": None,
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

        if self.is_video:
            with torch.no_grad():
                visual_embedding = self.vae.encode(
                    input_samples.to(self.vae.dtype), return_dict=False
                )[0]

            # This is specific feature of WAN2.1, other VAE_Model may not need this
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(input_samples.device, input_samples.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(input_samples.device, input_samples.dtype)
            visual_embedding = (visual_embedding.mean - latents_mean) * latents_std
        else:
            with torch.no_grad():
                scaling_factor = self.vae.config.scaling_factor
                visual_embedding = (
                    self.vae.encode(
                        input_samples.to(self.vae.dtype), return_dict=True
                    ).latent
                ) * scaling_factor

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

        noised_latent = self.scheduler.add_noise(
            original_samples=clean_latents, noise=noise, timesteps=timesteps
        )
        return noised_latent, noise, timesteps

    def sde_step_with_logprob(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        noise_level: float = 0.7,
        prev_sample: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_type: Optional[str] = "sde",
        return_sqrt_dt: Optional[bool] = False,
    ):
        # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
        model_output = model_output.float()
        sample = sample.float()
        if prev_sample is not None:
            prev_sample = prev_sample.float()

        step_index = [self.pipeline.scheduler.index_for_timestep(t) for t in timestep]
        prev_step_index = [step + 1 for step in step_index]
        self.pipeline.scheduler.sigmas = self.pipeline.scheduler.sigmas.to(
            sample.device
        )
        sigma = self.pipeline.scheduler.sigmas[step_index].view(
            -1, *([1] * (len(sample.shape) - 1))
        )
        sigma_prev = self.pipeline.scheduler.sigmas[prev_step_index].view(
            -1, *([1] * (len(sample.shape) - 1))
        )
        sigma_max = self.pipeline.scheduler.sigmas[1].item()
        dt = sigma_prev - sigma

        if sde_type == "sde":
            std_dev_t = (
                torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
                * noise_level
            )
            # our sde
            prev_sample_mean = (
                sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = (
                    prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
                )

            log_prob = (
                -((prev_sample.detach() - prev_sample_mean) ** 2)
                / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif sde_type == "cps":
            std_dev_t = sigma_prev * math.sin(
                noise_level * math.pi / 2
            )  # sigma_t in paper
            pred_original_sample = (
                sample - sigma * model_output
            )  # predicted x_0 in paper
            noise_estimate = sample + model_output * (
                1 - sigma
            )  # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (
                1 - sigma_prev
            ) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            # remove all constants
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        if return_sqrt_dt:
            return (
                prev_sample,
                log_prob,
                prev_sample_mean,
                std_dev_t,
                torch.sqrt(-1 * dt),
            )
        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    @torch.no_grad()
    def pipeline_with_logprob(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = None,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: int = 1024,
        width: int = 1024,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        noise_level: float = 0.7,
        complex_human_instruction: List[str] = [
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_4096_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 16:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.pipeline.image_processor.classify_height_width_bin(
                height, width, ratios=aspect_ratio_bin
            )

        self.pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_on_step_end_tensor_inputs,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._attention_kwargs = attention_kwargs
        self.pipeline._interrupt = False

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device
        lora_scale = (
            self.pipeline._attention_kwargs.get("scale", None)
            if self.pipeline._attention_kwargs is not None
            else None
        )

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt,
            self.pipeline.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )
        if self.pipeline.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # 4. Prepare timesteps
        timestep_device = device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            num_inference_steps,
            timestep_device,
            timesteps,
            sigmas,
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)  # noqa: F841

        all_latents = [latents]
        all_log_probs = []

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0
        )
        self.pipeline._num_timesteps = len(timesteps)

        transformer_dtype = self.transformer.dtype
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipeline.interrupt:
                    continue

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.pipeline.do_classifier_free_guidance
                    else latents
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                timestep = timestep * self.transformer.config.timestep_scale

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=self.pipeline.attention_kwargs,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if self.pipeline.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                latents, log_prob, _, _ = self.sde_step_with_logprob(
                    noise_pred.float(),
                    t.unsqueeze(0),
                    latents.float(),
                    noise_level=noise_level,
                )

                all_latents.append(latents)
                all_log_probs.append(log_prob)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.pipeline.scheduler.order == 0
                ):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            if use_resolution_binning:
                image = self.pipeline.image_processor.resize_and_crop_tensor(
                    image, orig_width, orig_height
                )

        if not output_type == "latent":
            image = self.pipeline.image_processor.postprocess(
                image, output_type=output_type
            )

        # Offload all models
        self.pipeline.maybe_free_model_hooks()

        return image, all_latents, all_log_probs

    def nft_prepare_transformer_input(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        **kwargs,
    ):
        """
        Prepare transformer input for training stage of DiffusionNFT
        Args:
            latents: Noised latent tensor
            prompt_embeds: Text embedding tensor
            prompt_attention_mask: Attention mask for text embedding
            pooled_prompt_embeds: Pooled text embedding tensor
            timestep: Timestep tensor
            num_frames: Number of frames to be generated
            height: Height of the image/video for generation
            width: Width of the image/video for generation
        Returns:
            transformer input args dict
        """
        return {
            "hidden_states": latents,
            "encoder_hidden_states": prompt_embeds,
            "attention_mask": prompt_attention_mask,
            "timestep": timestep,
            "return_dict": False,
            "attention_kwargs": self.pipeline.attention_kwargs,
        }
