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

import inspect
import math
import numpy as np
import torch
import torchvision
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Union

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
class CosmosPredict2_5Model(DiffuserModel):
    @staticmethod
    def supported_model_types():
        return ["Cosmos2_5_PredictBasePipeline"]

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

    def _sde_step_with_logprob(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        prev_sample: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        determistic: bool = False,
        return_pixel_log_prob: bool = False,
        return_dt_and_std_dev_t: bool = False,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
        process from the learned model outputs (most often the predicted velocity).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned flow model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        # prev_sample_mean, we must convert all variable to fp32
        model_output = model_output.float()
        sample = sample.float()
        if prev_sample is not None:
            prev_sample = prev_sample.float()

        step_index = [self.pipeline.scheduler.index_for_timestep(t) for t in timestep]
        prev_step_index = [step + 1 for step in step_index]

        self.pipeline.scheduler.sigmas = self.pipeline.scheduler.sigmas.to(
            sample.device
        )
        sigma = self.pipeline.scheduler.sigmas[step_index].view(-1, 1, 1, 1, 1)
        sigma_prev = self.pipeline.scheduler.sigmas[prev_step_index].view(
            -1, 1, 1, 1, 1
        )
        sigma_max = self.pipeline.scheduler.sigmas[1].item()
        sigma_min = self.pipeline.scheduler.sigmas[-1].item()
        dt = sigma_prev - sigma

        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                " `prev_sample` stays `None`."
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

        # No noise is added during evaluation
        if determistic:
            prev_sample = sample + dt * model_output

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        if return_dt_and_std_dev_t:
            return (
                prev_sample,
                log_prob,
                prev_sample_mean,
                std_dev_t,
                torch.sqrt(-1 * dt),
            )
        return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1 * dt)

    @torch.no_grad()
    def pipeline_with_logprob(
        self,
        image: PipelineImageInput | None = None,
        video: List[PipelineImageInput] | None = None,
        prompt: Union[str, List[str]] | None = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        conditional_frame_timestep: float = 0.1,
        determistic: bool = False,
        return_pixel_log_prob: bool = False,
    ):
        if self.pipeline.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.pipeline.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        # Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs
        )
        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._current_timestep = None
        self.pipeline._interrupt = False

        device = self.pipeline._execution_device

        if self.pipeline.safety_checker is not None:
            self.pipeline.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.pipeline.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.pipeline.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        num_frames_in = None
        if image is not None:
            if batch_size != 1:
                raise ValueError(
                    f"batch_size must be 1 for image input (given {batch_size})"
                )

            image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
            video = torch.cat(
                [image, torch.zeros_like(image).repeat(num_frames - 1, 1, 1, 1)], dim=0
            )
            video = video.unsqueeze(0)
            num_frames_in = 1
        elif video is None:
            video = torch.zeros(
                batch_size, num_frames, 3, height, width, dtype=torch.uint8
            )
            num_frames_in = 0
        else:
            num_frames_in = len(video)

            if batch_size != 1:
                raise ValueError(
                    f"batch_size must be 1 for video input (given {batch_size})"
                )

        assert video is not None
        video = self.pipeline.video_processor.preprocess_video(video, height, width)

        # pad with last frame (for video2world)
        num_frames_out = num_frames
        if video.shape[2] < num_frames_out:
            n_pad_frames = num_frames_out - num_frames_in
            last_frame = video[0, :, -1:, :, :]  # [C, T==1, H, W]
            pad_frames = last_frame.repeat(1, 1, n_pad_frames, 1, 1)  # [B, C, T, H, W]
            video = torch.cat((video, pad_frames), dim=2)

        assert (
            num_frames_in <= num_frames_out
        ), f"expected ({num_frames_in=}) <= ({num_frames_out=})"

        video = video.to(device=device, dtype=vae_dtype)

        num_channels_latents = self.transformer.config.in_channels - 1
        latents, cond_latent, cond_mask, cond_indicator = self.pipeline.prepare_latents(
            video=video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames_in=num_frames_in,
            num_frames_out=num_frames,
            do_classifier_free_guidance=self.pipeline.do_classifier_free_guidance,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        cond_timestep = torch.ones_like(cond_indicator) * conditional_frame_timestep
        cond_mask = cond_mask.to(transformer_dtype)

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        all_latents = [latents]
        all_log_probs = []

        # Denoising loop
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipeline.scheduler.timesteps
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.pipeline.scheduler.order
        )
        self.pipeline._num_timesteps = len(timesteps)
        gt_velocity = (latents - cond_latent) * cond_mask

        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t.cpu().item()

                # NOTE: assumes sigma(t) \in [0, 1]
                sigma_t = (
                    torch.tensor(self.pipeline.scheduler.sigmas[i].item())
                    .unsqueeze(0)
                    .to(device=device, dtype=transformer_dtype)
                )

                in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
                in_latents = in_latents.to(transformer_dtype)
                in_timestep = (
                    cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t
                )
                noise_pred = self.transformer(
                    hidden_states=in_latents,
                    condition_mask=cond_mask,
                    timestep=in_timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                # NOTE: replace velocity (noise_pred) with gt_velocity for conditioning inputs only
                noise_pred = gt_velocity + noise_pred * (1 - cond_mask)

                if self.do_classifier_free_guidance:
                    noise_pred_neg = self.transformer(
                        hidden_states=in_latents,
                        condition_mask=cond_mask,
                        timestep=in_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    # NOTE: replace velocity (noise_pred_neg) with gt_velocity for conditioning inputs only
                    noise_pred_neg = gt_velocity + noise_pred_neg * (1 - cond_mask)
                    noise_pred = noise_pred + self.guidance_scale * (
                        noise_pred - noise_pred_neg
                    )

                latents, log_prob, _, _ = self._sde_step_with_logprob(
                    noise_pred.float(),
                    t.unsqueeze(0),
                    latents.float(),
                    determistic=determistic,
                    return_pixel_log_prob=return_pixel_log_prob,
                )

                all_latents.append(latents)
                all_log_probs.append(log_prob)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.pipeline.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        latents_mean = self.latents_mean.to(latents.device, latents.dtype)
        latents_std = self.latents_std.to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean
        video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        video = self._match_num_frames(video, num_frames)

        assert self.safety_checker is not None
        self.safety_checker.to(device)
        video = self.pipeline.video_processor.postprocess_video(video, output_type="np")
        video = (video * 255).astype(np.uint8)
        video_batch = []
        for vid in video:
            vid = self.safety_checker.check_video_safety(vid)
            video_batch.append(vid)
        video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
        video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
        video = self.pipeline.video_processor.postprocess_video(
            video, output_type=output_type
        )

        # Offload all models
        self.pipeline.maybe_free_model_hooks()

        return video, all_latents, all_log_probs
