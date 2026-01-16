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
from typing import List


from cosmos_rl.policy.model.base import ModelRegistry
from cosmos_rl.policy.model.diffusers import DiffuserModel
from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
from cosmos_rl.policy.config import DiffusersConfig


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

    def _text_embedding(self, prompt_list: List[str], device="cuda"):
        """
        Text embedding of list of prompts
        """
        with torch.no_grad():
            # Fetch all default value of pipeline.__call__ that used by encode_prompt
            ignore_args = ["prompt", "do_classifier_free_guidance"]
            kwargs = {}
            sig_encode_pompt = inspect.signature(self.pipeline.encode_prompt)
            sig_call = inspect.signature(self.pipeline.__call__)
            for name, params in sig_encode_pompt.parameters.items():
                if name not in ignore_args and name in sig_call.parameters:
                    kwargs[name] = sig_call.parameters[name].default

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

        return {
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_attention_mask,
        }

    def _visual_embedding(
        self, input_visual_list, height=None, width=None, device="cuda"
    ):
        """
        Text embedding of list of preprocessed image tensor.
            input_visual_list: Only support List[torch.Tensor] now. Each tensor is [c,h,w] for image and [c,t,h,w] for video
        """
        input_samples = torch.stack(input_visual_list).to(device)

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

        return visual_embedding

    def _set_scheduler_timestep(self, timestep: int):
        """
        Set scheduler's timestep for nosie addition and noise removal process
        """
        self.scheduler.set_timesteps(num_inference_steps=timestep)
        self.timestep_map = torch.flip(self.scheduler.timesteps, dims=(0,)).to("cuda")

    def _add_noise(self, clean_latents, timestep=None, noise=None):
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
