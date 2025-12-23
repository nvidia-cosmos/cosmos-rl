import os
import re
import torch
import inspect
from typing import List, Tuple, Optional

from PIL import Image
from torch import nn
import torchvision.transforms as T
import diffusers
from diffusers import DiffusionPipeline
from diffusers import training_utils

from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.diffusers import DiffuserModel
from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import DiffusersConfig

from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
import inspect

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

@ModelRegistry.register(DiffuserModelWeightMapper)
class SD3Model(DiffuserModel):

    @staticmethod
    def supported_model_types():
        return ['StableDiffusion3Pipeline']

    def __init__(self, config: DiffusersConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.set_scheduler_timestep(timestep=self.train_sampling_steps)

    def text_embedding(self, prompt_list: List[str], device='cuda'):
        '''
            Text embedding of list of prompts
        '''
        # Move all text encoder to cuda if offload is enabled
        if self.offload:
            for model_tuple in self.separate_model_parts():
                if 'text' in model_tuple[0]:
                    model_tuple[1].to(device)

        with torch.no_grad():
            # Fetch all default value of pipeline.__call__ that used by encode_prompt
            ignore_args = ['prompt', 'do_classifier_free_guidance']
            kwargs = {}
            sig_encode_pompt = inspect.signature(self.pipeline.encode_prompt)
            sig_call = inspect.signature(self.pipeline.__call__)
            for name, params in sig_encode_pompt.parameters.items():
                if name not in ignore_args and name in sig_call.parameters:
                    kwargs[name] = sig_call.parameters[name].default

            # Training doesn't need to do cfg, only prompt embedding is needed
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = self.pipeline.encode_prompt(
                prompt_list,
                do_classifier_free_guidance = False,
                device=device,
                **kwargs,
            )
        
        if self.offload:
            for model_tuple in self.separate_model_parts():
                if 'text' in model_tuple[0]:
                    model_tuple[1].to('cpu')
        return prompt_embeds, pooled_prompt_embeds

    def visual_embedding(self, input_visual_list, height=None, width=None, device='cuda'):
        '''
            Text embedding of list of preprocessed image tensor.
                input_visual_list: Only support List[torch.Tensor] now. Each tensor is [c,h,w] for image and [c,t,h,w] for video
        '''
        if self.offload:
            self.vae.to('cuda')

        input_samples = torch.stack(input_visual_list).to(self.vae.device)

        with torch.no_grad():
            vae_config_shift_factor = self.vae.config.shift_factor
            vae_config_scaling_factor = self.vae.config.scaling_factor
            visual_embedding = (self.vae.encode(input_samples.to(self.vae.dtype), return_dict=True).latent_dist.sample() - vae_config_shift_factor) * vae_config_scaling_factor

        if self.offload:
            self.vae.to('cpu')
            torch.cuda.empty_cache()
        return visual_embedding

    def set_scheduler_timestep(self, timestep: int):
        '''
            Set scheduler's timestep for nosie addition and noise removal process
        '''
        self.scheduler.set_timesteps(num_inference_steps=timestep)
        self.timestep_map = torch.flip(self.scheduler.timesteps,dims=(0,)).to('cuda')

    def add_noise(self, clean_latents, timestep=None, noise=None):
        '''
            Add random noise by random sampling timestep index
        '''

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
            sample = clean_latents,
            noise = noise,
            timestep = timesteps)
        return noised_latent, noise, timesteps
    
    def training_step(self, clean_image, prompt_list, x_t=None, timestep=None, noise=None):
        '''
            Main training_step, do visual/text embedding on the fly
            Only support MSE loss now
        '''
        latents = self.visual_embedding(clean_image)
        prompt_embedding, pooled_prompt_embeds = self.text_embedding(prompt_list)
        noised_latents, noise, timesteps = self.add_noise(latents, timestep=timestep, noise=noise)

        if x_t is not None:
            noised_latents = x_t

        self.transformer.train()
        model_output = self.transformer(
            hidden_states = noised_latents.to(self.transformer.dtype),
            encoder_hidden_states=prompt_embedding,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            return_dict=False,
        )[0]

        target = noise - latents
        loss = mean_flat((target - model_output) ** 2)

        return {'loss': loss, 'x_t': noised_latents, "text_embedding":prompt_embedding, "visual_embedding": latents, "output": model_output}
