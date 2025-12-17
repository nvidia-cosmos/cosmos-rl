import os
import re
import torch
import inspect
from typing import List, Tuple, Optional

from PIL import Image
from torch import nn
import torchvision.transforms as T
import diffusers
from diffusers import SanaTransformer2DModel, AutoencoderDC, DPMSolverMultistepScheduler, AutoencoderKLWan, SanaVideoTransformer3DModel
from diffusers.image_processor import PixArtImageProcessor
from diffusers.video_processor import VideoProcessor
from diffusers import SanaVideoPipeline, training_utils
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN

from transformers import AutoModelForCausalLM, AutoTokenizer

from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
from cosmos_rl.policy.model.diffusers.constants import get_ratio_bin
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import DiffusersConfig

from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

@ModelRegistry.register(DiffuserModelWeightMapper)
class DiffuserModel(BaseModel):

    @staticmethod
    def supported_model_types():
        return ['diffusers']

    def __init__(self, config: DiffusersConfig, model_str: str = ""):
        super().__init__()
        complex_human_instruction = config.complex_human_instruction
        self.is_video = config.is_video
        self.load_models_from_hf(model_str)

        self.chi_prompt = "\n".join(complex_human_instruction)
        self.num_chi_prompt_tokens = None
        self.max_sequence_length = config.max_prompt_length
        self.train_sampling_steps = self.scheduler.config.num_train_timesteps
        self.weighting_scheme = config.weighting_scheme


        self.init_output_process(config)
        # init sigmas, 
        # in diffusers, https://github.com/huggingface/diffusers/blob/a748a839add5fe9f45a66e45dd93d8db0b45ce0f/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L433
        # total step will add 1, so we need minus 1 here to match train sampling steps
        self.scheduler.set_timesteps(num_inference_steps=self.train_sampling_steps - 1)
        self.set_timestep_map()
    
    def current_device(self):
        return next(self.transformer.parameters()).device

    def set_gradient_checkpointing_enabled(self, enabled: bool):
        '''
            Diffusers's transformer model support gradient checkpointing, just need to set it to True
        '''
        super().set_gradient_checkpointing_enabled(enabled)
        if hasattr(self.transformer, "_supports_gradient_checkpointing") and \
            self.transformer._supports_gradient_checkpointing and enabled:
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
        '''
            Return different model parts, diffusers usually contain transformer, vae_model and text_encoder
        '''
        return [self.transformer, self.vae_model, self.text_encoder]

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

    def init_output_process(self, config):
        '''
           For inference, output processer is needed to transfer latents back to visual output 
        '''
        if self.is_video:
            vae_scale_factor = self.vae_model.config.scale_factor_spatial
            self.output_processer = VideoProcessor(vae_scale_factor)
        else:
            vae_scale_factor = 2 ** (len(self.vae_model.config.encoder_block_out_channels) - 1)
            self.output_processer = PixArtImageProcessor(vae_scale_factor)
        self.ratio_bin = get_ratio_bin(config.ratio_bin)

    def cls_mapping(self):
        '''
           Get init cls for all pipeline parts
        '''
        # TODO: read out cls from pipeline config or load models by config
        if self.is_video:
            transformer_cls = SanaVideoTransformer3DModel
            scheduler_cls = DPMSolverMultistepScheduler
            vae_cls = AutoencoderKLWan
        else:
            transformer_cls = SanaTransformer2DModel
            scheduler_cls = DPMSolverMultistepScheduler
            vae_cls = AutoencoderDC

        return transformer_cls, scheduler_cls, vae_cls

    def load_models_from_hf(self, model_str, device='cuda'):
        '''
           Load all models
        '''
        # TODO (yy): get all cls from diffusers config
        # TODO (yy): maybe selective init, preprocessed embedding only need transformers
        transformer_cls, scheduler_cls, vae_cls = self.cls_mapping()
        self.transformer = transformer_cls.from_pretrained(
            model_str,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(device)

        # scheduler must init on cpu
        with torch.device('cpu'):
            self.scheduler = scheduler_cls.from_pretrained(
                model_str,
                subfolder="scheduler",
            )

        self.vae_model = vae_cls.from_pretrained(
            model_str,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.vae_model.eval()

        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            model_str,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16
        ).get_decoder().to(device)

        self.text_encoder.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_str,
            subfolder="tokenizer"
        )
        self.tokenizer.padding_side = "right"


    @classmethod
    def from_pretrained(cls, config, diffusers_config_args):
        '''
            Model initialize entrypoiny
        '''
        return cls(config.policy.diffusers_config, model_str=config.policy.model_name_or_path)

    
    @classmethod
    def get_nparams_and_flops(self):
        # TODO (yy): Support nparams and flops calculation
        pass

    def text_formatter(self, prompt):
        # TODO (yy): Should remove here and add it to dataset
        return prompt.lower().strip()

    def text_embedding(self, prompt_list: List[str], device='cuda', dtype=torch.bfloat16):
        '''
            Text embedding of list of prompts
        '''
        self.text_encoder.to('cuda')
        prompt = [self.chi_prompt + p for p in prompt_list]
        if self.num_chi_prompt_tokens is None:
            self.num_chi_prompt_tokens = len(self.tokenizer.encode(self.chi_prompt))

        max_length_all = self.num_chi_prompt_tokens + self.max_sequence_length - 2
        select_index = [0] + list(range(-self.max_sequence_length + 1, 0))

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        prompt_attention_mask = text_inputs.attention_mask
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids, 
                attention_mask=prompt_attention_mask
            )[0]
        prompt_embeds = prompt_embeds[:, select_index]
        prompt_attention_mask = prompt_attention_mask[:, select_index]
        self.text_encoder.to('cpu')
        torch.cuda.empty_cache()
        return prompt_embeds, prompt_attention_mask

    def visual_embedding(self, input_image_list, device='cuda'):
        '''
            Text embedding of list of preprocessed image tensor
        '''
        input_samples = torch.stack(input_image_list).to(device)
        if self.is_video:
            with torch.no_grad():
                visual_embedding = self.vae_model.encode(input_samples.to(torch.bfloat16), return_dict=False)[0]
            latents_mean = (
                torch.tensor(self.vae_model.config.latents_mean)
                .view(1, self.vae_model.config.z_dim, 1, 1, 1)
                .to(input_samples.device, input_samples.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_model.config.latents_std).view(1, self.vae_model.config.z_dim, 1, 1, 1).to(
                input_samples.device, input_samples.dtype
            )
            visual_embedding = (visual_embedding.mean - latents_mean) * latents_std
        else:
            scaling_factor = self.vae_model.config.scaling_factor
            with torch.no_grad():
                visual_embedding = self.vae_model.encode(input_samples, return_dict=False)[0] * scaling_factor
        torch.cuda.empty_cache()
        return visual_embedding

    def set_timestep_map(self):
        '''
            Since timestep index will be random generated, timestep_map is needed to retrieve true timestep
        '''
        # Since in diffusers, timestep is in decrease order and ignore step=0
        # Reverse and add back step=0
        self.timestep_map = torch.flip(self.scheduler.timesteps,dims=(0,)).to('cuda')
        zero = torch.zeros((1), dtype=self.timestep_map.dtype, device=self.timestep_map.device)
        self.timestep_map = torch.cat([zero, self.timestep_map], dim=0)

    def add_noise(self, clean_latents, device='cuda', timestep=None, noise=None):
        '''
            Add random noise by random sampling timestep index
        '''
        # random timestep
        if timestep is None:
            bsz = clean_latents.shape[0]
            u = training_utils.compute_density_for_timestep_sampling(
                weighting_scheme = self.weighting_scheme,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=None
            )
            timesteps_indices = (u * self.train_sampling_steps).long().to(device)
        else:
            timesteps_indices = timestep

        timesteps = self.timestep_map[timesteps_indices]
        if noise is None:
            noise = torch.randn_like(clean_latents).to(clean_latents.device)
        noised_latent = self.scheduler.add_noise(
            original_samples = clean_latents,
            noise = noise,
            timesteps = timesteps)
        return noised_latent, noise, timesteps
    
    def training_step(self, clean_image, prompt_list, x_t=None, timestep=None, noise=None):
        '''
            Main training_step, do visual/text embedding on the fly
            Only support MSE loss now
        '''
        latents = self.visual_embedding(clean_image)
        prompt_embedding, prompt_attention_mask = self.text_embedding(prompt_list)
        noised_latents, noise, timesteps = self.add_noise(latents, timestep=timestep, noise=noise)
        if x_t is not None:
            noised_latents = x_t
        self.transformer.train()
        model_output = self.transformer(
            noised_latents.to(torch.bfloat16),
            encoder_hidden_states=prompt_embedding,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
            return_dict=False,
        )[0]
        target = noise - latents
        loss = mean_flat((target - model_output) ** 2)
        return {'loss': loss, 'x_t': noised_latents, "text_embedding":prompt_embedding, "visual_embedding": latents, "output": model_output}
    
    def prepare_noise_latent(self, bsz, height, width, num_frames, dtype=torch.bfloat16, device='cuda'):
        '''
            Generate beginning noise latent for inference
        '''
        if self.is_video:
            vae_scale_factor = self.vae_model.config.scale_factor_spatial
            vae_scale_factor_temporal = self.vae_model.config.scale_factor_temporal
            num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
            latent_channel = self.transformer.config.in_channels

            shape = (
                bsz,
                latent_channel,
                num_latent_frames,
                int(height) // vae_scale_factor,
                int(width) // vae_scale_factor,
            )
        else:
            vae_scale_factor = 2 ** (len(self.vae_model.config.encoder_block_out_channels) - 1)
            latent_channel = self.transformer.config.in_channels

            shape = (
                bsz,
                latent_channel,
                int(height) // vae_scale_factor,
                int(width) // vae_scale_factor,
            )
        return torch.randn(shape, dtype=dtype, device=device)

    def inference(self, inference_step, height, width, prompt_list, guidance_scale, frames=None, negative_prompt=""):
        '''
            Main inference, do diffusers generation with given sampling parameters
        '''
        bsz = len(prompt_list)
        # CFG
        prompt_embedding, prompt_attention_mask = self.text_embedding(prompt_list)
        negative_prompt_embedding, negative_prompt_attention_mask = self.text_embedding([negative_prompt] * bsz)
        prompt_embeds = torch.cat([negative_prompt_embedding, prompt_embedding], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # Set scheduler's timesteps to inference_steps
        self.scheduler.set_timesteps(inference_step, device='cuda')
        timesteps = self.scheduler.timesteps
        
        # Prepare latent
        orig_width, orig_height = width, height
        height, width = self.output_processer.classify_height_width_bin(height, width, ratios=self.ratio_bin)
        latents = self.prepare_noise_latent(bsz, height, width, frames, dtype=torch.float32)

        # Denoise loop
        for i,t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            if hasattr(self.transformer.config, "timestep_scale"):
                timestep = timestep * self.transformer.config.timestep_scale
            with torch.no_grad():
                noise_pred = self.transformer(
                    latent_model_input.to(dtype=torch.float32),
                    encoder_hidden_states=prompt_embeds.to(dtype=torch.float32),
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents.to(self.vae_model.dtype)
        
        # Transform latents back to visual output
        if self.is_video:
            latents_mean = (
                torch.tensor(self.vae_model.config.latents_mean)
                .view(1, self.vae_model.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_model.config.latents_std).view(1, self.vae_model.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            with torch.no_grad():
                visual_output = self.vae_model.decode(latents, return_dict=False)[0]

            visual_output = self.output_processer.resize_and_crop_tensor(visual_output, orig_width, orig_height)
            visual_output = self.output_processer.postprocess_video(visual_output, output_type='pil')
        else:
            with torch.no_grad():
                visual_output = self.vae_model.decode(latents / self.vae_model.config.scaling_factor, return_dict=False)[0]

            visual_output = self.output_processer.resize_and_crop_tensor(visual_output, orig_width, orig_height)
            visual_output = self.output_processer.postprocess(visual_output, output_type='pil')

        # After inference, set scheduler's timesteps back to train_sampling_steps for following train steps
        self.scheduler.set_timesteps(self.train_sampling_steps - 1, device='cuda')

        return visual_output

    @property
    def parallelize_fn(self):
        from cosmos_rl.policy.model.diffusers.parallelize import parallelize

        return parallelize, self

    @property
    def trained_model(self):
        return [self.transformer]

    def check_tp_compatible(self, tp_size):
        assert tp_size == 1, "tp is not supported for DiffuserModel"
        
    def check_cp_compatible(self, cp_size: int, tp_size: int):
        assert cp_size == 1, "cp is not supported for DiffuserModel"


if __name__ == '__main__':
    from PIL import Image

    # test_image
    # model_str = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
    # sana_pipeline = DiffuserModel(model_str)
    # bsz = 4
    # prompt_list = ["A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece" for _ in range(bsz)]

    # visual_output = sana_pipeline.inference(
    #     prompt_list=prompt_list,
    #     height=1024,
    #     width=1024,
    #     guidance_scale=4.5,
    #     inference_step=20,
    # )
    # for i, image in enumerate(images):
    #     image.save(f"sana_{i}.png")

    # test video
    from diffusers.utils import export_to_video
    model_str = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    
    ## Test generate
    sana_pipeline = DiffuserModel(model_str, is_video=True)
    bsz = 1
    prompt_list = ["Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional." for _ in range(bsz)]

    visual_output = sana_pipeline.inference(
        prompt_list=prompt_list,
        height=480,
        width=832,
        frames=81,
        guidance_scale=4.5,
        inference_step=20,
    )
    
    for idx, video in enumerate(visual_output):
        export_to_video(video, f"sana_video_{idx}.mp4", fps=16)
