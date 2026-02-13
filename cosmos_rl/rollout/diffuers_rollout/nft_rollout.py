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

from typing import List

import torch

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.diffusers import DiffuserModel
from cosmos_rl.utils.parallelism import ParallelDims


@RolloutRegistry.register(rollout_type="diffusion_nft_rollout")
class NFTRollout(RolloutBase):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        device: torch.device,
        **kwargs,
    ):
        """
        Initialize the RolloutBase class.
        """
        super().__init__(config, parallel_dims, device)
        self.model_inited = False
        self.diffusers_config = config.policy.diffusers

    def post_init_hook(self, **kwargs):
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation
        self._model_param_map = None  # key: compatible name, value: param

    def set_neg_prompt_embed(self):
        neg_text_embedding_dict = self.model.text_embedding(
            [""],
            device=self.device,
            max_sequence_length=self.diffusers_config.max_prompt_length,
        )
        self.neg_prompt_embed = neg_text_embedding_dict["encoder_hidden_states"]
        self.neg_prompt_attention_mask = neg_text_embedding_dict[
            "encoder_attention_mask"
        ]
        self.neg_pooled_prompt_embed = neg_text_embedding_dict["pooled_projections"]

    def rollout_generation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        *args,
        **kwargs,
    ) -> List[RolloutResult]:
        self.model.transformer.eval()  # set transformer to eval mode for rollout
        self.model.transformer.set_adapter(
            "default"
        )  # ensure using the default adapter for rollout
        response = []
        for pl in payloads:
            prompts, metadatas = data_packer.get_rollout_input(
                payload=pl, n_generation=self.config.rollout.n_generation
            )
            text_embedding_dict = self.model.text_embedding(
                prompts,
                device=self.device,
                max_sequence_length=self.diffusers_config.max_prompt_length,
            )
            prompt_embeds = text_embedding_dict["encoder_hidden_states"]
            prompt_attention_mask = text_embedding_dict["encoder_attention_mask"]
            pooled_prompt_embeds = text_embedding_dict["pooled_projections"]
            prompt_ids = self.model.tokenizers[0](
                prompts,
                padding="max_length",
                max_length=self.diffusers_config.max_prompt_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)

            self.model.transformer.set_adapter("old")
            # Create generators with random seeds for each generation to ensure diversity
            generators = [
                torch.Generator(device=self.device).manual_seed(
                    int(torch.randint(0, 2**31, (1,), device=self.device).item())
                )
                for _ in range(self.config.rollout.n_generation)
            ]
            with torch.no_grad():
                # Inference with logprob computation
                # mm_datas contains the generated images/videos
                call_kwargs = {
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": self.neg_prompt_embed.repeat(
                        self.config.rollout.n_generation, 1, 1
                    ),
                    "num_inference_steps": self.diffusers_config.sample.num_steps,
                    "guidance_scale": self.diffusers_config.sample.guidance_scale,
                    "output_type": "pt",
                    "height": self.diffusers_config.inference_size[0],
                    "width": self.diffusers_config.inference_size[1],
                    "noise_level": self.diffusers_config.sample.noise_level,
                    "deterministic": self.diffusers_config.sample.deterministic_sampling,
                    "generator": generators,
                    "solver": self.diffusers_config.sample.solver,
                    "num_frames": self.diffusers_config.train_frames,
                }

                if pooled_prompt_embeds is not None:
                    call_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
                if prompt_attention_mask is not None:
                    call_kwargs["prompt_attention_mask"] = prompt_attention_mask
                if self.neg_pooled_prompt_embed is not None:
                    call_kwargs["negative_pooled_prompt_embeds"] = (
                        self.neg_pooled_prompt_embed.repeat(
                            self.config.rollout.n_generation, 1
                        )
                    )
                if self.neg_prompt_attention_mask is not None:
                    call_kwargs["negative_prompt_attention_mask"] = (
                        self.neg_prompt_attention_mask.repeat(
                            self.config.rollout.n_generation, 1
                        )
                    )
                mm_datas, latents, _ = self.model.pipeline_with_logprob(**call_kwargs)
                latents = torch.stack(latents, dim=1)
                timesteps = self.model.pipeline.scheduler.timesteps.repeat(
                    len(prompts), 1
                ).to(self.device)

            response.append(
                RolloutResult(
                    prompt=pl.prompt["prompt"],
                    completions=mm_datas,
                    completion_logprobs=None,
                    completion_token_ids=None,
                    extra_info={
                        "modality": "video" if self.model.is_video else "image",
                        "prompt_ids": prompt_ids,
                        "prompt_metadatas": metadatas,
                        "prompt_embeds": prompt_embeds,
                        "prompt_attention_mask": prompt_attention_mask,
                        "pooled_prompt_embeds": pooled_prompt_embeds,
                        "timesteps": timesteps,
                        "latents_clean": latents[:, -1],
                    },
                )
            )
        return response

    def init_engine(self, quantization: str, seed: int, load_format: str, **kwargs):
        pass

    def get_underlying_model(self):
        """Get the underlying model"""
        return self.model

    def set_underlying_model(self, model: DiffuserModel):
        """Set the underlying model"""
        self.model = model
        if not self.model_inited:
            self.model_inited = True
            self.set_neg_prompt_embed()
