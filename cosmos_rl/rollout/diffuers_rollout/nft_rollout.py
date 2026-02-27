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
from cosmos_rl.utils.logging import logger


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
            total_batch = prompt_embeds.shape[0]
            generators = [
                torch.Generator(device=self.device).manual_seed(
                    int(torch.randint(0, 2**31, (1,), device=self.device).item())
                )
                for _ in range(total_batch)
            ]
            with torch.no_grad():
                # Inference with logprob computation
                # mm_datas contains the generated images/videos
                base_call_kwargs = {
                    "num_inference_steps": self.diffusers_config.sample.num_steps,
                    "guidance_scale": self.diffusers_config.sample.guidance_scale,
                    "output_type": "pt",
                    "height": self.diffusers_config.inference_size[0],
                    "width": self.diffusers_config.inference_size[1],
                    "noise_level": self.diffusers_config.sample.noise_level,
                    "deterministic": self.diffusers_config.sample.deterministic_sampling,
                    "solver": self.diffusers_config.sample.solver,
                    "num_frames": self.diffusers_config.train_frames,
                }

                mini_batch = (
                    self.config.rollout.n_generation_mini_batch
                    if self.config.rollout.n_generation_mini_batch is not None
                    else total_batch
                )
                if mini_batch <= 0:
                    raise ValueError(
                        f"n_generation_mini_batch must be positive (got {mini_batch})."
                    )
                mini_batch = min(mini_batch, total_batch)
                num_mini_batch = (
                    total_batch + mini_batch - 1
                ) // mini_batch  # ceil division

                mm_datas_chunks = []
                latents_clean_chunks = []
                for mini_idx, start in enumerate(
                    range(0, total_batch, mini_batch), start=1
                ):
                    end = min(total_batch, start + mini_batch)
                    cur_bs = end - start

                    if num_mini_batch > 1:
                        logger.info(
                            f"Running rollout mini-batch {mini_idx}/{num_mini_batch} (batch_size={cur_bs})"
                        )

                    call_kwargs = dict(base_call_kwargs)
                    call_kwargs["prompt_embeds"] = prompt_embeds[start:end]
                    call_kwargs["negative_prompt_embeds"] = (
                        self.neg_prompt_embed.repeat(cur_bs, 1, 1)
                    )
                    call_kwargs["generator"] = generators[start:end]

                    if pooled_prompt_embeds is not None:
                        call_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds[
                            start:end
                        ]
                    if prompt_attention_mask is not None:
                        call_kwargs["prompt_attention_mask"] = prompt_attention_mask[
                            start:end
                        ]
                    if self.neg_pooled_prompt_embed is not None:
                        call_kwargs["negative_pooled_prompt_embeds"] = (
                            self.neg_pooled_prompt_embed.repeat(cur_bs, 1)
                        )
                    if self.neg_prompt_attention_mask is not None:
                        call_kwargs["negative_prompt_attention_mask"] = (
                            self.neg_prompt_attention_mask.repeat(cur_bs, 1)
                        )

                    mm_datas_chunk, latents_chunk, _ = self.model.pipeline_with_logprob(
                        **call_kwargs
                    )

                    mm_datas_chunks.append(mm_datas_chunk)
                    latents_clean_chunks.append(latents_chunk[-1])

                if len(mm_datas_chunks) == 1:
                    mm_datas = mm_datas_chunks[0]
                else:
                    if not isinstance(mm_datas_chunks[0], torch.Tensor):
                        raise TypeError(
                            "Expected pipeline_with_logprob to return a torch.Tensor when output_type='pt'."
                        )
                    mm_datas = torch.cat(mm_datas_chunks, dim=0)

                latents_clean = torch.cat(latents_clean_chunks, dim=0)
                timesteps = self.model.pipeline.scheduler.timesteps.repeat(
                    total_batch, 1
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
                        "latents_clean": latents_clean,
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
