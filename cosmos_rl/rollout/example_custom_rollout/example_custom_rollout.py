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

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult

from cosmos_rl.policy.config import Config as CosmosConfig
import torch
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from transformers import AutoModelForCausalLM, AutoTokenizer
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger
from torch.distributed.fsdp import (
    register_fsdp_forward_method,
    FSDPModule,
)
from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase


"""
HFRollout is a rollout engine that uses Hugging Face Transformers to generate sequences.
This rollout engine is just used for demonstration, testing and example purposes.
It demonstrates how to implement a custom rollout engine and how to register it to the RolloutRegistry to be used in the rollout worker.
It is not optimized for performance and may not be suitable for production use.
User can mimic this implementation to implement their own rollout engine.
"""


@RolloutRegistry.register("example_hf")
class ExampleHFRollout(RolloutBase):
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

    def post_init_hook(self, **kwargs):
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation
        self._model_param_map = None  # key: compatible name, value: param

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
        """Generate sequences"""
        assert (
            self.parallel_dims.world_size == self.parallel_dims.dp_shard
        ), "HF Rollout only supports world size equal to dp_shard"
        response = []
        if isinstance(self.model, FSDPModule):
            register_fsdp_forward_method(self.model, "generate")
        self.model.eval()
        for pl in payloads:
            prompt = data_packer.rollout_collate_fn(
                [data_packer.get_rollout_input(pl.prompt)]
            )[0]
            model_inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            generated_ids = self.model.generate(
                **model_inputs,
                **(
                    self.hf_generate_kwargs
                    if not is_validation
                    else self.hf_val_generate_kwargs
                ),
            ).to(self.model.device)
            generated_ids = [
                output_ids[len(model_inputs.input_ids) :]
                for output_ids in generated_ids
            ]
            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for text in texts:
                logger.debug(f"[ExampleHFRollout] Generated: {text}")
            response.append(
                RolloutResult(
                    prompt=pl.prompt,
                    completions=texts,
                    completion_logprobs=None,
                    completion_token_ids=None,
                )
            )
        return response

    def init_engine(self, quantization: str, seed: int, load_format: str, **kwargs):
        """Initialize the engine"""
        self._engine_initialized = True  # Set the engine initialized flag to True
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.policy.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        self.hf_generate_kwargs = {
            "num_return_sequences": self.config.rollout.n_generation,  # n=self.config.rollout.n_generation
            "top_p": self.config.rollout.sampling_config.top_p,  # top_p=self.config.rollout.sampling_config.top_p
            "top_k": self.config.rollout.sampling_config.top_k
            if self.config.rollout.sampling_config.top_k > 0
            else None,  # top_k=self.config.rollout.sampling_config.top_k
            "temperature": self.config.rollout.sampling_config.temperature,  # temperature=self.config.rollout.sampling_config.temperature
            "repetition_penalty": self.config.rollout.sampling_config.repetition_penalty,  # repetition_penalty=self.config.rollout.sampling_config.repetition_penalty
            "max_new_tokens": self.config.rollout.max_response_length,  # max_tokens=self.config.rollout.max_response_length
            "eos_token_id": self.tokenizer.eos_token_id,  # stop_token_ids=self.eos_token_ids
            "do_sample": True,
        }
        n_val_generation = self.config.validation.n_generation
        top_p_val = (
            self.config.validation.top_p
            if self.config.validation.top_p is not None
            else self.config.rollout.sampling_config.top_p
        )
        top_k_val = (
            self.config.validation.top_k
            if self.config.validation.top_k is not None
            else self.config.rollout.sampling_config.top_k
        )
        top_k_val = top_k_val if top_k_val > 0 else None
        temperature_val = (
            self.config.validation.temperature
            if self.config.validation.temperature is not None
            else self.config.rollout.sampling_config.temperature
        )
        repetition_penalty_val = (
            self.config.validation.repetition_penalty
            if self.config.validation.repetition_penalty is not None
            else self.config.rollout.sampling_config.repetition_penalty
        )
        max_response_length_val = (
            self.config.validation.max_response_length
            if self.config.validation.max_response_length is not None
            else self.config.rollout.max_response_length
        )
        self.hf_val_generate_kwargs = {
            "num_return_sequences": n_val_generation,  # n=self.config.validation.n_generation
            "top_p": top_p_val,  # top_p=calculated_value
            "top_k": top_k_val,  # top_k=calculated_value
            "temperature": temperature_val,  # temperature=calculated_value
            "repetition_penalty": repetition_penalty_val,  # repetition_penalty=calculated_value
            "max_new_tokens": max_response_length_val,  # max_tokens=calculated_value
            "eos_token_id": self.tokenizer.eos_token_id,  # stop_token_ids=self.eos_token_ids
            "do_sample": False,
        }

    def get_underlying_model(self):
        """Get the underlying model"""
        return self.model

    def set_underlying_model(self, model: torch.nn.Module):
        """Set the underlying model"""
        self.model = model
