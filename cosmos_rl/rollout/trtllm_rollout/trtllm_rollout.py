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

from typing import List, Optional

from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.utils.logging import logger

import tensorrt_llm

trtllm_version = tensorrt_llm.__version__

if trtllm_version == "1.0.0rc6":
    from tensorrt_llm import LLM
else:
    raise NotImplementedError(f"Unsupported trtllm version: {trtllm_version}")

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
import cosmos_rl.utils.util as util
from cosmos_rl.dispatcher.data.packer import BaseDataPacker

from transformers import AutoConfig
from transformers import GenerationConfig
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
)


class TRTLLM_Rollout(RolloutBase):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config)
        self.rollout_config = self.config.rollout

        hf_config_path = self.config.policy.model_name_or_path
        try:
            generation_config = util.retry(GenerationConfig.from_pretrained)(
                hf_config_path
            )
            self.eos_token_ids = generation_config.eos_token_id
            if isinstance(self.eos_token_ids, int):
                self.eos_token_ids = [self.eos_token_ids]
        except Exception as e:
            logger.warning(
                f"[Rollout] Failed to load generation config from {hf_config_path}: {str(e)}, use default eos_token_id."
            )
            # self.eos_token_ids = [tokenizer.eos_token_id]
            # TODO(lms): remove this
            self.eos_token_ids = [151645, 151643]

        self.model_config = util.retry(AutoConfig.from_pretrained)(
            self.config.policy.model_name_or_path
        )
        self._engine_initialized = True

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        model_path = self.config.policy.model_name_or_path
        rollout_parallelism = self.rollout_config.parallelism

        tp_size = rollout_parallelism.tp_size
        pp_size = rollout_parallelism.pp_size
        assert pp_size == 1, "TRTLLM only support pipeline parallelism 1 now."
        assert quantization is None, "TRTLLM Rollout does not support quantization now."

        moe_tensor_parallel_size = -1  # by default, trtllm uses tp for MoE.

        # Check if the model has MoE
        moe_model_type = {"qwen3_moe"}
        # multimodal_type = {"qwen2_5_vl"}

        model_type = self.model_config.model_type
        if model_type in moe_model_type:
            moe_tensor_parallel_size = tp_size

        kwargs = {}
        kwargs["disable_overlap_scheduler"] = True
        policy_config = self.config.policy
        # Check the prefix_caching like arguments default enabled?
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=self.rollout_config.gpu_memory_utilization,
            enable_block_reuse=False,  # disable KV cacheblock reuse for trtllm
        )
        self.rollout_engine = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            dtype="auto",
            backend="pytorch",
            skip_tokenizer_init=False,
            max_seq_len=policy_config.model_max_length,
            max_num_tokens=2048
            if 2048 >= policy_config.model_max_length
            else policy_config.model_max_length,
            enable_chunked_prefill=self.rollout_config.enable_chunked_prefill,
            load_format=load_format,
            trust_remote_code=True,
            moe_expert_parallel_size=moe_tensor_parallel_size,
            kv_cache_config=kv_cache_config,
            # For cosmos-rl
            cosmos_config=self.config,
            **kwargs,
        )
        logger.info("[Rollout] TRTLLM Engine initialized.")

    def rollout_generation(
        self,
        payloads: List[RLPayload],
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        sampling_params: SamplingParams,
    ) -> List[List[str]]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        # List of payloads.
        # [
        #   payload,
        #   payload,
        #   ...
        # ]

        # Pack the payloads into prompts for vllm.
        prompts = [
            data_packer.get_rollout_input(payload.prompt) for payload in payloads
        ]
        prompts = data_packer.rollout_collate_fn(prompts)

        # List of completions per prompt.
        # [
        #   [completion_str, completion_str, ...],
        #   [completion_str, completion_str, ...],
        #   ...
        # ]
        response: List[List[str]] = []
        try:
            results = self.rollout_engine.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            for output in results:
                response.append(
                    [output.outputs[i].text for i in range(len(output.outputs))]
                )
        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

        return response

    def is_engine_initialized(self):
        return self._engine_initialized

    def get_engine(self):
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine
