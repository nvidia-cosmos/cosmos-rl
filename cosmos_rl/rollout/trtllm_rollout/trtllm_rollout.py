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

from typing import List, Tuple, Optional


from transformers import GenerationConfig
from tensorrt_llm._torch import LLM
from tensorrt_llm import SamplingParams
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.dispatcher.data.packer import DataPacker
from transformers import AutoTokenizer


class DemoLLM:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        logger.info("[Rollout] LMS: generate")
        return [
            "Hello, world!",
        ]


class TRTLLM_Rollout(RolloutBase):
    def __init__(self, config: Config, tokenizer: AutoTokenizer, **kwargs):
        super().__init__(config, tokenizer)
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

        logger.info(f"[Rollout] LMS: init rollout engine with model: {model_path}")
        self.rollout_engine = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            seed=seed,
            load_format=load_format,
            trust_remote_code=True,
        )

        logger.info("[Rollout] LMS: init rollout engine done!")

    def rollout_generation(
        self,
        prompt_id_and_payload_list: List[Tuple[int, str]],
        data_packer: DataPacker,
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
        payloads = [x[1] for x in prompt_id_and_payload_list]

        # Pack the payloads into prompts for vllm.
        prompts = [data_packer.get_rollout_input(payload) for payload in payloads]
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
                prompts=prompts,
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

    def get_underlying_model(self):
        # we can't get the underlying model from trtllm
        return None

    def is_engine_initialized(self):
        return self._engine_initialized

    def get_engine(self):
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine
