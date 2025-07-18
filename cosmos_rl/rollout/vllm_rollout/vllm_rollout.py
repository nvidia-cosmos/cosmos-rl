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
import os

from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import apply_fp8_linear_patch

import vllm
import torch
from typing import List, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoConfig
from transformers import GenerationConfig
from vllm.entrypoints.llm import LLM
from vllm import SamplingParams
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.policy.config import RolloutConfig
from cosmos_rl.dispatcher.data.packer import DataPacker
from PIL import Image
from cosmos_rl.policy.model import WeightMapper


# Still needed for MLA
try:
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
except ImportError:
    from vllm.model_executor.model_loader.loader import (
        _process_weights_after_loading as process_weights_after_loading,
    )


def make_prompt(tokenizer, image_or_image_url, question=None):
    prompt_dict = {}

    if image_or_image_url is not None:
        # 1 image + text
        img_token = "<image>"
        question = "Describe the image with long caption."
        prompt_str = f"{img_token}\n{question}"
        if isinstance(image_or_image_url, str):
            image = Image.open(image_or_image_url)
        else:
            image = image_or_image_url
        prompt_dict["multi_modal_data"] = {"image": image}
    else:
        # text only
        prompt_str = "Could you talk about what model you are and how you were trained?"

    # Create chat template
    messages = [{"role": "user", "content": prompt_str}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # prompt = prompt.replace("<pad>", "")  # Remove the BOS token as the tokenization will add it later
    prompt = prompt.replace(
        "<｜begin▁of▁sentence｜>", ""
    )  # Remove the BOS token as the tokenization will add it later
    prompt_dict["prompt"] = prompt

    logger.info(f"[Rollout] Full test prompt: {prompt_dict}")

    return prompt_dict


def vllm_version_check(rollout_config: RolloutConfig):
    vllm_version = vllm.__version__
    if vllm_version < "0.9.0" and rollout_config.parallelism.pp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not supported for vLLM < 0.9.0, current version is %s"
            % vllm_version
        )


class vLLMRollout(RolloutBase):
    def __init__(self, config: Config, tokenizer: AutoTokenizer, **kwargs):
        """Rollout with vLLM as the backend.

        Args:
            config: Cosmos Config.
            tokenizer: Tokenizer of the model.
            hf_config_path: huggingface config file path.
            model_hf_config: the huggingface config to initiallize the generating model in vllm
        """
        super().__init__()

        self.config = config
        policy_config = self.config.policy
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation
        self.trust_remote_code = True

        vllm_version_check(self.rollout_config)

        model_path = policy_config.model_name_or_path

        self.model_config = util.retry(AutoConfig.from_pretrained)(
            model_path, trust_remote_code=True
        )

        self.pad_token_id = tokenizer.pad_token_id

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
        self.tokenizer = tokenizer
        self._engine_initialized = False
        self.rollout_engine = None

        self._model_param_map = None  # key: compatible name, value: param

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        if not self._engine_initialized:
            model_path = self.config.policy.model_name_or_path

            rollout_parallelism = self.rollout_config.parallelism

            # disable VLLM_DISABLE_COMPILE_CACHE
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

            tp_size = rollout_parallelism.tp_size
            pp_size = rollout_parallelism.pp_size

            enable_ep_parallelism = False
            disable_mm_preprocessor_cache = False

            # Check if the model has MoE
            moe_model_type = {"qwen3_moe"}
            multimodal_type = {"qwen2_5_vl"}

            model_type = self.model_config.model_type
            if model_type in moe_model_type:
                enable_ep_parallelism = True
            if model_type in multimodal_type:
                # for vllm nightly, this is only True for multimodal models, check here
                disable_mm_preprocessor_cache = True

            # FIXME(aazzolini) add to config
            disable_mm_preprocessor_cache = True
            enable_prefix_caching = False

            assert tp_size * pp_size == rollout_parallelism.world_size, (
                "[Rollout] For tensor parallel, the tp_size * pp_size must be equal to world size, but got tp_size: %d, pp_size: %d, world_size: %d"
                % (tp_size, pp_size, rollout_parallelism.world_size)
            )

            self.quantization = quantization

            policy_config = self.config.policy

            self.rollout_engine = LLM(
                model=model_path,
                enable_sleep_mode=False,  # enable sleep could corrupt the cuda allocator.
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                enable_expert_parallel=enable_ep_parallelism,
                distributed_executor_backend="external_launcher",
                dtype="auto",
                enforce_eager=self.rollout_config.enforce_eager,  # enable cuda graph
                gpu_memory_utilization=self.rollout_config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
                skip_tokenizer_init=False,
                max_model_len=policy_config.model_max_length,
                disable_log_stats=True,
                # default to 2048, this is related with chunked prefill. https://docs.vllm.ai/en/latest/performance/optimization.html
                max_num_batched_tokens=2048
                if 2048 >= policy_config.model_max_length
                else policy_config.model_max_length,
                enable_chunked_prefill=self.rollout_config.enable_chunked_prefill,
                enable_prefix_caching=enable_prefix_caching,
                trust_remote_code=self.trust_remote_code,
                quantization=self.quantization,
                seed=seed or 42,
                load_format=load_format,
                limit_mm_per_prompt={
                    "image": 32,
                    "video": 0,
                },  # TODO(aazzolini): pass as config? or figure out V1?
            )
            self._engine_initialized = True
            logger.info("[Rollout] Engine initialized.")
            # initialization done.

            # patch the vllm model to use rowwise fp8
            if self.quantization == "fp8":
                from vllm.config import set_current_vllm_config

                vllm_config = self.rollout_engine.llm_engine.vllm_config
                with set_current_vllm_config(vllm_config):
                    apply_fp8_linear_patch(self.get_underlying_model())

    GLOBAL_IDX = 0

    def process_weights_after_loading(self):
        process_weights_after_loading(
            self.rollout_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model,
            self.rollout_engine.llm_engine.model_config,
            torch.device("cuda"),  # TODO(aazzolini) generalize this.
        )

    @torch.no_grad()
    def rollout_generation(
        self,
        prompt_id_and_payload_list: List[Tuple[int, Any]],
        stream: torch.cuda.Stream,
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

        stream = torch.cuda.current_stream() if stream is None else stream
        try:
            # Get SLURM job ID
            # job_id = os.environ.get("SLURM_JOB_ID", "nojob")
            # Get global rank from SLURM_PROCID (works with MPI / SLURM)
            # global_rank = os.environ.get("SLURM_PROCID", "norank")
            # Compose filename
            # filename = f"/opt/cosmos-reason1/assets/output_rollout_prompts_{job_id}_rank{global_rank}_{self.GLOBAL_IDX}.txt"
            # logger.info(f"[Rollout] Pickling prompts to file {filename}")
            # import pickle
            # with open(filename, "wb") as file:
            #    pickle.dump(prompts, file)
            # self.GLOBAL_IDX += 1
            # logger.info(f"[Rollout] Pickled prompts to file {filename}")

            with torch.cuda.stream(stream):
                results = self.rollout_engine.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

            for idx, output in enumerate(results):
                repo = [output.outputs[i].text for i in range(len(output.outputs))]
                response.append(repo)

                for index, quote in enumerate(repo):
                    logger.info(f"[Rollout] {idx,index}: {quote}")

        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

        return response

    def _do_test_rollout(self, msg: str = ""):
        # image_url = "/home/aazzolini/lustre/RL/ref-rl/assets/20250213_car1_resize_384_384.jpg"
        llm = self.rollout_engine
        directory = "/opt/cosmos-reason1/assets/"

        logger.info(f"Generating sample output {msg}")
        for image_index in [None, 0, 1, 2, 3]:
            if image_index is not None:
                if image_index == 0:
                    image_url = f"{directory}/20250213_car1.jpg"
                elif image_index == 1:
                    image_url = (
                        f"{directory}/20250213_cosmos_image_caption_data_batch.png"
                    )
                elif image_index == 2:
                    image_url = f"{directory}/arch.png"
                elif image_index == 3:
                    image_url = f"{directory}/cosmos-rl-deepseek.png"
                else:
                    raise ValueError(f"Invalid image index: {image_index}")

                img = Image.open(image_url)
            else:
                img = None

            inputs = [make_prompt(llm.get_tokenizer(), img)]
            sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=128)

            outputs = llm.generate(inputs, sampling_params=sampling_params)
            logger.info(
                f"Sample output results {msg}, {image_index}: :: {outputs[0].outputs[0].text} ::"
            )

        return outputs

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    def get_engine(self):
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine

    def is_engine_initialized(self):
        return self._engine_initialized

    def fp8_quantization(self, weight: torch.Tensor):
        # convert to fp8
        from vllm import _custom_ops as ops

        # quantization of rowwise torch scaled_mm.
        # weight has shape [out_dim, in_dim]
        qweight, weight_scale = ops.scaled_fp8_quant(
            weight, scale=None, use_per_token_if_dynamic=True
        )

        return qweight.t(), weight_scale

    def model_param_map(self, weight_mapper: WeightMapper):
        if self._model_param_map:
            return self._model_param_map
        model = self.get_underlying_model()
        param_map = {}
        for name, param in model.named_parameters():
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(name)
            param_map[compatible_name] = param
        self._model_param_map = param_map
        return self._model_param_map
