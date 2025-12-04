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
import asyncio
import torch
import copy
import types
from typing import List, Optional, Dict, Any
from transformers import AutoConfig
from transformers import GenerationConfig
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine, AsyncEngineArgs
from vllm.sampling_params import SamplingParams, RequestOutputKind
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.dispatcher.data import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vllm_version_check, vLLMRollout
from cosmos_rl.utils.ipc import (
    ModuleLike,
    named_tensors_to_serialize,
    named_tensors_from_serialize,
)
from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import apply_fp8_linear_patch


class VLLMColocateWorkerExtension:
    """
    The extension designed to shared weight between the main process and the worker process via IPC.
    This way, the code can be compatible with both vLLM V0 and V1.
    NOTE: this class in a extension module, to use this class, you should pass the full qualified
    name as `worker_extension_cls` argument to the AsyncLLMEngine.from_engine_args() method.
    """

    def _get_model(self) -> torch.nn.Module:
        return self.model_runner.model

    def get_state_dict_ipc(self) -> Dict[str, Any]:
        """
        Get the CUDA IPC handles of the model weights.

        Returns:
            Dict[param_name, (ipc_handle, shape, dtype, device_id)]
        """

        state_dict = self._get_model().state_dict()
        # we also mark whether the weight tensor is also a parameter.
        param_keys = [name for name, _ in self._get_model().named_parameters()]

        not_parameter_names = set(state_dict.keys()) - set(param_keys)
        return named_tensors_to_serialize(state_dict), not_parameter_names

    def apply_fp8_linear_patch(self):
        """
        Apply the fp8 linear patch to the model when initialize the rollout engine.
        """
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.vllm_config):
            apply_fp8_linear_patch(self._get_model())


class vLLMRolloutAsync(RolloutBase):
    def __init__(self, config: Config, **kwargs):
        """Rollout with vLLM as the backend.

        Args:
            config: Cosmos Config.
            hf_config_path: huggingface config file path.
            model_hf_config: the huggingface config to initiallize the generating model in vllm
        """
        super().__init__(config)

        # TODO(zjx): refactor those methods to common methods in RolloutBase
        # reuse some of the vLLMRollout methods
        self.fp8_quantization = types.MethodType(vLLMRollout.fp8_quantization, self)
        self.mxfp4_quantization = types.MethodType(vLLMRollout.mxfp4_quantization, self)
        self.model_param_map = types.MethodType(vLLMRollout.model_param_map, self)
        self.preset_vllm_env = types.MethodType(vLLMRollout.preset_vllm_env, self)
        self.get_quantized_tensors = types.MethodType(
            vLLMRollout.get_quantized_tensors, self
        )

        policy_config = self.config.policy
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation

        vllm_version_check(self.rollout_config)

        model_path = policy_config.model_name_or_path

        self.model_config = util.retry(AutoConfig.from_pretrained)(model_path)
        self.underlying_model = None
        self.underlying_model_state_dict = None

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
        self._engine_initialized = False
        self.rollout_engine = None

        self._model_param_map = None  # key: compatible name, value: param
        self.is_vlm = getattr(self.model_config, "vision_config", None) is not None

        self.preset_vllm_env()

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        if not self._engine_initialized:
            trust_remote_code = True  # set trust remote code default to True.

            model_path = self.config.policy.model_name_or_path

            rollout_parallelism = self.rollout_config.parallelism

            tp_size = rollout_parallelism.tp_size
            pp_size = rollout_parallelism.pp_size

            enable_ep_parallelism = False
            disable_mm_preprocessor_cache = False

            # Check if the model has MoE
            # Note: even though deepseek_v3 is MoE, EP in rollout is not supported for it yet
            moe_model_type = {"qwen3_moe"}
            multimodal_type = {"qwen2_5_vl"}

            model_type = self.model_config.model_type
            if model_type in moe_model_type:
                enable_ep_parallelism = True
            if model_type in multimodal_type:
                # for vllm nightly, this is only True for multimodal models, check here
                disable_mm_preprocessor_cache = True
            assert tp_size * pp_size == rollout_parallelism.world_size, (
                "[Rollout] For tensor parallel, the tp_size * pp_size must be equal to world size, but got tp_size: %d, pp_size: %d, world_size: %d"
                % (tp_size, pp_size, rollout_parallelism.world_size)
            )

            if quantization == "none":
                quantization = None
            self.quantization = quantization

            policy_config = self.config.policy

            engine_args = AsyncEngineArgs(
                model=model_path,
                enable_sleep_mode=False,  # enable sleep could corrupt the cuda allocator.
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                enable_expert_parallel=enable_ep_parallelism,
                distributed_executor_backend="external_launcher",
                worker_extension_cls="cosmos_rl.rollout.vllm_rollout.vllm_rollout_async.VLLMColocateWorkerExtension",
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
                # Always disable prefix caching, since RL will change the underlying model.
                # The prefix cache will be invalid after training.
                enable_prefix_caching=False,
                trust_remote_code=trust_remote_code,
                quantization=self.quantization,
                seed=seed or 42,
                load_format=load_format,
            )

            self.rollout_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._engine_initialized = True
            logger.info("[Rollout] Engine initialized.")
            # initialization done.

            # patch the vllm model to use rowwise fp8
            if self.quantization == "fp8":
                asyncio.run(
                    self.rollout_engine.collective_rpc("apply_fp8_linear_patch")
                )

    def shutdown(self):
        if self._engine_initialized:
            self._engine_initialized = False
            self.rollout_engine.shutdown()

    def _get_request_id(self, prompt_idx: int, child_idx: int = 0):
        return str(f"req_{prompt_idx}_{child_idx}")

    async def _sub_generate_task(
        self, prompt: Any, sampling_params: SamplingParams, request_id: str
    ) -> str:
        async for result in self.rollout_engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if result.finished:
                return result.outputs[0].text

    async def rollout_generation_single_turn(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: DataPacker,
        sampling_params: SamplingParams,
    ) -> List[RolloutResult]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        # Here is a problem in vllm, when output_kind is not FINAL_ONLY, the count of result.outputs may not equal to the sampling_params.n
        # a valid solution is to set output_kind to FINAL_ONLY.
        assert (
            sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
        ), "vLLM async rollout must set output_kind to FINAL_ONLY."

        # TODO(zjx): should remove if vllm support putting multiple prompts in one call
        assert (
            len(payloads) == 1
        ), "vLLM async rollout only support one prompt at a time."

        # Pack the payloads into prompts for vllm.
        prompts = []
        for pl in payloads:
            assert (
                pl.prompt is not None
            ), "Prompt should not be None for single turn rollout generation."
            prompts.append(data_packer.get_rollout_input(pl.prompt))
        prompts = data_packer.rollout_collate_fn(prompts)
        if self.is_vlm:
            new_prompts = util.decode_vision_info(prompts)
        else:
            new_prompts = prompts

        completions = []
        stream = torch.cuda.current_stream() if stream is None else stream
        try:
            with torch.cuda.stream(stream):
                cur_prompt = new_prompts[0]
                cur_payload = payloads[0]
                n_generation = sampling_params.n

                # Manually control the generation of n requests to avoid long results slowing down generation speed in FINAL_ONLY mode
                sp = copy.deepcopy(sampling_params)
                sp.n = 1

                tasks = [
                    asyncio.create_task(
                        self._sub_generate_task(
                            cur_prompt,
                            sp,
                            self._get_request_id(cur_payload.prompt_idx, child_idx),
                        )
                    )
                    for child_idx in range(n_generation)
                ]
                results = await asyncio.gather(*tasks)
                completions = [result for result in results if result is not None]
        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []
        return [
            RolloutResult(
                prompt=payloads[0].prompt,
                completions=completions,
            )
        ]

    async def rollout_generation_multi_turn(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: DataPacker,
        sampling_params: SamplingParams,
    ) -> List[RolloutResult]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        raise NotImplementedError(
            "Multi-turn rollout is not supported in vLLM async rollout."
        )

    async def rollout_generation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: DataPacker,
        sampling_params: SamplingParams,
    ) -> List[RolloutResult]:
        if self.rollout_config.multi_turn_config.enable:
            return await self.rollout_generation_multi_turn(
                payloads, stream, data_packer, sampling_params
            )
        else:
            return await self.rollout_generation_single_turn(
                payloads, stream, data_packer, sampling_params
            )

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        if self.underlying_model is not None:
            return self.underlying_model

        # Note: state dict rather than serialize the whole model, have two benefits:
        # 1. Avoid unexpected object behavior when serializing the whole model.
        # 2. Avoid call `forward()` in the worker process, which is not safe.
        rpc_results = asyncio.run(
            self.rollout_engine.collective_rpc("get_state_dict_ipc")
        )
        # vllm backend may have multiple workers, we use the first worker's state dict to initialize the underlying model.
        sd_ipc_worker0, not_parameter_names = rpc_results[0]
        state_dict = named_tensors_from_serialize(sd_ipc_worker0)
        self.underlying_model = ModuleLike(state_dict, not_parameter_names)
        return self.underlying_model

    def get_engine(self):
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine

    def is_engine_initialized(self):
        return self._engine_initialized
