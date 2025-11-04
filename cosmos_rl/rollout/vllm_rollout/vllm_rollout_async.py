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
import uuid
from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import apply_fp8_linear_patch

import torch
import copy
from typing import List, Optional, Dict, Tuple
from transformers import AutoTokenizer, AutoConfig
from transformers import GenerationConfig
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine, AsyncEngineArgs
from vllm import SamplingParams
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.dispatcher.data.packer.multi_turn import (
    ConversationType,
)
from cosmos_rl.dispatcher.data import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vllm_version_check
from cosmos_rl.utils.ipc.tensor_util import (
    named_tensors_to_serialize,
    named_tensors_from_serialize,
)


class VLLMColocateWorkerExtension:
    """
    The extension designed to shared weight between the main process and the worker process via IPC.
    This way, the code can be compatible with both vLLM V0 and V1.
    NOTE: this class in a extension module, to use this class, you should pass the full qualified
    name as `worker_extension_cls` argument to the AsyncLLMEngine.from_engine_args() method.
    """

    def get_state_dict_ipc(self) -> Dict[str, Tuple]:
        """
        Get the CUDA IPC handles of the model weights.

        Returns:
            Dict[param_name, (ipc_handle, shape, dtype, device_id)]
        """
        model: torch.nn.Module = self.model_runner.model
        return named_tensors_to_serialize(model.named_parameters())


class vLLMRolloutAsync(RolloutBase):
    def __init__(self, config: Config, tokenizer: AutoTokenizer, **kwargs):
        """Rollout with vLLM as the backend.

        Args:
            config: Cosmos Config.
            tokenizer: Tokenizer of the model.
            hf_config_path: huggingface config file path.
            model_hf_config: the huggingface config to initiallize the generating model in vllm
        """
        super().__init__(config, tokenizer)
        policy_config = self.config.policy
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation

        vllm_version_check(self.rollout_config)

        model_path = policy_config.model_name_or_path

        self.model_config = util.retry(AutoConfig.from_pretrained)(model_path)

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
                from vllm.config import set_current_vllm_config

                vllm_config = self.rollout_engine.llm_engine.vllm_config
                with set_current_vllm_config(vllm_config):
                    apply_fp8_linear_patch(self.get_underlying_model())

    def shutdown(self):
        if self._engine_initialized:
            self._engine_initialized = False
            self.rollout_engine.shutdown()

    @torch.no_grad()
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

        response: List[RolloutResult] = []

        stream = torch.cuda.current_stream() if stream is None else stream
        try:
            with torch.cuda.stream(stream):
                async for result in self.rollout_engine.generate(
                    prompt=new_prompts[0],
                    sampling_params=sampling_params,
                    # there is no request tracking now.
                    request_id=str(uuid.uuid4()),
                ):
                    if result.finished:
                        response.append(
                            RolloutResult(
                                prompt=payloads[0].prompt,
                                completions=[result.outputs[0].text],
                            )
                        )
                        break
        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []
        return response

    @torch.no_grad()
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
        stream = torch.cuda.current_stream() if stream is None else stream

        request_id = str(uuid.uuid4())

        async def generation_multi_turn_for_one_payload(
            current_conversation: ConversationType,
        ):
            assistant_turn_count = 0
            assert (
                payload.conversation is not None
            ), "Conversation should not be None for multi-turn rollout generation."
            while (
                assistant_turn_count
                < self.rollout_config.multi_turn_config.max_assistant_turns
            ):
                # Pack the payloads into prompts for vllm.
                prompts = [data_packer.get_rollout_input(current_conversation)]
                prompts = data_packer.rollout_collate_fn(prompts)

                with torch.cuda.stream(stream):
                    async for result in self.rollout_engine.generate(
                        prompt=prompts[0],
                        sampling_params=sampling_params,
                        request_id=request_id,
                    ):
                        if result.finished:
                            responses = [result.outputs[0].text]
                            break

                # TODO(zjx): support multi-path conversations search for multi-turn rollout generation
                # extend the conversation with the rollout result
                current_conversation = data_packer.extend_conversation(
                    current_conversation,
                    responses,
                    ground_truth=payload.reference_answer,
                )

                # check if the sequence length is reached the max_sequence_length
                if (
                    len(result.prompt_token_ids) + len(result.outputs[0].token_ids)
                    > self.rollout_config.max_response_length
                ):
                    logger.warning(
                        "[Rollout] The sequence length is reached the max_response_length, stop the multi-turn generation."
                    )
                    break

                assistant_turn_count += 1

            # return the last assistant message as the completion to compute the reward in controller
            completion = current_conversation[-1].content
            return current_conversation, completion

        n_generation = sampling_params.n
        sampling_params = copy.deepcopy(sampling_params)
        sampling_params.n = 1
        response: List[RolloutResult] = []
        for payload in payloads:
            conversations = []
            completions = []
            for _ in range(n_generation):
                (
                    new_conversation,
                    completion,
                ) = await generation_multi_turn_for_one_payload(
                    copy.deepcopy(payload.conversation)
                )
                conversations.append(new_conversation)
                completions.append(completion)

            response.append(
                RolloutResult(
                    conversation=payload.conversation,
                    completions=completions,
                    completed_conversations=conversations,
                )
            )

        return response

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
        # TODO(zjx): get tensor via ipc in the future
        return torch.nn.Module()

    async def get_underlying_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get the state dict of the underlying model.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        named_tensors_ipc = await self.rollout_engine.collective_rpc(
            "get_state_dict_ipc"
        )
        sd_ipc_worker0 = named_tensors_ipc[0]
        return named_tensors_from_serialize(sd_ipc_worker0)

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

    def mxfp4_quantization(self, weight: torch.Tensor):
        """
        Quantize the original bf16 weight sent by policy to mxfp4 weight.
        """
        # https://github.com/vllm-project/vllm/pull/22259
        # Note: vLLM use triton kernel for mxfp4 moe when ep not specified.
        # We temporarily support this case first.
        # Reference: https://github.com/zyongye/vllm/blob/6a70830065701b163e36a86fd331b41b5feac401/vllm/model_executor/layers/quantization/mxfp4.py#L493

        # Note: For mxfp4 quantizaiton, vLLM will load original mxfp4 weight from hf fp4 weight, and do some post processing like padding and swizzle.
        # So we have two phases for quantization:
        # 1. Quantize the original bf16 weight sent by policy:
        # We use: https://github.com/openai/gpt-oss/blob/d0a300a40d6502a1bdd73d18464f3d69440656e0/gpt_oss/triton/model.py#L302

        # 2. Post process the quantized weight as vLLM did for triton kernel:
        # https://github.com/zyongye/vllm/blob/6a70830065701b163e36a86fd331b41b5feac401/vllm/model_executor/layers/quantization/mxfp4.py#L173
        # mxfp4_block_size = 32
        weight = weight.transpose(-2, -1).contiguous()
        # weight is bf16 moe weight with shape:
        # gate_up_proj: [num_experts, hidden_size, 2 * intermediate_size]
        # donw_proj:    [num_experts, intermediate_size, hidden_size]

        # 1. Quantize the original bf16 weight sent by policy:
        from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import quantize_mx4

        # weight_mxfp4 and weight_scale_mxfp4 are torch.Tensor
        weight_mxfp4, weight_scale_mxfp4 = quantize_mx4(weight.to(torch.bfloat16))
        weight_mxfp4 = weight_mxfp4.transpose(-2, -1).contiguous()  # Now torch.Tensor
        weight_scale_mxfp4 = weight_scale_mxfp4.transpose(-2, -1).contiguous()
        # For weight_mxfp4:
        # [num_experts, 2 * intermediate_size, hidden_size // mxfp4_block_size, 16] for gate_up_proj
        # [num_experts, hidden_size, intermediate_size // mxfp4_block_size, 16] for down_proj
        # For weight_scale_mxfp4:
        # [num_experts, 2 * intermediate_size, hidden_size // mxfp4_block_size] for gate_up_proj
        # [num_experts, hidden_size, intermediate_size // mxfp4_block_size] for down_proj

        # 2. Post process
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            _swizzle_mxfp4,
        )

        num_warps = 8
        swizzled_weight_mxfp4, _, swizzled_weight_scale_mxfp4 = _swizzle_mxfp4(
            weight_mxfp4, weight_scale_mxfp4, num_warps
        )
        return (
            swizzled_weight_mxfp4.storage.data,
            swizzled_weight_scale_mxfp4.storage.data,
        )

    def model_param_map(self, weight_mapper: WeightMapper) -> Dict[str, torch.Tensor]:
        """
        All the parameters of the rollout model:
            - All the parameters of the model.
            - All the scales of quantized weights.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        if self._model_param_map:
            return self._model_param_map
        model = self.get_underlying_model()
        param_map = {}
        for name, param in model.state_dict().items():
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(name)
            param_map[compatible_name] = param

        quantized_tensors = self.get_quantized_tensors(weight_mapper)
        param_map.update(quantized_tensors)

        self._model_param_map = param_map
        return self._model_param_map

    def preset_vllm_env(self):
        def log_env(env_name: str, env_value: str):
            logger.info(f"[Rollout] Setting vLLM {env_name} to {env_value}")
            os.environ[env_name] = env_value

        # disable VLLM_DISABLE_COMPILE_CACHE
        log_env("VLLM_DISABLE_COMPILE_CACHE", "1")

        # if flashinfer config is not enabled, avoid importing flashinfer
        if self.config.rollout.vllm_use_flashinfer:
            try:
                import flashinfer  # noqa: F401
            except ImportError:
                logger.warning(
                    "[Rollout] flashinfer is not installed, ignore rollout.vllm_use_flashinfer setting."
                )
            else:
                log_env("VLLM_ATTENTION_BACKEND", "FLASHINFER")

        if self.config.rollout.sampling_config.use_flashinfer:
            try:
                import flashinfer  # noqa: F401
            except ImportError:
                logger.warning(
                    "[Rollout] flashinfer is not installed, ignore rollout.sampling_config.use_flashinfer setting."
                )
            else:
                log_env("VLLM_USE_FLASHINFER_SAMPLER", "1")

        # Model specific logic
        model_type = self.model_config.model_type
        if model_type == "gpt_oss" and self.config.rollout.quantization == "mxfp4":
            # We disable flashinfer kernel for now temporarily in mxfp4 quantization
            log_env("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16", "0")
            log_env("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "0")
            log_env("VLLM_MXFP4_USE_MARLIN", "0")

    def get_quantized_tensors(
        self, weight_mapper: WeightMapper
    ) -> Dict[str, torch.Tensor]:
        """
        Get the quantized tensors of the rollout model.
        """
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        model = self.get_underlying_model()
        quantized_tensors = {}
        # Handle special cases for some quantized models
        if "gpt_oss" in self.model_config.model_type and self.quantization == "mxfp4":
            # FIXME: (lms) generally handle all quantized cases when refactoring the rollout param cache.
            # iterate all the modules in the model
            for module_name, module in model.named_modules():
                if hasattr(module, "w13_bias"):
                    # this is a mxfp4 quant layer
                    w13_weight_name = f"{module_name}.w13_weight"
                    w2_weight_name = f"{module_name}.w2_weight"
                    w13_compatible_name = weight_mapper._rollout_vllm_name_to_hf(
                        w13_weight_name
                    )
                    w2_compatible_name = weight_mapper._rollout_vllm_name_to_hf(
                        w2_weight_name
                    )
                    quantized_tensors[w13_compatible_name] = (
                        module.quant_method.w13_weight_triton_tensor.storage.data
                    )
                    quantized_tensors[w2_compatible_name] = (
                        module.quant_method.w2_weight_triton_tensor.storage.data
                    )
                    quantized_tensors[w13_compatible_name + "_scale"] = (
                        module.quant_method.w13_precision_config.weight_scale.storage.data
                    )
                    quantized_tensors[w2_compatible_name + "_scale"] = (
                        module.quant_method.w2_precision_config.weight_scale.storage.data
                    )

        return quantized_tensors
