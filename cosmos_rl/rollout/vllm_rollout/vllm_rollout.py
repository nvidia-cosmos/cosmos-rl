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

from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
    monkey_patch_for_fp8,
    simplify_process_weights_after_loading_for_fp8,
)

import vllm
import torch
import copy
from typing import List, Optional, Dict
from transformers import AutoConfig
from transformers import GenerationConfig
from vllm.entrypoints.llm import LLM
from vllm import SamplingParams
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
from cosmos_rl.policy.config import RolloutConfig
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.utils.tools_use import ToolParser
from cosmos_rl.dispatcher.data.packer.multi_turn import (
    ConversationType,
    add_tool_response_messages,
    add_assistant_message,
)
from cosmos_rl.utils.tools_use import OpenAIFunctionToolSchema
from cosmos_rl.dispatcher.data import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.dispatcher.command import (
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    Command,
)
import threading
import types
from functools import partial
from cosmos_rl.utils.constant import (
    COSMOS_ROLLOUT_STEP_INTERVAL,
    COSMOS_ROLLOUT_REPORT_INTERVAL,
)


def vllm_version_check(rollout_config: RolloutConfig):
    vllm_version = vllm.__version__
    if vllm_version < "0.9.0" and rollout_config.parallelism.pp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not supported for vLLM < 0.9.0, current version is %s"
            % vllm_version
        )


def _patch_vllm_rollout_locked_step(
    rollout, consume_command, reward_fetch, enable_validation
):
    llm_engine = rollout.get_engine().llm_engine
    orig_step = llm_engine.step

    def cmd_pred(cmd: Command, enable_validation: threading.Event):
        # Make sure no weight update happens during validation.
        # So filter out R2R and P2R commands when validation is enabled.
        if enable_validation.is_set() and (
            isinstance(cmd, RolloutToRolloutBroadcastCommand)
            or isinstance(cmd, PolicyToRolloutUnicastCommand)
        ):
            return False
        return True

    def step(self, *args, **kwargs):
        if not hasattr(self, "_cosmos_step_counter"):
            self._cosmos_step_counter = 0
        self._cosmos_step_counter += 1

        if (
            COSMOS_ROLLOUT_REPORT_INTERVAL > 0
            and self._cosmos_step_counter % COSMOS_ROLLOUT_REPORT_INTERVAL == 0
        ):
            _, is_validation, _, _ = reward_fetch()
            assert not is_validation, "Validation report should be handled in the broadcast command rather than step function."

        if (
            COSMOS_ROLLOUT_STEP_INTERVAL > 0
            and self._cosmos_step_counter % COSMOS_ROLLOUT_STEP_INTERVAL == 0
        ):
            # IMPORTANT:
            # If validation is enabled, R2R is not expected to be called in this step function
            # to avoid recursive inference execution.
            consume_command(
                cmd_pred=partial(cmd_pred, enable_validation=enable_validation)
            )
        return orig_step(*args, **kwargs)

    llm_engine.step = types.MethodType(step, llm_engine)


def update_conversation_wth_rollout_result(
    conversation: ConversationType,
    rollout_result: str,
    tool_parser: ToolParser,
    tools: list[OpenAIFunctionToolSchema],
) -> ConversationType:
    """
    Update the conversation with the rollout result.

    1. if the rollout result contains tool calls, add the tool response messages to the conversation
    2. otherwise, add the assistant message to the conversation
    """
    content, function_calls = tool_parser.extract_tool_calls(rollout_result)

    if function_calls:
        conversation = add_tool_response_messages(conversation, function_calls)
    else:
        conversation = add_assistant_message(conversation, content)

    return conversation


@RolloutRegistry.register(rollout_type="vllm")
class vLLMRollout(RolloutBase):
    def __init__(
        self,
        config: Config,
        parallel_dims: ParallelDims,
        device: torch.device,
        **kwargs,
    ):
        """Rollout with vLLM as the backend.

        Args:
            config: Cosmos Config.
            parallel_dims: Parallel dimensions for the rollout engine.
            device: The device on which the rollout engine will run.
            hf_config_path: huggingface config file path.
            model_hf_config: the huggingface config to initiallize the generating model in vllm
        """
        super().__init__(config, parallel_dims, device, **kwargs)

    def post_init_hook(self, **kwargs):
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation

        policy_config = self.config.policy

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

        self.rollout_engine = None
        self.is_vlm = getattr(self.model_config, "vision_config", None) is not None

        self.preset_vllm_env()

        # Set sampling params for rollout and validation.
        self.val_sampling_params = SamplingParams(
            n=self.config.validation.n_generation,
            logprobs=0,
            top_p=self.config.validation.top_p
            if self.config.validation.top_p is not None
            else self.config.rollout.sampling_config.top_p,
            top_k=self.config.validation.top_k
            if self.config.validation.top_k is not None
            else self.config.rollout.sampling_config.top_k,
            temperature=self.config.validation.temperature
            if self.config.validation.temperature is not None
            else self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.validation.repetition_penalty
            if self.config.validation.repetition_penalty is not None
            else self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.validation.max_response_length
            if self.config.validation.max_response_length is not None
            else self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )
        self.sampling_params = SamplingParams(
            n=self.config.rollout.n_generation,
            logprobs=0,
            top_p=self.config.rollout.sampling_config.top_p,
            top_k=self.config.rollout.sampling_config.top_k,
            temperature=self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
            prompt_logprobs=0,
        )

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
            llm_additional_kwargs = {}

            # Check if the model has MoE
            # Note: even though deepseek_v3 is MoE, EP in rollout is not supported for it yet
            moe_model_type = {"qwen3_moe", "qwen3_vl_moe", "deepseek_v3"}
            multimodal_type = {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}

            model_type = self.model_config.model_type
            if model_type in moe_model_type:
                enable_ep_parallelism = True
            if model_type in multimodal_type:
                # for vllm nightly, this is only True for multimodal models, check here
                llm_additional_kwargs["mm_processor_cache_gb"] = 0
            assert tp_size * pp_size == rollout_parallelism.world_size, (
                "[Rollout] For tensor parallel, the tp_size * pp_size must be equal to world size, but got tp_size: %d, pp_size: %d, world_size: %d"
                % (tp_size, pp_size, rollout_parallelism.world_size)
            )

            self.quantization = quantization

            policy_config = self.config.policy

            if seed is not None and seed < 0:
                seed = None

            # patch the vllm model to use rowwise fp8
            if self.quantization == "fp8":
                # patch for weight quantization. [weight loading]
                # patch must happen before `rollout_engine` is initialized.
                # vllm_config = self.rollout_engine.llm_engine.vllm_config
                simplify_process_weights_after_loading_for_fp8()

            logger.info(
                f"[Rollout] Initializing vLLM engine with quantization: {self.quantization}"
            )
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
                seed=seed,
                load_format=load_format,
                **llm_additional_kwargs,
            )
            self._engine_initialized = True
            logger.info("[Rollout] Engine initialized.")
            if self.quantization == "fp8":
                # Patch for computing kernel in rowwise fp8 manner. [computation]
                monkey_patch_for_fp8(
                    self.rollout_engine.llm_engine.vllm_config,
                    self.get_underlying_model(),
                )
            # initialization done.

    def post_init_engine_hook(
        self, consume_command_hook, report_rollouts_hook, validation_flag, **kwargs
    ):
        """
        Post initialization hook for the engine, which will be called after the engine is initialized.
        """
        _patch_vllm_rollout_locked_step(
            self,
            consume_command_hook,
            report_rollouts_hook,
            validation_flag,
        )

    def pre_get_params_for_sync_hook(
        self, quantization_type, weight_mapper, parallel_dims, **kwargs
    ):
        """
        Pre get sync param hook for the engine, which will be called before getting the sync params.
        """
        if quantization_type == "fp8":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
                cache_weight_of_quantized_module,
                replace_weight_of_quantized_module,
            )
        elif quantization_type == "mxfp4":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import (
                cache_weight_of_quantized_module,
                replace_weight_of_quantized_module,
            )

        vllm_hp_weight_map, vllm_quantized_weight_map = None, None
        if quantization_type is not None:
            promotion_dtype = util.str2torch_dtype(self.config.train.param_dtype)
            vllm_hp_weight_map, vllm_quantized_weight_map = (
                cache_weight_of_quantized_module(
                    self.get_underlying_model(),
                    promotion_dtype,
                    weight_mapper,
                    parallel_dims,
                )
            )
            # replace the weight of quantized module with the high precision weight.
            # let weight in vllm_weight_inplace_view_map always in high precision for recv
            # high precision weight from policy.
            replace_weight_of_quantized_module(
                self.get_underlying_model(),
                vllm_hp_weight_map,
                weight_mapper,
            )
        return vllm_hp_weight_map, vllm_quantized_weight_map

    def post_get_params_for_sync_hook(
        self,
        quantization_type,
        weight_mapper,
        weight_view_map,
        quantized_weight_map,
        **kwargs,
    ):
        """
        Post get sync param hook for the engine, which will be called after getting the sync params.
        """
        if quantization_type == "fp8":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
                replace_weight_of_quantized_module,
            )
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import (
                post_process_view_map_for_fp8 as post_process_view_map_for_lowp,
            )
        elif quantization_type == "mxfp4":
            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import (
                replace_weight_of_quantized_module,
            )

            from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_mxfp4 import (
                post_process_view_map_for_mxfp4 as post_process_view_map_for_lowp,
            )

        if quantization_type is not None:
            weight_view_map = post_process_view_map_for_lowp(weight_view_map)
            # Get vllm weight back into quantized.
            replace_weight_of_quantized_module(
                self.get_underlying_model(),
                quantized_weight_map,
                weight_mapper,
            )
        return weight_view_map

    @staticmethod
    def parse_logprobs(logprobs: List[dict]) -> List[float]:
        if logprobs is None:
            return []
        ret_logprobs = []
        ret_token_ids = []
        for logp in logprobs:
            assert (
                len(logp) == 1
            ), f"[Rollout] logprobs length should be 1, but got {logp}."
            ret_logprobs.append(list(logp.values())[0].logprob)
            ret_token_ids.append(list(logp.keys())[0])
        return ret_logprobs, ret_token_ids

    @torch.no_grad()
    def rollout_generation_single_turn(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        sampling_params: SamplingParams,
    ) -> List[RolloutResult]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        n_repeats = (
            sampling_params.n if self.config.rollout.n_generation_to_batch else 1
        )
        payloads = [p for p in payloads for _ in range(n_repeats)]
        if self.config.rollout.n_generation_to_batch:
            local_sampling_params = copy.deepcopy(sampling_params)
            local_sampling_params.n = 1
        else:
            local_sampling_params = sampling_params

        # Pack the payloads into prompts for vllm.
        prompts = []
        for pl in payloads:
            if not self.config.train.local_dataset:
                assert (
                    pl.prompt is not None
                ), "Prompt should not be None for single turn rollout generation."
            else:
                # Quert prompt from local dataset
                pass
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
                results = self.rollout_engine.generate(
                    new_prompts,
                    sampling_params=local_sampling_params,
                    use_tqdm=False,
                )
            assert len(results) % n_repeats == 0, (
                "[Rollout] The number of results %d is not divisible by n_repeats %d"
                % (len(results), n_repeats)
            )
            for i in range(0, len(results), n_repeats):
                outputs = results[i : i + n_repeats]
                logprobs = []
                token_ids = []
                prompt_logprobs = None
                # collect logprobs
                for output in outputs:
                    if self.config.train.train_policy.collect_rollout_logprobs:
                        if prompt_logprobs is None:
                            assert (
                                output.prompt_logprobs is not None
                                and len(output.prompt_logprobs) > 0
                            ), "Prompt logprobs should not be None or empty"
                            assert (
                                output.prompt_logprobs[0] is None
                            ), "Prompt logprobs should be None for the first token"
                            prompt_logprob, _ = self.parse_logprobs(
                                output.prompt_logprobs[1:]
                            )
                            prompt_logprobs = prompt_logprob

                    for j in range(len(output.outputs)):
                        if (
                            self.config.train.train_policy.rollout_as_token_ids
                            or self.config.train.train_policy.collect_rollout_logprobs
                        ):
                            logprob, token_id = self.parse_logprobs(
                                output.outputs[j].logprobs
                            )
                            if not self.config.train.train_policy.collect_rollout_logprobs:
                                logprob = []
                        else:
                            logprob = []
                            token_id = []
                        logprobs.append(logprob)
                        token_ids.append(token_id)
                response.append(
                    RolloutResult(
                        prompt=payloads[i].prompt,
                        completions=[
                            output.outputs[j].text
                            for output in outputs
                            for j in range(len(output.outputs))
                        ],
                        completion_logprobs=logprobs,
                        completion_token_ids=token_ids,
                        # Collect the cumulative logprob of the generated completions
                        # Used for reward calculation to find the most likely mode reward.
                        # This can indicate the most likelyhood of a generated completion.
                        cumulative_logprob=[
                            output.outputs[j].cumulative_logprob
                            for output in outputs
                            for j in range(len(output.outputs))
                        ],
                        prompt_logprobs=prompt_logprobs,
                    )
                )
        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []
        return response

    @torch.no_grad()
    def rollout_generation_multi_turn(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        sampling_params: SamplingParams,
    ) -> List[RolloutResult]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        stream = torch.cuda.current_stream() if stream is None else stream

        def generation_multi_turn_for_one_payload(
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
                    results = self.rollout_engine.generate(
                        prompts=prompts,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )

                assert (
                    len(results) == 1
                ), "[Rollout] Expected single result for multi-turn rollout generation"
                # TODO(zjx): support multi-path conversations search for multi-turn rollout generation
                # extend the conversation with the rollout result
                responses = [output.text for output in results[0].outputs]
                # Get token IDs (list of ints)

                # Manually decode to string
                logprobs = []
                token_ids = []
                prompt_logprobs = []
                if self.config.train.train_policy.collect_rollout_logprobs:
                    assert (
                        results[0].prompt_logprobs is not None
                        and len(results[0].prompt_logprobs) > 0
                    ), "Prompt logprobs should not be None or empty"
                    assert (
                        results[0].prompt_logprobs[0] is None
                    ), "Prompt logprobs should be None for the first token"
                    prompt_logprob, _ = self.parse_logprobs(
                        results[0].prompt_logprobs[1:]
                    )
                    prompt_logprobs = prompt_logprob

                if (
                    self.config.train.train_policy.rollout_as_token_ids
                    or not self.config.train.train_policy.collect_rollout_logprobs
                ):
                    assert (
                        len(results[0].outputs) == 1
                    ), "Expected single output for token ID extraction"
                    for output in results[0].outputs:
                        logprob, token_ids = self.parse_logprobs(output.logprobs)
                        if self.config.train.train_policy.collect_rollout_logprobs:
                            logprobs = logprob

                # Collect the cumulative logprob of the generated completions
                # Used for reward calculation to find the most likely mode reward.
                # This can indicate the most likelyhood of a generated completion.
                cumulative_logprob = [
                    output.cumulative_logprob for output in results[0].outputs
                ]

                current_conversation = data_packer.extend_conversation(
                    current_conversation,
                    responses,
                    ground_truth=payload.reference_answer,
                )

                # check if the sequence length is reached the max_sequence_length
                if (
                    len(results[0].prompt_token_ids)
                    + len(results[0].outputs[0].token_ids)
                    > self.rollout_config.max_response_length
                ):
                    logger.warning(
                        "[Rollout] The sequence length is reached the max_response_length, stop the multi-turn generation."
                    )
                    break

                assistant_turn_count += 1

            # return the last assistant message as the completion to compute the reward in controller
            completion = current_conversation[-1].content
            return (
                current_conversation,
                completion,
                logprobs,
                token_ids,
                cumulative_logprob,
                prompt_logprobs,
            )

        n_generation = sampling_params.n
        sampling_params = copy.deepcopy(sampling_params)
        sampling_params.n = 1
        response: List[RolloutResult] = []
        for payload in payloads:
            conversations = []
            completions = []
            logprobs_list = []
            token_ids_list = []
            cumulative_logprob_list = []
            prompt_logprobs_list = None
            for _ in range(n_generation):
                (
                    new_conversation,
                    completion,
                    logprobs,
                    token_ids,
                    cumulative_logprob,
                    prompt_logprobs,
                ) = generation_multi_turn_for_one_payload(
                    copy.deepcopy(payload.conversation)
                )
                conversations.append(new_conversation)
                completions.append(completion)
                logprobs_list.append(logprobs)
                token_ids_list.append(token_ids)
                cumulative_logprob_list.extend(cumulative_logprob)
                if prompt_logprobs_list is None:
                    prompt_logprobs_list = prompt_logprobs
                else:
                    assert (
                        prompt_logprobs_list == prompt_logprobs
                    ), "Prompt logprobs should be the same for all generations"
            response.append(
                RolloutResult(
                    conversation=payload.conversation,
                    completions=completions,
                    completed_conversations=conversations,
                    completion_logprobs=logprobs_list,
                    completion_token_ids=token_ids_list,
                    cumulative_logprob=cumulative_logprob_list,
                    prompt_logprobs=prompt_logprobs_list,
                )
            )

        return response

    def rollout_generation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        **kwargs,
    ) -> List[RolloutResult]:
        if self.rollout_config.multi_turn_config.enable:
            return self.rollout_generation_multi_turn(
                payloads,
                stream,
                data_packer,
                sampling_params=self.val_sampling_params
                if is_validation
                else self.sampling_params,
            )
        else:
            return self.rollout_generation_single_turn(
                payloads,
                stream,
                data_packer,
                sampling_params=self.val_sampling_params
                if is_validation
                else self.sampling_params,
            )

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

    def fp8_quantization(self, weight: torch.Tensor):
        """
        Quantize the weight to fp8.
        Args:
            weight: The weight to quantize, in high-precision dtype. If with shape [out_dim, in_dim], it's Linear case, if with shape [num_experts, out_dim, in_dim], it's MoE case.
        Returns:
            qweight: The quantized weight.
            weight_scale: The scale of the quantized weight.
        """
        from vllm import _custom_ops as ops

        # quantization of rowwise torch scaled_mm.
        # weight has shape [out_dim, in_dim]

        ndim = weight.dim()
        if ndim == 2:
            # Fp8LinearMethod
            # quantization of rowwise torch scaled_mm.
            # weight has shape [out_dim, in_dim]
            qweight, weight_scale = ops.scaled_fp8_quant(
                weight, scale=None, use_per_token_if_dynamic=True
            )

            return qweight.t(), weight_scale
        elif ndim == 3:
            # Fp8OnlineMoEMethod
            # per-tensor quantization for each expert.
            # weight has shape [num_experts, out_dim, in_dim]
            n_experts = weight.shape[0]
            qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
            weight_scale = torch.ones(
                n_experts, dtype=torch.float32, device=weight.device
            )
            for expert in range(n_experts):
                qweight[expert, :, :], weight_scale[expert] = ops.scaled_fp8_quant(
                    weight[expert, :, :]
                )

            return qweight, weight_scale

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
                    w13_compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(
                        w13_weight_name
                    )
                    w2_compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(
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
