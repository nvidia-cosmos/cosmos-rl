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

import types
import os
import weakref
from pydantic import Field

from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi.llm_utils import CachedModelLoader
import tensorrt_llm.llmapi.llm_args as tllm_llm_args
from tensorrt_llm.llmapi.inputs import create_input_processor
from tensorrt_llm.llmapi.llm_args import PybindMirror
from tensorrt_llm.llmapi.tokenizer import _xgrammar_tokenizer_info
from tensorrt_llm.builder import EngineConfig
from tensorrt_llm.llmapi.mpi_session import external_mpi_comm_available
from tensorrt_llm.llmapi.utils import print_colored
from tensorrt_llm.executor import PostprocWorkerConfig
from tensorrt_llm import bindings as tllm

from cosmos_rl.policy.config import Config as CosmosConfig


"""
Patches for trtllm.
"""


def add_method(obj, method, method_name):
    setattr(obj, method_name, types.MethodType(method, obj))


# 1. Add cosmos config to trtllm ExecutorConfig. patch the _build_model of `LLM` class.


def patch_trtllm_llm_args():
    class CosmosLLMArgs(tllm_llm_args.LlmArgs):
        cosmos_config = Field(default_factory=CosmosConfig)

    tllm_llm_args.LlmArgs = CosmosLLMArgs


patch_trtllm_llm_args()


def patch_trtllm_build_model(llm: LLM):
    def cosmos_build_model(self, *args, **kwargs):
        model_loader = CachedModelLoader(
            self.args,
            mpi_session=self.mpi_session,
            workspace=self.workspace,
            llm_build_stats=weakref.proxy(self.llm_build_stats),
        )
        self._engine_dir, self._hf_model_dir = model_loader()
        # update the model_dir to a local dir for the runtime, such as tokenizer loading.
        if self._engine_dir is not None:
            self.args.model = self._engine_dir

        # Tokenizer loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        self.input_processor = create_input_processor(
            self._hf_model_dir, self.tokenizer
        )
        self.tokenizer = self.input_processor.tokenizer

        max_batch_size = (
            self.args.max_batch_size or self.args.build_config.max_batch_size
        )
        max_num_tokens = (
            self.args.max_num_tokens or self.args.build_config.max_num_tokens
        )
        max_seq_len = self.args.max_seq_len or self.args.build_config.max_seq_len
        executor_config = tllm.ExecutorConfig(
            max_beam_width=self.args.build_config.max_beam_width,
            scheduler_config=PybindMirror.maybe_to_pybind(self.args.scheduler_config),
            batching_type=PybindMirror.maybe_to_pybind(self.args.batching_type)
            or tllm.BatchingType.INFLIGHT,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            gather_generation_logits=self.args.gather_generation_logits,
        )
        if self.args.backend is None:
            # also set executor_config.max_seq_len in TRT workflow, to deduce default max_tokens
            if max_seq_len is not None:
                executor_config.max_seq_len = max_seq_len
            else:
                engine_config = EngineConfig.from_json_file(
                    self._engine_dir / "config.json"
                )
                executor_config.max_seq_len = engine_config.build_config.max_seq_len
        if self.args.kv_cache_config is not None:
            executor_config.kv_cache_config = PybindMirror.maybe_to_pybind(
                self.args.kv_cache_config
            )
        if os.getenv("FORCE_DETERMINISTIC", "0") == "1":
            # Disable KV cache reuse for deterministic mode
            executor_config.kv_cache_config.enable_block_reuse = False
            executor_config.kv_cache_config.enable_partial_reuse = False
        if self.args.peft_cache_config is not None:
            executor_config.peft_cache_config = PybindMirror.maybe_to_pybind(
                self.args.peft_cache_config
            )
        elif self.args.build_config.plugin_config.lora_plugin:
            engine_config = EngineConfig.from_json_file(
                self._engine_dir / "config.json"
            )
            lora_config = engine_config.build_config.lora_config
            max_lora_rank = lora_config.max_lora_rank
            num_lora_modules = engine_config.pretrained_config.num_hidden_layers * len(
                lora_config.lora_target_modules + lora_config.missing_qkv_modules
            )
            executor_config.peft_cache_config = tllm.PeftCacheConfig(
                num_device_module_layer=max_lora_rank
                * num_lora_modules
                * self.args.max_loras,
                num_host_module_layer=max_lora_rank
                * num_lora_modules
                * self.args.max_cpu_loras,
            )
        if self.args.decoding_config is not None:
            executor_config.decoding_config = self.args.decoding_config
        if self.args.guided_decoding_backend == "xgrammar":
            executor_config.guided_decoding_config = tllm.GuidedDecodingConfig(
                backend=tllm.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
                **_xgrammar_tokenizer_info(self.tokenizer),
            )
        elif self.args.guided_decoding_backend is not None:
            raise ValueError(
                f"Unrecognized guided decoding backend {self.args.guided_decoding_backend}"
            )

        executor_config.normalize_log_probs = self.args.normalize_log_probs
        # lms: chunked prefill
        executor_config.enable_chunked_context = self.args.enable_chunked_prefill
        executor_config.max_beam_width = (
            self.args.max_beam_width or self.args.build_config.max_beam_width
        )
        if self.args.extended_runtime_perf_knob_config is not None:
            executor_config.extended_runtime_perf_knob_config = (
                PybindMirror.maybe_to_pybind(
                    self.args.extended_runtime_perf_knob_config
                )
            )
        if self.args.cache_transceiver_config is not None:
            executor_config.cache_transceiver_config = PybindMirror.maybe_to_pybind(
                self.args.cache_transceiver_config
            )
        from tensorrt_llm._torch.pyexecutor.config import update_executor_config

        update_executor_config(
            executor_config,
            backend=self.args.backend,
            pytorch_backend_config=self.pytorch_backend_config,
            mapping=self.args.parallel_config.to_mapping(),
            build_config=self.args.build_config,
            speculative_config=self.args.speculative_config,
            hf_model_dir=self._hf_model_dir,
            trt_engine_dir=self._engine_dir,
            max_input_len=self.args.max_input_len,
            max_seq_len=max_seq_len,
        )
        # lms: parallel config
        executor_config.llm_parallel_config = self.args.parallel_config

        """
        Cosmos-RL modification:
        - Add cosmos config to executor_config.
        """
        executor_config.cosmos_config = self.args.cosmos_config

        return_logits = self.args.gather_generation_logits or (
            self.args.build_config and self.args.build_config.gather_context_logits
        )
        # lms: create PyExecutor
        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=executor_config,
            batched_logits_processor=self.args.batched_logits_processor,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size
            ),
            return_logits=return_logits,
            postproc_worker_config=PostprocWorkerConfig(
                num_postprocess_workers=self.args.num_postprocess_workers,
                postprocess_tokenizer_dir=self.args.postprocess_tokenizer_dir,
            ),
            is_llm_executor=True,
            lora_config=self.args.lora_config,
        )
        print_colored(f"LMS: create executor of {self._executor}\n", "yellow")

    add_method(llm, cosmos_build_model, "_build_model")
