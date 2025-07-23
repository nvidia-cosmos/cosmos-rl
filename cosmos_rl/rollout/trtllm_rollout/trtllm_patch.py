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
from typing import Optional
from pydantic import Field
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.model_engine import (
    KV_CACHE_MANAGER_KEY,
)
from tensorrt_llm.lora_manager import (
    LoraConfig,
    get_default_trtllm_modules_to_hf_modules,
    load_torch_hf_lora,
)
from tensorrt_llm.llmapi.llm_utils import CachedModelLoader
import tensorrt_llm.llmapi.llm_args as tllm_llm_args
from tensorrt_llm.inputs import create_input_processor
from tensorrt_llm.llmapi.llm_args import PybindMirror
from tensorrt_llm.llmapi.tokenizer import _xgrammar_tokenizer_info
from tensorrt_llm.builder import EngineConfig
from tensorrt_llm.llmapi.mpi_session import external_mpi_comm_available
from tensorrt_llm.llmapi.utils import print_colored
from tensorrt_llm.executor import PostprocWorkerConfig
from tensorrt_llm.bindings import executor as tllm_executor
from tensorrt_llm._torch.pyexecutor import _util as tllm_util
from tensorrt_llm._torch.pyexecutor._util import _try_infer_num_experts
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    PeftCacheManager,
    ResourceManager,
)
from tensorrt_llm._torch.pyexecutor.scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    SimpleScheduler,
)
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    AttentionTypeCpp,
    create_kv_cache_transceiver,
)
from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_mla,
)

import tensorrt_llm.executor.worker as tllm_worker

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.rollout.trtllm_rollout.trtllm_worker import CosmosTRTLLMWorker

"""
Patches for trtllm.
"""


def add_method(obj, method, method_name):
    setattr(obj, method_name, types.MethodType(method, obj))


# 1. Add cosmos config to trtllm ExecutorConfig and `LLMArgs`. patch the _build_model of `LLM` class.


def patch_trtllm_llm_args():
    class CosmosLLMArgs(tllm_llm_args.LlmArgs):
        cosmos_config: Optional[CosmosConfig] = Field(
            default=None, description="Cosmos config that hacked by cosmos-rl"
        )

    tllm_llm_args.LlmArgs = CosmosLLMArgs


patch_trtllm_llm_args()


# 2. Patch the `create_py_executor_instance`, let PyExecutor to have a cosmos_config field and do the
# cosmos-rl specific initialization.


def extend_create_py_executor_instance():
    def create_py_executor_instance(
        dist,
        resources,
        mapping,
        pytorch_backend_config,
        executor_config,
        ctx_chunk_config,
        model_engine,
        draft_model_engine,
        start_worker,
        sampler,
        lora_config: Optional[LoraConfig] = None,
    ) -> PyExecutor:
        kv_cache_manager = resources.get(KV_CACHE_MANAGER_KEY, None)

        spec_config = model_engine.spec_config
        if (
            mapping.is_last_pp_rank()
            and executor_config.guided_decoding_config is not None
        ):
            if spec_config is not None:
                raise ValueError(
                    "Guided decoding is not supported with speculative decoding."
                )
            if not pytorch_backend_config.disable_overlap_scheduler:
                raise ValueError(
                    "Guided decoding is not supported with overlap scheduler."
                )

        logger.info(
            f"max_seq_len={executor_config.max_seq_len}, max_num_requests={executor_config.max_batch_size}, max_num_tokens={executor_config.max_num_tokens}"
        )

        for key, value in pytorch_backend_config.extra_resource_managers.items():
            if key in resources:
                raise ValueError(f"Cannot overwrite existing resource manager {key}.")
            resources[key] = value

        if lora_config is not None:
            from tensorrt_llm.bindings import LoraModule

            if len(lora_config.lora_dir) == 1:
                load_torch_hf_lora(lora_config)
            else:
                assert (
                    len(lora_config.lora_target_modules) >= 1
                ), "Expecting at least one lora target module"
                if not bool(lora_config.trtllm_modules_to_hf_modules):
                    lora_config.trtllm_modules_to_hf_modules = (
                        get_default_trtllm_modules_to_hf_modules()
                    )

            model_binding_config = (
                model_engine.model.model_config.get_bindings_model_config()
            )

            num_experts = _try_infer_num_experts(model_engine.model.model_config)

            lora_modules = LoraModule.create_lora_modules(
                lora_module_names=lora_config.lora_target_modules,
                hidden_size=model_binding_config.hidden_size,
                mlp_hidden_size=model_binding_config.mlp_hidden_size,
                num_attention_heads=model_binding_config.num_heads,
                num_kv_attention_heads=model_binding_config.num_heads,
                attention_head_size=model_binding_config.head_size,
                tp_size=mapping.tp_size,
                num_experts=num_experts,
            )
            model_binding_config.use_lora_plugin = True
            model_binding_config.lora_modules = lora_modules
            model_binding_config.max_lora_rank = lora_config.max_lora_rank

            max_lora_rank = lora_config.max_lora_rank
            num_lora_modules = (
                model_engine.model.model_config.pretrained_config.num_hidden_layers
                * len(lora_config.lora_target_modules + lora_config.missing_qkv_modules)
            )

            # TODO smor- need to figure out how to set these values
            executor_config.peft_cache_config = tllm_executor.PeftCacheConfig(
                num_device_module_layer=max_lora_rank
                * num_lora_modules
                * lora_config.max_loras,
                num_host_module_layer=max_lora_rank
                * num_lora_modules
                * lora_config.max_cpu_loras,
            )

            from tensorrt_llm.bindings import WorldConfig

            world_config = WorldConfig(
                tensor_parallelism=mapping.tp_size,
                pipeline_parallelism=mapping.pp_size,
                context_parallelism=mapping.cp_size,
                rank=dist.mapping.rank,
                gpus_per_node=dist.mapping.gpus_per_node,
            )
            peft_cache_manager = PeftCacheManager(
                peft_cache_config=executor_config.peft_cache_config,
                model_config=model_binding_config,
                world_config=world_config,
            )
            resources["peft_cache_manager"] = peft_cache_manager
            model_engine.set_lora_model_config(
                lora_config.lora_target_modules,
                lora_config.trtllm_modules_to_hf_modules,
            )

        max_num_sequences = executor_config.max_batch_size * mapping.pp_size

        resources["seq_slot_manager"] = SeqSlotManager(max_num_sequences)

        resource_manager = ResourceManager(resources)

        # Make sure the kv cache manager is always invoked last as it could
        # depend on the results of other resource managers.
        if kv_cache_manager is not None:
            resource_manager.resource_managers.move_to_end(
                "kv_cache_manager", last=True
            )

        capacity_scheduler = BindCapacityScheduler(
            max_num_sequences,
            kv_cache_manager.impl if kv_cache_manager is not None else None,
            executor_config.scheduler_config.capacity_scheduler_policy,
            two_step_lookahead=mapping.has_pp()
            or not pytorch_backend_config.disable_overlap_scheduler,
        )
        mb_scheduler = BindMicroBatchScheduler(
            executor_config.max_batch_size,
            executor_config.max_num_tokens,
            ctx_chunk_config,
        )
        scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)

        config = model_engine.model.model_config.pretrained_config
        attention_type = (
            AttentionTypeCpp.MLA if is_mla(config) else AttentionTypeCpp.DEFAULT
        )
        cache_transceiver_config = executor_config.cache_transceiver_config
        kv_cache_transceiver = create_kv_cache_transceiver(
            mapping, kv_cache_manager, attention_type, cache_transceiver_config
        )

        return CosmosTRTLLMWorker(
            resource_manager,
            scheduler,
            model_engine=model_engine,
            sampler=sampler,
            dist=dist,
            disable_overlap_scheduler=pytorch_backend_config.disable_overlap_scheduler,
            max_batch_size=executor_config.max_batch_size,
            max_draft_tokens=spec_config.max_draft_tokens
            if spec_config is not None
            else 0,
            kv_cache_transceiver=kv_cache_transceiver,
            draft_model_engine=draft_model_engine,
            start_worker=start_worker,
        )

    tllm_util.create_py_executor_instance = create_py_executor_instance


extend_create_py_executor_instance()


def patch_trtllm_create_py_executor_instance():
    original_create_py_executor_instance = tllm_util.create_py_executor_instance

    def cosmos_create_py_executor_instance(*args, **kwargs):
        executor_config = None
        for arg in args:
            if isinstance(arg, tllm_executor.ExecutorConfig):
                executor_config = arg
                break
        assert executor_config is not None, "Executor config not found in args."
        cosmos_config = executor_config.cosmos_config
        py_executor = original_create_py_executor_instance(
            *args, **kwargs
        )  # PyExecutor has been replaced by CosmosTRTLLMWorker
        logger.info(f"LMS: cosmos_config: {cosmos_config}")
        assert hasattr(
            py_executor, "set_cosmos_config"
        ), "PyExecutor must have a set_cosmos_config method"
        py_executor.set_cosmos_config(
            cosmos_config
        )  # set the cosmos config to the PyExecutor and do the cosmos-rl specific initialization.
        return py_executor

    tllm_util.create_py_executor_instance = cosmos_create_py_executor_instance


patch_trtllm_create_py_executor_instance()


# 3. Patch the `worker_main`, let it init the torch distributed environment that cosmos-rl uses.
def patch_trtllm_worker_main():
    original_worker_main = tllm_worker.worker_main

    def worker_main(*args, **kwargs):
        # init the torch distributed environment.
        # logger.info(f"LMS: init torch distributed environment")
        # rdzv_endpoint = os.environ.get("RDZV_ENDPOINT", "127.0.0.1:12371")
        # rdzv_host, rdzv_port = rdzv_endpoint.split(":")
        # init_distributed_with_MPI(rdzv_host, rdzv_port)
        # logger.info(f"LMS: init torch distributed environment done")

        original_worker_main(*args, **kwargs)

    tllm_worker.worker_main = worker_main


# patch_trtllm_worker_main()


def patch_trtllm_build_model():
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
        executor_config = tllm_executor.ExecutorConfig(
            max_beam_width=self.args.build_config.max_beam_width,
            scheduler_config=PybindMirror.maybe_to_pybind(self.args.scheduler_config),
            batching_type=PybindMirror.maybe_to_pybind(self.args.batching_type)
            or tllm_executor.BatchingType.INFLIGHT,
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
            executor_config.peft_cache_config = tllm_executor.PeftCacheConfig(
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
            executor_config.guided_decoding_config = tllm_executor.GuidedDecodingConfig(
                backend=tllm_executor.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
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
        logger.info(f"LMS: self.args.cosmos_config: {self.args}")
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

    LLM._build_model = cosmos_build_model


patch_trtllm_build_model()
