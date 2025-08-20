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
import weakref
from typing import Optional
from pydantic import Field

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.lora_manager import (
    LoraConfig,
    get_default_trtllm_modules_to_hf_modules,
    load_torch_lora,
)
from tensorrt_llm.llmapi.llm_utils import print_traceback_on_error

from tensorrt_llm.inputs import create_input_processor
from tensorrt_llm.llmapi.llm_args import PybindMirror
from tensorrt_llm.llmapi.tokenizer import (
    _xgrammar_tokenizer_info,
    _llguidance_tokenizer_info,
)
from tensorrt_llm.llmapi.mpi_session import external_mpi_comm_available
from tensorrt_llm.executor import PostprocWorkerConfig
from tensorrt_llm.executor.ipc import ZeroMqQueue as IpcQueue
from tensorrt_llm.bindings import executor as tllm_executor
from tensorrt_llm._torch.pyexecutor import _util as tllm_util
from tensorrt_llm._torch.pyexecutor._util import _try_infer_num_experts
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    PeftCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig, NGramDecodingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    SimpleScheduler,
)
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    AttentionTypeCpp,
    create_kv_cache_transceiver,
)
from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_mla,
)

from cosmos_rl.rollout.trtllm_rollout.trtllm_worker import (
    CosmosTRTLLMWorker,
    CosmosWorkerCommIpcAddrs,
)
from cosmos_rl.utils.mpi_distributed import init_distributed_with_MPI


"""
Patches for trtllm.
"""

# 1. Add cosmos config to trtllm ExecutorConfig and `LLMArgs`. patch the _build_model of `LLM` classes.

import tensorrt_llm
from tensorrt_llm.llmapi import llm_args as tllm_llm_args
import tensorrt_llm.llmapi.llm as llm_trllm


def patch_trtllm_llm_args():
    class CosmosLLMArgs(tllm_llm_args.TorchLlmArgs):
        cosmos_config: Optional[CosmosConfig] = Field(
            default=None, description="Cosmos config that hacked by cosmos-rl"
        )

    # Because trtllm in v1.0.0 use many `from import` to import the class, we need to patch all the class.
    tllm_llm_args.TorchLlmArgs = CosmosLLMArgs
    tllm_llm_args.LlmArgs = CosmosLLMArgs
    tensorrt_llm.LlmArgs = CosmosLLMArgs
    llm_trllm.LlmArgs = CosmosLLMArgs
    llm_trllm.TorchLlmArgs = CosmosLLMArgs


patch_trtllm_llm_args()


def extend_create_py_executor_instance():
    def cosmos_create_py_executor_instance(
        *,
        dist,
        resources,
        mapping,
        pytorch_backend_config,
        executor_config,
        ctx_chunk_config,
        model_engine,
        start_worker,
        sampler,
        drafter,
        guided_decoder: Optional[GuidedDecoder] = None,
        lora_config: Optional[LoraConfig] = None,
        garbage_collection_gen0_threshold: Optional[int] = None,
    ) -> PyExecutor:
        kv_cache_manager = resources.get(ResourceManagerType.KV_CACHE_MANAGER, None)

        spec_config = model_engine.spec_config

        logger.info(
            f"max_seq_len={executor_config.max_seq_len}, max_num_requests={executor_config.max_batch_size}, max_num_tokens={executor_config.max_num_tokens}, max_batch_size={executor_config.max_batch_size}"
        )

        for key, value in pytorch_backend_config.extra_resource_managers.items():
            if key in resources:
                raise ValueError(f"Cannot overwrite existing resource manager {key}.")
            resources[key] = value

        peft_cache_manager = None
        if lora_config is not None:
            from tensorrt_llm.bindings import LoraModule

            if len(lora_config.lora_dir) == 1:
                # Route to appropriate loader based on checkpoint source
                load_torch_lora(lora_config)
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

            num_kv_attention_heads_per_layer = (
                model_binding_config.num_kv_heads_per_layer
            )
            if max(num_kv_attention_heads_per_layer) != min(
                num_kv_attention_heads_per_layer
            ):
                logger.warning(
                    "Defining LORA with per-layer KV heads is not supported for LORA, using the max number of KV heads per layer"
                )
                num_kv_attention_heads = max(num_kv_attention_heads_per_layer)
            else:
                # all layers have the same number of KV heads
                num_kv_attention_heads = num_kv_attention_heads_per_layer[0]

            lora_modules = LoraModule.create_lora_modules(
                lora_module_names=lora_config.lora_target_modules,
                hidden_size=model_binding_config.hidden_size,
                mlp_hidden_size=model_binding_config.mlp_hidden_size,
                num_attention_heads=model_binding_config.num_heads,
                num_kv_attention_heads=num_kv_attention_heads,
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

            peft_cache_config_model = (
                PeftCacheConfig.from_pybind(executor_config.peft_cache_config)
                if executor_config.peft_cache_config is not None
                else PeftCacheConfig()
            )
            if lora_config.max_loras is not None:
                peft_cache_config_model.num_device_module_layer = (
                    max_lora_rank * num_lora_modules * lora_config.max_loras
                )
            if lora_config.max_cpu_loras is not None:
                peft_cache_config_model.num_host_module_layer = (
                    max_lora_rank * num_lora_modules * lora_config.max_cpu_loras
                )
            executor_config.peft_cache_config = peft_cache_config_model._to_pybind()

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
            resources[ResourceManagerType.PEFT_CACHE_MANAGER] = peft_cache_manager
            model_engine.set_lora_model_config(
                lora_config.lora_target_modules,
                lora_config.trtllm_modules_to_hf_modules,
            )

        max_num_sequences = executor_config.max_batch_size * mapping.pp_size

        resources[ResourceManagerType.SEQ_SLOT_MANAGER] = SeqSlotManager(
            max_num_sequences
        )

        resource_manager = ResourceManager(resources)

        # Make sure the kv cache manager is always invoked last as it could
        # depend on the results of other resource managers.
        if kv_cache_manager is not None:
            resource_manager.resource_managers.move_to_end(
                ResourceManagerType.KV_CACHE_MANAGER, last=True
            )

        capacity_scheduler = BindCapacityScheduler(
            max_num_sequences,
            kv_cache_manager.impl if kv_cache_manager is not None else None,
            peft_cache_manager.impl if peft_cache_manager is not None else None,
            executor_config.scheduler_config.capacity_scheduler_policy,
            two_step_lookahead=mapping.has_pp(),
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
            drafter=drafter,
            dist=dist,
            max_num_sequences=max_num_sequences,
            disable_overlap_scheduler=pytorch_backend_config.disable_overlap_scheduler,
            max_batch_size=executor_config.max_batch_size,
            max_beam_width=executor_config.max_beam_width,
            max_draft_len=spec_config.max_draft_len if spec_config is not None else 0,
            kv_cache_transceiver=kv_cache_transceiver,
            guided_decoder=guided_decoder,
            start_worker=start_worker,
            garbage_collection_gen0_threshold=garbage_collection_gen0_threshold,
            cosmos_config=executor_config.cosmos_config,
            cosmos_ipc_queues=executor_config.cosmos_ipc_queues_addrs,
        )

    tllm_util.create_py_executor_instance = cosmos_create_py_executor_instance


extend_create_py_executor_instance()

from tensorrt_llm.llmapi.llm import _TorchLLM


def patch_trtllm_build_model():
    def cosmos_build_model(self, *args, **kwargs):
        super(_TorchLLM, self)._build_model()
        assert self._engine_dir is None

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

        max_batch_size = self.args.max_batch_size
        max_num_tokens = self.args.max_num_tokens
        max_seq_len = self.args.max_seq_len

        kwargs = {}
        if self._on_trt_backend:
            kwargs["batching_type"] = (
                self.args.batching_type or tllm_executor.BatchingType.INFLIGHT
            )

        self._executor_config = tllm_executor.ExecutorConfig(
            max_beam_width=self.args.max_beam_width,
            scheduler_config=PybindMirror.maybe_to_pybind(self.args.scheduler_config),
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            gather_generation_logits=self.args.gather_generation_logits,
            fail_fast_on_attention_window_too_large=getattr(
                self.args, "fail_fast_on_attention_window_too_large", False
            ),
            **kwargs,
        )

        if self.args.kv_cache_config is not None:
            self._executor_config.kv_cache_config = PybindMirror.maybe_to_pybind(
                self.args.kv_cache_config
            )
        if os.getenv("FORCE_DETERMINISTIC", "0") == "1":
            # Disable KV cache reuse for deterministic mode
            self._executor_config.kv_cache_config.enable_block_reuse = False
            self._executor_config.kv_cache_config.enable_partial_reuse = False
        if self.args.peft_cache_config is not None:
            self._executor_config.peft_cache_config = PybindMirror.maybe_to_pybind(
                self.args.peft_cache_config
            )
        if self.args.decoding_config is not None:
            self._executor_config.decoding_config = self.args.decoding_config
        if self.args.guided_decoding_backend == "xgrammar":
            self._executor_config.guided_decoding_config = tllm_executor.GuidedDecodingConfig(
                backend=tllm_executor.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
                **_xgrammar_tokenizer_info(self.tokenizer),
            )
        elif self.args.guided_decoding_backend == "llguidance":
            self._executor_config.guided_decoding_config = tllm_executor.GuidedDecodingConfig(
                backend=tllm_executor.GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE,
                **_llguidance_tokenizer_info(self.tokenizer),
            )
        elif self.args.guided_decoding_backend is not None:
            raise ValueError(
                f"Unsupported guided decoding backend {self.args.guided_decoding_backend}"
            )

        if self._on_trt_backend:
            self._executor_config.normalize_log_probs = self.args.normalize_log_probs
        self._executor_config.enable_chunked_context = self.args.enable_chunked_prefill
        self._executor_config.max_beam_width = self.args.max_beam_width
        if self.args.cache_transceiver_config is not None:
            self._executor_config.cache_transceiver_config = (
                PybindMirror.maybe_to_pybind(self.args.cache_transceiver_config)
            )
        from tensorrt_llm._torch.pyexecutor.config import update_executor_config

        spec_config = self.args.speculative_config
        max_batch_size = self._executor_config.max_batch_size
        # Apply default heuristic to AutoDecodingConfig based on benchmark results
        # With concurrency <= 4, max_draft_len = 5, max_matching_ngram_size = 3
        # With concurrency <= 32, max_draft_len = 3, max_matching_ngram_size = 5
        # With concurrency > 32, speculative decoding is disabled.
        if spec_config is not None and spec_config.decoding_type == "AUTO":
            if not self.args.disable_overlap_scheduler:
                logger.info(
                    "Disable overlap scheduler to enable Auto speculative decoding with Ngram."
                )
                # From benchmark results, we found that NGram speculative decoding provides better performance than overlap scheduler with low concurrency <= 32.
                # Therefore, we disable overlap scheduler to enable NGram speculative decoding.
                self.args.disable_overlap_scheduler = True

            spec_config = NGramDecodingConfig(
                max_draft_len=5 if max_batch_size <= 4 else 3,
                max_matching_ngram_size=3 if max_batch_size <= 4 else 5,
                is_keep_all=True,
                is_use_oldest=True,
                is_public_pool=True,
                # Flag to indicate the NGramDecodingConfig is instantiated by auto heuristic.
                is_auto_heuristic=True,
            )

            logger.info(
                f"Apply heuristic to incomplete NGramDecodingConfig: max_draft_len={spec_config.max_draft_len}, max_matching_ngram_size={spec_config.max_matching_ngram_size}"
            )

        update_executor_config(
            self._executor_config,
            backend=self.args.backend,
            pytorch_backend_config=self.args.get_pytorch_backend_config()
            if self.args.backend in ["pytorch", "_autodeploy"]
            else None,
            mapping=self.args.parallel_config.to_mapping(),
            speculative_config=spec_config,
            hf_model_dir=self._hf_model_dir,
            max_input_len=self.args.max_input_len,
            max_seq_len=max_seq_len,
            checkpoint_format=None
            if self.args.backend == "_autodeploy"
            else self.args.checkpoint_format,
            checkpoint_loader=None
            if self.args.backend == "_autodeploy"
            else self.args.checkpoint_loader,
        )

        """
        Cosmos-RL modification start.
        - Add cosmos config to executor_config.
        """
        logger.info("[Rollout] update executor_config.cosmos_config")
        self._executor_config.cosmos_config = self.args.cosmos_config
        # set up IPCQueues that we need.
        self.cosmos_replica_name_queue = IpcQueue(
            is_server=True, name="cosmos_replica_name_queue"
        )
        self.cosmos_weight_sync_queue = IpcQueue(
            is_server=True, name="cosmos_weight_sync_queue"
        )
        # dynamically set the ipc_queues to the executor_config.
        self._executor_config.cosmos_ipc_queues_addrs = CosmosWorkerCommIpcAddrs(
            replica_name_queue_addr=self.cosmos_replica_name_queue.address,
            weight_sync_queue_addr=self.cosmos_weight_sync_queue.address,
        )
        """
        Cosmos-RL modification end.
        """

        # TODO: revisit gather_context_logits
        return_logits = self.args.gather_generation_logits

        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=self._executor_config,
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
            garbage_collection_gen0_threshold=self.args.garbage_collection_gen0_threshold,
        )

    _TorchLLM._build_model = cosmos_build_model


patch_trtllm_build_model()


# Patch worker_main
from tensorrt_llm.executor.worker import worker_main


@print_traceback_on_error
def cosmos_worker_main(*args, **kwargs) -> None:
    origin_worker_main = worker_main
    # init torch distributed environment
    init_distributed_with_MPI()

    return origin_worker_main(*args, **kwargs)


def patch_worker_main():
    import torch
    from tensorrt_llm.executor.proxy import GenerationExecutorProxy
    import concurrent.futures
    from tensorrt_llm.llmapi.tracer import enable_llm_tracer, get_tracer

    def cosmos_start_executor_workers(self, worker_kwargs):
        self_ref = weakref.ref(self)

        def mpi_done_callback(future: concurrent.futures.Future):
            # This is called when the MPI worker is done, so future.exception()
            # will not block.
            if future.exception() is not None:
                if self_ := self_ref():
                    self_._error_queue.put_nowait(future.exception())

        tracer_init_kwargs = get_tracer().init_kwargs if enable_llm_tracer() else None
        from tensorrt_llm._torch.models.modeling_auto import MODEL_CLASS_MAPPING

        torch.cuda.Stream()
        self.mpi_futures = self.mpi_session.submit(
            cosmos_worker_main,
            **worker_kwargs,
            worker_cls=self.worker_cls,
            tracer_init_kwargs=tracer_init_kwargs,
            _torch_model_class_mapping=MODEL_CLASS_MAPPING,
            ready_signal=GenerationExecutorProxy.READY_SIGNAL,
        )
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

        self.workers_started = True

        while True:
            if self.worker_init_status_queue.poll(1):
                ready_signal = self.worker_init_status_queue.get()
                break
            if any(fut.done() for fut in self.mpi_futures):
                logger.error("Executor worker died during initialization.")
                raise RuntimeError("Executor worker died during initialization")
            self._handle_background_error()

        if ready_signal != GenerationExecutorProxy.READY_SIGNAL:
            self.mpi_session.shutdown_abort(reason=ready_signal)
            raise RuntimeError("Executor worker returned error") from ready_signal
        self_ref = weakref.ref(self)

        def mpi_done_callback(future: concurrent.futures.Future):
            # This is called when the MPI worker is done, so future.exception()
            # will not block.
            if future.exception() is not None:
                if self_ := self_ref():
                    self_._error_queue.put_nowait(future.exception())

    GenerationExecutorProxy.worker_main = cosmos_worker_main
    GenerationExecutorProxy._start_executor_workers = cosmos_start_executor_workers


patch_worker_main()


# patch the ExecutorRequestQueue
import datetime
import queue
from typing import List
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)


def cosmos_get_from_request_queue(
    self, timeout: Optional[datetime.timedelta]
) -> List[RequestQueueItem]:
    items = []
    timeout_secs = timeout.total_seconds() if timeout is not None else None
    try:
        if self.request_queue.empty() and (timeout_secs is None or timeout_secs > 0):
            # if queue is empty and want to wait, wait
            # items.append(self.request_queue.get(timeout=timeout_secs))

            #### Cosmos-RL modification start.

            # Note: self.request_queue.get will block the thread.
            # We do not want it to block the thread.
            request = self.request_queue.get(False)
            items.append(request)

            #### Cosmos-RL modification end.
        else:
            # if not empty or don't want to wait, just return all items in queue
            while True:
                queue_item = self.request_queue.get_nowait()
                items.append(queue_item)
    except queue.Empty:
        pass
    return items


def patch_executor_request_queue():
    ExecutorRequestQueue._get_from_request_queue = cosmos_get_from_request_queue


patch_executor_request_queue()
