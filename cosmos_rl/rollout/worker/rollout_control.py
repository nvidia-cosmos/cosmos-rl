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
import time
import threading
import uuid
import torch
import atexit

import torch.distributed as dist

from torch.utils.data import Dataset

from queue import Queue, Empty as QueueEmpty, Full as QueueFull
from cosmos_rl.policy.model import ModelRegistry, WeightMapper
from typing import List, Optional, Callable, Union, Tuple
from functools import partial
from cosmos_rl.rollout import RolloutWorkerBase, State
from cosmos_rl.utils.model_config import load_model_config
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.constant import (
    COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK,
    COSMOS_REWARD_DISPATCHER_CONCURRENCY,
)
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.rollout.rollout_base import RolloutRegistry, RolloutBase
from cosmos_rl.dispatcher.protocol import RolloutRequest, ValidationReportRequest
from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    StopCommand,
    Command,
)
from cosmos_rl.utils.util import str2torch_dtype
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    bounded_drain_or_abort,
    nccl_broadcast,
    nccl_group_start,
    nccl_group_end,
)
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
    WeightSyncInstructionsGroup,
)
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
import cosmos_rl.utils.distributed as dist_util
import cosmos_rl.utils.util as util
from cosmos_rl.utils import constant
from cosmos_rl.dispatcher.data.schema import (
    RLPayload,
    ConversationType,
)
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    RolloutTaskScheduler,
    RolloutTask,
    CompletedRollout,
)
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.dispatcher.data.data_fetcher import WorkerDataFetcher
from cosmos_rl.collective.collective import P2RCollectiveManager
from cosmos_rl.rollout.worker.weight_sync import (
    AsyncR2RSyncMode,
    get_async_r2r_sync_mode,
    get_broadcast_all_params,
    ensure_wst,
    sync_buffer_to_live,
    process_wst_deferred_actions,
    do_nccl_broadcast_grouped,
    install_inference_sync,
)

"""
Keep in mind that torch distributed is not thread safe. So try to keep the usage in the same thread.
"""


# Bounded teardown drain of ``inference_stream``.  Backstop in case an
# in-flight NCCL op (e.g. the WeightSyncThread R2R broadcast) wedges the device;
# on timeout we abort all NCCL communicators so teardown always completes.
_TEARDOWN_DRAIN_TIMEOUT_S = float(os.getenv("COSMOS_TEARDOWN_DRAIN_TIMEOUT_S", "15.0"))


_LEGACY_PREFETCH_HOOK_WARNED = False


def _warn_legacy_prefetch_hook_once() -> None:
    """One-shot warning for backends that still define
    ``enqueue_prefetch_payloads`` directly on their ``RolloutBase``
    subclass instead of composing
    :class:`~cosmos_rl.rollout.generation_mixin.RolloutGenerationMixin`.

    The legacy hook is supported as a deprecation shim; the warning
    fires the first time the rollout controller hands a prefetched
    batch to a legacy backend, then stays silent for the rest of the
    process lifetime to avoid log spam in steady state.
    """
    global _LEGACY_PREFETCH_HOOK_WARNED
    if _LEGACY_PREFETCH_HOOK_WARNED:
        return
    _LEGACY_PREFETCH_HOOK_WARNED = True
    logger.warning(
        "[Rollout] enqueue_prefetch_payloads is deprecated; compose "
        "cosmos_rl.rollout.generation_mixin.RolloutGenerationMixin and "
        "override its hooks (_prepare_sample / _collate_batch / "
        "_generate / _postprocess) instead.  The legacy hook will be "
        "removed in a future release."
    )


class DisaggregatedRolloutControlWorker(RolloutWorkerBase):
    """
    DisaggregatedRolloutControlWorker will be a replica instance of single DP.
    DisaggregatedRolloutControlWorker should support scaling launch.
    """

    SUPPORT_ASYNC_BACKEND = ["vllm_async"]

    def __init__(
        self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs
    ) -> None:
        super(DisaggregatedRolloutControlWorker, self).__init__(config, parallel_dims)

        self.state = State()
        self.is_diffusers = self.config.policy.is_diffusers

        if self.config.rollout.parallelism.dp_shard_size == -1:
            self.config.rollout.parallelism.dp_shard_size = parallel_dims.dp_shard
        assert self.config.rollout.parallelism.dp_shard_size == parallel_dims.dp_shard
        assert self.config.rollout.parallelism.dp_shard_size > 0, (
            "[Rollout] dp_shard_size should be greater than 0."
        )

        # CommandQueue queried from controller.
        self._command_queue: Queue[Command] = Queue()

        # ``_single_producer_mode`` is a *live* property
        # (see below) -- intentionally not snapshotted here.  Backends
        # subclass ``RolloutBase`` and may mutate
        # ``config.rollout.prefetch_rollout`` from ``post_init_hook``,
        # which is invoked from ``RolloutBase.__init__`` *after* this
        # constructor returns.  Any value we cached here would be from
        # the pre-override config.  Reading the property lazily in
        # ``work()`` / ``_main_loop_impl`` / ``_prefetch_loop`` ensures
        # we observe the post-override value at the points where the
        # decision actually matters.
        #
        # Bounded queue (``maxsize=2``): in single-producer mode,
        # ``_prefetch_loop`` would otherwise race ahead of the
        # consumer and queue arbitrarily many batches.  Maxsize=2
        # keeps the prefetcher exactly one batch ahead.  In legacy
        # mode, queue depth never exceeds 1 anyway (``main_loop``
        # only calls ``request_new_prompts`` when the queue is
        # empty), so the cap is harmless there.
        self._prompt_queue: Queue[List[RLPayload]] = Queue(maxsize=2)

        # Vestigial lock from the dual-producer era: prior to the
        # single-producer-mode refactor, both ``main_loop`` (via
        # ``request_new_prompts``) and ``_prefetch_loop`` could put
        # on ``_prompt_queue``, and this lock serialized their
        # empty-check + fetch.  In the new world only one path puts
        # at a time, so the lock is uncontended; we keep it because
        # ``request_new_prompts`` is also called from non-main-loop
        # paths (validation, stream-rollout) that may be revisited
        # in future, and the cost is negligible.
        self._prompt_fetch_lock = threading.Lock()
        self.prefetch_thread: Optional[threading.Thread] = None
        self.current_weight_version = 0
        self._rollout_end_acknowledged = False

        # determine the quantization type
        self.quantization_type = None
        if self.config.rollout.quantization != "none":
            self.quantization_type = self.config.rollout.quantization

        self.rollout: RolloutBase = RolloutRegistry.get_rollout_cls(
            self.config.rollout.backend
        )(self.config, self.parallel_dims, self.device)

        # communicator index for the cached communicators in C++ binding.
        self.global_commnicator_idex = -1
        # rank in current rollout replicas.
        self.rank_in_rollout_repicas = -1

        self.batch_size = self.config.rollout.batch_size
        if self.config.validation.enable:
            self.val_batch_size = self.config.validation.batch_size or self.batch_size
            assert self.val_batch_size > 0, (
                "[Rollout] val_batch_size should be greater than 0."
            )
        else:
            self.val_batch_size = None
        self.background_thread: threading.Thread | None = None
        self.teacher_interact_thread: threading.Thread | None = None
        self.teacher_interact_queue: Queue = Queue()

        if self.is_diffusers:
            assert self.config.train.non_text, (
                "[Rollout] Diffusers rollout only support non-text training now."
            )
            model_type = "diffusers"
        else:
            self.eos_token = util.setup_tokenizer(
                self.config.policy.model_name_or_path
            ).eos_token

            # Routes through ``register_local_model_config`` so non-HF
            # ``model_name_or_path`` values (e.g. a ``.toml`` describing a
            # Gymnasium MLP) resolve before falling back to
            # ``AutoConfig.from_pretrained``. Default HF flow is unchanged.
            hf_config = util.retry(load_model_config)(
                self.config.policy.model_name_or_path, trust_remote_code=True
            )

            model_type = hf_config.model_type
            if not ModelRegistry.check_model_type_supported(model_type):
                logger.warning(
                    f"[Rollout] Replica can not find {model_type} in weight mapper, use {constant.COSMOS_HF_MODEL_TYPES} model type instead, with replica name: {self.replica_name}"
                )
                model_type = constant.COSMOS_HF_MODEL_TYPES
            self.weight_mapper = WeightMapper.get_weight_mapper(model_type)(hf_config)

            model_cls = ModelRegistry._MODEL_REGISTRY[model_type]
            if hasattr(model_cls, "preprocess_hf_config"):
                hf_config = model_cls.preprocess_hf_config(self.config)

            self.model_config = hf_config
            if self.quantization_type == "mxfp4":
                assert model_type == "gpt_oss", (
                    "[Rollout] Mxfp4 quantization is only supported for GPT-OSS now."
                )

        atexit.register(self.handle_shutdown)

        self.inference_stream = torch.cuda.Stream()

        # Holding temp tensors created in `recv_tensor_creator`. Do not remove this, or
        self.temp_recv_tensor_queue = Queue()
        self.misc_params = set()
        self.validation_flag = threading.Event()
        self.reward_dispatcher = RewardDispatcher(
            payload_per_task=COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK
        )
        self.data_fetcher = None

        self.p2r_collective_manager = P2RCollectiveManager(
            replica_name=self.replica_name,
            parallel_dims=self.parallel_dims,
            config=self.config,
            api_client=self.api_client,
            role=self.role,
        )

        # initialize variable for async rollout.
        self._is_async_rollout = False
        self.scheduler: Optional[RolloutTaskScheduler] = None
        if self.config.rollout.mode == "async":
            assert config.rollout.backend in self.SUPPORT_ASYNC_BACKEND, (
                f"DisaggregatedRolloutControlWorker async mode only supports {self.SUPPORT_ASYNC_BACKEND} backends, but got {config.rollout.backend}"
            )
            self._is_async_rollout = True

        self.setup(
            dataset=kwargs.get("dataset"),
            data_packer=kwargs.get("data_packer"),
            reward_fns=kwargs.get("reward_fns"),
            filter_reward_fns=kwargs.get("filter_reward_fns"),
            val_dataset=kwargs.get("val_dataset"),
            val_data_packer=kwargs.get("val_data_packer"),
            val_reward_fns=kwargs.get("val_reward_fns"),
        )
        self.non_trainable_params_received = False

    @property
    def _single_producer_mode(self) -> bool:
        """True when prefetch is on AND the worker is single-process.

        Read live from ``config.rollout.prefetch_rollout`` (and
        ``parallel_dims.world_size``) rather than snapshotted in
        ``__init__``: backends may flip ``prefetch_rollout`` from
        ``post_init_hook``, which runs *after* this constructor.

        See the comment in ``__init__`` next to ``self._prompt_queue``
        for the motivation.

        When True, ``_prefetch_loop`` is the *sole* producer for
        ``_prompt_queue`` and ``_main_loop_impl`` consumes only.
        When False (prefetch off, or multi-rank where
        ``request_new_prompts`` runs a distributed broadcast that
        must include all ranks), we keep the legacy flow:
        ``_main_loop_impl`` calls ``request_new_prompts`` itself
        and the prefetch thread is not started.
        """
        return bool(
            self.config.rollout.prefetch_rollout and self.parallel_dims.world_size == 1
        )

    def setup(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
        reward_fns: Optional[List[Callable]] = None,
        filter_reward_fns: Optional[List[Callable]] = None,
        val_dataset: Optional[Dataset] = None,
        val_data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
        val_reward_fns: Optional[List[Callable]] = None,
    ):
        # setup data packer first
        self.init_data_packer(
            data_packer=data_packer,
            val_data_packer=val_data_packer,
        )
        # Set up data fetcher
        self.data_fetcher = WorkerDataFetcher(
            config=self.config,
            dataset=dataset,
            val_dataset=val_dataset,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
            is_rl=True,
        )

        # Only the last stage of rollout pipeline and tp_coord == 0 need to report rollouts
        self.should_report = self.parallel_dims.tp_coord[0] == 0 and (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )

        self.reward_dispatcher.setup(
            config=self.config,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_reward_fns=val_reward_fns,
            data_packer=self.data_packer,
            val_data_packer=self.val_data_packer,
            num_workers=COSMOS_REWARD_DISPATCHER_CONCURRENCY
            if self.should_report
            else 0,
        )

        if self._is_async_rollout:
            self.scheduler = RolloutTaskScheduler(
                rollout_engine=self.rollout,
                data_packer=self.data_packer,
                max_concurrent_requests=self.config.rollout.async_config.max_concurrent_requests,
                stream=self.inference_stream,
            )

    def prepare_shard_infos_for_weight_sync_insts(self):
        # update the underlying model before prepare shard infos for weight sync instructions.
        self.hp_weight_map, self.quantized_weight_map = (
            self.rollout.pre_get_params_for_sync_hook(
                self.quantization_type, self.weight_mapper, self.parallel_dims
            )
        )

        self.weight_inplace_view_map, grouped_recv_param_key_n_rank_list = (
            self.weight_mapper.cosmos_rollout_prepare_recv(self.get_underlying_model())
        )
        # sorted by the param name for consistent order across ranks.
        self.recv_param_key_n_rank_list = []
        param_groups = []
        for group in grouped_recv_param_key_n_rank_list:
            self.recv_param_key_n_rank_list.extend(group)
            if len(group) > 1:
                param_groups.append([x[0] for x in group])
        self.recv_param_key_n_rank_list = sorted(
            self.recv_param_key_n_rank_list, key=lambda x: x[0]
        )
        local_shard_infos = ParallelTopoMapperGroup(
            self.parallel_dims,
            self.model_config,
            is_policy=False,
            underlying_model=self.get_underlying_model(),
            backend=self.config.rollout.backend,
            weight_mapper=self.weight_mapper,
        ).prepare_local_shard_infos(self.recv_param_key_n_rank_list, self.global_rank)

        # this must be done after prepare_local_shard_infos
        self.weight_inplace_view_map = self.rollout.post_get_params_for_sync_hook(
            self.quantization_type,
            self.weight_mapper,
            self.weight_inplace_view_map,
            self.quantized_weight_map,
        )

        self.all_rank_local_shard_infos = dist_util.all_gather_object_cpu(
            local_shard_infos
        )
        all_param_groups = dist_util.all_gather_object_cpu(param_groups)
        merged_groups = {}
        for r, param_groups in enumerate(all_param_groups):
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) != 0:
                continue
            for group in param_groups:
                group = sorted(group)
                key = tuple(group)
                if key not in merged_groups:
                    merged_groups[key] = group
        sorted_params_all_rank = dist_util.all_gather_object_cpu(
            [x[0] for x in self.recv_param_key_n_rank_list]
        )
        sorted_params_all_rank = [
            x
            for r, x in enumerate(sorted_params_all_rank)
            if self.parallel_dims.get_rank_in_dim("dp_cp_tp", r) == 0
        ]
        if self.global_rank == 0:
            self.api_client.post_rollout_shard_info(
                shard_infos=self.all_rank_local_shard_infos,
                param_groups=list(merged_groups.values()),
                sorted_params=sorted_params_all_rank,
            )

    def handle_shutdown(self):
        # Only call once
        if not hasattr(self, "_shutdown_handled"):
            self._shutdown_handled = True
            if not self.shutdown_signal.is_set():
                logger.info(
                    f"[Rollout] shutdown instruction of {self.replica_name}, setting shutdown signal"
                )
                self.shutdown_signal.set()
            if not self.shutdown_mp_signal.is_set():
                self.shutdown_mp_signal.set()
            # All joins below MUST be bounded.  Worker teardown blocking
            # on an unresponsive controller or a wedged daemon would
            # otherwise turn into multi-minute scheduler-timeout hangs
            # (e.g. ``unregister_from_controller`` -> ``requests.post``
            # to a wedged controller keeping the worker alive long
            # enough that the orchestrator hard-kills the whole job).
            # Worst-case delay caused by a missed join is bounded: the
            # background/heartbeat daemons are ``daemon=True`` /
            # ``mp.Process(daemon=True)`` with PR_SET_PDEATHSIG, so the
            # OS will reap them when the worker process exits.
            _JOIN_TIMEOUT_S = 15.0
            logger.info(
                "[Teardown] %s: handle_shutdown joining daemons "
                "(background/teacher/heartbeat)",
                self.replica_name,
            )
            if self.background_thread is not None:
                logger.info("[Rollout] handle_shutdown: joining command-query thread")
                self.background_thread.join(timeout=_JOIN_TIMEOUT_S)
                if self.background_thread.is_alive():
                    logger.warning(
                        "[Rollout] background_thread did not exit within "
                        "%.1fs of shutdown_signal; continuing teardown "
                        "(daemon will be reaped on process exit)",
                        _JOIN_TIMEOUT_S,
                    )
                self.background_thread = None
                logger.info("[Rollout] handle_shutdown: command-query thread joined")
            if self.teacher_interact_thread is not None:
                logger.info(
                    "[Rollout] handle_shutdown: joining teacher-interact thread"
                )
                self.teacher_interact_thread.join(timeout=_JOIN_TIMEOUT_S)
                if self.teacher_interact_thread.is_alive():
                    logger.warning(
                        "[Rollout] teacher_interact_thread did not exit "
                        "within %.1fs of shutdown_signal; continuing teardown",
                        _JOIN_TIMEOUT_S,
                    )
                self.teacher_interact_thread = None
                logger.info("[Rollout] handle_shutdown: teacher-interact thread joined")
            if self.scheduler is not None:
                self.scheduler.stop(wait=False)
                self.scheduler = None

            if self.heartbeat_thread is not None:
                logger.info("[Rollout] handle_shutdown: joining heartbeat process")
                self.heartbeat_thread.join(timeout=_JOIN_TIMEOUT_S)
                if self.heartbeat_thread.is_alive():
                    logger.warning(
                        "[Rollout] heartbeat process did not exit within "
                        "%.1fs of shutdown_signal; continuing teardown "
                        "(PR_SET_PDEATHSIG will reap it on process exit)",
                        _JOIN_TIMEOUT_S,
                    )
                self.heartbeat_thread = None
                logger.info("[Rollout] handle_shutdown: heartbeat process joined")
            logger.info(
                "[Teardown] %s: handle_shutdown daemons joined; "
                "calling unregister_from_controller",
                self.replica_name,
            )
            self.unregister_from_controller()
            logger.info(
                "[Teardown] %s: unregister_from_controller returned",
                self.replica_name,
            )

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in the rollout internal.
        """
        return self.rollout.get_underlying_model()

    @RolloutWorkerBase.register_rollout_command_handler(StopCommand)
    def handle_stop(self, command: StopCommand):
        """Controller-authoritative end-of-job stop (non-validation runs).

        The controller publishes STOP once every policy replica has
        unregistered (its main loop finished, so no policy will read this
        rollout's output again).  Setting the shutdown signals here breaks
        ``main_loop`` out of *any* branch -- normal drain, an empty queue,
        or the weight-version-gate spin that no longer clears once weight
        syncs have stopped (the residual hang the prompt-stream ``is_end``
        could not reach).  No NCCL collective is involved, and teardown's
        bounded ``cleanup_ucxx`` still lets any in-flight output read drain.
        """
        wst = getattr(self, "_weight_sync_thread", None)
        if wst is not None:
            fenced = wst.fence()
            if not fenced:
                logger.error(
                    "[ABNORMAL teardown] WeightSyncThread fence failed for %s; "
                    "continuing shutdown after the bounded abort path",
                    self.replica_name,
                )
        logger.info(
            "[Rollout] Received STOP from controller for %s; setting shutdown signal.",
            self.replica_name,
        )
        self.shutdown_signal.set()
        self.shutdown_mp_signal.set()

    @RolloutWorkerBase.register_rollout_command_handler(BuildMeshCommand)
    def build_global_mesh(self, build_mesh_command: BuildMeshCommand):
        # Ranked-ended disaggregated workers remain command participants until
        # STOP, so every rank in such a replica must accept topology rebuilds.
        # Keep only the single-process delivery-failure escape: that worker is
        # already shutting down and the controller never acknowledged it.
        if (
            self.state.prompt_consume_end()
            and getattr(self.parallel_dims, "world_size", 1) == 1
            and not getattr(self, "_rollout_end_acknowledged", False)
        ):
            logger.info(
                "[Rollout] Skipping BuildMeshCommand for %s: prompt "
                "source exhausted without an acknowledged checkout.",
                self.replica_name,
            )
            mesh_ready = getattr(self, "_mesh_rebuild_ready", None)
            if mesh_ready is not None:
                mesh_ready.set()
            return

        wst = getattr(self, "_weight_sync_thread", None)
        if wst is not None and not wst.fence():
            raise RuntimeError(
                "Weight-sync work did not drain before rollout mesh rebuild"
            )
        logger.info(f"[Rollout] Building global mesh for {self.replica_name}")

        replica_name_to_rank = build_mesh_command.replica_name_to_rank
        if self.replica_name not in replica_name_to_rank:
            raise RuntimeError(
                f"[Rollout] Replica {self.replica_name} not found in registered replicas."
            )
        self.rank_in_rollout_repicas = replica_name_to_rank[self.replica_name]
        # update the replcia_name to rank dict
        self.replica_name_to_rank = replica_name_to_rank

        if len(replica_name_to_rank) == 1:
            # only one rollout replica now, no need to build mesh.
            mesh_ready = getattr(self, "_mesh_rebuild_ready", None)
            if mesh_ready is not None:
                mesh_ready.set()
            return
        # generate key for storing the NCCL group id.
        # group_0: [rank 0 in replica 0, rank 0 in replica 1, ..., rank 0 in replica n-1]
        # group_1: [rank 1 in replica 0, rank 1 in replica 1, ..., rank 1 in replica n-1]
        # ...
        # group_m-1: [rank m-1 in replica 0, rank m-1 in replica 1, ..., rank m-1 in replica n-1]
        unique_rollout_group_key = self.get_group_unique_key(replica_name_to_rank)
        nccl_group_id = None
        if self.rank_in_rollout_repicas == 0:
            # only replica_rank == 0 have the right to generate nccl id.
            nccl_group_id = create_nccl_uid()
            self.api_client.post_nccl_comm_initiator(
                unique_rollout_group_key, nccl_group_id
            )

        if self.rank_in_rollout_repicas != 0:
            # other replicas should query the nccl group id from controller
            # all ranks need to wait for the rollout replica 0 finished the group_id post
            # and then they can get the group_id from controller
            # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                unique_rollout_group_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
                )

        # update the cached communicator index
        logger.debug(
            f"[Rollout] Creating nccl communicator for global mesh: {unique_rollout_group_key}"
        )
        self.global_commnicator_idex = create_nccl_comm(
            nccl_group_id, self.rank_in_rollout_repicas, len(replica_name_to_rank)
        )
        mesh_ready = getattr(self, "_mesh_rebuild_ready", None)
        if mesh_ready is not None:
            mesh_ready.set()

    def query_nccl_unique_id_from_controller(self, unique_id_key: str):
        # We don't have something like dist.barrier(), so just use while True loop to query it like synchronize.
        # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
        return self.api_client.post_nccl_comm_acceptor(unique_id_key)

    def prepare_trainable_params(self):
        # TODO: (lms/feng) Refactor the param management logic for P2R and R2R, incluing trainable params for P2R and non-trainable params for R2R.
        if not hasattr(self, "trainable_params"):
            if self.global_rank == 0:
                trainable_params = self.api_client.get_trainable_params()
            else:
                trainable_params = None
            trainable_params = dist_utils.broadcast_object_cpu(
                trainable_params,
            )
            self.trainable_params = set(trainable_params)
            logger.info(
                f"[Rollout] Finished fetching {len(self.trainable_params)} trainable params from controller."
            )
            # The splitted and unsplited version of param names should both added to handle for P2R and R2R cases separately.
            for p in trainable_params:
                self.trainable_params.add(
                    self.weight_mapper.get_unsplited_weight_name(p)
                )

            # Add weight scale of quantized weights to trainable params
            if self.quantization_type is not None:
                # Trivial params:
                # including tensors that need to be synced but not trainable in R2R. These
                # tensors will not be synced from P2R, so we have to add them to trainable params.
                for name, _ in self.rollout.model_param_map(self.weight_mapper).items():
                    if name.endswith("_scale"):
                        self.misc_params.add(name)
                self.trainable_params.update(self.misc_params)

            logger.info(
                f"[Rollout] Obtained {len(self.trainable_params)} trainable params after weight unsplit."
            )

    def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        insts_group: WeightSyncInstructionsGroup,
        mesh_key: str,
        trainable_only: bool,
        do_weight_sync_check: bool = False,
    ):
        target_dtype = str2torch_dtype(self.config.train.transfer_dtype)
        check_inside_group = do_weight_sync_check
        if self.quantization_type is not None:
            inst_group_weight_name = (
                insts_group.param_instructions[0].param_name
            )  # take a name from the inst group to determine the full weight name
            # the full weight name that this inst group handles.
            inst_group_full_weight_name = self.weight_mapper.get_unsplited_weight_name(
                inst_group_weight_name
            )
            is_lowp_quantized_module = (
                inst_group_full_weight_name in self.quantized_weight_map
            )
            check_inside_group = do_weight_sync_check and (not is_lowp_quantized_module)

        total_bytes_received = 0

        all_tensor_views_to_copy = []
        tensors_to_check = []

        if self.get_underlying_model() is not None:
            for m in self.get_underlying_model().modules():
                if isinstance(m, torch.distributed.fsdp.FSDPModule):
                    m.reshard()

        def recv_tensor_creator(underlying_tensor_view: torch.Tensor):
            recv_tensor = None
            inplace = True

            # clean up completed temp recv tensor in queue if the recv tensor queue is not empty.
            while (
                not self.temp_recv_tensor_queue.empty()
                and self.temp_recv_tensor_queue.queue[0][1].query()
            ):
                # pop the completed recv tensor if its event is finished.
                self.temp_recv_tensor_queue.get()

            # In case cpu part keeps inserting too many temp tensors without sync.
            # We synchronize and clear the queue to prevent memory issues.
            if (
                not underlying_tensor_view.is_contiguous()
                or underlying_tensor_view.dtype != target_dtype
            ):
                if (
                    self.temp_recv_tensor_queue.qsize()
                    >= constant.COSMOS_RECV_TENSOR_QUEUE_SIZE
                ):
                    num_to_clear = (
                        self.temp_recv_tensor_queue.qsize()
                        - constant.COSMOS_RECV_TENSOR_QUEUE_SIZE
                        + 1
                    )
                    for _ in range(num_to_clear):
                        _, event = self.temp_recv_tensor_queue.get()
                    event.synchronize()

            if underlying_tensor_view.device != self.device:
                recv_tensor = torch.empty_like(
                    underlying_tensor_view, device=torch.cuda.current_device()
                ).contiguous()
                inplace = False
            elif underlying_tensor_view.is_contiguous():
                recv_tensor = underlying_tensor_view
            else:
                # new a temp tensor
                recv_tensor = torch.empty_like(underlying_tensor_view).contiguous()
                inplace = False

            if underlying_tensor_view.dtype != target_dtype:
                recv_tensor = recv_tensor.to(target_dtype)
                inplace = False
            # Event for recv related operations completion tracking
            # Hold these recv_tensor, in case of buffer reusing by torch
            if not inplace:
                recv_complete_event = torch.cuda.Event()
                self.temp_recv_tensor_queue.put((recv_tensor, recv_complete_event))
            else:
                recv_complete_event = None
            return recv_tensor, recv_complete_event, inplace

        skipped_params_cnt = 0

        for insts_for_per_param in insts_group.param_instructions:
            # insts_for_per_param: WeightSyncInstructionsPerParam -> inst collection for a single tensor
            insts = insts_for_per_param.instructions
            # insts: List[Tuple[int, int, Dict[int, Any]]]
            inst_dest_name = insts_for_per_param.param_name

            if inst_dest_name not in self.trainable_params and trainable_only:
                logger.debug(
                    f"[Rollout] Skip {inst_dest_name} in P2R recv due to non trainable."
                )
                skipped_params_cnt += 1
                continue

            target_tensor = self.weight_inplace_view_map[inst_dest_name]
            if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                target_tensor = target_tensor.to_local()

            if check_inside_group:
                cloned_target_tensor = target_tensor.clone().cpu()
                # clear the current view
                target_tensor.zero_()

            for inst in insts:
                # Inst for different part of a tensor between policy and rollout.
                p_rank = inst.policy_rank
                r_rank = inst.rollout_rank
                tensor_split_strategys = inst.slice_strategy
                assert r_rank == global_rank_of_rollout

                underlying_tensor_view = target_tensor.cosmos_slice(
                    tensor_split_strategys
                )
                recv_tensor, recv_complete_event, inplace = recv_tensor_creator(
                    underlying_tensor_view
                )
                logger.debug(
                    f"[Rollout] Recving tensor {inst_dest_name} from policy rank {p_rank} to rollout rank {r_rank}, shape {underlying_tensor_view.shape} of {target_tensor.shape} with dtype {recv_tensor.dtype}."
                )
                self.p2r_collective_manager.recv(mesh_key, recv_tensor, p_rank)

                # inplace copy
                if not inplace:
                    all_tensor_views_to_copy.append(
                        (
                            underlying_tensor_view,
                            recv_tensor,
                            recv_complete_event,
                            inst_dest_name,
                        )
                    )

                total_bytes_received += recv_tensor.numel() * recv_tensor.element_size()

            if check_inside_group:
                tensors_to_check.append(
                    (cloned_target_tensor, target_tensor, insts, inst_dest_name)
                )

        post_process_list_for_lowp = []

        if not check_inside_group and self.quantization_type is not None:
            post_process_list_for_lowp.append(inst_group_full_weight_name)

        def completion_lambda(
            all_tensor_views_to_copy, tensors_to_check, post_process_list_for_lowp
        ):
            for (
                view,
                recv_tensor,
                recv_complete_event,
                inst_dest_name,
            ) in all_tensor_views_to_copy:
                self.weight_mapper.update_tensor_view(
                    view, recv_tensor, inst_dest_name, parallel_dims=self.parallel_dims
                )
                if recv_complete_event is not None:
                    recv_complete_event.record()
            for (
                cloned_target_tensor,
                target_tensor,
                insts,
                inst_dest_name,
            ) in tensors_to_check:
                cloned_target_tensor = cloned_target_tensor.to(target_dtype).to(
                    cloned_target_tensor.dtype
                )
                if not torch.allclose(cloned_target_tensor, target_tensor.cpu()):
                    raise ValueError(
                        f"Weight sync check failed after weight sync instruction: {insts} for {inst_dest_name}."
                    )
            tensors_to_check.clear()

            # here we got one full weight tensor sync done, if it is fp8/mxfp4 weight, we should do the quantization and check the numerical error.
            if self.quantization_type is not None:
                for inst_group_full_weight_name in post_process_list_for_lowp:
                    if self.quantization_type == "fp8":
                        if inst_group_full_weight_name in self.hp_weight_map:
                            weight_to_quantize = self.hp_weight_map[
                                inst_group_full_weight_name
                            ]  # [out_dim, in_dim]
                            quantized_weight, weight_scale = (
                                self.rollout.fp8_quantization(weight_to_quantize)
                            )
                            model_param_map = self.rollout.model_param_map(
                                self.weight_mapper
                            )
                            underlying_native_weight = model_param_map[
                                inst_group_full_weight_name
                            ]

                            # check weight sync
                            if do_weight_sync_check:
                                # allclose doesn't support fp8, promote it.
                                bf16_underlying_native_weight = (
                                    underlying_native_weight.to(torch.bfloat16)
                                )
                                bf16_quantized_weight = quantized_weight.to(
                                    torch.bfloat16
                                )
                                if not torch.allclose(
                                    bf16_underlying_native_weight, bf16_quantized_weight
                                ):
                                    raise ValueError(
                                        f"FP8 weight doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                    )
                            underlying_native_weight.copy_(quantized_weight)
                            # get the scale key.
                            scale_key = inst_group_full_weight_name.replace(
                                ".weight", ".weight_scale"
                            )
                            scale_tensor = model_param_map[scale_key]
                            assert scale_tensor.shape == weight_scale.shape, (
                                f"scale_tensor.shape: {scale_tensor.shape}, weight_scale.shape: {weight_scale.shape}"
                            )
                            scale_tensor.copy_(weight_scale)
                    elif self.quantization_type == "mxfp4":
                        # Note: For mxfp4, we don't do weight sync check for quantized weights.
                        if inst_group_full_weight_name in self.hp_weight_map:
                            if "gate_up_proj_bias" not in inst_group_full_weight_name:
                                # Weight to quantize:
                                # [local_num_experts, 2* local_intermediate_size, hidden_size] for gate_up_proj
                                # [local_num_experts, hidden_size, local_intermediate_size] for down_proj
                                weight_to_quantize = self.hp_weight_map[
                                    inst_group_full_weight_name
                                ]
                                quantized_weight, weight_scale = (
                                    self.rollout.mxfp4_quantization(weight_to_quantize)
                                )
                                # The quantized version of the weight has been removed by vLLM internally.
                                # https://github.com/zyongye/vllm/blob/6a70830065701b163e36a86fd331b41b5feac401/vllm/model_executor/layers/quantization/mxfp4.py#L328
                                # We can't get it from named_parameters.
                                underlying_native_weight = None
                                underlying_native_weight_scale = None

                                for (
                                    module_name,
                                    module,
                                ) in self.get_underlying_model().named_modules():
                                    w13_weight_name = f"{module_name}.w13_weight"
                                    w2_weight_name = f"{module_name}.w2_weight"
                                    w13_compatible_weight_name = self.weight_mapper.rollout_map_local_key_to_hf_key(
                                        w13_weight_name
                                    )
                                    w2_compatible_weight_name = self.weight_mapper.rollout_map_local_key_to_hf_key(
                                        w2_weight_name
                                    )

                                    # mxfp4 weight and mxfp4 weight scale are in int8 data type.
                                    # Two fp4 are packed into one int8 memory.
                                    if (
                                        inst_group_full_weight_name
                                        == w13_compatible_weight_name
                                    ):
                                        underlying_native_weight = module.quant_method.w13_weight_triton_tensor.storage.data
                                        underlying_native_weight_scale = module.quant_method.w13_precision_config.weight_scale.storage.data
                                        break
                                    elif (
                                        inst_group_full_weight_name
                                        == w2_compatible_weight_name
                                    ):
                                        underlying_native_weight = module.quant_method.w2_weight_triton_tensor.storage.data
                                        underlying_native_weight_scale = module.quant_method.w2_precision_config.weight_scale.storage.data
                                        break

                                assert underlying_native_weight is not None, (
                                    f"Failed to find the original weight for {inst_group_full_weight_name}"
                                )
                                assert underlying_native_weight_scale is not None, (
                                    f"Failed to find the original weight scale for {inst_group_full_weight_name}"
                                )

                                with torch.inference_mode():
                                    _, dim_1, dim_2 = quantized_weight.shape

                                    # check weight sync
                                    if do_weight_sync_check:
                                        valid_native_weight = underlying_native_weight[
                                            :, :dim_1, :dim_2
                                        ]
                                        if not torch.allclose(
                                            valid_native_weight, quantized_weight
                                        ):
                                            raise ValueError(
                                                f"MXFP4 weight doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                            )
                                    underlying_native_weight[:, :dim_1, :dim_2].copy_(
                                        quantized_weight
                                    )
                                    # check weight sync
                                    _, dim_1, dim_2 = weight_scale.shape
                                    if do_weight_sync_check:
                                        valid_native_weight_scale = (
                                            underlying_native_weight_scale[
                                                :, :dim_1, :dim_2
                                            ]
                                        )
                                        if not torch.allclose(
                                            valid_native_weight_scale, weight_scale
                                        ):
                                            raise ValueError(
                                                f"MXFP4 weight scale doesn't match after weight sync and dynamic quantization for full weight name: {inst_group_full_weight_name}."
                                            )
                                    underlying_native_weight_scale[
                                        :, :dim_1, :dim_2
                                    ].copy_(weight_scale)

                            else:
                                # For w13_bias, no need to quant, just copy the weight.
                                w13_bias_hp_weight = self.hp_weight_map[
                                    inst_group_full_weight_name
                                ]
                                model_param_map = self.rollout.model_param_map(
                                    self.weight_mapper
                                )
                                underlying_native_weight = model_param_map[
                                    inst_group_full_weight_name
                                ]
                                _, dim1 = w13_bias_hp_weight.shape
                                if do_weight_sync_check:
                                    if not torch.allclose(
                                        underlying_native_weight[:, :dim1],
                                        w13_bias_hp_weight,
                                    ):
                                        raise ValueError(
                                            f"gate_up_proj_bias doesn't match after weight sync for full weight name: {inst_group_full_weight_name}."
                                        )

                                underlying_native_weight[:, :dim1].copy_(
                                    w13_bias_hp_weight
                                )
            else:
                # For non-fp8/mxfp4 weights and fp8/mxfp4 not enabled cases, we just do nothing
                pass

        return (
            total_bytes_received,
            partial(
                completion_lambda,
                all_tensor_views_to_copy,
                tensors_to_check,
                post_process_list_for_lowp,
            ),
            skipped_params_cnt,
        )

    def do_validation(self):
        validation_queue = Queue()
        validation_payloads: List[RLPayload] = []
        is_end = False
        no_more_prompts = False

        # statistic the async rollout
        total_prompts_count = 0
        total_validation_payload_count = 0

        # Do validation here
        while True:
            payloads_list: List[RLPayload] = []
            rollout_results: List[RolloutResult] = []

            if self._is_async_rollout:
                if not no_more_prompts:
                    (
                        fetched_prompts,
                        no_more_prompts,
                    ) = self._stream_generation_feed_prompts(
                        self.val_batch_size,
                        validation_queue,
                        validation_step=self.current_step,
                    )
                    total_prompts_count += fetched_prompts

                is_end = (
                    no_more_prompts
                    and self.scheduler.is_idle()
                    and total_prompts_count == total_validation_payload_count
                )

                # get processed results
                completed_rollouts = self.scheduler.get_all()

                for cr in completed_rollouts:
                    payloads_list.append(cr.payload)
                    rollout_results.append(cr.result)

                total_validation_payload_count += len(payloads_list)
            else:
                is_end = self.request_new_prompts(
                    self.val_batch_size,
                    validation_queue,
                    validation_step=self.current_step,
                    rank_in_mesh=self.rank_in_rollout_repicas,
                )
                if not validation_queue.empty():
                    payloads_list: List[RLPayload] = validation_queue.get()

                    rollout_results: List[RolloutResult] = (
                        self._call_rollout_generation(
                            payloads=payloads_list,
                            stream=self.inference_stream,
                            data_packer=self.val_data_packer,
                            data_fetcher=self.data_fetcher,
                            is_validation=True,
                        )
                    )

            if rollout_results:
                for p, rr in zip(payloads_list, rollout_results):
                    p.completions = rr.completions
                    p.completion_logprobs = rr.completion_logprobs
                    p.completion_token_ids = rr.completion_token_ids
                    p.prompt_logprobs = rr.prompt_logprobs
                    p.prompt_token_ids = rr.prompt_token_ids
                    p.weight_version = self.current_weight_version
                    p.cumulative_logprob = rr.cumulative_logprob
                    p.extra_info = rr.extra_info
                    if self.config.rollout.multi_turn_config.enable:
                        p.completed_conversations = rr.completed_conversations
                    if self.config.train.local_dataset:
                        p.reference_answer = self.data_fetcher.query_reference_answer(
                            p.prompt_idx,
                            "val",
                        )
                validation_payloads.extend(payloads_list)

            if is_end:
                break

        # Clear the flag to indicate validation is done.
        self.validation_flag.clear()

        if self.should_report:
            self.reward_dispatcher.enqueue_rewards_cal(
                validation_payloads, True, self.current_step
            )
            payloads, is_validation, current_step, empty = self.report_rollouts(
                block=True
            )
            assert (is_validation and payloads is not None or payloads is None) and (
                not empty or len(validation_payloads) == 0
            ), (
                f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
            )
            while not empty:
                assert is_validation or payloads is None, (
                    f"Payloads must be for validation if not empty {is_validation}, {payloads}, {empty}"
                )
                if payloads is not None:
                    for i in range(len(payloads)):
                        # we don't need to upload completions, completed_conversations, completion_logprobs, completion_token_ids for validation.
                        # some other fields are removed inside `report_rollouts` function.
                        payloads[i].completions = None
                        payloads[i].completed_conversations = None
                        payloads[i].completion_logprobs = None
                        payloads[i].completion_token_ids = None

                        # For diffusers rollout, we don't need to upload extra_info for validation.
                        if self.is_diffusers:
                            payloads[i].extra_info = None

                    response = ValidationReportRequest(
                        src_replica_name=self.replica_name,
                        validation_step=current_step,
                        payloads=payloads,
                        is_end=True,
                    )
                    self.api_client.post_validation_report(response)
                payloads, is_validation, current_step, empty = (
                    self.reward_dispatcher.dequeue_rewards_cal()
                )

    def _start_async_rollout_scheduler(self, load_format):
        """
        Start the async rollout scheduler.
        """
        assert self.config.rollout.mode == "async", (
            "Async rollout scheduler is not enabled"
        )

        if self.scheduler.is_running():
            logger.info("[Rollout] Async rollout scheduler is already running")
            return

        def init_engine_hook(rollout_engine: RolloutBase):
            """
            This hook function is used to initialize the rollout engine in the async rollout scheduler.
            """
            rollout_engine.init_engine(
                quantization=self.quantization_type,
                seed=self.config.rollout.seed,
                load_format=load_format,
            )
            rollout_engine.post_init_engine_hook(
                self.consume_command,
                self.report_rollouts,
                self.validation_flag,
            )

        self.scheduler.start(init_engine_hook, wait_initialized=True)
        logger.info("[Rollout] Async rollout scheduler started")

    def lazy_initialize_rollout_engine(self, load_format):
        # lazy initialization of the rollout engine.
        already_initialized = self.rollout.is_engine_initialized()
        if not already_initialized:
            if self._is_async_rollout:
                # wait the scheduler thread to initialize the rollout engine.
                self._start_async_rollout_scheduler(load_format)
            else:
                self.rollout.init_engine(
                    quantization=self.quantization_type,
                    seed=self.config.rollout.seed,
                    load_format=load_format,
                )
                self.rollout.post_init_engine_hook(
                    self.consume_command,
                    self.report_rollouts,
                    self.validation_flag,
                )
            self.prepare_shard_infos_for_weight_sync_insts()

            async_mode = get_async_r2r_sync_mode(self)
            if async_mode != AsyncR2RSyncMode.DISABLED:
                logger.info(
                    "[Rollout] Model loaded — creating buffer + WeightSyncThread "
                    "(async_mode=%s).",
                    async_mode.value,
                )
                ensure_wst(self)

    @RolloutWorkerBase.register_rollout_command_handler(PolicyToRolloutUnicastCommand)
    @torch.no_grad()
    def policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        """Sync the weight from policy to rollout.

        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.

        When async R2R mode is enabled, P2R commands are routed to the
        WeightSyncThread which calls ``_execute_p2r_recv`` on its own
        CUDA stream.
        """
        # lazy initialization of the rollout engine.
        is_for_weight_resume = command.dst_replica_name == self.replica_name
        load_format = "auto" if is_for_weight_resume else "dummy"
        self.lazy_initialize_rollout_engine(load_format)

        if command.dst_replica_name != self.replica_name:
            return

        async_mode = get_async_r2r_sync_mode(self)
        if async_mode != AsyncR2RSyncMode.DISABLED:
            wst = self._weight_sync_thread
            wst.enqueue_p2r(command)
            logger.info(
                "[Rollout] Enqueued P2R to WeightSyncThread (step=%s).",
                command.weight_step,
            )
            return

        self._execute_p2r_recv(command, self.inference_stream)

    def _execute_p2r_recv(
        self,
        command: PolicyToRolloutUnicastCommand,
        stream: torch.cuda.Stream,
    ):
        """Execute the P2R NCCL receive on the given CUDA stream.

        Separated from ``policy_to_rollout_unicast`` so the
        WeightSyncThread can call this directly with its own stream,
        without needing to swap ``inference_stream``.
        """
        self.p2r_collective_manager.setup_manager(command)

        comm_id = None
        base_mesh_key = command.src_replica_name + "_" + command.dst_replica_name
        comm_id = (
            None
            if self.rl_mode == "colocated_separated"
            else self.p2r_collective_manager.query_nccl_comm_index(base_mesh_key)
        )

        if not hasattr(self, "policy_to_rollout_recv_insts"):
            logger.info(
                "[Rollout] Fetching policy_to_rollout_recv_insts from controller ..."
            )
            self.policy_to_rollout_recv_insts = (
                self.api_client.post_rollout_shard_recv_insts(self.global_rank)
            )
            logger.info(
                "[Rollout] Finished policy_to_rollout_recv_insts from controller."
            )
        else:
            assert command.trainable_only, (
                "only trainable params should be transferred at the not first time P2R"
            )

        self.prepare_trainable_params()

        total_recvs = 0
        total_params = 0
        for insts_group in self.policy_to_rollout_recv_insts:
            for insts_for_per_param in insts_group.param_instructions:
                total_params += 1
                total_recvs += len(insts_for_per_param.instructions)

        copy_stream = torch.cuda.Stream()

        assert total_params == len(self.recv_param_key_n_rank_list), (
            f"Mismatch in total params and received param keys: {total_params} != {len(self.recv_param_key_n_rank_list)}"
        )

        with torch.cuda.stream(stream):
            logger.info(
                f"[Rollout] Starting to execute {len(self.policy_to_rollout_recv_insts)}; {total_params}, {total_recvs} weight sync receives ..."
            )
            st = time.time()
            total_bytes_received = 0

            pending_bytes = [0]
            pending_completions = []
            pending_groups = 0

            def flush_completions(pending_bytes, pending_completions):
                recv_ready = torch.cuda.Event()
                recv_ready.record()
                copy_stream.wait_event(recv_ready)
                with torch.cuda.stream(copy_stream):
                    logger.debug(
                        f"[Rollout] Flushing {len(pending_completions)} completions, {pending_bytes[0] // 1024 // 1024} MB"
                    )
                    for completion in pending_completions:
                        completion()
                    pending_bytes[0] = 0
                    pending_completions.clear()

            if (
                self.rl_mode != "colocated_separated"
                and constant.COSMOS_P2R_NCCL_GROUP_SIZE > 0
            ):
                nccl_group_start(comm_id)

            skipped_params_cnt = 0
            transferred_params_cnt = 0
            skipped_groups_cnt = 0
            transferred_groups_cnt = 0

            for insts_group in self.policy_to_rollout_recv_insts:
                (
                    bytes_received,
                    completion_fn,
                    skipped_cnt,
                ) = self.recv_weight_shard(
                    self.global_rank,
                    insts_group,
                    base_mesh_key,
                    command.trainable_only,
                    command.do_weight_sync_check,
                )
                skipped_params_cnt += skipped_cnt
                transferred_params_cnt += (
                    len(insts_group.param_instructions) - skipped_cnt
                )
                if (
                    self.weight_mapper.get_unsplited_weight_name(
                        insts_group.param_instructions[0].param_name
                    )
                    != insts_group.param_instructions[0].param_name
                ):
                    skipped_groups_cnt += 1 if skipped_cnt > 0 else 0
                    transferred_groups_cnt += 0 if skipped_cnt > 0 else 1
                else:
                    skipped_groups_cnt += skipped_cnt
                    transferred_groups_cnt += (
                        len(insts_group.param_instructions) - skipped_cnt
                    )

                pending_bytes[0] += bytes_received
                pending_completions.append(completion_fn)
                total_bytes_received += bytes_received

                pending_groups += 1
                if pending_groups >= constant.COSMOS_P2R_NCCL_GROUP_SIZE:
                    if (
                        self.rl_mode != "colocated_separated"
                        and constant.COSMOS_P2R_NCCL_GROUP_SIZE > 0
                    ):
                        nccl_group_end(comm_id)
                    flush_completions(pending_bytes, pending_completions)
                    if (
                        self.rl_mode != "colocated_separated"
                        and constant.COSMOS_P2R_NCCL_GROUP_SIZE > 0
                    ):
                        nccl_group_start(comm_id)
                    pending_groups = 0

            if (
                self.rl_mode != "colocated_separated"
                and constant.COSMOS_P2R_NCCL_GROUP_SIZE > 0
            ):
                nccl_group_end(comm_id)

            flush_completions(pending_bytes, pending_completions)

            with torch.cuda.stream(copy_stream):
                copy_finished = torch.cuda.Event()
                copy_finished.record()

            stream.wait_event(copy_finished)
            self.temp_recv_tensor_queue.queue.clear()

            time_eclapsed = time.time() - st
            logger.info(
                f"[Rollout] All {len(self.policy_to_rollout_recv_insts)} at step {command.weight_step} recv operations finished in {time_eclapsed:.3f} seconds with {total_bytes_received / (1024 * 1024)} MB received. While {skipped_params_cnt} non-trainable splitted params skipped and {transferred_params_cnt} trainable splitted params transferred."
            )

            if command.trainable_only:
                assert self.non_trainable_params_received, (
                    "[Rollout] Non-trainable params must be received before trainable-only P2R."
                )
                if not hasattr(self, "p2r_synced_trainable_params_cnt"):
                    self.p2r_synced_trainable_params_cnt = transferred_groups_cnt
                assert self.p2r_synced_trainable_params_cnt == transferred_groups_cnt, (
                    f"Count of trainable unsplitted params which have been synced in P2R {transferred_groups_cnt} must match the synced_trainable_params attribute {self.p2r_synced_trainable_params_cnt}."
                )

            self.state.set_weight_synced()
        if not command.trainable_only:
            self.non_trainable_params_received = True

    @RolloutWorkerBase.register_rollout_command_handler(
        RolloutToRolloutBroadcastCommand
    )
    def broadcast_to_all_rollout_replica(
        self, broadcast_command: RolloutToRolloutBroadcastCommand
    ) -> None:
        """Broadcast the weight to all other rollout replicas.

        Will only happen between Rollout Replica 0 and all other Rollout
        Replicas.

        When ``async_r2r_sync`` is enabled the broadcast is enqueued to
        the WeightSyncThread which executes it on a dedicated CUDA stream
        with a Redis barrier.  When ``broadcast_all_params`` is enabled
        (or in async mode), the full state_dict is broadcast rather than
        only the trainable subset selected by ``model_param_map``.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        dst_replica_names: List[str] = broadcast_command.dst_replica_names

        # Forward-compat: flush any pending async NCCL sends (e.g. from data
        # packers) so they complete before weight sync reuses the communicator.
        if hasattr(self, "data_packer") and hasattr(
            self.data_packer, "flush_pending_sends"
        ):
            self.data_packer.flush_pending_sends()

        # lazy initialization of the rollout engine.
        if self.replica_name != src_replica_name:
            # for replicas that needs to be broadcasted, use dummy format.
            self.lazy_initialize_rollout_engine(load_format="dummy")

        was_synced = self.state.weight_synced()
        trainable_only = broadcast_command.trainable_only
        if not was_synced and trainable_only:
            logger.info(
                "[Rollout] First broadcast has trainable_only=True "
                "(race: rollout leader was faster). Forcing full broadcast."
            )
            trainable_only = False

        async_mode = get_async_r2r_sync_mode(self)
        broadcast_all = get_broadcast_all_params(self)

        if len(dst_replica_names) > 1:
            if async_mode != AsyncR2RSyncMode.DISABLED:
                # Enqueue to the WeightSyncThread.
                wst = self._weight_sync_thread
                wst.enqueue_r2r(broadcast_command)
                logger.info(
                    "[Rollout] Enqueued R2R to WeightSyncThread (mode=%s, step=%s).",
                    async_mode.value,
                    broadcast_command.weight_step,
                )
            elif broadcast_all:
                # Synchronous full-model broadcast via grouped NCCL.
                logger.info(
                    "[Rollout] Starting full-model broadcast (broadcast_all_params=true)."
                )
                t0 = time.time()
                transferred_params_cnt, bytes_broadcast = do_nccl_broadcast_grouped(
                    self,
                    src_replica_name,
                    self.inference_stream,
                )
                self.inference_stream.synchronize()
                elapsed = time.time() - t0
                logger.info(
                    "[Rollout] Finished full-model broadcast: %d params, "
                    "%.1f MB, %.3f s",
                    transferred_params_cnt,
                    bytes_broadcast / (1024 * 1024),
                    elapsed,
                )
                if not self.state.weight_synced():
                    self.state.set_weight_synced()
                if not trainable_only:
                    self.non_trainable_params_received = True
            else:
                # Original synchronous per-param broadcast path.
                self.prepare_trainable_params()
                skipped_params_cnt = 0
                transferred_params_cnt = 0
                logger.info(
                    "[Rollout] Starting broadcasting of parameters to all replicas."
                )
                with torch.cuda.stream(self.inference_stream):
                    assert self.rank_in_rollout_repicas >= 0, (
                        "[Rollout] rank in rollout replicas should be set before broadcast."
                    )
                    assert len(dst_replica_names) == len(self.replica_name_to_rank), (
                        "[Rollout] The vaild dst replicas num should match the replicas num that this worker holds."
                    )

                    src_rank = self.replica_name_to_rank[src_replica_name]
                    with torch.inference_mode():
                        for name, parameter in self.rollout.model_param_map(
                            self.weight_mapper
                        ).items():
                            if name not in self.trainable_params and trainable_only:
                                logger.debug(
                                    f"[Rollout] Skip {name} in R2R due to non trainable."
                                )
                                skipped_params_cnt += 1
                                continue
                            transferred_params_cnt += 1

                            recv_tensor = parameter
                            if not parameter.is_contiguous():
                                recv_tensor = parameter.contiguous()

                            nccl_broadcast(
                                recv_tensor, src_rank, self.global_commnicator_idex
                            )

                            if not parameter.is_contiguous():
                                parameter.copy_(recv_tensor)

                    if not self.state.weight_synced():
                        assert not trainable_only, (
                            "[Rollout] Trainable only must be set to False for the first broadcast."
                        )
                        self.state.set_weight_synced()

                logger.info(
                    f"[Rollout] Finished broadcasting of parameters to all replicas. While {skipped_params_cnt} unsplitted non-trainable params skipped and {transferred_params_cnt} unsplitted params transferred."
                )
                if not trainable_only:
                    self.non_trainable_params_received = True

                if trainable_only:
                    assert self.non_trainable_params_received, (
                        "[Rollout] Non-trainable params must be received before trainable-only R2R."
                    )
                    if not hasattr(self, "r2r_synced_trainable_params_cnt"):
                        self.r2r_synced_trainable_params_cnt = transferred_params_cnt
                    if hasattr(self, "p2r_synced_trainable_params_cnt"):
                        assert (
                            self.r2r_synced_trainable_params_cnt
                            == self.p2r_synced_trainable_params_cnt
                            + len(self.misc_params)
                        ), (
                            f"Synced params count in R2R {self.r2r_synced_trainable_params_cnt} must match the sum of count of attribute {self.p2r_synced_trainable_params_cnt} and {len(self.misc_params)}."
                        )

        # --- Post-broadcast bookkeeping (weight version, validation, shutdown) ---

        current_step = broadcast_command.weight_step

        # When async mode is enabled, the NCCL broadcast hasn't happened yet
        # (it's queued on the WST).  The WST's _execute_r2r will update
        # current_weight_version after the broadcast completes.
        if async_mode == AsyncR2RSyncMode.DISABLED:
            if current_step is not None:
                assert current_step >= self.current_weight_version, (
                    f"current_step: {current_step} must be greater than or equal to self.current_weight_version: {self.current_weight_version}"
                )
                self.current_weight_version = current_step
            else:
                current_step = self.current_weight_version

            if current_step is not None and current_step >= 0:
                is_initial_validation = (
                    current_step == 0 and self.config.validation.val_before_train
                )
                is_periodic_validation = (
                    current_step > 0 and current_step % self.config.validation.freq == 0
                )
                is_final_validation = current_step == broadcast_command.total_steps

                should_do_validation = self.config.validation.enable and (
                    is_initial_validation
                    or is_periodic_validation
                    or is_final_validation
                )

                if should_do_validation:
                    self.current_step = current_step
                    self.validation_flag.set()

            if broadcast_command.replica_should_stop():
                data = {
                    "is_end": True,
                    "prompt_idx": -1,
                    "completion_token_ids": [],
                }
                self.redis_controller.publish_teacher_request(data, self.replica_name)
                logger.info("[Rollout] Published end event to reference")
                if self.validation_flag.is_set():
                    self.do_validation()
                self.shutdown_signal.set()
                self.shutdown_mp_signal.set()

        # In async mode the WST's _execute_r2r calls set_weight_synced
        # after the broadcast actually completes.  Calling it here would
        # be premature (the NCCL transfer is only enqueued, not done).
        if async_mode == AsyncR2RSyncMode.DISABLED and not self.state.weight_synced():
            logger.info(
                "[Rollout] Setting weight_synced after broadcast (n_dst=%d, step=%s)",
                len(dst_replica_names),
                current_step,
            )
            self.state.set_weight_synced()

    def query_command_from_controller(self):
        """Background task to check commands from the controller.

        When async R2R mode is active and the WeightSyncThread is ready,
        P2R and R2R commands are routed directly to the WST instead of
        going through ``_command_queue``.  This avoids the latency of
        waiting for the main thread (which may be running a long
        simulation) to drain the queue before weight-sync begins.
        """
        while not self.shutdown_signal.is_set():
            commands = []
            try:
                # blocking request
                commands = self.redis_controller.subscribe_command(self.replica_name)
            except Exception as e:
                logger.error(
                    f"[Rollout] Failed in query commands from controller for replica {self.replica_name}\n: {str(e)}"
                )

            stop_received = False
            for instruction in commands:
                command = Command.depack(instruction)
                logger.debug(f"[Rollout] Received command: {command.command_type}")

                if isinstance(command, BuildMeshCommand):
                    mesh_ready = threading.Event()
                    self._mesh_rebuild_ready = mesh_ready
                    self._command_queue.put(command)
                    while not mesh_ready.wait(timeout=0.1):
                        if self.shutdown_signal.is_set():
                            return
                    continue

                wst = getattr(self, "_weight_sync_thread", None)
                if wst is not None and isinstance(
                    command, PolicyToRolloutUnicastCommand
                ):
                    if command.dst_replica_name == self.replica_name:
                        wst.enqueue_p2r(command)
                    else:
                        logger.debug(
                            "[Rollout] Skipping P2R for other replica %s",
                            command.dst_replica_name,
                        )
                    continue

                if wst is not None and isinstance(
                    command, RolloutToRolloutBroadcastCommand
                ):
                    wst.enqueue_r2r(command)
                    continue

                self._command_queue.put(command)
                if isinstance(command, StopCommand):
                    stop_received = True
                    break

            if stop_received:
                break

    def teacher_interact_loop(self):
        """Background task to interact with teacher model for distillation"""
        while not self.shutdown_signal.is_set():
            if not self.teacher_interact_queue.empty():
                data = self.teacher_interact_queue.get_nowait()
                self.redis_controller.publish_teacher_request(data, self.replica_name)
            time.sleep(0.01)

    def request_new_prompts(
        self,
        batch_size: int,
        prompt_queue: Queue,
        max_pending_batches: int = 1,
        **kwargs,
    ):
        """
        Request new prompts from the controller for both training and validation.

        ``max_pending_batches`` bounds how many batches may already be
        queued before this call fetches another.  The default of 1
        preserves the legacy fetch-when-empty cadence; the multi-rank
        prep-overlap path passes 2 so it can fetch one batch ahead and
        start that batch's preparation while the current one generates.
        The fetch decision is taken on rank 0 and broadcast, so every
        rank stays in lockstep regardless of this value.
        """
        prompts_and_is_end = (None, False)
        if self.global_rank == 0:
            # request new prompts for all ranks from controller only on global rank 0
            # this is to avoid getting different number of prompts at different ranks
            #
            # Hold _prompt_fetch_lock across the empty-check + fetch so that
            # main_loop and the optional prefetch thread (_prefetch_loop, gated
            # by config.rollout.prefetch_rollout) can never observe an empty
            # queue concurrently and double-fetch from the controller.
            with self._prompt_fetch_lock:
                if prompt_queue.qsize() < max_pending_batches:
                    # blocking request to get prompts from controller
                    # batch_size is per data parallel rank so we need to multiply it with data parallel size
                    payloads, is_end = self.api_client.get_next_prompt(
                        batch_size * self.parallel_dims.mesh["dp"].size(), **kwargs
                    )

                    assert all(payload["prompt_idx"] >= 0 for payload in payloads), (
                        "All payloads should have a valid prompt index"
                    )

                    if self.config.train.train_policy.data_dispatch_as_rank_in_mesh:
                        for payload in payloads:
                            assert (
                                payload["prompt_idx"] % len(self.replica_name_to_rank)
                                == self.rank_in_rollout_repicas
                            ), (
                                f"Payload prompt_idx {payload['prompt_idx']} mod {len(self.replica_name_to_rank)} must equal to rank in rollout replicas {self.rank_in_rollout_repicas}"
                            )
                    is_validation = kwargs.get("validation_step", None) is not None

                    if len(payloads) > 0:
                        if self.config.train.local_dataset:
                            for payload in payloads:
                                payload["prompt"] = (
                                    self.data_fetcher.get_payload_by_index(
                                        payload["prompt_idx"],
                                        is_validation=is_validation,
                                    )
                                )
                                payload["conversation"] = (
                                    self.data_fetcher.get_payload_by_index(
                                        payload["prompt_idx"],
                                        is_validation=is_validation,
                                        attr="conversation",
                                    )
                                )
                        payloads = [
                            RLPayload.model_validate(payload) for payload in payloads
                        ]
                    prompts_and_is_end = (
                        payloads if len(payloads) > 0 else None,
                        is_end,
                    )

        # Broadcast the prompts and is_end to all ranks
        prompts_and_is_end = dist_utils.broadcast_object_cpu(prompts_and_is_end)
        if self.parallel_dims.mesh["dp"].size() > 1:
            # Scatter the prompts to all data parallel ranks
            prompts, is_end = prompts_and_is_end
            if (
                prompts is not None
                and self.parallel_dims.mesh["dp"].get_local_rank() == 0
            ):
                # assert (
                #     len(prompts) % self.parallel_dims.mesh["dp"].size() == 0
                # ), f"Number of prompts {len(prompts)} must be divisible by data parallel size {self.parallel_dims.mesh['dp'].size()}"
                ranks_to_scatter = self.parallel_dims.mesh["dp"].size()

                # Distribute prompts in an interleaved (round-robin) fashion
                # Rank 0 gets indices [0, N, 2N, ...], Rank 1 gets [1, N+1, 2N+1, ...], etc.
                scattered_prompts_and_is_end = []
                for rank in range(ranks_to_scatter):
                    rank_prompts = prompts[rank::ranks_to_scatter]
                    scattered_prompts_and_is_end.append(
                        (
                            rank_prompts,
                            is_end,
                        )
                    )
            else:
                scattered_prompts_and_is_end = [
                    (None, is_end) for _ in range(self.parallel_dims.mesh["dp"].size())
                ]
            recv_prompts_and_is_end = [(None, False)]
            dist.scatter_object_list(
                recv_prompts_and_is_end,
                scattered_prompts_and_is_end,
                group=self.parallel_dims.mesh["dp"].get_group(),
                group_src=0,
            )
            prompts_and_is_end = recv_prompts_and_is_end[0]
        prompts, is_end = prompts_and_is_end
        if prompts is not None:
            prompt_queue.put(prompts)
        return is_end

    def consume_one_command(self, cmd_pred: Optional[Callable[[Command], bool]] = None):
        current_command = None
        if self.global_rank == 0:
            if not self._command_queue.empty():
                if cmd_pred is None:
                    current_command = self._command_queue.get()
                else:
                    if cmd_pred(self._command_queue.queue[0]):
                        current_command = self._command_queue.get()
                    else:
                        # Do not go on if the command is not expected
                        current_command = None

        current_command = dist_utils.broadcast_object_cpu(current_command)

        if current_command is not None:
            handler = self.get_rollout_command_handler(type(current_command))
            if handler is None:
                raise Exception(
                    f"No such command supoorted in rollout {current_command}"
                )
            try:
                logger.debug(
                    f"[Rollout] Executing command: {current_command._serialize()} for rank: {self.global_rank}"
                )
                handler(self, current_command)
                logger.debug(
                    f"[Rollout] Command executed: {current_command._serialize()} for rank: {self.global_rank}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Rollout] Command execution failed for {current_command._serialize()}"
                ) from e
        return current_command

    def consume_command(
        self,
        cmd_pred: Optional[Callable[[Command], bool]] = None,
        timeout=constant.COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT,
    ):
        """Consume all pending commands from the command queue.

        In async R2R mode, P2R/R2R commands are routed directly to the
        WeightSyncThread by ``query_command_from_controller`` and never
        appear in ``_command_queue``.  The "wait for R2R after P2R"
        logic only applies when both command types flow through the
        queue (synchronous mode).
        """
        async_wst_active = getattr(self, "_weight_sync_thread", None) is not None
        last_cmd = None
        none_cnt = 0
        start_time = time.time()
        while time.time() - start_time < float(timeout):
            cmd = self.consume_one_command(cmd_pred=cmd_pred)
            if cmd is not None:
                last_cmd = cmd
                none_cnt = 0
                start_time = time.time()
            else:
                none_cnt += 1
            if none_cnt >= constant.COSMOS_ROLLOUT_CMD_WAIT_TIMES and (
                async_wst_active
                or (
                    last_cmd is not None
                    and not isinstance(last_cmd, PolicyToRolloutUnicastCommand)
                )
                or last_cmd is None
            ):
                break
            time.sleep(constant.COSMOS_ROLLOUT_CMD_WAIT_INTERVAL)

    def send_end_signal(self):
        """
        Send end signal to the controller.
        This is used to notify the controller that the rollout worker has finished processing all prompts.
        """
        if self._rollout_end_acknowledged:
            return True

        payloads, is_validation, _, empty = self.report_rollouts(block=True)
        assert not is_validation and payloads is None and empty, (
            f"Payloads must be empty and not for validation when sending end signal {is_validation}, {payloads}, {empty}"
        )
        response = RolloutRequest(
            src_replica_name=self.replica_name,
            src_global_rank=self.global_rank,
            payloads=[],
            is_end=True,
        )
        logger.info(f"[Rollout] Posting rollout end signal to controller: {response}")
        self._rollout_end_acknowledged = bool(
            self.api_client.post_rollout_completion(response)
        )

        # Disaggregated workers whose end POST was acknowledged stay in the
        # command loop until the controller's explicit StopCommand.  Preserve
        # the existing local escape only when a single-process worker cannot
        # reach the controller, and for colocated mode, which has no separate
        # rollout StopCommand channel.
        if self.parallel_dims.world_size == 1 and (
            self.config.mode == "colocated" or not self._rollout_end_acknowledged
        ):
            self.shutdown_signal.set()

        return self._rollout_end_acknowledged

    def dynamic_sampling(self, payloads: List[RLPayload]):
        """
        Dynamic sampling: Filter out the rollouts that the rewards are all the same.
        This is used to filter out the rollouts that are not useful for training.
        """
        # DAPO only needs valid rollouts for training with dynamic sampling.
        # Separate valid and invalid rollouts for Dynamic Sampling
        # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
        valid_payloads = []
        metadata = {}
        for payload in payloads:
            # Collect the statistics for valid and filtered rollouts.
            if payload.valid:
                key = "sampled"
                metadata[key] = metadata.get(key, 0) + len(payload.completions)
                valid_payloads.append(payload)
            else:
                filter_reward = payload.filter_rewards[0]
                key = "filtered_positive" if filter_reward > 0 else "filtered_negative"
                metadata[key] = metadata.get(key, 0) + len(payload.completions)
        return valid_payloads, metadata

    def report_rollouts(self, block=False):
        while True:
            payloads, is_validation, step, empty = (
                self.reward_dispatcher.dequeue_rewards_cal()
            )
            if payloads is not None:
                if is_validation:
                    break

                metadata = {}
                if self.config.train.train_policy.variant == "dapo":
                    payloads, metadata_from_dapo = self.dynamic_sampling(payloads)
                    metadata.update(metadata_from_dapo)

                for i in range(len(payloads)):
                    (
                        payloads[i].completions,
                        payloads[i].completed_conversations,
                        payloads[i].completion_logprobs,
                        payloads[i].completion_token_ids,
                        _,
                    ) = self.data_packer.get_rollout_output(
                        payloads[i].completions,
                        payloads[i].completed_conversations,
                        payloads[i].completion_logprobs,
                        payloads[i].completion_token_ids,
                    )
                    # when using local dataset, we don't need to send the prompt/conversation to the controller
                    if self.config.train.local_dataset:
                        payloads[i].prompt = None
                        payloads[i].conversation = None
                    if self.config.train.train_policy.rollout_as_token_ids:
                        payloads[i].completions = [""] * len(payloads[i].completions)

                response = RolloutRequest(
                    src_replica_name=self.replica_name,
                    payloads=payloads,
                    metrics=metadata,
                    is_end=False,
                )
                self.api_client.post_rollout_completion(response)
            elif not block or empty:
                break
        return payloads, is_validation, step, empty

    def _call_rollout_generation(self, **kwargs) -> list:
        """Call ``rollout_generation`` with pre-generation buffer sync.

        All call sites should use this instead of calling
        ``self.rollout.rollout_generation`` directly so that async weight
        sync and weight-version injection happen consistently.
        """
        async_mode = get_async_r2r_sync_mode(self)
        if async_mode != AsyncR2RSyncMode.DISABLED:
            if async_mode == AsyncR2RSyncMode.INFERENCE and not getattr(
                self, "_inference_sync_installed", False
            ):
                install_inference_sync(self)
                self._inference_sync_installed = True
            sync_buffer_to_live(self)

        kwargs["current_weight_version"] = self.current_weight_version
        return self.rollout.rollout_generation(**kwargs)

    @torch.no_grad()
    def main_loop(self):
        async_mode = get_async_r2r_sync_mode(self)
        logger.info("[Rollout] main_loop async_r2r_sync mode: %s", async_mode.value)

        assert not (
            self._is_async_rollout and async_mode != AsyncR2RSyncMode.DISABLED
        ), (
            "async_r2r_sync is not supported with rollout.mode='async'. "
            "async_r2r_sync targets the synchronous rollout path; the async "
            "rollout scheduler (vllm_async) uses a separate generation path "
            "that bypasses the buffer model."
        )

        try:
            self._main_loop_impl()
        finally:
            # Stop the UCXX output server *before* any NCCL teardown/abort.
            # The rollout backend (e.g. rl-gym's ModularRolloutWorker via
            # UCXXRolloutMixin) serves this replica's generated output to the
            # trainer over a UCXX server.  cleanup_ucxx() is otherwise never
            # invoked, so at end-of-data a trainer can still be pulling the
            # final output when teardown begins -- and that concurrent UCXX
            # activity wedges ncclCommAbort on the straggler.  stop_server()
            # sets the shutdown flag and bounded-joins its threads (in-flight
            # reads finish first), so this is a graceful close, not a kill.
            self._cleanup_payload_server()
            wst = getattr(self, "_weight_sync_thread", None)
            if wst is not None:
                logger.info(
                    "[Teardown] %s: stopping WeightSyncThread", self.replica_name
                )
                _t_wst = time.monotonic()
                wst.stop()
                logger.info(
                    "[Teardown] %s: WeightSyncThread.stop() returned in %.2fs "
                    "(thread_alive=%s)",
                    self.replica_name,
                    time.monotonic() - _t_wst,
                    wst._thread.is_alive(),
                )

    def _cleanup_payload_server(self) -> None:
        """Best-effort, bounded shutdown of the rollout's UCXX output server.

        Invoked at the very start of teardown (before NCCL abort).  The server
        lives on the rollout backend (``self.rollout``) when it composes
        ``UCXXRolloutMixin``; guarded with ``getattr`` so backends without a
        UCXX server are unaffected.
        """
        rollout_engine = getattr(self, "rollout", None)
        cleanup = getattr(rollout_engine, "cleanup_ucxx", None)
        if not callable(cleanup):
            return
        logger.info(
            "[Teardown] %s: stopping UCXX output server (cleanup_ucxx)",
            self.replica_name,
        )
        _t0 = time.monotonic()
        try:
            cleanup()
        except Exception:
            logger.exception(
                "[Teardown] %s: cleanup_ucxx raised (continuing teardown)",
                self.replica_name,
            )
        logger.info(
            "[Teardown] %s: UCXX output server stopped in %.2fs",
            self.replica_name,
            time.monotonic() - _t0,
        )

    # Main-loop branch counters and prompt-version rejection logging.
    _MAINLOOP_LOG_INTERVAL_S = 1.0
    _VERSION_FAIL_LOG_INTERVAL_S = 5.0

    def _maybe_emit_mainloop_summary(self, now):
        """Emit a bounded summary of which branch the loop took.

        This is the *only* aggregated counter we keep in the rollout main
        loop.  We do not log per-iteration -- a hot-spinning loop can
        easily produce thousands of ``generate start`` lines per second per
        worker. Instead we tally
        branch hits in ``_mainloop_branch_counts`` and flush once per
        ``_MAINLOOP_LOG_INTERVAL_S`` (default 1s) so a stuck rollout
        produces a handful of lines/sec, not millions.
        """
        if now - self._mainloop_log_last_ts < self._MAINLOOP_LOG_INTERVAL_S:
            return
        c = self._mainloop_branch_counts
        # Skip emission when the previous window was completely idle
        # (loop-pump was blocked elsewhere).
        total = sum(c.values())
        if total == 0:
            self._mainloop_log_last_ts = now
            return
        logger.debug(
            "[Rollout main_loop %.1fs] rank=%d empty_q=%d consume_end=%d "
            "version_fail=%d gen_attempted=%d gen_succeeded=%d "
            "fetched_nonempty=%d weight_unsynced=%d",
            self._MAINLOOP_LOG_INTERVAL_S,
            self.global_rank,
            c["empty_q"],
            c["consume_end"],
            c["version_fail"],
            c["gen_attempted"],
            c["gen_succeeded"],
            c["fetched_nonempty"],
            c["weight_unsynced"],
        )
        for k in c:
            c[k] = 0
        self._mainloop_log_last_ts = now

    def _main_loop_impl(self):
        """Core main loop extracted for clean WST lifecycle management."""
        async_mode = get_async_r2r_sync_mode(self)

        # Per-worker branch counters for the legacy sync rollout path. The
        # async rollout path uses ``stream_generation_step`` instead.
        self._mainloop_branch_counts = {
            "empty_q": 0,
            "consume_end": 0,
            "version_fail": 0,
            "gen_attempted": 0,
            "gen_succeeded": 0,
            "fetched_nonempty": 0,
            "weight_unsynced": 0,
        }
        self._mainloop_log_last_ts = time.time()
        self._version_fail_last_log_ts = 0.0

        # Multi-rank prep overlap.  When prefetch is requested but the
        # worker is multi-rank, the dedicated rank-0 ``_prefetch_loop``
        # (fetch overlap) is unavailable -- its HTTP fetch ends in a
        # collective every rank must join.  Instead ``main_loop`` fetches
        # one batch ahead (in lockstep, via ``request_new_prompts`` with
        # ``max_pending_batches=2``) and hands the freshly-fetched batch
        # to ``_submit_prefetch_setup`` so per-prompt ``_prepare_sample``
        # runs on the mixin's bg setup thread while the batch ahead of it
        # generates.  Single-process workers (``_single_producer_mode``)
        # keep the dedicated fetch+prep loop and ``main_loop`` stays
        # consume-only, so prep_overlap is False for them here.
        prep_overlap = (
            self.config.rollout.prefetch_rollout and not self._single_producer_mode
        )
        if prep_overlap:
            self._bind_prefetch_context_once()

        while not self.shutdown_signal.is_set():
            self.consume_command(cmd_pred=None)

            # Process deferred validation/shutdown from the WST on the
            # main thread — never inside inference callbacks.
            if async_mode != AsyncR2RSyncMode.DISABLED:
                process_wst_deferred_actions(self)

            if self.validation_flag.is_set():
                self.do_validation()

            now = time.time()
            self._maybe_emit_mainloop_summary(now)

            # Multi-rank coordinated shutdown is driven by the controller's
            # explicit ``StopCommand`` (consumed above via ``consume_command``
            # -> ``handle_stop``).  Because ``consume_one_command`` broadcasts
            # the dequeued command across ranks, every rank sets the shutdown
            # signal at the same collective call and leaves ``main_loop`` in
            # lockstep -- so no rank strands a peer in a later collective.
            # This replaces the former Option-C per-iteration drain vote (a
            # CPU all-reduce gated on ``prompt_fetch_end``), which existed
            # only because the previous stop signal rode on the racy R2R
            # weight-sync broadcast.

            if not self.state.weight_synced():
                self._mainloop_branch_counts["weight_unsynced"] += 1
                continue

            _, is_validation, _, _ = self.report_rollouts()
            assert not is_validation, (
                "Validation report should be handled in the broadcast command rather than main loop."
            )

            if self._is_async_rollout:
                self.stream_generation_step()
                continue

            # In single-producer mode the prefetch thread owns
            # prompt fetching; ``main_loop`` is consume-only and
            # ``state.prompt_fetch_end`` is set by the prefetch
            # thread when the controller signals end-of-prompts.
            # In legacy mode ``main_loop`` is the producer and
            # drives ``request_new_prompts`` itself.
            if not self._single_producer_mode and not self.state.prompt_fetch_end():
                # Legacy cadence fetches one batch when the queue is
                # empty (depth 1).  Prep-overlap stages one batch ahead
                # (depth 2): with a batch already queued, its per-prompt
                # prep runs on the mixin's bg setup thread while the
                # batch in front of it generates.  The fetch is a
                # cross-rank collective, so every rank runs this loop in
                # lockstep -- the loop bound (queue depth, prompt_fetch_end,
                # whether the last fetch added a batch) is identical on
                # all ranks.
                target_depth = 2 if prep_overlap else 1
                while (
                    not self.state.prompt_fetch_end()
                    and self._prompt_queue.qsize() < target_depth
                ):
                    pre_qsize = self._prompt_queue.qsize()
                    no_more_prompts = self.request_new_prompts(
                        self.batch_size,
                        self._prompt_queue,
                        rank_in_mesh=self.rank_in_rollout_repicas,
                        max_pending_batches=target_depth,
                    )
                    made_progress = self._prompt_queue.qsize() > pre_qsize
                    if made_progress:
                        self._mainloop_branch_counts["fetched_nonempty"] += 1
                        # Kick off rank-local _prepare_sample for the
                        # batch we just enqueued (the deque tail).  No
                        # collective here -- only the fetch above is in
                        # lockstep.
                        if prep_overlap:
                            self._submit_prefetch_setup(self._prompt_queue.queue[-1])
                    if no_more_prompts:
                        logger.info(
                            f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation"
                        )
                        self.state.set_prompt_fetch_end()
                        if self._prompt_queue.empty():
                            self.state.set_prompt_consume_end()
                            if self.should_report:
                                self.send_end_signal()
                        break
                    # Controller had nothing to hand out right now: stop
                    # filling so we don't hot-spin the fetch RPC, and let
                    # the rest of the loop run.  Retried next iteration.
                    if not made_progress:
                        break

            if self.state.prompt_consume_end():
                assert self._prompt_queue.empty() and self.state.prompt_fetch_end(), (
                    "[Rollout] If prompt are all consumed, prompt queue should be empty and prompt end event should be set."
                )
                self._mainloop_branch_counts["consume_end"] += 1
                if self.should_report:
                    self.send_end_signal()
                continue
            elif self._prompt_queue.empty():
                # In single-producer mode the queue draining + a
                # set ``prompt_fetch_end`` means we're done; signal
                # the controller and let the next iteration hit the
                # ``prompt_consume_end`` gate above.  Otherwise this
                # is a transient empty -- prefetch hasn't filled
                # yet -- and we sleep briefly to avoid busy-waiting
                # while the producer races to fetch.  In legacy
                # mode an empty queue here means the controller has
                # nothing yet; the next iteration's
                # ``request_new_prompts`` will fetch.
                if self._single_producer_mode:
                    if self.state.prompt_fetch_end():
                        self.state.set_prompt_consume_end()
                        if self.should_report:
                            self.send_end_signal()
                    else:
                        time.sleep(0.05)
                self._mainloop_branch_counts["empty_q"] += 1
                continue
            else:
                logger.debug(f"[Rollout] generate start for rank {self.global_rank}")

                first_payload: RLPayload = self._prompt_queue.queue[0][0]
                allowed = self.config.train.train_policy.allowed_outdated_steps
                ceiling = self.current_weight_version + allowed
                is_valid_prompt_for_current_weight_version = (
                    first_payload.weight_version <= ceiling
                )

                if not is_valid_prompt_for_current_weight_version:
                    self._mainloop_branch_counts["version_fail"] += 1
                    # Explain prompt-version rejections at most once per 5s
                    # per worker.
                    if (
                        now - self._version_fail_last_log_ts
                        >= self._VERSION_FAIL_LOG_INTERVAL_S
                    ):
                        self._version_fail_last_log_ts = now
                        logger.info(
                            "[Rollout rank=%d] prompt rejected: "
                            "prompt.weight_version=%s current_weight_version=%s "
                            "allowed_outdated=%d ceiling=%s; head-of-queue "
                            "will be re-checked until current_weight_version advances",
                            self.global_rank,
                            first_payload.weight_version,
                            self.current_weight_version,
                            allowed,
                            ceiling,
                        )
                    # Back off before re-checking the head-of-queue
                    # prompt.  The rejection clears as soon as the next
                    # P->R broadcast advances ``current_weight_version``
                    # (typically every few seconds), so without a sleep
                    # this branch hot-spins.  50ms wakes well within
                    # the broadcast interval and is invisible to
                    # throughput.
                    time.sleep(0.05)
                    continue

                self._mainloop_branch_counts["gen_attempted"] += 1
                self.one_step_generation()
                self._mainloop_branch_counts["gen_succeeded"] += 1

                if self.state.prompt_fetch_end() and self._prompt_queue.empty():
                    self.state.set_prompt_consume_end()
                    if self.should_report:
                        self.send_end_signal()
        logger.info(f"[Rollout] Main loop of {self.replica_name} finished")

    def _filter_valid_rollout_results_and_report(
        self, rollout_results: List[RolloutResult], payloads_list: List[RLPayload]
    ) -> Tuple[List[RolloutResult], List[RLPayload]]:
        """
        Filter the rollout results with valid completions or valid completed_conversations.
        Returns the valid payloads and valid results for reporting.
        """
        # we need filter the result with valid completions or valid completed_conversations
        valid_result: List[RolloutResult] = []
        valid_payloads_list: List[RLPayload] = []
        if self.config.train.non_text:
            for payload, rr in zip(payloads_list, rollout_results):
                if rr.completions is not None and len(rr.completions) > 0:
                    valid_result.append(rr)
                    valid_payloads_list.append(payload)
        elif self.config.rollout.multi_turn_config.enable:
            for payload, rr in zip(payloads_list, rollout_results):
                valid_conversations: List[ConversationType] = []
                # remove those result without valid assistant message
                flag = False
                for conversation in rr.completed_conversations:
                    for msg in conversation:
                        if msg.role == "assistant" and msg.content != "":
                            flag = True
                            break
                    if flag:
                        valid_conversations.append(conversation)
                rr.completed_conversations = valid_conversations
                if len(rr.completed_conversations) > 0:
                    valid_result.append(rr)
                    valid_payloads_list.append(payload)
        else:
            # Remove empty completions
            for payload, rr in zip(payloads_list, rollout_results):
                completions = rr.completions
                skip_output = False
                total_generation_count = len(completions)
                empty_generation_count = 0
                output_texts: List[str] = []
                for j in range(total_generation_count):
                    output_text = completions[j]
                    # if output_text == "":
                    #     logger.warning(
                    #         f"[Rollout] Got empty completion for {i}th prompt {j}th generation"
                    #     )
                    #     empty_generation_count += 1
                    # else:
                    #     output_texts.append(output_text)

                    # Note: (jiaxinc)
                    # We still need to upload the output text, even if it is empty. (replace empty with eos_token)
                    # Because if fully synchronized mode is enabled, we need to make sure the expected
                    # number of global_batch_size is reached at exact time.
                    output_texts.append(
                        output_text if output_text != "" else self.eos_token
                    )
                # Skip the output if there is one or zero non-empty completions
                # We keep one completion case
                skip_output = (total_generation_count - empty_generation_count) <= 0
                if not skip_output:
                    rr.completions = output_texts
                    valid_result.append(rr)
                    valid_payloads_list.append(payload)

        should_report = self.should_report and len(valid_result) > 0
        if should_report:
            valid_payloads: List[RLPayload] = []
            # only the first tp rank in the rollout replica will post the completion to the controller.
            for old_payload, result in zip(valid_payloads_list, valid_result):
                # update payload
                old_payload.completions = result.completions
                old_payload.completion_logprobs = result.completion_logprobs
                old_payload.completion_token_ids = result.completion_token_ids
                old_payload.prompt_logprobs = result.prompt_logprobs
                old_payload.prompt_token_ids = result.prompt_token_ids
                old_payload.weight_version = self.current_weight_version
                old_payload.cumulative_logprob = result.cumulative_logprob
                old_payload.extra_info = result.extra_info
                if self.config.rollout.multi_turn_config.enable:
                    old_payload.completed_conversations = result.completed_conversations
                if self.config.train.local_dataset:
                    old_payload.reference_answer = (
                        self.data_fetcher.query_reference_answer(
                            old_payload.prompt_idx,
                        )
                    )
                valid_payloads.append(old_payload)
            valid_payloads = self.enqueue_teacher_calculation(valid_payloads)
            self.reward_dispatcher.enqueue_rewards_cal(
                valid_payloads,
                False,
                self.current_weight_version,
                bypass_reward=self.config.train.train_policy.bypass_reward,
            )
        return valid_payloads_list, valid_result

    def one_step_generation(
        self,
    ) -> Tuple[List[RLPayload], List[RolloutResult]]:
        """
        Perform one step of rollout generation.
        Returns the number of valid payloads generated.
        """
        generation_start_ts = time.time()
        try:
            _peek_wv = (
                self._prompt_queue.queue[0][0].weight_version
                if not self._prompt_queue.empty()
                else None
            )
        except Exception:
            _peek_wv = None
        # Paired with the matching exit log below for per-call latency
        # measurements when DEBUG logging is enabled.
        logger.debug(
            "[one_step_generation entry] rank=%d cur_wv=%s peek_prompt_wv=%s",
            self.global_rank,
            self.current_weight_version,
            _peek_wv,
        )

        payloads_list: List[RLPayload] = self._prompt_queue.get()

        rollout_results: List[RolloutResult] = self._call_rollout_generation(
            payloads=payloads_list,
            stream=self.inference_stream,
            data_packer=self.data_packer,
            data_fetcher=self.data_fetcher,
            is_validation=False,
        )

        if len(rollout_results) == 0:
            logger.debug(
                "[one_step_generation exit] rank=%d elapsed_ms=%.1f "
                "batch=%d produced=0 returned_false=True",
                self.global_rank,
                (time.time() - generation_start_ts) * 1000.0,
                len(payloads_list),
            )
            return False

        assert len(rollout_results) == len(payloads_list), (
            f"Error: Rollout engine returned {len(rollout_results)} for {len(payloads_list)}"
        )

        logger.debug(f"[Rollout] generate end for rank {self.global_rank}")

        result = self._filter_valid_rollout_results_and_report(
            rollout_results, payloads_list
        )
        logger.debug(
            "[one_step_generation exit] rank=%d elapsed_ms=%.1f "
            "batch=%d produced=%d returned_false=False",
            self.global_rank,
            (time.time() - generation_start_ts) * 1000.0,
            len(payloads_list),
            len(rollout_results),
        )
        return result

    def _stream_generation_feed_prompts(
        self,
        batch_size: int,
        prompt_queue: Queue,
        validation_step: Optional[int] = None,
    ) -> Tuple[int, bool]:
        """
        Perform one step of stream rollout generation.
        feed the prompts to the rollout_scheduler and collect the rollout results, report the rollout results to the controller.

        This function is non-blocking.

        Args:
            batch_size (int): the batch size of the prompts to fetch
            prompt_queue (Queue): the queue to store the prompts
            validation_step (Optional[int]): the validation step, if None, means no validation.

        Return:
            feed_prompts_count (int): the number of prompts fed to the scheduler
            is_end (bool): whether there is no more prompts to fetch
        """
        if self.scheduler.is_busy():
            # skip fetching new prompts if the scheduler is busy
            return 0, False

        request_prompts_count = min(
            batch_size,
            self.scheduler.max_concurrent_requests
            - self.scheduler.pending_tasks()
            - self.scheduler.active_tasks(),
        )
        if request_prompts_count <= 0:
            return 0, False

        is_end = self.request_new_prompts(
            request_prompts_count,
            prompt_queue,
            validation_step=validation_step,
            rank_in_mesh=self.rank_in_rollout_repicas,
        )

        is_validation = validation_step is not None

        # Check if the prompt is valid for the current weight version
        if not is_validation and not self._prompt_queue.empty():
            first_payload: RLPayload = self._prompt_queue.queue[0][0]
            is_valid_prompt_for_current_weight_version = (
                first_payload.weight_version
                <= self.current_weight_version
                + self.config.train.train_policy.allowed_outdated_steps
            )
            if not is_valid_prompt_for_current_weight_version:
                return 0, False

        # try to get the prompts from the prompt queue, even if the prompt queue is empty.
        try:
            payloads_list: List[RLPayload] = prompt_queue.get_nowait()
        except QueueEmpty:
            # if the prompt queue is empty, just skip feed the scheduler.
            return 0, is_end

        # packing the prompts into tasks and put into the scheduler
        tasks = [
            RolloutTask(
                idx=payload.prompt_idx,
                payload=payload,
                is_validation=False,
            )
            for payload in payloads_list
        ]
        self.scheduler.put_rollout_batch(tasks)
        return len(payloads_list), is_end

    def _stream_generation_collect_results(self):
        """
        Collect the rollout results from the scheduler.

        This function is non-blocking.
        """
        results: List[CompletedRollout] = self.scheduler.get_all()

        if len(results) == 0:
            return

        payloads_list: List[RLPayload] = []
        rollout_results: List[RolloutResult] = []
        for cr in results:
            payloads_list.append(cr.payload)
            rollout_results.append(cr.result)

        self._filter_valid_rollout_results_and_report(rollout_results, payloads_list)

    def stream_generation_step(self):
        """
        Perform the stream rollout generation step, include 3 sub-steps:
        1. update the state of the rollout generation worker.
        1. get prompts from controller and feed to the scheduler.
        3. collect the rollout results from the scheduler and enqueue the reward calculation.

        This function is non-blocking.
        """
        # update the state of the rollout generation worker
        if (
            self.state.prompt_fetch_end()
            and self.scheduler.is_all_tasks_completed()
            # all reward calculation tasks are reported
            and self.reward_dispatcher.is_empty()
        ):
            self.state.set_prompt_consume_end()

        if not self.state.prompt_fetch_end():
            _, is_end = self._stream_generation_feed_prompts(
                self.batch_size, self._prompt_queue, validation_step=None
            )
            if is_end:
                logger.info(
                    f"[Rollout] Receive prompt end, wait for {self.replica_name} to finish all rollouts generation"
                )
                self.state.set_prompt_fetch_end()

        self._stream_generation_collect_results()

        # Check if all prompts are consumed, if so, send end signal to the controller.
        if self.state.prompt_consume_end():
            # Each reporting rank owns an independent reward dispatcher;
            # flush it and report that rank's local drain exactly once.
            if self.should_report:
                self.send_end_signal()

    def enqueue_teacher_calculation(self, payloads: List[RLPayload]) -> List[RLPayload]:
        """
        Enqueue the teacher calculation for the payloads.
        Args:
            payloads: The payloads to enqueue the teacher calculation for.
        Returns:
            The payloads with the teacher result uuid.
        """
        if not self.config.distillation.enable:
            return payloads
        assert all(payload.completion_token_ids is not None for payload in payloads), (
            "All payloads must have completion token ids"
        )
        for payload in payloads:
            data = {
                "prompt_idx": payload.prompt_idx,
                "completion_token_ids": payload.completion_token_ids,
            }
            if payload.prompt_token_ids is not None:
                data["prompt_token_ids"] = payload.prompt_token_ids
            uuid_values = []
            for _ in payload.completion_token_ids:
                uuid_value = str(uuid.uuid4())
                uuid_values.append(uuid_value)
            data["teacher_result_uuid"] = uuid_values
            self.teacher_interact_queue.put_nowait(data)
            payload.teacher_result_uuids = uuid_values
            if self.config.distillation.trainer_token_ids_from_teacher:
                # offload the verbose token ids out of the payload for efficient communication
                # only keep the first token id which is selected
                # the full token ids will be fetched from teacher model during distillation
                payload.completion_token_ids = [
                    [t[0:1] for t in compl] for compl in payload.completion_token_ids
                ]
                payload.prompt_token_ids = [t[0:1] for t in payload.prompt_token_ids]
        return payloads

    def _bind_prefetch_context_once(self):
        """Bind data_packer / data_fetcher into the mixin's prefetch
        context so the bg setup worker resolves payloads via the same
        packer the consumer uses.  Idempotent; no-op for backends that
        don't compose RolloutGenerationMixin.
        """
        if getattr(self, "_prefetch_context_bound", False):
            return
        bind = getattr(self.rollout, "bind_prefetch_context", None)
        if callable(bind):
            try:
                bind(
                    data_packer=self.data_packer,
                    data_fetcher=self.data_fetcher,
                )
            except Exception:
                logger.exception(
                    "[Rollout] bind_prefetch_context failed; "
                    "prefetch will run with empty context."
                )
        self._prefetch_context_bound = True

    def _submit_prefetch_setup(self, payloads):
        """Hand a freshly-fetched batch to the backend's prefetch setup so
        per-prompt ``_prepare_sample`` runs on the mixin's bg setup
        thread, overlapping with in-flight generation on earlier batches.

        Rank-local and keyed by ``prompt_idx`` -- it runs no collective --
        so it is safe to call from a multi-rank ``main_loop`` (unlike the
        controller prompt fetch, which must stay in lockstep across
        ranks).  Backends compose ``RolloutGenerationMixin`` (preferred)
        or define the legacy ``enqueue_prefetch_payloads`` hook; a backend
        with neither is legal and simply falls back to inline
        ``_prepare_sample`` in the consumer.
        """
        submit = getattr(self.rollout, "submit_setup", None)
        if callable(submit):
            try:
                submit(payloads)
            except Exception:
                logger.exception(
                    "[Rollout] submit_setup failed for batch of %d",
                    len(payloads),
                )
            return
        legacy = getattr(self.rollout, "enqueue_prefetch_payloads", None)
        if callable(legacy):
            _warn_legacy_prefetch_hook_once()
            try:
                legacy(payloads)
            except Exception:
                logger.exception(
                    "[Rollout] enqueue_prefetch_payloads failed for batch of %d",
                    len(payloads),
                )

    def _prefetch_loop(self):
        """Sole producer of ``_prompt_queue`` in single-producer mode.

        Active only when :attr:`_single_producer_mode` is True
        (``config.rollout.prefetch_rollout`` set + single-process
        worker; see :meth:`__init__`).  The thread is started from
        :meth:`work` and runs on global rank 0.

        Per iteration: fetch a prompt batch from the controller,
        notify the rollout backend via ``submit_setup`` so
        :class:`RolloutGenerationMixin` can start ``_prepare_sample``
        on its own bg thread, then ``put`` onto ``_prompt_queue``.
        Back-pressure is provided by the bounded queue: ``put``
        blocks when ``main_loop`` is behind, so this thread runs
        exactly as fast as the consumer drains -- no polling, no
        ``time.sleep``, no lock.

        ``submit_setup`` is called *before* ``put`` so the bg setup
        worker has a head start.  By the time ``main_loop`` pops
        the batch and reaches ``_gather_prepared_samples``, the
        future is typically done and the consumer waits ~0 ms.

        Backends that haven't migrated to the mixin can still
        implement the legacy ``enqueue_prefetch_payloads(payloads)``
        hook on their ``RolloutBase`` subclass; this is supported as
        a deprecation shim with a one-shot warning at first use.

        Lifecycle:
          * Wait for first weight-sync (a one-shot sticky bit; we
            poll only because ``State`` doesn't expose it as an
            event, and only at startup).
          * On controller ``is_end=True``, set
            ``state.prompt_fetch_end`` and exit; ``main_loop``
            drains the queue and signals consume-end.
          * On shutdown, exit ASAP; the bounded queue's blocking
            ``put`` is woken by a small timeout so the thread can
            check ``shutdown_signal`` even when ``main_loop`` is
            stuck.
          * On unexpected exception, set ``prompt_fetch_end`` from
            ``finally`` so ``main_loop`` doesn't wait forever for
            a producer that died.

        Multi-rank scope: this thread overlaps the controller *fetch*
        (the ``get_next_prompt`` HTTP round-trip), which ends in a
        distributed broadcast/scatter that all ranks must join -- driving
        it from rank 0 only would deadlock the peers, so the fetch-overlap
        loop requires ``world_size == 1`` (:attr:`_single_producer_mode`).
        The *prep* overlap (per-prompt ``_prepare_sample`` on the mixin's
        bg setup thread) does not need a collective and is available to
        multi-rank workers too -- see :meth:`_submit_prefetch_setup`,
        which ``_main_loop_impl`` calls after each in-lockstep fetch.
        """
        self._bind_prefetch_context_once()

        # Wait for first weight-sync.  ``State.weight_synced`` is a
        # sticky bit (set once, never cleared per
        # ``cosmos_rl/rollout/__init__.py:69``), so a tight poll
        # here is a one-shot and the loop body below has no
        # weight-sync gate.
        while not self.shutdown_signal.is_set() and not self.state.weight_synced():
            time.sleep(0.05)

        try:
            while not self.shutdown_signal.is_set():
                if self.state.prompt_fetch_end():
                    return

                # Sole producer: no lock, no empty-check (we always
                # try to fetch the next batch and let the bounded
                # queue's ``put`` back-pressure us off).
                try:
                    payloads, is_end = self.api_client.get_next_prompt(
                        self.batch_size,
                        rank_in_mesh=self.rank_in_rollout_repicas,
                    )
                except Exception:
                    logger.exception("[Rollout] Prefetch fetch failed")
                    # Backoff to avoid hammering an unhealthy
                    # controller.  Re-check shutdown immediately on
                    # the next iteration.
                    time.sleep(0.5)
                    continue

                if is_end:
                    self.state.set_prompt_fetch_end()
                if not payloads:
                    continue

                # Mirror request_new_prompts' local_dataset / RLPayload
                # validation so main_loop sees identical objects when
                # it pops from the queue.
                if self.config.train.local_dataset:
                    for payload in payloads:
                        payload["prompt"] = self.data_fetcher.get_payload_by_index(
                            payload["prompt_idx"],
                            is_validation=False,
                        )
                        payload["conversation"] = (
                            self.data_fetcher.get_payload_by_index(
                                payload["prompt_idx"],
                                is_validation=False,
                                attr="conversation",
                            )
                        )
                payloads = [RLPayload.model_validate(p) for p in payloads]

                # Notify the backend BEFORE put: the bg setup worker
                # starts preparing samples while we're still preparing
                # to enqueue, so prep overlaps in-flight generation.
                self._submit_prefetch_setup(payloads)

                # Blocking put with a small timeout so we stay
                # responsive to ``shutdown_signal`` even if the
                # consumer is hung.  ``queue.Full`` is the expected
                # case during steady-state back-pressure.
                while not self.shutdown_signal.is_set():
                    try:
                        self._prompt_queue.put(payloads, timeout=0.5)
                        break
                    except QueueFull:
                        continue
                if self.shutdown_signal.is_set():
                    return

                logger.info(
                    "[Rollout] Prefetched %d payloads (prompt_idxs=%s%s)",
                    len(payloads),
                    [p.prompt_idx for p in payloads[:5]],
                    " ..." if len(payloads) > 5 else "",
                )
        finally:
            # Defensive: if we crash or exit early, set fetch_end so
            # main_loop doesn't wait forever for a producer that
            # died.  Idempotent (the bit is sticky).
            self.state.set_prompt_fetch_end()

    def work(self):
        # Start the thread with daemon=True, so it will exit when the main program exits.
        if self.global_rank == 0:
            # create a thread to query command as a producer
            self.background_thread = threading.Thread(
                target=self.query_command_from_controller, daemon=True
            )
            self.background_thread.start()
            if self._single_producer_mode:
                logger.info(
                    "[Rollout] Prefetch enabled (single-producer mode); "
                    "starting background thread"
                )
                self.prefetch_thread = threading.Thread(
                    target=self._prefetch_loop,
                    daemon=True,
                    name="rollout-prefetch",
                )
                self.prefetch_thread.start()
            elif self.config.rollout.prefetch_rollout:
                # Multi-rank worker (world_size>1) with prefetch on.
                # The dedicated rank-0 ``_prefetch_loop`` can't run here
                # because its HTTP fetch ends in a distributed
                # broadcast/scatter that every rank must join, so the
                # *fetch* overlap stays single-process.  The *prep*
                # overlap is still active: ``_main_loop_impl`` fetches
                # one batch ahead (in lockstep across ranks) and hands
                # it to ``_submit_prefetch_setup`` so per-prompt
                # ``_prepare_sample`` runs on the mixin's bg setup
                # thread while earlier batches generate.
                logger.info(
                    "[Rollout] config.rollout.prefetch_rollout=True with "
                    "world_size=%d (>1): per-prompt prep overlap is ENABLED "
                    "via main_loop look-ahead; the rank-0 fetch-overlap loop "
                    "stays single-process (set dp/tp/pp/cp=1 to also overlap "
                    "the controller fetch).",
                    self.parallel_dims.world_size,
                )
        if self.config.distillation.enable:
            # create a thread to interact with teacher model
            self.teacher_interact_thread = threading.Thread(
                target=self.teacher_interact_loop, daemon=True
            )
            self.teacher_interact_thread.start()

        self.main_loop()
        # [Teardown trace] These calls are otherwise silent; emitting a
        # breadcrumb before/after each one means the last line in a wedged
        # worker's log names the exact blocking call (e.g. a hung
        # inference_stream.synchronize() from a cross-replica NCCL teardown
        # race) instead of an un-attributable silent gap before unregister.
        logger.info(
            "[Teardown] %s: main_loop returned; draining inference_stream",
            self.replica_name,
        )
        # Synchronous-path backstop: in async_r2r_sync=disabled there is no
        # WeightSyncThread, so R2R/P2R run on inference_stream; an orphaned
        # collective here is drained-or-aborted the same way.  Quiet on healthy
        # runs now that the controller waits for rollout checkout.
        bounded_drain_or_abort(
            self.inference_stream,
            _TEARDOWN_DRAIN_TIMEOUT_S,
            f"inference_stream[{self.replica_name}]",
        )
        self.handle_shutdown()
        logger.info("[Teardown] %s: work() returning", self.replica_name)
