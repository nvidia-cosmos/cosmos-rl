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

import torch.distributed as dist
import uuid
from typing import Dict, Callable, Type, Optional, Any, Union
import copy
import time
import atexit
import threading
from abc import ABC, abstractmethod
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.utils.network_util import get_local_ip
from cosmos_rl.dispatcher.command import (
    PolicyCommandRegistry,
    RolloutCommandRegistry,
    Command,
)
from cosmos_rl.dispatcher.data.packer import BaseDataPacker, DecoderOnlyLLMDataPacker
from cosmos_rl.dispatcher.data.packer import (
    HFVLMDataPacker,
)

from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.constant as constant
import cosmos_rl.utils.distributed as dist_utils
from cosmos_rl.dispatcher.protocol import MESH_NAMES
import cosmos_rl.utils.util as util
from cosmos_rl.utils.payload_transport import (
    PayloadTransportRegistry,
    RedisEndpoint,
    get_payload_transfer_mode,
    is_payload_transfer_mode_explicit,
)
from transformers import AutoConfig
import multiprocessing as mp
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.colocated.api_client import ColocatedAPIClient

try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None


class CommMixin:
    policy_command_handler_registry = PolicyCommandRegistry()
    rollout_command_handler_registry = RolloutCommandRegistry()

    @classmethod
    def register_policy_command_handler(cls, command_type: Type[Command]):
        def decorator(func):
            cls.policy_command_handler_registry.register(command_type, func)
            return func

        return decorator

    @classmethod
    def register_rollout_command_handler(
        cls, command_type: Type[Command], backend: str = "vllm"
    ):
        def decorator(func):
            cls.rollout_command_handler_registry.register(command_type, func, backend)
            return func

        return decorator

    @classmethod
    def get_policy_command_handler(
        cls, command_type: Type[Command]
    ) -> Optional[Callable]:
        return cls.policy_command_handler_registry.get_command_handler(command_type)

    @classmethod
    def get_rollout_command_handler(
        cls, command_type: Type[Command], backend: str = "vllm"
    ) -> Optional[Callable]:
        return cls.rollout_command_handler_registry.get_command_handler(
            command_type, backend
        )

    def init_comm(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        logger.info(
            f"{self.role} Replica started at global rank {self.global_rank}, with replica name: {self.replica_name}"
        )

        self.api_client = (
            ColocatedAPIClient(self.role)
            if hasattr(self, "colocated")
            else APIClient(self.role)
        )

        policy_type = self.config.train.train_policy.type
        if policy_type != "sft" or self.config.policy.parallelism.n_init_replicas > 1:
            # if not sft, we have to init redis
            # Or if sft but with multiple replicas, we also need redis for coordination
            self.init_redis()

        self.register_to_controller()

    def init_data_packer(
        self,
        data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
        val_data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
    ):
        if not self.config.policy.is_diffusers:
            hf_config = util.retry(AutoConfig.from_pretrained)(
                self.config.policy.model_name_or_path, trust_remote_code=True
            )
            is_vlm = getattr(hf_config, "vision_config", None) is not None
            model_type = hf_config.model_type
        else:
            model_type = "diffusers"
            is_vlm = False

        if data_packer:
            if isinstance(data_packer, Callable):
                self.data_packer = data_packer(self.config)
            elif isinstance(data_packer, BaseDataPacker):
                self.data_packer = data_packer
            else:
                raise ValueError(
                    "Data packer must be a BaseDataPacker instance or a factory function that "
                    " returns a BaseDataPacker instance"
                )
            logger.info(f"Using user-provided data packer: {self.data_packer}")
        else:
            try:
                self.data_packer = BaseDataPacker.get_default_data_packer(model_type)
                logger.info(f"Using default data packer: {self.data_packer}")
            except ValueError:
                self.data_packer = (
                    DecoderOnlyLLMDataPacker() if not is_vlm else HFVLMDataPacker()
                )
                logger.warning(
                    f"No default data packer found for {model_type}, using {type(self.data_packer).__name__} as default"
                )

        util.call_setup(self.data_packer, self.config)

        if val_data_packer:
            if isinstance(val_data_packer, Callable):
                self.val_data_packer = val_data_packer(self.config)
            elif isinstance(val_data_packer, BaseDataPacker):
                self.val_data_packer = val_data_packer
            else:
                raise ValueError(
                    "Validation data packer must be a BaseDataPacker instance or a factory function that "
                    " returns a BaseDataPacker instance"
                )
            logger.info(
                f"Using user-provided validation data packer: {self.val_data_packer}"
            )
        else:
            try:
                self.val_data_packer = BaseDataPacker.get_default_data_packer(
                    model_type
                )
                logger.info(
                    f"Using default validation data packer: {self.val_data_packer}"
                )
            except ValueError:
                self.val_data_packer = (
                    DecoderOnlyLLMDataPacker() if not is_vlm else HFVLMDataPacker()
                )
                logger.warning(
                    f"No default validation data packer found for {model_type}, using {type(self.val_data_packer).__name__} as default"
                )

        util.call_setup(self.val_data_packer, self.config)

        self._attach_payload_transport()

    # ------------------------------------------------------------------
    # Worker-side payload-transport attachment.
    #
    # Two compatibility surfaces are preserved here -- search for these
    # comments if downstream code (e.g. Yuxiao's PolicyDataPacker
    # introduced in PR #670 / commit 55745c) breaks:
    #
    #   (1) NCCL fast path:
    #       NcclPayloadTransport.attach_data_packer assigns
    #       packer.redis_client and THEN calls
    #       packer.post_redis_injection().  This is the literal
    #       contract from 55745c, hardened to survive ping failures
    #       without crashing init.
    #
    #   (2) Opportunistic Redis injection compat fallback:
    #       Pre-55745c convention (and any downstream packer that
    #       lazily depends on ``redis_client`` without selecting NCCL
    #       mode) expected a Redis client to land on the packer
    #       regardless of the active transport.  We replicate that
    #       behavior for any packer whose ``redis_client`` attribute is
    #       ``None`` after attach -- but ONLY when the active mode is
    #       NOT "nccl", to avoid double-injection (NCCL's attach
    #       already handled it).
    #
    # The deprecation shim ``_inject_redis_into_data_packers`` below
    # delegates to this method, keeping the old name reachable for
    # downstream callers that monkey-patched or invoked it directly.
    # ------------------------------------------------------------------
    def _attach_payload_transport(self):
        """Unified worker-side hook to wire payload transport into packers.

        Replaces the inline Redis-injection logic that used to live in
        ``_inject_redis_into_data_packers``.  Looks up the active
        :class:`PayloadTransport` from the registry, then for each data
        packer:

        1. Calls ``transport.attach_data_packer(...)`` so the transport
           can wire its own state in (e.g. NCCL injects a Redis client
           and runs ``post_redis_injection()``; UCXX starts prefetch
           threads; Redis default no-ops).
        2. Falls back to opportunistic Redis injection for non-NCCL
           modes to preserve pre-55745c behavior for packers that
           passively expose a ``redis_client`` attribute.

        Failure handling follows the ``explicit_fatal`` policy: when
        the user explicitly selected the transport in config, ``ImportError``
        / ``RuntimeError`` from attach is re-raised so misconfiguration
        (e.g. ``payload_transfer="ucxx"`` with no ``ucxx-cu12`` installed)
        crashes loudly.  Other failures are logged and swallowed.
        """
        mode = get_payload_transfer_mode(self.config)
        explicit = is_payload_transfer_mode_explicit(self.config)
        transport = PayloadTransportRegistry.get_optional(mode)
        endpoint = self._build_redis_endpoint()
        device = getattr(self, "device", None)

        packers = [(self.data_packer, "data_packer")]
        if (
            hasattr(self, "val_data_packer")
            and self.val_data_packer is not self.data_packer
        ):
            packers.append((self.val_data_packer, "val_data_packer"))

        for packer, packer_name in packers:
            if transport is not None:
                try:
                    transport.attach_data_packer(
                        packer,
                        config=self.config,
                        device=device,
                        redis_endpoint=endpoint,
                    )
                except (ImportError, RuntimeError) as exc:
                    if explicit:
                        # User-chosen transport failed hard (e.g.
                        # ucxx-cu12 not installed but config sets
                        # payload_transfer="ucxx").  Re-raise per
                        # explicit_fatal policy so the misconfig is
                        # visible immediately.
                        raise
                    logger.warning(
                        f"[{self.role}] {mode} attach_data_packer for "
                        f"{packer_name} failed: {exc}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[{self.role}] {mode} attach_data_packer for "
                        f"{packer_name} raised "
                        f"{type(exc).__name__}: {exc}"
                    )

            # (2) Opportunistic Redis injection compat fallback.  See
            # the long-form comment above for why this exists.  Skipped
            # when the active mode is NCCL because NCCL's attach is
            # already responsible for the assignment-then-hook order
            # from 55745c -- running it again here would either
            # double-inject or stomp on the assigned client.
            if mode != "nccl":
                self._opportunistic_inject_redis(packer, packer_name, endpoint)

    def _opportunistic_inject_redis(self, packer, packer_name: str, endpoint):
        """Best-effort Redis injection for non-NCCL modes.

        Compatibility surface (2) from ``_attach_payload_transport``:
        preserves pre-55745c permissive injection for downstream packers
        that lazily depend on ``redis_client`` without selecting NCCL
        mode.  No-op when the packer has no ``redis_client`` attribute,
        already has one wired up, or the redis library is unavailable.
        """
        if not hasattr(packer, "redis_client"):
            return
        if getattr(packer, "redis_client", None) is not None:
            # Some other code path (e.g. a transport's attach hook) has
            # already wired one in -- don't stomp on it.
            return
        if endpoint is None or _redis_lib is None:
            return
        try:
            client = _redis_lib.Redis(
                host=endpoint.host,
                port=endpoint.port,
                db=endpoint.db,
                decode_responses=True,
            )
            client.ping()
        except Exception as exc:
            logger.warning(
                f"[{self.role}] Opportunistic Redis injection for "
                f"{packer_name} failed: {exc}"
            )
            return
        packer.redis_client = client
        logger.info(
            f"[{self.role}] Injected Redis client into {packer_name} "
            f"(host={endpoint.host}, port={endpoint.port}, db={endpoint.db})"
        )
        hook = getattr(packer, "post_redis_injection", None)
        if callable(hook):
            try:
                hook()
            except Exception as exc:
                logger.warning(
                    f"[{self.role}] post_redis_injection on {packer_name} "
                    f"raised {type(exc).__name__}: {exc}"
                )

    def _inject_redis_into_data_packers(self):
        """DEPRECATED: use :meth:`_attach_payload_transport`.

        Kept as a one-line shim because some downstream forks (search
        PR #670 / commit 55745c) call this method by name or
        monkey-patch it directly.  Remove no earlier than two minor
        releases after the unification PR lands.
        """
        return self._attach_payload_transport()

    def _build_redis_endpoint(self) -> Optional[RedisEndpoint]:
        """Return a :class:`RedisEndpoint` for the worker's Redis, or None.

        Resolves to the live ``redis_controller`` connection coordinates
        when available, falling back to ``("localhost", 6379, 0)``
        overridden by ``config.redis`` (port-only, historical).
        """
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0
        redis_controller = getattr(self, "redis_controller", None)
        if redis_controller is not None and hasattr(redis_controller, "redis_clients"):
            clients = redis_controller.redis_clients
            if clients:
                conn_kwargs = clients[0].connection_pool.connection_kwargs
                redis_host = conn_kwargs.get("host", redis_host)
                redis_port = conn_kwargs.get("port", redis_port)
                redis_db = conn_kwargs.get("db", redis_db)
        config = getattr(self, "config", None)
        if config is not None and hasattr(config, "redis") and config.redis:
            redis_port = int(config.redis)
        return RedisEndpoint(host=redis_host, port=int(redis_port), db=int(redis_db))

    def register_to_controller(self):
        if hasattr(self, "_is_registered"):
            return

        target_mesh_names = copy.deepcopy(MESH_NAMES)
        ranks = []
        group_size = []
        for mesh_name in MESH_NAMES:
            if (
                self.parallel_dims.mesh.mesh_dim_names
                and mesh_name in self.parallel_dims.mesh.mesh_dim_names
            ):
                ranks.append(self.parallel_dims.mesh[mesh_name].get_local_rank())
                group_size.append(self.parallel_dims.mesh[mesh_name].size())
            else:
                ranks.append(0)
                group_size.append(1)

        host_info_tuple = get_local_ip()
        if host_info_tuple is None:
            raise Exception("Failed to get local IP address")
        host_ip, host_name = host_info_tuple
        self.api_client.register(
            replica_name=self.replica_name,
            role=self.role,
            mesh_names=target_mesh_names,
            ranks=ranks,
            group_size=group_size,
            global_rank=self.global_rank,
            host_ip=host_ip,
            host_name=host_name,
        )

        dist.barrier()  # wait all the atoms registered.

        self.shutdown_signal = threading.Event()
        self.shutdown_mp_signal = mp.Event()  # Must be a multiprocessing event

        if self.global_rank == 0:
            logger.info(
                f"{self.role} Replica {self.replica_name} registered to controller"
            )
            # Start the thread with daemon=True, so it will exit when the main program exits.
            process = mp.Process(
                target=self.heartbeat_trigger,
                args=(self.shutdown_mp_signal,),
                daemon=True,  # Dies when main process exits
            )
            process.start()
            self.heartbeat_thread = process
        else:
            self.heartbeat_thread = None

        self._is_registered = True
        atexit.register(self.unregister_from_controller)

    def unregister_from_controller(self):
        if not hasattr(self, "_is_registered"):
            return
        elif hasattr(self, "_is_unregistered"):
            return
        else:
            self._is_unregistered = True
        self._is_registered = False
        # let only rank == 0 send the unregister request
        if self.global_rank == 0:
            self.api_client.unregister(self.replica_name)

    def get_group_unique_key(self, replica_name_to_rank: Dict[str, int]):
        return (
            "_".join(
                [
                    k
                    for k, _ in sorted(
                        replica_name_to_rank.items(), key=lambda item: item[1]
                    )
                ]
            )
            + "_"
            + str(self.global_rank)
        )

    def init_redis(self):
        assert self.api_client.remote_ips is not None, (
            "Please init the api client first"
        )
        # For command fetch via redis connection
        self.redis_controller = RedisStreamHandler(
            ips=self.api_client.remote_ips, port=int(self.config.redis)
        )
        logger.debug(
            f"[{self.role}] Init redis at {self.api_client.remote_ips}:{self.redis_controller.port}"
        )

    def heartbeat_trigger(self, shutdown_signal: threading.Event):
        while True:
            self.api_client.post_heartbeat(self.replica_name)

            # If the heartbeat interval is greater than 1, we need to check the shutdown signal every second
            # for faster shutdown check
            if constant.COSMOS_HEARTBEAT_SEND_INTERVAL > 1:
                early_break = False
                for _ in range(int(constant.COSMOS_HEARTBEAT_SEND_INTERVAL)):
                    if shutdown_signal.is_set():
                        early_break = True
                        break
                    else:
                        time.sleep(1)
                if early_break:
                    break
            else:
                time.sleep(constant.COSMOS_HEARTBEAT_SEND_INTERVAL)
                if shutdown_signal.is_set():
                    break


class WorkerBase(ABC):
    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def execute(self):
        raise RuntimeError("execute method must be implemented")

    @abstractmethod
    def build_runner(self, **kwargs):
        raise RuntimeError("build_runner method must be implemented")

    @abstractmethod
    def destroy_worker(self):
        raise RuntimeError("destroy method must be implemented")
