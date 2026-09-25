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

"""
API client for the dispatcher.
"""

import os
import re
import requests
import msgpack
import threading
from uuid import uuid4
from functools import partial
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urljoin

from cosmos_rl.dispatcher.protocol import (
    Role,
    ValidationReportRequest,
    RolloutRequest,
    SetProfileRequest,
    SetTracePathRequest,
)
from cosmos_rl.utils.network_util import make_request_with_retry
from cosmos_rl.utils import constant
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_STATUS_SUFFIX,
    COSMOS_API_META_SUFFIX,
    COSMOS_API_REQUEST_STOP_SUFFIX,
    COSMOS_API_TRAINING_BOUNDARY_SUFFIX,
    COSMOS_API_REGISTER_SUFFIX,
    COSMOS_API_SET_PROFILE_SUFFIX,
    COSMOS_API_SET_TRACE_PATH_SUFFIX,
    COSMOS_API_UNREGISTER_SUFFIX,
    COSMOS_API_HEARTBEAT_SUFFIX,
    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX,
    COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX,
    COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX,
    COSMOS_API_NCCL_COMM_ERROR_SUFFIX,
    COSMOS_API_NCCL_COMM_STORE_CLEAR_SUFFIX,
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_ROLLOUT_SUFFIX,
    COSMOS_API_VALIDATION_REPORT_SUFFIX,
    COSMOS_API_POLICY_TRAIN_ACK_SUFFIX,
    COSMOS_API_POLICY_SHARD_INFOS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX,
    COSMOS_API_POLICY_SHARD_SEND_INSTS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX,
    COSMOS_API_GET_TRAINABLE_PARAMS_SUFFIX,
    COSMOS_API_IPC_INFO_SUFFIX,
    COSMOS_API_QUERY_IPC_INFO_SUFFIX,
    COSMOS_API_RESUME_INFO_SUFFIX,
)
from cosmos_rl.utils.parallelism_map import WeightSyncInstructionsGroup
from cosmos_rl.utils.util import list_to_b64, sanitize, b64_to_list
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.resume import ResumeMetadataMismatch


class APIClient(object):
    """
    API client for the dispatcher.
    """

    def __init__(
        self,
        role: Role,
        remote_ips: Optional[List[str]] = None,
        remote_port: Optional[int] = None,
        controller_execution_id: Optional[str] = None,
    ):
        self.role = role
        self.controller_execution_id = controller_execution_id
        self._report_session_id = uuid4().hex
        self._report_sequence = 0
        self._report_lock = threading.Lock()
        self._report_failed = False
        self._fetch_sequence = 0
        self._fetch_lock = threading.Lock()
        self._fetch_failed = False
        self._registered_global_rank = None
        self._registered_replica_name = None

        self.remote_ips = remote_ips
        self.remote_port = remote_port
        self.base_urls = []
        self.__update_base_urls()

        self.max_retries = constant.COSMOS_HTTP_RETRY_CONFIG.max_retries

    def __update_base_urls(self):
        if self.remote_ips is None or self.remote_port is None:
            # parser remote_hosts from env COSMOS_CONTROLLER_HOST
            remote_hosts = os.environ["COSMOS_CONTROLLER_HOST"]
            remote_ips, remote_port = remote_hosts.split(":")
            remote_ips = remote_ips.split(";")
            self.remote_ips = remote_ips
            self.remote_port = remote_port

        # Verify in the format of host:port
        for remote_ip in self.remote_ips:
            if not re.match(
                r"^([a-zA-Z0-9_.-]+):([1-9][0-9]{0,4})$",
                f"{remote_ip}:{self.remote_port}",
            ):
                raise ValueError(f"Invalid remote host: {remote_ip}:{self.remote_port}")

        self.base_urls = [
            f"http://{remote_ip}:{self.remote_port}{COSMOS_API_META_SUFFIX}"
            for remote_ip in self.remote_ips
        ]

    def get_alternative_urls(self, suffix: str):
        urls = []
        for base_url in self.base_urls:
            urls.append(urljoin(base_url, suffix))
        return urls

    def request_stop(self, reason: str) -> bool:
        def parse(response):
            if response.status_code != 409:
                response.raise_for_status()

        response = make_request_with_retry(
            partial(requests.post, json={"reason": reason}),
            self.get_alternative_urls(COSMOS_API_REQUEST_STOP_SUFFIX),
            response_parser=parse,
            max_retries=self.max_retries,
        )
        if response.status_code == 409:
            raise RuntimeError(f"Stop request rejected: {response.text}")
        return response.json()["accepted"]

    def training_boundary(
        self, replica_name, completed_step, *, checkpoint_complete=False
    ):
        def parse(response):
            if response.status_code != 409:
                response.raise_for_status()

        response = make_request_with_retry(
            partial(
                requests.post,
                json={
                    "replica_name": replica_name,
                    "completed_step": completed_step,
                    "checkpoint_complete": checkpoint_complete,
                },
            ),
            self.get_alternative_urls(COSMOS_API_TRAINING_BOUNDARY_SUFFIX),
            response_parser=parse,
            max_retries=self.max_retries,
        )
        if response.status_code == 409:
            raise RuntimeError(f"Training boundary rejected: {response.text}")
        return response.json()

    def get_controller_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from the controller with retry logic.
        """
        try:
            r = make_request_with_retry(
                partial(
                    requests.get,
                ),
                self.get_alternative_urls(COSMOS_API_META_SUFFIX),
                max_retries=self.max_retries,
            )
            metadata = r.json()
        except Exception as e:
            logger.error(f"Failed to communicate with controller after attempts: {e}")
            raise e

        # update base urls from controller
        remote_eth_ips = metadata.get("config", {}).get("eth_ips", [])
        if remote_eth_ips:
            remote_ips = set(self.remote_ips)
            remote_ips.update(remote_eth_ips.split(";"))
            self.remote_ips = list(remote_ips)
        self.__update_base_urls()
        return metadata

    def register(
        self,
        replica_name: str,
        role: Role,
        mesh_names: List[str],
        ranks: List[int],
        group_size: List[int],
        global_rank: int,
        host_ip: str,
        host_name: str,
        validation_reporter: Optional[bool] = None,
        rollout_reporter: Optional[bool] = None,
    ):
        if self._registered_replica_name not in (None, replica_name) or (
            self._registered_global_rank not in (None, global_rank)
        ):
            raise ValueError("API client cannot change its registered source")
        self._registered_replica_name = replica_name
        self._registered_global_rank = global_rank
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={
                        "replica_name": replica_name,
                        "role": role,
                        "mesh_names": mesh_names,
                        "ranks": ranks,
                        "group_size": group_size,
                        "global_rank": global_rank,
                        "host_ip": host_ip,
                        "host_name": host_name,
                        "validation_reporter": validation_reporter,
                        "rollout_reporter": rollout_reporter,
                        "report_session_id": self._report_session_id
                        if role in (Role.ROLLOUT, Role.POLICY)
                        else None,
                    },
                ),
                self.get_alternative_urls(COSMOS_API_REGISTER_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to register to controller: {e}")
            raise e

    def delegate_rollout_reporting(self) -> dict:
        """Transfer the unused fetch/report streams to a backend wrapper."""
        with self._fetch_lock, self._report_lock:
            if (
                self.role != Role.ROLLOUT
                or self._registered_replica_name is None
                or self._registered_global_rank is None
                or self._report_sequence != 0
                or self._report_failed
                or self._fetch_sequence != 0
                or self._fetch_failed
            ):
                raise ValueError(
                    "Only an unused registered rollout source can delegate"
                )
            self._report_failed = True
            self._fetch_failed = True
            return {
                "replica_name": self._registered_replica_name,
                "global_rank": self._registered_global_rank,
                "report_session_id": self._report_session_id,
                "controller_execution_id": self.controller_execution_id,
            }

    def adopt_rollout_reporting(self, source: dict) -> None:
        """Bind once to the executor's source, without registering another atom."""
        with self._fetch_lock, self._report_lock:
            if (
                self.role != Role.ROLLOUT
                or self._registered_replica_name is not None
                or self._report_sequence != 0
                or self._report_failed
                or self._fetch_sequence != 0
                or self._fetch_failed
                or not isinstance(source, dict)
                or not isinstance(source.get("replica_name"), str)
                or not source["replica_name"]
                or type(source.get("global_rank")) is not int
                or source["global_rank"] < 0
                or not isinstance(source.get("report_session_id"), str)
                or not source["report_session_id"]
                or source.get("controller_execution_id") != self.controller_execution_id
            ):
                raise ValueError("Invalid or already bound delegated rollout source")
            self._registered_replica_name = source["replica_name"]
            self._registered_global_rank = source["global_rank"]
            self._report_session_id = source["report_session_id"]

    def unregister(self, replica_name: str):
        # ``unregister`` is called on the shutdown path (handle_shutdown).
        # ``requests.post`` with no ``timeout`` blocks forever when the
        # controller is wedged (e.g. event loop frozen by another bug),
        # which then deadlocks worker teardown -- the worker process
        # never reaches ``destroy_distributed()`` and its UCXX server
        # threads keep polling until the orchestrator hard-kills the
        # job. Make one bounded attempt, without the operational request
        # retry/backoff chain: the controller may already have exited.
        # This is best-effort cleanup, not a correctness requirement.
        try:
            requests.post(
                self.get_alternative_urls(COSMOS_API_UNREGISTER_SUFFIX)[0],
                json={"replica_name": replica_name},
                timeout=constant.COSMOS_CONTROL_HTTP_TIMEOUT,
            )
        except Exception as e:
            logger.error(f"Failed to unregister from controller: {e}")

    def post_heartbeat(self, replica_name: str):
        # Per-attempt timeout matters here too: the heartbeat daemon
        # blocks shutdown_signal polling while inside ``requests.post``,
        # so an unresponsive controller would keep the heartbeat
        # process alive (and ``heartbeat_thread.join()`` hung) for the
        # full configurable retry chain.  10s is generous relative to a
        # healthy controller round-trip while still ensuring the daemon
        # checks shutdown_signal at most every ~10s.
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={"replica_name": replica_name},
                    # Bounded so a stuck heartbeat post cannot block the heartbeat
                    # process indefinitely (which would also wedge its join()).
                    timeout=constant.COSMOS_CONTROL_HTTP_TIMEOUT,
                ),
                self.get_alternative_urls(COSMOS_API_HEARTBEAT_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to send heartbeat to controller: {e}")

    def get_status(self) -> Dict[str, Any]:
        try:
            r = make_request_with_retry(
                partial(
                    requests.get,
                ),
                self.get_alternative_urls(COSMOS_API_STATUS_SUFFIX),
                max_retries=self.max_retries,
            )
            return r.json()
        except Exception as e:
            logger.error(f"Failed to get status from controller: {e}")
            raise e

    def post_profile(self, profile: SetProfileRequest):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json=profile.model_dump(),
                ),
                self.get_alternative_urls(COSMOS_API_SET_PROFILE_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to set profile to controller: {e}")
            raise e

    def post_trace_path(self, trace_path: SetTracePathRequest):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json=trace_path.model_dump(),
                ),
                self.get_alternative_urls(COSMOS_API_SET_TRACE_PATH_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to set trace path to controller: {e}")
            raise e

    def post_ipc_info(self, mesh_key: str, ipc_addr: str):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={"mesh_key": mesh_key, "ipc_addr": ipc_addr},
                ),
                self.get_alternative_urls(COSMOS_API_IPC_INFO_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(
                f"Failed to post ipc info for mesh key {mesh_key} to controller: {e}"
            )
            raise e

    def query_ipc_info(self, mesh_key: str) -> str:
        try:
            r = make_request_with_retry(
                partial(
                    requests.post,
                    json={"mesh_key": mesh_key},
                ),
                self.get_alternative_urls(COSMOS_API_QUERY_IPC_INFO_SUFFIX),
                max_retries=self.max_retries,
            )
            ipc_addr = r.json()["ipc_addr"]
            return ipc_addr
        except Exception as e:
            raise RuntimeError(
                f"[{self.role}] Failed in get ipc_addr for mesh key {mesh_key} from controller after retries {e}."
            )

    def post_nccl_comm_initiator(self, unique_pair_name: str, nccl_uuid: List[int]):
        base64_nccl_group_id = list_to_b64(nccl_uuid)
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={
                        "unique_pair_name": unique_pair_name,
                        "handle_base64": base64_nccl_group_id,
                    },
                ),
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[{self.role}] Failed in post nccl group_id to controller after retries {e}."
            )

    def post_nccl_comm_acceptor(self, unique_pair_name: str) -> List[int]:
        try:
            r = make_request_with_retry(
                partial(
                    requests.post,
                    json={"unique_pair_name": unique_pair_name},
                ),
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX),
                max_retries=self.max_retries,
            )
            base64_nccl_group_id = r.json()["handle_base64"]
            return b64_to_list(base64_nccl_group_id)
        except Exception as e:
            raise RuntimeError(
                f"[{self.role}] Failed in get nccl group_id from controller after retries {e}."
            )

    def post_nccl_comm_error(self, replica_name: str, error: Exception):
        # Failure reporting must not delay the worker's terminal/rebuild path.
        # One bounded best-effort request; a lost response is not a reason to
        # replay a failure notification through the operational retry schedule.
        try:
            response = requests.post(
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_ERROR_SUFFIX)[0],
                json={"replica_name": replica_name, "error": str(error)},
                timeout=constant.COSMOS_CONTROL_HTTP_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning("[%s] Could not report NCCL failure: %s", self.role, e)

    def post_clear_nccl_comm_store(self, unique_pair_name: str):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={"unique_pair_name": unique_pair_name},
                ),
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_STORE_CLEAR_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[{self.role}] Failed in clear nccl comm store from controller after retries {e}."
            )

    def list_nccl_comm_infos(self) -> List[Dict[str, Any]]:
        """
        List all the NCCL communicators stored in the controller.
        """
        try:
            r = make_request_with_retry(
                partial(
                    requests.get,
                ),
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX),
                max_retries=self.max_retries,
            )
            comm_info = r.json()["comm_info"]
            comm_dict = {}
            for key, value in comm_info.items():
                comm_dict[key] = str(b64_to_list(value))
            return comm_dict
        except Exception as e:
            raise RuntimeError(
                f"[{self.role}] Failed in list nccl comm from controller after retries {e}."
            )

    def post_policy_shard_info(
        self,
        shard_infos: List[Dict[str, Any]],
        param_groups: List[List[str]],
        sorted_params: List[List[str]],
        trainable_params: List[str],
    ):
        data = msgpack.packb(
            {
                "shard_infos": shard_infos,
                "param_groups": param_groups,
                "sorted_params": sorted_params,
                "trainable_params": trainable_params,
            }
        )
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    data=data,
                    headers={"Content-Type": "application/msgpack"},
                ),
                self.get_alternative_urls(COSMOS_API_POLICY_SHARD_INFOS_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Policy] Failed to post policy shard infos to controller after retries {e}."
            )

    def post_rollout_shard_info(
        self,
        shard_infos: List[Dict[str, Any]],
        param_groups: List[List[str]],
        sorted_params: List[List[str]],
    ):
        data = msgpack.packb(
            {
                "shard_infos": shard_infos,
                "param_groups": param_groups,
                "sorted_params": sorted_params,
            }
        )
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    data=data,
                    headers={"Content-Type": "application/msgpack"},
                ),
                self.get_alternative_urls(COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Rollout] Failed in post shard infos to controller after retries {e}."
            )

    def post_policy_shard_send_insts(
        self, rank: int
    ) -> List[WeightSyncInstructionsGroup]:
        try:
            insts_meta = make_request_with_retry(
                partial(
                    requests.post,
                    json={"rank": rank},
                ),
                self.get_alternative_urls(COSMOS_API_POLICY_SHARD_SEND_INSTS_SUFFIX),
                max_retries=self.max_retries,
            )
            insts = msgpack.unpackb(insts_meta.content, strict_map_key=False)
            return [WeightSyncInstructionsGroup.from_dict(inst) for inst in insts]
        except Exception as e:
            raise RuntimeError(
                f"[Policy] Failed in post policy shard send insts to controller after retries {e}."
            )

    def post_rollout_shard_recv_insts(
        self, rank: int
    ) -> List[WeightSyncInstructionsGroup]:
        try:
            insts_meta = make_request_with_retry(
                partial(
                    requests.post,
                    json={"rank": rank},
                ),
                self.get_alternative_urls(COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX),
                max_retries=self.max_retries,
            )
            insts = msgpack.unpackb(insts_meta.content, strict_map_key=False)
            return [WeightSyncInstructionsGroup.from_dict(inst) for inst in insts]
        except Exception as e:
            raise RuntimeError(
                f"[Rollout] Failed in fetching rollout shard recv insts from controller after retries {e}."
            )

    def post_policy_train_ack(
        self,
        replica_name: str,
        weight_step: int,
        total_steps: int,
        profile_finished: bool,
        report_data: Dict[str, Any],
    ):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={
                        "replica_name": replica_name,
                        "weight_step": weight_step,
                        "total_steps": total_steps,
                        "profile_finished": profile_finished,
                        "report_data": sanitize(report_data),
                        "report_session_id": self._report_session_id,
                        "src_global_rank": self._registered_global_rank,
                    },
                ),
                self.get_alternative_urls(COSMOS_API_POLICY_TRAIN_ACK_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Policy] Failed in in send train ack to controller after retries {e}."
            )

    def post_resume_info(self, resume_info: Dict[str, Any]):
        """
        Post the resumed extra info to the controller.
        Args:
            resume_info: The resumed extra info to post.
        """

        def check_response(response):
            # Return a permanent conflict through the retry helper without
            # raising inside it; transport failures and other statuses retain
            # the existing retry policy.
            if response.status_code != 409:
                response.raise_for_status()

        try:
            response = make_request_with_retry(
                partial(
                    requests.post,
                    json={"ckpt_extra_info": resume_info},
                ),
                self.get_alternative_urls(COSMOS_API_RESUME_INFO_SUFFIX),
                max_retries=self.max_retries,
                response_parser=check_response,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Policy] Failed in post resume info to controller after retries {e}."
            )
        if response.status_code == 409:
            raise ResumeMetadataMismatch(
                "Controller rejected checkpoint resume agreement; training must not continue."
            )

    def get_trainable_params(self) -> List[str]:
        try:
            r = make_request_with_retry(
                partial(
                    requests.get,
                ),
                self.get_alternative_urls(COSMOS_API_GET_TRAINABLE_PARAMS_SUFFIX),
                max_retries=self.max_retries,
            )
            return r.json()["trainable_params"]
        except Exception as e:
            raise RuntimeError(
                f"[Rollout] Failed in fetching trainable params from controller after retries {e}."
            )

    def post_validation_report(self, report: ValidationReportRequest):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json=report.model_dump(),
                    timeout=constant.COSMOS_CONTROL_HTTP_TIMEOUT,
                ),
                self.get_alternative_urls(COSMOS_API_VALIDATION_REPORT_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Rollout] Failed in sending validation report to controller after retries {e}."
            ) from e

    def get_next_prompt(
        self,
        batch_size: int,
        validation_step: Optional[int] = None,
        rank_in_mesh: Optional[int] = None,
        *,
        validation_round_id: Optional[str] = None,
        src_replica_name: Optional[str] = None,
        fetch_sequence: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if self.role == Role.ROLLOUT and validation_step is None:
            return self._get_training_prompt(batch_size, rank_in_mesh)
        try:
            params = {
                "n": batch_size,
                "validation_step": validation_step,
                "rank_in_mesh": rank_in_mesh,
            }
            if validation_round_id is not None:
                params.update(
                    validation_round_id=validation_round_id,
                    src_replica_name=src_replica_name,
                    fetch_sequence=fetch_sequence,
                )
            request_options = (
                {"timeout": constant.COSMOS_CONTROL_HTTP_TIMEOUT}
                if validation_round_id is not None
                else {}
            )
            r = make_request_with_retry(
                partial(
                    requests.get,
                    params=params,
                    **request_options,
                ),
                self.get_alternative_urls(COSMOS_API_NEXT_PROMPT_SUFFIX),
                max_retries=self.max_retries,
            )
            r = r.json()
            payloads = r["payloads_list"]
            is_end = r["is_end"]
            return payloads, is_end
        except Exception as e:
            if validation_round_id is not None:
                raise RuntimeError("Validation fetch failed after retries") from e
            logger.error(
                f"[Rollout] Failed in fetching next prompt from controller after retries {e}."
            )
            return [], False

    @property
    def training_fetch_failed(self) -> bool:
        return self._fetch_failed

    def _get_training_prompt(self, batch_size, rank_in_mesh):
        with self._fetch_lock:
            if self._fetch_failed:
                raise RuntimeError("Training fetch failed; source cannot continue")
            if self._registered_replica_name is None:
                self._fetch_failed = True
                raise ValueError("Training fetch requires its registered source")
            params = {
                "n": batch_size,
                "rank_in_mesh": rank_in_mesh,
                "src_replica_name": self._registered_replica_name,
                "src_global_rank": self._registered_global_rank,
                "fetch_session_id": self._report_session_id,
                "fetch_sequence": self._fetch_sequence,
                "controller_execution_id": self.controller_execution_id,
            }
            try:
                response = make_request_with_retry(
                    partial(
                        requests.get,
                        params=params,
                        timeout=constant.COSMOS_CONTROL_HTTP_TIMEOUT,
                    ),
                    self.get_alternative_urls(COSMOS_API_NEXT_PROMPT_SUFFIX),
                    max_retries=min(self.max_retries, 3),
                    initial_delay=0.1,
                    max_delay=0.5,
                ).json()
                payloads, is_end = response["payloads_list"], response["is_end"]
                if not isinstance(payloads, list) or type(is_end) is not bool:
                    raise ValueError("Invalid training fetch response")
            except Exception as error:
                self._fetch_failed = True
                raise RuntimeError(
                    "Training fetch failed after bounded HTTP attempts"
                ) from error
            self._fetch_sequence += 1
            return payloads, is_end

    def post_rollout_completion(self, response: RolloutRequest) -> bool:
        # A producer may report from more than one thread, but its mutation
        # stream must stay ordered across retries and normal/terminal reports.
        with self._report_lock:
            if self._report_failed:
                raise RuntimeError("Rollout reporting failed; source cannot continue")
            if response.src_replica_name != self._registered_replica_name:
                raise ValueError("Rollout report requires its registered source")
            if response.src_global_rank not in (None, self._registered_global_rank):
                raise ValueError(
                    "Rollout report rank differs from its registered source"
                )
            if response.report_session_id is None:
                response.report_session_id = self._report_session_id
                response.report_sequence = self._report_sequence
            if response.report_session_id != self._report_session_id:
                raise ValueError("Rollout report belongs to another source incarnation")
            if response.report_sequence is None:
                raise ValueError("Rollout report is missing its sequence")
            result = self._post_rollout_receipt(response)
            if result:
                self._report_sequence = max(
                    self._report_sequence, response.report_sequence + 1
                )
            else:
                self._report_failed = True
            return result

    def _post_rollout_receipt(self, response: RolloutRequest) -> bool:
        payload = response.model_dump()
        payload["controller_execution_id"] = self.controller_execution_id
        payload["src_global_rank"] = self._registered_global_rank

        def check_response(result):
            # An old attempt cannot be retried into the current execution.
            if result.status_code != 410:
                result.raise_for_status()

        try:
            result = make_request_with_retry(
                partial(
                    requests.post,
                    json=payload,
                    timeout=constant.COSMOS_CONTROL_HTTP_TIMEOUT,
                ),
                self.get_alternative_urls(COSMOS_API_ROLLOUT_SUFFIX),
                max_retries=min(self.max_retries, 3),
                initial_delay=0.1,
                max_delay=0.5,
                response_parser=check_response,
            )
            return result.status_code != 410
        except Exception as e:
            self._report_failed = True
            raise RuntimeError(
                "Rollout report failed after bounded HTTP attempts"
            ) from e
