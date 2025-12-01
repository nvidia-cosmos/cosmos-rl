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
)
from cosmos_rl.utils.parallelism_map import WeightSyncInstructionsGroup
from cosmos_rl.utils.util import list_to_b64, sanitize, b64_to_list
from cosmos_rl.utils.logging import logger


class APIClient(object):
    """
    API client for the dispatcher.
    """

    def __init__(
        self,
        role: Role,
        remote_ips: Optional[List[str]] = None,
        remote_port: Optional[int] = None,
    ):
        self.role = role

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
    ):
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
                    },
                ),
                self.get_alternative_urls(COSMOS_API_REGISTER_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to register to controller: {e}")
            raise e

    def unregister(self, replica_name: str):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={"replica_name": replica_name},
                ),
                self.get_alternative_urls(COSMOS_API_UNREGISTER_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to unregister from controller: {e}")

    def post_heartbeat(self, replica_name: str):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={"replica_name": replica_name},
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
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json={"replica_name": replica_name, "error": str(error)},
                ),
                self.get_alternative_urls(COSMOS_API_NCCL_COMM_ERROR_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[{self.role}] Failed in post nccl comm error to controller after retries {e}."
            )

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
                    },
                ),
                self.get_alternative_urls(COSMOS_API_POLICY_TRAIN_ACK_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Policy] Failed in in send train ack to controller after retries {e}."
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
                ),
                self.get_alternative_urls(COSMOS_API_VALIDATION_REPORT_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Rollout] Failed in sending validation report to controller after retries {e}."
            )

    def get_next_prompt(
        self, batch_size: int, validation_step: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], bool]:
        try:
            r = make_request_with_retry(
                partial(
                    requests.get,
                    params={"n": batch_size, "validation_step": validation_step},
                ),
                self.get_alternative_urls(COSMOS_API_NEXT_PROMPT_SUFFIX),
                max_retries=self.max_retries,
            )
            r = r.json()
            payloads = r["payloads_list"]
            is_end = r["is_end"]
            return payloads, is_end
        except Exception as e:
            logger.error(
                f"[Rollout] Failed in fetching next prompt from controller after retries {e}."
            )
            return [], False

    def post_rollout_completion(self, response: RolloutRequest):
        try:
            make_request_with_retry(
                partial(
                    requests.post,
                    json=response.model_dump(),
                ),
                self.get_alternative_urls(COSMOS_API_ROLLOUT_SUFFIX),
                max_retries=self.max_retries,
            )
        except Exception as e:
            logger.error(
                f"[Rollout] Failed in sending rollout completion to controller after retries {e}."
            )
