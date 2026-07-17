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
from enum import IntEnum
from pathlib import Path

CACHE_DIR = Path(
    os.environ.get("COSMOS_CACHE_DIR", Path.home() / ".cache" / "cosmos_rl")
)

COSMOS_TCP_STORE_TIMEOUT = 10000
COSMOS_ROLLOUT_TRAJECTORY_SIZE = 30

# Heartbeat used to make sure the main thread is alive.
# Mostly, Heartbeat report is non-blocking in a separate thread,
# so we can use a shorter timeout threshold.
COSMOS_HEARTBEAT_TIMEOUT = int(os.environ.get("COSMOS_HEARTBEAT_TIMEOUT", "200"))
COSMOS_HEARTBEAT_SEND_INTERVAL = int(
    os.environ.get("COSMOS_HEARTBEAT_SEND_INTERVAL", "60")
)

# Bound for control-plane HTTP calls (unregister/heartbeat). Without a timeout,
# requests.post blocks forever on a hung/saturated controller socket during
# teardown -- which strands the clean unregister.
COSMOS_CONTROL_HTTP_TIMEOUT = float(os.environ.get("COSMOS_CONTROL_HTTP_TIMEOUT", "30"))

COSMOS_ROLLOUT_SCAN_INTERVAL = int(os.environ.get("COSMOS_ROLLOUT_SCAN_INTERVAL", "10"))
# Opt-in escalation: when truthy, the controller's replica-status monitor
# initiates a controller-wide shutdown (SIGTERM to self -> FastAPI lifespan
# shutdown) once every policy replica has been marked dead by the heartbeat
# timeout.  Default is off because cosmos-rl supports dynamic replica
# scaling (scale-to-zero, rolling restart) where ``len(policy_replicas)
# == 0`` is a legitimate transient state and a fatal shutdown would be
# wrong.  Deployments without auto-respawn -- where one trainer death
# means the whole job is lost -- can set this to ``1`` to free the
# allocation immediately instead of waiting for the wall-clock timeout.
# See ``cosmos_rl/dispatcher/run_web_panel.py::monitor_replica_status``
# for the escalation logic.
COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS = os.environ.get(
    "COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS", "0"
).lower() in ("1", "true", "yes")
COSMOS_ROLLOUT_STEP_INTERVAL = int(
    os.environ.get("COSMOS_ROLLOUT_STEP_INTERVAL", "100")
)
COSMOS_NCCL_ERROR_CLEAN_REPLICA_DELAY = int(
    os.environ.get("COSMOS_NCCL_ERROR_CLEAN_REPLICA_DELAY", "10")
)
# FIXME: (lms) Setting this greater than 1 could cause P2R NCCL hang when PP and FSDP are both enabled.
COSMOS_P2R_NCCL_GROUP_SIZE = int(os.environ.get("COSMOS_P2R_NCCL_GROUP_SIZE", "0"))
COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT = int(
    os.environ.get("COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT", "600")
)
COSMOS_ROLLOUT_CMD_WAIT_INTERVAL = float(
    os.environ.get("COSMOS_ROLLOUT_CMD_WAIT_INTERVAL", "0.001")
)
COSMOS_ROLLOUT_CMD_WAIT_TIMES = int(
    os.environ.get("COSMOS_ROLLOUT_CMD_WAIT_TIMES", "0")
)
COSMOS_ROLLOUT_REPORT_INTERVAL = int(
    os.environ.get("COSMOS_ROLLOUT_REPORT_INTERVAL", "100")
)

COSMOS_RECV_TENSOR_QUEUE_SIZE = int(
    os.environ.get("COSMOS_RECV_TENSOR_QUEUE_SIZE", "8")
)

COSMOS_GLOO_TIMEOUT = int(os.environ.get("COSMOS_GLOO_TIMEOUT", "600"))

# Internal model type for HFModel
COSMOS_HF_MODEL_TYPES = "hfmodel"


class CosmosHttpRetryConfig:
    max_retries: int = 60
    retries_per_delay: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0


COSMOS_HTTP_RETRY_CONFIG = CosmosHttpRetryConfig()
COSMOS_HTTP_LONG_WAIT_MAX_RETRY = 100
# Streaming poll reads (subscribe_command / subscribe_rollout) run inside their
# own `while not shutdown_signal` loops, so the loop itself is the retry
# mechanism. They must NOT run the deep (max_retries=60) exponential-backoff
# storm internally: when the controller finalizes and tears down its embedded
# Redis during shutdown, that storm blocks for ~50min ignoring shutdown_signal,
# which hangs the teardown join in handle_shutdown. Fail fast (one attempt) and
# let the outer loop re-check shutdown_signal and re-poll.
COSMOS_HTTP_STREAM_POLL_MAX_RETRY = 1

COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK = int(
    os.environ.get("COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK", "64")
)

COSMOS_REWARD_DISPATCHER_CONCURRENCY = int(
    os.environ.get("COSMOS_REWARD_DISPATCHER_CONCURRENCY", "2")
)

COSMOS_TEACHER_RESULT_GET_TIMEOUT = float(
    os.environ.get("COSMOS_TEACHER_RESULT_GET_TIMEOUT", "1800.0")
)
COSMOS_TEACHER_RESULT_RETRY_TIMEOUT_INTERVAL = float(
    os.environ.get("COSMOS_TEACHER_RESULT_RETRY_TIMEOUT_INTERVAL", "0.01")
)
COSMOS_TEACHER_RESULT_SET_TIMEOUT = float(
    os.environ.get("COSMOS_TEACHER_RESULT_SET_TIMEOUT", "1800.0")
)


class Algo:
    GRPO = "grpo"
    PPO = "ppo"


class RewardFn:
    DIRECT_MATH = "direct_math"
    BOXED_MATH = "boxed_math"
    SINGLE_CHOICE = "single_choice"
    GSM8K = "gsm8k"
    FORMAT = "format"
    OVERLONG = "overlong"

    @classmethod
    def from_string(cls, value: str):
        mapping = {
            "direct_math": cls.DIRECT_MATH,
            "boxed_math": cls.BOXED_MATH,
            "single_choice": cls.SINGLE_CHOICE,
            "gsm8k": cls.GSM8K,
            "format": cls.FORMAT,
            "overlong": cls.OVERLONG,
        }
        if value not in mapping:
            raise ValueError(f"Invalid value: {value}")
        return mapping[value]


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001
    # Added for Vision API
    INVALID_IMAGE = 40002
    ALREADY_EXISTS = 40003

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303
    INVALID_REQUEST = 400304

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    REQUEST_CANCELLED = 49901

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004

    SERVICE_UNAVAILABLE = 50301


class RedisStreamConstant:
    CMD_READING_TIMEOUT_MS = 10 * 1000  # 10 seconds
    CMD_FETCH_SIZE = 5
    STREAM_MAXLEN = 10000  # Keep latest n message entries
    ROLLOUT_READING_TIMEOUT_MS = 10 * 1000  # 10 seconds
    ROLLOUT_FETCH_SIZE = 8
    TEACHER_REQUEST_READING_TIMEOUT_MS = 10 * 1000  # 10 seconds
    TEACHER_REQUEST_FETCH_SIZE = 8
