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

"""Backward-compatibility shim.

The contents of this module moved to
:mod:`cosmos_rl.utils.payload_transport.nccl` as part of the
payload-transport registry refactor.  This file re-exports the public
names so existing imports continue to work; new code should import from
``cosmos_rl.utils.payload_transport.nccl`` directly.
"""

from cosmos_rl.utils.payload_transport.nccl import (
    NCCL_COMPLETION_PREFIX,
    NCCL_REDIS_NAMESPACE,
    NcclPayloadTransport,
    build_cleanup_channel,
    build_nccl_prefix,
    build_request_channel,
    build_rollout_prefix,
    build_transfer_rollout_candidates,
    parse_transfer_rollout_idx,
)

__all__ = [
    "NCCL_COMPLETION_PREFIX",
    "NCCL_REDIS_NAMESPACE",
    "NcclPayloadTransport",
    "build_cleanup_channel",
    "build_nccl_prefix",
    "build_request_channel",
    "build_rollout_prefix",
    "build_transfer_rollout_candidates",
    "parse_transfer_rollout_idx",
]
