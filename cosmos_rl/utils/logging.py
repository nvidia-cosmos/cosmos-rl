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

import logging
import os

logger = logging.getLogger("cosmos")

# Level selection and handler installation are separate concerns, so they must
# not share a guard.  ``hasHandlers()`` walks the ancestor chain, so a handler
# on the ROOT logger -- which any embedding application installs the moment it
# calls ``logging.basicConfig(...)`` -- used to make this whole block a no-op.
# COSMOS_RL's own entry point sets COSMOS_LOG_LEVEL before importing anything,
# so it never noticed; an embedder silently lost the variable, and the only
# symptom was missing output.  Setting the level is always safe: it decides
# which records this logger CREATES, and says nothing about where they go.
_level = getattr(logging, os.getenv("COSMOS_LOG_LEVEL", "INFO").upper(), None)
if not isinstance(_level, int):
    # A typo would otherwise resolve to some unrelated attribute of ``logging``
    # (``getattr`` happily returns a class), which setLevel rejects at a point
    # far from the cause.
    _level = logging.INFO
logger.setLevel(_level)

# Only the handler is conditional: if the host has already configured logging,
# its handlers own the routing and formatting, and records reach them by
# propagation.  Installing ours on top would bypass the host's configuration
# and double-print for anyone who deliberately routes the "cosmos" logger.
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(_level)
    formatter = logging.Formatter(
        "[cosmos] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
