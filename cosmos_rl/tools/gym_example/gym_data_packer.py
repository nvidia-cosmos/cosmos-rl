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

"""Gym-flavored data packer.

Subclasses :class:`TensorDataPacker` to:

* Parse the dataset's ``prompt`` field as JSON-encoded initial
  conditions (typically just ``{"seed": 42}``) before handing it to
  the rollout engine.
* Echo the trajectory dict produced by the rollout engine straight
  through to the trainer (no transport-side translation needed for
  the in-process / Redis demo).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from cosmos_rl.dispatcher.data.packer.tensor_data_packer import (
    EPISODE_LENGTH,
    OBSERVATIONS,
    TensorDataPacker,
)
from cosmos_rl.utils.logging import logger


class GymDataPacker(TensorDataPacker):
    """Tensor data packer for Gymnasium-driven rollouts."""

    def get_rollout_input(self, item: Any, **kwargs) -> Dict[str, Any]:
        """Decode the dataset ``prompt`` JSON into a dict of init conditions.

        Cosmos-RL conventionally puts ``{"prompt": "..."}`` on the
        dataset; for a non-text task we treat the prompt string as a
        small JSON document carrying e.g. the env seed and any
        environment-specific overrides.  Plain strings (or missing
        prompts) fall back to an empty dict so the rollout engine can
        use sensible defaults.
        """
        prompt = item.get("prompt") if isinstance(item, dict) else None
        if not prompt:
            return {}
        if isinstance(prompt, dict):
            return prompt
        try:
            return json.loads(prompt)
        except (TypeError, json.JSONDecodeError):
            logger.debug(
                f"[GymDataPacker] prompt is not JSON, passing through verbatim: "
                f"{prompt!r}"
            )
            return {"prompt": prompt}

    def policy_compute_max_len(self, processed_samples) -> int:
        """Same logic as the base class but defaults to the OBS-shape
        fallback first since the Gym example always emits
        ``OBSERVATIONS`` and ``EPISODE_LENGTH`` is sometimes a Python
        int rather than a tensor.  Inherits the safe-floor of 1."""
        max_len = 1
        for item in processed_samples:
            if not isinstance(item, dict):
                continue
            ep_len = item.get(EPISODE_LENGTH)
            if ep_len is None:
                obs = item.get(OBSERVATIONS)
                if hasattr(obs, "shape") and len(obs.shape) > 0:
                    ep_len = int(obs.shape[0])
            if ep_len is None:
                continue
            try:
                ep_len_int = int(ep_len.item() if hasattr(ep_len, "item") else ep_len)
            except (TypeError, ValueError):
                continue
            if ep_len_int > max_len:
                max_len = ep_len_int
        return max_len


__all__ = ["GymDataPacker"]
