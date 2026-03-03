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
WFM Policy Rollout: plugs :class:`CosmosPolicyVLA` into the VLA rollout
engine.

Subclasses :class:`RolloutBase` and re-uses ``EnvManager`` +
``LiberoEnvWrapper`` from the VLA rollout infrastructure.  The rollout loop
mirrors ``OpenVLARollout._do_rollout`` while using the
``CosmosPolicyVLA.process_input`` / ``generate_action`` interface that is
identical to OpenVLA and PI05.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cosmos_rl.policy.config import Config
from cosmos_rl.policy.model.wfm.cosmos_policy import (
    CosmosPolicyVLA,
    CosmosPolicyVLAConfig,
)
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.simulators.env_manager import EnvManager
from cosmos_rl.utils.logging import logger


def _eval_cfg_from_custom(custom: dict) -> CosmosPolicyVLAConfig:
    """Build a :class:`CosmosPolicyVLAConfig` from ``config.custom``."""
    valid = {f.name for f in CosmosPolicyVLAConfig.__dataclass_fields__.values()}
    return CosmosPolicyVLAConfig(**{k: v for k, v in custom.items() if k in valid})


@RolloutRegistry.register(rollout_type="wfm-policy")
class WFMPolicyRollout(RolloutBase):
    """Rollout backend for :class:`CosmosPolicyVLA` on LIBERO.

    Re-uses ``EnvManager`` + ``LiberoEnvWrapper`` from the existing VLA
    rollout infrastructure.  Only ``init_engine`` and ``post_init_hook`` are
    customised – the rollout loop and env management are standard.
    """

    def __init__(self, config: Config, parallel_dims, device, **kwargs):
        custom = getattr(config, "custom", {}) or {}
        self.vla_cfg = _eval_cfg_from_custom(custom)
        self.task_suite_name = custom.get("task_suite_name", "libero_10")
        self.max_steps = custom.get("max_steps", 300)
        self._num_envs = custom.get("num_envs", 1)
        self._save_video = custom.get("save_video", True)
        super().__init__(config, parallel_dims, device, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def post_init_hook(self, **kwargs):
        self._model_param_map = None
        sim_cfg = SimpleNamespace(
            num_envs=self._num_envs,
            task_suite_name=self.task_suite_name,
            max_steps=self.max_steps,
        )
        from cosmos_rl.simulators.libero.env_wrapper import LiberoEnvWrapper

        self.env_manager = EnvManager(
            cfg=sim_cfg,
            rank=int(os.environ.get("LOCAL_RANK", 0)),
            env_cls=LiberoEnvWrapper,
            use_subprocess=True,
        )
        self.env_manager.start_simulator()
        self.num_envs = self._num_envs
        self.obs_keys = ["full_images", "wrist_images", "states", "proprio_9d"]

    def init_engine(self, quantization=None, seed=42, load_format="dummy", **kwargs):
        if self._engine_initialized:
            return
        logger.info("[WFMPolicyRollout] Building CosmosPolicyVLA ...")
        self.model = CosmosPolicyVLA.from_config(self.vla_cfg)
        self.model.eval()
        self._engine_initialized = True
        logger.info("[WFMPolicyRollout] Engine initialised.")

    def get_underlying_model(self) -> torch.nn.Module:
        return getattr(self, "model", None)

    def set_underlying_model(self, model: torch.nn.Module):
        self.model = model

    # ------------------------------------------------------------------
    # Standalone evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        task_ids: Optional[List[int]] = None,
        trials_per_task: int = 1,
    ) -> Dict[str, Any]:
        """Run evaluation without the full training framework.

        Args:
            task_ids: Task indices to evaluate (``None`` → all tasks).
            trials_per_task: Number of trials per task.

        Returns:
            ``{"success_rate": float, "per_task": [...]}``.
        """
        if not self._engine_initialized:
            self.init_engine()

        from cosmos_rl.simulators.libero.utils import get_benchmark_overridden

        suite = get_benchmark_overridden(self.task_suite_name)()
        if task_ids is None:
            task_ids = list(range(suite.n_tasks))

        jobs = [(tid, trial) for tid in task_ids for trial in range(trials_per_task)]
        results: List[Dict[str, Any]] = []
        done = 0

        while done < len(jobs):
            batch = min(self.num_envs, len(jobs) - done)
            batch_jobs = jobs[done : done + batch]
            env_ids = list(range(batch))
            b_tids = [j[0] for j in batch_jobs]
            b_trials = [j[1] for j in batch_jobs]

            images_and_states, task_descriptions = self.env_manager.reset(
                env_ids, b_tids, b_trials, [True] * batch,
            )
            sim = {**images_and_states, "task_descriptions": task_descriptions}

            active = np.ones(batch, dtype=bool)
            completes = np.zeros(batch, dtype=bool)

            while active.any():
                ids = [i for i in range(batch) if active[i]]
                active_sim: Dict[str, Any] = {
                    "task_descriptions": [sim["task_descriptions"][i] for i in ids],
                }
                for k in self.obs_keys:
                    active_sim[k] = sim[k][ids] if sim[k] is not None else None

                data_batch = self.model.process_input(active_sim, "")
                out = self.model.generate_action(data_batch, is_valid=True)
                actions = out["action"]
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()
                actions[..., -1] = np.sign(actions[..., -1])
                for t in range(actions.shape[1]):
                    step_res = self.env_manager.step(ids, actions[:, t])

                for k in self.obs_keys:
                    if step_res.get(k) is not None:
                        for j, eid in enumerate(ids):
                            sim[k][eid] = step_res[k][j]

                for j, eid in enumerate(ids):
                    if not step_res["active"][j]:
                        active[eid] = False
                        completes[eid] = step_res["complete"][j]

            if self._save_video:
                out_dir = os.path.join(
                    getattr(self.config, "train", SimpleNamespace(output_dir="./outputs")).output_dir,
                    "wfm_rollouts",
                )
                self.env_manager.save_validation_videos(out_dir, env_ids)

            for j, (tid, trial) in enumerate(batch_jobs):
                ok = bool(completes[j])
                results.append({"task_id": tid, "trial_id": trial, "complete": ok})
                logger.info(f"  Task {tid} trial {trial}: {'SUCCESS' if ok else 'FAIL'}")

            done += batch

        n_ok = sum(r["complete"] for r in results)
        sr = n_ok / len(results) * 100 if results else 0.0
        logger.info(f"Overall success rate: {sr:.1f}% ({n_ok}/{len(results)})")
        return {"success_rate": sr, "n_success": n_ok, "n_total": len(results), "per_task": results}

    # ------------------------------------------------------------------
    # Framework-compatible interface
    # ------------------------------------------------------------------

    def rollout_generation(self, payloads, stream, data_packer, data_fetcher, is_validation=False, **kwargs):
        result = self.evaluate()
        completions = [
            {"complete": r["complete"], "finish_step": -1, "trajectory_id": ""}
            for r in result["per_task"]
        ]
        return [RolloutResult(completions=completions)]
