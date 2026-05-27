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

"""Toy policy-gradient trainer for the Gymnasium Classic Control demo.

First upstream consumer of :class:`TrajectoryExpansionMixin`.
Composes the mixin in **rollout mode** (``chunk_size = None``,
override :meth:`_train_one_rollout`) because gym tensor trajectories
are small enough that per-rollout iteration is the right granularity.

This trainer is deliberately **toy**:

* Plain :class:`torch.optim.Adam` over the policy parameters; no
  optimizer-container plumbing, no FSDP, no gradient accumulation
  beyond the per-rollout sum the mixin's outer loop already provides.
* No real LR scheduler.
* No real persistence (``export_safetensors`` /
  ``model_load_from_hf`` / ``model_resume_from_checkpoint`` are
  no-ops with a warning).
* No real validation step.
* Loss is :func:`compute_simple_pg_loss` — MSE between the policy's
  raw output and the (one-hot) sampled action, weighted by the
  discounted return at each step. **Not PPO, not A2C, not REINFORCE
  with a baseline.** It exists to validate the contract end-to-end,
  not to be a competitive RL implementation.

What this trainer **does** validate:

* :class:`TrajectoryExpansionMixin` composes cleanly with
  :class:`Trainer` (mixin leftmost, base trainer rightmost).
* The packer-protocol assertion fires on a misconfigured packer.
* ``_begin_training_step`` -> N x ``_train_one_rollout`` ->
  ``_finalize_training_step`` ordering with real gradients applied.
* The mixin's outer loop is correctly delegated through
  ``packer.iter_rollouts(rollouts)`` (default body yields a single
  batch of all rollouts, so the trainer sees them in input order).

Launcher integration (colocated single-replica mode): the trainer
exposes the surface that :class:`~cosmos_rl.policy.worker.rl_worker`
introspects (``model.trainable_params``, ``model.weight_sync_transforms``,
``model.weight_mapper``, ``trainer.weight_mapper``,
``trainer.map_w_from_policy_to_rollout``, ``trainer.weight_resume``,
``trainer.update_lr_schedulers``, ``trainer.sync_all_states``).  These
are the **minimum** surface needed by the rl_worker; checkpoint
resume, LR decay and inter-replica state sync are no-ops appropriate
for a single-replica colocated toy.  Disaggregated / multi-replica
launches are explicitly out of scope (no NCCL P2R weight transport
plumbing) — see ``tools/gym_example/README.md`` for the supported
launch matrix.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import IdentityWeightMapper
from cosmos_rl.policy.trainer.base import Trainer, TrainerRegistry
from cosmos_rl.policy.trainer.trajectory_mixin import TrajectoryExpansionMixin
from cosmos_rl.tools.gym_example.gym_algo import (
    compute_returns,
    compute_simple_pg_loss,
)
from cosmos_rl.tools.gym_example.gym_data_packer import (
    _episode_length,
    _trajectory_from_rollout,
)
from cosmos_rl.tools.gym_example.gym_policy import GymMLPConfig, GymPolicy
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims


_DEFAULT_GAMMA = 0.99
_DEFAULT_LR = 3e-4
_DEFAULT_GRAD_NORM_CLIP = 1.0


def _resolve_device(preferred: Optional[torch.device] = None) -> torch.device:
    """Pick a sensible default device for the toy trainer.

    Preference order: explicit override > CUDA if available > CPU.
    The base :class:`Trainer.__init__` defaults to ``cuda:LOCAL_RANK``
    even when CUDA isn't available, which trips up CPU-only test runs;
    we override that here.
    """
    if preferred is not None:
        return preferred
    if torch.cuda.is_available():
        return torch.device("cuda", int(torch.cuda.current_device()))
    return torch.device("cpu")


class _NoOpLRScheduler:
    """Object-shaped placeholder for an LR scheduler.

    Has the two methods :func:`update_lr_schedulers` and the trainer's
    own loop touch — :meth:`step` and :meth:`state_dict`. Both are
    no-ops; the toy trainer doesn't decay the LR.
    """

    def step(self) -> None:
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return None


@TrainerRegistry.register(trainer_type="gym_pg")
class GymTrainer(TrajectoryExpansionMixin, Trainer):
    """Toy policy-gradient trainer composing :class:`TrajectoryExpansionMixin`.

    The mixin (leftmost in the MRO) provides ``step_training``; this
    class fills in the per-rollout work and the framework-required
    abstract methods. ``chunk_size = None`` selects rollout-mode
    iteration: the mixin walks one rollout at a time and dispatches to
    :meth:`_train_one_rollout`.

    Args:
        config: The cosmos-rl :class:`Config`.
        parallel_dims: Parallelism descriptor (unused by this toy
            trainer; the model is single-rank, no sharding).
        **kwargs: Forwarded to :class:`Trainer`. Notable keys:
            * ``data_packer`` (required, must satisfy
              :class:`TrajectoryPacker`).
            * ``device`` (optional override; defaults to CUDA if
              available else CPU).
            * ``policy`` (optional pre-built :class:`GymPolicy` for
              tests; if omitted, the trainer constructs one from the
              ``[model]`` section of the policy TOML referenced by
              ``config.policy.model_name_or_path``).
    """

    chunk_size = None  # rollout-mode iteration

    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        **kwargs: Any,
    ) -> None:
        # The base Trainer hard-codes self.device = torch.device(f"cuda:{LOCAL_RANK}")
        # which fails on CPU-only systems. Override before calling super().
        device_override = kwargs.pop("device", None)
        super().__init__(config=config, parallel_dims=parallel_dims, **kwargs)
        self.device = _resolve_device(device_override)

        # Toy hyperparameters live in [custom] so they don't pollute the
        # main config schema. Defaults are conservative.
        custom = getattr(config, "custom", None) or {}
        if not isinstance(custom, dict):
            # Pydantic model in some configs; fall back to ``model_dump``.
            try:
                custom = custom.model_dump()
            except Exception:
                custom = {}
        self.gamma = float(custom.get("gamma", _DEFAULT_GAMMA))
        self.max_grad_norm = float(custom.get("max_grad_norm", _DEFAULT_GRAD_NORM_CLIP))

        self.model = self._build_policy(config, kwargs.get("policy"))
        self.model.to(self.device)

        # Tier-B launcher surface (single-replica colocated mode).
        # The rl_worker introspects ``self.trainer.model.{trainable_params,
        # weight_sync_transforms, weight_mapper}`` plus ``self.trainer.{
        # weight_mapper, map_w_from_policy_to_rollout}`` unconditionally
        # during ``prepare_shard_infos_for_weight_sync_insts``; we expose
        # them here without making :class:`GymPolicy` inherit from
        # :class:`BaseModel` (8 abstract methods, none meaningful for a
        # toy MLP).  ``map_w_from_policy_to_rollout`` is the empty dict
        # because in colocated mode the rollout backend shares the same
        # nn.Module reference (see ``rollout/worker/colocated/rollout_control.py``
        # ``set_underlying_model(api_client.get_policy_model())``); no
        # tensor needs to be transferred between policy and rollout.
        self.weight_mapper = IdentityWeightMapper(None)
        self.map_w_from_policy_to_rollout: Dict[
            str, Union[torch.Tensor, Callable[[], torch.Tensor]]
        ] = {}
        self._attach_weight_sync_surface(self.model, self.weight_mapper)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[_NoOpLRScheduler] = None

        # Per-step scratch state populated by _begin_training_step and
        # consumed by _finalize_training_step. Set to None outside of a
        # training step so misordered hook calls fail loudly.
        self._step_losses: Optional[List[float]] = None
        self._step_returns: Optional[List[float]] = None
        self._step_episode_lengths: Optional[List[int]] = None
        self._step_start_time: Optional[float] = None
        self._step_num_rollouts: int = 0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_weight_sync_surface(
        model: torch.nn.Module, weight_mapper: IdentityWeightMapper
    ) -> None:
        """Monkey-patch the cosmos-rl P2R-sync surface onto a plain ``nn.Module``.

        ``rl_worker.prepare_shard_infos_for_weight_sync_insts`` reads
        three attributes on ``trainer.model`` unconditionally:

        * ``trainable_params: List[str]`` — HF-canonical parameter names
          for parameters that are being trained.
        * ``weight_sync_transforms: List[Tuple[str, Tensor | Callable]]``
          — ``(name, value)`` pairs the worker uses to assemble the
          P2R sync plan.  For a single-rank, non-sharded model the
          value is just the parameter tensor itself.
        * ``weight_mapper: WeightMapper`` — the mapper used to canonicalize
          parameter names; :class:`IdentityWeightMapper` is the right
          choice for the toy MLP (no name remapping, no FSDP splitting).

        :class:`BaseModel` provides these as ``@property`` /
        ``@cached_property``, but inheriting from :class:`BaseModel`
        forces eight unrelated abstract methods on us
        (``parallelize_fn``, ``apply_pipeline_split``, ``load_hf_weights``,
        etc.).  For a toy demo, in-place attribute assignment is the
        more honest minimal-delta approach.
        """
        model.weight_mapper = weight_mapper
        named = list(model.named_parameters())
        trainable_names: List[str] = []
        sync_transforms: List[
            Tuple[str, Union[torch.Tensor, Callable[[], torch.Tensor]]]
        ] = []
        for name, param in named:
            hf_name = weight_mapper.policy_map_local_key_to_hf_key(name)
            if param.requires_grad:
                trainable_names.append(hf_name)
            sync_transforms.append((hf_name, param))
        model.trainable_params = trainable_names
        model.weight_sync_transforms = sync_transforms

    @staticmethod
    def _build_policy(config: CosmosConfig, prebuilt: Optional[GymPolicy]) -> GymPolicy:
        """Resolve a :class:`GymPolicy` from config or accept a prebuilt one."""
        if prebuilt is not None:
            return prebuilt
        # Resolve from [policy].model_name_or_path which (for the gym
        # demo) points at a TOML carrying [model] obs_dim/action_dim/...
        path = getattr(getattr(config, "policy", None), "model_name_or_path", None)
        if path:
            import toml

            try:
                data = toml.load(path)
                section = data.get("model", data)
                cfg = GymMLPConfig(
                    obs_dim=int(section.get("obs_dim", 4)),
                    action_dim=int(section.get("action_dim", 2)),
                    hidden_dim=int(section.get("hidden_dim", 64)),
                    discrete=bool(section.get("discrete", True)),
                )
                return GymPolicy(cfg)
            except Exception as e:  # pragma: no cover - exercised in launch path
                logger.warning(
                    f"[GymTrainer] Failed to load policy config from {path!r}: {e}; "
                    "falling back to default GymMLPConfig (CartPole shape)."
                )
        return GymPolicy(GymMLPConfig())

    # ------------------------------------------------------------------
    # Trainer abstract methods
    # ------------------------------------------------------------------

    def build_optimizers(self) -> torch.optim.Optimizer:
        """Build a plain Adam optimizer over the policy parameters."""
        lr = float(
            getattr(getattr(self.config, "train", None), "optm_lr", _DEFAULT_LR)
            or _DEFAULT_LR
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return self.optimizer

    def build_lr_schedulers(self) -> _NoOpLRScheduler:
        """Return a no-op LR scheduler. The toy trainer does not decay LR."""
        self.lr_scheduler = _NoOpLRScheduler()
        return self.lr_scheduler

    def step_validation(self) -> Dict[str, Any]:
        """Toy trainer skips validation. Returns empty metrics."""
        return {}

    def export_safetensors(
        self,
        output_dir: str,
        rel_path: str,
        trainable_only: bool = False,
        is_final: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        logger.warning(
            "[GymTrainer] export_safetensors is a no-op in the toy demo "
            f"(would have written to {output_dir}/{rel_path})."
        )

    def model_load_from_hf(self) -> None:
        logger.warning(
            "[GymTrainer] model_load_from_hf is a no-op in the toy demo; "
            "the policy is constructed fresh from the [model] section of the "
            "policy TOML."
        )

    def model_resume_from_checkpoint(self) -> None:
        logger.warning(
            "[GymTrainer] model_resume_from_checkpoint is a no-op in the toy demo."
        )

    # ------------------------------------------------------------------
    # Launcher-driven methods (called by rl_worker beyond Trainer's ABC)
    # ------------------------------------------------------------------

    def weight_resume(self) -> Dict[str, Any]:
        """Return ckpt-extra-info for :class:`WeightResumeCommand`.

        The toy demo does not persist checkpoints, so there is nothing
        to resume; return an empty dict so the controller's
        ``post_resume_info`` path stays consistent.
        """
        return {}

    def update_lr_schedulers(self, total_steps: int) -> None:
        """Step the LR schedule. The toy trainer uses a constant LR.

        ``rl_worker.execute_data_fetch`` calls this once per training
        step.  The base ``_NoOpLRScheduler.step`` is also called from
        :meth:`_finalize_training_step`; both are no-ops here.
        """
        del total_steps  # Unused; constant LR.

    def sync_all_states(
        self,
        is_send: bool,
        send_hook: Callable[..., Any],
        recv_hook: Callable[..., Any],
        reference_model: bool = False,
    ) -> int:
        """Inter-replica state sync.  Not exercised in single-replica colocated.

        ``rl_worker.execute_policy_to_policy_{broadcast,unicast}`` calls
        this when there is more than one policy replica or when
        ``train.train_policy.kl_beta != 0`` (reference-model copy).
        Neither applies to the toy demo's intended single-replica
        launch; we return 0 (parameters transferred) so that if the
        path is reached accidentally the caller sees a well-formed
        no-op rather than a crash.
        """
        del is_send, send_hook, recv_hook, reference_model
        return 0

    # ------------------------------------------------------------------
    # TrajectoryExpansionMixin hooks (rollout mode)
    # ------------------------------------------------------------------

    def _begin_training_step(
        self, rollouts: Sequence[Rollout], *args: Any, **kwargs: Any
    ) -> None:
        """Set the model to train mode, zero grads, init per-step buffers.

        Precomputes :attr:`_step_num_rollouts` so :meth:`_train_one_rollout`
        can divide its per-rollout backward by the batch size — this is
        the deliberate trainer-owned "mean over rollouts" length-normalization
        choice (see the contract feature doc).
        """
        if self.optimizer is None:
            self.build_optimizers()
        self.model.train()
        self.optimizer.zero_grad()
        self._step_losses = []
        self._step_returns = []
        self._step_episode_lengths = []
        self._step_num_rollouts = max(1, len(rollouts))
        self._step_start_time = time.perf_counter()

    def _train_one_rollout(self, rollout: Rollout, *args: Any, **kwargs: Any) -> None:
        """Run forward+backward on a single trajectory.

        Reads the trajectory dict from the packer, computes discounted
        returns, runs the policy forward over the valid prefix of
        observations, and applies a return-weighted MSE loss between
        the predicted and sampled actions. The backward call is scaled
        by ``1 / num_rollouts`` so the step-level effective loss is the
        mean across rollouts (matching the Adam optimizer's expectation
        that a single ``step()`` follows a sum-of-mean-batch-losses).
        """
        traj = _trajectory_from_rollout(rollout)
        ep_len = _episode_length(traj)
        if ep_len == 0:
            logger.debug("[GymTrainer] Skipping zero-length rollout.")
            return

        observations = self._as_tensor(traj["observations"][:ep_len], torch.float32)
        actions = self._as_tensor(traj["actions"][:ep_len], None)
        rewards = self._as_tensor(traj["rewards"][:ep_len], torch.float32)

        returns = compute_returns(rewards, gamma=self.gamma)

        predicted = self.model.forward(observations)
        loss, metrics = compute_simple_pg_loss(predicted, actions, returns)

        scaled = loss / self._step_num_rollouts
        scaled.backward()

        self._step_losses.append(metrics["loss"])
        self._step_returns.append(metrics["mean_return"])
        self._step_episode_lengths.append(metrics["num_steps"])

    def _finalize_training_step(
        self, rollouts: Sequence[Rollout], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Clip gradients, step the optimizer, return aggregated metrics."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        losses = self._step_losses or []
        n = max(1, len(losses))
        loss_avg = float(sum(losses) / n)
        loss_max = float(max(losses)) if losses else 0.0
        iter_time = (
            float(time.perf_counter() - self._step_start_time)
            if self._step_start_time is not None
            else 0.0
        )
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        # ``rl_worker`` passes ``current_step`` via kwargs; default to 0
        # so unit tests that don't supply it still get a valid metric.
        current_step = int(kwargs.get("current_step", 0))

        # Standard cosmos-rl per-step report keys consumed by the
        # controller (``cosmos_rl/dispatcher/status.py:train_ack``).
        # Required (no ``.get`` default in train_ack):
        #   ``train/loss_avg``, ``train/loss_max``,
        #   ``train/learning_rate``, ``train/iteration_time``,
        #   ``train_step`` (note: NO ``train/`` prefix).
        # The rest are optional and tolerated via ``.get(..., 0)``.
        metrics: Dict[str, Any] = {
            "train/loss_avg": loss_avg,
            "train/loss_max": loss_max,
            "train/learning_rate": learning_rate,
            "train/iteration_time": iter_time,
            "train/grad_norm": float(grad_norm),
            "train_step": current_step,
            # Gym-specific extras (kept for parity with our unit tests
            # and for whoever wants to grep wandb logs by trajectory).
            "train/mean_return": float(sum(self._step_returns or []) / n),
            "train/mean_episode_length": float(
                sum(self._step_episode_lengths or []) / n
            ),
            "train/num_rollouts": int(self._step_num_rollouts),
        }

        # Clear scratch state.
        self._step_losses = None
        self._step_returns = None
        self._step_episode_lengths = None
        self._step_num_rollouts = 0
        self._step_start_time = None
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _as_tensor(self, value: Any, dtype: Optional[torch.dtype]) -> torch.Tensor:
        """Coerce numpy arrays or scalars to a tensor on ``self.device``.

        Existing torch tensors are moved to the right device / dtype
        as needed without an unnecessary copy.
        """
        if isinstance(value, torch.Tensor):
            t = value
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            if t.device != self.device:
                t = t.to(self.device)
            return t
        t = torch.as_tensor(value)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if t.device != self.device:
            t = t.to(self.device)
        return t


__all__ = ["GymTrainer"]
