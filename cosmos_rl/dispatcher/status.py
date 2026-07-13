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

import time
import math
from queue import Empty, Queue
from strenum import StrEnum
from typing import Dict, List, Iterator, Any, Optional, Callable
from cosmos_rl.utils.constant import COSMOS_HEARTBEAT_TIMEOUT
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import RollingDict
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.replica import Replica, Atom, Rollout
from cosmos_rl.dispatcher.protocol import MESH_NAMES, Role
import cosmos_rl.dispatcher.command as command
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.utils.payload_transport import PayloadTransportRegistry
from cosmos_rl.utils.report.wandb_logger import (
    is_wandb_available,
    log_wandb,
)
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from transformers import AutoTokenizer
import numpy as np
from cosmos_rl.utils.util import aggregate_report_data


# Debug-only accounting log for ``samples_on_the_fly``. Only the mutation
# sites that can drift from dispatch accounting call this helper; the
# dispatch-side increment is intentionally not wrapped because prompt fetches
# can be hot.
def _log_samples_on_the_fly_mutation(
    source: str,
    before: int,
    after: int,
    *,
    extra: str = "",
) -> None:
    delta = after - before
    extra_str = f" {extra}" if extra else ""
    logger.debug(
        "[Controller samples_on_the_fly] source=%s before=%d delta=%+d after=%d%s",
        source,
        before,
        delta,
        after,
        extra_str,
    )


def need_weight_sync(
    *,
    step: int,
    total_steps: int,
    sync_weight_interval: int,
    validation_enabled: bool,
    validation_freq: Optional[int],
) -> bool:
    """Pure decision: should the controller trigger a P2R/R2R weight sync
    after training ``step``?

    Weight sync after step N exists to ship the freshly trained weights to
    the rollouts so they generate the rollouts trained at step N+1.  On the
    **final** step there is no N+1, so the sync is never consumed -- and
    issuing it anyway (the old ``or step == total_steps`` "ending signal")
    raced rollout self-terminate at end-of-data: the policy's deferred final
    P2R landed after the recipient rollout had already drained + aborted,
    orphaning the P2R recv and wedging ``ncclCommAbort`` (see
    rollout_multirank_shutdown.md).

    So for non-validation runs we **never** sync on the final step; rollouts
    stop via the unified prompt-stream ``is_end`` path
    (``controller.get_batched_prompt`` forces ``is_end`` once training is
    finished, feeding the ``prompt_consume_end`` fast path), with the
    explicit ``StopCommand`` broadcast as the backstop for any rank that
    never observes ``is_end`` (e.g. one wedged on the weight-version gate).

    Validation-enabled runs are the exception: the final/periodic validation
    is driven by the controller R2R broadcast and the rollout needs the
    synced weights to run it, so the last-step sync is kept (mirrored by the
    validation gating in the ``trigger_weight_sync`` exclusion and the STOP
    broadcast suppression).
    """
    need = step % sync_weight_interval == 0
    if validation_enabled:
        if validation_freq:
            need = need or (step % validation_freq == 0)
        need = need or step == total_steps
        return need
    # Non-validation: suppress the sync entirely on the final step.
    if step == total_steps:
        return False
    return need


def should_broadcast_stop(
    *,
    n_policy: int,
    had_policy_replicas: bool,
    stop_broadcast_sent: bool,
    validation_enabled: bool,
    training_finished: bool,
    all_rollouts_ended: bool,
) -> bool:
    """Pure decision: should the controller broadcast the end-of-job
    ``StopCommand`` to the rollout set now?

    STOP is the NCCL-free, authoritative end-of-job signal for the rollouts
    (see :class:`~cosmos_rl.dispatcher.command.StopCommand`).  It must fire
    exactly once, at the moment the policy side is genuinely done:

    - ``n_policy == 0`` **and** ``had_policy_replicas``: every policy replica
      has unregistered after finishing its main loop (or been reaped).  A
      bare ``n_policy == 0`` at cold start -- before any policy has been
      seen -- must not trigger it.
    - ``training_finished`` **or** ``all_rollouts_ended``: distinguishes
      genuine end-of-job from a transient ``n_policy == 0`` during dynamic
      policy rescaling (scale-to-zero / rolling restart,
      ``current_step < total_steps``), where stopping the rollouts would be
      wrong.  ``all_rollouts_ended`` covers the post-``is_end`` case where
      the buffer-only recompute still leaves ``total_steps > current_step``
      (untrained buffer backlog the policy will never drain) so
      ``training_finished()`` is false even though the policy has already
      unregistered -- without this OR-guard STOP never fires and rollouts
      wedge (the GRPO end-of-data failure in CI).
    - ``not validation_enabled``: validation runs keep the final weight sync
      and stop via the R2R ``replica_should_stop`` broadcast instead, so STOP
      would race that path.
    - ``not stop_broadcast_sent``: one-shot; once sent we never re-broadcast.

    The caller still decides separately whether any rollout replicas remain
    to receive it; this is purely the "are we at the stop point?" gate.
    """
    policy_side_done = training_finished or all_rollouts_ended
    return (
        n_policy == 0
        and had_policy_replicas
        and not stop_broadcast_sent
        and not validation_enabled
        and policy_side_done
    )


def should_coalesce_skip(
    *,
    coalesce_enabled: bool,
    forced: bool,
    last_staged_step: int,
    max_adopted_version: int,
) -> bool:
    """Pure decision: should the controller *skip* (coalesce) this weight-sync
    round instead of issuing a fresh P2R+R2R?

    Depth-1 drop-to-latest.  A round the controller issued at
    ``last_staged_step`` is still *in flight* exactly while the rollouts have
    not yet adopted it -- i.e. ``last_staged_step > max_adopted_version``
    (``max_adopted_version`` is the freshest weight version any rollout has
    generated with, tracked from the stamped ``rollout.weight_version``).
    Issuing another round while one is in flight would only pile redundant
    ~1.6 GB transfers onto the rollout's ``WeightSyncThread`` queue -- every
    intermediate version is superseded before it is ever adopted (the rollout's
    ``sync_buffer_to_live`` always jumps to the latest ``buf_ver``).  So we skip;
    once the rollouts catch up the next tick issues at ``current_step`` (the
    latest) -> drop-to-latest.

    Note "in flight" is *derived*, not tracked: there is exactly one source of
    truth (the adopted-version comparison), so no counter can drift.

    ``forced`` overrides the skip for the one round that must never be coalesced
    away: a validation-trigger step, where the rollout needs that exact version
    to validate.  (The first sync issues naturally -- ``last_staged_step`` starts
    at -1, so the comparison is false -- and the staleness ceiling is enforced
    independently by ``filter_outdated_rollouts``.)

    When ``coalesce_enabled`` is False this is always False -> behaviour is
    identical to the unconditional every-interval sync.
    """
    if not coalesce_enabled or forced:
        return False
    return last_staged_step > max_adopted_version


class ReplicaScalingEnum(StrEnum):
    """
    Enum for replica scaling event.
    """

    REPLICA_SCALING_UP = "replica_scaling_up"
    REPLICA_SCALING_DOWN = "replica_scaling_down"


class ReplicaScalingLog:
    event: ReplicaScalingEnum
    replica_name: str
    timestamp: int

    def __init__(
        self, event: ReplicaScalingEnum, replica_name: str, timestamp: int = None
    ):
        self.event = event
        self.replica_name = replica_name
        self.timestamp = timestamp if timestamp is not None else int(time.time())

    @staticmethod
    def up(replica: Replica):
        return ReplicaScalingLog(ReplicaScalingEnum.REPLICA_SCALING_UP, replica.name)

    @staticmethod
    def down(replica: Replica):
        return ReplicaScalingLog(ReplicaScalingEnum.REPLICA_SCALING_DOWN, replica.name)


class PolicyStatus(StrEnum):
    """
    Enum for policy status.
    There are 7 statuses:
    UNINITIALIZED: The policy is uninitialized.
    READY: The policy is ready to run.
    RUNNING: The policy is running.
    REDUCED: The policy has finished reduce.
    END: The policy has finished.
    VALIDATED: The policy has finished validation.
    """

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    REDUCED = "reduced"
    END = "end"
    VALIDATED = "validated"


class JobPhase(StrEnum):
    """Controller job phase for non-validation RL shutdown.

    STOPPING and DONE remain implicit via ``stop_broadcast_sent``,
    ``should_broadcast_stop``, and ``_maybe_finalize``.
    """

    RUNNING = "running"
    DRAINING = "draining"


class PolicyStatusManager:
    """
    A class to manage the status of a policy.
    """

    policy_replicas: Dict[str, Replica]
    policy_init_done: bool = False
    replica_scaling_log: List[ReplicaScalingLog]

    # Global status
    remain_samples_num: int
    current_step: int
    total_steps: int

    # Instance status
    status: Dict[str, PolicyStatus]

    def __init__(self):
        self.policy_replicas = {}
        # number of steps that needed to interate over all the samples across all the epochs.
        self.total_steps = 0
        # current step of the policy training, this step could won't reach to total_steps because of dynmaic sampling.
        # Some samples could be filtered out due to dynamic sampling and they won't be used for policy training.
        # This step is the actual weight update step, it is also binded to the weight version.
        self.current_step = 0

        self.rollout_buffer = Queue()
        self.remain_samples_num = 0
        self.samples_on_the_fly = 0

        # Actual rollout count for each in-flight real training command.
        # Entries are keyed by the command step and consumed after its full
        # policy ACK set, keeping samples_on_the_fly accounting symmetric.
        self.dispatched_rollouts_by_step: Dict[int, int] = {}

        self.status = {}

        self.train_report_data = RollingDict(maxlen=20)

        self.replica_scaling_log = []

        # NCCL payload transfer cleanup: disabled by default, auto-enabled
        # when the first nccl:-prefixed rollout is seen.
        self._nccl_cleanup_enabled = False

        # Validation related
        self.val_report_data: Dict[int, List[Any]] = {}

        # Indicate whether on-policy rollout collection has completed for the current policy step
        self.on_policy_rollout_completed: bool = False

        # Record filter rewards distribution for dynamic sampling
        self.filter_records = {}

        # Non-validation RL: explicit end-of-data phase (see ``JobPhase``).
        self.job_phase = JobPhase.RUNNING
        self.draining_total_steps: Optional[int] = None
        self.last_real_datafetch_acked_step: Optional[int] = None
        self.last_real_datafetch_acked_total_steps: Optional[int] = None
        self.completion_step: Optional[int] = None
        self.completion_recipients: set[str] = set()
        self.completion_acks: set[str] = set()
        self.terminal_complete = False

        # For rank specific data dispatch
        self.rollout_buffer_per_rank: List[Queue] = []

        # --- Weight-sync coalescing (depth-1 drop-to-latest) state ---
        # ``_weight_last_staged_step``: weight_step of the most recent issued
        #   round (-1 = none issued yet, so the first sync issues naturally).
        # ``_weight_max_adopted_version``: highest rollout-reported weight
        #   version seen (updated from put_rollouts).  A round is "in flight"
        #   iff _weight_last_staged_step > _weight_max_adopted_version -- the
        #   single derived source of truth the coalescing gate keys on.
        # ``_weight_coalesced_skips``: count of coalesced (skipped) rounds, for
        #   the bench report (observability only).
        self._weight_last_staged_step: int = -1
        self._weight_max_adopted_version: int = -1
        self._weight_coalesced_skips: int = 0
        # Latest accepted-rollout staleness percentiles, merged into the
        # train_ack report_data (operator-facing tuning signal).
        self._weight_staleness_recent: Dict[str, int] = {}

    def setup(
        self,
        config: Config,
        redis_handler: RedisStreamHandler,
        data_fetcher: ControllerDataFetcher,
        remain_samples_num: int,
        samples_per_epoch: int,
        tokenizer: Optional[AutoTokenizer] = None,
        current_step: int = 0,
        max_num_steps: Optional[int] = None,
        custom_logger_fns: Optional[List[Callable]] = None,
        hook_fns: Optional[Dict[str, Callable]] = None,
    ):
        self.redis_handler = redis_handler
        self.config = config
        self.remain_samples_num = remain_samples_num
        self.samples_per_epoch = samples_per_epoch
        self.tokenizer = tokenizer
        self.current_step = current_step
        self.max_num_steps = max_num_steps
        self.custom_logger_fns = (
            custom_logger_fns if custom_logger_fns is not None else []
        )
        self.hook_fns = hook_fns if hook_fns is not None else {}
        self.data_fetcher = data_fetcher

        self.recompute_total_steps()
        # For resume case to activate dataloader and validation if needed
        if (
            self.config.train.resume
            and self.config.validation.enable
            and self.current_step > 0
            and (
                self.current_step % self.config.validation.freq == 0
                or self.current_step == self.total_steps
            )
        ):
            self.data_fetcher.validation_activate_dataloader(self.current_step)

    def n_atoms_per_replica(self) -> int:
        """
        Get the number of GPUs per replica.
        """
        if len(self.policy_replicas) == 0:
            return 0
        return next(iter(self.policy_replicas.values())).n_atoms_per_replica()

    def __len__(self) -> int:
        """
        Get the number of policies.
        """
        return len(self.policy_replicas)

    def __iter__(self) -> Iterator[Replica]:
        """
        Iterate over the policy replicas.
        """
        for replica in sorted(self.policy_replicas.values(), key=lambda x: x.name):
            yield replica

    def __contains__(self, replica_name: str) -> bool:
        """
        Check if the replica is in the status manager.
        """
        return replica_name in self.policy_replicas

    def __getitem__(self, replica_name: str) -> Replica:
        """
        Get the replica from the status manager.
        """
        return self.policy_replicas.get(replica_name)

    def training_finished(self) -> bool:
        """
        Check if the training is finished.
        """
        total_steps = self.training_horizon()
        return self.terminal_complete or (
            self.current_step >= total_steps and total_steps > 0
        )

    def training_horizon(self) -> int:
        """Return the immutable drain horizon once rollout input is closed."""
        if self.draining_total_steps is not None:
            return self.draining_total_steps
        return self.total_steps

    def rollout_admission_closed(self) -> bool:
        """Whether new rollout results can no longer feed a training step."""
        return self.terminal_complete or self.completion_step is not None

    def maintain_life_status(self):
        """
        Maintain the life status of the rollout.
        """
        dead_replicas = set()
        now = time.time()
        for replica in self:
            if now - replica.status.heartbeat_timestamp > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Policy {replica.name} is dead")
                dead_replicas.add(replica.name)
        for replica_name in dead_replicas:
            self.unregister(replica_name)

    def set_status(self, name: str, status: PolicyStatus):
        """
        Set the status of the policy.
        """
        if name not in self.status:
            assert status == PolicyStatus.UNINITIALIZED, (
                "Policy status should be UNINITIALIZED when first created"
            )
            self.status[name] = status
            return
        assert status != PolicyStatus.UNINITIALIZED, (
            "Policy status should not be UNINITIALIZED when already created"
        )
        self.status[name] = status

    def recompute_total_steps(
        self, explicit_num_remaining_samples: Optional[int] = None
    ):
        """
        Set the ranks of the policies.
        """
        if self.job_phase == JobPhase.DRAINING or self.training_finished():
            # Training is finished, do not recompute total steps
            return
        # Update total_steps based on remaining samples and replicas
        num_policy_replicas = len(self.get_all_atoms_arrived_replicas())
        if num_policy_replicas == 0:
            return

        num_remaining_samples = (
            explicit_num_remaining_samples
            if explicit_num_remaining_samples is not None
            else self.remain_samples_num
        )

        steps_by_dataset = self.current_step + num_remaining_samples // (
            self.config.train.train_batch_per_replica * num_policy_replicas
        )

        # If max_num_steps is set, honour the smaller one.
        if self.config.train.max_num_steps is not None:
            self.total_steps = min(steps_by_dataset, self.config.train.max_num_steps)
        else:
            self.total_steps = steps_by_dataset

    def _total_steps_from_remaining_samples(self, num_remaining_samples: int) -> int:
        """Compute ``total_steps`` from a sample count without mutating state."""
        num_policy_replicas = len(self.get_all_atoms_arrived_replicas())
        if num_policy_replicas == 0:
            return self.total_steps
        steps_by_dataset = self.current_step + num_remaining_samples // (
            self.config.train.train_batch_per_replica * num_policy_replicas
        )
        if self.config.train.max_num_steps is not None:
            return min(steps_by_dataset, self.config.train.max_num_steps)
        return steps_by_dataset

    def enter_draining_phase(self) -> None:
        """Close rollout input and freeze the advertised training horizon."""
        if self.job_phase != JobPhase.RUNNING:
            return
        self.job_phase = JobPhase.DRAINING
        self.draining_total_steps = self.total_steps
        logger.info(
            "[Controller] Job phase RUNNING -> DRAINING at frozen horizon %d",
            self.draining_total_steps,
        )

    def finish_draining_phase(
        self,
        rollout_status_manager: "RolloutStatusManager",
    ) -> None:
        """Dispatch complete tail batches, then complete without inventing work."""
        if not rollout_status_manager.all_rollouts_ended():
            return
        if self.job_phase == JobPhase.RUNNING:
            self.enter_draining_phase()

        if self.rollout_admission_closed():
            self.cleanup_buffered_rollouts()
            return
        if not self.all_ready_or_reduced():
            return

        frozen_total = self.training_horizon()
        if self.current_step < frozen_total and self.rollouts_enough_for_one_step():
            previous_step = self.current_step
            self.try_trigger_data_fetch_and_training()
            if self.current_step > previous_step:
                return

        self.cleanup_buffered_rollouts()
        if self.real_terminal_command_acked():
            self.terminal_complete = True
            return
        self.trigger_training_complete()

    def on_rollout_is_end(
        self,
        rollout_status_manager: "RolloutStatusManager",
    ) -> None:
        """Single entry for rollout HTTP ``is_end`` POST (not prompt fetch)."""
        if self.config.validation.enable:
            return
        if not rollout_status_manager.all_rollouts_ended():
            return
        self.finish_draining_phase(rollout_status_manager)

    def should_weight_sync_after_train_ack(
        self,
        step: int,
        rollout_status_manager: "RolloutStatusManager",
    ) -> bool:
        """Whether ``train_ack`` should schedule P2R/R2R for ``step``."""
        if self.job_phase == JobPhase.DRAINING:
            return False
        need_sync_weight = need_weight_sync(
            step=step,
            total_steps=self.total_steps,
            sync_weight_interval=self.config.train.sync_weight_interval,
            validation_enabled=self.config.validation.enable,
            validation_freq=(
                self.config.validation.freq if self.config.validation.enable else None
            ),
        )
        if rollout_status_manager.all_rollouts_ended():
            # Validation runs can exhaust the training prompt stream (``is_end``)
            # before the final ``train_ack`` lands.  ``status.ended`` only means
            # "no more training prompts", not "validation + shutdown complete".
            # Keep the final-step R2R so rollout receives ``validation_flag``
            # and ``replica_should_stop``; suppressing it here wedged final
            # validation at 0/N with the controller val dataloader activated.
            if not (
                self.config.validation.enable
                and need_sync_weight
                and step == self.total_steps
            ):
                return False
        if need_sync_weight:
            targets = self._weight_sync_rollout_targets(rollout_status_manager)
            if not targets:
                logger.warning(
                    "[Controller] Suppressing weight sync for step=%s because "
                    "there are no rollout replicas available to receive it "
                    "(validation_enabled=%s total_steps=%s)",
                    step,
                    self.config.validation.enable,
                    self.total_steps,
                )
                return False
        return need_sync_weight

    def trigger_training_complete(self) -> None:
        """Stop policy replicas at synthetic coordinates without advancing K."""
        if self.data_fetcher.activated_val_iter is not None:
            return

        arrived_replicas = self.get_all_atoms_arrived_replicas()
        if len(arrived_replicas) == 0:
            return
        if self.completion_step is not None:
            return

        frozen_total = self.training_horizon()
        completion_step = self.current_step + 1
        recipients = {replica.name for replica in arrived_replicas}

        # Stage the complete recipient snapshot before the first publish.  A
        # partial publication can then hang, but can never look successful or
        # admit late rollout results that no future trainer can consume.
        self.completion_step = completion_step
        self.completion_recipients = recipients
        self.completion_acks = set()
        for replica in arrived_replicas:
            self.set_status(replica.name, PolicyStatus.RUNNING)

        do_save = bool(frozen_total > 0 and self.config.train.ckpt.enable_checkpoint)
        for replica in arrived_replicas:
            command.TrainingCompleteCommand.trigger(
                replica=replica,
                global_step=completion_step,
                total_steps=completion_step,
                final_step=self.current_step,
                checkpoint_total_steps=frozen_total,
                remain_samples_num=self.remain_samples_num,
                do_save=do_save,
                redis_handler=self.redis_handler,
            )

    def record_completion_ack(self, replica_name: str, step: int) -> bool:
        """Record one ACK for the active synthetic completion command."""
        if (
            step != self.completion_step
            or replica_name not in self.completion_recipients
        ):
            return False
        if replica_name in self.completion_acks:
            return True
        self.completion_acks.add(replica_name)
        if self.completion_acks == self.completion_recipients:
            self.terminal_complete = True
        return True

    def _weight_sync_rollout_targets(
        self,
        rollout_status_manager: "RolloutStatusManager",
    ) -> List[Replica]:
        """Return a communicator-safe rollout recipient set."""
        return rollout_status_manager.get_safe_weight_sync_replicas(
            validation_enabled=self.config.validation.enable
        )

    def _expected_validation_rollout_count(self) -> int:
        val_datasize = getattr(self.data_fetcher, "val_datasize", 0)
        if not val_datasize and getattr(self.data_fetcher, "val_dataloader", None):
            val_datasize = len(self.data_fetcher.val_dataloader)
        return val_datasize * self.config.validation.n_generation

    def _reported_validation_rollout_count(self, validation_step: int) -> int:
        return sum(len(x) for x in self.val_report_data.get(validation_step, []))

    def get_status(self, name: str) -> PolicyStatus:
        """
        Get the status of the policy.
        """
        if name not in self.status:
            raise KeyError(f"Policy {name} not found")
        return self.status[name]

    def all_with_status(self, status: List[PolicyStatus]) -> bool:
        """
        Check if all policies have the given status.
        """
        return all([x in status for x in self.status.values()])

    def any_with_status(self, status: List[PolicyStatus]) -> bool:
        """
        Check if any policies have the given status.
        """
        return any([x in status for x in self.status.values()])

    def all_reduced(self) -> bool:
        """
        Check if all policies are reduced.
        """
        return self.all_with_status([PolicyStatus.REDUCED])

    def all_ready(self) -> bool:
        """
        Check if all policies are ready.
        """
        return self.all_with_status([PolicyStatus.READY])

    def all_ready_or_reduced(self) -> bool:
        """
        Check if all policies are ready or reduced.
        """
        return self.all_with_status([PolicyStatus.READY, PolicyStatus.REDUCED])

    def set_ncclerror(self, replica_name: str, timestamp: int):
        """
        Set the timeout ack of the policy.
        """
        self[replica_name].status.nccl_error_timestamp = timestamp

    def clear_ncclerror(self):
        """
        Clear the timeout ack of the policy.
        """
        for replica in self:
            replica.status.nccl_error_timestamp = None

    def get_all_policy_report_ncclerror(self) -> Dict[str, int]:
        """
        Get all the timeout ack of the policies.
        """
        return {
            replica.name: replica.status.nccl_error_timestamp
            for replica in self
            if replica.status.nccl_error_timestamp is not None
        }

    def heartbeat(self, replica_name: str):
        timestamp: int = int(time.time())
        if replica_name not in self:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in policy status manager."
            )
            return
        self[replica_name].status.heartbeat_timestamp = timestamp

    def shutdown(self):
        """
        Shutdown the status manager.
        """
        self.policy_init_done = False

    def unregister(self, replica_name: str):
        """
        Unregister the replica from the status manager.
        """
        assert replica_name in self, (
            f"Replica {replica_name} not found in policy status manager"
        )

        replica = self.policy_replicas.pop(replica_name)
        self.status.pop(replica_name)
        self.replica_scaling_log.append(ReplicaScalingLog.down(replica))

        if self.training_finished() or replica_name in self.completion_acks:
            # This policy replica is normally finished
            # Do not trigger rebuild mesh since everything is gonna be finished shortly
            logger.info(f"[Controller] Replica {replica_name} is stopping.")
            return

        valid_replicas = self.get_all_atoms_arrived_replicas()
        if replica.in_mesh and len(valid_replicas) > 0:
            self.trigger_rebuild_mesh(valid_replicas)

    def register(
        self,
        atom: Atom,
        config: Config,
        rollout_status_manager: "RolloutStatusManager",
        **kwargs,
    ):
        """
        Register the atom to the status manager.
        """
        replica = self[atom.replica_name]
        if replica is None:
            replica = Replica(atom.replica_name, Role.POLICY, [atom])
            self.policy_replicas[atom.replica_name] = replica
        else:
            replica.arrive(atom)
        atom.bind_replica(replica)
        current_policy_replica = replica

        # post register hook
        if not self.policy_init_done:
            if len(self.policy_replicas) > config.policy.parallelism.n_init_replicas:
                config.policy.parallelism.n_init_replicas = len(self.policy_replicas)
                logger.info(
                    f"[Controller] Update policy n_init_replicas to {config.policy.parallelism.n_init_replicas} replicas"
                )

        # Check if all atoms of the replica have arrived
        if replica.all_atoms_arrived:
            if replica.start_time == -1:
                replica.start_time = int(time.time())
            logger.info(
                f"[Controller] All atoms of {Role.POLICY} Replica {replica.name} has been set."
            )
            self.set_status(replica.name, PolicyStatus.UNINITIALIZED)
            # Check total valid policy replicas
            valid_replicas = []
            if not hasattr(self, "policy_atoms_in_replica"):
                self.policy_atoms_in_replica = int(math.prod(atom.group_size))

            for r in self.policy_replicas.values():
                if r.all_atoms_arrived:
                    valid_replicas.append(r)

            # Load weight for the first loaded replica policy
            if len(valid_replicas) == 1:
                assert not hasattr(self, "_first_policy_replica_arrived"), (
                    "Expect only one policy replica to load weight during training process"
                )
                self._first_policy_replica_arrived = True
                # This is the first policy replica to arrive, it is responsible for weight initialization
                command.WeightResumeCommand.trigger(
                    current_policy_replica, redis_handler=self.redis_handler
                )

                # Check whether there is any valid rollout replicas
                any_valid_rollout_replica = None
                sorted_rollout_replicas = sorted(
                    rollout_status_manager.rollout_replicas.values(),
                    key=lambda x: x.start_time,
                )
                valid_rollout_replicas = []
                for r in sorted_rollout_replicas:
                    if r.all_atoms_arrived:
                        valid_rollout_replicas.append(r)
                        if any_valid_rollout_replica is None:
                            any_valid_rollout_replica = r
                if any_valid_rollout_replica:
                    command.PolicyToRolloutUnicastCommand.trigger(
                        src_replica=current_policy_replica,
                        dst_replica=any_valid_rollout_replica,
                        src_replica_size=self.policy_atoms_in_replica,
                        dst_replica_size=rollout_status_manager.rollout_atoms_in_replica,
                        weight_step=None,
                        total_steps=None,
                        redis_handler=self.redis_handler,
                    )
                    if (
                        len(valid_rollout_replicas)
                        >= config.rollout.parallelism.n_init_replicas
                    ):
                        command.RolloutToRolloutBroadcastCommand.trigger(
                            src_replica=any_valid_rollout_replica,
                            dst_replicas=valid_rollout_replicas,
                            weight_step=self.current_step,  # we must pass the current step to rollout replicas to track the weight version even in resume ckpt.
                            total_steps=None,
                            redis_handler=self.redis_handler,
                        )
                    logger.info(
                        f"[Controller] Trigger PolicyToRolloutUnicastCommand to {any_valid_rollout_replica.name} via Policy registration"
                    )
                else:
                    logger.info(
                        "[Controller] No valid rollout replicas found, skip PolicyToRolloutUnicastCommand"
                    )
            self.post_register_hook(
                valid_replicas,
                atom.replica,
                config,
                rollout_status_manager,
            )
        return replica

    def trigger_rebuild_mesh(self, valid_replicas: List[Replica]):
        # Always tell the policy to rebuild mesh even there is only one policy replica
        sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)
        command.BuildMeshCommand.trigger(
            sorted_valid_replicas, redis_handler=self.redis_handler
        )
        self.recompute_total_steps()
        self.data_fetcher.set_policy_global_mesh_size(len(sorted_valid_replicas))
        self.rearrange_rollout_buffer_after_mesh_rebuild(sorted_valid_replicas)

    def rearrange_rollout_buffer_after_mesh_rebuild(
        self, sorted_valid_replicas: List[Replica]
    ):
        # Only handle the case when data dispatch as rank in mesh is enabled for GRPO
        # Currently SFT does not support rank specific data dispatch
        if self.config.train.train_policy.data_dispatch_as_rank_in_mesh:
            new_rollout_buffer_per_rank: List[Queue[Rollout]] = [
                Queue() for _ in range(len(sorted_valid_replicas))
            ]
            for q in self.rollout_buffer_per_rank:
                while not q.empty():
                    rollout: Rollout = q.get()
                    new_rollout_buffer_per_rank[
                        rollout.prompt_idx % len(sorted_valid_replicas)
                    ].put(rollout)
            self.rollout_buffer_per_rank = new_rollout_buffer_per_rank

    def post_register_hook(
        self,
        valid_replicas: List[Replica],
        target_replica: Replica,
        config: Config,
        rollout_status_manager: "RolloutStatusManager",
    ):
        sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)

        if config.validation.enable and config.validation.val_before_train:
            self.data_fetcher.validation_activate_dataloader(0)

        if (
            not self.policy_init_done
            and len(valid_replicas) >= config.policy.parallelism.n_init_replicas
        ):
            # This is the case when all required replicas have arrived

            self.policy_init_done = True
            # Trigger mesh building (Typically only occurs during initialization)

            # we need buildmesh, event there is only one replica. (trigger HANccl buildmesh)
            # 1. Trigger mesh building
            self.trigger_rebuild_mesh(valid_replicas)

            # 2. Trigger weight/optimizer state synchronization
            if len(valid_replicas) > 1:
                # Only broadcast when there are multiple policy replicas
                initialized_replica = None
                for replica in sorted_valid_replicas:
                    # We will select the first replica that has weights loaded in view of command
                    if (
                        replica.weights_loaded_in_view_of_command
                        and replica in valid_replicas
                    ):
                        initialized_replica = replica
                        break
                assert initialized_replica is not None, (
                    "No replica was selected to load weights"
                )
                command.PolicyToPolicyBroadcastCommand.trigger(
                    src_replica=initialized_replica,
                    dst_replicas=valid_replicas,
                    total_steps=self.total_steps,
                    redis_handler=self.redis_handler,
                )
            # Set all policy replicas to `ready`
            for replica in valid_replicas:
                self.set_status(replica.name, PolicyStatus.READY)

            if self.config.mode == "colocated":
                # In colocated mode, we initially trigger data fetch for step 1 since the rollouts are generated locally.
                self.current_step += 1
                if self.config.validation.enable and (
                    self.current_step % self.config.validation.freq == 0
                    or self.current_step == self.total_steps
                ):
                    self.data_fetcher.validation_activate_dataloader(self.current_step)

                for replica in valid_replicas:
                    self.remain_samples_num -= self.config.train.train_batch_per_replica
                    command.DataFetchCommand.trigger(
                        replica=replica,
                        items_count=self.config.train.train_batch_per_replica,
                        global_step=self.current_step,
                        total_steps=self.total_steps,
                        # `remain_samples_num` is just for checkpointing the training progress
                        remain_samples_num=self.remain_samples_num,
                        # Only `do_save` when checkpointing is enabled
                        do_save=False,
                        redis_handler=self.redis_handler,
                    )
                    self.set_status(replica.name, PolicyStatus.RUNNING)
                    logger.info(
                        f"[Controller] Policy Replica {replica.name} is ready in colocated mode."
                    )
        elif (
            not self.policy_init_done
            and len(valid_replicas) < config.policy.parallelism.n_init_replicas
        ):
            # This is the case when replicas are in the initialization stage
            logger.info(
                f"Waiting for {config.policy.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )
        else:
            # This is the case when the dynamic scaling is triggered
            assert self.policy_init_done, (
                "Policy initialization must be done before building another mesh"
            )

            assert target_replica.status.mesh_rank == -1, (
                "Target replica should not be in the mesh"
            )

            # This occurs when new dynamic scaling is triggered
            initialized_replica = None
            for replica in sorted_valid_replicas:
                if (
                    replica.weights_loaded_in_view_of_command
                    and replica in valid_replicas
                ):
                    # We will select the first replica that has weights loaded in view of command
                    # to broadcast weights
                    initialized_replica = replica
                    break
            assert initialized_replica is not None, (
                "No replica was selected to load weights"
            )
            self.trigger_rebuild_mesh(valid_replicas)

            command.PolicyToPolicyUnicastCommand.trigger(
                src_replica=initialized_replica,
                dst_replica=target_replica,
                total_steps=self.total_steps,
                redis_handler=self.redis_handler,
            )
            self.set_status(target_replica.name, PolicyStatus.READY)

    def validation_report_validation_results(
        self,
        validation_step: int,
        validation_results: List[List[Rollout]],
        rollout_status_manager: "RolloutStatusManager",
    ):
        if validation_step not in self.val_report_data:
            self.val_report_data[validation_step] = []

        self.val_report_data[validation_step].extend(validation_results)
        n_items_of_this_step = self._reported_validation_rollout_count(validation_step)

        validation_finished = n_items_of_this_step == (
            self._expected_validation_rollout_count()
        )

        if self.data_fetcher.activated_val_tqdm:
            self.data_fetcher.activated_val_tqdm.update(
                n_items_of_this_step // self.config.validation.n_generation
            )
        else:
            logger.error("[Controller] Validation tqdm is not activated")
        # Check if all rollout replicas have reported validation results
        if validation_finished and self.data_fetcher.activated_val_iter is not None:
            # Validation is finished, trigger next step training
            self.data_fetcher.clear_validation_status()

            try:
                all_rollouts_lists: List[List[Rollout]] = self.val_report_data[
                    validation_step
                ]
                if all_rollouts_lists:
                    rewards = []
                    for rollouts in all_rollouts_lists:
                        rewards.extend([r.reward for r in rollouts])
                    avg_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    max_reward = np.max(rewards)
                    min_reward = np.min(rewards)

                    report_data = {
                        "val/reward_avg": avg_reward,
                        "val/reward_std": std_reward,
                        "val/reward_max": max_reward,
                        "val/reward_min": min_reward,
                        "val/rollout_count": len(rewards),
                        "val/step": validation_step,
                        "val/train_total_steps": self.total_steps,  # the total steps of the training when current validation step is triggered. This total_steps may change due to dynamic sampling.
                    }
                    logger.info(
                        f"[Controller] Validation finished, average reward: {avg_reward}, total rollouts: {len(rewards)}, max reward: {max_reward}, min reward: {min_reward}, std reward: {std_reward} at step {validation_step}"
                    )
                    report_data_list = [
                        rollout.report_metrics
                        if rollout.report_metrics is not None
                        else {}
                        for rollouts in all_rollouts_lists
                        for rollout in rollouts
                    ]
                    report_data = aggregate_report_data(
                        report_data_list, report_data, prefix="val/"
                    )
                    report_data_str = ", ".join(
                        [f"{k}: {v}" for k, v in report_data.items()]
                    )
                    logger.info(
                        f"[Controller] Validation report data from total {sum(len(rollouts) for rollouts in all_rollouts_lists)} rollouts: {report_data_str}"
                    )
                    if "wandb" in self.config.logging.logger and is_wandb_available():
                        log_wandb(
                            data=report_data,
                            step=validation_step,
                        )

                    # call custom logger fns
                    for custom_logger_fn in self.custom_logger_fns:
                        try:
                            custom_logger_fn(report_data, validation_step)
                        except Exception as e:
                            logger.warning(
                                f"[Controller] Error calling custom logger function: {e}"
                            )

            except Exception as e:
                logger.error(f"[Controller] Error reporting validation results: {e}")

            # The order is important, because the previous code block logs the previous step's validation results
            # while `try_trigger_data_fetch_and_training` will immediately report the next step's results
            self.try_trigger_data_fetch_and_training()

    def total_pending_rollouts(self) -> int:
        """
        Get the total pending rollouts.
        """
        if self.config.train.train_policy.data_dispatch_as_rank_in_mesh:
            return sum(q.qsize() for q in self.rollout_buffer_per_rank)
        return self.rollout_buffer.qsize()

    @staticmethod
    def _parse_non_negative_count(metrics: Dict[str, Any], key: str) -> int:
        value = metrics.get(key, 0)
        if type(value) is int and value >= 0:
            return value
        logger.warning(
            "[Controller] Ignoring malformed DAPO metric %s=%r; expected a "
            "non-negative integer",
            key,
            value,
        )
        return 0

    @classmethod
    def parse_dynamic_sampling_counts(
        cls, metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Sanitize DAPO counts once before any metric or counter mutation."""
        metrics = metrics or {}
        return {
            key: cls._parse_non_negative_count(metrics, key)
            for key in ("sampled", "filtered_positive", "filtered_negative")
        }

    def _settle_samples_on_the_fly(self, count: int, source: str) -> None:
        if count <= 0:
            return
        before = self.samples_on_the_fly
        self.samples_on_the_fly = max(0, before - count)
        _log_samples_on_the_fly_mutation(
            source,
            before,
            self.samples_on_the_fly,
            extra=f"settled_count={count}",
        )

    def _discard_rollouts(self, rollouts: List[Rollout], source: str) -> int:
        if not rollouts:
            return 0
        self._publish_payload_transport_cleanup(rollouts, [])
        self._settle_samples_on_the_fly(len(rollouts), source)
        return len(rollouts)

    def cleanup_buffered_rollouts(self) -> int:
        """Release every buffered rollout without consuming resumable budget."""
        dropped: List[Rollout] = []
        queues = (
            self.rollout_buffer_per_rank
            if self.config.train.train_policy.data_dispatch_as_rank_in_mesh
            else [self.rollout_buffer]
        )
        for rollout_queue in queues:
            while True:
                try:
                    dropped.append(rollout_queue.get_nowait())
                except Empty:
                    break
        return self._discard_rollouts(dropped, "terminal_buffer_cleanup")

    def cleanup_terminal_rollouts(
        self,
        rollouts: List[Rollout],
        metrics: Optional[Dict[str, Any]],
        *,
        is_dapo: bool,
    ) -> int:
        """Settle a post-terminal HTTP result without normal admission."""
        if rollouts:
            self._publish_payload_transport_cleanup(rollouts, [])
        settled_count = len(rollouts)
        if is_dapo:
            counts = self.parse_dynamic_sampling_counts(metrics)
            if counts["sampled"] != len(rollouts):
                logger.warning(
                    "[Controller] DAPO sampled=%d does not match %d extracted "
                    "terminal rollouts; using extracted count",
                    counts["sampled"],
                    len(rollouts),
                )
            settled_count += counts["filtered_positive"] + counts["filtered_negative"]
        self._settle_samples_on_the_fly(settled_count, "terminal_result_cleanup")
        return settled_count

    def real_terminal_command_acked(self) -> bool:
        """Whether this controller observed a fully ACKed real T/T command."""
        terminal_step = self.training_horizon()
        return (
            self.last_real_datafetch_acked_step == terminal_step
            and self.last_real_datafetch_acked_total_steps == terminal_step
        )

    def record_real_datafetch_acked(self, step: int, total_steps: int) -> None:
        """Record a fully ACKed real command and close natural-final input."""
        self.last_real_datafetch_acked_step = step
        self.last_real_datafetch_acked_total_steps = total_steps
        terminal_step = self.training_horizon()
        if step != terminal_step or total_steps != terminal_step:
            return
        # Activate first so a result handler cannot re-admit work between the
        # final ACK and cleanup.  FastAPI handlers do not yield in this region.
        self.terminal_complete = True
        self.cleanup_buffered_rollouts()

    def get_all_atoms_arrived_replicas(self) -> List[Replica]:
        """
        Get all the replicas that have all atoms arrived.
        """
        return [
            replica
            for replica in self.policy_replicas.values()
            if replica.all_atoms_arrived
        ]

    def put_rollout(self, rollout: Rollout):
        """
        Dispatch the rollout to the policy replicas in a round-robin manner.
        It is that replica's responsibility to dispatch the rollout to further (DP_SHARD) atoms.
        """
        if self.config.rollout.include_stop_str_in_output:
            if self.tokenizer.eos_token is not None and rollout.completion is not None:
                if not rollout.completion.endswith(self.tokenizer.eos_token):
                    rollout.completion = rollout.completion + self.tokenizer.eos_token
                    if (
                        self.config.rollout.multi_turn_config.enable
                        and rollout.completed_conversation[-1].role == "assistant"
                    ):
                        rollout.completed_conversation[
                            -1
                        ].content += self.tokenizer.eos_token
        if self.config.train.train_policy.data_dispatch_as_rank_in_mesh:
            # Dispatch based on prompt idx
            target_rank = rollout.prompt_idx % len(self.rollout_buffer_per_rank)
            self.rollout_buffer_per_rank[target_rank].put(rollout)
        else:
            self.rollout_buffer.put(rollout)
        self.try_trigger_data_fetch_and_training()

    def put_rollouts(self, rollouts: List[Rollout]):
        """
        Put the rollouts to the rollout buffer.

        Note on ``on_policy_rollout_completed``: this flag is a notification
        primitive set here when the pending queue drains so the trainer knows
        the current on-policy step is complete. It is reset by the trainer
        step-completion handler. It must NOT be used as a producer-side
        admission gate: the controller's prompt dispatch (driven by
        ``try_trigger_data_fetch_and_training`` inside ``put_rollout``) can
        issue step ``N+1`` prompts before the trainer wakes up and resets the
        flag, so step ``N+1`` rollouts can legitimately arrive while the flag
        is still ``True``. Their on-policy validity was already established
        at prompt-dispatch time (weight-version check); dropping them here
        destroys valid training data and, in the on-policy producer-consumer
        pipeline, deterministically deadlocks the trainer.
        """
        completion_tokens_count = 0
        n_samples = 0

        for rollout in rollouts:
            if self.config.train.train_policy.rollout_as_token_ids:
                completion_tokens_count += len(rollout.completion_token_ids)
            elif not self.config.train.non_text:
                completion_tokens_count += len(
                    self.tokenizer.encode(rollout.completion)
                )
            n_samples += 1
            self.put_rollout(rollout)
            if self.config.train.train_policy.on_policy:
                if self.total_pending_rollouts() == 0:
                    self.on_policy_rollout_completed = True
                    # Do not break: keep admitting any remaining rollouts in
                    # this batch. They are valid data for the next step and
                    # dropping them starves the consumer.

        return completion_tokens_count, n_samples

    def update_dynamic_sampling_statistics(self, filter_records: Dict[str, int]):
        """
        Update the dynamic sampling statistics.
        """
        counts = self.parse_dynamic_sampling_counts(filter_records)
        for k in ["sampled", "filtered_positive", "filtered_negative"]:
            self.filter_records[k] = self.filter_records.get(k, 0) + counts[k]

        # Update the remaining samples number to reflect the filtering results
        filtered_count = counts["filtered_positive"] + counts["filtered_negative"]
        self.remain_samples_num -= filtered_count
        # Filtered DAPO generations have no payload and can never reach a
        # training ACK, so settle their prompt-side in-flight accounting here.
        self._settle_samples_on_the_fly(filtered_count, "dapo_filter")

    def filter_outdated_rollouts(self, rollouts: List[Rollout]) -> List[Rollout]:
        """
        Filter out the outdated rollouts based on the current step.

        When NCCL payload transfer is active, discarded rollouts may hold
        GPU buffers on the rollout worker.  This method publishes explicit
        cleanup messages so the rollout worker releases them immediately
        instead of waiting for age-based cleanup.
        """
        allowed_outdated_steps = self.config.train.train_policy.allowed_outdated_steps
        filtered_rollouts = []
        accepted_staleness: List[int] = []
        discarded_staleness: List[int] = []
        for idx, rollout in enumerate(rollouts):
            assert rollout.weight_version <= self.current_step, (
                f"Rollout weight version {rollout.weight_version} is greater than current step {self.current_step}"
            )
            # Adoption signal for weight-sync coalescing: the freshest weight
            # version any rollout has actually generated with confirms that
            # round was delivered + adopted (deadlock-free re-arm signal).
            if rollout.weight_version > self._weight_max_adopted_version:
                self._weight_max_adopted_version = rollout.weight_version
            # Estimate the step when this rollout will be used for training
            # This is estimated based on the current step, the number of pending rollouts,
            # and the number of rollouts before this rollout in the current batch.
            estimated_step = self.current_step + (
                idx + self.total_pending_rollouts()
            ) // (
                self.config.train.train_batch_per_replica
                * max(len(self.get_all_atoms_arrived_replicas()), 1)
            )
            staleness = estimated_step - rollout.weight_version
            if staleness <= allowed_outdated_steps:
                filtered_rollouts.append(rollout)
                accepted_staleness.append(staleness)
            else:
                discarded_staleness.append(staleness)
                logger.debug(
                    f"[Controller] Filtered out outdated rollout with version {rollout.weight_version}, current step {self.current_step}, estimated step {estimated_step}, pending rollouts {self.total_pending_rollouts()}, preceeding rollouts in this batch {idx}, allowed_outdated_steps {allowed_outdated_steps}"
                )

        # Operator-facing weight-version staleness (see weight-sync coalescing
        # plan, Phase 1b): how outdated the *accepted* rollouts are vs the live
        # weight version, so the sync frequency can be tuned.  Stashed for the
        # train_ack report (`rollout/weight_staleness_*`) and logged inline with
        # the config knobs so the headroom vs allowed_outdated_steps is obvious.
        if accepted_staleness:
            p50 = int(np.percentile(accepted_staleness, 50))
            p99 = int(np.percentile(accepted_staleness, 99))
            smax = int(np.max(accepted_staleness))
            self._weight_staleness_recent = {
                "rollout/weight_staleness_p50": p50,
                "rollout/weight_staleness_p99": p99,
                "rollout/weight_staleness_max": smax,
            }
            logger.info(
                "[Controller] weight staleness: accepted p50=%d p99=%d max=%d, "
                "discarded=%d/%d (allowed_outdated_steps=%d, sync_weight_interval=%d)",
                p50,
                p99,
                smax,
                len(discarded_staleness),
                len(rollouts),
                allowed_outdated_steps,
                self.config.train.sync_weight_interval,
            )

        discarded_count = len(rollouts) - len(filtered_rollouts)

        # Update remaining samples number
        self.remain_samples_num -= discarded_count
        k = "outdated"
        self.filter_records[k] = self.filter_records.get(k, 0) + discarded_count

        if discarded_count > 0:
            self._settle_samples_on_the_fly(discarded_count, "filter_outdated")
            self._publish_payload_transport_cleanup(rollouts, filtered_rollouts)

        return filtered_rollouts

    def _publish_payload_transport_cleanup(
        self,
        rollouts: List[Rollout],
        filtered: List[Rollout],
    ) -> None:
        """Delegate per-transport cleanup dispatch to the registry.

        The grouping/dispatch logic lives in
        :meth:`PayloadTransportRegistry.handle_discarded`.  This wrapper
        only resolves the controller's Redis client and flips the
        ``_nccl_cleanup_enabled`` "first-detection" flag (used to
        debounce the "now active" log line so it appears at most once).

        Called by :meth:`filter_outdated_rollouts` whenever any rollout
        is discarded; ``handle_discarded`` itself is a cheap no-op when
        no payload-transport-prefixed rollouts are present, so calling
        it unconditionally is safe.
        """
        redis_client = self._resolve_cleanup_redis_client()
        published = PayloadTransportRegistry.handle_discarded(
            rollouts,
            filtered,
            config=self.config,
            redis_client=redis_client,
        )
        if published and not self._nccl_cleanup_enabled:
            self._nccl_cleanup_enabled = True
            logger.info(
                "[Controller] Detected payload-transport-prefixed rollouts; "
                "transport cleanup publishing is now active."
            )

    def _resolve_cleanup_redis_client(self) -> Any:
        """Return the controller's Redis client (or None) for cleanup."""
        redis_handler = getattr(self, "redis_handler", None)
        if redis_handler is None:
            return None
        if hasattr(redis_handler, "redis_clients") and redis_handler.redis_clients:
            return redis_handler.redis_clients[0]
        if hasattr(redis_handler, "redis_client"):
            return redis_handler.redis_client
        return None

    def sft_report_summary(
        self,
        train_step: int,
        total_steps: int,
        is_validation: bool = False,
    ):
        try:
            report_data = {}
            report_data = aggregate_report_data(self.report_data_list, report_data)
            self.report_data_list = []
            report_data_str = ", ".join([f"{k}: {v}" for k, v in report_data.items()])
            logger.debug(
                f"[Controller] {'Validation' if is_validation else 'Train'} report data from total {self.config.train.train_batch_per_replica * len(self.get_all_atoms_arrived_replicas())} data batch: {report_data_str}"
            )
            if "wandb" in self.config.logging.logger and is_wandb_available():
                log_wandb(
                    data=report_data,
                    step=train_step,
                )
            if "console" in self.config.logging.logger:
                if is_validation:
                    logger.info(
                        f"[SFT] Validation Loss: {report_data['val/avg_loss']:.5f} at step {train_step}/{total_steps}, epoch {self.data_fetcher.epoch - 1}."
                    )
                else:
                    logger.info(
                        f"Step: {train_step}/{total_steps}, Loss: {report_data['train/loss_avg']:.5f}, Max Loss {report_data['train/loss_max']:.5f}, Grad norm: {report_data['optimizer/grad_norm']:.5f}, Iteration time: {report_data['train/iteration_time']:.2f}s."
                    )
            for custom_logger_fn in self.custom_logger_fns:
                # We add a separate try-except block to handle the error of custom logger function.
                # This is to avoid the error of custom logger function affecting the fundamental logging system.
                for custom_logger_fn in self.custom_logger_fns:
                    try:
                        custom_logger_fn(report_data, train_step)
                    except Exception as e:
                        logger.warning(
                            f"[Controller] Error calling custom logger function: {e}"
                        )
        except Exception as e:
            import traceback

            logger.warning(
                f"[Controller] Warning reporting training results: {e}\n{traceback.format_exc()}"
            )
        for replica in self.get_all_atoms_arrived_replicas():
            self.set_status(replica.name, PolicyStatus.RUNNING)

    def sft_train_ack(
        self,
        replica_name: str,
        report_data: Dict[str, Any],
        step: int,
        total_steps: int,
    ):
        if "val/avg_loss" in report_data:
            # This is a validation ack from SFT validation step
            self.set_status(replica_name, PolicyStatus.VALIDATED)
            if self.all_with_status([PolicyStatus.VALIDATED]):
                # First validation ack received in this step
                # Trigger validation report
                self.sft_report_summary(
                    train_step=step,
                    total_steps=total_steps,
                    is_validation=True,
                )
            return
        if not self.any_with_status([PolicyStatus.REDUCED]):
            # For SFT, we increment current_step at first train_ack received in each step
            self.current_step += 1
            if self.config.validation.enable and (
                self.current_step % self.config.validation.freq == 0
                or self.current_step == self.total_steps
            ):
                self.data_fetcher.validation_activate_dataloader(self.current_step)
        self.set_status(replica_name, PolicyStatus.REDUCED)
        if self.all_reduced():
            # All replicas have been reduced, trigger remain_samples_num update and report
            self.remain_samples_num -= (
                self.config.train.train_batch_per_replica
            ) * len(self.get_all_atoms_arrived_replicas())
            self.sft_report_summary(
                train_step=step,
                total_steps=total_steps,
            )

    def train_ack(
        self,
        replica_name: str,
        step: int,
        total_steps: int,
        profile_finished: bool,
        report_data: Dict[str, Any],
        rollout_status_manager: "RolloutStatusManager",
    ):
        if replica_name not in self:
            raise Exception(f"Replica {replica_name} not found")

        # Synthetic completion ACKs have their own persistent recipient set.
        # Record them before logging/reduction bookkeeping and before the
        # worker can unregister.  They are never real DataFetch evidence.
        if self.record_completion_ack(replica_name, step):
            self.set_status(replica_name, PolicyStatus.REDUCED)
            return

        if not hasattr(self, "report_data_list"):
            self.report_data_list = []
        self.report_data_list.append(report_data)

        if self.config.train.train_policy.type == "sft":
            # For SFT with multiple replicas, we handle train_ack differently
            return self.sft_train_ack(
                replica_name,
                report_data,
                step,
                total_steps,
            )

        self.set_status(replica_name, PolicyStatus.REDUCED)

        if self.all_reduced():
            _sotf_before = self.samples_on_the_fly
            # Settle exactly the rollout count recorded for this real command.
            _missing_dispatch = object()
            _dispatch_record = self.dispatched_rollouts_by_step.pop(
                step, _missing_dispatch
            )
            _train_decrement = (
                0 if _dispatch_record is _missing_dispatch else _dispatch_record
            )
            if _dispatch_record is _missing_dispatch and step not in (
                self.total_steps,
                self.total_steps - 1,
            ):
                # Unexpected: a real step had no dispatch record. Either the
                # dispatch record was already consumed (double-ack) or a
                # step number is mismatched.  Log loudly but do not crash;
                # ``samples_on_the_fly`` stays balanced regardless.
                logger.warning(
                    "[Controller] train_ack for step=%d found no dispatch "
                    "record (current_step=%d total_steps=%d).  "
                    "Decrementing samples_on_the_fly by 0; this may "
                    "indicate a double-ack or step-numbering bug.",
                    step,
                    self.current_step,
                    self.total_steps,
                )
            self.samples_on_the_fly -= _train_decrement
            _log_samples_on_the_fly_mutation(
                "train_ack",
                _sotf_before,
                self.samples_on_the_fly,
                extra=(
                    f"step={step} replica={replica_name} "
                    f"recorded_dispatch={_train_decrement}"
                ),
            )
            assert self.samples_on_the_fly >= 0, (
                "samples_on_the_fly should not be negative"
            )
            if (
                getattr(self.config, "mode", None) != "colocated"
                and not self.config.validation.enable
                and _dispatch_record is not _missing_dispatch
                and _train_decrement > 0
            ):
                self.record_real_datafetch_acked(step, total_steps)
            # All replicas have been reduced; decide whether to weight-sync.
            # See ``need_weight_sync`` for the end-of-data rationale (the
            # final-step sync is suppressed for non-validation runs because it
            # is never consumed and races rollout teardown).
            #
            need_sync_weight = self.should_weight_sync_after_train_ack(
                step, rollout_status_manager
            )

            if profile_finished:
                # Only reset the do_profile flag if the profile is finished
                logger.debug(f"[Controller] Unset the profile mode of {replica_name}")
                self[replica_name].sub_profiler_config.do_profile = False

            # Sum and report data
            if self.config.logging.logger and not all(
                [not data for data in self.report_data_list]
            ):
                try:
                    total_loss_avg = np.mean(
                        [data["train/loss_avg"] for data in self.report_data_list]
                    )
                    total_loss_max = np.max(
                        [data["train/loss_max"] for data in self.report_data_list]
                    )
                    total_learning_rate = self.report_data_list[0][
                        "train/learning_rate"
                    ]
                    total_iter_time_avg = np.mean(
                        [data["train/iteration_time"] for data in self.report_data_list]
                    )
                    # KL loss
                    total_kl_loss_avg = np.mean(
                        [
                            data.get("train/kl_loss_avg", 0)
                            for data in self.report_data_list
                        ]
                    )
                    total_kl_loss_max = np.max(
                        [
                            data.get("train/kl_loss_max", 0)
                            for data in self.report_data_list
                        ]
                    )
                    total_grad_norm = np.mean(
                        [
                            data.get("train/grad_norm", 0)
                            for data in self.report_data_list
                        ]
                    )
                    total_entropy = np.mean(
                        [data.get("train/entropy", 0) for data in self.report_data_list]
                    )
                    total_effective_entropy = np.mean(
                        [
                            data.get("train/effective_entropy", 0)
                            for data in self.report_data_list
                        ]
                    )
                    train_step = self.report_data_list[0]["train_step"]
                    policy_report_data = {
                        "train/loss_avg": total_loss_avg,
                        "train/loss_max": total_loss_max,
                        "train/learning_rate": total_learning_rate,
                        "train/iteration_time": total_iter_time_avg,
                        "train/kl_loss_avg": total_kl_loss_avg,
                        "train/kl_loss_max": total_kl_loss_max,
                        "train/grad_norm": total_grad_norm,
                        "train/entropy": total_entropy,
                        "train/effective_entropy": total_effective_entropy,
                        "train/total_steps": total_steps,
                    }
                    policy_report_data = aggregate_report_data(
                        self.report_data_list, policy_report_data
                    )
                    if self.config.mode == "colocated":
                        for data in self.report_data_list:
                            # Handle dynamic sampling statistics update in colocated mode
                            self.update_dynamic_sampling_statistics(data)

                    if len(self.filter_records) > 0:
                        total_samples_for_filtering = sum(
                            v for v in self.filter_records.values()
                        )
                        if total_samples_for_filtering > 0:
                            for k, v in self.filter_records.items():
                                policy_report_data.update(
                                    {
                                        f"rollout/{k}_ratio": v
                                        / total_samples_for_filtering
                                    }
                                )
                    # Operator-facing weight-version staleness + coalescing
                    # activity (see weight-sync coalescing plan, Phase 1b/2).
                    if self._weight_staleness_recent:
                        policy_report_data.update(self._weight_staleness_recent)
                    policy_report_data["rollout/weight_coalesced_skips"] = (
                        self._weight_coalesced_skips
                    )
                    self.train_report_data.setdefault(train_step, {}).update(
                        policy_report_data
                    )
                    self.report_data_list = []

                    report_data_str = ", ".join(
                        [
                            f"{k}: {v}"
                            for k, v in self.train_report_data[train_step].items()
                            if k not in ["rollout_images", "rollout_videos"]
                        ]
                    )
                    logger.info(
                        f"[Controller] Train report data from total {self.config.train.train_batch_per_replica * len(self.get_all_atoms_arrived_replicas())} rollouts: {report_data_str}"
                    )

                    if "wandb" in self.config.logging.logger and is_wandb_available():
                        # Convert multimodal data to wandb compatible format if needed
                        import wandb

                        for modality in ["rollout_images", "rollout_videos"]:
                            if modality in self.train_report_data[train_step]:
                                # We only support logging a list of images/videos for now, and the caption of each image/video is set as the prompt and reward of the rollout that generated this image/video.
                                def _caption(prompt: str, reward_val: Any) -> str:
                                    return (
                                        f"{prompt[:100]} | avg: {float(reward_val):.2f}"
                                    )

                                raw_data = self.train_report_data[train_step][modality]
                                if modality == "rollout_images":
                                    wandb_mm_data = [
                                        wandb.Image(
                                            mm_result_sample["path"],
                                            caption=_caption(
                                                mm_result_sample["prompt"],
                                                mm_result_sample["reward"],
                                            ),
                                        )
                                        for mm_result_sample in raw_data
                                    ]
                                else:
                                    wandb_mm_data = [
                                        wandb.Video(
                                            mm_result_sample["path"],
                                            caption=_caption(
                                                mm_result_sample["prompt"],
                                                mm_result_sample["reward"],
                                            ),
                                            format="mp4",
                                        )
                                        for mm_result_sample in raw_data
                                    ]
                                self.train_report_data[train_step][modality] = (
                                    wandb_mm_data
                                )
                        log_wandb(
                            data=self.train_report_data[train_step],
                            step=train_step,
                        )
                    if "console" in self.config.logging.logger:
                        logger.info(
                            f"Step: {train_step}/{total_steps}, Reward Mean: {self.train_report_data[train_step]['train/reward_mean']:.4f}, Reward Std: {self.train_report_data[train_step]['train/reward_std']:.4f}, Reward Max: {self.train_report_data[train_step]['train/reward_max']:.4f}, Reward Min: {self.train_report_data[train_step]['train/reward_min']:.4f}, Completion Length Mean: {self.train_report_data[train_step]['rollout/completion_length_mean']:.2f}, Completion Length Max: {self.train_report_data[train_step]['rollout/completion_length_max']:.2f}, Average loss: {total_loss_avg:.5f}, Max loss: {total_loss_max:.5f}, Learning rate: {total_learning_rate:.5e}, Entropy: {total_entropy:.5f}, Effective Entropy: {total_effective_entropy:.5f}, Grad Norm: {total_grad_norm:.5f}, KL Loss Avg: {total_kl_loss_avg:.5f}, KL Loss Max: {total_kl_loss_max:.5f}, Iteration time: {total_iter_time_avg:.2f}s."
                        )
                        if len(self.filter_records) > 0:
                            logger.info(
                                f"Dynamic sampling rewards distribution so far: {self.filter_records}."
                            )
                    self.filter_records = {}
                    for custom_logger_fn in self.custom_logger_fns:
                        # We add a separate try-except block to handle the error of custom logger function.
                        # This is to avoid the error of custom logger function affecting the fundamental logging system.
                        try:
                            custom_logger_fn(
                                self.train_report_data[train_step], train_step
                            )
                        except Exception as e:
                            logger.warning(
                                f"[Controller] [Controller] Error calling custom logger function: {e}"
                            )
                except Exception as e:
                    import traceback

                    logger.warning(
                        f"[Controller] Warning reporting training results: {e}\n{traceback.format_exc()}"
                    )

            # All replicas have been reduced, trigger weight sync
            any_loaded_replica = None
            sorted_replicas = sorted(
                self.get_all_atoms_arrived_replicas(), key=lambda x: x.start_time
            )
            for replica in sorted_replicas:
                if any_loaded_replica is None:
                    any_loaded_replica = replica
                self.set_status(replica.name, PolicyStatus.READY)

            # P->R & R->R
            if need_sync_weight:
                # Weight-sync coalescing (depth-1 drop-to-latest): while a
                # previously issued round is still in flight to the rollouts
                # (last_staged > max_adopted), skip issuing redundant P2R+R2R
                # rounds -- every intermediate version is superseded on the
                # rollout before it is adopted.  Once the rollouts catch up, the
                # next tick issues at the latest step.
                tp = self.config.train.train_policy
                coalesce_enabled = (
                    self.config.train.coalesce_weight_sync
                    and getattr(tp, "allowed_outdated_steps", 0) > 0
                    and not getattr(tp, "on_policy", False)
                )
                forced = self._weight_sync_forced(step, self.total_steps)
                if should_coalesce_skip(
                    coalesce_enabled=coalesce_enabled,
                    forced=forced,
                    last_staged_step=self._weight_last_staged_step,
                    max_adopted_version=self._weight_max_adopted_version,
                ):
                    self._weight_coalesced_skips += 1
                    logger.info(
                        "[Controller] Coalesced weight-sync skip "
                        "(last_staged=%d, current=%d, max_adopted=%d, "
                        "total_skips=%d)",
                        self._weight_last_staged_step,
                        step,
                        self._weight_max_adopted_version,
                        self._weight_coalesced_skips,
                    )
                else:
                    self.trigger_weight_sync(
                        any_loaded_replica,
                        rollout_status_manager,
                        step,
                        self.total_steps,
                    )
            # Trigger/finalize only after every policy status is normalized;
            # otherwise a command published mid-ACK can be overwritten READY.
            if self.job_phase == JobPhase.DRAINING:
                self.finish_draining_phase(rollout_status_manager)
            else:
                self.try_trigger_data_fetch_and_training()
            if self.config.train.train_policy.on_policy:
                # Reset on-policy rollout completed flag for next step
                self.on_policy_rollout_completed = False

    def trigger_weight_sync(
        self,
        policy_replica: Replica,
        rollout_status_manager: "RolloutStatusManager",
        current_step: int,
        total_steps: int,
    ):
        valid_rollout_replicas = self._weight_sync_rollout_targets(
            rollout_status_manager
        )
        any_loaded_rollout_replica = (
            valid_rollout_replicas[0] if valid_rollout_replicas else None
        )
        if any_loaded_rollout_replica is None:
            return
        command.PolicyToRolloutUnicastCommand.trigger(
            src_replica=policy_replica,
            dst_replica=any_loaded_rollout_replica,
            src_replica_size=self.policy_atoms_in_replica,
            dst_replica_size=rollout_status_manager.rollout_atoms_in_replica,
            weight_step=current_step,
            total_steps=total_steps,
            redis_handler=self.redis_handler,
        )

        command.RolloutToRolloutBroadcastCommand.trigger(
            src_replica=any_loaded_rollout_replica,
            dst_replicas=valid_rollout_replicas,
            weight_step=current_step,
            total_steps=total_steps,
            redis_handler=self.redis_handler,
        )

        # Weight-sync coalescing: record this as the latest staged round.  It
        # counts as "in flight" until the rollouts adopt it
        # (_weight_max_adopted_version catches up), gating the next round.
        self._weight_last_staged_step = current_step

    def _weight_sync_forced(self, step: int, total_steps: int) -> bool:
        """The one round that must never be coalesced away: a validation-trigger
        step, where the rollout needs that exact weight version to validate.

        Nothing else needs forcing -- the first sync issues naturally (the gate
        comparison is false while ``last_staged_step`` is -1), and the staleness
        ceiling is enforced independently by ``filter_outdated_rollouts``.
        """
        val = self.config.validation
        if getattr(val, "enable", False):
            freq = getattr(val, "freq", 0)
            if freq and step % freq == 0:
                return True
            if step == total_steps:
                return True
        return False

    def rollouts_enough_for_one_step(self) -> bool:
        """
        Check if the rollouts are enough.
        """
        if self.config.mode == "colocated":
            # Colocated mode always has enough rollouts since they are locally prepared.
            return True

        if self.config.train.train_policy.data_dispatch_as_rank_in_mesh:
            if not self.rollout_buffer_per_rank:
                return False
            # In this dispatch mode, each rank has its own rollout buffer.
            return all(
                q.qsize() >= self.config.train.train_batch_per_replica
                for q in self.rollout_buffer_per_rank
            )

        return self.total_pending_rollouts() >= (
            self.config.train.train_batch_per_replica
            * len(self.get_all_atoms_arrived_replicas())
        )

    def check_checkpoint_saving(self, required_rollouts: int):
        # Decide whether to save checkpoint
        # First check if we need to save checkpoint based on epoch
        do_save = False
        if self.current_step == self.training_horizon():
            # Always save checkpoint at the last step
            do_save = True
        elif self.config.train.ckpt.save_freq_in_epoch > 0:
            # Checkpointing based on epoch if `save_freq_in_epoch` is set
            if (
                self.remain_samples_num + required_rollouts - 1
            ) // self.samples_per_epoch != (
                self.remain_samples_num - 1
            ) // self.samples_per_epoch:
                # New epoch begins and old epoch ends
                # So check the epoch number against save_freq_in_epoch for saving checkpoint
                epoch = (
                    self.config.train.epoch
                    - (self.remain_samples_num + required_rollouts - 1)
                    // self.samples_per_epoch
                )
                do_save = epoch % self.config.train.ckpt.save_freq_in_epoch == 0
                if do_save:
                    logger.info(
                        f"[Controller] Epoch {epoch} ends, triggering checkpoint saving at step {self.current_step}"
                    )
        else:
            # Checkpointing based on step if `save_freq_in_epoch` is not set
            do_save = (
                self.current_step % self.config.train.ckpt.save_freq == 0
                and self.current_step > 0
            )
        # Finally check if checkpointing is enabled
        # Only `do_save` when checkpointing is enabled
        return do_save and self.config.train.ckpt.enable_checkpoint

    def try_trigger_data_fetch_and_training(self):
        # If the validation dataloader is activated, do not trigger data fetch and training
        if self.data_fetcher.activated_val_iter is not None:
            return

        arrived_replicas = self.get_all_atoms_arrived_replicas()
        # no replicas arrived, do nothing
        if len(arrived_replicas) == 0:
            return

        if self.training_finished():
            return

        training_horizon = self.training_horizon()

        items_count = self.config.train.train_batch_per_replica
        required_rollouts = items_count * len(arrived_replicas)
        all_ready_or_reduced = (
            self.all_ready_or_reduced() and self.rollouts_enough_for_one_step()
        )

        if all_ready_or_reduced:
            rollouts_of_this_step: List[Rollout] = []
            # Decrease the consumed rollouts number.
            self.remain_samples_num -= required_rollouts

            # From controller's perspective, the training step is already increased
            self.current_step += 1

            # Record the count echoed by this real command's eventual ACK set.
            self.dispatched_rollouts_by_step[self.current_step] = required_rollouts

            if self.config.validation.enable and (
                self.current_step % self.config.validation.freq == 0
                or self.current_step == training_horizon
            ):
                self.data_fetcher.validation_activate_dataloader(self.current_step)

            # FIXME: (lms) will this dipatch style cause non-alignment with VeRL?
            # This dispatch style will cause rollouts from same prompt may be dispatched to different replicas.
            # Interleave-style data dispatch
            if not self.config.mode == "colocated":
                # Colocated mode no need real rollout dispatching since they are all local.
                if self.config.train.train_policy.data_dispatch_as_rank_in_mesh:
                    # Helper function to sort a queue by item.prompt_idx
                    def sort_queue_by_prompt_idx(q):
                        # Step 1: Extract all items
                        items: List[Rollout] = []
                        while not q.empty():
                            items.append(q.get())

                        # Step 2: Sort by prompt_idx
                        items.sort(key=lambda item: item.prompt_idx)

                        # Step 3: Put sorted items back
                        for item in items:
                            q.put(item)

                    sorted_valid_replicas = sorted(
                        arrived_replicas, key=lambda x: x.start_time
                    )
                    for index, replica in enumerate(sorted_valid_replicas):
                        sort_queue_by_prompt_idx(self.rollout_buffer_per_rank[index])
                        for _ in range(items_count):
                            rollout = self.rollout_buffer_per_rank[index].get()
                            replica.put_rollout(rollout, self.redis_handler)
                            rollouts_of_this_step.append(rollout)
                else:
                    for _ in range(items_count):
                        for replica in arrived_replicas:
                            rollout = self.rollout_buffer.get()
                            replica.put_rollout(rollout, self.redis_handler)
                            rollouts_of_this_step.append(rollout)

            # Decide whether to save checkpoint
            do_save = self.check_checkpoint_saving(required_rollouts)

            for replica in arrived_replicas:
                command.DataFetchCommand.trigger(
                    replica=replica,
                    items_count=items_count,
                    global_step=self.current_step,
                    total_steps=training_horizon,
                    # `remain_samples_num` is just for checkpointing the training progress
                    remain_samples_num=self.remain_samples_num,
                    # do_save from `check_checkpoint_saving` indicates whether the replica should save checkpoint after this training step
                    do_save=do_save,
                    redis_handler=self.redis_handler,
                )
                self.set_status(replica.name, PolicyStatus.RUNNING)

            # Report the reward, length, etc.
            # These properties are already ready to be reported before being trained
            if self.config.logging.logger and rollouts_of_this_step:
                rewards = []
                completion_lengths = []
                advantages = []
                filter_rewards = []
                for rollout in rollouts_of_this_step:
                    rewards.append(rollout.reward)
                    completion_length = (
                        (
                            len(rollout.completion_token_ids)
                            if self.config.train.train_policy.rollout_as_token_ids
                            else len(self.tokenizer.encode(rollout.completion))
                        )
                        if not self.config.train.non_text
                        else 1
                    )
                    advantages.extend([rollout.advantage] * completion_length)
                    filter_rewards.append(rollout.filter_reward)
                    completion_lengths.append(completion_length)
                report_data = {
                    "train/reward_mean": np.mean(rewards),
                    "train/reward_std": np.std(rewards),
                    "train/reward_max": np.max(rewards),
                    "train/reward_min": np.min(rewards),
                    "rollout/completion_length_mean": np.mean(completion_lengths),
                    "rollout/completion_length_std": np.std(completion_lengths),
                    "rollout/completion_length_max": np.max(completion_lengths),
                    "rollout/completion_length_min": np.min(completion_lengths),
                    "rollout/advantage_mean": np.mean(advantages),
                    "rollout/advantage_std": np.std(advantages),
                    "rollout/advantage_max": np.max(advantages),
                    "rollout/advantage_min": np.min(advantages),
                    "rollout/filter_reward_mean": np.mean(filter_rewards),
                    "rollout/filter_reward_std": np.std(filter_rewards),
                    "rollout/filter_reward_max": np.max(filter_rewards),
                    "rollout/filter_reward_min": np.min(filter_rewards),
                }

                report_data_list = [
                    rollout.report_metrics if rollout.report_metrics is not None else {}
                    for rollout in rollouts_of_this_step
                ]
                report_data = aggregate_report_data(
                    report_data_list, report_data, prefix="train/"
                )
                self.train_report_data[self.current_step] = report_data


class RolloutStatusManager:
    """
    A class to manage the status of rollout replicas.
    """

    rollout_replicas: Dict[str, Replica]
    rollout_init_done: bool
    replica_scaling_log: List[ReplicaScalingLog]

    def __init__(self):
        self.rollout_replicas = {}
        self.rollout_init_done = False
        self.replica_scaling_log = []
        self._ended_reporters: Dict[str, set[int]] = {}
        self._command_participant_ended_replicas: set[str] = set()

    def setup(
        self,
        config: Config,
        redis_handler: RedisStreamHandler,
        policy_status_manager: PolicyStatusManager,
        data_fetcher: ControllerDataFetcher,
    ):
        self.redis_handler = redis_handler
        self.config = config
        # Rollout status manager has to access some information throug policy status manager.
        self.policy_status_manager = policy_status_manager
        # Data fetcher is needed to set global mesh size when rebuilding mesh for replica specific dispatch.
        self.data_fetcher = data_fetcher
        """
        Maintain the life status of the policy and rollout replicas.
        """
        return len(self.rollout_replicas)

    def n_atoms_per_replica(self) -> int:
        """
        Get the number of GPUs per replica.
        """
        if len(self.rollout_replicas) == 0:
            return 0
        return next(iter(self.rollout_replicas.values())).n_atoms_per_replica()

    def __len__(self) -> int:
        """
        Get the number of rollout replicas.
        """
        return len(self.rollout_replicas)

    def __iter__(self) -> Iterator[Replica]:
        """
        Iterate over the policy replicas.
        """
        for replica in sorted(self.rollout_replicas.values(), key=lambda x: x.name):
            yield replica

    def __contains__(self, replica_name: str) -> bool:
        """
        Check if the replica is in the status manager.
        """
        return replica_name in self.rollout_replicas

    def __getitem__(self, replica_name: str) -> Replica:
        """
        Get the replica from the status manager.
        """
        return self.rollout_replicas.get(replica_name)

    def maintain_life_status(self, policy_status_manager: PolicyStatusManager):
        """
        Maintain the life status of the rollout.
        """
        now = time.time()
        dead_replicas = set()
        for replica in self:
            if now - replica.status.heartbeat_timestamp > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Rollout {replica.name} is dead")
                dead_replicas.add(replica.name)
        for replica_name in dead_replicas:
            self.unregister(replica_name, policy_status_manager=policy_status_manager)

    def heartbeat(self, replica_name: str):
        timestamp: int = int(time.time())
        if replica_name not in self:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in both policy and rollout."
            )
            return
        self[replica_name].status.heartbeat_timestamp = timestamp

    ############################################################
    # utility functions
    ############################################################
    def get_all_atoms_arrived_replicas(self) -> List[Replica]:
        """
        Get all the replicas that have all atoms arrived.
        """
        return [
            replica
            for replica in self.rollout_replicas.values()
            if replica.all_atoms_arrived
        ]

    def unregister(self, replica_name: str, policy_status_manager: PolicyStatusManager):
        """
        Unregister the replica from the status manager.
        """
        assert replica_name in self, (
            f"Replica {replica_name} not found in policy status manager"
        )

        replica = self.rollout_replicas.pop(replica_name)
        self._ended_reporters.pop(replica_name, None)
        self._command_participant_ended_replicas.discard(replica_name)
        self.replica_scaling_log.append(ReplicaScalingLog.down(replica))
        if policy_status_manager.training_finished():
            # This policy replica is normally finished
            # Do not trigger rebuild mesh since everything is gonna be finished shortly
            logger.info(f"[Controller] Replica {replica_name} is stopping.")
            return

        # Workers that promise to remain in their command loop until STOP are
        # safe mesh members. Legacy rankless-ended workers make no such
        # guarantee and must remain excluded.
        safe_survivors = [
            survivor
            for survivor in self.get_all_atoms_arrived_replicas()
            if not survivor.status.ended
            or survivor.name in self._command_participant_ended_replicas
        ]
        if replica.in_mesh and safe_survivors:
            # A one-member rebuild is still required: async R2R must replace
            # the departed replica's communicator before version bookkeeping.
            self.trigger_rebuild_mesh(safe_survivors)
        elif replica.in_mesh:
            logger.info(
                "[Controller] Replica %s unregistering with no safe rollout "
                "mesh survivors; skipping rebuild.",
                replica_name,
            )

    def register(
        self,
        atom: Atom,
        config: Config,
        policy_status_manager: PolicyStatusManager,
        **kwargs,
    ):
        """
        Register the atom to the status manager.
        """
        replica = self[atom.replica_name]
        if replica is None:
            replica = Replica(atom.replica_name, Role.ROLLOUT, [atom])
            self.rollout_replicas[atom.replica_name] = replica
        else:
            replica.arrive(atom)
        atom.bind_replica(replica)

        # post register hook
        if not self.rollout_init_done:
            if len(self.rollout_replicas) > config.rollout.parallelism.n_init_replicas:
                config.rollout.parallelism.n_init_replicas = len(self.rollout_replicas)
                logger.info(
                    f"[Controller] Update rollout n_init_replicas to {config.rollout.parallelism.n_init_replicas} replicas"
                )

        # Check if all atoms of the replica have arrived
        if replica.all_atoms_arrived:
            if replica.start_time == -1:
                replica.start_time = int(time.time())
            logger.info(
                f"[Controller] All atoms of {Role.ROLLOUT} Replica {replica.name} has been set."
            )
            # Check total valid rollout replicas
            valid_replicas = []
            if not hasattr(self, "rollout_atoms_in_replica"):
                self.rollout_atoms_in_replica = int(math.prod(atom.group_size))
            for replica in self.rollout_replicas.values():
                if replica.all_atoms_arrived:
                    valid_replicas.append(replica)
            self.post_register_hook(
                valid_replicas,
                atom.replica,
                config,
                policy_status_manager,
            )
        return replica

    @staticmethod
    def _expected_reporting_ranks(replica: Replica) -> set[int]:
        return {
            atom.global_rank
            for atom in replica.atoms.values()
            if atom.tp_rank() == 0
            and atom.pp_rank() == atom.group_size[MESH_NAMES.index("pp")] - 1
        }

    def get_safe_weight_sync_replicas(
        self, *, validation_enabled: bool
    ) -> List[Replica]:
        """Return an R2R recipient set that matches a valid communicator."""
        replicas = sorted(
            self.get_all_atoms_arrived_replicas(), key=lambda item: item.start_time
        )
        if validation_enabled:
            return replicas
        if any(
            replica.status.ended
            and replica.name not in self._command_participant_ended_replicas
            for replica in replicas
        ):
            # A rankless legacy checkout does not promise to keep consuming
            # commands. Publishing to it or to a subset of its communicator
            # can deadlock, so wait for unregister/rebuild.
            return []
        # Command-participating ended workers remain available through STOP.
        # Keep the complete registered topology; an R2R subset would use a
        # stale all-replica communicator.
        return replicas

    def rollout_end(
        self,
        replica_name: str,
        src_global_rank: Optional[int] = None,
        stays_command_participant: bool = False,
    ) -> bool:
        """
        Record a local reporter checkout and return a whole-replica transition.

        Rankless requests retain the legacy whole-replica meaning. They may
        explicitly promise to remain command participants through STOP.
        """
        replica = self[replica_name]
        if replica is None:
            logger.warning(
                f"[Controller] Rollout {replica_name} not found in RolloutStatusManager"
            )
            return False
        if replica.status.ended:
            return False
        if src_global_rank is None:
            if stays_command_participant:
                self._command_participant_ended_replicas.add(replica_name)
            else:
                self._command_participant_ended_replicas.discard(replica_name)
            replica.status.ended = True
            return True

        expected = self._expected_reporting_ranks(replica)
        if src_global_rank not in expected:
            logger.warning(
                "[Controller] Ignoring rollout end from unexpected rank %s "
                "for %s; expected one of %s",
                src_global_rank,
                replica_name,
                sorted(expected),
            )
            return False
        reporters = self._ended_reporters.setdefault(replica_name, set())
        reporters.add(src_global_rank)
        if expected and expected.issubset(reporters):
            self._command_participant_ended_replicas.add(replica_name)
            replica.status.ended = True
            return True
        return False

    def all_rollouts_ended(self) -> bool:
        """
        Check if all rollouts have ended.
        """
        return len(self.rollout_replicas) > 0 and all(
            [replica.status.ended for replica in self.rollout_replicas.values()]
        )

    def trigger_rebuild_mesh(
        self,
        valid_replicas: List[Replica],
    ):
        sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)
        command.BuildMeshCommand.trigger(
            sorted_valid_replicas, redis_handler=self.redis_handler
        )
        self.data_fetcher.set_rollout_global_mesh_size(len(sorted_valid_replicas))

    def post_register_hook(
        self,
        valid_replicas: List[Replica],
        target_replica: Replica,
        config: Config,
        policy_status_manager: PolicyStatusManager,
    ):
        assert target_replica in valid_replicas
        any_loaded_policy_replica = None
        sorted_valid_policy_replicas = sorted(
            [r for r in policy_status_manager], key=lambda x: x.start_time
        )
        for replica in sorted_valid_policy_replicas:
            if replica.weights_loaded_in_view_of_command:
                # We will select the first replica that has weights loaded in view of command
                # to broadcast weights
                any_loaded_policy_replica = replica
                break

        # First P->R Unicast if the policy is ready and all rollout replicas are not ready
        if (
            all(
                [
                    not replica.weights_loaded_in_view_of_command
                    for replica in valid_replicas
                ]
            )
            and any_loaded_policy_replica is not None
        ):
            command.PolicyToRolloutUnicastCommand.trigger(
                src_replica=any_loaded_policy_replica,
                dst_replica=target_replica,
                src_replica_size=policy_status_manager.policy_atoms_in_replica,
                dst_replica_size=self.rollout_atoms_in_replica,
                weight_step=None,
                total_steps=None,
                redis_handler=self.redis_handler,
            )
            logger.info(
                f"[Controller] Trigger PolicyToRolloutUnicastCommand to {target_replica.name} via Rollout registration"
            )
        else:
            logger.info(
                "[Controller] No valid policy replicas found in Rollout registration or some rollout already get weight from policy, skip PolicyToRolloutUnicastCommand"
            )

        was_already_initialized = self.rollout_init_done

        if (
            not was_already_initialized
            and len(valid_replicas) == config.rollout.parallelism.n_init_replicas
        ):
            self.rollout_init_done = True
            self.trigger_rebuild_mesh(valid_replicas)

            # ONLY ONCE PER LIFE CYCLE
            # Trigger RolloutToRolloutBroadcastCommand only once after all initial rollout replicas are loaded
            any_loaded_rollout_replica = None
            sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)
            for replica in sorted_valid_replicas:
                if (
                    replica.weights_loaded_in_view_of_command
                    and replica in valid_replicas
                ):
                    # We will select the first replica that has weights loaded in view of command
                    # to broadcast weights
                    any_loaded_rollout_replica = replica
                    break
            if any_loaded_rollout_replica is not None:
                command.RolloutToRolloutBroadcastCommand.trigger(
                    src_replica=any_loaded_rollout_replica,
                    dst_replicas=valid_replicas,
                    weight_step=self.policy_status_manager.current_step,  # we must pass the current step to rollout replicas to track the weight version even in resume ckpt.
                    total_steps=None,
                    redis_handler=self.redis_handler,
                )
        elif not self.rollout_init_done:
            assert len(valid_replicas) < config.rollout.parallelism.n_init_replicas
            logger.info(
                f"Waiting for {config.rollout.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )
        else:
            # Dynamic mesh building, no matter what the length of valid_replicas is,
            # we will always trigger mesh building if there are more than one rollout replicas
            self.trigger_rebuild_mesh(valid_replicas)
