# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-owned admission transaction; no parallel sample accounting."""

from cosmos_rl.dispatcher.data.admission import (
    CompletionDisposition,
    AdmissionCommitted,
    SourceWindow,
)
from cosmos_rl.reward.admission import COMPLETION_ADMISSION_METRIC_PREFIX
from cosmos_rl.utils.application_callbacks import CallbackContext, CallbackRegistry
from cosmos_rl.utils.logging import logger


class CompletionAdmissionState:
    def __init__(self, *, window_size=4096, hook_fns=None):
        self.window_size = window_size
        self.sources = {}
        self.failed = False
        self.reason_labels = set()
        self.callbacks = None
        if hook_fns is not None:
            self._configure_callbacks(hook_fns)

    def _configure_callbacks(self, hook_fns):
        hooks = dict(hook_fns or {})
        if hooks.get("completion_settled") is not None:
            if hooks.get("admission_committed") is not None:
                raise ValueError(
                    "Use admission_committed without deprecated completion_settled"
                )
            logger.warning("completion_settled is deprecated; use admission_committed")
            hooks["admission_committed"] = hooks.pop("completion_settled")
        self.callbacks = CallbackRegistry(hooks, CallbackContext(role="controller"))
        self.callbacks.validate("admission_committed", borrowed=False)

    def close(self):
        if self.callbacks is not None:
            self.callbacks.close()

    def prepare(self, controller, request, rollouts):
        """Validate and evaluate everything before the first accounting mutation.

        Called under the controller lifecycle lock. Quality selection must
        already have happened on the producer, before advantage computation.
        """
        if self.failed:
            raise RuntimeError(
                "Admission settlement failed; controller restart required"
            )
        status = controller.policy_status_manager
        sources = controller.rollout_status_manager
        # Keep state only for live replica objects, bounding churn as well as
        # the sequence window. Reusing a replica name for a new incarnation is
        # prohibited by the wire contract; restart must use a fresh replica ID.
        self.sources = {
            key: value
            for key, value in self.sources.items()
            if sources[key[0]] is value[0]
        }
        replica = sources[request.src_replica_name]
        if replica is None or replica.status.ended:
            raise ValueError("Admission report source is not a live replica")
        rank = request.src_global_rank
        if type(rank) is not int or not 0 <= rank < replica.n_atoms_per_replica():
            raise ValueError("Admission report requires a valid source rank")
        identities = request.completion_identities
        if identities is None or len(identities) != len(rollouts):
            raise ValueError("Admission requires one identity per completion")
        if request.completion_diagnostics is not None and len(
            request.completion_diagnostics
        ) != len(identities):
            raise ValueError(
                "Admission requires diagnostics to match completion identities"
            )
        if "discarded_samples" in request.metrics or any(
            key.startswith(COMPLETION_ADMISSION_METRIC_PREFIX)
            for key in request.metrics
        ):
            raise ValueError(
                "Identified admission cannot mix legacy settlement metrics"
            )
        all_ids = identities + [
            failure.identity for failure in request.completion_failures
        ]
        if any(identity.weight_version > status.current_step for identity in all_ids):
            raise ValueError("Completion originates from a future weight version")
        if any(
            identity.weight_version != rollout.weight_version
            for identity, rollout in zip(identities, rollouts)
        ):
            raise ValueError("Completion identity and payload weight version disagree")
        if any(
            failure.payload is not None
            and failure.payload.weight_version != failure.identity.weight_version
            for failure in request.completion_failures
        ):
            raise ValueError("Rejected payload and identity weight version disagree")
        key = (request.src_replica_name, rank)
        _, window = self.sources.setdefault(
            key, (replica, SourceWindow(self.window_size))
        )
        unseen = window.unseen(all_ids)
        decisions = []
        late_payloads = []
        closed = status.rollout_admission_closed()
        for index, (identity, rollout) in enumerate(zip(identities, rollouts)):
            if not unseen[index]:
                if identity.sequence in window.failed_without_payload:
                    late_payloads.append((identity, rollout))
                continue
            if closed:
                disposition = CompletionDisposition(outcome="rejected", reason="closed")
            else:
                # Never perform per-completion quality filtering here: the
                # received advantages describe the producer's selected group.
                disposition = CompletionDisposition(outcome="accepted")
            decisions.append((identity, rollout, disposition))
        for index, failure in enumerate(
            request.completion_failures, start=len(identities)
        ):
            if (
                not unseen[index]
                and failure.payload is not None
                and failure.identity.sequence in window.failed_without_payload
            ):
                late_payloads.append((failure.identity, failure.payload))
            if unseen[index]:
                decisions.append(
                    (
                        failure.identity,
                        failure.payload,
                        CompletionDisposition(
                            outcome="rejected", reason=failure.reason
                        ),
                    )
                )
        if len(decisions) > status.samples_on_the_fly:
            raise ValueError(
                "Admission report exceeds outstanding completion reservations"
            )
        return window, all_ids, decisions, closed, late_payloads

    def settle(
        self,
        controller,
        request,
        plan,
        *,
        requested_versions=None,
        requested_policy_ranks=None,
    ):
        """Use existing cleanup, settlement and versioned refill mechanisms.

        An infrastructure exception after mutations poisons this instance;
        retries cannot turn a partial settlement into apparent success.
        """
        window, identities, decisions, closed, late_payloads = plan
        status = controller.policy_status_manager
        accepted = []
        try:
            for identity, rollout in late_payloads:
                status._publish_payload_transport_cleanup([rollout], [])
                window.failed_without_payload.remove(identity.sequence)
            for identity, rollout, disposition in decisions:
                if disposition.outcome == "accepted":
                    accepted.append(rollout)
                    continue
                if rollout is not None:
                    status._publish_payload_transport_cleanup([rollout], [])
                else:
                    window.failed_without_payload.add(identity.sequence)
                report_id = (
                    f"admission:{request.src_replica_name}:"
                    f"{request.src_global_rank}:{identity.sequence}"
                )
                if closed:
                    status._settle_samples_on_the_fly(1, "terminal_buffer_cleanup")
                else:
                    status.settle_discarded_samples(
                        source_replica=request.src_replica_name,
                        report_id=report_id,
                        count=1,
                        weight_version=(
                            identity.weight_version
                            if requested_versions is None
                            else requested_versions[
                                (
                                    identity.reservation.work_id,
                                    identity.reservation.slot,
                                )
                            ]
                        ),
                        policy_rank=(
                            requested_policy_ranks.get(
                                (
                                    identity.reservation.work_id,
                                    identity.reservation.slot,
                                )
                            )
                            if requested_policy_ranks is not None
                            else None
                        ),
                    )
                # Bounded metric cardinality: application diagnostics remain
                # on the disposition; aggregate terminal admission separately.
                status.filter_records["application_rejected"] = (
                    status.filter_records.get("application_rejected", 0) + 1
                )
                reason = disposition.reason
                if reason not in self.reason_labels:
                    if len(self.reason_labels) >= 64:
                        reason = "other"
                    else:
                        self.reason_labels.add(reason)
                metric = f"application_rejected/{reason}"
                status.filter_records[metric] = status.filter_records.get(metric, 0) + 1
            window.commit(identities)
            if self.callbacks is None:
                self._configure_callbacks(getattr(status, "hook_fns", {}))
            if self.callbacks.has("admission_committed") and decisions:
                diagnostics = {
                    identity.sequence: diagnostic
                    for identity, diagnostic in zip(
                        request.completion_identities,
                        request.completion_diagnostics
                        or [None] * len(request.completion_identities),
                    )
                }
                diagnostics.update(
                    (failure.identity.sequence, failure.diagnostics)
                    for failure in request.completion_failures
                )
                self.callbacks.invoke(
                    "admission_committed",
                    tuple(
                        AdmissionCommitted(
                            source_replica=request.src_replica_name,
                            source_rank=request.src_global_rank,
                            identity=identity,
                            disposition=disposition,
                            # Telemetry workers can retain only detached small
                            # metadata, independent of this request and payload.
                            diagnostics=(
                                diagnostics[identity.sequence].model_copy(deep=True)
                                if diagnostics.get(identity.sequence) is not None
                                else None
                            ),
                        )
                        for identity, _, disposition in decisions
                    ),
                    borrowed=False,
                )
            return accepted
        except BaseException:
            self.failed = True
            raise
