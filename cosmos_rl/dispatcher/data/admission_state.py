# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-owned admission transaction; no parallel sample accounting."""

from cosmos_rl.dispatcher.data.admission import (
    CompletionContext,
    CompletionDisposition,
    SourceWindow,
)
from cosmos_rl.reward.admission import COMPLETION_ADMISSION_METRIC_PREFIX


class CompletionAdmissionState:
    def __init__(self, adapter, *, window_size=4096):
        self.adapter = adapter
        self.window_size = window_size
        self.sources = {}
        self.failed = False
        self.reason_labels = set()

    def prepare(self, controller, request, rollouts):
        """Validate and evaluate everything before the first accounting mutation.

        Called under the controller lifecycle lock. The synchronous adapter
        cannot interleave with another ingestion/registration coroutine.
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
                disposition = self.adapter.admit(
                    CompletionContext(request.src_replica_name, rank, identity),
                    rollout.model_copy(deep=True),
                )
                if not isinstance(disposition, CompletionDisposition):
                    raise TypeError(
                        "Admission adapter must return CompletionDisposition"
                    )
            decisions.append((identity, rollout, disposition))
        for index, failure in enumerate(
            request.completion_failures, start=len(identities)
        ):
            if unseen[index]:
                decisions.append(
                    (
                        failure.identity,
                        None,
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

    def settle(self, controller, request, plan):
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
                        weight_version=identity.weight_version,
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
            return accepted
        except BaseException:
            self.failed = True
            raise
