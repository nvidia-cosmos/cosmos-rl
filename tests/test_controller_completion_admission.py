# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from types import SimpleNamespace
from unittest.mock import Mock
from pydantic import ValidationError

from cosmos_rl.dispatcher.data.admission import (
    CompletionDisposition,
    CompletionIdentity,
    SourceWindow,
    CompletionFailure,
)
from cosmos_rl.dispatcher.data.admission_state import CompletionAdmissionState
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.dispatcher.status import PolicyStatusManager


def identity(sequence, version=0):
    return CompletionIdentity(sequence=sequence, weight_version=version)


def test_out_of_order_and_duplicate_reports():
    window = SourceWindow(capacity=4)
    window.commit([identity(2)])
    assert window.unseen([identity(1), identity(2)]) == [True, False]
    window.commit([identity(1), identity(2)])
    assert window.unseen([identity(1)]) == [False]


def test_eviction_never_allows_repeated_settlement():
    window = SourceWindow(capacity=4)
    for sequence in range(100):
        window.commit([identity(sequence)])
        assert len(window.versions) <= 4
    with pytest.raises(ValueError, match="expired"):
        window.unseen([identity(0)])
    with pytest.raises(ValueError, match="expired"):
        window.unseen([identity(95)])


def test_origin_version_cannot_change_on_retry():
    window = SourceWindow()
    window.commit([identity(0, 2)])
    with pytest.raises(ValueError, match="originating version"):
        window.unseen([identity(0, 3)])


@pytest.mark.parametrize("sequence", [-1, True, "1", 1.5])
def test_malformed_identity(sequence):
    with pytest.raises(ValidationError):
        identity(sequence)


def test_reasons_are_diagnostic_not_identity():
    assert CompletionDisposition(outcome="accepted").reason is None
    assert (
        CompletionDisposition(outcome="rejected", reason="quality").reason == "quality"
    )
    with pytest.raises(ValidationError):
        CompletionDisposition(outcome="rejected")
    with pytest.raises(ValidationError):
        CompletionDisposition(outcome="accepted", reason="quality")


def test_report_validation_is_atomic():
    window = SourceWindow(capacity=4)
    for report in ([identity(0), identity(0)], [identity(0), identity(4)]):
        with pytest.raises(ValueError):
            window.commit(report)
        assert window.highest == -1
        assert window.versions == {}


def harness(closed=False):
    status = PolicyStatusManager()
    status.current_step = 2
    status.samples_on_the_fly = 10
    status.rollout_admission_closed = lambda: closed
    status._publish_payload_transport_cleanup = Mock()
    refill = Mock()
    status.set_discard_refill_hook(refill)
    controller = SimpleNamespace(
        policy_status_manager=status,
        rollout_status_manager={
            "source": SimpleNamespace(
                n_atoms_per_replica=lambda: 1, status=SimpleNamespace(ended=False)
            )
        },
    )
    return controller, CompletionAdmissionState(), refill


def report(sequences, version=2, failures=()):
    return RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        payloads=[],
        completion_identities=[identity(sequence, version) for sequence in sequences],
        completion_failures=list(failures),
    )


def apply(state, controller, request):
    rollouts = [
        Rollout(
            completion=f"payload-{item.sequence}", weight_version=item.weight_version
        )
        for item in request.completion_identities
    ]
    return state.settle(
        controller, request, state.prepare(controller, request, rollouts)
    )


def test_partial_rejection_uses_existing_versioned_settlement_once():
    controller, state, refill = harness()
    request = report(
        [1], failures=[CompletionFailure(identity=identity(0, 2), reason="quality")]
    )
    assert len(apply(state, controller, request)) == 1
    status = controller.policy_status_manager
    assert status.samples_on_the_fly == 9
    assert status.filter_records["application_rejected/quality"] == 1
    # No rejected payload was transferred; its resources stay producer-owned.
    status._publish_payload_transport_cleanup.assert_not_called()
    refill.assert_called_once()
    assert apply(state, controller, request) == []
    assert status.samples_on_the_fly == 9
    refill.assert_called_once()


def test_malformed_failure_does_not_settle_any_of_the_report():
    controller, state, refill = harness()
    request = report(
        [0], failures=[CompletionFailure(identity=identity(0, 2), reason="quality")]
    )
    with pytest.raises(ValueError, match="duplicate identities"):
        apply(state, controller, request)
    assert controller.policy_status_manager.samples_on_the_fly == 10
    controller.policy_status_manager._publish_payload_transport_cleanup.assert_not_called()
    refill.assert_not_called()
    assert not state.failed


def test_producer_failure_and_late_result_share_one_terminal_identity():
    controller, state, refill = harness()
    failure = CompletionFailure(identity=identity(0, 2), reason="producer_failed")
    apply(state, controller, report([], failures=[failure]))
    assert apply(state, controller, report([0])) == []
    assert controller.policy_status_manager.samples_on_the_fly == 9
    refill.assert_called_once()
    controller.policy_status_manager._publish_payload_transport_cleanup.assert_called_once()
    assert apply(state, controller, report([0])) == []
    controller.policy_status_manager._publish_payload_transport_cleanup.assert_called_once()


def test_quality_rejection_transfers_cleanup_once_without_training():
    controller, state, refill = harness()
    rejected = Rollout(completion="rejected-reference", weight_version=2)
    request = report(
        [],
        failures=[
            CompletionFailure(
                identity=identity(0, 2), reason="quality", payload=rejected
            )
        ],
    )
    for _ in range(2):
        assert apply(state, controller, request) == []
    controller.policy_status_manager._publish_payload_transport_cleanup.assert_called_once_with(
        [rejected], []
    )
    refill.assert_called_once()


def test_failure_then_rejected_payload_cleans_up_without_resettling():
    controller, state, refill = harness()
    failure = CompletionFailure(identity=identity(0, 2), reason="quality")
    apply(state, controller, report([], failures=[failure]))
    failure = failure.model_copy(
        update={"payload": Rollout(completion="late-reference", weight_version=2)}
    )
    for _ in range(2):
        assert apply(state, controller, report([], failures=[failure])) == []
    controller.policy_status_manager._publish_payload_transport_cleanup.assert_called_once()
    refill.assert_called_once()


def test_rejected_payload_version_mismatch_is_atomic():
    controller, state, refill = harness()
    request = report(
        [],
        failures=[
            CompletionFailure(
                identity=identity(0, 2),
                reason="quality",
                payload=Rollout(weight_version=1),
            )
        ],
    )
    with pytest.raises(ValueError, match="weight version disagree"):
        apply(state, controller, request)
    assert controller.policy_status_manager.samples_on_the_fly == 10
    refill.assert_not_called()


def test_closed_admission_discards_without_refill():
    controller, state, refill = harness(closed=True)
    assert apply(state, controller, report([0, 1])) == []
    assert controller.policy_status_manager.samples_on_the_fly == 8
    refill.assert_not_called()
    assert apply(state, controller, report([0, 1])) == []
    assert (
        controller.policy_status_manager._publish_payload_transport_cleanup.call_count
        == 2
    )


def test_departed_source_cannot_resettle_after_state_is_pruned():
    controller, state, refill = harness()
    apply(state, controller, report([0]))
    # Status manager lookup returns None for a departed replica.
    controller.rollout_status_manager["source"] = None
    with pytest.raises(ValueError, match="live replica"):
        apply(state, controller, report([0]))
    assert state.sources == {}
    refill.assert_not_called()


def test_settlement_failure_poisoning_prevents_false_retry_success():
    controller, state, _ = harness(closed=True)
    controller.policy_status_manager._publish_payload_transport_cleanup.side_effect = (
        RuntimeError("control plane failed")
    )
    with pytest.raises(RuntimeError, match="control plane failed"):
        apply(state, controller, report([0]))
    with pytest.raises(RuntimeError, match="restart required"):
        apply(state, controller, report([0]))


def test_all_rejected_mixed_versions_refill_each_origin_separately():
    controller, state, refill = harness()
    request = report(
        [],
        failures=[
            CompletionFailure(identity=identity(0, 1), reason="quality"),
            CompletionFailure(identity=identity(1, 2), reason="quality"),
        ],
    )
    assert apply(state, controller, request) == []
    assert [call.args[:2] for call in refill.call_args_list] == [(1, 1), (2, 1)]
    assert controller.policy_status_manager.samples_on_the_fly == 8


@pytest.mark.parametrize(
    "fault", ["missing", "future", "rank", "legacy", "reservation", "ended"]
)
def test_invalid_report_has_no_accounting_effect(fault):
    controller, state, refill = harness()
    request = report([0])
    if fault == "missing":
        request.completion_identities = []
        rollouts = [Rollout(weight_version=2)]
    else:
        rollouts = [Rollout(weight_version=2)]
    if fault == "future":
        request.completion_identities = [identity(0, 3)]
    if fault == "rank":
        request.src_global_rank = 2
    if fault == "legacy":
        request.metrics = {"discarded_samples": 1}
    if fault == "reservation":
        controller.policy_status_manager.samples_on_the_fly = 0
    if fault == "ended":
        controller.rollout_status_manager["source"].status.ended = True
    before = controller.policy_status_manager.samples_on_the_fly
    with pytest.raises(ValueError):
        state.prepare(controller, request, rollouts)
    assert controller.policy_status_manager.samples_on_the_fly == before
    refill.assert_not_called()


def test_changed_failure_reason_does_not_change_identity():
    controller, state, refill = harness()
    for reason in ("timeout", "cancelled", "quality"):
        apply(
            state,
            controller,
            report(
                [], failures=[CompletionFailure(identity=identity(0, 2), reason=reason)]
            ),
        )
    assert controller.policy_status_manager.samples_on_the_fly == 9
    refill.assert_called_once()


def test_reason_metric_cardinality_is_bounded():
    controller, state, _ = harness()
    controller.policy_status_manager.samples_on_the_fly = 100
    for sequence in range(100):
        apply(
            state,
            controller,
            report(
                [],
                failures=[
                    CompletionFailure(
                        identity=identity(sequence, 2), reason=f"reason_{sequence}"
                    )
                ],
            ),
        )
    metrics = controller.policy_status_manager.filter_records
    assert (
        len([key for key in metrics if key.startswith("application_rejected/")]) <= 65
    )
    assert (
        sum(
            value
            for key, value in metrics.items()
            if key.startswith("application_rejected/")
        )
        == 100
    )


def test_controller_cannot_reselect_precomputed_advantage_group():
    controller, state, _ = harness()
    accepted = apply(state, controller, report([0]))
    assert accepted[0].completion == "payload-0"
    assert not hasattr(state, "adapter")
