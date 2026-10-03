# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Autonomous SFT receipts belong to the mesh published before training."""

from types import SimpleNamespace
from unittest.mock import Mock
import asyncio

import pytest

from cosmos_rl.dispatcher.protocol import MESH_NAMES, Role
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.status import PolicyStatus, PolicyStatusManager


def atom(name):
    return Atom(
        global_rank=0,
        host_ip="127.0.0.1",
        host_name="host",
        trace_path="",
        ranks=[0] * len(MESH_NAMES),
        group_size=[1] * len(MESH_NAMES),
        replica_name=name,
        report_session_id=f"process-{name}",
    )


def manager(*, publish=True):
    status = PolicyStatusManager()
    status.current_step = 2
    status.total_steps = 10
    status.remain_samples_num = 100
    status.config = SimpleNamespace(
        policy=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=2)),
        validation=SimpleNamespace(enable=True, freq=1),
        train=SimpleNamespace(
            train_batch_per_replica=4,
            train_policy=SimpleNamespace(
                type="sft", data_dispatch_as_rank_in_mesh=False
            ),
        ),
    )
    status.policy_replicas = {
        name: Replica(name, Role.POLICY, [atom(name)]) for name in ("a", "b")
    }
    status.status = {name: PolicyStatus.RUNNING for name in status.policy_replicas}
    status.redis_handler = Mock()
    status.data_fetcher = Mock()
    status.recompute_total_steps = Mock()
    status.sft_report_summary = Mock()
    if publish:
        status.trigger_rebuild_mesh(status.get_all_atoms_arrived_replicas())
    return status


def ack(status, name, step=3, *, validation=False, report=None):
    if report is None:
        report = {"val/avg_loss": 1.0} if validation else {}
    status.train_ack(
        name,
        step,
        10,
        False,
        report,
        Mock(),
        report_session_id=f"process-{name}",
        src_global_rank=0,
    )


def test_membership_is_sealed_before_first_mesh_publication():
    status = manager(publish=False)

    def publish(*_):
        assert status.sft_cohort == status.policy_replicas
        assert not status.sft_ack_groups

    status.redis_handler.publish_command.side_effect = publish
    status.trigger_rebuild_mesh(status.get_all_atoms_arrived_replicas())
    assert status.redis_handler.publish_command.call_count == 2


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("after_first_ack", [False, True])
def test_live_late_join_cannot_change_published_ack_set(validation, after_first_ack):
    status = manager()
    if after_first_ack:
        ack(status, "a", validation=validation)
    status.policy_replicas["late"] = Replica("late", Role.POLICY, [atom("late")])
    ack(status, "a", validation=validation)
    ack(status, "b", validation=validation)
    group = status.sft_ack_groups[(validation, 3)]
    assert group.settled and group.participants == {"a", "b"}
    assert status.remain_samples_num == (100 if validation else 92)


@pytest.mark.parametrize("validation", [False, True])
def test_missing_live_member_does_not_shrink_original_ack_set(validation):
    status = manager()
    status.policy_replicas.pop("b")
    ack(status, "a", validation=validation)
    group = status.sft_ack_groups[(validation, 3)]
    assert not group.settled and group.participants == {"a", "b"}
    assert status.remain_samples_num == 100
    assert not status.training_finished()


@pytest.mark.parametrize("before_first_ack", [False, True])
def test_departure_contains_uncertain_work_without_rebuild(before_first_ack):
    status = manager()
    if not before_first_ack:
        ack(status, "a")
    with pytest.raises(RuntimeError, match="SFT completion is uncertain"):
        status.unregister("b")
    assert status.terminal_error is not None
    assert status.redis_handler.publish_command.call_count == 2
    assert not status.training_finished()
    assert status.remain_samples_num == 100
    with pytest.raises(RuntimeError, match="SFT completion is uncertain"):
        ack(status, "a")
    assert not any(g.settled for g in status.sft_ack_groups.values())


def test_first_final_ack_is_progress_not_completion():
    status = manager()
    ack(status, "a", 10)
    assert status.current_step == 10
    assert not status.training_finished()
    with pytest.raises(RuntimeError, match="before its final ACK"):
        status.unregister("b")
    assert not status.training_finished()


def test_final_ack_allows_healthy_exit_before_peer_final_report():
    status = manager()
    ack(status, "a", 10)
    status.unregister("a")
    assert status.terminal_error is None and not status.training_finished()
    ack(status, "b", 10)
    assert status.training_finished()
    assert status.remain_samples_num == 92
    status.unregister("b")
    assert status.terminal_error is None
    assert status.redis_handler.publish_command.call_count == 2


@pytest.mark.parametrize("same_names", [False, True])
def test_active_mesh_cannot_be_replaced_even_with_same_names(same_names):
    status = manager()
    participants = status.get_all_atoms_arrived_replicas()
    if not same_names:
        participants = participants[:1]
    with pytest.raises(RuntimeError, match="active mesh rebuild"):
        status.trigger_rebuild_mesh(participants)
    assert set(status.sft_cohort) == {"a", "b"}
    assert status.redis_handler.publish_command.call_count == 2
    assert not status.training_finished()


def test_late_registration_rejected_before_mutation_but_retries_preserved():
    status = manager()
    original = status.policy_replicas["a"]
    assert status.register(atom("a"), Mock(), Mock()) is original
    with pytest.raises(RuntimeError, match="SFT membership is sealed"):
        status.register(atom("late"), Mock(), Mock())
    assert "late" not in status.policy_replicas
    assert status.redis_handler.publish_command.call_count == 2
    ack(status, "a")
    ack(status, "b")
    assert status.sft_ack_groups[(False, 3)].settled


def test_name_reuse_cannot_supply_original_participants_ack():
    status = manager()
    status.policy_replicas["a"] = Replica("a", Role.POLICY, [atom("a")])
    with pytest.raises(ValueError, match="original mesh participant"):
        ack(status, "a")
    assert not status.sft_ack_groups
    assert status.remain_samples_num == 100


def test_partial_publication_never_retries_as_new_generation():
    status = manager(publish=False)
    status.redis_handler.publish_command.side_effect = [None, OSError("lost reply")]
    with pytest.raises(OSError, match="lost reply"):
        status.trigger_rebuild_mesh(status.get_all_atoms_arrived_replicas())
    assert status.terminal_error is not None
    with pytest.raises(OSError, match="lost reply"):
        ack(status, "a")
    assert not status.training_finished()


def test_ack_before_mesh_publication_cannot_create_work():
    status = manager(publish=False)
    with pytest.raises(ValueError, match="before mesh publication"):
        ack(status, "a")
    assert not status.sft_ack_groups


@pytest.mark.parametrize("validation", [False, True])
def test_expired_receipt_cannot_recreate_and_resettle_old_work(monkeypatch, validation):
    monkeypatch.setattr("cosmos_rl.dispatcher.status._REPORT_DEDUP_WINDOW", 2)
    status = manager()
    for step in (3, 4, 5):
        ack(status, "a", step, validation=validation)
        ack(status, "b", step, validation=validation)
    before = status.remain_samples_num
    with pytest.raises(ValueError, match="expired report group"):
        ack(status, "a", 3, validation=validation)
    assert status.remain_samples_num == before
    assert (validation, 3) not in status.sft_ack_groups
    assert len(status.sft_ack_groups) == 2


def test_pending_groups_are_bounded_and_never_dropped(monkeypatch):
    monkeypatch.setattr("cosmos_rl.dispatcher.status._REPORT_DEDUP_WINDOW", 2)
    status = manager()
    ack(status, "a", 3)
    ack(status, "a", 4)
    with pytest.raises(RuntimeError, match="too many unsettled"):
        ack(status, "a", 5)
    assert set(status.sft_ack_groups) == {(False, 3), (False, 4)}
    assert status.remain_samples_num == 100


def test_changed_retry_remains_rejected_without_mutating_accounting():
    status = manager()
    ack(status, "a")
    with pytest.raises(ValueError, match="retry changed"):
        ack(status, "a", report={"loss": 1})
    assert status.remain_samples_num == 100
    ack(status, "b")
    assert status.remain_samples_num == 92


def test_final_ack_does_not_hide_older_unsettled_work():
    status = manager()
    ack(status, "a", 3)
    ack(status, "a", 10)
    ack(status, "b", 10)
    assert not status.training_finished()
    ack(status, "b", 3)
    assert status.training_finished()


def test_successful_early_stop_remains_complete_without_final_horizon_ack():
    status = manager()
    ack(status, "a")
    ack(status, "b")
    status.terminal_complete = True  # Existing all-cohort final checkpoint protocol.
    assert status.training_finished()
    status.unregister("a")
    status.unregister("b")
    assert status.terminal_error is None


def test_single_replica_sft_does_not_require_multi_replica_ack_protocol():
    status = manager(publish=False)
    status.config.policy.parallelism.n_init_replicas = 1
    status.policy_replicas.pop("b")
    status.trigger_rebuild_mesh(status.get_all_atoms_arrived_replicas())
    assert status.sft_cohort is None
    status.current_step = 10
    assert status.training_finished()


@pytest.mark.parametrize(
    "session,rank", [(None, 0), ("replacement", 0), ("process-a", 1)]
)
def test_stale_or_wrong_rank_session_cannot_acknowledge_work(session, rank):
    status = manager()
    with pytest.raises(ValueError, match="original mesh participant"):
        status.train_ack(
            "a",
            3,
            10,
            False,
            {},
            Mock(),
            report_session_id=session,
            src_global_rank=rank,
        )
    assert not status.sft_ack_groups
    assert status.remain_samples_num == 100


def test_same_name_replacement_registration_is_not_a_retry():
    status = manager()
    replacement = atom("a")
    replacement.report_session_id = "new-process"
    with pytest.raises(ValueError, match="changed identity"):
        status.register(replacement, status.config, Mock())
    assert status.redis_handler.publish_command.call_count == 2
    ack(status, "a")
    ack(status, "b")
    assert status.sft_ack_groups[(False, 3)].settled


def test_legacy_multi_sft_registration_rejected_before_mesh_publication():
    status = manager(publish=False)
    legacy = atom("a")
    legacy.report_session_id = None
    with pytest.raises(ValueError, match="policy process session ID"):
        status.register(legacy, status.config, Mock())
    status.redis_handler.publish_command.assert_not_called()


def test_extra_atom_cannot_extend_sealed_participant_identity():
    status = manager()
    extra = atom("a")
    extra.global_rank = 1
    extra.report_session_id = "extra-process"
    with pytest.raises(RuntimeError, match="SFT membership is sealed"):
        status.register(extra, status.config, Mock())
    assert len(status.policy_replicas["a"].atoms) == 1


@pytest.mark.parametrize("step,total", [(3, 11), (-1, 10), (11, 10)])
def test_schedule_mismatch_cannot_create_or_evict_ack_group(step, total):
    status = manager()
    with pytest.raises(ValueError, match="published training schedule"):
        status.train_ack(
            "a",
            step,
            total,
            False,
            {},
            Mock(),
            report_session_id="process-a",
            src_global_rank=0,
        )
    assert not status.sft_ack_groups
    assert status.remain_samples_num == 100


def test_policy_client_keeps_registered_incarnation_on_ack_retries(monkeypatch):
    from cosmos_rl.dispatcher.api.client import APIClient

    send = Mock(return_value=Mock(status_code=200))
    monkeypatch.setattr("requests.post", send)
    client = APIClient(Role.POLICY, remote_ips=["localhost"], remote_port=12345)
    client.register(
        "a",
        Role.POLICY,
        MESH_NAMES,
        [0] * len(MESH_NAMES),
        [1] * len(MESH_NAMES),
        0,
        "127.0.0.1",
        "host",
    )
    registration = send.call_args.kwargs["json"]
    client.post_policy_train_ack("a", 3, 10, False, {})
    first = send.call_args.kwargs["json"]
    client.post_policy_train_ack("a", 3, 10, False, {})
    assert first == send.call_args.kwargs["json"]
    assert first["report_session_id"] == registration["report_session_id"]
    assert first["src_global_rank"] == registration["global_rank"] == 0


@pytest.mark.parametrize("session", ["process-a", "restarted-a", None])
def test_http_ack_route_preserves_incarnation_check(monkeypatch, session):
    from cosmos_rl.dispatcher import run_web_panel as web
    from cosmos_rl.dispatcher.protocol import TrainAckRequest

    status = manager()
    monkeypatch.setattr(
        web,
        "controller",
        SimpleNamespace(policy_status_manager=status, rollout_status_manager=Mock()),
    )
    response = asyncio.run(
        web.train_ack(
            TrainAckRequest(
                replica_name="a",
                weight_step=3,
                total_steps=10,
                report_session_id=session,
                src_global_rank=0,
            )
        )
    )
    if session == "process-a":
        assert response == {"message": "Ack completed"}
        assert status.sft_ack_groups[(False, 3)].report_digests.keys() == {"a"}
    else:
        assert response.status_code == 409
        assert not status.sft_ack_groups
