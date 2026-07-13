# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cosmos_rl.dispatcher.command import TrainingCompleteCommand
from cosmos_rl.dispatcher.protocol import MESH_NAMES, Role
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.status import (
    PolicyStatus,
    PolicyStatusManager,
    RolloutStatusManager,
)


def _rollout_atom(replica_name: str, global_rank: int, dp_rank: int) -> Atom:
    ranks = [0] * len(MESH_NAMES)
    group_size = [1] * len(MESH_NAMES)
    ranks[MESH_NAMES.index("dp_shard")] = dp_rank
    group_size[MESH_NAMES.index("dp_shard")] = 2
    return Atom(
        global_rank=global_rank,
        host_ip="127.0.0.1",
        host_name="localhost",
        trace_path="",
        ranks=ranks,
        group_size=group_size,
        replica_name=replica_name,
    )


class TestRankedRolloutEnd(unittest.TestCase):
    def test_replica_ends_only_after_every_reporting_rank(self):
        replica = Replica(
            "rollout-0",
            Role.ROLLOUT,
            [_rollout_atom("rollout-0", 0, 0), _rollout_atom("rollout-0", 1, 1)],
        )
        manager = RolloutStatusManager()
        manager.rollout_replicas = {replica.name: replica}

        self.assertFalse(manager.rollout_end(replica.name, src_global_rank=99))
        self.assertFalse(replica.status.ended)
        self.assertFalse(manager.rollout_end(replica.name, src_global_rank=0))
        self.assertFalse(replica.status.ended)
        self.assertTrue(manager.rollout_end(replica.name, src_global_rank=1))
        self.assertTrue(replica.status.ended)
        self.assertFalse(manager.rollout_end(replica.name, src_global_rank=1))

    def test_rankless_legacy_end_marks_whole_replica(self):
        replica = SimpleNamespace(status=SimpleNamespace(ended=False), atoms={})
        manager = RolloutStatusManager()
        manager.rollout_replicas = {"legacy": replica}
        self.assertTrue(manager.rollout_end("legacy"))
        self.assertTrue(replica.status.ended)

    def test_rankless_command_participant_remains_safe_for_sync(self):
        replica = _registered_rollout("trtllm")
        manager = RolloutStatusManager()
        manager.rollout_replicas = {replica.name: replica}

        self.assertTrue(
            manager.rollout_end(replica.name, stays_command_participant=True)
        )
        self.assertEqual(
            manager.get_safe_weight_sync_replicas(validation_enabled=False),
            [replica],
        )

    def test_http_forwards_rankless_command_participant_capability(self):
        from cosmos_rl.dispatcher import run_web_panel

        rollout_end = MagicMock(return_value=False)
        fake_controller = SimpleNamespace(
            rollout_status_manager=SimpleNamespace(rollout_end=rollout_end)
        )
        request = SimpleNamespace(
            is_end=True,
            src_replica_name="trtllm",
            src_global_rank=None,
            stays_command_participant=True,
        )
        with patch.object(run_web_panel, "controller", fake_controller):
            response = asyncio.run(run_web_panel.put_rollout_group(request))

        self.assertEqual(response, {"message": "Rollout end signal received"})
        rollout_end.assert_called_once_with(
            "trtllm",
            src_global_rank=None,
            stays_command_participant=True,
        )


def _registered_rollout(name: str, *, ended: bool = False):
    return SimpleNamespace(
        name=name,
        start_time=0,
        all_atoms_arrived=True,
        in_mesh=True,
        status=SimpleNamespace(ended=ended),
    )


class TestWeightSyncTopology(unittest.TestCase):
    def test_ranked_checkout_keeps_full_registered_communicator(self):
        manager = RolloutStatusManager()
        manager.rollout_replicas = {
            "r0": _registered_rollout("r0", ended=True),
            "r1": _registered_rollout("r1"),
            "r2": _registered_rollout("r2"),
        }
        manager._command_participant_ended_replicas = {"r0"}

        targets = manager.get_safe_weight_sync_replicas(validation_enabled=False)

        self.assertEqual([replica.name for replica in targets], ["r0", "r1", "r2"])

    def test_legacy_checkout_suppresses_subset_sync(self):
        manager = RolloutStatusManager()
        manager.rollout_replicas = {
            "legacy": _registered_rollout("legacy", ended=True),
            "live": _registered_rollout("live"),
        }

        self.assertEqual(
            manager.get_safe_weight_sync_replicas(validation_enabled=False),
            [],
        )

    def test_unregister_rebuilds_all_safe_survivors_including_one(self):
        policy_status = SimpleNamespace(training_finished=lambda: False)
        for survivor_names in (["live", "ranked"], ["live"]):
            with self.subTest(survivor_names=survivor_names):
                manager = RolloutStatusManager()
                manager.rollout_replicas = {
                    "departing": _registered_rollout("departing"),
                    **{
                        name: _registered_rollout(name, ended=name == "ranked")
                        for name in survivor_names
                    },
                }
                manager._command_participant_ended_replicas = {
                    name for name in survivor_names if name == "ranked"
                }
                rebuilds = []
                manager.trigger_rebuild_mesh = lambda replicas: rebuilds.append(
                    [replica.name for replica in replicas]
                )

                manager.unregister("departing", policy_status)

                self.assertEqual(rebuilds, [survivor_names])


class TestTrainingCompleteCoordinates(unittest.TestCase):
    def test_constructor_preserves_legacy_positional_prefix(self):
        command = TrainingCompleteCommand(
            "policy-0",
            2,
            2,
            7,
            True,
        )

        self.assertEqual(command.remain_samples_num, 7)
        self.assertTrue(command.do_save)
        self.assertEqual(command.final_step, 1)
        self.assertEqual(command.checkpoint_total_steps, 2)

    def test_completion_is_staged_before_publish_and_does_not_advance_real_step(self):
        replica = SimpleNamespace(
            name="policy-0",
            sub_profiler_config=SimpleNamespace(
                do_profile=False,
                active_steps=None,
                rank_filter=None,
                record_shape=None,
                profile_memory=None,
                with_stack=None,
                with_modules=None,
            ),
        )
        manager = PolicyStatusManager()
        manager.current_step = 1
        manager.total_steps = 99
        manager.draining_total_steps = 5
        manager.remain_samples_num = 3
        manager.data_fetcher = SimpleNamespace(activated_val_iter=None)
        manager.config = SimpleNamespace(
            train=SimpleNamespace(ckpt=SimpleNamespace(enable_checkpoint=True))
        )
        manager.redis_handler = object()
        manager.policy_replicas = {replica.name: replica}
        manager.status = {replica.name: PolicyStatus.READY}
        manager.get_all_atoms_arrived_replicas = lambda: [replica]
        published = []

        def fake_trigger(**kwargs):
            self.assertEqual(manager.completion_step, 2)
            self.assertEqual(manager.completion_recipients, {replica.name})
            self.assertEqual(manager.completion_acks, set())
            self.assertEqual(manager.status[replica.name], PolicyStatus.RUNNING)
            self.assertTrue(manager.rollout_admission_closed())
            published.append(kwargs)

        with patch.object(TrainingCompleteCommand, "trigger", fake_trigger):
            manager.trigger_training_complete()

        self.assertEqual(manager.current_step, 1)
        self.assertEqual(manager.dispatched_rollouts_by_step, {})
        self.assertEqual(published[0]["global_step"], 2)
        self.assertEqual(published[0]["total_steps"], 2)
        self.assertEqual(published[0]["final_step"], 1)
        self.assertEqual(published[0]["checkpoint_total_steps"], 5)
        self.assertTrue(published[0]["do_save"])

    def test_zero_horizon_and_resume_at_horizon_use_synthetic_completion(self):
        for current_step, horizon, expect_save in ((0, 0, False), (5, 5, True)):
            with self.subTest(current_step=current_step, horizon=horizon):
                replica = SimpleNamespace(
                    name="policy-0",
                    sub_profiler_config=SimpleNamespace(
                        do_profile=False,
                        active_steps=None,
                        rank_filter=None,
                        record_shape=None,
                        profile_memory=None,
                        with_stack=None,
                        with_modules=None,
                    ),
                )
                manager = PolicyStatusManager()
                manager.current_step = current_step
                manager.total_steps = horizon
                manager.data_fetcher = SimpleNamespace(activated_val_iter=None)
                manager.config = SimpleNamespace(
                    train=SimpleNamespace(ckpt=SimpleNamespace(enable_checkpoint=True))
                )
                manager.redis_handler = object()
                manager.policy_replicas = {replica.name: replica}
                manager.status = {replica.name: PolicyStatus.READY}
                manager.get_all_atoms_arrived_replicas = lambda: [replica]
                published = []

                with patch.object(
                    TrainingCompleteCommand,
                    "trigger",
                    side_effect=lambda **kwargs: published.append(kwargs),
                ):
                    manager.trigger_training_complete()

                self.assertEqual(manager.current_step, current_step)
                self.assertEqual(published[0]["global_step"], current_step + 1)
                self.assertEqual(published[0]["total_steps"], current_step + 1)
                self.assertEqual(published[0]["final_step"], current_step)
                self.assertEqual(published[0]["checkpoint_total_steps"], horizon)
                self.assertEqual(published[0]["do_save"], expect_save)

    def test_partial_publication_cannot_shrink_completion_recipients(self):
        replicas = [
            SimpleNamespace(
                name=name,
                sub_profiler_config=SimpleNamespace(
                    do_profile=False,
                    active_steps=None,
                    rank_filter=None,
                    record_shape=None,
                    profile_memory=None,
                    with_stack=None,
                    with_modules=None,
                ),
            )
            for name in ("p0", "p1")
        ]
        manager = PolicyStatusManager()
        manager.current_step = 1
        manager.total_steps = 3
        manager.data_fetcher = SimpleNamespace(activated_val_iter=None)
        manager.config = SimpleNamespace(
            train=SimpleNamespace(ckpt=SimpleNamespace(enable_checkpoint=False))
        )
        manager.redis_handler = object()
        manager.policy_replicas = {replica.name: replica for replica in replicas}
        manager.status = {replica.name: PolicyStatus.READY for replica in replicas}
        manager.get_all_atoms_arrived_replicas = lambda: replicas

        def publish(**kwargs):
            if kwargs["replica"].name == "p1":
                raise RuntimeError("publish failed")

        with (
            patch.object(TrainingCompleteCommand, "trigger", side_effect=publish),
            self.assertRaisesRegex(RuntimeError, "publish failed"),
        ):
            manager.trigger_training_complete()

        self.assertEqual(manager.completion_recipients, {"p0", "p1"})
        self.assertEqual(manager.completion_acks, set())
        self.assertTrue(manager.rollout_admission_closed())
        self.assertFalse(manager.terminal_complete)


class TestTailCleanup(unittest.TestCase):
    def _manager(self):
        manager = PolicyStatusManager()
        manager.config = SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False)
            )
        )
        manager.rollout_buffer = Queue()
        manager.rollout_buffer_per_rank = []
        manager.samples_on_the_fly = 3
        manager.remain_samples_num = 11
        manager.cleaned = []
        manager._publish_payload_transport_cleanup = lambda dropped, retained: (
            manager.cleaned.append((dropped, retained))
        )
        return manager

    def test_buffer_cleanup_releases_payloads_without_consuming_budget(self):
        manager = self._manager()
        items = [object(), object(), object()]
        for item in items:
            manager.rollout_buffer.put(item)
        self.assertEqual(manager.cleanup_buffered_rollouts(), 3)
        self.assertTrue(manager.rollout_buffer.empty())
        self.assertEqual(manager.samples_on_the_fly, 0)
        self.assertEqual(manager.remain_samples_num, 11)
        self.assertEqual(manager.cleaned, [(items, [])])

    def test_rank_specific_cleanup_drains_uneven_tail(self):
        manager = self._manager()
        manager.config.train.train_policy.data_dispatch_as_rank_in_mesh = True
        manager.rollout_buffer_per_rank = [Queue(), Queue()]
        items = [object(), object(), object()]
        manager.rollout_buffer_per_rank[0].put(items[0])
        manager.rollout_buffer_per_rank[0].put(items[1])
        manager.rollout_buffer_per_rank[1].put(items[2])

        self.assertEqual(manager.cleanup_buffered_rollouts(), 3)

        self.assertTrue(all(queue.empty() for queue in manager.rollout_buffer_per_rank))
        self.assertEqual(manager.samples_on_the_fly, 0)
        self.assertEqual(manager.remain_samples_num, 11)
        self.assertEqual(manager.cleaned, [(items, [])])

    def test_terminal_dapo_cleanup_settles_payload_and_filtered_counts(self):
        manager = self._manager()
        manager.samples_on_the_fly = 8
        manager.filter_records = {"sampled": 9}
        payloads = [object(), object()]
        settled = manager.cleanup_terminal_rollouts(
            payloads,
            {
                "sampled": 2,
                "filtered_positive": 3,
                "filtered_negative": 1,
            },
            is_dapo=True,
        )
        self.assertEqual(settled, 6)
        self.assertEqual(manager.samples_on_the_fly, 2)
        self.assertEqual(manager.remain_samples_num, 11)
        self.assertEqual(manager.filter_records, {"sampled": 9})
        self.assertEqual(manager.cleaned, [(payloads, [])])

    def test_malformed_dapo_counts_are_non_throwing_zero(self):
        manager = self._manager()
        manager.samples_on_the_fly = 5
        counts = manager.parse_dynamic_sampling_counts(
            {
                "sampled": "2",
                "filtered_positive": True,
                "filtered_negative": -1,
            }
        )
        self.assertEqual(
            counts,
            {"sampled": 0, "filtered_positive": 0, "filtered_negative": 0},
        )
        manager.update_dynamic_sampling_statistics(
            {"filtered_positive": None, "filtered_negative": 1.5}
        )
        self.assertEqual(manager.samples_on_the_fly, 5)
        self.assertEqual(manager.remain_samples_num, 11)

    def test_normal_dapo_path_settles_filtered_only_work(self):
        manager = self._manager()
        manager.samples_on_the_fly = 7
        manager.filter_records = {}
        manager.update_dynamic_sampling_statistics(
            {"sampled": 2, "filtered_positive": 3, "filtered_negative": 1}
        )
        self.assertEqual(manager.samples_on_the_fly, 3)
        self.assertEqual(manager.remain_samples_num, 7)
        self.assertEqual(
            manager.filter_records,
            {"sampled": 2, "filtered_positive": 3, "filtered_negative": 1},
        )


class TestTerminalHttpAdmission(unittest.TestCase):
    def test_terminal_cleanup_precedes_dapo_stats_and_outdated_filtering(self):
        from cosmos_rl.dispatcher import run_web_panel

        extracted = object()
        cleaned = []
        policy_status = SimpleNamespace(
            rollout_admission_closed=lambda: True,
            cleanup_terminal_rollouts=lambda rollouts, metrics, is_dapo: cleaned.append(
                (rollouts, metrics, is_dapo)
            ),
            update_dynamic_sampling_statistics=lambda _metrics: (_ for _ in ()).throw(
                AssertionError("terminal DAPO stats must be bypassed")
            ),
            filter_outdated_rollouts=lambda _rollouts: (_ for _ in ()).throw(
                AssertionError("terminal outdated filtering must be bypassed")
            ),
        )
        fake_controller = SimpleNamespace(
            policy_status_manager=policy_status,
            config=SimpleNamespace(
                train=SimpleNamespace(train_policy=SimpleNamespace(variant="dapo"))
            ),
            put_rollouts=AsyncMock(
                side_effect=AssertionError("terminal work must not be admitted")
            ),
        )
        request = SimpleNamespace(
            is_end=False,
            src_replica_name="rollout-0",
            payloads=[object()],
            metrics={"filtered_positive": 2},
        )
        with (
            patch.object(run_web_panel, "controller", fake_controller),
            patch.object(run_web_panel, "extract_rollouts", return_value=[[extracted]]),
        ):
            response = asyncio.run(run_web_panel.put_rollout_group(request))

        self.assertEqual(response, {"message": "Terminal rollout cleaned"})
        self.assertEqual(
            cleaned,
            [([extracted], {"filtered_positive": 2}, True)],
        )
        fake_controller.put_rollouts.assert_not_awaited()


class TestNaturalTerminalEvidence(unittest.TestCase):
    def test_real_final_ack_closes_admission_and_cleans_surplus(self):
        manager = PolicyStatusManager()
        manager.config = SimpleNamespace(
            mode="disaggregated",
            train=SimpleNamespace(
                train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False)
            ),
        )
        manager.current_step = 2
        manager.total_steps = 2
        manager.samples_on_the_fly = 2
        manager.remain_samples_num = 10
        manager.rollout_buffer = Queue()
        manager.rollout_buffer_per_rank = []
        manager._publish_payload_transport_cleanup = lambda *_args: None
        manager.rollout_buffer.put(object())
        manager.rollout_buffer.put(object())

        manager.record_real_datafetch_acked(2, 2)

        self.assertEqual(manager.last_real_datafetch_acked_step, 2)
        self.assertTrue(manager.rollout_admission_closed())
        self.assertTrue(manager.terminal_complete)
        self.assertTrue(manager.rollout_buffer.empty())
        self.assertEqual(manager.samples_on_the_fly, 0)
        self.assertEqual(manager.remain_samples_num, 10)

    def test_ack_with_stale_advertised_horizon_is_not_terminal(self):
        manager = PolicyStatusManager()
        manager.config = SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False)
            )
        )
        manager.current_step = 2
        manager.total_steps = 2
        manager.samples_on_the_fly = 0
        manager.rollout_buffer = Queue()
        manager.rollout_buffer_per_rank = []
        manager._publish_payload_transport_cleanup = lambda *_args: None

        manager.record_real_datafetch_acked(2, 3)

        self.assertFalse(manager.real_terminal_command_acked())
        self.assertFalse(manager.rollout_admission_closed())
        self.assertFalse(manager.terminal_complete)

    def test_completion_requires_every_prestaged_recipient_ack(self):
        manager = PolicyStatusManager()
        manager.policy_replicas = {
            "p0": SimpleNamespace(name="p0"),
            "p1": SimpleNamespace(name="p1"),
        }
        manager.status = {
            "p0": PolicyStatus.RUNNING,
            "p1": PolicyStatus.RUNNING,
        }
        manager.completion_step = 4
        manager.completion_recipients = {"p0", "p1"}
        manager.completion_acks = set()
        manager.config = SimpleNamespace(
            train=SimpleNamespace(train_policy=SimpleNamespace(type="grpo"))
        )

        self.assertTrue(manager.record_completion_ack("p0", 4))
        self.assertFalse(manager.terminal_complete)
        self.assertTrue(manager.record_completion_ack("p1", 4))
        self.assertTrue(manager.terminal_complete)
        self.assertEqual(manager.completion_acks, {"p0", "p1"})

    def test_completion_ack_survives_early_recipient_unregister(self):
        manager = PolicyStatusManager()
        manager.policy_replicas = {
            name: SimpleNamespace(name=name, in_mesh=True) for name in ("p0", "p1")
        }
        manager.status = {
            name: PolicyStatus.RUNNING for name in manager.policy_replicas
        }
        manager.completion_step = 4
        manager.completion_recipients = {"p0", "p1"}
        rollout_status = SimpleNamespace()

        manager.train_ack("p0", 4, 4, False, {}, rollout_status)
        manager.unregister("p0")
        manager.train_ack("p1", 4, 4, False, {}, rollout_status)

        self.assertEqual(manager.completion_acks, {"p0", "p1"})
        self.assertTrue(manager.terminal_complete)


class TestTerminalMatrix(unittest.TestCase):
    @staticmethod
    def _manager(accepted_count):
        replica = SimpleNamespace(
            name="policy-0",
            start_time=0,
            put_rollout=MagicMock(),
            sub_profiler_config=SimpleNamespace(
                do_profile=False,
                active_steps=None,
                rank_filter=None,
                record_shape=None,
                profile_memory=None,
                with_stack=None,
                with_modules=None,
            ),
        )
        manager = PolicyStatusManager()
        manager.current_step = 0
        manager.total_steps = 2
        manager.remain_samples_num = 20
        manager.samples_on_the_fly = accepted_count
        manager.rollout_buffer = Queue()
        for _ in range(accepted_count):
            manager.rollout_buffer.put(object())
        manager.policy_replicas = {replica.name: replica}
        manager.status = {replica.name: PolicyStatus.READY}
        manager.get_all_atoms_arrived_replicas = lambda: [replica]
        manager.data_fetcher = SimpleNamespace(activated_val_iter=None)
        manager.redis_handler = object()
        manager.samples_per_epoch = 20
        manager.config = SimpleNamespace(
            mode="disaggregated",
            validation=SimpleNamespace(enable=False, freq=1),
            logging=SimpleNamespace(logger=[]),
            train=SimpleNamespace(
                train_batch_per_replica=2,
                epoch=1,
                non_text=False,
                ckpt=SimpleNamespace(
                    enable_checkpoint=False,
                    save_freq=1,
                    save_freq_in_epoch=0,
                ),
                train_policy=SimpleNamespace(
                    data_dispatch_as_rank_in_mesh=False,
                    rollout_as_token_ids=False,
                ),
            ),
        )
        manager._publish_payload_transport_cleanup = lambda *_args: None
        return manager, replica

    def test_complete_batches_train_and_only_partial_tail_is_synthetic(self):
        rollout_status = SimpleNamespace(all_rollouts_ended=lambda: True)
        expected_real_steps = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 6: 2}
        for accepted_count, expected_steps in expected_real_steps.items():
            with self.subTest(accepted_count=accepted_count):
                manager, _replica = self._manager(accepted_count)
                real_commands = []
                completion_commands = []

                def datafetch_trigger(**kwargs):
                    real_commands.append(kwargs)

                def completion_trigger(**kwargs):
                    completion_commands.append(kwargs)

                with (
                    patch(
                        "cosmos_rl.dispatcher.command.DataFetchCommand.trigger",
                        datafetch_trigger,
                    ),
                    patch.object(
                        TrainingCompleteCommand, "trigger", completion_trigger
                    ),
                ):
                    manager.finish_draining_phase(rollout_status)
                    while len(real_commands) < expected_steps:
                        manager.status["policy-0"] = PolicyStatus.READY
                        manager.finish_draining_phase(rollout_status)

                    if expected_steps == 2:
                        manager.record_real_datafetch_acked(2, 2)
                    elif expected_steps == 1:
                        manager.status["policy-0"] = PolicyStatus.READY
                        manager.finish_draining_phase(rollout_status)

                self.assertEqual(len(real_commands), expected_steps)
                self.assertEqual(
                    [command["total_steps"] for command in real_commands],
                    [2] * expected_steps,
                )
                self.assertEqual(
                    len(completion_commands),
                    0 if expected_steps == 2 else 1,
                )
                self.assertEqual(manager.current_step, expected_steps)
                self.assertEqual(manager.total_steps, 2)

    def test_tail_dispatch_uses_snapshot_after_late_total_step_write(self):
        rollout_status = SimpleNamespace(all_rollouts_ended=lambda: True)
        for late_total_steps in (1, 99):
            with self.subTest(late_total_steps=late_total_steps):
                manager, _replica = self._manager(2)
                manager.current_step = 1
                manager.config.train.ckpt.enable_checkpoint = True
                manager.enter_draining_phase()
                manager.total_steps = late_total_steps
                real_commands = []

                with patch(
                    "cosmos_rl.dispatcher.command.DataFetchCommand.trigger",
                    side_effect=lambda **kwargs: real_commands.append(kwargs),
                ):
                    manager.finish_draining_phase(rollout_status)

                self.assertEqual(len(real_commands), 1)
                self.assertEqual(real_commands[0]["global_step"], 2)
                self.assertEqual(real_commands[0]["total_steps"], 2)
                self.assertTrue(real_commands[0]["do_save"])


if __name__ == "__main__":
    unittest.main()
