# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Repeated end-of-data and training-only preparation at validation boundaries."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.rollout import State
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.rollout.worker.weight_sync import process_wst_deferred_actions
from cosmos_rl.rollout.worker import weight_sync as ws
from test_rollout_generation_mixin import _FakeBackend, _FakePayload
from test_r2r_unseeded_source import FakeWorker, FakeCommand, make_thread


@pytest.mark.parametrize("waiting_for", ["generation", "reward", None])
def test_async_end_waits_for_drain_and_remains_a_command_participant(waiting_for):
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.state = State()
    worker.state.set_weight_synced()
    worker.state.set_prompt_fetch_end()
    worker.scheduler = SimpleNamespace(
        is_idle=Mock(return_value=waiting_for != "generation")
    )
    worker.reward_dispatcher = SimpleNamespace(
        is_empty=Mock(return_value=waiting_for != "reward")
    )
    worker._stream_generation_collect_results = Mock()
    worker.should_report = True
    worker._rollout_end_acknowledged = False
    worker.report_rollouts = Mock(return_value=(None, False, None, True))
    worker.replica_name, worker.global_rank = "rollout-0", 0
    worker.parallel_dims = SimpleNamespace(world_size=1)
    worker.config = SimpleNamespace(mode="disaggregated")
    worker.api_client = SimpleNamespace(post_rollout_completion=Mock(return_value=True))
    worker.shutdown_signal = threading.Event()
    worker.shutdown_mp_signal = threading.Event()
    if waiting_for is not None:
        worker.stream_generation_step()
        assert not worker.state.prompt_consume_end()
        worker.api_client.post_rollout_completion.assert_not_called()
        worker.scheduler.is_idle.return_value = True
        worker.reward_dispatcher.is_empty.return_value = True
    for _ in range(3):
        worker.stream_generation_step()
    assert worker.state.prompt_consume_end()
    worker.api_client.post_rollout_completion.assert_called_once()
    assert not worker.shutdown_signal.is_set()
    worker.handle_stop(SimpleNamespace())
    assert worker.shutdown_signal.is_set() and worker.shutdown_mp_signal.is_set()


class PhaseBackend(_FakeBackend):
    def _prepare_sample(
        self, payload, *, data_packer=None, data_fetcher=None, is_validation=False
    ):
        self._record("_prepare_sample")
        return payload.prompt, is_validation

    def _collate_batch(self, samples, **kwargs):
        return samples

    def _generate(self, batch, **kwargs):
        return batch

    def _postprocess(self, raw, payloads, **kwargs):
        return raw


@pytest.mark.parametrize("prefetch", [False, True])
def test_validation_cannot_consume_a_training_preparation_with_the_same_index(prefetch):
    backend = PhaseBackend(prefetch=prefetch)
    train, validation = _FakePayload(0, "train"), _FakePayload(0, "validation")
    try:
        backend.submit_setup([train])
        assert backend.rollout_generation([validation], is_validation=True) == [
            ("validation", True)
        ]
        assert backend.rollout_generation([train]) == [("train", False)]
        assert backend.events.count("_prepare_sample") == 2
        assert not backend._setup_futures
    finally:
        backend.shutdown_generation()


@pytest.mark.parametrize("validation_enabled", [False, True])
@pytest.mark.parametrize("already_seeded", [False, True])
def test_final_sync_broadcast_defers_validation_and_shutdown_until_generation_returns(
    validation_enabled, already_seeded
):
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.replica_name = "rollout-0"
    worker.state = State()
    if already_seeded:
        worker.state.set_weight_synced()
    worker.current_weight_version = 0
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            async_r2r_sync="disabled",
            broadcast_all_params=False,
            prefetch_rollout=False,
        ),
        validation=SimpleNamespace(
            enable=validation_enabled, val_before_train=False, freq=1
        ),
    )
    worker.validation_flag = threading.Event()
    worker.shutdown_signal = threading.Event()
    worker.shutdown_mp_signal = threading.Event()
    worker.redis_controller = SimpleNamespace(publish_teacher_request=Mock())
    worker.consume_command = Mock()
    worker.report_rollouts = Mock()
    worker._maybe_emit_mainloop_summary = Mock()
    generating = True

    def validate():
        assert not generating, "validation re-entered an active generation call"
        assert worker.state.weight_synced()
        assert not worker.shutdown_signal.is_set()
        worker.validation_flag.clear()

    worker.do_validation = Mock(side_effect=validate)
    command = SimpleNamespace(
        src_replica_name=worker.replica_name,
        dst_replica_names=[worker.replica_name],
        trainable_only=False,
        weight_step=1,
        total_steps=1,
        replica_should_stop=lambda: True,
    )
    worker.broadcast_to_all_rollout_replica(command)
    worker.do_validation.assert_not_called()
    worker.redis_controller.publish_teacher_request.assert_not_called()
    assert not worker.shutdown_signal.is_set()

    generating = False
    worker._main_loop_impl()
    assert worker.do_validation.call_count == int(validation_enabled)
    assert worker.shutdown_signal.is_set() and worker.shutdown_mp_signal.is_set()
    worker.report_rollouts.assert_not_called()
    worker.redis_controller.publish_teacher_request.assert_called_once()
    process_wst_deferred_actions(worker)
    worker.redis_controller.publish_teacher_request.assert_called_once()


@pytest.mark.parametrize("trainable_only", [False, True])
@pytest.mark.parametrize("outcome", ["healthy", "cancelled", "failed"])
def test_wst_full_broadcast_marks_frozen_parameters_received_only_after_success(
    monkeypatch, trainable_only, outcome
):
    worker = FakeWorker(replica_name="rollout-b", buffer_version=0)
    worker.non_trainable_params_received = False
    worker.state = State()
    worker._buffer_weight_version = 0
    worker.shutdown_signal = threading.Event()
    worker.shutdown_mp_signal = threading.Event()
    worker.config = SimpleNamespace(
        validation=SimpleNamespace(enable=False, val_before_train=False, freq=1)
    )
    thread = make_thread(worker)
    thread._stream = object()
    thread._queue = SimpleNamespace(qsize=lambda: 0)
    thread._executed = 0
    command = FakeCommand(dsts=("rollout-a", "rollout-b"))
    command.trainable_only = trainable_only
    monkeypatch.setattr(ws, "r2r_barrier", lambda *a, **k: outcome != "cancelled")
    monkeypatch.setattr(ws.torch.cuda, "Event", lambda: SimpleNamespace(record=Mock()))

    def broadcast(*args):
        assert not worker.non_trainable_params_received
        if outcome == "failed":
            raise RuntimeError("injected broadcast failure")
        return 2, 16

    monkeypatch.setattr(ws, "do_nccl_broadcast_grouped", broadcast)
    if outcome == "failed":
        with pytest.raises(RuntimeError, match="injected broadcast failure"):
            thread._execute_r2r(command)
    else:
        thread._execute_r2r(command)
    assert worker.non_trainable_params_received is (outcome == "healthy")
