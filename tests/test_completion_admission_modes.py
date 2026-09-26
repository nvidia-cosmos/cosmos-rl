# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the same producer -> rewards -> reporting path in both modes."""

import asyncio
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cosmos_rl.colocated.controller import ColocatedController
from cosmos_rl.dispatcher import run_web_panel
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.admission_state import CompletionAdmissionState
from cosmos_rl.reward.identity import CompletionReporter
from cosmos_rl.dispatcher.algo.grpo import GRPO
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.reward.local_calculator import LocalRewardCalculator
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils.payload import extract_rollouts


class RecordingGRPO(GRPO):
    def __init__(self):
        super().__init__(reward_fn=None)
        self.advantage_inputs = []

    def compute_reward(self, completions, reference, **kwargs):
        rewards = [0.0, 1.0, 100.0]
        return rewards, rewards, [{}, {}, {}]

    def compute_advantage(self, rewards):
        self.advantage_inputs.append(list(rewards))
        return super().compute_advantage(rewards)


@pytest.mark.parametrize(
    "mode", ["disaggregated", "identified", "colocated", "colocated_centralized"]
)
@pytest.mark.parametrize(
    "mask", [[True, True, False], [True, False, False], [False] * 3, [True] * 3]
)
@pytest.mark.parametrize("minimum", [1, 2])
def test_filter_then_advantage_then_ingestion(mode, mask, minimum, monkeypatch):
    config = SimpleNamespace(
        mode="disaggregated"
        if mode in ("disaggregated", "identified")
        else "colocated",
        train=SimpleNamespace(
            non_text=True,
            local_dataset=False,
            train_policy=SimpleNamespace(
                variant="grpo",
                bypass_reward=False,
                rollout_as_token_ids=False,
                min_filter_prefix_tokens=0,
                uncentralized_training=mode != "colocated_centralized",
            ),
        ),
        rollout=SimpleNamespace(
            n_generation=3, multi_turn_config=SimpleNamespace(enable=False)
        ),
    )
    algo = RecordingGRPO()
    algo.minimum_trainable_completions = minimum
    calculator = object.__new__(LocalRewardCalculator)
    calculator.config = config
    calculator.rl_algo = algo
    pending = Queue()

    def enqueue(payloads, is_validation, step, **kwargs):
        pending.put((*calculator.compute_rewards(payloads, is_validation, step), False))

    cls = (
        DisaggregatedRolloutControlWorker
        if mode in ("disaggregated", "identified")
        else ColocatedRolloutControlWorker
    )
    worker = object.__new__(cls)
    worker.config = config
    worker.replica_name = "source"
    worker.current_weight_version = 0
    worker.should_report = True
    worker._report_discarded_samples = Mock()
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    worker.reward_dispatcher = SimpleNamespace(
        enqueue_rewards_cal=enqueue,
        dequeue_rewards_cal=lambda: pending.get()
        if not pending.empty()
        else (None, False, -1, True),
    )
    worker.data_packer = SimpleNamespace(
        get_rollout_output=lambda *values: (*values, None)
    )
    worker.api_client = SimpleNamespace(post_rollout_completion=Mock())
    source = RLPayload(prompt_idx=0, prompt="p", reference_answer="a")
    if mode == "identified":
        worker.completion_reporter = CompletionReporter("source", 0)
        worker.completion_reporter.reserve([source], 3, 0)
    worker._filter_valid_rollout_results_and_report(
        [
            RolloutResult(
                completions=["zero", "one", "outlier"], completion_trainable=mask
            )
        ],
        [source],
    )
    worker.report_rollouts()
    request = worker.api_client.post_rollout_completion.call_args.args[0]
    selected = [reward for reward, keep in zip([0.0, 1.0, 100.0], mask) if keep]
    trainable = len(selected) >= algo.minimum_trainable_completions
    assert algo.advantage_inputs == ([selected] if trainable else [])
    expected_count = len(selected) if trainable else 0
    if mode == "identified":
        assert len(request.completion_failures) == 3 - expected_count
    else:
        assert request.metrics.get("discarded_samples", 0) == 3 - expected_count

    if mode in ("disaggregated", "identified"):
        status = PolicyStatusManager()
        status.current_step = 0
        status.samples_on_the_fly = 3
        status.rollout_admission_closed = lambda: False
        status.next_rollout_training_step = lambda: 1
        status.filter_outdated_rollouts = lambda rollouts: rollouts
        controller = SimpleNamespace(
            config=config,
            policy_status_manager=status,
            completion_admission=None,
            register_discarded_samples_for_refill=Mock(),
            put_rollouts=AsyncMock(),
        )
        if mode == "identified":
            controller.completion_admission = CompletionAdmissionState()
            controller.life_cycle_lock = asyncio.Lock()
            controller.rollout_status_manager = {
                "source": SimpleNamespace(
                    n_atoms_per_replica=lambda: 1, status=SimpleNamespace(ended=False)
                )
            }
            status._publish_payload_transport_cleanup = Mock()
            controller.put_application_rollouts = (
                lambda request, rollouts: Controller.put_application_rollouts(
                    controller, request, rollouts
                )
            )
        from rollout_receipt_fixture import install_report_source

        request = install_report_source(controller, request)
        monkeypatch.setattr(run_web_panel, "controller", controller)
        response = asyncio.run(run_web_panel.put_rollout_group(request))
        assert response == {
            "message": "Identified rollout report processed"
            if mode == "identified"
            else "Rollout put"
        }
        received = controller.put_rollouts.call_args.args[0]
        assert status.samples_on_the_fly == expected_count
    else:
        controller = object.__new__(ColocatedController)
        controller.config = config
        controller.current_step = 0
        controller.train_report_data = {}
        controller.policy = SimpleNamespace(data_queue=Queue())
        mesh = SimpleNamespace(get_group=lambda: None, get_local_rank=lambda: 0)
        controller.rollout = SimpleNamespace(
            parallel_dims=SimpleNamespace(mesh={"dp": mesh}, cp_coord=(0, 1))
        )
        gather = Mock(side_effect=lambda rollouts, **kwargs: [rollouts])
        monkeypatch.setattr(
            "cosmos_rl.colocated.controller.dist_util.all_gather_object_cpu", gather
        )
        controller.put_rollouts(request)
        received = list(controller.policy.data_queue.queue)
        assert gather.call_count == int(mode == "colocated_centralized")
        metrics = controller.train_report_data[0]
        assert "discard_report_id" not in metrics
        assert (
            metrics["rollout/completion_admission_training_discarded_count"]
            == 3 - expected_count
        )
    assert len(received) == expected_count
    if trainable:
        assert [r.reward for r in received] == selected
        assert [r.advantage for r in received] == pytest.approx(
            GRPO(reward_fn=None).compute_advantage(selected)
        )
    if mask == [True, True, False]:
        assert [r.advantage for r in received] == pytest.approx([-0.999998, 0.999998])


def test_ingestion_rejects_late_quality_mask():
    payload = RLPayload(
        completions=["a", "b"],
        rewards=[0.0, 100.0],
        advantages=[-1.0, 1.0],
        completion_trainable=[True, False],
    )
    with pytest.raises(ValueError, match="before computing advantages"):
        extract_rollouts([payload], False)


def test_colocated_minor_step_handles_empty_generation():
    worker = object.__new__(ColocatedRolloutControlWorker)
    worker.config = SimpleNamespace(rollout=SimpleNamespace(n_generation=3))
    worker.global_rank = 0
    worker.current_weight_version = 0
    worker.batch_size = 1
    worker._prompt_queue = Queue()
    worker._prompt_queue.put([RLPayload(prompt="p", weight_version=0)])
    worker.request_new_prompts = Mock(return_value=False)
    worker._call_rollout_generation = Mock(return_value=[])
    worker._report_discarded_samples = Mock()
    worker.inference_stream = worker.data_packer = worker.data_fetcher = None

    assert worker.rollout_for_one_minor_step() == (False, 0)
    assert worker._prompt_queue.empty()
    worker._report_discarded_samples.assert_called_once_with(3, training_rejections=[])
