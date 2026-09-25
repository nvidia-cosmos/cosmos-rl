"""Empty DP slots preserve fetch/version cadence without invented prompts."""

import threading
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cosmos_rl.rollout import State
from cosmos_rl.rollout.prompt_batch import PromptBatch, required_weight_version
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker as Worker,
)
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)


def worker_fixture(rank=0, supported=True, prefetch=False):
    worker = object.__new__(Worker)
    worker.global_rank = rank
    worker.replica_name = "empty-test"
    worker.rank_in_rollout_repicas = 0
    worker._prompt_fetch_lock = threading.Lock()
    worker._prompt_queue = Queue(maxsize=2)
    mesh = SimpleNamespace(
        size=lambda: 2, get_local_rank=lambda: rank, get_group=lambda: None
    )
    worker.parallel_dims = SimpleNamespace(world_size=2, mesh={"dp": mesh})
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            local_dataset=False,
            train_policy=SimpleNamespace(
                data_dispatch_as_rank_in_mesh=False, allowed_outdated_steps=0
            ),
        ),
        rollout=SimpleNamespace(prefetch_rollout=prefetch, async_r2r_sync="disabled"),
    )
    worker.rollout = SimpleNamespace(
        supports_empty_dp_batches=supported,
        rollout_generation=Mock(),
        submit_setup=Mock(),
    )
    worker.api_client = SimpleNamespace(
        get_next_prompt=Mock(return_value=([], False)), post_validation_report=Mock()
    )
    return worker


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_common_guard_precedes_scatter_and_retains_empty_slot(rank, supported):
    worker = worker_fixture(rank, supported)
    payload = SimpleNamespace(prompt_idx=4, weight_version=7)

    def scatter(output, values, **kwargs):
        output[0] = ([payload] if rank == 0 else [], True)

    with (
        patch(
            "cosmos_rl.rollout.worker.rollout_control.dist_utils.broadcast_object_cpu",
            return_value=([payload], True),
        ),
        patch(
            "torch.distributed.scatter_object_list", side_effect=scatter
        ) as scatter_mock,
    ):
        if not supported:
            with pytest.raises(ValueError, match="empty DP slices"):
                worker.request_new_prompts(1, worker._prompt_queue)
            scatter_mock.assert_not_called()
            assert worker._prompt_queue.empty()
        else:
            assert worker.request_new_prompts(1, worker._prompt_queue)
            slot = worker._prompt_queue.get_nowait()
            assert len(slot) == (1 if rank == 0 else 0)
            assert required_weight_version(slot) == 7
    worker.rollout.rollout_generation.assert_not_called()
    worker.rollout.submit_setup.assert_not_called()


def test_unknown_backend_is_not_opted_in():
    assert RolloutBase.supports_empty_dp_batches is False


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("is_end", [False, True])
def test_globally_empty_response_does_not_create_a_slot(rank, is_end):
    worker = worker_fixture(rank)

    def scatter(output, values, **kwargs):
        output[0] = (None, is_end)

    with (
        patch(
            "cosmos_rl.rollout.worker.rollout_control.dist_utils.broadcast_object_cpu",
            return_value=(None, is_end),
        ),
        patch("torch.distributed.scatter_object_list", side_effect=scatter),
    ):
        assert worker.request_new_prompts(1, worker._prompt_queue) is is_end
    assert worker._prompt_queue.empty()


@pytest.mark.parametrize("rank", [0, 1])
def test_nonempty_slices_share_maximum_global_version(rank):
    worker = worker_fixture(rank)
    payloads = [
        SimpleNamespace(prompt_idx=i, weight_version=v) for i, v in enumerate((2, 7))
    ]

    def scatter(output, values, **kwargs):
        output[0] = ([payloads[rank]], False)

    with (
        patch(
            "cosmos_rl.rollout.worker.rollout_control.dist_utils.broadcast_object_cpu",
            return_value=(payloads, False),
        ),
        patch("torch.distributed.scatter_object_list", side_effect=scatter),
    ):
        worker.request_new_prompts(1, worker._prompt_queue)
    assert required_weight_version(worker._prompt_queue.get_nowait()) == 7


def test_required_version_includes_future_prompt_not_just_first():
    assert (
        required_weight_version([SimpleNamespace(weight_version=v) for v in (2, 9, 4)])
        == 9
    )
    with pytest.raises(ValueError, match="global batch metadata"):
        required_weight_version([])


@pytest.mark.parametrize("prefetch", [False, True])
def test_main_loop_empty_slot_waits_for_common_version_and_skips_backend(prefetch):
    worker = worker_fixture(prefetch=prefetch)
    worker.state = State()
    worker.state.set_weight_synced()
    worker.state.set_prompt_fetch_end()
    worker._prompt_queue.put(PromptBatch([], 7))
    worker.current_weight_version = 6
    worker.shutdown_signal = threading.Event()
    worker.validation_flag = threading.Event()
    worker._is_async_rollout = False
    worker.batch_size = 1
    worker.should_report = False
    worker._bind_prefetch_context_once = Mock()
    worker._maybe_emit_mainloop_summary = Mock()
    worker.report_rollouts = Mock(return_value=(None, False, None, True))
    worker.one_step_generation = Mock(side_effect=AssertionError("empty backend call"))
    turns = []

    def consume(**kwargs):
        turns.append(worker._prompt_queue.qsize())
        if len(turns) == 2:
            assert not worker._prompt_queue.empty()
            worker.current_weight_version = 7
        if len(turns) == 3:
            worker.shutdown_signal.set()

    worker.consume_command = consume
    worker._main_loop_impl()
    assert turns == [1, 1, 0]
    worker.one_step_generation.assert_not_called()


def test_preparation_and_validation_forward_skip_empty_slot():
    worker = worker_fixture()
    slot = PromptBatch([], 4)
    worker._submit_prefetch_setup(slot)
    assert worker._call_rollout_generation(payloads=slot, is_validation=True) == []
    worker.rollout.submit_setup.assert_not_called()
    worker.rollout.rollout_generation.assert_not_called()


def test_colocated_empty_slot_uses_same_version_gate():
    worker = worker_fixture()
    worker.batch_size = 1
    worker.current_weight_version = 2
    worker.request_new_prompts = Mock(return_value=True)
    worker.one_step_generation = Mock()
    worker._prompt_queue.put(PromptBatch([], 3))
    assert ColocatedRolloutControlWorker.rollout_for_one_minor_step(worker) == (True, 0)
    assert worker._prompt_queue.qsize() == 1
    worker.current_weight_version = 3
    assert ColocatedRolloutControlWorker.rollout_for_one_minor_step(worker) == (True, 0)
    assert worker._prompt_queue.empty()
    worker.one_step_generation.assert_not_called()


def test_empty_validation_rank_still_reports_terminal_without_reward_work():
    worker = worker_fixture()
    worker._is_async_rollout = False
    worker.val_batch_size = 1
    worker.current_step = 4
    worker.validation_flag = threading.Event()
    worker.should_report = True
    worker.inference_stream = worker.val_data_packer = worker.data_fetcher = None

    def fetch(batch_size, queue, **kwargs):
        queue.put(PromptBatch([], 4))
        return True

    worker.request_new_prompts = fetch
    worker.reward_dispatcher = Mock()
    worker.do_validation()
    request = worker.api_client.post_validation_report.call_args.args[0]
    assert request.is_end and request.payloads == [] and request.validation_step == 4
    worker.reward_dispatcher.enqueue_rewards_cal.assert_not_called()
