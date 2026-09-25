# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real vLLM generation across the actual worker's async validation boundary.

Uses dummy model weights, local request/reward fixtures and no controller server.
This tests phase/drain integration, not P2R/R2R quiescence or transport recovery.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import toml
import torch

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.policy.config import Config
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.rollout.vllm_rollout.vllm_rollout_async import vLLMRolloutAsync
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    RolloutTask,
    RolloutTaskScheduler,
)
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils import async_utils
from cosmos_rl.utils.parallelism import ParallelDims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--live-weight-fence", action="store_true")
    args = parser.parse_args()
    if args.live_weight_fence:
        from async_live_weight_canary import install_canary_extension

        install_canary_extension()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    async_utils.unsafe_enable_nest_asyncio()
    config = Config.from_dict(toml.load(args.config))
    config.rollout.backend = "vllm_async"
    config.rollout.mode = "async"
    config.rollout.parallelism.tp_size = config.rollout.parallelism.pp_size = 1
    config.rollout.parallelism.dp_shard_size = 1
    config.rollout.enforce_eager = True
    config.rollout.gpu_memory_utilization = 0.5
    config.rollout.max_response_length = 16
    config.rollout.n_generation = 1
    config.policy.model_max_length = 512
    config.validation.enable = True
    config.validation.n_generation = 2
    config.validation.temperature = 0.8
    config.validation.max_response_length = 8
    dims = ParallelDims.from_config(config.rollout.parallelism)
    engine = vLLMRolloutAsync(config, parallel_dims=dims, device=torch.device("cuda:0"))
    counts = {False: 0, True: 0}
    observed = []

    def packer(validation):
        def pack(prompt):
            assert prompt.startswith("Validation" if validation else "Training")
            counts[validation] += 1
            return prompt

        return SimpleNamespace(
            get_rollout_input=pack,
            rollout_collate_fn=lambda prompts: prompts,
            get_rollout_output=lambda *values: (*values, None),
        )

    train_packer, val_packer = packer(False), packer(True)
    generate = engine.rollout_generation

    async def checked_generation(**kwargs):
        if kwargs["payloads"][0].prompt == "Training injected failure":
            print("INJECTED_GENERATION_FAILURE", flush=True)
            raise RuntimeError("injected generation failure before native submission")
        results = await generate(**kwargs)
        if kwargs["payloads"][0].prompt == "Training child failure":
            assert results == []
            return results
        phase = kwargs["is_validation"]
        assert len(results) == 1
        assert len(results[0].completions) == (2 if phase else 1)
        observed.append((phase, kwargs["payloads"][0].prompt_idx))
        return results

    engine.rollout_generation = checked_generation
    scheduler = RolloutTaskScheduler(
        engine,
        train_packer,
        val_data_packer=val_packer,
        max_concurrent_requests=2,
        check_interval=0.01,
        stream=torch.cuda.Stream(),
    )
    initialized = threading.Event()
    init_errors = []

    def initialize(rollout):
        try:
            rollout.init_engine(quantization=None, seed=42, load_format="dummy")
        except BaseException as error:
            init_errors.append(error)
            raise
        finally:
            initialized.set()

    scheduler.start(initialize)
    assert initialized.wait(240), "engine initialization timed out"
    if init_errors:
        raise init_errors[0]
    deadline = time.monotonic() + 5
    while not scheduler.is_running():
        assert time.monotonic() < deadline
        time.sleep(0.01)

    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.scheduler, worker._is_async_rollout = scheduler, True
    worker._prompt_queue, worker.rank_in_rollout_repicas = Queue(), 0
    worker.current_step = worker.current_weight_version = 7
    worker.val_batch_size, worker.should_report = 2, True
    worker.replica_name, worker.is_diffusers = "phase-canary", False
    worker.config = SimpleNamespace(
        validation=SimpleNamespace(n_generation=2),
        rollout=SimpleNamespace(
            n_generation=1, multi_turn_config=SimpleNamespace(enable=False)
        ),
        train=SimpleNamespace(
            non_text=True,
            local_dataset=False,
            train_policy=SimpleNamespace(
                variant="grpo", rollout_as_token_ids=False, bypass_reward=False
            ),
        ),
    )
    worker.data_packer = train_packer
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    worker._report_discarded_samples = Mock()
    worker.shutdown_signal, worker.validation_flag = (
        threading.Event(),
        threading.Event(),
    )
    worker.validation_flag.set()
    worker.api_client = SimpleNamespace(
        post_rollout_completion=Mock(), post_validation_report=Mock()
    )
    worker.reward_dispatcher = RewardDispatcher()
    worker.reward_dispatcher.is_remote = False
    with ThreadPoolExecutor(max_workers=2) as reward_pool:

        def reward(payloads, phase, step):
            time.sleep(0.05)
            for payload in payloads:
                payload.rewards = [1.0] * len(payload.completions)
            return payloads, phase, step

        def enqueue(payloads, phase, step, **kwargs):
            worker.reward_dispatcher.task_queue.put(
                reward_pool.submit(reward, payloads, phase, step)
            )

        worker.reward_dispatcher.enqueue_rewards_cal = enqueue

        def fetch(count, queue, validation_step, **kwargs):
            assert validation_step == 7 and scheduler.is_idle()
            assert worker.reward_dispatcher.is_empty()
            reports = worker.api_client.post_rollout_completion.call_args_list
            assert sum(len(call.args[0].payloads) for call in reports) == 3
            queue.put(
                [
                    RLPayload(prompt_idx=i, prompt=f"Validation phase prompt {i}:")
                    for i in range(2)
                ]
            )
            return True

        worker.request_new_prompts = fetch
        try:
            scheduler.put_rollout_batch(
                [
                    RolloutTask(
                        i, RLPayload(prompt_idx=i, prompt=f"Training phase prompt {i}:")
                    )
                    for i in range(3)
                ]
                + [
                    RolloutTask(
                        3, RLPayload(prompt_idx=3, prompt="Training injected failure")
                    )
                ]
            )
            worker.do_validation()
            assert scheduler.is_idle() and scheduler.total_processed == 6
            assert counts == {False: 3, True: 2}
            assert sorted(observed) == [
                (False, 0),
                (False, 1),
                (False, 2),
                (True, 0),
                (True, 1),
            ]
            assert (
                sum(
                    call.args[0]
                    for call in worker._report_discarded_samples.call_args_list
                )
                == 1
            )
            reports = worker.api_client.post_validation_report.call_args_list
            assert sum(len(call.args[0].payloads) for call in reports) == 2
            assert all(
                len(payload.rewards) == 2
                for call in reports
                for payload in call.args[0].payloads
            )
            print(
                "ASYNC_VLLM_PHASE_PASS train=3 failed=1 validation=2 val_n=2 shared_indices=True",
                flush=True,
            )

            # Fail one child after its sibling has entered actual vLLM generation.
            # The prompt cannot become terminal until that child is also drained.
            child_entered, child_finished = threading.Event(), threading.Event()
            original_child = engine._sub_generate_task

            async def child_generate(prompt, params, request_id):
                if request_id.endswith("_0"):
                    while not child_entered.is_set():
                        await asyncio.sleep(0.001)
                    print("INJECTED_CHILD_FAILURE sibling_entered=True", flush=True)
                    raise RuntimeError("injected child generation failure")
                child_entered.set()
                try:
                    result = await original_child(prompt, params, request_id)
                    # Keep the child alive long enough to expose early gather
                    # return even if this tiny real-engine generation is quick.
                    await asyncio.sleep(0.2)
                    return result
                finally:
                    child_finished.set()

            engine._sub_generate_task = child_generate
            engine.sampling_params.n = worker.config.rollout.n_generation = 2
            scheduler.put_rollout(
                RolloutTask(9, RLPayload(prompt_idx=9, prompt="Training child failure"))
            )
            deadline = time.monotonic() + 30
            while not scheduler.is_all_tasks_completed():
                assert time.monotonic() < deadline, "child-drain deadline expired"
                time.sleep(0.01)
            assert child_entered.is_set() and child_finished.is_set()
            worker._stream_generation_collect_results()
            assert scheduler.is_idle() and scheduler.total_processed == 7
            assert (
                sum(
                    call.args[0]
                    for call in worker._report_discarded_samples.call_args_list
                )
                == 3
            )
            print("ASYNC_CHILD_DRAIN_PASS terminal_after_all_children=True", flush=True)
            if args.live_weight_fence:
                from async_live_weight_canary import exercise_live_weight_fence

                engine._sub_generate_task = original_child
                engine.sampling_params.n = worker.config.rollout.n_generation = 1
                exercise_live_weight_fence(worker, engine, scheduler)
        finally:
            scheduler.stop()
            assert not scheduler._worker_thread.is_alive()


if __name__ == "__main__":
    main()
