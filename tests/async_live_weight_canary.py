# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real vLLM worker/IPC lifetime injection used by async_rollout_phase_canary.

Not a controller or native weight-transfer test: writes use the actual worker
barrier and shared model storage, with delayed CUDA work on both sides.
"""

import asyncio
import threading
import time

import torch

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.vllm_rollout import vllm_rollout_async as backend
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import RolloutTask


def _arm_reader(worker, name):
    probe = worker._get_model().state_dict()[name].view(-1)[:1024]
    expected = probe.clone()
    snapshot = torch.empty_like(probe)
    ready, finished = torch.cuda.Event(), torch.cuda.Event()
    ready.record()
    stream = torch.cuda.Stream()
    stream.wait_event(ready)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(500_000_000)
        snapshot.copy_(probe)
        finished.record()
    worker._cosmos_audit_reader = (stream, finished, expected, snapshot)
    assert not finished.query(), "injected backend read was not pending"
    return True


def _check_reader(worker):
    _, finished, expected, snapshot = worker._cosmos_audit_reader
    assert finished.query(), "backend fence returned before its reader finished"
    assert torch.equal(snapshot, expected), "live weight write corrupted a backend read"
    return True


def _read_probe(worker, name):
    return worker._get_model().state_dict()[name].view(-1)[:16].cpu().tolist()


def _fail_fence(worker):
    raise RuntimeError("injected backend fence failure")


class LiveWeightCanaryExtension(backend.VLLMColocateWorkerExtension):
    """Named test RPCs avoid enabling pickle-based callable deserialization."""

    def _canary_arm_reader(self, name):
        return _arm_reader(self, name)

    def _canary_check_reader(self):
        return _check_reader(self)

    def _canary_read_probe(self, name):
        return _read_probe(self, name)

    def _canary_fail_fence(self):
        return _fail_fence(self)


def install_canary_extension():
    original = backend.AsyncEngineArgs

    def engine_args(**kwargs):
        kwargs["worker_extension_cls"] = (
            "async_live_weight_canary.LiveWeightCanaryExtension"
        )
        return original(**kwargs)

    backend.AsyncEngineArgs = engine_args


def exercise_live_weight_fence(worker, engine, scheduler):
    worker.rollout = engine
    worker.inference_stream = scheduler.stream
    model = engine.get_underlying_model()
    name, parameter = next(iter(model.state_dict().items()))
    assert parameter.is_contiguous() and parameter.numel() >= 1024
    shared = parameter.view(-1)[:1024]
    original = shared.clone()
    increment = torch.ones_like(shared)
    # First-use CUDA kernel loading can synchronize prior work. Warm both
    # write kernels before injecting a delay, or the delay is silently drained
    # before the assertion and never tests the writer's completion boundary.
    shared.add_(increment)
    shared.copy_(original)
    torch.cuda.synchronize()
    native = engine.rollout_engine
    original_rpc = native.collective_rpc
    original_fence = engine.synchronize_generation
    original_add, original_abort = native.add_request, native.abort
    original_params = (
        engine.sampling_params.n,
        engine.sampling_params.max_tokens,
        engine.sampling_params.min_tokens,
        engine.sampling_params.ignore_eos,
    )

    def rpc(method, *args):
        return asyncio.run_coroutine_threadsafe(
            original_rpc(method, args=args), scheduler.get_event_loop()
        ).result(timeout=30)

    def checked_fence(timeout):
        # Device work can outlive a request coroutine, especially after abort.
        # Inject a delayed real read in the actual backend worker process.
        assert rpc("_canary_arm_reader", name) == [True]
        original_fence(timeout)
        assert rpc("_canary_check_reader") == [True]

    engine.synchronize_generation = checked_fence
    try:
        for index, cancelled in enumerate([False, True]):
            admitted, aborted = threading.Event(), threading.Event()

            async def add_request(*args, **kwargs):
                result = await original_add(*args, **kwargs)
                admitted.set()
                return result

            async def abort(*args, **kwargs):
                await original_abort(*args, **kwargs)
                aborted.set()

            native.add_request, native.abort = add_request, abort
            engine.sampling_params.n = 1
            engine.sampling_params.max_tokens = 128
            engine.sampling_params.min_tokens = 128
            engine.sampling_params.ignore_eos = True
            scheduler.put_rollout(
                RolloutTask(
                    100 + index,
                    RLPayload(
                        prompt_idx=100 + index, prompt="Training live-weight fence:"
                    ),
                )
            )
            assert admitted.wait(30), "real backend request was not admitted"
            if cancelled:

                def cancel_active():
                    assert scheduler._active_tasks
                    for task in list(scheduler._active_tasks):
                        task.cancel()

                scheduler.get_event_loop().call_soon_threadsafe(cancel_active)

            written = torch.cuda.Event()
            written.record()
            written.synchronize()
            with worker._paused_async_live_weights():
                assert not scheduler._active_tasks and scheduler.is_paused()
                if cancelled:
                    assert aborted.is_set(), "cancellation did not reach the backend"
                with torch.cuda.stream(worker.inference_stream):
                    torch.cuda._sleep(500_000_000)
                    shared.add_(increment)
                    written.record()
                assert not written.query(), "injected write was not pending"
            assert written.query(), "admission resumed before the write completed"
            assert not scheduler.is_paused()
            assert rpc("_canary_read_probe", name) == [
                (original[:16] + 1).cpu().tolist()
            ]
            results = scheduler.get_all()
            assert len(results) == 1
            assert bool(results[0].result.completions) is not cancelled

            with worker._paused_async_live_weights():
                with torch.cuda.stream(worker.inference_stream):
                    shared.copy_(original)
            assert rpc("_canary_read_probe", name) == [original[:16].cpu().tolist()]
            print(
                f"ASYNC_LIVE_WEIGHT_PASS cancelled={cancelled} backend_read=True delayed_write=True ipc_parity=True",
                flush=True,
            )

        async def failing_rpc(method, *args, **kwargs):
            return await original_rpc(
                "_canary_fail_fence" if method == "synchronize_generation" else method,
                *args,
                **kwargs,
            )

        native.collective_rpc = failing_rpc
        engine.synchronize_generation = original_fence
        try:
            with worker._paused_async_live_weights():
                raise AssertionError("failed backend fence permitted a write")
        except Exception as error:
            # vLLM wraps worker exceptions in its utility-RPC envelope; retain
            # the specific injection check rather than assuming its type.
            assert "injected backend fence failure" in str(error)
        else:
            raise AssertionError("backend failure was swallowed")
        assert scheduler.is_paused() and worker._async_weight_write_failed
        # A later queued prompt cannot run after an uncertain/failed write phase.
        submitted = scheduler.total_processed
        scheduler.put_rollout(
            RolloutTask(999, RLPayload(prompt_idx=999, prompt="Training blocked:"))
        )
        time.sleep(0.05)
        assert scheduler.total_processed == submitted
        print(
            "ASYNC_LIVE_WEIGHT_FAILURE_PASS admission_remains_paused=True", flush=True
        )
    finally:
        native.collective_rpc = original_rpc
        native.add_request, native.abort = original_add, original_abort
        engine.synchronize_generation = original_fence
        (
            engine.sampling_params.n,
            engine.sampling_params.max_tokens,
            engine.sampling_params.min_tokens,
            engine.sampling_params.ignore_eos,
        ) = original_params
