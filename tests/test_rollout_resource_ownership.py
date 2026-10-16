# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker ownership and rollback contracts; no controller or model download."""

import asyncio
import threading
import os
import ast
import atexit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import cosmos_rl

from cosmos_rl.policy.config import Config
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.rollout import RolloutWorkerBase
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.rollout.worker import llm_worker, rollout_control
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    RolloutTaskScheduler,
)


class WorkFailure(RuntimeError):
    pass


def test_generation_restart_requires_completed_shutdown():
    from cosmos_rl.rollout.generation_mixin import RolloutGenerationMixin

    started, release = threading.Event(), threading.Event()

    class PreparingBackend(RolloutGenerationMixin):
        config = SimpleNamespace(rollout=SimpleNamespace(prefetch_rollout=True))

        def _prepare_sample(self, payload, **kwargs):
            started.set()
            assert release.wait(15)
            return payload

    backend = PreparingBackend()
    backend.setup_generation()
    old_worker = backend._setup_thread
    try:
        payload = SimpleNamespace(prompt_idx=1)
        backend.submit_setup([payload])
        assert started.wait(5)
        active = next(iter(backend._setup_futures.values()))
        with pytest.raises(TimeoutError, match="still using"):
            backend.shutdown_generation()
        assert not active.cancelled()
        assert active in backend._setup_futures.values()
        with pytest.raises(RuntimeError, match="incomplete"):
            backend.setup_generation()
        assert backend._setup_thread is old_worker
        release.set()
        assert active.result(timeout=5) is payload
        old_worker.join(5)
        assert not old_worker.is_alive()
        backend.shutdown_generation()
        backend.setup_generation()
        assert backend._setup_thread is not old_worker
    finally:
        release.set()
        old_worker.join(5)
        backend.shutdown_generation()


class Backend(RolloutBase):
    def post_init_hook(self, **kwargs):
        self.events = kwargs["events"]
        self.events.append("acquire engine")
        if kwargs.get("fail"):
            raise WorkFailure("backend construction")

    def init_engine(self, *args, **kwargs):
        self._engine_initialized = True

    def rollout_generation(self, *args, **kwargs):
        self.events.append("generate")
        return []

    def get_underlying_model(self):
        return None

    def shutdown(self):
        self.events.append("engine")


def owner(control):
    worker = llm_worker.LLMRolloutWorker.__new__(llm_worker.LLMRolloutWorker)
    worker.rollout_worker = control
    worker._owned_process_group = None
    return worker


def control(events, *, shared=False, fail_work=False, fail_close=False, backend=None):
    worker = rollout_control.DisaggregatedRolloutControlWorker.__new__(
        rollout_control.DisaggregatedRolloutControlWorker
    )
    worker.shutdown_signal = threading.Event()
    worker.shutdown_mp_signal = threading.Event()
    worker.rollout = backend or Backend(Config(), events=events)
    worker.global_rank = 1
    worker.config = SimpleNamespace(distillation=SimpleNamespace(enable=False))
    worker.replica_name = "test"
    worker._cleanup_payload_server = lambda: events.append("producer transport")
    worker.close_payload_transports = lambda: events.append("packer transport")
    worker.unregister_from_controller = lambda: events.append("unregister")

    def close_train():
        events.append("train packer")
        if fail_close:
            raise RuntimeError("close failed")

    worker.data_packer = SimpleNamespace(close=close_train)
    worker.val_data_packer = (
        worker.data_packer
        if shared
        else SimpleNamespace(close=lambda: events.append("validation packer"))
    )

    def work():
        events.append("work")
        worker.rollout.rollout_generation()
        if fail_work:
            raise WorkFailure("work failed")

    worker.main_loop = work
    return worker


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("fail_work", [False, True])
def test_custom_backend_work_and_cleanup(shared, fail_work):
    events = []
    worker = owner(control(events, shared=shared, fail_work=fail_work))
    if fail_work:
        with pytest.raises(WorkFailure, match="work failed"):
            worker.execute()
    else:
        worker.execute()
    worker.destroy_worker()
    assert events == [
        "acquire engine",
        "work",
        "generate",
        "producer transport",
        "engine",
        "packer transport",
        "train packer",
        *([] if shared else ["validation packer"]),
        "unregister",
    ]


@pytest.mark.parametrize("fail_work", [False, True])
def test_cleanup_failure_is_visible_but_never_replaces_work_failure(fail_work):
    events = []
    worker = owner(control(events, fail_work=fail_work, fail_close=True))
    expected = WorkFailure if fail_work else RuntimeError
    with pytest.raises(
        expected, match="work failed" if fail_work else "cleanup failed"
    ):
        worker.execute()
    assert events[-2:] == ["validation packer", "unregister"]
    before = list(events)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        worker.destroy_worker()
    assert events == before


def test_backend_post_init_failure_unwinds_acquisition():
    events = []
    with pytest.raises(WorkFailure, match="backend construction"):
        Backend(Config(), events=events, fail=True)
    assert events == ["acquire engine", "engine"]


def test_real_control_constructor_unwinds_engine(monkeypatch):
    events = []
    config = Config()
    config.rollout.parallelism.dp_shard_size = 1
    dims = SimpleNamespace(dp_shard=1)

    def base_init(self, config, parallel_dims):
        self.config, self.parallel_dims, self.device = config, parallel_dims, None

    monkeypatch.setattr(RolloutWorkerBase, "__init__", base_init)
    monkeypatch.setattr(
        RolloutRegistry,
        "get_rollout_cls",
        lambda _: (lambda *args: Backend(config, events=events)),
    )
    cls = rollout_control.DisaggregatedRolloutControlWorker

    def fail(self):
        raise WorkFailure("after engine acquisition")

    monkeypatch.setattr(cls, "_configure_prompt_queue", fail)
    monkeypatch.setattr(cls, "_cleanup_payload_server", lambda _: None)
    with pytest.raises(WorkFailure, match="after engine acquisition"):
        cls(config, dims)
    assert events == ["acquire engine", "engine"]


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("failure", [None, "work", "construction", "group setup"])
def test_real_process_group_ownership(tmp_path, monkeypatch, borrowed, failure):
    assert not dist.is_initialized()
    events = []
    group_backend = os.environ.get("COSMOS_TEST_GROUP_BACKEND", "gloo")
    if group_backend == "nccl":
        assert torch.cuda.is_available()
        torch.cuda.set_device(0)
    device = "cuda" if group_backend == "nccl" else "cpu"

    def initialize():
        if not dist.is_initialized():
            dist.init_process_group(
                group_backend,
                init_method=f"file://{tmp_path / 'group'}",
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=20),
            )
        if failure == "group setup":
            raise WorkFailure("group setup")

    if borrowed:
        dist.init_process_group(
            group_backend,
            init_method=f"file://{tmp_path / 'group'}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=20),
        )
    config = Config()
    api = Mock()
    api.get_controller_metadata.return_value = {"config": config.model_dump()}
    monkeypatch.setattr(llm_worker, "APIClient", lambda **_: api)
    monkeypatch.setattr(llm_worker, "init_distributed", initialize)
    monkeypatch.setattr(llm_worker.ParallelDims, "from_config", lambda **_: Mock())

    def initialize_control(self, config, dims, **kwargs):
        self.__dict__.update(control(events, fail_work=failure == "work").__dict__)
        shutdown = self.rollout.shutdown

        def dependent_cleanup():
            assert dist.is_initialized()
            value = torch.tensor(3, device=device)
            dist.all_reduce(value)
            assert value.item() == 3
            shutdown()

        self.rollout.shutdown = dependent_cleanup
        if failure == "construction":
            raise WorkFailure("construction")

    monkeypatch.setattr(
        rollout_control.DisaggregatedRolloutControlWorker,
        "_initialize",
        initialize_control,
    )
    try:
        if failure:
            with pytest.raises(WorkFailure):
                llm_worker.LLMRolloutWorker().execute()
        else:
            llm_worker.LLMRolloutWorker().execute()
        assert dist.is_initialized() is borrowed
        if borrowed:
            value = torch.tensor(7, device=device)
            dist.all_reduce(value)
            assert value.item() == 7
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_live_producer_prevents_resource_retirement():
    events = []
    worker = control(events)
    thread = Mock()
    thread.is_alive.return_value = True
    worker.prefetch_thread = thread
    with pytest.raises(RuntimeError, match="prefetch_thread"):
        owner(worker).destroy_worker()
    assert "engine" not in events
    assert "train packer" not in events
    assert "unregister" in events


def test_thread_start_failure_does_not_prevent_cleanup():
    events = []
    worker = control(events)
    worker.background_thread = threading.Thread(target=lambda: None)
    worker.handle_shutdown()
    assert "engine" in events
    assert "train packer" in events
    assert worker._shutdown_resources_released


def test_scheduler_stop_checks_initializing_thread():
    scheduler = RolloutTaskScheduler(Mock(), Mock())
    scheduler._worker_thread = Mock()
    scheduler._worker_thread.is_alive.return_value = True
    with pytest.raises(TimeoutError, match="scheduler"):
        scheduler.stop(wait=True, shutdown_engine=False)
    assert scheduler._stop_requested.is_set()
    scheduler.rollout_engine.shutdown.assert_not_called()


def test_scheduler_initialization_failure_is_returned_to_owner():
    backend = Mock()
    backend.is_engine_initialized.return_value = False
    scheduler = RolloutTaskScheduler(backend, Mock())

    def fail(_):
        raise WorkFailure("initialization hook")

    with pytest.raises(RuntimeError, match="initialization failed") as error:
        scheduler.start(fail, wait_initialized=True)
    assert isinstance(error.value.__cause__, WorkFailure)
    scheduler.stop(shutdown_engine=False)
    assert not scheduler._worker_thread.is_alive()


def test_scheduler_runtime_failure_is_returned_to_owner():
    backend = Mock()
    backend.is_engine_initialized.return_value = False
    scheduler = RolloutTaskScheduler(backend, Mock())
    release = threading.Event()

    async def fail():
        while not release.is_set():
            await asyncio.sleep(0.001)
        raise WorkFailure("generation loop")

    scheduler._worker_loop = fail
    scheduler.start(lambda _: None, wait_initialized=True)
    release.set()
    scheduler._worker_thread.join(timeout=5)
    assert not scheduler._worker_thread.is_alive()
    with pytest.raises(RuntimeError, match="scheduler worker failed") as error:
        scheduler.raise_if_failed()
    assert isinstance(error.value.__cause__, WorkFailure)
    scheduler.stop(shutdown_engine=False)


def test_reward_dispatcher_shutdown_is_bounded():
    dispatcher = RewardDispatcher()
    dispatcher.executor = ThreadPoolExecutor(max_workers=1)
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(timeout=5)

    dispatcher.executor.submit(block)
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(TimeoutError, match="Reward executor"):
            dispatcher.shutdown(timeout_s=0.01)
        assert dispatcher.executor is not None
    finally:
        release.set()
        dispatcher.shutdown(timeout_s=5)
    assert dispatcher.executor is None


def test_reward_dispatcher_shutdown_failure_retains_executor():
    dispatcher = RewardDispatcher()
    executor = Mock()
    executor.shutdown.side_effect = WorkFailure("executor shutdown")
    dispatcher.executor = executor
    with pytest.raises(RuntimeError, match="Reward executor shutdown failed") as error:
        dispatcher.shutdown(timeout_s=5)
    assert isinstance(error.value.__cause__, WorkFailure)
    assert dispatcher.executor is executor


def test_reward_shutdown_timeout_retains_dependent_resources():
    events = []
    worker = control(events)
    worker.reward_dispatcher = SimpleNamespace(
        shutdown=Mock(side_effect=TimeoutError("reward task still running"))
    )
    with pytest.raises(RuntimeError, match="reward executor"):
        worker.handle_shutdown()
    assert "engine" not in events
    assert "train packer" not in events
    assert not worker._shutdown_resources_released
    assert "unregister" in events


def test_scheduler_failure_is_reported_after_safe_resource_release():
    events = []
    worker = control(events)
    worker.scheduler = SimpleNamespace(
        stop=Mock(),
        raise_if_failed=Mock(side_effect=WorkFailure("generation loop")),
    )
    with pytest.raises(RuntimeError, match="scheduler worker"):
        worker.handle_shutdown()
    assert "engine" in events
    assert "train packer" in events
    assert worker._shutdown_resources_released
    assert "unregister" in events


def test_scheduler_failure_probe_is_optional_for_custom_scheduler():
    events = []
    worker = control(events)
    worker.scheduler = SimpleNamespace(stop=Mock())
    worker.handle_shutdown()
    worker.scheduler.stop.assert_called_once_with(wait=True, shutdown_engine=False)
    assert worker._shutdown_resources_released


def test_stop_during_scheduler_initialization_cannot_restart_it():
    backend = Mock()
    backend.is_engine_initialized.return_value = False
    scheduler = RolloutTaskScheduler(backend, Mock())
    entered, release = threading.Event(), threading.Event()

    def initialize(_):
        entered.set()
        assert release.wait(timeout=5)

    scheduler.start(initialize)
    try:
        assert entered.wait(timeout=5)
        scheduler.stop(wait=False, shutdown_engine=False)
    finally:
        release.set()
    scheduler.stop(shutdown_engine=False)
    assert not scheduler._running.is_set()
    assert not scheduler._worker_thread.is_alive()


def test_colocated_control_does_not_own_shared_policy_resources():
    events = []
    worker = control(events)
    worker.colocated = True
    worker.handle_shutdown()
    assert events == ["acquire engine", "producer transport", "unregister"]


def test_failed_backend_shutdown_retains_dependent_packers():
    events = []
    worker = control(events)
    worker.rollout.shutdown = Mock(side_effect=TimeoutError("preparation still alive"))
    with pytest.raises(RuntimeError, match="engine"):
        owner(worker).destroy_worker()
    assert "packer transport" not in events
    assert "train packer" not in events
    assert "unregister" in events


def test_acquired_packer_is_closed_when_later_factory_fails(monkeypatch):
    from cosmos_rl.comm import base
    from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
        DecoderOnlyLLMDataPacker,
    )

    events = []

    class Packer(DecoderOnlyLLMDataPacker):
        def setup(self, config):
            events.append("packer setup")

        def close(self):
            events.append("packer close")

    def failed_factory(config):
        # A factory owns resources not yet returned to the caller.
        try:
            events.append("factory acquire")
            raise WorkFailure("validation factory")
        finally:
            events.append("factory unwind")

    monkeypatch.setattr(
        base, "load_model_config", lambda *a, **k: SimpleNamespace(model_type="test")
    )
    cls = rollout_control.DisaggregatedRolloutControlWorker

    def initialize(self, config, parallel_dims, **kwargs):
        self.config = config
        self.init_data_packer(
            data_packer=lambda _: Packer(), val_data_packer=failed_factory
        )

    monkeypatch.setattr(cls, "_initialize", initialize)
    with pytest.raises(WorkFailure, match="validation factory"):
        cls(Config(), None)
    assert events == [
        "packer setup",
        "factory acquire",
        "factory unwind",
        "packer close",
    ]


def test_failed_constructor_retains_group_when_cleanup_cannot_quiesce(
    tmp_path, monkeypatch
):
    assert not dist.is_initialized()

    def initialize():
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'group'}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=20),
        )

    config = Config()
    api = Mock()
    api.get_controller_metadata.return_value = {"config": config.model_dump()}
    monkeypatch.setattr(llm_worker, "APIClient", lambda **_: api)
    monkeypatch.setattr(llm_worker, "init_distributed", initialize)
    monkeypatch.setattr(llm_worker.ParallelDims, "from_config", lambda **_: Mock())

    def fail(self, *args, **kwargs):
        self.__dict__.update(control([]).__dict__)
        self.prefetch_thread = Mock()
        self.prefetch_thread.is_alive.return_value = True
        raise WorkFailure("construction")

    monkeypatch.setattr(
        rollout_control.DisaggregatedRolloutControlWorker, "_initialize", fail
    )
    try:
        with pytest.raises(WorkFailure, match="construction"):
            llm_worker.LLMRolloutWorker()
        assert dist.is_initialized(), "a live dependent must keep its process group"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("v1", [False, True])
def test_existing_sync_vllm_backend_shutdown(v1):
    pytest.importorskip("vllm")
    from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vLLMRollout

    backend = vLLMRollout.__new__(vLLMRollout)
    resource = Mock()
    backend.rollout_engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=resource if v1 else None, model_executor=resource
        )
    )
    backend.shutdown()
    backend.shutdown()
    resource.shutdown.assert_called_once()


def test_failed_sync_vllm_shutdown_retains_engine():
    pytest.importorskip("vllm")
    from cosmos_rl.rollout.vllm_rollout.vllm_rollout import vLLMRollout

    backend = vLLMRollout.__new__(vLLMRollout)
    resource = Mock()
    resource.shutdown.side_effect = WorkFailure("engine shutdown")
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=resource, model_executor=resource)
    )
    backend.rollout_engine = engine
    with pytest.raises(WorkFailure, match="engine shutdown"):
        backend.shutdown()
    assert backend.rollout_engine is engine


@pytest.mark.parametrize("fail_work", [False, True])
def test_existing_async_vllm_backend_teardown(fail_work):
    pytest.importorskip("vllm")
    from cosmos_rl.rollout.vllm_rollout.vllm_rollout_async import vLLMRolloutAsync

    backend = vLLMRolloutAsync.__new__(vLLMRolloutAsync)
    engine = backend.rollout_engine = Mock()
    backend._engine_initialized = threading.Event()
    # Deliberately unset: a partially initialized engine still needs shutdown.
    backend.rollout_generation = Mock()
    events = []
    worker = owner(control(events, fail_work=fail_work, backend=backend))
    if fail_work:
        with pytest.raises(WorkFailure):
            worker.execute()
    else:
        worker.execute()
    worker.destroy_worker()
    backend.shutdown()
    engine.shutdown.assert_called_once()


def test_failed_async_vllm_shutdown_retains_engine():
    pytest.importorskip("vllm")
    from cosmos_rl.rollout.vllm_rollout.vllm_rollout_async import vLLMRolloutAsync

    backend = vLLMRolloutAsync.__new__(vLLMRolloutAsync)
    engine = Mock()
    engine.shutdown.side_effect = WorkFailure("engine shutdown")
    backend.rollout_engine = engine
    backend._engine_initialized = threading.Event()
    backend._engine_initialized.set()
    with pytest.raises(WorkFailure, match="engine shutdown"):
        backend.shutdown()
    assert backend.rollout_engine is engine
    assert backend._engine_initialized.is_set()


def trt_rollout_class():
    """Exercise the backend lifecycle without importing the optional SDK."""
    path = Path(cosmos_rl.__file__).parent / "rollout/trtllm_rollout/trtllm_rollout.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TRTLLM_Rollout"
    )
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"post_init_hook", "shutdown", "init_engine", "is_engine_initialized"}
    ]
    namespace = {
        "Optional": __import__("typing").Optional,
        "KvCacheConfig": Mock,
        "LLM": Mock,
        "logger": rollout_control.logger,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TRTLLM_Rollout"]


def test_trt_backend_tracks_engine_acquisition_and_failed_shutdown():
    cls = trt_rollout_class()
    backend = cls.__new__(cls)
    backend.config = SimpleNamespace(
        policy=SimpleNamespace(model_name_or_path="model", model_max_length=128),
        rollout=SimpleNamespace(
            parallelism=SimpleNamespace(tp_size=1, pp_size=1),
            gpu_memory_utilization=0.5,
            enable_chunked_prefill=False,
        ),
    )
    backend.rollout_config = backend.config.rollout
    backend.model_config = SimpleNamespace(model_type="dense")
    backend._engine_initialized = False
    backend.init_engine()
    assert backend.is_engine_initialized()
    engine = backend.rollout_engine
    engine.shutdown.side_effect = WorkFailure("engine shutdown")
    with pytest.raises(WorkFailure, match="engine shutdown"):
        backend.shutdown()
    assert backend.rollout_engine is engine
    assert backend.is_engine_initialized()
    engine.shutdown.side_effect = None
    backend.shutdown()
    assert backend.rollout_engine is None
    assert not backend.is_engine_initialized()


def trt_wrapper_class():
    """Exercise installed wrapper methods without importing the optional MPI SDK.

    This is lifecycle fault injection, not native TRT-LLM qualification.
    """
    path = (
        Path(cosmos_rl.__file__).parent
        / "rollout/trtllm_rollout/trtllm_rollout_wrapper.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TRTLLMRolloutWrapper"
    )
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "__init__",
            "handle_shutdown",
            "life_control_loop",
            "_life_control_loop",
            "work",
            "main_loop",
        }
    ]
    namespace = {
        "CosmosConfig": Config,
        "atexit": atexit,
        "torch": torch,
        "threading": threading,
        "logger": rollout_control.logger,
        **{
            name: type(name, (), {})
            for name in (
                "ShutdownInstruction",
                "ValidationInstruction",
                "RolloutWrapperInstruction",
            )
        },
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TRTLLMRolloutWrapper"]


def test_trt_wrapper_idle_receiver_stops_before_engine_close():
    cls = trt_wrapper_class()
    worker = cls.__new__(cls)
    events = []
    worker.shutdown_signal = threading.Event()
    worker.rollout_wrapper_event = threading.Event()
    queue = Mock()
    polled = threading.Event()

    def poll(timeout):
        polled.set()
        worker.shutdown_signal.wait(timeout=0.001)
        return False

    queue.socket.poll.side_effect = poll
    queue.close.side_effect = lambda: events.append("queue")
    worker.cosmos_weight_sync_queue = queue

    def work():
        assert polled.wait(timeout=5)
        raise WorkFailure("work failed")

    worker.main_loop = work
    worker.rollout = SimpleNamespace(shutdown=lambda: events.append("engine"))
    worker.close_payload_transports = lambda: events.append("transport")
    worker.data_packer = worker.val_data_packer = SimpleNamespace(
        close=lambda: events.append("packer")
    )
    worker.unregister_from_controller = lambda: events.append("unregister")
    with pytest.raises(WorkFailure, match="work failed"):
        owner(worker).execute()
    assert not worker.life_control_thread.is_alive()
    worker.handle_shutdown()
    assert events == ["queue", "engine", "transport", "packer", "unregister"]


def test_trt_control_failure_unblocks_registration_wait():
    cls = trt_wrapper_class()
    worker = cls.__new__(cls)
    worker.shutdown_signal = threading.Event()
    worker.rollout_wrapper_event = threading.Event()
    worker.rollout = SimpleNamespace(
        rollout_config=SimpleNamespace(multi_turn_config=SimpleNamespace(enable=False))
    )
    worker.cosmos_replica_name_queue = Mock()
    worker.cosmos_replica_name_queue.socket.poll.side_effect = (
        lambda timeout: worker.shutdown_signal.wait(0.001) and False
    )
    worker.cosmos_weight_sync_queue = Mock()
    worker.cosmos_weight_sync_queue.setup_lazily.side_effect = WorkFailure(
        "control initialization"
    )
    with pytest.raises(WorkFailure, match="control initialization"):
        worker.work()
    worker.life_control_thread.join(timeout=2)
    assert not worker.life_control_thread.is_alive()
    worker.cosmos_weight_sync_queue.close.assert_called_once()


def test_trt_queue_close_failure_is_not_silent_success():
    cls = trt_wrapper_class()
    worker = cls.__new__(cls)
    worker.shutdown_signal = threading.Event()
    worker.shutdown_signal.set()
    worker.cosmos_weight_sync_queue = Mock()
    worker.cosmos_weight_sync_queue.close.side_effect = RuntimeError(
        "queue close failed"
    )
    worker.life_control_loop()
    worker.unregister_from_controller = Mock()
    worker.close_payload_transports = Mock()
    worker.rollout = Mock()
    with pytest.raises(RuntimeError, match="control thread"):
        worker.handle_shutdown()
    assert str(worker._life_control_error) == "queue close failed"
    worker.rollout.shutdown.assert_called_once()
    worker.unregister_from_controller.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
