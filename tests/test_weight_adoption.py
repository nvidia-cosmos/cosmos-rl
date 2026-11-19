# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual buffer/writer/adoption entrypoints; CPU ordering and CUDA data checks."""

import os
from pathlib import Path
import subprocess
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.reward.identity import CompletionReporter
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker import weight_sync as ws
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker


class _CPUStream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class _CPUEvent:
    def record(self, stream=None):
        self.stream = stream

    def synchronize(self):
        pass


@pytest.fixture
def device(monkeypatch):
    value = torch.device(os.environ.get("COSMOS_WEIGHT_DEVICE", "cpu"))
    if value.type == "cuda":
        assert torch.cuda.is_available(), "Required CUDA adoption gate cannot skip"
    else:
        monkeypatch.setattr(torch.cuda, "Stream", _CPUStream)
        monkeypatch.setattr(torch.cuda, "Event", _CPUEvent)
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    return value


def make_worker(device, *, tied=True):
    model = torch.nn.Module()
    model.register_parameter(
        "embedding", torch.nn.Parameter(torch.zeros(32, 8, device=device))
    )
    if tied:
        model.register_parameter("head", model.embedding)
        model.register_buffer("offset", model.embedding.detach()[4:12, ::2])
        model.register_buffer("transpose", model.embedding.detach().T)
    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.rollout = SimpleNamespace(get_underlying_model=lambda: model)
    worker.device = device
    worker.inference_stream = torch.cuda.Stream()
    worker.parallel_dims = SimpleNamespace(world_size=1)
    worker.replica_name = "rollout-0"
    worker.current_weight_version = 0
    worker.shutdown_signal = threading.Event()
    worker.shutdown_mp_signal = threading.Event()
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            async_r2r_sync="generation",
            n_generation=1,
            multi_turn_config=SimpleNamespace(enable=False),
        ),
        train=SimpleNamespace(
            non_text=True,
            local_dataset=False,
            train_policy=SimpleNamespace(bypass_reward=False),
        ),
        validation=SimpleNamespace(enable=False, val_before_train=False, freq=1),
    )
    worker.state = SimpleNamespace(weight_synced=lambda: True)
    ws.create_buffer_model(worker, device=device)
    worker._weight_sync_thread = ws.WeightSyncThread(worker)
    worker._execute_p2r_recv = lambda command, stream: _write(
        worker, command.weight_step, stream
    )
    return worker, model


def _write(worker, value, stream):
    with torch.cuda.stream(stream), torch.no_grad():
        worker._buffer_state_dict["embedding"].fill_(value)


def _drain(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def test_queued_rounds_preserve_source_weights_and_version(device, monkeypatch):
    """A later P2R must not overwrite the source of an already accepted R2R."""
    worker, _ = make_worker(device, tied=False)
    worker.replica_name_to_rank = {worker.replica_name: 0, "peer": 1}
    worker.rank_in_rollout_repicas = 0
    wst = worker._weight_sync_thread
    broadcasts = []
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(ws, "r2r_barrier", lambda *args, **kwargs: True)

    def broadcast(worker, source, stream):
        with torch.cuda.stream(stream):
            broadcasts.append(
                (
                    worker._buffer_weight_version,
                    worker._buffer_state_dict["embedding"].detach().clone(),
                )
            )
        if len(broadcasts) == 2:
            wst._stop.set()
        return 1, worker._buffer_state_dict["embedding"].numel() * 4

    monkeypatch.setattr(ws, "do_nccl_broadcast_grouped", broadcast)
    for step in (1, 2):
        command = SimpleNamespace(
            weight_step=step,
            total_steps=3,
            src_replica_name=worker.replica_name,
            dst_replica_names=list(worker.replica_name_to_rank),
            replica_should_stop=lambda: False,
        )
        wst.enqueue_p2r(command)
        wst.enqueue_r2r(command)
    wst._run()
    _drain(device)
    assert not wst._task_failed
    assert [version for version, _ in broadcasts] == [1, 2]
    for step, (_, value) in enumerate(broadcasts, start=1):
        torch.testing.assert_close(value, torch.full_like(value, step))
    assert worker._buffer_weight_version == 2
    assert wst._queue.unfinished_tasks == 0


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("payload flush failed"), SystemExit(86), ws.R2RAborted("canceled")],
)
def test_weight_sync_failure_before_buffer_write_stops_worker(
    device, monkeypatch, failure
):
    worker, _ = make_worker(device, tied=False)
    wst = worker._weight_sync_thread
    wst._execute_p2r(SimpleNamespace(weight_step=1))
    worker.data_packer = SimpleNamespace(flush_pending_sends=Mock(side_effect=failure))
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    wst.enqueue_r2r(SimpleNamespace(weight_step=1))
    wst.start()
    try:
        assert worker.shutdown_signal.wait(timeout=1), (
            "A failed accepted weight-sync command must not leave generation "
            "waiting for a version that will never arrive"
        )
        assert worker.shutdown_mp_signal.is_set()
        assert wst._task_failed
    finally:
        wst._stop.set()
        wst._thread.join(timeout=2)
    assert not wst._thread.is_alive()
    assert wst._queue.unfinished_tasks == 0


def test_accepted_weight_write_has_a_device_completion_deadline():
    script = f"""
import os, sys, threading, time
from contextlib import nullcontext
from types import SimpleNamespace
import torch
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from test_weight_adoption import make_worker, _CPUStream, _CPUEvent
from cosmos_rl.rollout.worker import weight_sync as ws
torch.cuda.Stream = _CPUStream
torch.cuda.Event = _CPUEvent
torch.cuda.stream = lambda stream: nullcontext()
worker, _ = make_worker(torch.device('cpu'), tied=False)
class StuckEvent(_CPUEvent):
    def query(self):
        return False
    def synchronize(self):
        threading.Event().wait(30)
torch.cuda.Event = StuckEvent
worker._weight_sync_thread._execute_p2r(SimpleNamespace(weight_step=1))
print('RETURNED_WITH_UNFINISHED_WEIGHT_WRITE', flush=True)
time.sleep(0.5)
os._exit(99)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "COSMOS_NCCL_TIMEOUT_MS": "50"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 86, result.stdout + result.stderr
    assert "RETURNED_WITH_UNFINISHED_WEIGHT_WRITE" not in result.stdout
    assert "[Transport FATAL]" in result.stderr


def test_weight_sync_device_initialization_failure_stops_owner(device, monkeypatch):
    worker, _ = make_worker(device, tied=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", Mock(side_effect=RuntimeError("device unavailable"))
    )
    wst = worker._weight_sync_thread
    wst.start()
    try:
        assert worker.shutdown_signal.wait(timeout=1)
        assert worker.shutdown_mp_signal.is_set()
        assert wst._task_failed
    finally:
        wst._stop.set()
        wst._thread.join(timeout=2)
    assert not wst._thread.is_alive()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("mode", ["buffer", "sync-full", "sync-trainable"])
def test_native_weight_peer_stall_exits_with_terminal_status(mode):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cosmos_rl.launcher.torchrun",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).with_name("weight_sync_completion_canary.py")),
            "--mode",
            mode,
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 86, output
    context = (
        "weight buffer rollout-1 step 1"
        if mode == "buffer"
        else "live weight update rollout-1"
    )
    assert context in output
    assert "R2R_SOURCE_STALLED_AFTER_READINESS" in output
    assert "UNEXPECTED_WEIGHT_COMPLETION" not in output


def test_storage_aliases_offsets_and_strides_survive_adoption(device):
    worker, model = make_worker(device)
    live = model.state_dict()
    buffer = worker._buffer_state_dict
    assert len({t.untyped_storage()._cdata for t in buffer.values()}) == 1
    assert (
        buffer["head"].untyped_storage()._cdata != live["head"].untyped_storage()._cdata
    )
    for key in buffer:
        assert buffer[key].stride() == live[key].stride()
        assert buffer[key].storage_offset() == live[key].storage_offset()
    worker.weight_inplace_view_map = {
        "renamed_offset": live["offset"][1:3],
        "embedding": live["embedding"],
    }
    ws.redirect_view_map_to_buffer(worker)
    target = worker.weight_inplace_view_map["renamed_offset"]
    assert target.storage_offset() == live["offset"][1:3].storage_offset()
    worker._weight_sync_thread._execute_p2r(SimpleNamespace(weight_step=7))
    assert worker.current_weight_version == 0
    with torch.no_grad():
        ws.sync_buffer_to_live(worker)
    _drain(device)
    assert worker.current_weight_version == 7
    torch.testing.assert_close(model.embedding, torch.full_like(model.embedding, 7))
    torch.testing.assert_close(model.head, model.embedding)
    torch.testing.assert_close(model.offset, model.embedding[4:12, ::2])


@pytest.mark.parametrize("mode", ["generation", "inference"])
@pytest.mark.parametrize("writing", [False, True])
def test_early_ready_cannot_start_generation_before_first_adoption(
    device, mode, writing
):
    worker, _ = make_worker(device)
    worker.config.rollout.async_r2r_sync = mode
    worker.rollout.rollout_generation = Mock(return_value=[])
    assert worker.state.weight_synced()
    if writing:
        worker._buffer_version = 1
        worker._buffer_writing = True
    with pytest.raises(RuntimeError, match="adopted initial weight buffer"):
        worker._call_rollout_generation(payloads=[], is_validation=False)
    worker.rollout.rollout_generation.assert_not_called()
    assert worker._buffer_synced_version == 0

    worker._buffer_writing = False
    worker._weight_sync_thread._execute_p2r(SimpleNamespace(weight_step=7))
    assert worker._call_rollout_generation(payloads=[], is_validation=False) == []
    worker.rollout.rollout_generation.assert_called_once()
    assert (
        worker.rollout.rollout_generation.call_args.kwargs["current_weight_version"]
        == 7
    )
    assert worker._buffer_synced_version > 0


def test_shared_storage_across_dtypes_and_empty_views(device):
    raw = torch.arange(16, dtype=torch.uint8, device=device)
    model = torch.nn.Module()
    model.register_buffer("raw", raw)
    model.register_buffer("words", raw.view(torch.int32))
    model.register_buffer("empty", raw[4:4])
    worker = SimpleNamespace(
        rollout=SimpleNamespace(get_underlying_model=lambda: model)
    )
    ws.create_buffer_model(worker, device=device)
    state = worker._buffer_state_dict
    assert len({t.untyped_storage()._cdata for t in state.values()}) == 1
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(state[key], tensor)
    assert state["empty"].storage_offset() == 4


def test_pending_writer_does_not_block_main_thread_or_publish_partial_buffer(device):
    worker, model = make_worker(device)
    wst = worker._weight_sync_thread
    wst._execute_p2r(SimpleNamespace(weight_step=1))
    entered, release = threading.Event(), threading.Event()
    errors = []

    def receive(command, stream):
        _write(worker, 2, stream)
        entered.set()
        assert release.wait(5)

    worker._execute_p2r_recv = receive

    def run():
        try:
            wst._execute_p2r(SimpleNamespace(weight_step=2))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        with torch.no_grad():
            ws.sync_buffer_to_live(worker)
        _drain(device)
        assert worker.current_weight_version == 0
        assert worker._buffer_version == 1
        assert not torch.count_nonzero(model.embedding)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    with torch.no_grad():
        ws.sync_buffer_to_live(worker)
    _drain(device)
    assert worker.current_weight_version == 2
    assert torch.all(model.embedding == 2)


@pytest.mark.parametrize("writer", ["p2r", "r2r"])
def test_next_write_waits_for_final_adoption_read(device, writer, monkeypatch):
    # Isolate the fence defect from the independent tied-storage defect.
    worker, model = make_worker(device, tied=False)
    wst = worker._weight_sync_thread
    wst._execute_p2r(SimpleNamespace(weight_step=1))
    if device.type == "cuda":
        with torch.cuda.stream(worker.inference_stream):
            torch.cuda._sleep(100_000_000)
    with torch.no_grad():
        ws.sync_buffer_to_live(worker)
    adoption = getattr(worker, "_buffer_adopt_event", None)
    if writer == "p2r":
        wst._execute_p2r(SimpleNamespace(weight_step=2))
    else:
        monkeypatch.setattr(ws, "r2r_barrier", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            ws,
            "do_nccl_broadcast_grouped",
            lambda worker, source, stream: (_write(worker, 2, stream) or (1, 1024)),
        )
        wst._execute_r2r(
            SimpleNamespace(
                weight_step=2,
                src_replica_name="rollout-1",
                dst_replica_names=["rollout-0", "rollout-1"],
                total_steps=10,
                replica_should_stop=lambda: False,
            )
        )
    _drain(device)
    # Writer 2 may finish, but must not corrupt adoption 1's pending read.
    assert torch.all(model.embedding == 1)
    assert torch.all(worker._buffer_state_dict["embedding"] == 2)
    assert worker.current_weight_version == 1
    if device.type == "cpu":
        assert adoption in wst._stream.waits
    with torch.no_grad():
        ws.sync_buffer_to_live(worker)
    _drain(device)
    assert worker.current_weight_version == 2
    assert torch.all(model.embedding == 2)


def test_failed_transfer_poison_is_not_adopted_or_reused(device):
    worker, model = make_worker(device)
    wst = worker._weight_sync_thread

    def failure(command, stream):
        _write(worker, 9, stream)
        raise RuntimeError("injected transfer failure")

    worker._execute_p2r_recv = failure
    with pytest.raises(RuntimeError, match="injected transfer"):
        wst._execute_p2r(SimpleNamespace(weight_step=9))
    assert worker.shutdown_signal.is_set() and worker.shutdown_mp_signal.is_set()
    assert worker._buffer_version == 0 and worker.current_weight_version == 0
    with pytest.raises(RuntimeError, match="failed weight transfer"):
        ws.sync_buffer_to_live(worker)
    with pytest.raises(RuntimeError, match="unavailable"):
        wst._execute_p2r(SimpleNamespace(weight_step=10))
    _drain(device)
    assert not torch.count_nonzero(model.embedding)


@pytest.mark.parametrize("world_size", [2, 4])
def test_unsupported_multirank_mode_rejected_before_worker_setup(
    world_size, monkeypatch
):
    config = SimpleNamespace(rollout=SimpleNamespace(async_r2r_sync="generation"))
    base_init = Mock(side_effect=AssertionError("must reject before base worker init"))
    monkeypatch.setattr(
        DisaggregatedRolloutControlWorker.__bases__[0], "__init__", base_init
    )
    with pytest.raises(ValueError, match="one rank"):
        DisaggregatedRolloutControlWorker(
            config, SimpleNamespace(world_size=world_size)
        )
    base_init.assert_not_called()


@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_generation_reports_oldest_adopted_not_later_received_version(
    device, identified, masked
):
    worker, model = make_worker(device)
    wst = worker._weight_sync_thread
    wst._execute_p2r(SimpleNamespace(weight_step=1))
    worker.should_report = True
    worker.completion_reporter = CompletionReporter("source", 0) if identified else None
    worker.enqueue_teacher_calculation = lambda payloads: payloads
    worker.reward_dispatcher = SimpleNamespace(enqueue_rewards_cal=Mock())
    worker.api_client = SimpleNamespace(post_rollout_completion=Mock(return_value=True))
    payload = RLPayload(prompt_idx=1)

    def generate(**kwargs):
        assert kwargs["current_weight_version"] == 1
        wst._execute_p2r(SimpleNamespace(weight_step=2))
        # Simulate the supported inference-boundary adoption during generation.
        with torch.no_grad():
            ws.sync_buffer_to_live(worker)
        return [
            RolloutResult(
                completions=["result"], completion_trainable=[True] if masked else None
            )
        ]

    worker.rollout.rollout_generation = generate
    with torch.no_grad():
        result = worker._call_rollout_generation(
            payloads=[payload], is_validation=False
        )
    worker._filter_valid_rollout_results_and_report(result, [payload])
    _drain(device)
    assert worker.current_weight_version == 2
    assert result[0].weight_version == 1
    assert payload.weight_version == 1
    assert worker.reward_dispatcher.enqueue_rewards_cal.call_args.args[2] == 1


@pytest.mark.parametrize("rank,local_rank", [(0, 0), (1, 1), (1, 0)])
def test_canary_separates_global_communicator_rank_from_local_device(
    monkeypatch, rank, local_rank
):
    from unittest.mock import Mock, create_autospec
    import weight_adoption_canary as canary

    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    select_device = Mock()
    monkeypatch.setattr(canary.torch.cuda, "set_device", select_device)
    monkeypatch.setattr(canary.dist, "init_process_group", Mock())
    monkeypatch.setattr(canary.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(canary.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(canary, "create_nccl_uid", lambda: [7])
    monkeypatch.setattr(
        canary.dist,
        "broadcast_object_list",
        lambda value, src: value.__setitem__(0, [7]),
    )
    create = create_autospec(canary.create_nccl_comm, return_value=17)
    monkeypatch.setattr(canary, "create_nccl_comm", create)
    assert canary.initialize_transport() == (rank, torch.device("cuda", local_rank), 17)
    select_device.assert_called_once_with(local_rank)
    create.assert_called_once_with([7], rank, 2, timeout_ms=60_000)


def test_trt_wrapper_readiness_follows_native_weight_completion():
    """Execute the real handler without importing TensorRT-LLM/model engines."""
    import ast
    from contextlib import contextmanager, nullcontext
    import cosmos_rl

    path = Path(cosmos_rl.__file__).parent / "rollout/trtllm_rollout/trtllm_worker.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CosmosTRTLLMWorker"
    )
    handler = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "broadcast_to_all_rollout_replica"
    )
    handler.decorator_list = []
    calls = []

    @contextmanager
    def complete(stream, context):
        calls.append("accepted")
        yield
        calls.append("device-complete")

    namespace = dict(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(stream=lambda stream: nullcontext())
        ),
        completed_weight_transfer=complete,
        nccl_broadcast=lambda *args: calls.append("enqueue"),
        RolloutToRolloutBroadcastCommand=object,
        RolloutWrapperInstruction=lambda: "ready",
        List=list,
    )
    exec(
        compile(ast.Module(body=[handler], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    worker = SimpleNamespace(
        _engine_initialized=True,
        replica_name="rollout-0",
        inference_stream=object(),
        rank_in_rollout_repicas=0,
        replica_name_to_rank={"rollout-0": 0, "rollout-1": 1},
        global_commnicator_idex=7,
        get_underlying_model=lambda: SimpleNamespace(
            state_dict=lambda: {"weight": torch.ones(1)}
        ),
        cosmos_state=SimpleNamespace(
            weight_synced=lambda: False,
            set_weight_synced=lambda: calls.append("ready-state"),
        ),
        cosmos_weight_sync_queue=SimpleNamespace(put=lambda value: calls.append(value)),
    )
    command = SimpleNamespace(
        src_replica_name="rollout-0",
        dst_replica_names=list(worker.replica_name_to_rank),
        weight_step=-1,
        replica_should_stop=lambda: False,
    )
    namespace[handler.name](worker, command)
    assert calls == ["accepted", "enqueue", "device-complete", "ready-state", "ready"]


def test_startup_router_cannot_overtake_commands_waiting_for_model_setup(monkeypatch):
    from queue import Queue
    from cosmos_rl.dispatcher.command import (
        PolicyToRolloutUnicastCommand,
        RolloutToRolloutBroadcastCommand,
        StopCommand,
    )
    from cosmos_rl.rollout.worker import rollout_control as control

    worker = object.__new__(DisaggregatedRolloutControlWorker)
    worker.replica_name = "rollout-0"
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(async_r2r_sync="generation", broadcast_all_params=False)
    )
    worker._command_queue = Queue()
    worker.global_rank = 0
    worker.shutdown_signal = threading.Event()
    first_queued, model_started, reader_done = (threading.Event() for _ in range(3))
    accepted, errors = [], []
    commands = []
    for version in (1, 2):
        commands.extend(
            [
                PolicyToRolloutUnicastCommand(
                    "policy",
                    "rollout-0",
                    1,
                    1,
                    weight_step=version,
                    total_steps=3,
                    trainable_only=False,
                ),
                RolloutToRolloutBroadcastCommand(
                    "rollout-0", ["rollout-0"], version, 3, False
                ),
            ]
        )
    commands.append(StopCommand("rollout-0"))

    def incoming():
        yield commands[0].pack()
        first_queued.set()
        assert model_started.wait(2)
        for command in commands[1:]:
            yield command.pack()

    worker.redis_controller = SimpleNamespace(
        subscribe_command=lambda replica: incoming()
    )

    def initialize(load_format):
        worker._weight_sync_thread = SimpleNamespace(
            enqueue_p2r=lambda cmd: accepted.append(("p2r", cmd.weight_step)),
            enqueue_r2r=lambda cmd: accepted.append(("r2r", cmd.weight_step)),
        )
        model_started.set()
        assert reader_done.wait(2)

    worker.lazy_initialize_rollout_engine = initialize
    worker._paused_async_live_weights = nullcontext
    worker.get_rollout_command_handler = lambda cls: {
        PolicyToRolloutUnicastCommand: DisaggregatedRolloutControlWorker.policy_to_rollout_unicast,
        RolloutToRolloutBroadcastCommand: DisaggregatedRolloutControlWorker.broadcast_to_all_rollout_replica,
        StopCommand: lambda owner, cmd: owner.shutdown_signal.set(),
    }[cls]
    monkeypatch.setattr(control.dist_utils, "broadcast_object_cpu", lambda value: value)
    monkeypatch.setattr(control.constant, "COSMOS_ROLLOUT_CMD_WAIT_TIMES", 1)
    monkeypatch.setattr(control.constant, "COSMOS_ROLLOUT_CMD_WAIT_INTERVAL", 0)

    def read():
        try:
            worker.query_command_from_controller()
        except BaseException as error:
            errors.append(error)
        finally:
            reader_done.set()

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert first_queued.wait(2)
        worker.consume_command(timeout=3)
    finally:
        worker.shutdown_signal.set()
        model_started.set()
        reader.join(3)
    assert not reader.is_alive() and not errors
    assert accepted == [("p2r", 1), ("r2r", 1), ("p2r", 2), ("r2r", 2)]
    assert worker._wst_direct_routing_ready
    worker.shutdown_signal.clear()
    later = [
        PolicyToRolloutUnicastCommand(
            "policy", "rollout-0", 1, 1, weight_step=3, total_steps=4
        ),
        RolloutToRolloutBroadcastCommand("rollout-0", ["rollout-0"], 3, 4, False),
        StopCommand("rollout-0"),
    ]
    worker.redis_controller.subscribe_command = lambda replica: [
        cmd.pack() for cmd in later
    ]
    worker.query_command_from_controller()
    assert accepted[-2:] == [("p2r", 3), ("r2r", 3)]
    assert worker._command_queue.qsize() == 1
    assert isinstance(worker._command_queue.get_nowait(), StopCommand)


def test_one_member_startup_publishes_ready_after_staged_transfer(device, monkeypatch):
    from cosmos_rl.rollout import State

    worker, model = make_worker(device, tied=False)
    worker.state = State()
    worker.config.rollout.broadcast_all_params = False
    worker.replica_name_to_rank = {worker.replica_name: 0}
    wst = worker._weight_sync_thread
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    mark_ready = worker.state.set_weight_synced

    def ready():
        mark_ready()
        wst._stop.set()

    worker.state.set_weight_synced = ready
    command = SimpleNamespace(
        weight_step=7,
        total_steps=8,
        src_replica_name=worker.replica_name,
        dst_replica_names=[worker.replica_name],
        trainable_only=False,
        replica_should_stop=lambda: False,
    )
    wst.enqueue_p2r(command)
    worker._execute_rollout_broadcast(command)
    assert not worker.state.weight_synced()
    wst.start()
    try:
        wst._thread.join(timeout=3)
        assert not wst._thread.is_alive(), "One-member R2R did not publish readiness"
    finally:
        wst._stop.set()
        wst._thread.join(timeout=2)
    assert worker.state.weight_synced() and not wst._task_failed
    assert worker._buffer_weight_version == 7 and worker._buffer_version == 2
    assert wst._queue.unfinished_tasks == 0
    with torch.no_grad():
        ws.sync_buffer_to_live(worker)
    _drain(device)
    torch.testing.assert_close(model.embedding, torch.full_like(model.embedding, 7))
