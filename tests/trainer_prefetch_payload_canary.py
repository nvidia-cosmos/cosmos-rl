# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real payloads and stock worker, matched OFF/ON/barrier controls.

torchrun --nproc-per-node=2 tests/trainer_prefetch_payload_canary.py --backend nccl
Also supports two nodes, one rank per node. Rank zero serves fresh depth-bounded
payloads; rank one runs real SGD. The world group is fixture-only, not a policy
mesh. Use trainer_prefetch_cohort_canary.py for multi-policy collective checks.
Outputs JSON timings including cold start and final consumption, per-fetch and
compute spans, useful optimizer updates and sampled/peak CUDA memory. No speedup
threshold: small/barrier workloads may get no benefit. Producer admission is
bounded independently of the consumer's depth-one memory check. UCXX references
are single-use; no control may reuse a reference already consumed by another.
"""

import argparse
import asyncio
from datetime import timedelta
import gc
import json
import os
import socket
import subprocess
import threading
import time
from queue import Queue, Empty
from types import SimpleNamespace
from unittest.mock import patch

import redis
import torch
import torch.distributed as dist

from cosmos_rl.dispatcher.command import DataFetchCommand
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.trainer.prefetch import TrainerPayloadPrefetch
from cosmos_rl.policy.trainer.batching import (
    ExpandedSampleBatching,
    ExpandedTrainingBatch,
)
from cosmos_rl.utils import distributed as dist_util
from cosmos_rl.utils.redis_stream import RedisStreamHandler
from cosmos_rl.dispatcher.status import RolloutStatusManager, PolicyStatus
from dispatch_commit_canary import worker as make_worker
from test_nccl_e2e import _ComposedPacker, _make_trajectory
from test_trainer_payload_prefetch import manager


class PayloadTrainer:
    batching_contract = ExpandedSampleBatching(partial_tail="include")
    trace_mode = None

    def __init__(self, packer, device, compute):
        self.packer = packer
        self.data_packer = packer
        self.device = device
        self.config = SimpleNamespace(
            train=SimpleNamespace(
                train_policy=SimpleNamespace(mini_batch=2, mu_iterations=1)
            )
        )
        self.weight = torch.nn.Parameter(torch.ones((), device=device))
        self.optimizer = torch.optim.SGD([self.weight], lr=0.001)
        self.scheduler_calls = self.updates = 0
        self.expected = 1.0
        self.compute = compute
        self.matrix = torch.ones((1024, 1024), device=device)
        self.spans = []

    def prepare_training_batch(self, rollouts):
        return ExpandedTrainingBatch((tuple(rollouts),))

    def step_expanded_training(self, batch, **kwargs):
        assert len(batch.minibatches) == 1
        return self.step_training(rollouts=batch.minibatches[0], **kwargs)

    def update_lr_schedulers(self, total_steps):
        self.scheduler_calls += 1

    def step_training(self, *, rollouts, current_step, **kwargs):
        if self.trace_mode is None:
            return self._step_training(
                rollouts=rollouts, current_step=current_step, **kwargs
            )
        with torch.profiler.record_function(
            f"{self.trace_mode.upper()}_TRAIN_{current_step}"
        ):
            return self._step_training(
                rollouts=rollouts, current_step=current_step, **kwargs
            )

    def _step_training(self, *, rollouts, current_step, **kwargs):
        if getattr(self.packer._transport_strategy, "_receive_budget", None) is None:
            self.packer.start_prefetch(rollouts)
            self.packer.wait_prefetch()
        else:
            # The bounded expanded entrypoint acquires current payloads before
            # preparation. Fetching again here would be a second batch while
            # the first lease is still in use, even in the OFF control.
            from cosmos_rl.utils.payload_transport.receive_memory import ReceivedBatch

            assert isinstance(self.packer._prefetch_cache, ReceivedBatch)
        started = time.monotonic()
        if hasattr(self, "on_begin"):
            self.on_begin(current_step)
        values = []
        for item in rollouts:
            payload = self.packer.get_policy_input(rollout_output=item.completion)
            assert payload is not None, "Healthy payload must not be silently skipped"
            observations = payload["observations"]
            expected = item.prompt_idx % 4 + 1
            assert torch.all(observations == expected).item()
            values.append(observations.mean())
        self.optimizer.zero_grad()
        mean = torch.stack(values).mean()
        (self.weight * mean).backward()
        for _ in range(self.compute):
            torch.mm(self.matrix, self.matrix)
        self.optimizer.step()
        torch.cuda.current_stream().synchronize()
        self.expected -= (
            0.001 * sum(r.prompt_idx % 4 + 1 for r in rollouts) / len(rollouts)
        )
        torch.testing.assert_close(
            self.weight, torch.full_like(self.weight, self.expected)
        )
        self.updates += 1
        assert self.updates == current_step
        self.spans.append((started, time.monotonic()))
        return {"train_step": current_step}


def independent_control(
    args, mode, instance, client, deliver, memory, reports, weight_group
):
    """Stock controller, readers and training handler; data arrives mid-update.

    The separate two-rank weight channel is a synthetic snapshot publication
    control, not a production P2R/VLA qualification. It shares the GPUs with
    real native payload transfers and publishes before the next optimizer call.
    """
    status, replica = manager(0)
    replica.name = instance.replica_name = f"policy-{mode}"
    status.policy_replicas = {replica.name: replica}
    status.status = {replica.name: PolicyStatus.READY}
    status.total_steps = args.steps
    status.remain_samples_num = status.samples_per_epoch = 2 * args.steps
    status.config.train.prefetch_payloads = mode == "on"
    status.config.train.sync_weight_interval = args.sync_interval
    status.config.train.coalesce_weight_sync = False
    status.config.train.train_policy.allowed_outdated_steps = args.steps + 1
    status.config.rollout = SimpleNamespace(include_stop_str_in_output=False)
    status.policy_init_done = True
    status.should_weight_sync_after_train_ack = (
        lambda step, _: step % args.sync_interval == 0
    )
    rollout_status = RolloutStatusManager()
    rollout_status.rollout_replicas = {}
    rollout_status.all_rollouts_ended = lambda: False
    endpoint = client.connection_pool.connection_kwargs
    handler = RedisStreamHandler([endpoint["host"]], endpoint["port"])
    status.redis_handler = instance.redis_controller = handler
    instance.api_client._report_session_id = "test-session"
    instance.shutdown_signal = threading.Event()
    instance.fetch_command_buffer = Queue()
    instance.teacher_prefetch_queue = Queue()
    instance.kv_store = SimpleNamespace(
        broadcast_command=lambda command, **_: command,
        broadcast_command_bounded=lambda command, **_: command,
    )
    begun, failures, snapshots = Queue(), [], []
    instance.trainer.on_begin = begun.put_nowait

    def publish_weights(_policy, _rollouts, step, _total):
        assert instance.trainer.updates == step
        assert len(instance.api_client.acks) == step
        snapshot = instance.trainer.weight.detach().clone()
        client.rpush(
            "payload-request",
            json.dumps({"mode": mode, "step": step, "weight": snapshot.item()}),
        )
        dist.send(snapshot, dst=0, group=weight_group)
        receipt = client.blpop("weight-reply", timeout=30)
        assert receipt is not None and int(receipt[1]) == step
        snapshots.append((step, snapshot.item()))

    status.trigger_weight_sync = publish_weights

    def admit(step):
        rows = deliver(step, enqueue=False)
        with status._lifecycle_lock:
            status.samples_on_the_fly += len(rows)
            for row in rows:
                status.put_rollout(row)

    def feed():
        while not instance.shutdown_signal.is_set():
            try:
                step = begun.get(timeout=0.05)
            except Empty:
                continue
            if step < args.steps:
                admit(step + 1)

    def guarded(callback):
        try:
            callback()
        except Exception as error:
            failures.append(error)

    threads = [
        threading.Thread(
            target=guarded, args=(lambda: asyncio.run(instance.fetch_command()),)
        ),
        threading.Thread(
            target=guarded, args=(lambda: asyncio.run(instance.fetch_rollouts()),)
        ),
        threading.Thread(target=guarded, args=(feed,)),
    ]
    for thread in threads:
        thread.start()
    try:
        admit(1)
        for step in range(1, args.steps + 1):
            command = instance.fetch_command_buffer.get(timeout=45)
            assert isinstance(command, DataFetchCommand) and command.global_step == step
            assert command.prefetch_next_batch_id is None
            instance.execute_data_fetch(command)
            assert not failures, failures
            status.train_ack(
                *instance.api_client.acks[-1], rollout_status_manager=rollout_status
            )
            client.rpush("payload-retire", json.dumps({"mode": mode, "step": step}))
            reports.append(instance.api_client.acks[-1][-1])
            if mode == "off":
                instance.data_packer.finish_payload_batch(
                    streams=(instance.train_stream,)
                )
            memory.append(torch.cuda.memory_allocated())
        assert status.samples_on_the_fly == 0 and status._payload_lookahead is None
        assert len(snapshots) == args.steps // args.sync_interval
        print(
            "CROSS_SYNC_SNAPSHOTS "
            + json.dumps(
                {"mode": mode, "interval": args.sync_interval, "snapshots": snapshots}
            ),
            flush=True,
        )
    finally:
        if instance.payload_prefetch is not None:
            instance.payload_prefetch.stop_notifications()
        instance.shutdown_signal.set()
        for thread in threads:
            thread.join(35)
        assert all(not thread.is_alive() for thread in threads)
        assert not failures, failures


def consume(args, client, config, device, weight_group):
    results = []
    memory_failures = []
    for mode in ("off", "on", "barrier"):
        packer = _ComposedPacker()
        if args.backend == "nccl":
            from cosmos_rl.utils.payload_transport.nccl.strategy import (
                compose_nccl_transport,
            )

            compose_nccl_transport(
                packer,
                device=device,
                redis_client=client,
                config=config,
                prefetch_timeout=45,
                max_attempts=2,
                recv_timeout=15,
            )
        else:
            from cosmos_rl.utils.payload_transport.ucxx.strategy import (
                compose_ucxx_transport,
            )

            compose_ucxx_transport(
                packer,
                device=device,
                prefetch_timeout=45,
                max_attempts=2,
                read_timeout=15,
            )
        instance, acks = make_worker(device, None)
        instance.api_client.acks = acks
        instance.device = device
        instance.train_stream = torch.cuda.current_stream()
        instance.inter_policy_nccl = SimpleNamespace(
            allreduce=lambda *a, **k: None,
            wait_comm_ready=lambda: None,
            world_size=lambda: 1,
            replica_name_to_rank={"policy-0": 0},
        )
        instance.data_packer = packer
        instance.config.train.train_policy.data_dispatch_as_rank_in_mesh = False
        instance.payload_prefetch = (
            TrainerPayloadPrefetch(packer, device=device) if mode != "off" else None
        )
        instance.trainer = PayloadTrainer(packer, device, args.compute)
        if args.trace_dir:
            instance.trainer.trace_mode = mode
        fetches, lock = [], threading.Lock()
        original = packer._fetch_batch

        def fetch(tasks):
            start = time.monotonic()
            result = original(tasks)
            with lock:
                fetches.append((start, time.monotonic()))
            return result

        packer._fetch_batch = fetch
        if args.receive_budget_bytes:
            assert args.backend == "nccl"
            assert (
                getattr(packer._transport_strategy, "_receive_budget", None) is not None
            )

        def deliver(step, *, enqueue=True):
            client.rpush("payload-request", json.dumps({"mode": mode, "step": step}))
            reply = client.blpop("payload-reply", timeout=30)
            assert reply is not None, "Producer did not publish the requested batch"
            metadata = json.loads(reply[1])
            rows = []
            for index in range(2):
                sample = 2 * (step - 1) + index
                rows.append(
                    Rollout(
                        prompt_idx=sample,
                        completion=metadata[index],
                        reward=0.0,
                        advantage=1.0,
                        weight_version=0,
                    )
                )
            if enqueue:
                for row in rows:
                    instance.data_queue.put(row)
            return rows

        memory, reports = [], []
        try:
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            independent_control(
                args, mode, instance, client, deliver, memory, reports, weight_group
            )
            if instance.payload_prefetch is not None:
                instance.payload_prefetch.drain()
            elapsed = time.monotonic() - started
            # A reusable receive pool may grow lazily when current and next
            # payload lifetimes overlap. Idle cached storage is not a live
            # batch leak. Prove retirement, then evict that cache outside the
            # timed region before checking the terminal allocation footprint.
            strategy = packer._transport_strategy
            budget = getattr(strategy, "_receive_budget", None)
            if budget is not None:
                terminal_credit = budget.snapshot()
                assert terminal_credit["reserved_bytes"] == 0, terminal_credit
                assert terminal_credit["decoded_leased_bytes"] == 0, terminal_credit
            torch.cuda.synchronize()
            terminal_allocated = torch.cuda.memory_allocated()
            pool = getattr(strategy, "_receive_pool", None)
            cached_bytes = 0
            if pool is not None:
                assert not pool._quarantined
                cached_bytes = pool._cached_bytes
                pool.close()
            # UCXX cached mappings retain a bound lease-release callback,
            # forming collectible Python cycles. Measure live storage after
            # those unreachable wrappers are collected, not before GC runs.
            gc.collect()
            torch.cuda.synchronize()
            drained_allocated = torch.cuda.memory_allocated()
            assert terminal_allocated - drained_allocated >= cached_bytes
            assert (
                instance.trainer.updates
                == instance.trainer.scheduler_calls
                == args.steps
            )
            assert packer._prepared_prefetch_future is None
            assert len(fetches) == args.steps
            overlap = sum(
                max(0, min(b, d) - max(a, c))
                for a, b in fetches
                for c, d in instance.trainer.spans
            )
            results.append(
                dict(
                    mode=mode,
                    backend=args.backend,
                    steps=args.steps,
                    elapsed_s=elapsed,
                    updates_per_s=args.steps / elapsed,
                    fetch_compute_overlap_s=overlap,
                    compute=args.compute,
                    sync_interval=args.sync_interval,
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    sampled_allocated=memory,
                    terminal_allocated=terminal_allocated,
                    terminal_cached_bytes=cached_bytes,
                    drained_allocated=drained_allocated,
                    fetch_spans=fetches,
                    compute_spans=instance.trainer.spans,
                    parameter=instance.trainer.weight.item(),
                    reports=reports,
                )
            )
            # Asynchronous samples see wire/decoded phases of one next batch,
            # not a constant footprint. Final drain must retain NO next batch.
            print("PREFETCH_PAYLOAD_SAMPLE " + json.dumps(results[-1]), flush=True)
            batch_bytes = 2 * (256 * 8192 * 4 + 256 * 2 * 4 + 256 * 4 + 8)
            stable = memory[8:] or memory
            if (
                max(stable) - min(stable) > 2 * batch_bytes + 1024 * 1024
                or drained_allocated - min(stable) > 1024 * 1024
            ):
                memory_failures.append(mode)
        finally:
            packer.shutdown_prefetch()
            packer._transport_strategy.shutdown()
    assert all(
        abs(item["parameter"] - results[0]["parameter"]) < 1e-6 for item in results
    )
    assert not memory_failures, (
        f"Memory range exceeded control threshold: {memory_failures}"
    )
    print("PREFETCH_PAYLOAD_PASS " + json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nccl", "ucxx"), required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--compute", type=int, default=64)
    parser.add_argument("--sync-interval", type=int, default=3)
    parser.add_argument(
        "--trace-dir",
        help="Export an optional consumer CPU/CUDA trace; do not benchmark with profiling enabled.",
    )
    parser.add_argument(
        "--receive-budget-bytes",
        type=int,
        default=0,
        help="Require the bounded receive API when nonzero (NCCL only).",
    )
    args = parser.parse_args()
    assert args.steps > 0 and args.compute >= 0 and args.sync_interval > 0
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    weight_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=60))
    server = producer = None
    try:
        endpoint = [None]
        if rank == 0:
            with socket.socket() as probe:
                probe.bind(("0.0.0.0", 0))
                port = probe.getsockname()[1]
            server = subprocess.Popen(
                [
                    "redis-server",
                    "--bind",
                    "0.0.0.0",
                    "--protected-mode",
                    "no",
                    "--port",
                    str(port),
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                ],
                stdout=subprocess.DEVNULL,
            )
            endpoint[0] = (os.environ["MASTER_ADDR"], port)
        dist.broadcast_object_list(endpoint, src=0)
        host, port = endpoint[0]
        client = redis.Redis(host=host, port=port, socket_timeout=35)
        deadline = time.monotonic() + 10
        while True:
            try:
                client.ping()
                break
            except redis.ConnectionError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name="trainer-prefetch-canary"),
            custom={"nccl_receive_budget_bytes": args.receive_budget_bytes},
        )
        if rank == 0:
            dims = dict(max_steps=256, obs_dim=8192, action_dim=2)
            if args.backend == "nccl":
                from cosmos_rl.utils.payload_transport.nccl.producer import (
                    NCCLPayloadProducer,
                )

                producer = NCCLPayloadProducer()
                producer.setup(
                    replica_id="rollout-0",
                    rollout_idx=0,
                    redis_client=client,
                    config=config,
                    sender_rank=0,
                    device=device,
                    **dims,
                )
            else:
                from cosmos_rl.utils.payload_transport.ucxx.producer import (
                    UCXXPayloadProducer,
                )
                from cosmos_rl.utils.payload_transport.ucxx.ucxx_buffer import (
                    UCXXBufferConfig,
                )

                producer = UCXXPayloadProducer()
                # Use an explicit server base port, as deployment callers do.
                # A multi-listener server needs a non-privileged base, not
                # port zero followed by privileged ports one through three.
                with socket.socket() as probe:
                    probe.bind(("0.0.0.0", 0))
                    ucxx_port = probe.getsockname()[1]
                producer.setup(
                    replica_id="rollout-0",
                    port=ucxx_port,
                    config=UCXXBufferConfig(
                        max_entries=8, entry_size_bytes=9 * 1024 * 1024
                    ),
                    **dims,
                )
            outstanding = {}
            served = 0
            while served < 3 * args.steps or outstanding:
                request = client.blpop(
                    ["payload-retire", "payload-request"], timeout=30
                )
                assert request is not None, (
                    "Consumer stopped requesting/retiring batches"
                )
                kind, raw = request
                item = json.loads(raw)
                if "weight" in item:
                    received = torch.empty((), device=device)
                    dist.recv(received, src=1, group=weight_group)
                    expected = 1.0 - 0.001 * sum(
                        ((2 * index) % 4 + 1 + (2 * index + 1) % 4 + 1) / 2
                        for index in range(item["step"])
                    )
                    torch.testing.assert_close(
                        received, torch.full_like(received, expected)
                    )
                    torch.testing.assert_close(
                        received, torch.full_like(received, item["weight"])
                    )
                    client.rpush("weight-reply", str(item["step"]))
                    continue
                key = (item["mode"], item["step"])
                if kind == b"payload-retire":
                    refs = outstanding.pop(key)
                    if args.backend == "nccl":
                        for ref in refs:
                            assert producer._nccl_registry.free(ref["_transfer_id"])
                    continue
                assert key not in outstanding and len(outstanding) < 2
                refs = []
                for index in range(2):
                    sample = 2 * (item["step"] - 1) + index
                    trajectory = _make_trajectory(device, ep_len=256, obs_dim=8192)
                    trajectory["observations"].fill_(sample % 4 + 1)
                    ref = producer.publish(trajectory)
                    assert ref is not None
                    refs.append(ref)
                outstanding[key] = refs
                client.rpush("payload-reply", json.dumps(refs))
                served += 1
        if rank == 1:
            # Fixture ranks are producer/consumer, not a distributed trainer.
            with (
                patch.object(
                    dist_util,
                    "all_reduce_tensor_object_cpu",
                    side_effect=lambda t, **k: t,
                ),
                patch.object(dist, "is_initialized", return_value=False),
            ):
                if args.trace_dir:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profile:
                        consume(args, client, config, device, weight_group)
                    os.makedirs(args.trace_dir, exist_ok=True)
                    profile.export_chrome_trace(
                        os.path.join(args.trace_dir, f"{args.backend}-consumer.json")
                    )
                    print(f"DEVICE_TRACE_PASS backend={args.backend}", flush=True)
                else:
                    consume(args, client, config, device, weight_group)
        dist.barrier()
    finally:
        if producer is not None:
            producer.close()
        if server is not None:
            server.terminate()
            server.wait(timeout=10)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
