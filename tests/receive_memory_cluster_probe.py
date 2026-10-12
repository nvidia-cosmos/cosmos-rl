# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real two-GPU receive-memory probe; synthetic payloads, not NDAS training.

Run on an allocated node: python tests/receive_memory_cluster_probe.py --output out.json
Uses the same Redis and NCCL producer as the transport end-to-end tests.
"""

import argparse
import json
import os
import queue
import time
import traceback
import uuid
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp


def wait_key(client, key, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = client.get(key)
        if value is not None:
            return value
        time.sleep(0.01)
    raise TimeoutError(f"No Redis value for {key}")


def worker(rank, port, run_id, batch_size, steps, bounded, results):
    producer = packer = None
    try:
        import redis
        from test_nccl_e2e import _ConsumerPacker
        from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
        from cosmos_rl.utils.payload_transport.receive_memory import storage_bytes
        from cosmos_rl.utils.trajectory import build_trajectory_schema, schema_layout

        os.environ["RANK"] = str(rank)
        os.environ["SLURM_JOB_ID"] = run_id
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        client = redis.Redis(host="127.0.0.1", port=port, decode_responses=True)
        dims = dict(max_steps=1024, obs_dim=4096, action_dim=2)
        _, payload_bytes = schema_layout(build_trajectory_schema(dims))
        budget_bytes = (batch_size + 1) * payload_bytes + 32 if bounded else 0
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name=run_id),
            custom=dict(
                nccl_max_steps=1024,
                nccl_obs_dim=4096,
                nccl_action_dim=2,
                nccl_receive_budget_bytes=budget_bytes,
                nccl_receive_admission_timeout=60,
            ),
        )
        if rank == 0:
            producer = NCCLRolloutMixin()
            producer.setup_nccl(
                replica_id="probe-producer",
                rollout_idx=0,
                redis_client=client,
                config=config,
                device=device,
                sender_rank=0,
                registry_capacity=batch_size + 4,
                **dims,
            )
            for step in range(steps):
                trajectory = {
                    "observations": torch.full((1024, 4096), step + 1.0, device=device),
                    "actions": torch.ones((1024, 2), device=device),
                    "rewards": torch.full((1024,), 0.5, device=device),
                    "episode_length": 1024,
                }
                metas = [
                    producer.write_to_buffer(trajectory) for _ in range(batch_size)
                ]
                assert all(meta is not None for meta in metas)
                client.set(f"{run_id}:batch:{step}", json.dumps(metas), ex=1800)
                wait_key(client, f"{run_id}:ack:{step}")
                del trajectory, metas
            results.put(dict(rank=rank, steps=steps, status="PASS"))
        else:
            packer = _ConsumerPacker()
            packer._setup_nccl_data_packer(
                device=device,
                redis_client=client,
                config=config,
                prefetch_timeout=120,
                recv_timeout=15,
                first_transfer_timeout=60,
            )
            readers = [
                torch.cuda.Stream(device=device),
                torch.cuda.Stream(device=device),
            ]
            metrics = []
            for step in range(steps):
                metas = json.loads(wait_key(client, f"{run_id}:batch:{step}"))
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.monotonic()
                packer.start_prefetch(metas)
                packer.wait_prefetch()
                fetch_seconds = time.monotonic() - start
                assert len(packer._prefetch_cache) == batch_size, "dropped payloads"
                ready = torch.cuda.Event()
                ready.record()
                checks = []
                for meta in metas:
                    first = packer.get_policy_input(rollout_output=meta)
                    second = packer.get_policy_input(rollout_output=meta)
                    assert first is second, "repeated read unexpectedly refetched"
                    for stream in readers:
                        stream.wait_event(ready)
                        with torch.cuda.stream(stream):
                            # View readers model two independent final consumers.
                            checks.append(first["observations"][::2].mean())
                    assert first["episode_length"].item() == 1024
                    assert first["actions"].mean().item() == 1
                    assert first["rewards"].mean().item() == 0.5
                actual = storage_bytes(packer._prefetch_cache.values())
                assert actual == batch_size * payload_bytes
                del first, second
                if bounded:
                    packer.release_prefetch(streams=readers)
                    assert (
                        packer._transport_strategy.receive_memory_stats()[
                            "reserved_bytes"
                        ]
                        == 0
                    )
                else:
                    for stream in readers:
                        stream.synchronize()
                    # Retain the old cache until next collect, as the legacy path does.
                assert all(value.item() == step + 1 for value in checks)
                del checks
                torch.cuda.synchronize(device)
                stats = packer._transport_strategy.receive_memory_stats()
                if bounded:
                    assert stats["peak_tensor_bytes"] <= budget_bytes
                    assert stats["peak_reserved_bytes"] <= budget_bytes
                row = dict(
                    step=step + 1,
                    fetch_seconds=fetch_seconds,
                    allocated_peak=torch.cuda.max_memory_allocated(device),
                    reserved_peak=torch.cuda.max_memory_reserved(device),
                    decoded_bytes=actual,
                    **stats,
                )
                metrics.append(row)
                print(json.dumps(dict(case=run_id, progress=row)), flush=True)
                client.set(f"{run_id}:ack:{step}", "1", ex=1800)
            packer.release_prefetch(streams=readers)
            results.put(
                dict(
                    rank=rank,
                    steps=steps,
                    status="PASS",
                    metrics=metrics,
                    gpu=torch.cuda.get_device_name(device),
                )
            )
    except BaseException:
        results.put(dict(rank=rank, status="FAIL", traceback=traceback.format_exc()))
        raise
    finally:
        if packer is not None:
            packer.shutdown_nccl_data_packer()
        if producer is not None:
            producer.cleanup_nccl()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "need two GPUs"
    import test_nccl_e2e as e2e

    assert e2e._ensure_redis(), "Redis is required; refusing to skip"
    cases = []
    for batch_size in (16, 32, 64):
        for bounded in (False, True):
            run_id = (
                f"receive-memory-{batch_size}-{int(bounded)}-{uuid.uuid4().hex[:8]}"
            )
            ctx = mp.get_context("spawn")
            results = ctx.Queue()
            processes = [
                ctx.Process(
                    target=worker,
                    args=(
                        rank,
                        e2e.REDIS_PORT,
                        run_id,
                        batch_size,
                        args.steps,
                        bounded,
                        results,
                    ),
                )
                for rank in (0, 1)
            ]
            for process in processes:
                process.start()
            deadline = time.monotonic() + 600
            reports = []
            while len(reports) < 2 and time.monotonic() < deadline:
                try:
                    reports.append(results.get(timeout=1))
                except queue.Empty:
                    if any(p.exitcode not in (None, 0) for p in processes):
                        break
            for process in processes:
                process.join(timeout=15)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            case = dict(
                batch_size=batch_size,
                bounded=bounded,
                reports=reports,
                exitcodes=[p.exitcode for p in processes],
            )
            cases.append(case)
            with open(args.output, "w") as output:
                json.dump(cases, output, indent=2)
            assert len(reports) == 2 and all(r["status"] == "PASS" for r in reports), (
                case
            )
            assert all(p.exitcode == 0 for p in processes), case
    # Compare sustained receive peaks, excluding the cold first iteration.
    for baseline, bounded in zip(cases[::2], cases[1::2]):

        def peak(case):
            report = next(r for r in case["reports"] if r["rank"] == 1)
            return max(m["allocated_peak"] for m in report["metrics"][1:])

        assert peak(bounded) < peak(baseline), (
            baseline["batch_size"],
            peak(baseline),
            peak(bounded),
        )
    print(
        "PASS: all six cases completed, data checks passed, bounded receive peaks lower",
        flush=True,
    )


if __name__ == "__main__":
    main()
