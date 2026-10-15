# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU readiness/accepted-payload canary with owned control-plane services.

Run directly with Python, not torchrun: the parent owns service cleanup even
when injected native failures exit workers immediately. Requires redis-server.
The readiness HTTP fixture uses the production registry and client, but is not
a full controller. Payload tests use the actual producer, strategy and NCCL.
"""

import argparse
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import redis
import torch
import torch.distributed as dist

from cosmos_rl.collective.collective import P2RCollectiveManager
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.dispatcher.transfer_readiness import (
    TransferReadiness,
    TransferReadyRequest,
)
from cosmos_rl.utils import constant
from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin
from cosmos_rl.utils.payload_transport.nccl.strategy import compose_nccl_transport
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.pynccl import nccl_abort, nccl_abort_all


class BasePacker:
    def get_policy_input(self, sample=None, rollout_output=None, *args, **kwargs):
        return rollout_output


class Packer(PrefetchDataPackerMixin, BasePacker):
    pass


def shared_stream_worker(args, device, rank):
    """One real producer, two warmed consumers, one shared send stream.

    Withhold one consumer's native receive after acceptance, then request work
    from the other consumer. The required outcome is bounded process failure,
    not transparent pair recovery or a promise that the healthy peer completes.
    No fatal callback, native send, or accepted-operation timer is replaced.
    """
    from cosmos_rl.utils import pynccl

    client = redis.Redis(
        host=args.service_host,
        port=args.redis_port,
        protocol=2,
        socket_timeout=2,
        decode_responses=True,
    )
    config = SimpleNamespace(
        logging=SimpleNamespace(experiment_name="shared-stream-contract"),
        custom={"nccl_max_steps": 8, "nccl_obs_dim": 4, "nccl_action_dim": 2},
    )
    producer, packer = None, None
    phase = ["warm"]
    if rank == 0:
        producer = NCCLRolloutMixin()
        producer.setup_nccl(
            replica_id="producer",
            rollout_idx=0,
            redis_client=client,
            config=config,
            device=device,
            sender_rank=0,
            max_steps=8,
            obs_dim=4,
            action_dim=2,
            stream_pool_size=1,
            num_sender_threads=2,
        )
        original_send = producer._send
        original_native_send = pynccl.nccl_send
        shared_stream = producer._nccl_streams.acquire().cuda_stream

        def send(entry, receiver_rank, uid_key, receiver_replica, **kwargs):
            if phase[0] == "fault":
                # _send is reached only after rendezvous accepts the operation.
                print(f"SHARED_ACCEPTED peer={receiver_replica}", flush=True)
            return original_send(
                entry, receiver_rank, uid_key, receiver_replica, **kwargs
            )

        def native_send(*pos, **kwargs):
            assert kwargs["stream"].cuda_stream == shared_stream
            if phase[0] == "fault":
                print("SHARED_NATIVE_POST shared_stream=True", flush=True)
                client.set("shared-native-post", "1")
            return original_native_send(*pos, **kwargs)

        producer._send = send
        pynccl.nccl_send = native_send
    else:
        packer = Packer()
        packer._nccl_dp_receiver_replica = f"consumer{rank}"
        compose_nccl_transport(
            packer,
            device=device,
            redis_client=client,
            config=config,
            prefetch_timeout=40,
            recv_timeout=8,
            first_transfer_timeout=25,
            max_attempts=1,
        )
        if rank == 1 and args.case == "shared-stream-peer-stall":
            original_recv = pynccl.nccl_recv

            def recv(*pos, **kwargs):
                if phase[0] == "fault":
                    print(f"SHARED_STREAM_INJECT t={time.monotonic()}", flush=True)
                    threading.Event().wait(30)
                    raise AssertionError("Accepted receive outlived its deadline")
                return original_recv(*pos, **kwargs)

            pynccl.nccl_recv = recv

    for step in range(2):
        metadata = [None]
        if rank == 0:
            metadata[0] = producer.write_to_buffer(
                {
                    "observations": torch.full((8, 4), step + 1.0, device=device),
                    "actions": torch.ones(8, 2, device=device),
                    "rewards": torch.arange(8, dtype=torch.float32, device=device),
                    "episode_length": 8,
                }
            )
        dist.broadcast_object_list(metadata, 0)
        if step:
            phase[0] = "fault"
        dist.barrier()  # all producer/consumer injection hooks are armed
        if rank:
            if step and rank == 2:
                deadline = time.monotonic() + 10
                while not client.exists("shared-native-post"):
                    assert time.monotonic() < deadline, "First send was not posted"
                    time.sleep(0.005)
            if args.prefetch:
                packer.start_prefetch(metadata)
                packer.wait_prefetch()
            result = packer.get_policy_input(rollout_output=metadata[0])
            torch.testing.assert_close(
                result["observations"], torch.full((8, 4), step + 1.0, device=device)
            )
            torch.testing.assert_close(
                result["actions"], torch.ones(8, 2, device=device)
            )
            torch.testing.assert_close(
                result["rewards"], torch.arange(8, dtype=torch.float32, device=device)
            )
            del result
            packer.release_prefetch()
            if step and args.case == "shared-stream-peer-stall" and rank == 1:
                raise AssertionError("Withheld receive became a successful payload")
            print(f"SHARED_PAYLOAD_PASS rank={rank} step={step} exact=True", flush=True)
        dist.barrier()
    if producer:
        producer.cleanup_nccl(timeout=10)
    if packer:
        packer.close_transport(timeout=10)
    client.close()
    dist.destroy_process_group()


def check_shared_stream_output(output, returncode, case, now):
    """Require warm parity and both accepted peers before crediting containment."""
    for rank in (1, 2):
        assert f"SHARED_PAYLOAD_PASS rank={rank} step=0 exact=True" in output
        assert f"SHARED_ACCEPTED peer=consumer{rank}" in output
    assert "SHARED_NATIVE_POST shared_stream=True" in output
    if case == "shared-stream-peer-stall":
        assert returncode != 0 and "[Transport FATAL]" in output
        injection = next(
            line
            for line in output.splitlines()
            if line.startswith("SHARED_STREAM_INJECT t=")
        )
        elapsed = now - float(injection.split("t=", 1)[1])
        assert 0 <= elapsed < 20, f"Shared-stream containment took {elapsed:.1f}s"
        assert "SHARED_PAYLOAD_PASS rank=1 step=1 exact=True" not in output
        return elapsed
    assert case == "shared-stream-healthy"
    assert returncode == 0, returncode
    for rank in (1, 2):
        assert f"SHARED_PAYLOAD_PASS rank={rank} step=1 exact=True" in output
    return None


def worker(args):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    if args.case.startswith("shared-stream-"):
        assert dist.get_world_size() == 3
        shared_stream_worker(args, device, rank)
        return
    assert dist.get_world_size() == 2
    if args.case.startswith("ready"):
        role = Role.POLICY if rank == 0 else Role.ROLLOUT
        api = APIClient(role, [args.service_host], args.http_port)
        manager = P2RCollectiveManager(
            "policy" if rank == 0 else "rollout",
            SimpleNamespace(world_size=1),
            SimpleNamespace(mode="disaggregated"),
            api,
            role,
        )
        manager.global_rank = 0  # two single-rank replicas, not one two-rank replica
        constant.COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS = 5000
        invalidation = args.case.startswith("ready-abort-")
        previous_index = None
        previous_uid = None
        rounds = 4 if invalidation else (2 if args.case == "ready-delay" else 1)
        for step in range(rounds):
            if invalidation and step == 2:
                side = "source" if rank == 0 else "receiver"
                if args.case in (f"ready-abort-{side}", "ready-abort-both"):
                    assert nccl_abort_all() == 1
                    print(
                        f"P2R_CACHE_ABORT rank={rank} after_completed_step=1",
                        flush=True,
                    )
                dist.barrier()
            commands = [
                PolicyToRolloutUnicastCommand(
                    "policy",
                    "rollout",
                    1,
                    1,
                    ready_deadline=time.time()
                    + (2 if args.case == "ready-missing" else 25),
                )
                if rank == 0
                else None
            ]
            dist.broadcast_object_list(commands, 0)
            command = commands[0]
            if args.case == "ready-missing":
                if rank == 0:
                    try:
                        manager.setup_manager(command)
                    except TimeoutError:
                        assert manager.nccl_comm_cache == {}
                        print("READINESS_MISSING_PASS no_native_init=True", flush=True)
                    else:
                        raise AssertionError("Absent receiver was declared ready")
            else:
                if rank == 1 and args.case == "ready-delay":
                    time.sleep(7)  # deliberately longer than native build budget
                manager.setup_manager(command)
                if invalidation:
                    index = manager.nccl_comm_cache["policy_rollout"]
                    uid = manager.unique_ids_cache["policy_rollout"]
                    if step in (1, 3):
                        assert index == previous_index and uid == previous_uid
                    elif step == 2:
                        assert index != previous_index and uid != previous_uid
                    previous_index, previous_uid = index, uid
                value = torch.full(
                    (4096,), step + 1.0 if rank == 0 else -1.0, device=device
                )
                if rank == 0:
                    manager.send("policy_rollout", value, 0)
                else:
                    manager.recv("policy_rollout", value, 0)
                torch.cuda.synchronize()
                torch.testing.assert_close(value, torch.full_like(value, step + 1.0))
                print(
                    f"READINESS_PASS rank={rank} warm={bool(step)} parity=True",
                    flush=True,
                )
                if invalidation:
                    print(
                        f"P2R_CACHE_PASS rank={rank} step={step} parity=True",
                        flush=True,
                    )
            dist.barrier()
        for comm in manager.nccl_comm_cache.values():
            nccl_abort(comm)
        dist.destroy_process_group()
        return

    client = redis.Redis(
        host=args.service_host,
        port=args.redis_port,
        protocol=2,
        socket_timeout=2,
        decode_responses=True,
    )
    config = SimpleNamespace(
        logging=SimpleNamespace(experiment_name="transfer-contract"),
        custom={
            "nccl_max_steps": 8,
            "nccl_obs_dim": 4,
            "nccl_action_dim": 2,
            "nccl_receive_budget_bytes": 4096 if args.bounded else 0,
        },
    )
    producer, packer = None, None
    injected = False
    if rank == 0:
        producer = NCCLRolloutMixin()
        producer.setup_nccl(
            replica_id="producer",
            rollout_idx=0,
            redis_client=client,
            config=config,
            device=device,
            sender_rank=0,
            max_steps=8,
            obs_dim=4,
            action_dim=2,
        )
        original = producer._send
        original_build = producer._nccl_comm_cache._build_fn
        sends = [0]

        def send(*pos, **kw):
            sends[0] += 1
            if args.case == "queue-uid" and sends[0] == 1:
                time.sleep(2)
                assert not client.keys("*:nccl_uid:*")
                print("QUEUED_UID_EXPIRED captured_uid=True", flush=True)
            if args.case == "warm-dead-peer" and sends[0] == 3:
                print(
                    f"TRANSFER_INJECT case={args.case} t={time.monotonic()}", flush=True
                )
                threading.Event().wait(30)
                raise AssertionError("Accepted-operation deadline did not fire")
            return original(*pos, **kw)

        def build(*pos, **kw):
            if args.case == "accepted-failure":
                print(
                    f"TRANSFER_INJECT case={args.case} t={time.monotonic()}", flush=True
                )
                raise RuntimeError("injected accepted sender initialization failure")
            return original_build(*pos, **kw)

        producer._send = send
        if args.case == "accepted-failure":
            producer._nccl_comm_cache._build_fn = build
    else:
        packer = Packer()
        compose_nccl_transport(
            packer,
            device=device,
            redis_client=client,
            config=config,
            prefetch_timeout=40,
            recv_timeout=3,
            first_transfer_timeout=20,
            max_attempts=1,
        )
        packer._transport_strategy._rendezvous._uid_ttl_s = 1
    for step in range(3):
        metadata = [None]
        if rank == 0:
            metadata[0] = producer.write_to_buffer(
                {
                    "observations": torch.full((8, 4), step + 1.0, device=device),
                    "actions": torch.ones(8, 2, device=device),
                    "rewards": torch.arange(8, dtype=torch.float32, device=device),
                    "episode_length": 8,
                }
            )
        dist.broadcast_object_list(metadata, 0)
        if rank == 1:
            injected = args.case == "accepted-failure" or (
                args.case == "warm-dead-peer" and step == 2
            )
            if args.prefetch:
                packer.start_prefetch(metadata)
                packer.wait_prefetch()
            result = packer.get_policy_input(rollout_output=metadata[0])
            if injected:
                print("UNSAFE_CONTINUATION", flush=True)
                raise AssertionError("Native failure became a normal payload result")
            torch.testing.assert_close(
                result["observations"], torch.full((8, 4), step + 1.0, device=device)
            )
            torch.testing.assert_close(
                result["actions"], torch.ones(8, 2, device=device)
            )
            torch.testing.assert_close(
                result["rewards"], torch.arange(8, dtype=torch.float32, device=device)
            )
            del result
            packer.release_prefetch()
        dist.barrier()
        print(f"PAYLOAD_PASS rank={rank} step={step} exact=True", flush=True)
    if producer:
        producer.cleanup_nccl(timeout=10)
    if packer:
        packer.close_transport(timeout=10)
    client.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "ready-delay",
            "ready-missing",
            "ready-abort-source",
            "ready-abort-receiver",
            "ready-abort-both",
            "queue-uid",
            "accepted-failure",
            "warm-dead-peer",
            "shared-stream-healthy",
            "shared-stream-peer-stall",
        ),
    )
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--service-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=0)
    parser.add_argument("--http-port", type=int, default=0)
    args = parser.parse_args()
    if args.bounded and not args.prefetch:
        parser.error("bounded leases require prefetch")
    if args.bounded and args.case.startswith("shared-stream-"):
        parser.error("shared-stream probe tests the unbounded-memory transport path")
    if args.worker:
        worker(args)
        return
    registry = TransferReadiness()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = TransferReadyRequest(
                **json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(registry.arrive(request)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http_thread = threading.Thread(target=http.serve_forever, daemon=True)
    http_thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen(
        [
            "redis-server",
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
    )
    try:
        probe = redis.Redis(host="127.0.0.1", port=port, socket_timeout=1, protocol=2)
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    probe.ping()
                    break
                except redis.ConnectionError:
                    assert server.poll() is None and time.monotonic() < deadline
                    time.sleep(0.01)
        finally:
            probe.close()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=3"
            if args.case.startswith("shared-stream-")
            else "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            args.case,
            "--redis-port",
            str(port),
            "--http-port",
            str(http.server_port),
        ]
        if args.prefetch:
            command.append("--prefetch")
        if args.bounded:
            command.append("--bounded")
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        output = result.stdout + result.stderr
        print(output, end="", flush=True)
        if args.case.startswith("shared-stream-"):
            elapsed = check_shared_stream_output(
                output, result.returncode, args.case, time.monotonic()
            )
            if elapsed is not None:
                print(
                    f"SHARED_CONTAINMENT_PASS elapsed={elapsed:.3f} pair_recovery=False"
                )
        elif args.case in ("accepted-failure", "warm-dead-peer"):
            assert result.returncode != 0 and "[Transport FATAL]" in output
            assert "TRANSFER_INJECT" in output and "UNSAFE_CONTINUATION" not in output
            if args.case == "accepted-failure":
                # Either the initial poll or the independent peer observer can
                # win. Both must propagate this operation's FAILED outcome.
                assert (
                    "peer reported accepted transfer failure" in output
                    or "Accepted sender failed" in output
                )
            if args.case == "warm-dead-peer":
                assert "PAYLOAD_PASS rank=1 step=1 exact=True" in output
        else:
            assert result.returncode == 0, result.returncode
            if args.case == "ready-delay":
                for rank in (0, 1):
                    for warm in (False, True):
                        assert (
                            f"READINESS_PASS rank={rank} warm={warm} parity=True"
                            in output
                        )
            elif args.case == "ready-missing":
                assert "READINESS_MISSING_PASS no_native_init=True" in output
            elif args.case.startswith("ready-abort-"):
                for rank in (0, 1):
                    for step in range(4):
                        assert (
                            f"P2R_CACHE_PASS rank={rank} step={step} parity=True"
                            in output
                        )
                assert "P2R_CACHE_ABORT" in output
            else:
                assert "QUEUED_UID_EXPIRED captured_uid=True" in output
                for rank in (0, 1):
                    assert f"PAYLOAD_PASS rank={rank} step=2 exact=True" in output
        print(
            f"TRANSFER_CONTRACT_PASS case={args.case} prefetch={args.prefetch} bounded={args.bounded}",
            flush=True,
        )
    finally:
        server.terminate()
        server.wait(timeout=5)
        http.shutdown()
        http.server_close()
        http_thread.join(5)


if __name__ == "__main__":
    main()
