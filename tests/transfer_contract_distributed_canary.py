# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Supervise the portable transfer faults across two nodes, one GPU per rank.

Run via torchrun with two ranks. Supervisors own HTTP/Redis and child processes;
their independent Gloo group survives injected native worker exit. Child failure
is observed before peer termination. This is test cleanup, not a job launcher
feature or evidence of scheduler-level failure propagation.
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
import tempfile
import threading
import time

import redis
import torch.distributed as dist

from cosmos_rl.dispatcher.transfer_readiness import (
    TransferReadiness,
    TransferReadyRequest,
)


def free_port():
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "ready-delay",
            "ready-missing",
            "queue-uid",
            "accepted-failure",
            "warm-dead-peer",
        ),
    )
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--bounded", action="store_true")
    args = parser.parse_args()
    if args.bounded and not args.prefetch:
        parser.error("bounded leases require prefetch")
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    server = http = child = thread = None
    registry = TransferReadiness()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = TransferReadyRequest(
                **json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(registry.arrive(request)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    try:
        connection = [None]
        if rank == 0:
            http = ThreadingHTTPServer(("", 0), Handler)
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            redis_port = free_port()
            server = subprocess.Popen(
                [
                    "redis-server",
                    "--port",
                    str(redis_port),
                    "--bind",
                    "0.0.0.0",
                    "--protected-mode",
                    "no",
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                ],
                stdout=subprocess.DEVNULL,
            )
            probe = redis.Redis(
                host="127.0.0.1", port=redis_port, protocol=2, socket_timeout=1
            )
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
            connection[0] = (
                os.environ["MASTER_ADDR"],
                redis_port,
                http.server_port,
                free_port(),
            )
        dist.broadcast_object_list(connection, src=0)
        host, redis_port, http_port, worker_port = connection[0]
        env = dict(os.environ, MASTER_ADDR=host, MASTER_PORT=str(worker_port))
        env.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        # Children use a separate process group/store, never the supervisors'.
        command = [
            sys.executable,
            str(Path(__file__).with_name("transfer_contract_canary.py")),
            "--worker",
            "--case",
            args.case,
            "--service-host",
            host,
            "--redis-port",
            str(redis_port),
            "--http-port",
            str(http_port),
        ]
        if args.prefetch:
            command.append("--prefetch")
        if args.bounded:
            command.append("--bounded")
        with tempfile.TemporaryFile(mode="w+t") as output:
            child = subprocess.Popen(
                command, env=env, stdout=output, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + 120
            failed_at = None
            while True:
                statuses = [None, None]
                dist.all_gather_object(statuses, child.poll())
                if all(code is not None for code in statuses):
                    break
                now = time.monotonic()
                if any(code is not None and code != 0 for code in statuses):
                    failed_at = failed_at or now
                if failed_at is not None and now - failed_at > 5:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=5)
                    break
                assert now < deadline, "Canary timed out without a terminal outcome"
                time.sleep(0.1)
            child.wait(timeout=5)
            output.seek(0)
            reports = [None, None]
            dist.all_gather_object(reports, (child.returncode, output.read()))
        combined = "\n".join(report[1] for report in reports)
        if rank == 0:
            print(combined, end="", flush=True)
        if args.case in ("accepted-failure", "warm-dead-peer"):
            assert any(report[0] != 0 for report in reports)
            assert (
                "TRANSFER_INJECT" in combined and "UNSAFE_CONTINUATION" not in combined
            )
            # The actual receiver must reach its own terminal boundary; killing
            # a healthy peer from this supervisor is not a passing fault result.
            assert "[Transport FATAL]" in reports[1][1]
            if args.case == "warm-dead-peer":
                assert "PAYLOAD_PASS rank=1 step=1 exact=True" in combined
        else:
            assert all(report[0] == 0 for report in reports), reports
            if args.case == "ready-missing":
                assert "READINESS_MISSING_PASS no_native_init=True" in combined
            elif args.case == "ready-delay":
                for peer in (0, 1):
                    for warm in (False, True):
                        assert (
                            f"READINESS_PASS rank={peer} warm={warm} parity=True"
                            in combined
                        )
            else:
                assert "QUEUED_UID_EXPIRED captured_uid=True" in combined
                for peer in (0, 1):
                    assert f"PAYLOAD_PASS rank={peer} step=2 exact=True" in combined
        print(
            f"DISTRIBUTED_TRANSFER_PASS rank={rank} case={args.case} prefetch={args.prefetch} bounded={args.bounded}",
            flush=True,
        )
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
        if http is not None:
            http.shutdown()
            http.server_close()
            thread.join(5)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
