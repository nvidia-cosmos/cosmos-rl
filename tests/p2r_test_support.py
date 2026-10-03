# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP readiness for the preconnected native P2R weight-copy harness.

The harness creates its communicator collectively before exercising weight
transfer. It still must declare all ranks ready through the production client
and controller ledger; a cached communicator is not a readiness bypass.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.command import Command
from cosmos_rl.dispatcher.transfer_readiness import (
    TransferReadiness,
    TransferReadyRequest,
)
from cosmos_rl.utils.api_suffix import COSMOS_API_P2R_READY_SUFFIX


class P2RReadinessServer:
    def __init__(self, command):
        self.command = command
        self.readiness = TransferReadiness()
        ledger = self.readiness

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    if self.path != COSMOS_API_P2R_READY_SUFFIX:
                        raise ValueError("Unexpected readiness route")
                    request = TransferReadyRequest.model_validate_json(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                    response, status = ledger.arrive(request), 200
                except (ValueError, KeyError) as error:
                    response, status = {"error": str(error)}, 400
                body = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def environment(self):
        return {
            "COSMOS_TEST_P2R_READY_PORT": str(self.server.server_port),
            "COSMOS_TEST_P2R_COMMAND": json.dumps(self.command._serialize()),
        }

    def assert_all_ready(self):
        operation = self.readiness._operations[self.command.uuid_value]
        assert operation["error"] is None
        assert set(operation["arrivals"]) == {
            (side, rank)
            for side, size in (
                ("source", self.command.src_replica_size),
                ("receiver", self.command.dst_replica_size),
            )
            for rank in range(size)
        }
        assert not any(arrival[0] for arrival in operation["arrivals"].values())

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "Readiness HTTP fixture did not stop"


def p2r_test_client(role):
    return APIClient(
        role,
        remote_ips=["127.0.0.1"],
        remote_port=int(os.environ["COSMOS_TEST_P2R_READY_PORT"]),
    )


def p2r_test_command():
    # Both torchrun groups receive one parent-issued command, not independently
    # generated operation IDs or almost-equal wall-clock deadlines.
    return Command.deserialize(json.loads(os.environ["COSMOS_TEST_P2R_COMMAND"]))
