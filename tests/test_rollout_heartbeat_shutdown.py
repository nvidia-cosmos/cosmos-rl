"""Live heartbeat-loop audit during controlled slow engine cleanup, not a backend test."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import pytest
import requests

from cosmos_rl.comm.base import CommMixin
from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.protocol import Role
from cosmos_rl.rollout.worker.rollout_control import DisaggregatedRolloutControlWorker
from cosmos_rl.utils import constant


def heartbeat_child(port, stopped, deadline):
    constant.COSMOS_HEARTBEAT_SEND_INTERVAL = 0.05
    client = APIClient(Role.ROLLOUT, remote_ips=["127.0.0.1"], remote_port=port)
    CommMixin.heartbeat_trigger(
        SimpleNamespace(
            api_client=client,
            replica_name="heartbeat-http-audit",
            _heartbeat_shutdown_deadline=deadline,
        ),
        stopped,
    )


@pytest.mark.parametrize(
    "owner", ["control", "sync", "async", "expired", "expired-error"]
)
def test_actual_heartbeat_posts_continue_during_essential_cleanup(owner):
    received = []
    guard = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with guard:
                received.append((self.path, time.monotonic()))
            body = b'{"message":"ok"}'
            self.send_response(503 if owner == "expired-error" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    context = mp.get_context("spawn")
    stopped = context.Event()
    shutdown_deadline = context.Value("d", 0.0)
    child = context.Process(
        target=heartbeat_child,
        args=(server.server_port, stopped, shutdown_deadline),
        daemon=True,
    )
    child.start()
    cleanup_posts = []
    unregistered = []

    def cleanup(*args, **kwargs):
        began = time.monotonic()
        time.sleep(0.6)
        with guard:
            cleanup_posts.extend(entry for entry in received if entry[1] >= began)

    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            with guard:
                started = len(received) >= 2
            if started:
                break
            assert child.is_alive(), f"heartbeat child exited {child.exitcode}"
            time.sleep(0.02)
        assert started, "heartbeat did not start before cleanup"
        if owner in ("expired", "expired-error"):
            # The child must stop even without parent-side shutdown progress.
            shutdown_deadline.value = time.monotonic() + 0.15
            child.join(timeout=3)
            assert not child.is_alive()
            assert stopped.is_set()
            return
        if owner == "control":
            cleanup()
        else:
            worker = SimpleNamespace(
                replica_name="heartbeat-http-audit",
                shutdown_signal=threading.Event(),
                shutdown_mp_signal=stopped,
                _heartbeat_shutdown_deadline=shutdown_deadline,
                background_thread=None,
                teacher_interact_thread=None,
                heartbeat_thread=child,
                scheduler=SimpleNamespace(stop=cleanup) if owner == "async" else None,
                rollout=SimpleNamespace(shutdown=cleanup),
                unregister_from_controller=lambda: unregistered.append(True),
            )
            DisaggregatedRolloutControlWorker.handle_shutdown(worker)
            assert unregistered == [True]
            assert stopped.is_set()
            assert not child.is_alive()
        assert len(cleanup_posts) >= 3, (
            f"owner={owner}, actual HTTP heartbeats during cleanup={len(cleanup_posts)}, "
            f"stop_requested={stopped.is_set()}"
        )
    finally:
        stopped.set()
        child.join(timeout=3)
        if child.is_alive():
            child.terminate()
            child.join(timeout=3)
        server.shutdown()
        server.server_close()
        serving.join(timeout=3)


@pytest.mark.parametrize("stop_after_failure", [False, True])
def test_heartbeat_checks_stop_before_each_alternative(monkeypatch, stop_after_failure):
    client = APIClient(Role.ROLLOUT, remote_ips=["127.0.0.1"], remote_port=1)
    monkeypatch.setattr(client, "get_alternative_urls", lambda _: ["first", "second"])
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        assert kwargs["timeout"] == constant.COSMOS_CONTROL_HTTP_TIMEOUT
        if url == "first":
            raise requests.Timeout("injected lost heartbeat reply")
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(requests, "post", post)
    client.post_heartbeat(
        "heartbeat-http-audit",
        should_stop=lambda: stop_after_failure and bool(calls),
    )
    assert calls == (["first"] if stop_after_failure else ["first", "second"])


def test_stopped_heartbeat_does_not_issue_request(monkeypatch):
    client = APIClient(Role.ROLLOUT, remote_ips=["127.0.0.1"], remote_port=1)

    def unexpected_post(*args, **kwargs):
        pytest.fail("heartbeat posted after terminal stop")

    monkeypatch.setattr(requests, "post", unexpected_post)
    client.post_heartbeat("heartbeat-http-audit", should_stop=lambda: True)
