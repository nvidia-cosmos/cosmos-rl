# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/Gloo regression tests for policy shutdown at the final-sync deadline.

Run from the repository root with:
    python tests/test_policy_shutdown_deadline.py

Requires PyTorch with Gloo and loopback socket access, but no GPU or model.
Load the actual worker methods and transport helper from this checkout or the
installed package without importing the trainer's GPU dependencies. Only clocks,
background IO, command handlers, and teardown are substituted. A timeout is a
test failure.
"""

import ast
import datetime
import importlib.util
import multiprocessing
import os
import queue
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

import torch
import torch.distributed as dist


def _run_rank(
    rank, world_size, early_rank, late_command, validation, rendezvous, release, results
):
    """Run one real Gloo rank with deterministic deadline and command arrival."""
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), GLOO_SOCKET_IFNAME="lo"
    )
    root = Path(__file__).resolve().parents[1] / "cosmos_rl"
    if not root.is_dir():
        spec = importlib.util.find_spec("cosmos_rl")
        assert spec is not None and spec.origin, "cosmos_rl is not importable"
        root = Path(spec.origin).resolve().parent
    worker_path = root / "policy/worker/rl_worker.py"
    worker_tree = ast.parse(worker_path.read_text(), filename=str(worker_path))
    methods = [
        method
        for cls in worker_tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "RLPolicyWorker"
        for method in cls.body
        if isinstance(method, ast.FunctionDef)
        and method.name in {"main_loop", "broadcast_command"}
    ]
    transport_path = root / "utils/distributed.py"
    transport_tree = ast.parse(transport_path.read_text(), filename=str(transport_path))
    transport = [
        node
        for node in transport_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "broadcast_object_cpu"
    ]
    transport_namespace = {"os": os, "torch": torch, "dist": dist}
    exec(
        compile(
            ast.Module(body=transport, type_ignores=[]), str(transport_path), "exec"
        ),
        transport_namespace,
    )
    clock_calls = 0
    collective_calls = 0
    handled = []
    command_sent = False

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 1:
            return 0.0
        expiry_call = 4 if early_rank is not None and rank != early_rank else 3
        return 30.001 if clock_calls >= expiry_call else 29.999

    def sleep(_seconds):
        nonlocal command_sent
        # Arrive after an empty shutdown poll, before the next deadline check.
        if late_command and rank == 0 and not command_sent:
            worker.fetch_command_buffer.put_nowait("final-sync")
            command_sent = True

    def broadcast(*args, **kwargs):
        nonlocal collective_calls
        collective_calls += 1
        return transport_namespace["broadcast_object_cpu"](*args, **kwargs)

    def execute(command):
        handled.append(command)
        return command == "end"

    namespace = {
        "GRPOTrainer": object,
        "threading": types.SimpleNamespace(
            Thread=lambda **kwargs: types.SimpleNamespace(start=lambda: None)
        ),
        "time": types.SimpleNamespace(time=clock, sleep=sleep),
        "COSMOS_FINAL_WEIGHT_SYNC_WAIT_S": 30,
        "torch": torch,
        "dist_util": types.SimpleNamespace(broadcast_object_cpu=broadcast),
        "logger": types.SimpleNamespace(info=lambda *args: None),
        "bounded_drain_or_abort": lambda *args: None,
        "os": os,
    }
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(worker_path), "exec"),
        namespace,
    )
    worker = types.SimpleNamespace(
        global_rank=rank,
        parallel_dims=types.SimpleNamespace(pp_cp_tp_coord=(1, 0, 0)),
        config=types.SimpleNamespace(
            validation=types.SimpleNamespace(enable=validation)
        ),
        fetch_command_buffer=queue.Queue(),
        command_buffer=queue.Queue(),
        execute_command=execute,
        train_stream=None,
        replica_name="shutdown-test",
        handle_shutdown=lambda: None,
    )
    worker.broadcast_command = types.MethodType(namespace["broadcast_command"], worker)
    if rank == 0:
        worker.fetch_command_buffer.put_nowait("end")
    error = None
    try:
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=3),
        )
        namespace["main_loop"](worker)
    except Exception:
        error = traceback.format_exc()
    finally:
        results.put(
            {
                "rank": rank,
                "error": error,
                "collective_calls": collective_calls,
                "handled": handled,
            }
        )
        # Keep either peer alive so divergence reports a timeout, not socket closure.
        release.wait(10)
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
class TestPolicyShutdownDeadline(unittest.TestCase):
    """Every rank must exit and handle each accepted command exactly once."""

    def _check_shutdown(
        self, early_rank=None, late_command=False, validation=True, world_size=2
    ):
        """Bound worker lifetime and assert shutdown behavior, not a fixed poll count."""
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="policy-shutdown-test-") as directory:
            release = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_run_rank,
                    args=(
                        rank,
                        world_size,
                        early_rank,
                        late_command,
                        validation,
                        f"file://{directory}/rendezvous",
                        release,
                        results,
                    ),
                )
                for rank in range(world_size)
            ]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 20
                rows = [
                    results.get(timeout=max(0.01, deadline - time.monotonic()))
                    for _ in processes
                ]
                release.set()
                for process in processes:
                    process.join(timeout=3)
                    self.assertEqual(
                        process.exitcode, 0, f"Worker did not exit: {rows}"
                    )
            finally:
                release.set()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=3)
                        if process.is_alive():
                            process.kill()
                            process.join()
                results.close()
        self.assertEqual([row["error"] for row in rows], [None] * world_size, rows)
        self.assertEqual(len({row["collective_calls"] for row in rows}), 1, rows)
        expected = ["end", "final-sync"] if late_command else ["end"]
        for row in rows:
            self.assertEqual(row["handled"], expected, rows)

    def test_aligned_deadlines(self):
        self._check_shutdown()

    def test_rank0_deadline_first(self):
        self._check_shutdown(early_rank=0)

    def test_rank1_deadline_first(self):
        self._check_shutdown(early_rank=1)

    def test_late_command_aligned_deadlines(self):
        self._check_shutdown(late_command=True)

    def test_late_command_rank0_deadline_first(self):
        self._check_shutdown(early_rank=0, late_command=True)

    def test_late_command_rank1_deadline_first(self):
        self._check_shutdown(early_rank=1, late_command=True)

    def test_validation_disabled(self):
        self._check_shutdown(early_rank=0, validation=False)

    def test_single_rank(self):
        self._check_shutdown(world_size=1)


if __name__ == "__main__":
    unittest.main()
