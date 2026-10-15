# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the GPU test harness, without controller import effects."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import signal
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).with_name("test_high_availability_nccl.py")


def load_nodes(predicate, namespace):
    # Execute the actual helper/entrypoint without importing GPU dependencies or
    # changing controller environment variables at module import time.
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if predicate(node)]
    assert tree.body
    exec(compile(tree, str(SOURCE), "exec"), namespace)


class Mesh:
    def __init__(self, *members):
        self.replica_name_to_rank = {name: i for i, name in enumerate(members)}


def mesh_helper():
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    namespace = {
        "time": SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep),
        "BuildMeshCommand": Mesh,
    }
    load_nodes(
        lambda node: isinstance(node, ast.FunctionDef)
        and node.name == "wait_for_mesh_command",
        namespace,
    )
    return namespace["wait_for_mesh_command"]


def test_waits_for_exact_membership_not_first_ready_snapshot():
    final = Mesh("a", "b", "c", "d")
    batches = iter([[Mesh("a")], [], [object(), Mesh("a", "b")], [final]])

    def fetch(*, block):
        assert block is False
        return next(batches)

    assert mesh_helper()(fetch, {"a", "b", "c", "d"}) is final


def test_scale_down_ignores_stale_or_same_size_wrong_membership():
    final = Mesh("a", "c", "d")
    commands = [Mesh("a", "b", "c", "d"), Mesh("a", "b", "c"), final]
    assert mesh_helper()(lambda **kw: commands, {"a", "c", "d"}) is final


def test_missing_expected_membership_has_bounded_diagnostic():
    with pytest.raises(TimeoutError, match="last observed.*a"):
        mesh_helper()(lambda **kw: [Mesh("a")], {"a", "b"}, timeout=0.03)


def test_generated_config_disables_resume_as_boolean(tmp_path):
    namespace = {"os": os, "WORK_DIR": str(tmp_path)}
    load_nodes(
        lambda node: isinstance(node, ast.FunctionDef)
        and node.name == "write_train_config",
        namespace,
    )
    path = namespace["write_train_config"]()
    config = tomllib.loads(Path(path).read_text())
    # A string is an explicit checkpoint path, not a false boolean. Fail-fast
    # resume correctly rejects the old fixture's nonexistent path "False".
    assert config["train"]["resume"] is False


def test_controller_launch_owns_the_python_process():
    process = Mock()
    popen = Mock(return_value=process)
    owners = []
    namespace = {
        "os": os,
        "sys": sys,
        "CTRL_PORT": 8010,
        "logger": Mock(),
        "subprocess": SimpleNamespace(Popen=popen),
        "_controller_processes": owners,
    }
    load_nodes(
        lambda node: isinstance(node, ast.FunctionDef)
        and node.name == "launch_controller",
        namespace,
    )
    assert namespace["launch_controller"]("config with spaces.toml") == [process]
    args, kwargs = popen.call_args
    assert args[0] == [
        sys.executable,
        "-m",
        "cosmos_rl.dispatcher.run_web_panel",
        "--port",
        "8010",
        "--config",
        "config with spaces.toml",
    ]
    assert not kwargs.get("shell", False)
    assert owners == [process]


@pytest.mark.parametrize("stalled", [False, True])
def test_cleanup_only_stops_owned_live_controller_and_is_idempotent(stalled):
    running, exited = Mock(), Mock()
    running.poll.return_value = None
    exited.poll.return_value = 0
    if stalled:
        running.wait.side_effect = [subprocess.TimeoutExpired("controller", 10), 0]
    owners = [running, exited]
    namespace = {
        "_controller_processes": owners,
        "signal": signal,
        "subprocess": subprocess,
    }
    load_nodes(
        lambda node: isinstance(node, ast.FunctionDef) and node.name == "cleanup",
        namespace,
    )
    namespace["cleanup"]()
    namespace["cleanup"]()
    running.send_signal.assert_called_once_with(signal.SIGINT)
    if stalled:
        running.kill.assert_called_once_with()
    else:
        running.kill.assert_not_called()
    exited.send_signal.assert_not_called()
    exited.kill.assert_not_called()
    assert not owners


@pytest.mark.parametrize("exit_code", [0, 7])
def test_wrapper_propagates_actual_child_exit_code(exit_code):
    calls = []

    def run(command, *, env):
        calls.append(command)
        assert env["RECURSIVE_ENTRYPOINT"] == "1"
        assert command[0] == "torchrun"
        return subprocess.run([sys.executable, "-c", f"raise SystemExit({exit_code})"])

    namespace = {
        "__name__": "__main__",
        "__file__": str(SOURCE),
        "os": SimpleNamespace(path=os.path, environ={}),
        "sys": sys,
        "subprocess": SimpleNamespace(run=run),
    }
    with pytest.raises(SystemExit) as error:
        load_nodes(lambda node: isinstance(node, ast.If), namespace)
    assert error.value.code == exit_code
    assert len(calls) == 1
