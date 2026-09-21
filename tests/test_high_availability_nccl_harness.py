# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the GPU test harness, without controller import effects."""

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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
