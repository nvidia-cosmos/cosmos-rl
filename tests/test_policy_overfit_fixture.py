# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for cache publication and teardown in the overfit fixture."""

import functools
import os
import signal
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

import test_policy_overfit as overfit
from subprocess_helpers import kill_process_group


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "spawn_error"])
def test_prepares_config_before_launch_and_always_reaps(outcome):
    events = []
    children = []

    class Child:
        def __init__(self, command, **kwargs):
            assert events and events[0] == "config"
            assert isinstance(command, list) and kwargs["start_new_session"]
            assert not kwargs.get("shell", False)
            self.role = "policy" if command[0] == "torchrun" else "controller"
            if self.role == "policy" and outcome == "spawn_error":
                raise OSError("injected spawn failure")
            self.returncode = None
            events.append(self.role)
            children.append(self)

        def communicate(self, timeout):
            assert self.role == "policy" and timeout > 0
            if outcome == "timeout":
                raise subprocess.TimeoutExpired("policy", timeout)
            self.returncode = 7 if outcome == "failure" else 0

    def prepare(model):
        assert model
        events.append("config")

    def reap(child, *, owned_session):
        assert owned_session
        events.append("reap_" + child.role)

    with tempfile.TemporaryDirectory() as directory:
        temporary_file = functools.partial(tempfile.NamedTemporaryFile, dir=directory)
        with (
            patch.object(overfit, "load_model_config", prepare, create=True),
            patch.object(
                overfit.network_util, "find_available_port", return_value=8123
            ),
            patch.object(overfit.subprocess, "Popen", Child),
            patch.object(overfit, "kill_process_group", reap, create=True),
            patch.object(overfit.tempfile, "NamedTemporaryFile", temporary_file),
            patch.dict(os.environ, {}, clear=False),
        ):
            run = overfit.TestPolicyOverfit("test_policy_overfit").test_policy_overfit
            if outcome == "success":
                run()
            else:
                error = OSError if outcome == "spawn_error" else AssertionError
                with pytest.raises(error):
                    run()
    assert events[0] == "config"
    assert events[-1] == "reap_controller"
    assert ["reap_" + child.role for child in reversed(children)] == events[
        -len(children) :
    ]


def test_config_failure_does_not_launch_workers():
    with (
        patch.object(
            overfit,
            "load_model_config",
            side_effect=ValueError("bad config"),
            create=True,
        ),
        patch.object(overfit.network_util, "find_available_port", return_value=8123),
        patch.object(overfit.subprocess, "Popen") as launch,
    ):
        with pytest.raises(ValueError, match="bad config"):
            overfit.TestPolicyOverfit("test_policy_overfit").test_policy_overfit()
    launch.assert_not_called()


def test_owned_group_is_reaped_after_its_leader_exits():
    # The orphan inherits stdout, exactly the pipe that otherwise strands tee.
    code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print('started', flush=True)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "started"
        assert child.wait(timeout=5) == 0
        # A dead leader's group cannot safely be inferred without ownership.
        kill_process_group(child)
        with pytest.raises(subprocess.TimeoutExpired):
            child.communicate(timeout=0.1)
        kill_process_group(child, owned_session=True)
        child.communicate(timeout=5)
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def test_owned_session_rejects_a_child_in_the_callers_group():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(ValueError, match="owned session"):
            kill_process_group(child, owned_session=True)
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=5)
