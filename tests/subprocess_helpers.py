# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared subprocess teardown helpers for GPU integration tests."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request


def _kill_descendants(parent_pid: int) -> None:
    """Best-effort SIGKILL of ``parent_pid``'s descendant processes (Linux)."""
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            child_pid = int(line)
        except ValueError:
            continue
        _kill_descendants(child_pid)
        try:
            os.kill(child_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def kill_process_group(process):
    """SIGKILL the whole process tree of ``process`` and reap it."""
    if process.poll() is None:
        try:
            _kill_descendants(process.pid)
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    try:
        process.wait(timeout=30)
    except Exception:
        pass


def wait_for_controller_ready(
    testcase, controller_process, port, timeout_s=120, context=""
):
    """Poll ``/api/meta`` until the controller finishes setup or exits."""
    url = f"http://127.0.0.1:{port}/api/meta"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        exit_code = controller_process.poll()
        if exit_code is not None:
            testcase.fail(
                f"Controller exited with code {exit_code} before becoming ready"
                f"{f' ({context})' if context else ''}."
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    testcase.fail(
        f"Controller did not become ready within {timeout_s:.0f}s"
        f"{f' ({context})' if context else ''}."
    )


def wait_all_or_fail(testcase, processes, timeout_s, context):
    """Bounded wait for subprocess trees; always reaps process groups."""
    deadline = time.monotonic() + timeout_s
    try:
        try:
            for process in processes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd=context, timeout=timeout_s)
                process.communicate(timeout=remaining)
                testcase.assertEqual(
                    process.returncode,
                    0,
                    f"Process failed with code: {process.returncode} ({context})",
                )
        except subprocess.TimeoutExpired:
            testcase.fail(
                f"Timed out after {timeout_s:.0f}s waiting for processes to exit "
                f"cleanly ({context})."
            )
    finally:
        for process in processes:
            kill_process_group(process)
