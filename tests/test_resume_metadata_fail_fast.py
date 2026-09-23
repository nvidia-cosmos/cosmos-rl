# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.utils.resume import ResumeMetadataMismatch, validate_resume_metadata


@pytest.mark.parametrize(
    "key",
    [
        "step",
        "total_steps",
        "remain_samples_num",
        "checkpoint_path",
        "optimizer_updates",
        "resume_contract",
    ],
)
def test_contract_disagreement_is_fatal(key):
    with pytest.raises(ResumeMetadataMismatch, match=key):
        validate_resume_metadata({key: 1}, {key: 2})


def test_matching_contract_and_empty_fresh_start():
    validate_resume_metadata({}, {})
    validate_resume_metadata({"step": 0}, {"step": 0})


def test_missing_and_extra_contract_fields_report_keys_not_values():
    with pytest.raises(ResumeMetadataMismatch) as error:
        validate_resume_metadata(
            {"step": 3, "private": "secret"}, {"step": 4, "other": "secret"}
        )
    assert "missing=['private']" in str(error.value)
    assert "unexpected=['other']" in str(error.value)
    assert "changed=['step']" in str(error.value)
    assert "secret" not in str(error.value)


def test_fetcher_uses_explicit_validation():
    fetcher = ControllerDataFetcher.__new__(ControllerDataFetcher)
    fetcher.ckpt_extra_info = {"step": 2}
    with pytest.raises(ResumeMetadataMismatch):
        fetcher.validate_after_resume({"step": 1})


def test_validation_survives_python_optimized_mode():
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from cosmos_rl.utils.resume import validate_resume_metadata; validate_resume_metadata({'step': 1}, {'step': 2})",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ResumeMetadataMismatch" in result.stderr


def response(status):
    result = requests.Response()
    result.status_code = status
    return result


def client():
    result = APIClient.__new__(APIClient)
    result.max_retries = 3
    result.get_alternative_urls = Mock(return_value=["http://controller/resume"])
    return result


def test_conflict_is_never_retried(monkeypatch):
    post = Mock(return_value=response(409))
    sleep = Mock(side_effect=AssertionError("must not back off"))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", sleep)
    with pytest.raises(ResumeMetadataMismatch):
        client().post_resume_info({"step": 1})
    post.assert_called_once()
    sleep.assert_not_called()


def test_transient_connection_failure_still_retries(monkeypatch):
    post = Mock(side_effect=[requests.ConnectionError("temporary"), response(200)])
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("cosmos_rl.utils.network_util.time.sleep", Mock())
    client().post_resume_info({"step": 1})
    assert post.call_count == 2


def test_endpoint_sends_conflict_then_exits_nonzero(monkeypatch):
    from cosmos_rl.dispatcher import run_web_panel as panel

    fetcher = ControllerDataFetcher.__new__(ControllerDataFetcher)
    fetcher.ckpt_extra_info = {"step": 2}
    monkeypatch.setattr(panel, "controller", SimpleNamespace(data_fetcher=fetcher))
    exit_process = Mock()
    monkeypatch.setattr(os, "_exit", exit_process)
    result = asyncio.run(
        panel.resume_info(SimpleNamespace(ckpt_extra_info={"step": 1}))
    )
    assert result.status_code == 409
    exit_process.assert_not_called()
    asyncio.run(result.background())
    exit_process.assert_called_once_with(1)


def test_matching_endpoint_does_not_schedule_exit(monkeypatch):
    from cosmos_rl.dispatcher import run_web_panel as panel

    fetcher = ControllerDataFetcher.__new__(ControllerDataFetcher)
    fetcher.ckpt_extra_info = {"step": 0}
    monkeypatch.setattr(panel, "controller", SimpleNamespace(data_fetcher=fetcher))
    result = asyncio.run(
        panel.resume_info(SimpleNamespace(ckpt_extra_info={"step": 0}))
    )
    assert result == {"message": "Resume info received and processed"}


def test_real_controller_exit_follows_conflict_response():
    program = """
import asyncio
from types import SimpleNamespace
from cosmos_rl.dispatcher import run_web_panel as panel
from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
fetcher = ControllerDataFetcher.__new__(ControllerDataFetcher)
fetcher.ckpt_extra_info = {'step': 2}
panel.controller = SimpleNamespace(data_fetcher=fetcher)
async def run():
    response = await panel.resume_info(SimpleNamespace(ckpt_extra_info={'step': 1}))
    async def send(message):
        if message['type'] == 'http.response.start':
            print('STATUS', message['status'], flush=True)
        if message['type'] == 'http.response.body':
            print('BODY', message['body'].decode(), flush=True)
    await response({'type': 'http'}, None, send)
    raise RuntimeError('controller survived fatal disagreement')
asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 1
    assert "STATUS 409" in result.stdout
    assert "resume_metadata_mismatch" in result.stdout
    assert "controller survived" not in result.stderr
