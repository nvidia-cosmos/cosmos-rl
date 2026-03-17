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

import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs

import numpy as np
import toml
import torch

from cosmos_rl.utils import network_util


FAKE_PICKSCORE_VALUES = [
    0.6708330512046814,
    0.6815550923347473,
    0.713772714138031,
    0.6735506057739258,
    0.6758633852005005,
    0.7138779759407043,
    0.6701474785804749,
    0.6949127912521362,
    0.6782714128494263,
    0.71278977394104,
    0.712783932685852,
    0.6760029792785645,
    0.7097694873809814,
    0.7138543725013733,
    0.7137541770935059,
    0.7107489109039307,
    0.7137821316719055,
    0.6759352684020996,
    0.6772384643554688,
    0.6808035373687744,
    0.6688560843467712,
    0.7138133645057678,
    0.6761638522148132,
    0.7128953337669373,
]


def _build_fake_scores(batch_size: int) -> list[float]:
    if batch_size <= 0:
        return []
    repeats = (batch_size + len(FAKE_PICKSCORE_VALUES) - 1) // len(
        FAKE_PICKSCORE_VALUES
    )
    return (FAKE_PICKSCORE_VALUES * repeats)[:batch_size]


class _MockRewardHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.request_records: dict[str, dict[str, Any]] = {}
        self.records_lock = threading.Lock()


class _MockRewardServiceHandler(BaseHTTPRequestHandler):
    server: _MockRewardHTTPServer

    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path == "/api/reward/enqueue":
            self._handle_enqueue()
            return
        if self.path in {"/api/reward/pull", "/api/rewardpull"}:
            self._handle_pull()
            return
        self._send_json(404, {"error": f"Unsupported endpoint: {self.path}"})

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(content_length)

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_enqueue(self) -> None:
        body = self._read_body()
        metadata_bytes, separator, array_bytes = body.partition(b"\n")
        if not separator:
            self._send_json(
                400, {"error": "Expected payload format: <json>\\n<npy-bytes>"}
            )
            return

        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            mm_array = np.load(io.BytesIO(array_bytes), allow_pickle=False)
        except Exception as exc:
            self._send_json(400, {"error": f"Failed to parse enqueue payload: {exc}"})
            return

        request_uuid = str(uuid.uuid4())
        reward_fn_names = list(metadata.get("reward_fn", {}).keys()) or ["pickscore"]
        input_info = {
            "shape": list(mm_array.shape),
            "dtype": f"torch.{mm_array.dtype}",
            "min": f"{float(mm_array.min()):.3f}" if mm_array.size else "0.000",
            "max": f"{float(mm_array.max()):.3f}" if mm_array.size else "0.000",
        }
        with self.server.records_lock:
            self.server.request_records[request_uuid] = {
                "batch_size": int(mm_array.shape[0]) if mm_array.ndim > 0 else 0,
                "input_info": input_info,
                "reward_fn_names": reward_fn_names,
            }
        self._send_json(200, {"uuid": request_uuid, "replica_id": "mock-replica-0"})

    def _handle_pull(self) -> None:
        body = self._read_body()
        try:
            request_data = {
                key: values[0]
                for key, values in parse_qs(body.decode("utf-8")).items()
                if values
            }
        except UnicodeDecodeError as exc:
            self._send_json(400, {"error": f"Failed to parse pull payload: {exc}"})
            return

        request_uuid = request_data.get("uuid")
        reward_type = request_data.get("type", "pickscore")
        if not request_uuid:
            self._send_json(400, {"error": "Missing required field: uuid"})
            return

        with self.server.records_lock:
            record = self.server.request_records.get(request_uuid)
        if record is None:
            self._send_json(404, {"error": f"Unknown uuid: {request_uuid}"})
            return

        fake_scores = _build_fake_scores(record["batch_size"])
        score_keys = set(record["reward_fn_names"])
        score_keys.add(reward_type)
        response_json = {
            "scores": {key: list(fake_scores) for key in sorted(score_keys)},
            "input_info": record["input_info"],
            "duration": "0.61",
            "decoded_duration": "0.00",
            "type": reward_type,
        }
        self._send_json(200, response_json)


class MockedRewardService:
    def __init__(self, host: str = "127.0.0.1", start_port: int = 8080):
        self.host = host
        self.start_port = start_port
        self.port: Optional[int] = None
        self._server: Optional[_MockRewardHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        assert self.port is not None, "Mocked reward service has not been started."
        return f"http://localhost:{self.port}"

    @property
    def enqueue_url(self) -> str:
        return f"{self.base_url}/api/reward/enqueue"

    @property
    def pull_url(self) -> str:
        return f"{self.base_url}/api/reward/pull"

    @property
    def rewardpull_url(self) -> str:
        return f"{self.base_url}/api/rewardpull"

    def start(self) -> "MockedRewardService":
        if self._server is not None:
            return self

        self.port = network_util.find_available_port(self.start_port)
        self._server = _MockRewardHTTPServer(
            (self.host, self.port), _MockRewardServiceHandler
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mocked_reward_service",
            daemon=True,
        )
        self._thread.start()
        self._wait_until_ready()
        return self

    def _wait_until_ready(self, timeout_s: float = 5.0) -> None:
        assert self.port is not None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise TimeoutError(
            f"Timed out waiting for mocked reward service on port {self.port}."
        )

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> "MockedRewardService":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def _build_cosmos_rl_command(config_path: str) -> list[str]:
    cli_path = shutil.which("cosmos-rl")
    if cli_path is not None:
        return [
            cli_path,
            "--config",
            config_path,
            "cosmos_rl.tools.dataset.diffusion_nft",
        ]
    return [
        sys.executable,
        "-m",
        "cosmos_rl.launcher.launch_all",
        "--config",
        config_path,
        "cosmos_rl.tools.dataset.diffusion_nft",
    ]


def _prepare_local_pickscore_dataset(cache_dir: str) -> None:
    dataset_dir = os.path.join(cache_dir, "diffusion_nft", "pickscore")
    os.makedirs(dataset_dir, exist_ok=True)
    prompts = [
        "A photo of a small red cube on a wooden table.",
        "A watercolor painting of mountains under a blue sky.",
    ]
    for split in ("train", "test"):
        with open(
            os.path.join(dataset_dir, f"{split}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(prompts) + "\n")


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _attach_live_output_stream(
    process: subprocess.Popen,
) -> tuple[list[str], threading.Thread]:
    assert process.stdout is not None
    output_lines: list[str] = []

    def _reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

    reader_thread = threading.Thread(
        target=_reader,
        name="diffusion_rl_e2e_subprocess_output",
        daemon=True,
    )
    reader_thread.start()
    return output_lines, reader_thread


def _make_smoke_test_config(base_config_path: str, tmpdir: str) -> str:
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)

    config["logging"]["logger"] = ["console"]
    config["train"]["epoch"] = 1
    config["train"]["max_num_steps"] = 1
    config["train"]["train_batch_per_replica"] = 4
    config["train"]["output_dir"] = os.path.join(tmpdir, "outputs")
    config["train"]["ema_enable"] = True
    config["train"]["ckpt"]["enable_checkpoint"] = False
    config["train"]["ckpt"]["export_safetensors"] = False
    config["train"]["ckpt"]["upload_hf"] = False
    config["train"]["ckpt"]["upload_s3"] = False

    config["policy"]["lora"]["r"] = 4
    config["policy"]["lora"]["lora_alpha"] = 8
    config["policy"]["parallelism"]["n_init_replicas"] = 1
    config["policy"]["parallelism"]["tp_size"] = 1
    config["policy"]["parallelism"]["dp_shard_size"] = 1
    config["policy"]["diffusers"]["inference_size"] = [512, 512]
    config["policy"]["diffusers"]["sample"]["num_steps"] = 1
    config["policy"]["diffusers"]["sample"]["eval_num_steps"] = 1

    config["rollout"]["n_generation"] = 2
    config["rollout"]["parallelism"]["n_init_replicas"] = 1
    config["rollout"]["parallelism"]["tp_size"] = 1
    config["rollout"]["parallelism"]["dp_shard_size"] = 1

    config["train"]["train_policy"]["dataset"]["name"] = "pickscore"
    config["train"]["train_policy"]["mini_batch"] = 4
    config["validation"]["enable"] = False
    config["validation"]["dataset"]["name"] = "pickscore"

    config_path = os.path.join(tmpdir, "diffusion_rl_e2e.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)
    return config_path


class TestDiffusionRLE2E(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for diffusion RL e2e."
    )
    def test_diffusion_rl_with_mocked_reward_service(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_config_path = os.path.join(
            repo_root,
            "configs",
            "stable-diffusion-3-5",
            "stable-diffusion-3-5-medium-nft.toml",
        )
        timeout_s = int(os.environ.get("COSMOS_DIFFUSION_RL_E2E_TIMEOUT_S", "1800"))

        with tempfile.TemporaryDirectory(prefix="diffusion_rl_e2e_") as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            _prepare_local_pickscore_dataset(cache_dir)
            config_path = _make_smoke_test_config(base_config_path, tmpdir)

            with MockedRewardService(start_port=8080) as reward_service:
                env = os.environ.copy()
                env["COSMOS_CACHE_DIR"] = cache_dir
                env["REMOTE_REWARD_ENQUEUE_URL"] = reward_service.enqueue_url
                env["REMOTE_REWARD_FETCH_URL"] = reward_service.pull_url
                env["REMOTE_REWARD_TOKEN"] = "mock-token"
                env["COSMOS_DISABLE_REMOTE_REWARD_USE_REPLICA"] = "1"
                env["WANDB_MODE"] = "disabled"
                env["TOKENIZERS_PARALLELISM"] = "false"
                env.setdefault("CUDA_VISIBLE_DEVICES", "0")

                process = subprocess.Popen(
                    _build_cosmos_rl_command(config_path),
                    cwd=repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                output_lines, output_thread = _attach_live_output_stream(process)
                try:
                    process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    _terminate_process_group(process)
                    output_thread.join(timeout=10.0)
                    output = "".join(output_lines)
                    self.fail(
                        "Timed out waiting for diffusion RL e2e subprocess.\n"
                        f"Command output:\n{output}"
                    )
                finally:
                    _terminate_process_group(process)
                    output_thread.join(timeout=10.0)
                    if process.stdout is not None:
                        process.stdout.close()

                output = "".join(output_lines)

                self.assertEqual(
                    process.returncode,
                    0,
                    msg=f"Diffusion RL subprocess failed.\nCommand output:\n{output}",
                )
                self.assertGreater(
                    len(reward_service._server.request_records),
                    0,
                    msg="Mocked reward service did not receive any enqueue requests.",
                )


if __name__ == "__main__":
    unittest.main()
