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

# Pass `--stream` (or `--verbose`) when running this script to print all training logs in real time.
import os
import sys
import subprocess
import tempfile
import toml
import re
import threading
import time


from cosmos_rl.utils import util
import torch
import signal

try:
    import psutil
except ImportError:
    psutil = None


def _calc_world_size(parallel_cfg: dict, is_rollout: bool = False) -> int:
    tp = int(parallel_cfg.get("tp_size", 1))
    pp = int(parallel_cfg.get("pp_size", 1))
    if is_rollout:
        return tp * pp or 1
    dp_shard = int(parallel_cfg.get("dp_shard_size", 1))
    if dp_shard == -1:
        dp_shard = 1  # fallback when auto infer disabled in smoke test
    dp_rep = int(parallel_cfg.get("dp_replicate_size", 1))
    return max(1, tp * pp * dp_shard * dp_rep)


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
CTRL_ENTRY = "cosmos_rl.dispatcher.run_web_panel"
POLICY_ENTRY = "cosmos_rl.policy.train"
ROLLOUT_ENTRY = "cosmos_rl.rollout.rollout_entrance"


def _launch_controller(cfg_file: str, port: int):
    """Start controller web panel on given port."""
    cmd = f"{PYTHON} -m {CTRL_ENTRY} --config {cfg_file} --port {port}"
    env = dict(os.environ, COSMOS_ROLE="Controller")
    return subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _launch_torchrun(entry: str, world_size: int, cuda_devices: str | None = None):
    """Launch a torchrun worker running module entry with given world size"""
    cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--role=rank",
        "--tee=3",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        "-m",
        entry,
    ]
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _prepare_cfg(src_cfg: str) -> str:
    """Overwrite dataset.name to local tiny set and dump to temp file."""
    with open(src_cfg, "r") as f:
        cfg = toml.load(f)
    tmp = tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w+")
    toml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _kill_tree(proc):
    """Kill *proc* and all its children recursively."""
    if proc is None:
        return
    if psutil is not None:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except psutil.NoSuchProcess:
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


# Stream flag
STREAM_LOGS = False
if "--stream" in sys.argv or "--verbose" in sys.argv:
    STREAM_LOGS = True
    # remove the flag to avoid unittest/pytest complaining
    sys.argv = [arg for arg in sys.argv if arg not in ("--stream", "--verbose")]


def _reader_thread(proc, bucket):
    """Read lines from proc.stdout, store and optionally echo."""
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="ignore")
        bucket.append(text)
        if STREAM_LOGS:
            sys.stdout.write(text)
            sys.stdout.flush()


def run_smoke(cfg_name: str, need_rollout: bool):
    """Launch controller/policy/(rollout) and wait until they finish."""
    cfg_file = _prepare_cfg(os.path.join(CUR_DIR, "configs", cfg_name))

    with open(cfg_file, "r") as f:
        cfg_dict = toml.load(f)

    port = util.find_available_port(13000)

    controller = _launch_controller(cfg_file, port)
    os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"

    policy_ws = _calc_world_size(cfg_dict["policy"]["parallelism"], is_rollout=False)
    total_gpus = torch.cuda.device_count()

    if policy_ws > total_gpus:
        raise RuntimeError(
            f"Policy requires {policy_ws} GPUs but only {total_gpus} are visible"
        )

    policy_dev = ",".join(str(i) for i in range(policy_ws))

    rollout_ws = None
    rollout_dev = None
    if need_rollout:
        rollout_ws = _calc_world_size(
            cfg_dict["rollout"]["parallelism"], is_rollout=True
        )
        if policy_ws + rollout_ws > total_gpus:
            raise RuntimeError(
                f"Total GPUs required (policy {policy_ws} + rollout {rollout_ws}) exceeds available {total_gpus}"
            )
        rollout_dev = ",".join(str(i) for i in range(policy_ws, policy_ws + rollout_ws))

    policy = _launch_torchrun(
        POLICY_ENTRY, world_size=policy_ws, cuda_devices=policy_dev
    )
    rollout = (
        _launch_torchrun(ROLLOUT_ENTRY, world_size=rollout_ws, cuda_devices=rollout_dev)
        if need_rollout
        else None
    )

    procs = [p for p in (controller, policy, rollout) if p]

    # Start readers
    controller_logs: list[str] = []
    threading.Thread(
        target=_reader_thread, args=(controller, controller_logs), daemon=True
    ).start()
    policy_logs: list[str] = []
    threading.Thread(
        target=_reader_thread, args=(policy, policy_logs), daemon=True
    ).start()
    rollout_logs: list[str] = []
    if rollout:
        threading.Thread(
            target=_reader_thread, args=(rollout, rollout_logs), daemon=True
        ).start()

    timeout_sec = 1000
    try:
        # Wait for policy / rollout to finish
        for worker in (policy, rollout):
            if worker is None:
                continue
            try:
                worker.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                _kill_tree(worker)
                raise RuntimeError(f"{worker.args} timed out after {timeout_sec}s")
            if worker.returncode != 0:
                raise RuntimeError(
                    f"{worker.args} exited with code {worker.returncode}"
                )

        # Controller should finish after workers
        if controller.poll() is None:
            controller.terminate()
        controller.wait(timeout=30)

        # Compose logs
        captured_logs = policy_logs + rollout_logs + controller_logs
        all_logs = "\n".join(captured_logs)

        try:
            if need_rollout:
                reward_vals = [
                    float(m)
                    for m in re.findall(
                        r"Reward\s+Max\s*:\s*([0-9eE+\-.]+)",
                        all_logs,
                        flags=re.IGNORECASE,
                    )
                ]
                assert reward_vals, "No Reward Max found in logs"
                assert any(v > 0 for v in reward_vals), "All Reward Max are 0"
                print(
                    f"[INTEGRATION TEST] GRPO reward check passed, Reward Max list: {reward_vals}"
                )
            else:
                loss_vals = [
                    float(m)
                    for m in re.findall(r"Validation loss:\s*([0-9eE+\-.]+)", all_logs)
                ]
                assert loss_vals, "No Validation loss found in logs"
                max_loss = max(loss_vals)
                assert max_loss <= 1.3, f"Validation loss too high: {max_loss} > 1.3"
                print(
                    f"[INTEGRATION TEST] SFT validation check passed, Validation loss: {max_loss}"
                )
        except AssertionError:
            print("\n=== Captured logs (tail) ===")
            print(all_logs[-5000:])
            raise
    finally:
        for p in procs:
            if p.poll() is None:
                _kill_tree(p)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _kill_tree(p)


if __name__ == "__main__":
    cases = [
        ("sft_integration_test.toml", False),  # SFT
        ("grpo_integration_test.toml", True),  # GRPO
    ]

    total_start = time.time()
    for cfg, need_rollout in cases:
        case_start = time.time()
        print(f"[INTEGRATION TEST] Running {cfg}  (rollout={need_rollout})")
        run_smoke(cfg, need_rollout)
        case_elapsed = time.time() - case_start
        print(f"[INTEGRATION TEST] ✅  {cfg} finished in {case_elapsed:.1f} seconds")
    total_elapsed = time.time() - total_start
    print(f"[INTEGRATION TEST] All cases finished in {total_elapsed:.1f} seconds")
