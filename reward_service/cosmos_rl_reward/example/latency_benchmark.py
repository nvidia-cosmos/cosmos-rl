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

"""Benchmark average end-to-end latency of reward server.

This script follows the same request format as example/image_client.py:
- POST /api/reward/ping with form field "info_data" containing JSON
- POST /api/reward/enqueue with body: json.dumps(meta) + "\n" + npy_bytes
- POST /api/reward/pull polling until status_code == 200

It reports per-request end-to-end latency (enqueue -> pull success) and summary stats.

Optionally, it can print the JSON responses returned by /api/reward/pull.

Example:
  python reward_service/cosmos_rl_reward/example/latency_benchmark.py \
    --host https://<your-host> \
    --token <token> \
    --reward-fn hpsv2 \
    --num-requests 20
"""

from __future__ import annotations

import argparse
import io
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def make_headers(
    token: str,
    replica_id: str | None = None,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if replica_id:
        headers["X-Lepton-Replica-Target"] = replica_id
    if extra:
        headers.update(extra)
    return headers


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _format_json_for_log(obj: object, max_chars: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        s = repr(obj)
    if max_chars is not None and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "...<truncated>"
    return s


def _recommend_enqueue_timeout_s(payload_bytes: int, assume_upload_mbps: float) -> float:
    """Heuristic to avoid socket write timeouts for large uploads.

    requests/urllib3 doesn't expose an explicit "write timeout"; the socket timeout can still
    trigger during send() when the server/network is slow. We estimate upload time from payload
    size and assumed throughput (MB/s), then add a safety factor.
    """

    if payload_bytes <= 0:
        return 30.0
    if assume_upload_mbps <= 0:
        return 60.0

    payload_mb = payload_bytes / 1_000_000.0
    est_upload_s = payload_mb / assume_upload_mbps
    # Safety: *2 for jitter/backpressure + 30s fixed overhead.
    return max(30.0, est_upload_s * 2.0 + 30.0)


def _parse_shape(shape: str) -> Tuple[int, ...]:
    # Accept "2,512,512,3" or "[2,512,512,3]".
    cleaned = shape.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid --shape: {shape!r}")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError(f"Invalid --shape (dims must be >0): {shape!r}")
    return dims


def _parse_reward_fn_specs(specs: Optional[List[str]], default_name: str) -> Dict[str, float]:
    # Allow repeated flags: --reward-fn hpsv2 or --reward-fn hpsv2=1.0
    if not specs:
        return {default_name: 1.0}
    reward_fn: Dict[str, float] = {}
    for spec in specs:
        s = spec.strip()
        if not s:
            continue
        if "=" in s:
            name, weight = s.split("=", 1)
            name = name.strip()
            weight = float(weight.strip())
        else:
            name = s
            weight = 1.0
        if not name:
            raise ValueError(f"Invalid --reward-fn value: {spec!r}")
        reward_fn[name] = weight
    if not reward_fn:
        raise ValueError("No valid --reward-fn provided")
    return reward_fn


def build_fake_image_npy_bytes(shape: Tuple[int, ...]) -> bytes:
    # Matches image_client.py: np.save(uint8 array) and send as .npy bytes.
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(
            f"Image shape must be [B,H,W,3]. Got {shape}"
        )
    arr = ((np.random.rand(*shape) * 255.0).clip(0, 255)).astype(np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def build_fake_video_torch_bytes(shape: Tuple[int, ...], dtype: str) -> bytes:
    # Matches video_client.py: torch.save(tensor, buffer) and send as bytes.
    # Typical latent shape: [B, 16, 24, 54, 96]
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Video mode requires PyTorch. Install torch or use --media-type image."
        ) from e

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported --video-dtype: {dtype!r}. Choose from {sorted(dtype_map)}")

    # Use rand() in float32 then cast; avoids bfloat16 rand limitations on some backends.
    tensor = torch.from_numpy(np.random.rand(*shape).astype(np.float32)).to(dtype_map[dtype])
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def ping(session: requests.Session, url: str, token: str, info_data: Dict) -> None:
    response = session.post(
        url,
        data={"info_data": json.dumps(info_data)},
        headers=make_headers(token),
        timeout=(5, 10),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Ping failed with status {response.status_code}: {response.text}"
        )


def enqueue(
    session: requests.Session,
    url: str,
    token: str,
    meta: Dict,
    npy_bytes: bytes,
    timeout: Tuple[float, float],
) -> Tuple[str, str | None]:
    payload = json.dumps(meta).encode("utf-8") + b"\n" + npy_bytes
    response = session.post(
        url,
        data=payload,
        headers={
            **make_headers(token, None, {"Content-Type": "application/octet-stream"}),
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Enqueue failed with status {response.status_code}: {response.text}"
        )
    response_json = response.json()
    uuid = response_json.get("uuid")
    if not uuid:
        raise KeyError(f"Enqueue response missing 'uuid': {response_json}")
    return uuid, response_json.get("replica_id")


def pull_until_ready(
    session: requests.Session,
    url: str,
    token: str,
    replica_id: str | None,
    uuid: str,
    reward_type: str,
    poll_interval_s: float,
    timeout: Tuple[float, float],
    max_wait_s: float,
) -> Dict:
    deadline = time.perf_counter() + max_wait_s
    while True:
        response = session.post(
            url,
            data={"uuid": uuid, "type": reward_type},
            headers=make_headers(token, replica_id),
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()

        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"Pull timed out after {max_wait_s}s. last_status={response.status_code}, last_body={response.text}"
            )
        time.sleep(poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark average end-to-end latency of reward server (enqueue + poll pull until ready)."
    )
    parser.add_argument("--host", type=str, required=True, help="Server host, e.g. https://... (no trailing slash)")
    parser.add_argument("--token", type=str, default="", help="Bearer token (optional)")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of benchmark requests")
    parser.add_argument(
        "--media-type",
        type=str,
        choices=["image", "video"],
        default="image",
        help="Payload media type to benchmark",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="2,512,512,3",
        help="Tensor shape. image expects B,H,W,3. video expects B,... (e.g. 1,16,24,54,96)",
    )
    parser.add_argument(
        "--video-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Video tensor dtype used for torch.save() payload",
    )
    parser.add_argument(
        "--reward-fn",
        action="append",
        default=None,
        help="Reward function spec. Repeatable. Format: name or name=weight (e.g. --reward-fn hpsv2 --reward-fn ocr=1.0)",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between pull retries")
    parser.add_argument(
        "--max-wait",
        type=float,
        default=120.0,
        help="Max seconds to wait for pull to become ready per request",
    )
    parser.add_argument(
        "--enqueue-timeout",
        type=float,
        default=60.0,
        help="Enqueue request timeout in seconds (applies to both connect/read as a single value)",
    )
    parser.add_argument(
        "--assume-upload-mbps",
        type=float,
        default=20.0,
        help="Heuristic upload throughput (MB/s) used to auto-recommend enqueue timeout based on payload size.",
    )
    parser.add_argument(
        "--pull-timeout",
        type=float,
        default=10.0,
        help="Pull request timeout in seconds (applies to both connect/read as a single value)",
    )
    parser.add_argument(
        "--no-ping",
        action="store_true",
        help="Skip ping check",
    )
    parser.add_argument(
        "--print-response",
        type=str,
        choices=["none", "last", "all"],
        default="none",
        help="Print /api/reward/pull response JSON: none, last (per request), or all (every pull success).",
    )
    parser.add_argument(
        "--print-max-chars",
        type=int,
        default=4000,
        help="Max characters to print for each response JSON (truncated if longer).",
    )

    args = parser.parse_args()

    api_ping = "/api/reward/ping"
    api_enqueue = "/api/reward/enqueue"
    api_pull = "/api/reward/pull"

    ping_url = args.host + api_ping
    enqueue_url = args.host + api_enqueue
    pull_url = args.host + api_pull

    shape = _parse_shape(args.shape)
    default_reward = "hpsv2" if args.media_type == "image" else "cosmos_reason1"
    reward_fn = _parse_reward_fn_specs(args.reward_fn, default_name=default_reward)

    if args.media_type == "image":
        batch = shape[0]
        payload_bytes = build_fake_image_npy_bytes(shape)
        meta: Dict = {
            "media_type": "image",
            "prompts": ["Latency benchmark prompt." for _ in range(batch)],
            "reward_fn": reward_fn,
        }
    else:
        batch = shape[0]
        payload_bytes = build_fake_video_torch_bytes(shape, dtype=args.video_dtype)
        # Matches video_client.py: include video_infos; also include media_type for clarity.
        meta = {
            "media_type": "video",
            "prompts": ["Latency benchmark prompt." for _ in range(batch)],
            "reward_fn": reward_fn,
            "video_infos": [{"video_fps": 16.0} for _ in range(batch)],
        }

    log(
        f"[LatencyBenchmark] Prepared payload: media_type={args.media_type}, shape={shape}, bytes={len(payload_bytes)}"
    )

    recommended_enqueue_timeout = _recommend_enqueue_timeout_s(
        payload_bytes=len(payload_bytes),
        assume_upload_mbps=args.assume_upload_mbps,
    )
    effective_enqueue_timeout = max(float(args.enqueue_timeout), recommended_enqueue_timeout)
    if effective_enqueue_timeout != float(args.enqueue_timeout):
        log(
            "[LatencyBenchmark] Auto-increasing enqueue timeout to reduce write timeouts: "
            f"configured={float(args.enqueue_timeout):.1f}s recommended={recommended_enqueue_timeout:.1f}s effective={effective_enqueue_timeout:.1f}s "
            f"(assume_upload_mbps={args.assume_upload_mbps})"
        )

    session = requests.Session()

    if not args.no_ping:
        log(f"[LatencyBenchmark] Pinging server: {ping_url}")
        ping(session, ping_url, args.token, meta)

    latencies_s: List[float] = []

    for i in range(args.num_requests):
        t0 = time.perf_counter()
        uuid, replica_id = enqueue(
            session,
            enqueue_url,
            args.token,
            meta,
            payload_bytes,
            timeout=(10, effective_enqueue_timeout),
        )
        if replica_id:
            log(f"[LatencyBenchmark] Enqueue assigned replica_id={replica_id}")
        # Poll for all reward types; end-to-end latency is until the last one is ready.
        last_responses: Dict[str, object] = {}
        for reward_type in reward_fn.keys():
            resp_json = pull_until_ready(
                session,
                pull_url,
                args.token,
                replica_id,
                uuid,
                reward_type=reward_type,
                poll_interval_s=args.poll_interval,
                timeout=(5, args.pull_timeout),
                max_wait_s=args.max_wait,
            )
            last_responses[reward_type] = resp_json
            if args.print_response == "all":
                log(
                    "[LatencyBenchmark] pull response "
                    f"uuid={uuid} type={reward_type}: "
                    f"{_format_json_for_log(resp_json, args.print_max_chars)}"
                )
        t1 = time.perf_counter()
        dt = t1 - t0
        latencies_s.append(dt)
        log(f"[LatencyBenchmark] {i+1}/{args.num_requests}: uuid={uuid}, latency={dt:.3f}s")
        if args.print_response == "last":
            log(
                "[LatencyBenchmark] last pull responses "
                f"uuid={uuid}: {_format_json_for_log(last_responses, args.print_max_chars)}"
            )

    latencies_sorted = sorted(latencies_s)
    mean = sum(latencies_s) / len(latencies_s) if latencies_s else float("nan")
    log("[LatencyBenchmark] Done.")
    log(
        f"[LatencyBenchmark] n={len(latencies_s)} mean={mean:.3f}s min={latencies_sorted[0]:.3f}s max={latencies_sorted[-1]:.3f}s"
    )
    log(
        "[LatencyBenchmark] p50={:.3f}s p90={:.3f}s p95={:.3f}s".format(
            _percentile(latencies_sorted, 50),
            _percentile(latencies_sorted, 90),
            _percentile(latencies_sorted, 95),
        )
    )


if __name__ == "__main__":
    main()
