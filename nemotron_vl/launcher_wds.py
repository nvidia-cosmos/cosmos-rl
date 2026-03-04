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

import json
from typing import Optional
import os, sys
os.environ["USE_QWEN_VL_PROCESS"] = "1"
if os.environ.get("APPLY_SIGLIP2_PATCH"):
    os.environ["USE_SIGLIP2_PROCESS"] = "1"
# Enable EP mesh to be represented by TP mesh, and also treat EP as a sub-group of Data Parallelism.
os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"
import webdataset as wds
from torch.utils.data import IterableDataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable DecompressionBombWarning for large images
import io
from typing import Any, Iterator, Union, List, Optional, Sequence
import glob
import torch, re
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from qwen_vl_utils.vision_process import smart_nframes, calculate_video_frame_range
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nemotron_parallelize import parallelize
from weight_converter import convert_weight_from_hf
from torch.utils.data import Dataset
from cosmos_rl.policy.config import Config as CosmosConfig

from webdataset.tariterators import (
    base_plus_ext as _wds_base_plus_ext,
    valid_sample as _wds_valid_sample,
    url_opener as _wds_url_opener,
    tar_file_expander as _wds_tar_file_expander,
)

try:
    import decord
except ImportError:
    decord = None

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    VideoDecoder = None

try:
    import wandb
except ImportError:
    wandb = None


def _check_torchcodec() -> tuple[bool, str]:
    """Return (available, reason) for torchcodec."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, f"ImportError: {e}"
    except RuntimeError as e:
        return False, f"RuntimeError: {e}"


@lru_cache(maxsize=1)
def _get_video_backend() -> str:
    """Select video decoding backend: torchcodec > decord.

    Honours ``FORCE_QWENVL_VIDEO_READER`` env-var to pin a specific backend.
    """
    forced = os.environ.get("FORCE_QWENVL_VIDEO_READER")
    if forced:
        if forced == "torchcodec":
            assert VideoDecoder is not None, "torchcodec is not available"
        elif forced == "decord":
            assert decord is not None, "decord is not available"
        else:
            raise ValueError(f"Invalid FORCE_QWENVL_VIDEO_READER value: {forced}, must be 'torchcodec' or 'decord'")
        return forced
    if VideoDecoder is not None:
        return "torchcodec"
    elif decord is not None:
        return "decord"
    raise ImportError(
        "No video decoding backend found. Install torchcodec (preferred) or decord."
    )
########################################################
# Auxiliary helper functions for MoE load balancing tracking.
########################################################

def _to_cpu_np(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def _balance_metrics(counts_1d: np.ndarray):
    counts = counts_1d.astype(np.float32)
    total = float(counts.sum() + 1e-9)
    p = counts / total
    entropy = float(-(p * np.log(p + 1e-9)).sum())
    max_fraction = float(p.max())
    std = float(counts.std())
    return entropy, max_fraction, std, total

def report_moe_load_to_wandb(step: int, loads_by_layer: dict, prefix="moe", log_every=10):
    if len(loads_by_layer) == 0:
        return None
    if (step % log_every) != 0:
        return None

    layer_names = sorted(loads_by_layer.keys())
    layer_loads = [_to_cpu_np(loads_by_layer[name]).reshape(-1) for name in layer_names]

    if wandb is not None:
        table = wandb.Table(columns=["step", "layer", "layer_name", "expert", "tokens", "frac"])
    else:
        table = None
    entropies, max_fracs, stds, totals = [], [], [], []
    for li, (lname, counts) in enumerate(zip(layer_names, layer_loads)):
        entropy, max_fraction, std, total = _balance_metrics(counts)
        entropies.append(entropy); max_fracs.append(max_fraction); stds.append(std); totals.append(total)

        if table is not None:
            frac = counts / (counts.sum() + 1e-9)
            for ei in range(counts.shape[0]):
                table.add_data(int(step), int(li), lname, int(ei), float(counts[ei]), float(frac[ei]))
 

    return {
        f"{prefix}/layer_expert_table": table,
        f"{prefix}/entropy_mean": float(np.mean(entropies)),
        f"{prefix}/entropy_min": float(np.min(entropies)),
        f"{prefix}/max_fraction_mean": float(np.mean(max_fracs)),
        f"{prefix}/max_fraction_max": float(np.max(max_fracs)),
        f"{prefix}/tokens_per_layer_mean": float(np.mean(totals)),
        f"{prefix}/tokens_per_layer_std_mean": float(np.mean(stds)),
    }

########################################################

def _sanitize_wds_filename(fname: str) -> str:
    """Replace dots in the __key__ portion of a tar entry filename.

    WebDataset's ``base_plus_ext`` splits at the *first* dot to obtain
    (key, extension).  If the original sample id contains dots (e.g.
    ``sharegpt4v(llava)_processed_new.json16311``), every sample from
    that source ends up with the same key prefix and they all get merged
    into one giant sample.  We fix this by replacing every dot in the
    key portion (everything before the last dot) with an underscore.
    """
    last_dot = fname.rfind(".")
    if last_dot <= 0:
        return fname
    key_part = fname[:last_dot]
    ext_part = fname[last_dot:]           # includes the leading dot
    return key_part.replace(".", "_") + ext_part


# ---------------------------------------------------------------------------
# Tolerant tarfile_to_samples: handles both dots-in-key and duplicate IDs
# without losing samples.
# ---------------------------------------------------------------------------

def _group_by_keys_tolerant(data, keys=_wds_base_plus_ext, lcase=True,
                            suffixes=None, handler=None):
    """Like ``webdataset.tariterators.group_by_keys`` but duplicate-tolerant.

    When a duplicate suffix is encountered for the current sample (e.g. two
    files with the same key and extension), the current sample is yielded and
    a *new* sample is started instead of raising ``ValueError``.  This keeps
    both copies of a duplicated sample instead of losing one.
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            if filesample == {}:
                if _wds_valid_sample(current_sample):
                    yield current_sample
                current_sample = None
                continue
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if current_sample is None or prefix != current_sample["__key__"]:
                if _wds_valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffix in current_sample:
                # Duplicate suffix -- yield the current (complete) sample and
                # start a fresh one so both copies survive.
                if _wds_valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
            local_path = filesample.get("__local_path__")
            if local_path is not None:
                current_sample["__local_path__"] = local_path
        except Exception as exn:
            exn.args = exn.args + (filesample.get("stream"), filesample.get("url"))
            if handler and handler(exn):
                continue
            else:
                break
    if _wds_valid_sample(current_sample):
        yield current_sample


def _tarfile_samples_tolerant(src, handler=wds.reraise_exception,
                              select_files=None, rename_files=None):
    """Drop-in replacement for ``wds.tarfile_samples`` that uses
    ``_group_by_keys_tolerant`` to survive duplicate sample IDs."""
    streams = _wds_url_opener(src, handler=handler)
    files = _wds_tar_file_expander(streams, handler=handler,
                                   select_files=select_files,
                                   rename_files=rename_files)
    samples = _group_by_keys_tolerant(files, handler=handler)
    return samples


_tarfile_to_samples_tolerant = wds.filters.pipelinefilter(
    _tarfile_samples_tolerant
)


# Per-image pixel cap: at most ~1960 vision tokens per image.
# 1960 * (patch_size * merge_size)^2 = 1960 * 32 * 32
IMAGE_MAX_PIXELS = 1960 * 32 * 32


def approx_max_pixels(messages, max_seq_len):
    """Estimate the available pixel budget for vision content.

    Approximates text token count from the JSON byte length, then converts
    the remaining token budget to pixels using (patch_size * merge_size)^2 = 1024.
    Returns None when the text alone nearly fills the context (< 128 tokens
    worth of vision capacity), signalling the caller to skip the sample.
    """
    # Estimate text length by summing only the serialisable text portions.
    # messages may already contain PIL Images / video frames after
    # _attach_media_from_sample, so json.dumps(messages) would fail.
    text_bytes = 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            text_bytes += len(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_bytes += len(c.get("text", ""))
    text_token_length = int(text_bytes / 3.0)
    max_pixels = (max_seq_len - text_token_length) * 0.9 * 32 * 32
    if max_pixels < 128 * 32 * 32:
        return None
    return max_pixels


def modify_messages(messages, max_seq_len=None):
    """Set per-content pixel budgets and normalise content format.

    * Computes a *per-sample* pixel budget from the remaining token capacity
      so that vision tokens + text tokens ≤ model_max_length.
    * Divides the budget across all vision items (images + videos) in the
      sample, and additionally caps each image at IMAGE_MAX_PIXELS.
    * Returns **None** when the sample cannot fit any vision content,
      signalling the caller to drop it.
    """
    max_pixels = None
    if max_seq_len is not None:
        max_pixels = approx_max_pixels(messages, max_seq_len)
        if max_pixels is None:
            return None  # text alone already fills the context

    # Count total vision items so we can split the budget fairly.
    n_vision_items = 0
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get('type') in ('image', 'video'):
                    n_vision_items += 1

    # Per-item pixel share of the total budget.
    per_item_pixels = None
    if max_pixels is not None and n_vision_items > 0:
        per_item_pixels = int(max_pixels / n_vision_items)

    for message in messages:
        if isinstance(message['content'], str):
            # Only normalize user messages to list format; system/assistant
            # messages must remain strings for the Jinja chat template.
            if message.get('role') in ('system', 'assistant'):
                continue
            message['content'] = [{'type': 'text', 'text': message['content']}]
        if not isinstance(message['content'], list):
            continue
        for content in message['content']:
            if content.get('type') == 'image':
                pixel_cap = IMAGE_MAX_PIXELS
                if per_item_pixels is not None:
                    pixel_cap = min(pixel_cap, per_item_pixels)
                content['max_pixels'] = pixel_cap
            elif content.get('type') == 'video':
                if per_item_pixels is not None:
                    content['total_pixels'] = per_item_pixels
    return messages

def modify_messages_siglip2(messages, max_num_patches=256, max_frame_num_patches=196,
                            scale_factor=1024, max_seq_len=None, video_max_frames=30):
    """Hybrid SigLIP2 pixel budget: fixed patch counts + dynamic overflow cap.

    Computes fixed per-item budgets from patch counts, then optionally caps
    with a dynamic budget derived from remaining context length.

    - Images (<4 per message): max_num_patches * scale_factor pixels
    - Images (>=4 per message): max_frame_num_patches * scale_factor pixels
    - Videos: fps=1, max_frames=video_max_frames,
              total_pixels = max_frame_num_patches * video_max_frames * scale_factor

    If max_seq_len is set, computes a dynamic cap via approx_max_pixels() and
    takes min(fixed_budget, dynamic_share) per item to prevent context overflow.

    Returns None if text alone fills the context (signals caller to drop sample).
    """
    # Step 1: Dynamic overflow cap (optional)
    dynamic_per_item = None
    if max_seq_len is not None:
        dynamic_max = approx_max_pixels(messages, max_seq_len)
        if dynamic_max is None:
            return None  # text alone fills the context

        # Count vision items for fair sharing of the dynamic budget
        n_vision = 0
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get('type') in ('image', 'video'):
                        n_vision += 1
        if n_vision > 0:
            dynamic_per_item = int(dynamic_max / n_vision)

    # Step 2: Apply fixed budgets per message, capped by dynamic budget
    for message in messages:
        if isinstance(message['content'], str):
            if message.get('role') in ('system', 'assistant'):
                continue
            message['content'] = [{'type': 'text', 'text': message['content']}]
        if not isinstance(message['content'], list):
            continue

        # Count images in this message for the <4 / >=4 threshold
        num_images = 0
        for content in message['content']:
            if isinstance(content, dict) and content.get('type') == 'image':
                num_images += 1

        for content in message['content']:
            if content.get('type') == 'image':
                if num_images < 4:
                    fixed_px = max_num_patches * scale_factor
                else:
                    fixed_px = max_frame_num_patches * scale_factor
                if dynamic_per_item is not None:
                    content['max_pixels'] = min(fixed_px, dynamic_per_item)
                else:
                    content['max_pixels'] = fixed_px
            elif content.get('type') == 'video':
                # fps/max_frames are already set during decoding in
                # _attach_media_from_sample — only the pixel budget matters here.
                fixed_px = max_frame_num_patches * video_max_frames * scale_factor
                if dynamic_per_item is not None:
                    content['total_pixels'] = min(fixed_px, dynamic_per_item)
                else:
                    content['total_pixels'] = fixed_px
    return messages


def _expand_wds_urls(name: Union[str, Sequence[str]]) -> List[str]:
    """
    Returns a LIST of shard paths/urls suitable for wds.WebDataset(urls).

    Accepts:
      - brace pattern str: "/path/train-{000000..000999}.tar"  -> returns ["that pattern"] (WebDataset expands)
      - dir str: "/data/wds_dir" -> returns sorted list of .tar in that dir
      - tar str: "/data/a.tar" -> returns ["/data/a.tar"]
      - list/tuple of dirs or tars: ["/d1", "/d2", "/x/a.tar"] -> concatenated shard list
      - comma-separated str: "/d1,/d2" -> treated as list
      - file list: "@/path/shards.txt" -> each line is a shard path (comments allowed)
    """
    def expand_one(x: str) -> List[str]:
        x = x.strip()
        if not x:
            return []
        # filelist support: "@/path/shards.txt"
        if x.startswith("@"):
            fl = x[1:]
            out = []
            with open(fl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    out.append(line)
            return out

        # brace pattern: let WebDataset handle expansion
        if any(tok in x for tok in ["{", "}", ".."]):
            return [x]

        if os.path.isdir(x):
            shards = sorted(glob.glob(os.path.join(x, "*.tar")))
            if not shards:
                raise ValueError(f"No .tar shards found in directory: {x}")
            return shards

        if os.path.isfile(x) and x.endswith(".tar"):
            return [x]

        raise ValueError(f"Unsupported WebDataset path: {x}")

    # If it's a list/tuple
    if isinstance(name, (list, tuple)):
        urls: List[str] = []
        for item in name:
            urls.extend(expand_one(str(item)))
        return urls

    # If it's a string
    if isinstance(name, str):
        # allow comma-separated list
        if "," in name and not any(tok in name for tok in ["{", "}", ".."]):
            parts = [p.strip() for p in name.split(",") if p.strip()]
            urls: List[str] = []
            for p in parts:
                urls.extend(expand_one(p))
            return urls
        return expand_one(name)

    raise TypeError(f"Unsupported type for dataset.name: {type(name)}")

def _iter_media_items(messages: list[dict]):
    """
    Yield content dicts that reference webdataset media via 'wds_field'.
    """
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            # Only normalize user messages to list format; system/assistant
            # messages must stay as strings so the chat template can
            # concatenate them (Jinja does ``str + system_message``).
            if msg.get("role") == "user":
                msg["content"] = [{"type": "text", "text": content}]
                content = msg["content"]
            else:
                continue
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and "wds_field" in c:
                yield c


def _decode_video_torchcodec(blob: bytes, ele: dict,
                              decoding_timeout: int = 60) -> tuple[list[Image.Image], float, float]:
    """Decode video bytes using torchcodec (preferred backend).

    Returns (pil_frames, sample_fps, video_fps).
    """

    decoder = VideoDecoder(blob)
    metadata = decoder.metadata
    total_frames = metadata.num_frames
    video_fps = metadata.average_fps

    if total_frames is None or video_fps is None:
        raise ValueError("torchcodec metadata missing num_frames or average_fps")
    if total_frames == 0:
        raise ValueError("Video has 0 frames")
    if video_fps < 1:
        raise ValueError(f"Video FPS {video_fps} < 1")

    orig_total_frames = total_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    # Clamp to valid absolute frame indices within [start_frame, end_frame].
    idx = [min(max(i, start_frame), end_frame) for i in idx]

    # Timeout guard — any decoder could hang on corrupt video
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: decoder.get_frames_at(idx).data)
    try:
        # get_frames_at returns FrameBatch with .data as NCHW tensor
        frames_tensor = future.result(timeout=decoding_timeout)
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"Video decoding (torchcodec) timed out after {decoding_timeout}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Convert NCHW -> NHWC numpy for PIL
    frames_np = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()
    pil_frames = [Image.fromarray(frame) for frame in frames_np]
    sample_fps = nframes / max(orig_total_frames, 1e-6) * video_fps

    del decoder
    return pil_frames, sample_fps, video_fps


def _decode_video_decord(blob: bytes, ele: dict,
                          decoding_timeout: int = 60) -> tuple[list[Image.Image], float, float]:
    """Decode video bytes using decord (fallback backend).

    Returns (pil_frames, sample_fps, video_fps).
    """

    video_buffer = io.BytesIO(blob)
    vr = decord.VideoReader(video_buffer, num_threads=1)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    if total_frames == 0:
        raise ValueError("Video has 0 frames")
    if video_fps < 1:
        raise ValueError(f"Video FPS {video_fps} < 1")

    orig_total_frames = total_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    # Clamp to valid absolute frame indices within [start_frame, end_frame].
    idx = [min(max(i, start_frame), end_frame) for i in idx]

    # Timeout guard against hanging decord
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: vr.get_batch(idx).asnumpy())
    try:
        frames = future.result(timeout=decoding_timeout)
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"Video decoding (decord) timed out after {decoding_timeout}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    pil_frames = [Image.fromarray(frame) for frame in frames]
    sample_fps = nframes / max(orig_total_frames, 1e-6) * video_fps

    del vr
    return pil_frames, sample_fps, video_fps


def _decode_video_from_bytes(blob: bytes, ele: dict,
                              decoding_timeout: int = 60) -> tuple[list[Image.Image], float, float]:
    """Decode video bytes in-memory and return sampled frames as PIL Images.

    Dispatches to the best available backend (torchcodec > decord).
    Matches qwen_vl_utils._read_video_decord behaviour: uses the full content
    dict *ele* so that smart_nframes respects per-sample fps / min_frames /
    max_frames, and calculate_video_frame_range honours video_start / video_end.

    Returns (pil_frames, sample_fps, video_fps).
    """
    backend = _get_video_backend()
    if backend == "torchcodec":
        return _decode_video_torchcodec(blob, ele, decoding_timeout)
    else:
        return _decode_video_decord(blob, ele, decoding_timeout)


def _attach_media_from_sample(messages: list[dict], sample: dict,
                               video_sample_fps: float = 1.0,
                               default_max_frames: int = 768) -> list[dict] | None:
    """
    Replace each content item containing 'wds_field' with actual media payload.

    For images: set c["image"] = PIL.Image
    For videos: decode in-memory to list[PIL.Image] and set c["video"],
                c["sample_fps"], and c["raw_fps"].

    *video_sample_fps* and *default_max_frames* are applied as fallbacks when the
    JSON content item does not specify per-sample ``fps`` / ``max_frames``.

    Returns None if any video decode fails (caller should skip the sample).
    """
    for c in _iter_media_items(messages):
        field = c.pop("wds_field")
        if field not in sample:
            # Missing blob — neutralize to avoid downstream crash in
            # process_vision_info which expects a "video"/"image" key.
            c["type"] = "text"
            c["text"] = ""
            continue

        blob = sample[field]
        # blob may be memoryview/bytearray; normalize
        if isinstance(blob, (memoryview, bytearray)):
            blob = bytes(blob)

        # infer from c["type"] if present
        kind = c.get("type", None)

        if kind == "image":
            try:
                img = Image.open(io.BytesIO(blob)).convert("RGB")
            except Exception as e:
                print(f"WARNING: Failed to decode image (field={field}): {e}")
                return None
            c["image"] = img  # IMPORTANT: this matches many VLM preprocessors
        elif kind == "video":
            # Ensure defaults so smart_nframes / calculate_video_frame_range
            # see the same keys that qwen_vl_utils._read_video_decord would.
            c.setdefault("fps", video_sample_fps)
            c.setdefault("max_frames", default_max_frames)

            try:
                pil_frames, sample_fps, video_fps = _decode_video_from_bytes(
                    blob, ele=c)
            except Exception as e:
                print(f"WARNING: Failed to decode video (field={field}): {e}")
                return None  # signals preprocess_sample to skip

            c["video"] = pil_frames
            c["sample_fps"] = sample_fps
            c["raw_fps"] = video_fps

            # Clean up WDS metadata fields that downstream processors don't expect
            for extra_key in ("ext", "orig_path"):
                c.pop(extra_key, None)
        else:
            # if kind missing, fall back by extension
            # (you can extend this if needed)
            c["blob"] = blob

    return messages


class CustomWebDatasetDataset(Dataset):
    """
    Streaming dataset for Nemotron-3-Nano VLM alignment using WebDataset shards.

    NOTE: This inherits from Dataset (map-style) rather than IterableDataset because
    the cosmos_rl framework requires a map-style Dataset interface. However, internally
    it wraps a streaming WebDataset pipeline. As a result:
      - __getitem__(idx) ignores the idx parameter and returns the next streamed sample.
      - Shuffling is handled internally by the WebDataset shuffle buffer, NOT by the
        DataLoader sampler.
      - Shard splitting across nodes/workers is handled by WebDataset's split_by_node
        and split_by_worker inside the pipeline.

    Expects each WDS sample to include:
      - "json": bytes of a JSON object containing 'messages'
      - media fields referenced by messages via content item {"wds_field": "...", "type": "image"/"video", ...}
    """
    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.cosmos_config = config
        # Controls (put these in config.custom)
        custom = config.custom if hasattr(config, "custom") and config.custom is not None else {}
        self.include_video = bool(custom.get("include_video", False))
        self.video_sample_fps = float(custom.get("video_sample_fps", 1.0))
        self.shuffle_buf = int(custom.get("wds_shuffle", 2000))  # 0 disables

        self.n_shards_to_skip = 0
        self.n_samples_to_skip_in_shard = 0
        resume_path = config.train.resume
        if resume_path and isinstance(resume_path, str) and os.path.exists(resume_path):
            # Check the current step from the resume checkpoint to determine 
            # how much shards to skip and how much samples to skip in the current shard.
            rank_0_state_path = os.path.join(resume_path, "extra_info_rank_0.pth")
            if os.path.exists(rank_0_state_path):
                extra_info = torch.load(rank_0_state_path, map_location="cpu", weights_only=False)
                resume_step = extra_info.get("step", 0)

                # TODO(jiaxinc): 50000 is a rough number for all tar files.
                # Get worker number, the more workers, the more steps per shard
                num_workers = self.cosmos_config.train.train_policy.dataloader_num_workers or 1
                n_steps_per_shard = 50000 * num_workers // self.cosmos_config.train.train_batch_per_replica
                self.n_shards_to_skip = resume_step // n_steps_per_shard
                self.n_samples_to_skip_in_shard = (resume_step % n_steps_per_shard) * self.cosmos_config.train.train_batch_per_replica // num_workers
                self.n_samples_to_skip_in_shard += int(self.shuffle_buf * 0.5)

        # Fake the resume logic by skipping a certain number of shards and samples in the current shard.
        # self.n_shards_to_skip = 5
        # self.n_samples_to_skip_in_shard = 100
        if self.n_shards_to_skip > 0 or self.n_samples_to_skip_in_shard > 0:
            print(f"Resuming from previous checkpoint, skipping {self.n_shards_to_skip} shards and {self.n_samples_to_skip_in_shard} samples in the current shard.")

        webdataset_root_paths = custom["webdataset_root_paths"]
        if isinstance(webdataset_root_paths, str):
            webdataset_root_paths = [webdataset_root_paths]

        total_items = 0
        # Check `meta.json` inside each path
        for root_path in webdataset_root_paths:
            meta_json_path = os.path.join(root_path, "meta.json")
            if not os.path.exists(meta_json_path):
                raise FileNotFoundError(f"meta.json not found in {root_path}")
            with open(meta_json_path, "r") as f:
                meta_json = json.load(f)
            total_items += meta_json["total_items"]

        # Optional length hint (only used if someone calls len())
        self._length_hint = total_items
        self.urls = _expand_wds_urls(webdataset_root_paths)

        # Per-sample pixel budget is computed dynamically in modify_messages()
        # using approx_max_pixels(), so we only store max_seq_len here.
        self.max_seq_len = config.policy.model_max_length

        # Max video frames: use config value (consistent with modify_messages_siglip2),
        # falling back to 30 to match launcher.py's hardcoded default.
        self.max_video_frames = int(custom.get('video_max_frames', 30))

        self.include_video = bool(custom.get("include_video", False))
        self.video_sample_fps = float(custom.get("video_sample_fps", 1.0))
        self.shuffle_buf = int(custom.get("wds_shuffle", 2000))  # 0 disables

        # SigLIP2 mode: fixed patch-count pixel budget with dynamic overflow cap
        self.siglip2_mode = bool(os.environ.get("APPLY_SIGLIP2_PATCH"))
        if self.siglip2_mode:
            self.scale_factor = (16 * 2) ** 2  # 1024
            self.max_num_patches = int(custom.get('single_image_max_num_patches', 256))
            self.max_frame_num_patches = int(custom.get('single_frame_max_num_patches', 196))
        self.setup_wds_dataset()

    def setup_wds_dataset(self):
        # Build WDS pipeline

        # version-safe
        nodesplitter = getattr(wds, "split_by_node", None) or wds.shardlists.split_by_node
        workersplitter = getattr(wds, "split_by_worker", None) or wds.shardlists.split_by_worker
        ResampledShards = getattr(wds, "ResampledShards", None) or wds.shardlists.ResampledShards
        wds_pool_shuffle_seed = self.cosmos_config.custom.get("wds_pool_shuffle_seed", 42)
        wds_sharding_seed = self.cosmos_config.custom.get("wds_sharding_seed", 12345)

        def skip_first_n_shards(n: int):
            def _skip(src):
                it = iter(src)
                for _ in range(n):
                    next(it, None)   # discard shard url/path
                yield from it
            return _skip

        def skip_first_samples(n: int):
            def _skip(src):
                it = iter(src)
                for _ in range(n):
                    next(it, None)   # discard decoded sample dict
                yield from it
            return _skip

        sharding_pipe = [
            # Sample shards with replacement
            ResampledShards(self.urls, deterministic=True, seed=wds_sharding_seed),
            # NB: if `resampled=True`, and even if we split by node and worker, we still expect to see potential duplicates between nodes and workers.
            # This is expected and hopefully could be uniformly distributed.
            #
            # Split stream by nodes, since we do not expect to see duplicates between nodes
            nodesplitter,          # shard-level split across ranks
            # Split stream by workers, since iterable dataset is not randomly accessible, each worker should see different samples
            workersplitter,        # shard-level split across dataloader workers
        ]

        if self.n_shards_to_skip > 0:
            sharding_pipe.append(skip_first_n_shards(self.n_shards_to_skip))
        
        sharding_pipe.append(_tarfile_to_samples_tolerant(handler=wds.warn_and_continue,rename_files=_sanitize_wds_filename))

        if self.n_samples_to_skip_in_shard > 0:
            sharding_pipe.append(skip_first_samples(self.n_samples_to_skip_in_shard))

        pipeline = wds.DataPipeline(
            *sharding_pipe,
            # Shuffle samples inside a buffer/pool of size `shuffle_buf`
            wds.detshuffle(self.shuffle_buf, seed=wds_pool_shuffle_seed),   # sample-level shuffle buffer
            # Do preprocessing to attach the media payloads (PIL for images; bytes/path for videos)
            wds.map(self.preprocess_sample),
            # Filter out samples that are None(i.e. failed to preprocess)
            wds.select(lambda x: x is not None),
        )

        self.ds = pipeline

    def __len__(self):
        if self._length_hint is None:
            raise TypeError("CustomWebDatasetDataset has no known length. Ensure each dataset root path contains a valid meta.json with 'total_items'.")
        return int(self._length_hint)

    def __getitem__(self, idx: int) -> list[dict]:
        if not hasattr(self, "_wds_iter") or self._wds_iter is None:
            self._wds_iter = iter(self.ds)

        max_retries = 50
        retries = 0
        while True:
            try:
                return next(self._wds_iter)
            except StopIteration:
                retries += 1
                # Normally with resampled=True this should not happen,
                # but it CAN happen if upstream filters drop everything temporarily,
                # or if the pipeline got exhausted for some reason.
                # Recreate the iterator and keep going.
                if retries >= max_retries:
                    raise RuntimeError(
                        f"Pipeline exhausted {max_retries} consecutive times. "
                        f"Check that dataset shards are non-empty and filters are not dropping all samples."
                    )
                self._wds_iter = iter(self.ds)
            except Exception as e:
                retries += 1
                print(f"WARNING: Failed to fetch sample (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise RuntimeError(
                        f"Failed to fetch a valid sample after {max_retries} consecutive attempts. "
                        f"Last error: {e}"
                    ) from e
                continue

    def preprocess_sample(self, sample: dict) -> dict:
        # shard = sample.get("__url__", "<no __url__>")
        # key = sample.get("__key__", "<no __key__>")

        # print(f"[WDS] shard={shard} key={key} on worker={torch.utils.data.get_worker_info() if torch.utils.data.get_worker_info() else 'main'}")

        # Parse JSON
        js = sample.get("json", None)
        if js is None:
            return None
        if isinstance(js, (memoryview, bytearray)):
            js = bytes(js)
        obj = json.loads(js.decode("utf-8"))

        messages = obj.get("messages", [])
        if not isinstance(messages, list):
            return None

        # Normalize system/assistant content from list→string.
        # Some datasets store content as [{"type":"text","text":"..."}] even for
        # non-user messages. The Jinja chat template does string concatenation on
        # system_message (line 44) and requires it to be a string.
        for msg in messages:
            if msg.get("role") in ("system", "assistant"):
                content = msg.get("content")
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                    msg["content"] = "".join(texts)

        # Optional filter: skip samples containing videos if include_video=False
        if not self.include_video:
            has_video = False
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "video":
                            has_video = True
                            break
                if has_video:
                    break
            if has_video:
                return None

        # Attach actual media payloads (PIL for images; decoded frames for videos)
        messages = _attach_media_from_sample(
            messages, sample,
            video_sample_fps=self.video_sample_fps,
            default_max_frames=self.max_video_frames,
        )
        if messages is None:
            return None

        # Apply per-sample pixel budget (max_pixels for images, total_pixels for videos).
        # Video frame selection (fps, max_frames) was already applied during decoding in
        # _attach_media_from_sample — the functions below only set pixel budgets.
        # Both modify_messages variants return None when the text alone fills the context → skip.
        if self.siglip2_mode:
            messages = modify_messages_siglip2(
                messages,
                max_num_patches=self.max_num_patches,
                max_frame_num_patches=self.max_frame_num_patches,
                scale_factor=self.scale_factor,
                max_seq_len=self.max_seq_len,
                video_max_frames=self.max_video_frames,
            )
        else:
            messages = modify_messages(messages, self.max_seq_len)
        if messages is None:
            return None
        assert isinstance(messages, list), "messages should be a list of dicts"
        return messages

@torch.no_grad()
def policy_map_local_key_for_export_tensor(self, name, expert_weight: torch.Tensor):

    # Only For Nemotron-3-Nano Base LLM Model naming convention
    # Leave the prefix "model." for Nemotron-3-Nano Vision-Language Model naming convention
    if name.startswith("model.backbone."):
        name = name[len("model."):]

    if match := re.search(
        r"backbone\.layers\.(\d+)\.mixer\.experts\.(gate_and_up_projs|down_projs)",
        name,
    ):
        def yield_weight(n_experts, expert_weight, w_name, layer_id):
            for expert_id in range(n_experts):
                single_expert_weight = expert_weight[expert_id].contiguous()
                yield (
                    f"backbone.layers.{layer_id}.mixer.experts.{expert_id}.{w_name}.weight",
                    single_expert_weight,
                )
        layer_id = int(match.group(1))
        w_name = match.group(2)
        n_experts = expert_weight.shape[0]
        if w_name == "gate_and_up_projs":
            # gate_and_up_projs is `up_proj` in nemotron
            # shape: [experts, ffn_dim, hidden_dim]
            yield from yield_weight(
                n_experts, expert_weight, "up_proj", layer_id
            )
        else:
            yield from yield_weight(n_experts, expert_weight, "down_proj", layer_id)
    elif match := re.search(
        r"model\.language_model\.layers\.(\d+)\.mixer\.experts\.(gate_and_up_projs|down_projs)",
        name,
    ):
        def yield_weight(n_experts, expert_weight, w_name, layer_id):
            for expert_id in range(n_experts):
                single_expert_weight = expert_weight[expert_id].contiguous()
                yield (
                    f"model.language_model.layers.{layer_id}.mixer.experts.{expert_id}.{w_name}.weight",
                    single_expert_weight,
                )
        layer_id = int(match.group(1))
        w_name = match.group(2)
        n_experts = expert_weight.shape[0]
        if w_name == "gate_and_up_projs":
            # gate_and_up_projs is `up_proj` in nemotron
            # shape: [experts, ffn_dim, hidden_dim]
            yield from yield_weight(
                n_experts, expert_weight, "up_proj", layer_id
            )
        else:
            yield from yield_weight(n_experts, expert_weight, "down_proj", layer_id)
    else:
        yield name, expert_weight

def patched_parallelize_fn(self):
    # whatever you want to return
    return parallelize, self

# MoE: Aux-free load balancing update bias after each step update.
def step_hook(self, step: int) -> Optional[dict]:
    if not hasattr(self, "_stateful_expert_load_per_layer"):
        self._stateful_expert_load_per_layer = {}

    enable_moe_load_balancing_training = self.cosmos_config.custom.get("enable_moe_load_balancing_training", True)
    report_every = self.cosmos_config.custom.get("n_step_per_workload_report", 10)

    if enable_moe_load_balancing_training:
        for name, module in self.language_model.named_modules():
            if 'NemotronHBlock' in type(module).__name__ and module.block_type == "moe":
                local_expert_load = module.mixer.gate.update_bias()
                with torch.no_grad():
                    if name not in self._stateful_expert_load_per_layer:
                        self._stateful_expert_load_per_layer[name] = local_expert_load
                    else:
                        self._stateful_expert_load_per_layer[name] += local_expert_load
    elif not hasattr(self, "_warn_moe_load_balancing_training_once"):
        self._warn_moe_load_balancing_training_once = True
        print("WARNING: MoE load balancing training is disabled. Please set enable_moe_load_balancing_training to True in the config['custom'] to enable it.")
    
    report_data = None
    if step % report_every == 0:
        report_data = report_moe_load_to_wandb(step, self._stateful_expert_load_per_layer, prefix="moe")
        # reset window accumulation
        self._stateful_expert_load_per_layer = {}
    return report_data

def get_dataset(config: CosmosConfig):
    return CustomWebDatasetDataset()

if __name__ == "__main__":
    import cosmos_rl

    if os.environ.get("APPLY_QWEN3VL_PATCH") and os.environ.get("APPLY_SIGLIP2_PATCH"):
        raise RuntimeError(
            "APPLY_QWEN3VL_PATCH and APPLY_SIGLIP2_PATCH cannot both be set. "
            "Please set exactly one to select the model-specific patches."
        )

    if os.environ.get("APPLY_QWEN3VL_PATCH"):
        # Qwen3VL: skip NemotronH-specific overrides, apply Qwen3VL-specific patches
        # from qwen3_vl_patches import apply_qwen3vl_patches
        # apply_qwen3vl_patches()
        pass
    elif os.environ.get("APPLY_SIGLIP2_PATCH"):
        # SigLIP2: uses NemotronH as LLM backbone, same monkey patches apply
        cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(patched_parallelize_fn)
        cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
        cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
        cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor
    else:
        # NemotronH: monkey patching for EP parallelization, weight conversion, etc.
        cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(patched_parallelize_fn)
        cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
        cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
        cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor

    # Launch the worker
    cosmos_rl.launcher.worker_entry.main(
        # Uncomment this if you want to use a custom dataset
        dataset=get_dataset,
    )
