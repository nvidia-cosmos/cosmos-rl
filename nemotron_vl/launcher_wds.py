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
import copy
from typing import Optional
import os, sys
# Enable EP mesh to be represented by TP mesh, and also treat EP as a sub-group of Data Parallelism.
os.environ["TP_EP_INTERCHANGABLE_WITH_DP_FUSED"] = "1"
import webdataset as wds
from torch.utils.data import IterableDataset
from PIL import Image
import io
from typing import Any, Iterator, Union, List, Optional, Sequence
import glob
import torch, re
import numpy as np
import decord
from concurrent.futures import ThreadPoolExecutor
from qwen_vl_utils.vision_process import smart_nframes, calculate_video_frame_range
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nemotron_parallelize import parallelize
from weight_converter import convert_weight_from_hf
from torch.utils.data import Dataset
from cosmos_rl.policy.config import Config as CosmosConfig
try:
    import wandb
except ImportError:
    wandb = None
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

def modify_messages(messages, max_pixels=None, max_video_frames=None):
    for message in messages:
        if isinstance(message['content'], str):
            message['content'] = [{'type': 'text', 'text': message['content']}]
        for content in message['content']:
            if content['type'] == 'image':
                if max_pixels is not None:
                    content['max_pixels'] = max_pixels
            elif content['type'] == 'video':
                content['fps'] = 1
                if max_video_frames is not None:
                    content['max_frames'] = max_video_frames
                if max_pixels is not None:
                    content['total_pixels'] = max_pixels
    return messages

class CustomJSONLDataset(Dataset):
    '''
    Custom dataset for Nemotron-3-Nano Vision-Language Model alignment.
    Assume the dataset are in JSONL format stored in `config.train.train_policy.dataset.name`, and each line is a JSON object with 'messages' key.
    '''
    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.data_list = []
        data_path = config.train.train_policy.dataset.name
        jsonl_files = sorted(
            [f for f in os.listdir(data_path) if f.endswith(".jsonl")]
        )
        for file_name in jsonl_files:
            if not config.custom.get("include_video", False) and 'webvid' in file_name:
                continue
            with open(os.path.join(data_path, file_name)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data_list.append(json.loads(line)['messages'])
        self.max_pixels = config.policy.model_max_length * 0.9 * ((16 * 2) ** 2)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = copy.deepcopy(self.data_list[idx])
        sample = modify_messages(sample, self.max_pixels)
        return sample

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
            # normalize (same as your jsonl loader did)
            msg["content"] = [{"type": "text", "text": content}]
            content = msg["content"]
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and "wds_field" in c:
                yield c


def _decode_video_from_bytes(blob: bytes, ele: dict,
                              decoding_timeout: int = 60) -> tuple[list[Image.Image], float, float]:
    """Decode video bytes in-memory and return sampled frames as PIL Images.

    Matches qwen_vl_utils._read_video_decord behaviour: uses the full content
    dict *ele* so that smart_nframes respects per-sample fps / min_frames /
    max_frames, and calculate_video_frame_range honours video_start / video_end.

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

    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()

    # Timeout guard against hanging decord (same pattern as video_decoder_qwen.py)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: vr.get_batch(idx).asnumpy())
    try:
        frames = future.result(timeout=decoding_timeout)
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"Video decoding timed out after {decoding_timeout}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    pil_frames = [Image.fromarray(frame) for frame in frames]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # Cleanup
    vr.seek(0)
    del vr

    return pil_frames, sample_fps, video_fps


def _attach_media_from_sample(messages: list[dict], sample: dict,
                               default_fps: float = 1.0,
                               default_max_frames: int = 768) -> list[dict] | None:
    """
    Replace each content item containing 'wds_field' with actual media payload.

    For images: set c["image"] = PIL.Image
    For videos: decode in-memory to list[PIL.Image] and set c["video"],
                c["sample_fps"], and c["raw_fps"].

    *default_fps* and *default_max_frames* are applied as fallbacks when the
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
            c.setdefault("fps", default_fps)
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

        webdataset_root_paths = [
            # "/workspace/simonz/data/vlm-data-debug/laion_recaption_filter_train_shuf_3m",
            # "/workspace/simonz/data/vlm-data-debug/infinitmm_stage1_qwenvl_filter_clip_id_shuf_5m",
            # "/workspace/simonz/data/vlm-data-debug/synthdog-en-chat-sample-250k-stage1",
            # "/workspace/simonz/data/vlm-data-debug/synthdog-zh-chat-sample-250k-stage1",
            # "/workspace/simonz/data/vlm-data-debug/unsplash_filter_001_800k_grounding_caption_chat_stage1_train",
            "/workspace/simonz/data/vlm-data-debug/webvid_caption_openai_format_1M_stage1_train",
        ]

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

        # Keep your existing max_pixels heuristic
        self.max_pixels = config.policy.model_max_length * 0.9 * ((16 * 2) ** 2)

        # Cap video frames so tokens fit in model_max_length.
        # Each frame produces at least VIDEO_MIN_TOKEN_NUM=128 tokens in qwen_vl_utils.
        # Reserve ~half the context for video, round down to FRAME_FACTOR=2.
        max_vf = max(4, int(config.policy.model_max_length * 0.5 / 128))
        self.max_video_frames = max_vf // 2 * 2  # align to FRAME_FACTOR

        # Controls (put these in config.custom if you want)
        custom = config.custom if hasattr(config, "custom") and config.custom is not None else {}
        self.include_video = bool(custom.get("include_video", False))
        self.shuffle_buf = int(custom.get("wds_shuffle", 2000))  # 0 disables

        self.setup_wds_dataset()

    def setup_wds_dataset(self):
        # Build WDS pipeline

        # version-safe
        nodesplitter = getattr(wds, "split_by_node", None) or wds.shardlists.split_by_node
        workersplitter = getattr(wds, "split_by_worker", None) or wds.shardlists.split_by_worker
        ResampledShards = getattr(wds, "ResampledShards", None) or wds.shardlists.ResampledShards


        pipeline = wds.DataPipeline(
            # Sample shards with replacement
            ResampledShards(self.urls),
            # NB: if `resampled=True`, and even if we split by node and worker, we still expect to see potential duplicates between nodes and workers.
            # This is expected and hopefully could be uniformly distributed.
            #
            # Split stream by nodes, since we do not expect to see duplicates between nodes
            nodesplitter,          # shard-level split across ranks
            # Split stream by workers, since iterable dataset is not randomly accessible, each worker should see different samples
            workersplitter,        # shard-level split across dataloader workers
            wds.tarfile_to_samples(),
            # Shuffle samples inside a buffer/pool of size `shuffle_buf`
            wds.shuffle(self.shuffle_buf),   # sample-level shuffle buffer
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
                # Normally with resampled=True this should not happen,
                # but it CAN happen if upstream filters drop everything temporarily,
                # or if the pipeline got exhausted for some reason.
                # Recreate the iterator and keep going.
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
            default_fps=1.0,  # matches modify_messages' hardcoded fps=1
            default_max_frames=self.max_video_frames,
        )
        if messages is None:
            return None

        # Apply your existing mutations (max_pixels, fps, max_frames, etc.)
        messages = modify_messages(messages, self.max_pixels, self.max_video_frames)
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
    # Do some monkey patching to support Nemotron-3-Nano Vision-Language Model parallelization.
    import cosmos_rl
    # Override the parallelize_fn to support EP parallelization.
    # cosmos_rl.policy.model.hf_models.HFModel.parallelize_fn = property(patched_parallelize_fn)
    # # Override the convert_weight_from_hf to support EP weight sharding during initialization
    # cosmos_rl.policy.model.hf_models.convert_weight_from_hf = convert_weight_from_hf
    # # Override the step_hook to enable aux-free load balancing update bias after each step update.
    # cosmos_rl.policy.model.hf_models.HFModel.step_hook = step_hook
    # # Map the weight name from custom DeepEP convention back to HF convention for safetensor saving.
    # cosmos_rl.policy.model.hf_models.weight_mapper.HFModelWeightMapper.policy_map_local_key_for_export_tensor = policy_map_local_key_for_export_tensor
    
    # Launch the worker
    cosmos_rl.launcher.worker_entry.main(
        # Uncomment this if you want to use a custom dataset
        dataset=get_dataset,
    )

