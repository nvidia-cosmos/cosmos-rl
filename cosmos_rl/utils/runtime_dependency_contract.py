# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time compatibility checks for Cosmos-RL runtime dependencies."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
_QWEN_FORCE_ANCHOR = (
    'FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)\n'
)
_QWEN_WORKER_HELPERS = '''

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_forced_video_reader(video_reader_backend: str) -> None:
    """Install the baked GPU reader inside spawned DataLoader workers."""
    if video_reader_backend != "pynvvideocodec":
        return
    if video_reader_backend in VIDEO_READER_BACKENDS:
        return

    from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader

    register_pynv_video_reader(
        cache_size=int(os.getenv("TAO_PYNV_VIDEO_CACHE_SIZE", "0")),
        decoder_cache_size=int(os.getenv("TAO_PYNV_DECODER_CACHE_SIZE", "4")),
        video_override_map=os.getenv("TAO_PYNV_VIDEO_OVERRIDE_MAP") or None,
        strict=_env_flag("TAO_PYNV_VIDEO_STRICT", True),
    )
    if video_reader_backend not in VIDEO_READER_BACKENDS:
        raise RuntimeError(
            "pynvvideocodec registration completed without installing its Qwen backend"
        )
'''
_QWEN_OLD_BACKEND = """@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend
"""
_QWEN_NEW_BACKEND = """@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    forced_video_reader = os.getenv(
        "FORCE_QWENVL_VIDEO_READER", FORCE_QWENVL_VIDEO_READER
    )
    if forced_video_reader is not None:
        video_reader_backend = forced_video_reader
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    _ensure_forced_video_reader(video_reader_backend)
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend
"""
_QWEN_OLD_FALLBACK = """        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
"""
_QWEN_NEW_FALLBACK = """        except Exception as e:
            if video_reader_backend == "pynvvideocodec" or _env_flag(
                "TAO_PYNV_VIDEO_STRICT", False
            ):
                raise RuntimeError(
                    "strict GPU video decoding failed; CPU fallback is disabled"
                ) from e
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
"""
_QWEN_FETCH_ANCHOR = """def fetch_video(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False,
                return_video_metadata: bool = False) -> Union[torch.Tensor, List[Image.Image]]:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
"""
_QWEN_FETCH_WITH_PIXEL_NORMALIZATION = """def fetch_video(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False,
                return_video_metadata: bool = False) -> Union[torch.Tensor, List[Image.Image]]:
    if os.getenv("FORCE_QWENVL_VIDEO_READER", FORCE_QWENVL_VIDEO_READER) == "pynvvideocodec":
        from cosmos_rl.utils.video_pixel_bounds import normalize_video_pixel_bounds

        normalize_video_pixel_bounds(ele, image_patch_size, sys.modules[__name__])
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
"""


def repair_qwen_pynv_worker_source(source: str) -> tuple[str, bool]:
    """Install strict PyNvVideoCodec registration in qwen-vl-utils 0.0.14."""
    worker_markers = (
        "def _ensure_forced_video_reader(",
        "_ensure_forced_video_reader(video_reader_backend)",
        'os.getenv("TAO_PYNV_DECODER_CACHE_SIZE", "4")',
        "strict GPU video decoding failed; CPU fallback is disabled",
    )
    pixel_markers = (
        "from cosmos_rl.utils.video_pixel_bounds import normalize_video_pixel_bounds",
        "normalize_video_pixel_bounds(ele, image_patch_size, sys.modules[__name__])",
    )
    worker_markers_present = tuple(marker in source for marker in worker_markers)
    pixel_markers_present = tuple(marker in source for marker in pixel_markers)
    if all(worker_markers_present) and all(pixel_markers_present):
        normalized = source if source.endswith("\n") else source + "\n"
        return normalized, normalized != source
    if any(worker_markers_present) and not all(worker_markers_present):
        raise RuntimeError(
            "Unrecognized qwen-vl-utils video implementation; refusing blind patch"
        )
    if any(pixel_markers_present) and not all(pixel_markers_present):
        raise RuntimeError(
            "Unrecognized qwen-vl-utils pixel-bound contract; refusing blind patch"
        )

    repaired = source
    if not all(worker_markers_present):
        if (
            repaired.count(_QWEN_FORCE_ANCHOR) != 1
            or repaired.count(_QWEN_OLD_BACKEND) != 1
            or repaired.count(_QWEN_OLD_FALLBACK) != 1
        ):
            raise RuntimeError(
                "Unrecognized qwen-vl-utils video implementation; refusing blind patch"
            )
        repaired = repaired.replace(
            _QWEN_FORCE_ANCHOR,
            _QWEN_FORCE_ANCHOR + _QWEN_WORKER_HELPERS,
        )
        repaired = repaired.replace(_QWEN_OLD_BACKEND, _QWEN_NEW_BACKEND)
        repaired = repaired.replace(_QWEN_OLD_FALLBACK, _QWEN_NEW_FALLBACK)

    if not all(pixel_markers_present):
        if repaired.count(_QWEN_FETCH_ANCHOR) != 1:
            raise RuntimeError(
                "Unrecognized qwen-vl-utils fetch_video implementation; refusing blind patch"
            )
        repaired = repaired.replace(
            _QWEN_FETCH_ANCHOR,
            _QWEN_FETCH_WITH_PIXEL_NORMALIZATION,
        )
    if not repaired.endswith("\n"):
        repaired += "\n"
    return repaired, True


def _qwen_vision_process_path() -> Path:
    spec = importlib.util.find_spec("qwen_vl_utils.vision_process")
    if spec is None or spec.origin is None:
        raise RuntimeError("qwen-vl-utils vision_process.py is not installed")
    return Path(spec.origin)


def repair_qwen_pynv_worker() -> None:
    """Patch the installed qwen-vl-utils tree during the image build."""
    path = _qwen_vision_process_path()
    repaired, changed = repair_qwen_pynv_worker_source(path.read_text(encoding="utf-8"))
    if changed:
        path.write_text(repaired, encoding="utf-8")
        importlib.invalidate_caches()
    verify_qwen_pynv_worker()


def verify_qwen_pynv_worker() -> None:
    """Verify spawned workers lazily install strict GPU video decoding."""
    path = _qwen_vision_process_path()
    source = path.read_text(encoding="utf-8")
    required = (
        "def _ensure_forced_video_reader(",
        "_ensure_forced_video_reader(video_reader_backend)",
        'os.getenv("TAO_PYNV_DECODER_CACHE_SIZE", "4")',
        "strict GPU video decoding failed; CPU fallback is disabled",
        "from cosmos_rl.utils.video_pixel_bounds import normalize_video_pixel_bounds",
        "normalize_video_pixel_bounds(ele, image_patch_size, sys.modules[__name__])",
    )
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise RuntimeError(
            "qwen-vl-utils strict GPU worker contract is missing: " + ", ".join(missing)
        )
    print(f"qwen-vl-utils strict GPU worker contract ready: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-qwen-pynv-worker", action="store_true")
    parser.add_argument("--verify-qwen-pynv-worker", action="store_true")
    args = parser.parse_args()
    if not any(
        (
            args.repair_qwen_pynv_worker,
            args.verify_qwen_pynv_worker,
        )
    ):
        parser.error("select at least one runtime dependency check")
    if args.repair_qwen_pynv_worker:
        repair_qwen_pynv_worker()
    elif args.verify_qwen_pynv_worker:
        verify_qwen_pynv_worker()


if __name__ == "__main__":
    main()
