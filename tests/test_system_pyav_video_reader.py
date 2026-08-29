from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest
import torch

from cosmos_rl.utils import system_pyav_video_reader as reader


def test_missing_video_decoder_fails_before_native_decode(monkeypatch, tmp_path):
    video_path = tmp_path / "unsupported.mp4"
    video_path.touch()
    container = SimpleNamespace(
        streams=SimpleNamespace(video=[SimpleNamespace(codec_context=None)]),
        close=lambda: None,
    )
    monkeypatch.setattr(reader.av, "open", lambda _path: container)

    with pytest.raises(RuntimeError, match="no decoder for the primary video stream"):
        reader._decode_sparse({"video": str(video_path), "nframes": 8})


def test_registration_replaces_qwen_torchvision_backend(monkeypatch):
    import qwen_vl_utils.vision_process as vision_process

    monkeypatch.setenv("FORCE_QWENVL_VIDEO_READER", "torchvision")
    reader.register_system_pyav_video_reader()
    assert (
        vision_process.VIDEO_READER_BACKENDS["torchvision"]
        is reader.read_video_system_pyav
    )
    assert vision_process.fetch_video is reader.fetch_video_system_pyav_cached


def test_registration_rejects_cuda_codec_aliases(monkeypatch):
    class FakeCodec:
        def __init__(self, name, _mode):
            self.name = f"{name}_cuvid"

    monkeypatch.setattr(reader.av, "Codec", FakeCodec)
    with pytest.raises(RuntimeError, match="requires software h264/hevc codecs"):
        reader.register_system_pyav_video_reader()


def test_video_conversation_hook_accepts_explicit_torchvision_contract(monkeypatch):
    from cosmos_rl.tools.custom_hooks.tao_sft_example import (
        CustomConfig,
        configure_video_decoder,
    )

    monkeypatch.setenv("FORCE_QWENVL_VIDEO_READER", "torchvision")
    config = CustomConfig.model_validate(
        {
            "train_dataset": {"annotation_path": "/tmp/train.json"},
            "video_decoder": "torchvision",
        }
    )
    assert configure_video_decoder(config) == {
        "backend": "torchvision",
        "implementation": "system_pyav_sparse",
    }
    assert reader.os.environ["COSMOS_DATALOADER_VIDEO_DECODER"] == "system_pyav"


def test_video_conversation_hook_rejects_implicit_torchvision_contract(monkeypatch):
    from cosmos_rl.tools.custom_hooks.tao_sft_example import (
        CustomConfig,
        configure_video_decoder,
    )

    monkeypatch.delenv("FORCE_QWENVL_VIDEO_READER", raising=False)
    config = CustomConfig.model_validate(
        {
            "train_dataset": {"annotation_path": "/tmp/train.json"},
            "video_decoder": "torchvision",
        }
    )
    with pytest.raises(RuntimeError, match="FORCE_QWENVL_VIDEO_READER=torchvision"):
        configure_video_decoder(config)


def test_video_conversation_hook_forwards_explicit_pynv_cache_contract(monkeypatch):
    from cosmos_rl.tools.custom_hooks import tao_sft_example as hook
    from cosmos_rl.utils import pynv_video_reader

    calls = []
    monkeypatch.setenv("COSMOS_ROLE", "Policy")
    monkeypatch.setattr(
        pynv_video_reader,
        "register_pynv_video_reader",
        lambda **kwargs: calls.append(kwargs) or {"backend": "pynvvideocodec"},
    )
    config = hook.CustomConfig.model_validate(
        {
            "train_dataset": {"annotation_path": "/tmp/train.json"},
            "video_decoder": "pynvvideocodec",
            "video_cache_size": 341,
            "video_decoder_cache_size": 341,
        }
    )

    assert hook.configure_video_decoder(config) == {
        "backend": "pynvvideocodec"
    }
    assert calls == [
        {
            "cache_size": 341,
            "decoder_cache_size": 341,
            "video_override_map": None,
        }
    ]


def test_video_conversation_hook_defaults_to_validated_pynv_cache_profile() -> None:
    from cosmos_rl.tools.custom_hooks import tao_sft_example as hook

    config = hook.CustomConfig.model_validate(
        {"train_dataset": {"annotation_path": "/tmp/train.json"}}
    )

    assert config.video_cache_size == 0
    assert config.video_decoder_cache_size == 4


def test_repeated_parallel_reads_are_single_flight(monkeypatch):
    reader.clear_video_cache()
    monkeypatch.setattr(reader, "_CACHE_MAX_ITEMS", 8)
    calls = 0
    calls_lock = threading.Lock()

    def fake_decode(element):
        nonlocal calls
        with calls_lock:
            calls += 1
        return torch.zeros(8, 3, 2, 2), {"frames_indices": list(range(8))}, 1.0

    monkeypatch.setattr(reader, "_decode_sparse", fake_decode)
    element = {"video": "/tmp/repeated.mp4", "nframes": 8}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: reader.read_video_system_pyav(element), range(16))
        )

    assert calls == 1
    assert all(result[0] is results[0][0] for result in results)


def test_cache_key_separates_sampling_configuration():
    assert reader._cache_key({"video": "a.mp4", "nframes": 8}) != reader._cache_key(
        {"video": "a.mp4", "nframes": 16}
    )


def test_repeated_parallel_fetches_cache_processed_video(monkeypatch):
    reader.clear_video_cache()
    monkeypatch.setattr(reader, "_CACHE_MAX_ITEMS", 8)
    calls = 0
    calls_lock = threading.Lock()
    processed = torch.zeros(8, 3, 2, 2, dtype=torch.float32)

    def fake_fetch(
        element,
        image_patch_size=14,
        return_video_sample_fps=False,
        return_video_metadata=False,
    ):
        nonlocal calls
        with calls_lock:
            calls += 1
        threading.Event().wait(0.02)
        return (processed, {"fps": 1.0})

    monkeypatch.setattr(reader, "_ORIGINAL_FETCH_VIDEO", fake_fetch)
    element = {"video": "/tmp/repeated.mp4", "nframes": 8, "max_pixels": 64}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: reader.fetch_video_system_pyav_cached(
                    element, image_patch_size=16, return_video_metadata=True
                ),
                range(16),
            )
        )

    assert calls == 1
    assert all(result is results[0] for result in results)


def test_processed_cache_key_separates_resize_configuration():
    common = ({"video": "a.mp4", "nframes": 8}, 16, False, True)
    assert reader._processed_cache_key(*common) != reader._processed_cache_key(
        {"video": "a.mp4", "nframes": 8, "max_pixels": 64}, 16, False, True
    )


def test_distinct_cold_decodes_are_serialized(monkeypatch):
    reader.clear_video_cache()
    monkeypatch.setattr(reader, "_CACHE_MAX_ITEMS", 0)
    active = 0
    peak_active = 0
    calls_lock = threading.Lock()

    def fake_decode(element):
        nonlocal active, peak_active
        with calls_lock:
            active += 1
            peak_active = max(peak_active, active)
        threading.Event().wait(0.02)
        with calls_lock:
            active -= 1
        return torch.zeros(8, 3, 2, 2), {"frames_indices": list(range(8))}, 1.0

    monkeypatch.setattr(reader, "_decode_sparse", fake_decode)
    elements = [
        {"video": f"/tmp/video-{index}.mp4", "nframes": 8} for index in range(4)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(reader.read_video_system_pyav, elements))

    assert peak_active == 1
