from __future__ import annotations

import pytest

from cosmos_rl.tools.custom_hooks import tao_vl_reason_daft_sft_example as hook
from cosmos_rl.utils import pynv_video_reader, system_pyav_video_reader


def _config(video_decoder: str) -> hook.CustomConfig:
    return hook.CustomConfig.model_validate(
        {
            "train_dataset": {"annotation_path": "/tmp/train.json"},
            "video_decoder": video_decoder,
        }
    )


@pytest.mark.parametrize("video_decoder", ["cpu", "torchvision"])
def test_system_decoder_is_registered_outside_controller(
    monkeypatch, video_decoder: str
):
    calls = []
    monkeypatch.setenv("COSMOS_ROLE", "Policy")
    monkeypatch.setattr(
        system_pyav_video_reader,
        "register_system_pyav_video_reader",
        lambda: calls.append(video_decoder),
    )

    assert hook.configure_video_decoder(_config(video_decoder)) == {
        "backend": "torchvision",
        "implementation": "system_pyav_sparse",
        "requested": video_decoder,
    }
    assert calls == [video_decoder]
    assert hook.os.environ["COSMOS_DATALOADER_VIDEO_DECODER"] == "system_pyav"


def test_controller_does_not_register_system_decoder(monkeypatch):
    monkeypatch.setenv("COSMOS_ROLE", "Controller")
    monkeypatch.setattr(
        system_pyav_video_reader,
        "register_system_pyav_video_reader",
        lambda: pytest.fail("controller must not register a worker decoder"),
    )

    assert hook.configure_video_decoder(_config("torchvision")) is None


def test_pynv_decoder_registration_is_unchanged(monkeypatch):
    calls = []
    expected = {"backend": "pynvvideocodec"}
    monkeypatch.setenv("COSMOS_ROLE", "Policy")
    monkeypatch.setattr(
        pynv_video_reader,
        "register_pynv_video_reader",
        lambda **kwargs: calls.append(kwargs) or expected,
    )

    assert hook.configure_video_decoder(_config("pynvvideocodec")) == expected
    assert calls == [{"cache_size": 2, "video_override_map": None}]
    assert "COSMOS_DATALOADER_VIDEO_DECODER" not in hook.os.environ
