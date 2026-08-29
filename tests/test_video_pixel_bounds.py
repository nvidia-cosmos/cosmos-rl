from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "cosmos_rl" / "utils" / "video_pixel_bounds.py"
SPEC = importlib.util.spec_from_file_location("video_pixel_bounds", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
video_pixel_bounds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video_pixel_bounds)
normalize_video_pixel_bounds = video_pixel_bounds.normalize_video_pixel_bounds


RUNTIME = SimpleNamespace(VIDEO_MIN_TOKEN_NUM=128, SPATIAL_MERGE_SIZE=2)


def test_missing_minimum_is_normalized_on_the_reader_input_object() -> None:
    source = {"video": "clip.mp4", "max_pixels": 81920}

    normalized = normalize_video_pixel_bounds(source, 16, RUNTIME)

    assert normalized == {
        "video": "clip.mp4",
        "min_pixels": 81920,
        "max_pixels": 81920,
    }
    assert normalized is source
    assert source["min_pixels"] == source["max_pixels"]


def test_string_pixel_bounds_are_canonicalized_on_the_reader_input_object() -> None:
    source = {"video": "clip.mp4", "max_pixels": "81920"}

    normalized = normalize_video_pixel_bounds(source, 16, RUNTIME)

    assert normalized is source
    assert normalized["min_pixels"] == 81920
    assert normalized["max_pixels"] == 81920
    assert isinstance(normalized["min_pixels"], int)
    assert isinstance(normalized["max_pixels"], int)


def test_runtime_default_is_preserved_when_it_fits_explicit_maximum() -> None:
    source = {"video": "clip.mp4", "max_pixels": 262144}
    assert normalize_video_pixel_bounds(source, 16, RUNTIME) is source


def test_consistent_explicit_bounds_remain_authoritative() -> None:
    source = {
        "video": "clip.mp4",
        "min_pixels": 65536,
        "max_pixels": 81920,
    }
    assert normalize_video_pixel_bounds(source, 16, RUNTIME) is source


def test_contradictory_explicit_bounds_fail() -> None:
    with pytest.raises(ValueError, match="contradictory"):
        normalize_video_pixel_bounds(
            {"min_pixels": 131072, "max_pixels": 81920}, 16, RUNTIME
        )


def test_launcher_propagates_every_child_failure() -> None:
    launcher = (
        Path(__file__).parents[1] / "cosmos_rl" / "launcher" / "launch_all.py"
    ).read_text()
    assert "controller_id == -1 or i == controller_id" not in launcher
    assert "Propagate every child failure immediately" in launcher
