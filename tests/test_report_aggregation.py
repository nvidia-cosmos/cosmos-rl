# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report values may contain application metadata alongside scalar metrics."""

import math

import numpy as np
import pytest
import torch

from cosmos_rl.utils.report.aggregation import aggregate_report_data


@pytest.mark.parametrize(
    "suffix, expected",
    [
        ("", 3.0),
        ("_mean", 3.0),
        ("_avg", 3.0),
        ("_max", 4.0),
        ("_min", 2.0),
        ("_sum", 6.0),
        ("_count", 6.0),
        ("_cnt", 6.0),
        ("_std", math.sqrt(10.0)),
    ],
)
def test_numeric_reducers_skip_unsupported_and_nonfinite_values(suffix, expected):
    key = f"score{suffix}"
    reports = [
        {key: value}
        for value in (
            2.0,
            np.float32(4.0),
            None,
            [1.0, 2.0],
            [3.0],
            {"reason": "dropped"},
            float("nan"),
            float("inf"),
            -float("inf"),
            np.float32("nan"),
        )
    ]
    assert aggregate_report_data(reports, {}) == {key: pytest.approx(expected)}


@pytest.mark.parametrize(
    "value",
    [
        2,
        2.0,
        True,
        np.int8(2),
        np.uint64(2),
        np.float32(2),
        np.float64(2),
        np.bool_(True),
    ],
)
def test_supported_scalar_types(value):
    assert aggregate_report_data([{"value": value}], {}) == {"value": float(value)}


@pytest.mark.parametrize(
    "value",
    [
        None,
        float("nan"),
        np.float64("inf"),
        "2.0",
        2.0 + 0j,
        [1.0],
        [],
        {"reason": "dropped"},
        np.array(2.0),
        np.array([2.0]),
        torch.tensor(2.0),
        torch.tensor([2.0]),
    ],
)
def test_keys_with_only_invalid_values_are_omitted_even_when_missing_elsewhere(value):
    assert aggregate_report_data([{"value": value}, {}], {}) == {}


@pytest.mark.parametrize(
    "suffix, expected",
    [
        ("", 1.0),
        ("_mean", 1.0),
        ("_avg", 1.0),
        ("_max", 2.0),
        ("_min", 0.0),
        ("_sum", 2.0),
        ("_count", 2.0),
        ("_cnt", 2.0),
        ("_std", math.sqrt(2.0)),
    ],
)
def test_missing_key_keeps_zero_contribution(suffix, expected):
    key = f"score{suffix}"
    reports = [{key: 2.0}, {}, {key: None}, {key: float("nan")}]
    assert aggregate_report_data(reports, {}) == {key: pytest.approx(expected)}


def test_missing_max_preserves_zero_for_negative_observations():
    assert aggregate_report_data([{"score_max": -2}, {}], {}) == {"score_max": 0}


class ConversionForbidden:
    def __float__(self):
        raise AssertionError("metadata must not be converted")

    def __array__(self, *args, **kwargs):
        raise AssertionError("metadata must not be converted")

    def __str__(self):
        raise AssertionError("metadata must not be stringified")


def test_input_metadata_and_media_are_preserved_without_conversion(monkeypatch):
    metadata = ConversionForbidden()
    tensor = torch.tensor(4.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("excluded tensors must not be read or copied")

    for name in ("item", "cpu", "numpy", "to", "__array__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    ragged = [[1.0, 2.0], [3.0]]
    diagnostic = {"reason": metadata}
    images, videos = [metadata], {"video": metadata}
    reports = [
        {
            "metadata": metadata,
            "tensor": tensor,
            "steps": ragged,
            "diagnostics": diagnostic,
            "rollout_images": images,
            "score": 2.0,
        },
        {"rollout_images": object(), "rollout_videos": videos, "score": 4.0},
    ]
    original_keys = [set(report) for report in reports]
    result = aggregate_report_data(reports, {}, prefix="train/")
    assert result.keys() == {
        "train/rollout_images",
        "train/rollout_videos",
        "train/score",
    }
    assert result["train/score"] == 3.0
    assert result["train/rollout_images"] is images
    assert result["train/rollout_videos"] is videos
    assert [set(report) for report in reports] == original_keys
    for key, value in (
        ("metadata", metadata),
        ("tensor", tensor),
        ("steps", ragged),
        ("diagnostics", diagnostic),
        ("rollout_images", images),
    ):
        assert reports[0][key] is value


def test_caller_destination_wins_and_is_returned():
    existing = ConversionForbidden()
    destination = {"train/score": existing, "unrelated": 5}
    result = aggregate_report_data(
        [{"score": 2, "other_sum": 3}], destination, "train/"
    )
    assert result is destination
    assert result["train/score"] is existing
    assert result["train/other_sum"] == 3
    assert result["unrelated"] == 5


def test_media_uses_first_present_value_even_when_none():
    reports = [{}, {"rollout_images": None}, {"rollout_images": ["later"]}]
    assert aggregate_report_data(reports, {}) == {"rollout_images": None}


def test_empty_input_leaves_destination_unchanged():
    destination = {"existing": 3}
    assert aggregate_report_data([], destination) is destination
    assert destination == {"existing": 3}
    assert aggregate_report_data([{}, {}], {}) == {}


def test_util_import_keeps_existing_public_entry_point():
    from cosmos_rl.utils.util import aggregate_report_data as public_aggregate

    assert public_aggregate is aggregate_report_data
    assert public_aggregate([{"score": 2.0}, {}], {}) == {"score": 1.0}
