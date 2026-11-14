# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numeric/media projection of producer-owned reports for metric loggers."""

import math
from typing import Any, Dict

import numpy as np


def _is_finite_scalar(value: Any) -> bool:
    # Check the supported types before doing any conversion. In particular,
    # tensors and arrays must not be copied to the host or synchronized here.
    if not isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
        return False
    return not isinstance(value, (float, np.floating)) or bool(np.isfinite(value))


def aggregate_report_data(
    report_data_list, report_data, prefix: str = ""
) -> Dict[str, Any]:
    """Add numeric/media aggregates to ``report_data`` without changing inputs.

    Finite Python ``int``, ``float`` and ``bool`` values, and NumPy integer,
    floating and boolean scalars, are numeric observations. Explicit ``None``,
    non-finite values, arrays, tensors (including single-element tensors), and
    arbitrary metadata are excluded without conversion. Keys with no supported
    observation are omitted; their original values stay in the producer reports
    for application consumers.

    Missing keys still contribute zero whenever a key has at least one finite
    scalar observation. This preserves the existing sparse-report convention;
    an explicitly invalid value is excluded rather than counted as missing.
    Suffix reducers, first-reporter media selection, prefixing, and existing
    destination entries retain their original behavior. ``_std`` remains the
    approximate root mean square of reported standard deviations.
    """
    all_keys = set().union(*[r.keys() for r in report_data_list])
    for key in all_keys:
        prefixed_key = f"{prefix}{key}" if prefix else key
        if prefixed_key in report_data:
            continue
        if key in ("rollout_images", "rollout_videos"):
            # Only the master rank normally reports media; preserve its object.
            report_data[prefixed_key] = next(
                data[key] for data in report_data_list if key in data
            )
            continue

        values = []
        missing = 0
        for data in report_data_list:
            if key not in data:
                missing += 1
            elif _is_finite_scalar(data[key]):
                values.append(data[key])
        if not values:
            continue
        values.extend([0] * missing)

        if key.endswith("_max"):
            report_data[prefixed_key] = np.max(values)
        elif key.endswith("_min"):
            report_data[prefixed_key] = np.min(values)
        elif key.endswith(("_sum", "_count", "_cnt")):
            report_data[prefixed_key] = np.sum(values)
        elif key.endswith("_std"):
            report_data[prefixed_key] = math.sqrt(
                sum(value**2 for value in values) / len(values)
            )
        else:
            # Includes _mean, _avg, and names with no recognized suffix.
            report_data[prefixed_key] = np.mean(values)
    return report_data
