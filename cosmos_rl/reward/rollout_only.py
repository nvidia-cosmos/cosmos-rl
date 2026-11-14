# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Score generation outputs without preparing a training objective."""

import math


def prepare_scored_payload(payload, scores=None, metrics=None):
    """Keep every completion and its quality annotations.

    Zero advantages/rewards in unscored mode are wire-compatibility placeholders;
    the public rollout result consumer receives score=None for unscored outputs.
    """
    size = len(payload.completions)
    if scores is not None and len(scores) != size:
        raise ValueError("Scorer must return one score per generated completion")
    if metrics is not None and len(metrics) != size:
        raise ValueError("Scorer metrics must match generated completions")
    numeric_scores = None if scores is None else [float(value) for value in scores]
    if numeric_scores is not None and not all(
        math.isfinite(value) for value in numeric_scores
    ):
        raise ValueError("Scorer must return finite numeric scores")
    payload.rewards = numeric_scores if numeric_scores is not None else [0.0] * size
    payload.advantages = [0.0] * size
    payload.filter_rewards = [0.0] * size
    payload.report_metrics = (
        list(metrics) if metrics is not None else [{} for _ in range(size)]
    )
    if payload.completed_conversations is None:
        payload.completed_conversations = [[] for _ in range(size)]
    if payload.completion_token_ids is None:
        payload.completion_token_ids = [[] for _ in range(size)]
    if payload.completion_logprobs is None:
        payload.completion_logprobs = [[] for _ in range(size)]
    payload.n_ignore_prefix_tokens = [0] * size
    return payload
