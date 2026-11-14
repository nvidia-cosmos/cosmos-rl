# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Serve real prompt/report APIs and the controller monitor with CPU fixtures."""

import itertools
import sys

import pytest
import uvicorn

from cosmos_rl.dispatcher import run_web_panel as web
from cosmos_rl.dispatcher.data.schema import RLPayload
from test_training_fetch_receipts import setup


if __name__ == "__main__":
    patch = pytest.MonkeyPatch()
    item, _, _ = setup(patch)
    status = item.policy_status_manager
    policy = item.config.train.train_policy
    policy.on_policy, policy.allowed_outdated_steps = False, 1
    policy.max_inflight_steps, policy.max_retry_for_on_policy = 2, 1
    status.set_discard_refill_hook(item.register_discarded_samples_for_refill)
    indices = itertools.count()
    item.data_fetcher.get_batched_prompt.side_effect = lambda n, *args, **kwargs: (
        [RLPayload(prompt_idx=next(indices)) for _ in range(n)],
        False,
    )
    # No cluster or Redis is needed. Keep real HTTP dispatch, settlement,
    # receipt handling, lifespan supervision and process termination.
    status.maintain_life_status = lambda: None
    status.redis_handler._publication_failure = None
    item.rollout_status_manager.maintain_life_status = lambda _: None
    web.COSMOS_ROLLOUT_SCAN_INTERVAL = 0.01
    uvicorn.run(web.app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
