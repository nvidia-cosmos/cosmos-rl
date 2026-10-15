"""Simulator action conventions must not reverse the ALOHA right gripper."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cosmos_rl.policy.model.vla import OpenVLA


def test_generation_preserves_robotwin_actions_and_libero_conversion():
    actions = np.arange(28, dtype=np.float32).reshape(1, 2, 14) / 28
    actions[..., 6] = [0.0, 1.0]
    actions[..., 13] = [1.0, 0.0]
    responses, logprobs = object(), object()
    fake = SimpleNamespace(
        model=SimpleNamespace(
            generate_action=lambda **kw: (actions.copy(), responses, logprobs)
        ),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    inputs = dict(input_ids=None, attention_mask=None, pixel_values=None)
    with patch("torch.autocast", return_value=nullcontext()):
        robotwin = OpenVLA.generate_action(fake, inputs, simulator_type="robotwin")
        libero = OpenVLA.generate_action(fake, inputs)
    np.testing.assert_array_equal(robotwin["action"], actions)
    np.testing.assert_array_equal(libero["action"][..., -1], [[-1.0, 1.0]])
    np.testing.assert_array_equal(libero["action"][..., :-1], actions[..., :-1])
    assert robotwin["responses"] is responses
    assert robotwin["old_log_probs"] is logprobs
