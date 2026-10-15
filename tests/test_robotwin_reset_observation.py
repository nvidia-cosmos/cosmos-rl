"""Observation reads may advance randomized lighting; reset must not re-render."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from cosmos_rl.simulators.robotwin.env_wrapper import EnvStates, RoboTwinEnvWrapper


def check_reset(asynchronous):
    env = object.__new__(RoboTwinEnvWrapper)
    env.env_states = [EnvStates(env_idx=i) for i in range(3)]
    env._pending_reset_obs = {}
    observations = [
        {"instruction": "left", "image": 17},
        {"instruction": "right", "image": 29},
    ]
    env.env = SimpleNamespace(
        reset=Mock(return_value=observations),
        get_obs=Mock(side_effect=AssertionError("re-rendered initial observation")),
    )
    env._extract_image_and_state = lambda obs: {
        "full_images": np.array([x["image"] for x in obs]),
        "states": np.zeros((len(obs), 1)),
    }
    if asynchronous:
        env.reset_async([2, 0], [1, 1], [30024, 30042], False)
        obs, descriptions = env.reset_wait([0, 2])
        assert obs["full_images"].tolist() == [29, 17]
        assert descriptions == ["right", "left"]
        assert not env._pending_reset_obs
    else:
        obs, descriptions = env.reset([2, 0], [1, 1], [30024, 30042], False)
        assert obs["full_images"].tolist() == [17, 29]
        assert descriptions == ["left", "right"]
    assert env.env_states[1].current_obs is None
    assert env.env_states[2].current_obs["full_images"] == 17
    assert env.env_states[0].current_obs["full_images"] == 29
    env.env.get_obs.assert_not_called()


def test_sync_reset_preserves_initial_observation():
    check_reset(False)


def test_async_reset_preserves_initial_observation():
    check_reset(True)
