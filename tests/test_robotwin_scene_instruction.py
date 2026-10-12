"""Scene-dependent instruction metadata must refresh on every reset."""

from unittest.mock import patch

from cosmos_rl.simulators.robotwin.venv import SubEnv


def test_reset_uses_actual_scene_metadata_and_retry_seed():
    class Task:
        def setup_demo(self, now_ep_num, seed, **kwargs):
            if seed == 2:
                raise RuntimeError("invalid requested scene")
            self.seed = seed

        def get_info(self):
            return {"arm": "left" if self.seed == 1 else "right"}

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_instruction(self):
            return self.instruction

        def get_obs(self):
            return {}

        def close_env(self):
            pass

    env = object.__new__(SubEnv)
    env.task = None
    env.task_args = {}
    env.env_id = 0
    env.env_seed = 1
    env.current_task_name = None
    env.create_instruction = (
        lambda task: f"{env.episode_info_list[0]['arm']}:{env.env_seed}"
    )
    with (
        patch(
            "cosmos_rl.simulators.robotwin.venv._class_decorator",
            side_effect=lambda _: Task(),
        ),
        patch(
            "cosmos_rl.simulators.robotwin.venv._update_obs", side_effect=lambda x: x
        ),
    ):
        assert env.reset("hammer", 1)["instruction"] == "left:1"
        assert env.reset("hammer", 2)["instruction"] == "right:3"
        assert env.reset("hammer", 1)["instruction"] == "left:1"
