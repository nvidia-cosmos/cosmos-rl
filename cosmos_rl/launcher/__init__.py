import os


def launch():
    role = os.environ.get("COSMOS_ROLE")
    assert role in ["Policy", "Rollout", "Controller"], f"Invalid role: {role}"
    if role == "Policy":
        from cosmos_rl.policy.train import main as policy_main

        policy_main()
    elif role == "Rollout":
        from cosmos_rl.rollout.rollout_entrance import run_rollout

        run_rollout()
    pass


if __name__ == "__main__":
    launch()
