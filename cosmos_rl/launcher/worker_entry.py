import os
from typing import Optional, List, Callable, Union
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.decorators import monitor_status
from cosmos_rl.policy.config import Config as CosmosConfig
from torch.utils.data import Dataset
import argparse


@monitor_status(name="Cosmos-RL Worker", mode="worker")
def main(
    dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
    data_packer: Optional[DataPacker] = None,
    reward_fns: Optional[List[Callable]] = None,
    val_dataset: Optional[Dataset] = None,
    val_reward_fns: Optional[List[Callable]] = None,
    val_data_packer: Optional[DataPacker] = None,
    custom_logger_fns: Optional[List[Callable]] = None,
    args: Optional[argparse.Namespace] = None,
    **kwargs,
):
    if kwargs:
        logger.warning(f"Unused kwargs: {list(kwargs.keys())}")

    role = os.environ.get("COSMOS_ROLE")
    assert role in ["Policy", "Rollout", "Controller"], f"Invalid role: {role}"

    try:
        if role == "Controller":
            from cosmos_rl.dispatcher.run_web_panel import main as controller_main

            controller_main(
                dataset=dataset,
                data_packer=data_packer,
                reward_fns=reward_fns,
                val_dataset=val_dataset,
                val_reward_fns=val_reward_fns,
                val_data_packer=val_data_packer,
            )
        elif role == "Policy":
            from cosmos_rl.policy.train import main as policy_main

            policy_main()
            return
        else:
            from cosmos_rl.rollout.rollout_entrance import run_rollout

            run_rollout()
            return
    except (KeyboardInterrupt, SystemError) as e:
        logger.warning(f"[Worker {role}] Worker was interrupted: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"[Worker {role}] Worker failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
