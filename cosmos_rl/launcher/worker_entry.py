import os
from typing import Optional, List, Callable, Union, Dict
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
from torch.utils.data import Dataset
import argparse


def main(
    dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
    data_packer: Optional[BaseDataPacker] = None,
    reward_fns: Optional[List[Callable]] = None,
    filter_reward_fns: Optional[List[Callable]] = None,
    val_dataset: Optional[Dataset] = None,
    val_reward_fns: Optional[List[Callable]] = None,
    val_data_packer: Optional[BaseDataPacker] = None,
    custom_logger_fns: Optional[List[Callable]] = None,
    hook_fns: Optional[Dict[str, Callable]] = None,
    sampler: Optional[Callable] = None,
    batch_sampler: Optional[Callable] = None,
    val_sampler: Optional[Callable] = None,
    val_batch_sampler: Optional[Callable] = None,
    args: Optional[argparse.Namespace] = None,
    **kwargs,
):
    if kwargs:
        logger.warning(f"Unused kwargs: {list(kwargs.keys())}")

    role = os.environ.get("COSMOS_ROLE")
    assert role in ["Policy", "Rollout", "Controller"], f"Invalid role: {role}"
    if role == "Controller":
        from cosmos_rl.dispatcher.run_web_panel import main as controller_main

        controller_main(
            dataset=dataset,
            data_packer=data_packer,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_dataset=val_dataset,
            val_reward_fns=val_reward_fns,
            val_data_packer=val_data_packer,
            custom_logger_fns=custom_logger_fns,
            hook_fns=hook_fns,
            sampler=sampler,
            batch_sampler=batch_sampler,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
            args=args,
        )
    elif role == "Policy":
        from cosmos_rl.policy.train import main as policy_main

        policy_main(
            dataset=dataset,
            data_packer=data_packer,
            val_dataset=val_dataset,
            val_data_packer=val_data_packer,
            sampler=sampler,
            batch_sampler=batch_sampler,
            custom_logger_fns=custom_logger_fns,
            hook_fns=hook_fns,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
        )
        return
    else:
        from cosmos_rl.rollout.rollout_entry import run_rollout

        run_rollout(
            dataset=dataset,
            reward_fns=reward_fns,
            filter_reward_fns=filter_reward_fns,
            val_dataset=val_dataset,
            val_reward_fns=val_reward_fns,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            custom_logger_fns=custom_logger_fns,
            hook_fns=hook_fns,
        )
        return


if __name__ == "__main__":
    main()
