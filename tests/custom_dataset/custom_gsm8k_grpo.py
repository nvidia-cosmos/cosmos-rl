from cosmos_rl.dispatcher.data.packer import worker_entry_parser
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.tools.dataset.gsm8k_grpo import (
    GSM8kDataset,
    GSM8kValDataset,
    custom_reward_fn,
    custom_logger_fn,
    GSM8kDataPacker,
)
from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
from torch.utils.data import Dataset


class MinimalGSM8kDataset(GSM8kDataset):
    def __len__(self):
        return 8


class MinimalGSM8kValDataset(GSM8kValDataset):
    def __len__(self):
        return 8


if __name__ == "__main__":
    parser = worker_entry_parser()

    # Users can add custom arguments here
    parser.add_argument(
        "--foo", type=str, default="bar", help="The custom optional argument name."
    )
    parser.add_argument(
        "x_arg",
        type=str,
        default=None,
        help="The custom positional argument name.",
    )

    try:
        args = parser.parse_args()
        assert args.x_arg == "cosmos_rl"
        assert args.foo == "cosmos"
    except SystemExit as e:
        logger.error("Error parsing arguments.")
        raise e

    def get_dataset(config: CosmosConfig) -> Dataset:
        return MinimalGSM8kDataset()

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        return MinimalGSM8kValDataset()

    launch_worker(
        dataset=get_dataset,
        val_dataset=get_val_dataset,
        reward_fns=[custom_reward_fn],
        data_packer=GSM8kDataPacker(),
        val_data_packer=GSM8kDataPacker(),
        custom_logger_fns=[custom_logger_fn],
        args=args,  # Note: args must be passed if you want to use custom arguments
    )
