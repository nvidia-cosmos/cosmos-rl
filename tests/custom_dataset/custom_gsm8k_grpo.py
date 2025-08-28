from cosmos_rl.dispatcher.data.packer import worker_entry_parser
from cosmos_rl.utils.logging import logger
import sys

if __name__ == "__main__":
    parser = worker_entry_parser()

    # add custom arguments here
    parser.add_argument("--foo", type=str, default="bar", help="The custom optional argument name.")
    parser.add_argument(
        "x_arg",
        type=str,
        default=None,
        help="The custom positional argument name.",
    )
    try:
        args = parser.parse_args()
        logger.info(f"LMS: args: {args}")
        # assert args.x_arg == "cosmos_rl"
        # assert args.foo == "cosmos"
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        sys.exit(1)
    sys.exit(0)