import argparse

from cosmos_rl_reward.utils.redis import start_redis_server
from cosmos_rl_reward.utils.logging import logger


def main() -> None:
    """Start a standalone Redis server for cosmos-rl-reward."""

    parser = argparse.ArgumentParser(
        description="Start a standalone Redis server for Cosmos-RL-Reward."
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Preferred Redis port (will pick next available if occupied).",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="/tmp/redis.log",
        help="Redis logfile path (default: /tmp/redis.log).",
    )

    args = parser.parse_args()

    port = start_redis_server(args.port, redis_logfile_path=args.logfile)
    if port == -1:
        raise SystemExit(1)

    logger.info(f"[cosmos-rl-reward-redis] Redis started on port {port}")


if __name__ == "__main__":
    main()
