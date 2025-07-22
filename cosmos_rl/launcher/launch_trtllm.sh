#!/usr/bin/env bash

# Default values
NGPU=2
NNODES=1
RDZV_ENDPOINT="localhost:0"
SCRIPT=""
CONFIG=""

print_help() {
  echo ""
  echo "Usage: ./launch_trtllm.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --nnodes <int>                     Number of nodes to launch. Default: 1"
  echo "  --ngpus <int>                      Number of GPUs per node. Default: 2"
  echo "  --rdzv-endpoint <host:port>        Rendezvous endpoint for distributed training. Default: localhost:0"
  echo "  --script <script>                  The user script to run before launch."
  echo "  --config <path>                    The path to the config file."
  echo "  --help                             Show this help message"
  echo "Examples:"
  echo "  ./launch_trtllm.sh --ngpus 4 --log-rank 0,1"
  echo ""
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --ngpus)
    NGPU="$2"
    shift 2
    ;;
  --nnodes)
    NNODES="$2"
    shift 2
    ;;
  --log-rank)
    LOG_RANKS="$2"
    shift 2
    ;;
  --type)
    TYPE="$2"
    shift 2
    ;;
  --rdzv-endpoint)
    RDZV_ENDPOINT="$2"
    shift 2
    ;;
  --script)
    SCRIPT="$2"
    shift 2
    ;;
  --config)
    CONFIG="$2"
    shift 2
    ;;
  --help)
    print_help
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    print_help
    exit 1
    ;;
  esac
done


export TORCH_CPP_LOG_LEVEL="ERROR"
DEFAULT_MODULE="cosmos_rl.rollout.rollout_entrance"
export COSMOS_ROLE="Rollout"
export RDZV_ENDPOINT="$RDZV_ENDPOINT"

COSMOS_WORLD_SIZE=$((NNODES * NGPU))
export COSMOS_WORLD_SIZE
COSMOS_LOCAL_WORLD_SIZE=$((NGPU))
export COSMOS_LOCAL_WORLD_SIZE
export COSMOS_ROLLOUT_BACKEND="trtllm"

if [ -z "$COSMOS_CONTROLLER_HOST" ]; then
  echo "Error: COSMOS_CONTROLLER_HOST is not set. Please pass it in like:"
  echo "  COSMOS_CONTROLLER_HOST=<controller_host>:<controller_port> ./launch_replica.sh"
  exit 1
fi

LAUNCH_CMD=python


if [ -n "$SCRIPT" ]; then
  if [[ "$SCRIPT" != *.py ]]; then
    LAUNCH_CMD+=(
      -m "$SCRIPT"
    )
  else
    LAUNCH_CMD+=(
      "$SCRIPT"
    )
  fi
else
  LAUNCH_CMD+=(
    -m "$DEFAULT_MODULE"
  )
fi

if [ -n "$CONFIG" ]; then
  LAUNCH_CMD+=(
    --config "$CONFIG"
  )
fi

"${LAUNCH_CMD[@]}"
