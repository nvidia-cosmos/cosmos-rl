#!/usr/bin/env bash

# Default values
NGPU=2
NNODES=1
LOG_RANKS=""
TYPE=""
RDZV_ENDPOINT="localhost:0"
SCRIPT=""
CONFIG=""
print_help() {
  echo ""
  echo "Usage: ./launch_replica.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --type <rollout|policy>            Required. Type of replica to launch."
  echo "  --nnodes <int>                     Number of nodes to launch. Default: 1"
  echo "  --ngpus <int>                      Number of GPUs per node. Default: 2"
  echo "  --log-rank <comma-separated ints>  Comma-separated list of ranks to enable logging. Default: Empty for all ranks."
  echo "  --rdzv-endpoint <host:port>        Rendezvous endpoint for distributed training. Default: localhost:0"
  echo "  --script <script>                  The user script to run before launch."
  echo "  --config <path>                    The path to the config file."
  echo "  --help                             Show this help message"
  echo "Examples:"
  echo "  ./launch_replica.sh --type rollout --ngpus 4 --log-rank 0,1"
  echo "  ./launch_replica.sh --type policy --ngpus 8 --log-rank 0"
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

if [ -z "$TYPE" ]; then
  echo "Error: --type is required"
  print_help
  exit 1
fi

export TORCH_CPP_LOG_LEVEL="ERROR"
if [ "$TYPE" == "rollout" ]; then
  DEFAULT_MODULE="cosmos_rl.rollout.rollout_entrance"
  export COSMOS_ROLE="Rollout"
elif [ "$TYPE" == "policy" ]; then
  DEFAULT_MODULE="cosmos_rl.policy.train"
  export COSMOS_ROLE="Policy"
else
  echo "Error: Invalid --type value '$TYPE'. Must be 'rollout' or 'policy'."
  print_help
  exit 1
fi

if [ -z "$COSMOS_CONTROLLER_HOST" ]; then
  echo "Error: COSMOS_CONTROLLER_HOST is not set. Please pass it in like:"
  echo "  COSMOS_CONTROLLER_HOST=<controller_host>:<controller_port> ./launch_replica.sh"
  exit 1
fi

TORCHRUN_CMD=(
  torchrun
  --nproc-per-node="$NGPU"
  --nnodes="$NNODES"
  --role rank
  --tee 3
  --rdzv_backend c10d
  --rdzv_endpoint="$RDZV_ENDPOINT"
)

if [ -n "$LOG_RANKS" ]; then
  TORCHRUN_CMD+=(--local-ranks-filter "$LOG_RANKS")
fi

if [ -n "$SCRIPT" ]; then
  if [[ "$SCRIPT" != *.py ]]; then
    TORCHRUN_CMD+=(
      -m "$SCRIPT"
    )
  else
    TORCHRUN_CMD+=(
      "$SCRIPT"
    )
  fi
else
  TORCHRUN_CMD+=(
    -m "$DEFAULT_MODULE"
  )
fi

if [ -n "$CONFIG" ]; then
  TORCHRUN_CMD+=(
    --config "$CONFIG"
  )
fi

"${TORCHRUN_CMD[@]}"
