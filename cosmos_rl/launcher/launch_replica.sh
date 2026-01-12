#!/usr/bin/env bash

# Default values
NGPU=2
NNODES=1
LOG_RANKS=""
TYPE=""
RDZV_ENDPOINT="localhost:0"
SCRIPT=""
SCRIPT_ARGS=()
CONFIG=""
BACKEND="vllm"
WFM_MODE="False"

print_help() {
  echo ""
  echo "Usage: ./launch_replica.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --type <rollout|policy|reference>     Required. Type of replica to launch."
  echo "  --nnodes <int>                        Number of nodes to launch. Default: 1"
  echo "  --ngpus <int>                         Number of GPUs per node. Default: 2"
  echo "  --log-rank <comma-separated ints>     Comma-separated list of ranks to enable logging. Default: Empty for all ranks."
  echo "  --rdzv-endpoint <host:port>           Rendezvous endpoint for distributed training. Default: localhost:0"
  echo "  --script <script>                     The user script to run before launch."
  echo "  --config <path>                       The path to the config file."
  echo "  --backend <vllm|vllm_async|trtllm>    The backend to use for the job. Default: vllm"
  echo "  --wfm-mode <True|False>               Whether to launch in wfm mode. Default: False"
  echo "  --help                                Show this help message"
  echo "Examples:"
  echo "  ./launch_replica.sh --type rollout --ngpus 4 --log-rank 0,1"
  echo "  ./launch_replica.sh --type policy --ngpus 8 --log-rank 0"
  echo ""
}

set_env() {
  local env_name="$1"
  local env_value="$2"
  local upper_type="${TYPE^^}"
  echo "[Cosmos-RL] $upper_type Pre-setting environment variable $env_name=$env_value"
  export "$env_name=$env_value"
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
  --backend)
    BACKEND="$2"
    shift 2
    ;;
  --wfm-mode)
    WFM_MODE="$2"
    shift 2
    ;;
  --help)
    print_help
    exit 0
    ;;
  *)
    SCRIPT_ARGS+=("$1")
    shift
    ;;
  esac
done

if [ -z "$TYPE" ]; then
  echo "Error: --type is required"
  print_help
  exit 1
fi

# NCCL related
set_env "NCCL_CUMEM_ENABLE" "1"

if [ "$BACKEND" == "trtllm" ]; then
  # BACKEND won't have affect on policy.
  # But we still need user speicify rollout backend when launch policy.
  # to set this variable.
  set_env "NCCL_RUNTIME_CONNECT" "0"
fi

# Torch related
set_env "TORCH_CPP_LOG_LEVEL" "ERROR"

if [ "$WFM_MODE" == "True" ]; then
  set_env "COSMOS_IS_WFM" "True"
fi

LAUNCH_BINARY="torchrun"

if [ "$TYPE" == "rollout" ]; then
  DEFAULT_MODULE="cosmos_rl.rollout.rollout_entry"
  export COSMOS_ROLE="Rollout"
  if [ "$BACKEND" == "trtllm" ]; then
    LAUNCH_BINARY="mpirun"
  fi
elif [ "$TYPE" == "policy" ]; then
  DEFAULT_MODULE="cosmos_rl.policy.train"
  export COSMOS_ROLE="Policy"
  set_env "PYTORCH_CUDA_ALLOC_CONF" "expandable_segments:True"
elif [ "$TYPE" == "reference" ]; then
  DEFAULT_MODULE="cosmos_rl.reference.reference_entry"
  export COSMOS_ROLE="Reference"
  # Set a longer timeout for reference to avoid timeout when waiting for teacher requests
  # when reference is not used such as in validation mode.
  export COSMOS_GLOO_TIMEOUT="6000"
  set_env "PYTORCH_CUDA_ALLOC_CONF" "expandable_segments:True"
else
  echo "Error: Invalid --type value '$TYPE'. Must be 'rollout' or 'policy' or 'reference'."
  print_help
  exit 1
fi

if [ -z "$COSMOS_CONTROLLER_HOST" ]; then
  echo "Error: COSMOS_CONTROLLER_HOST is not set. Please pass it in like:"
  echo "  COSMOS_CONTROLLER_HOST=<controller_host>:<controller_port> ./launch_replica.sh"
  exit 1
fi

LAUNCH_CMD=("$LAUNCH_BINARY")

if [ "$TYPE" == "policy" ]; then
  LAUNCH_CMD+=(
    --nproc-per-node="$NGPU"
    --nnodes="$NNODES"
    --role rank
    --tee 3
    --rdzv_backend c10d
    --rdzv_endpoint="$RDZV_ENDPOINT"
  )

  if [ -n "$LOG_RANKS" ]; then
    LAUNCH_CMD+=(--local-ranks-filter "$LOG_RANKS")
  fi
elif [ "$TYPE" == "rollout" ]; then
  if [ "$BACKEND" == "trtllm" ]; then
    COSMOS_WORLD_SIZE=$((NNODES * NGPU))
    export COSMOS_WORLD_SIZE
    COSMOS_LOCAL_WORLD_SIZE=$((NGPU))
    export COSMOS_LOCAL_WORLD_SIZE
    export COSMOS_RDZV_ENDPOINT="$RDZV_ENDPOINT"

    # Set np to 1 just for trtllm to get OMP_* entvironments.
    LAUNCH_CMD+=(
      -np 1
      --allow-run-as-root
      --oversubscribe
      python
    )

    echo "Launching trtllm as the backend, ignoring:
            --log-rank flags."
  else
    LAUNCH_CMD+=(
      --nproc-per-node="$NGPU"
      --nnodes="$NNODES"
      --role rank
      --tee 3
      --rdzv_backend c10d
      --rdzv_endpoint="$RDZV_ENDPOINT"
    )

    if [ -n "$LOG_RANKS" ]; then
      LAUNCH_CMD+=(--local-ranks-filter "$LOG_RANKS")
    fi
  fi
elif [ "$TYPE" == "reference" ]; then
  LAUNCH_CMD+=(
    --nproc-per-node="$NGPU"
    --nnodes="$NNODES"
    --role rank
    --tee 3
    --rdzv_backend c10d
    --rdzv_endpoint="$RDZV_ENDPOINT"
  )

  if [ -n "$LOG_RANKS" ]; then
    LAUNCH_CMD+=(--local-ranks-filter "$LOG_RANKS")
  fi
fi


if [ -n "$SCRIPT" ]; then
  if [[ "$SCRIPT" != *.py ]]; then
    LAUNCH_CMD+=(
      -m "$SCRIPT"
    )
    LAUNCH_CMD+=(
      "${SCRIPT_ARGS[@]}"
    )
  else
    LAUNCH_CMD+=(
      "$SCRIPT"
    )
    LAUNCH_CMD+=(
      "${SCRIPT_ARGS[@]}"
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

echo "Launching command: ${LAUNCH_CMD[@]}"

"${LAUNCH_CMD[@]}"
