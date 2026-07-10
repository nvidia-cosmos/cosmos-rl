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
NSYS="0"
NSYS_EXPLICIT="0"
NSYS_OUTPUT_DIR=""
NSYS_OUTPUT_PREFIX="nsys"
NSYS_TRACE="cuda,nvtx,osrt,cudnn,cublas"
NSYS_CAPTURE_RANGE="cudaProfilerApi"
NSYS_CAPTURE_RANGE_END="stop"
NSYS_EXTRA_ARGS=()

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
  echo "  --nsys                                Wrap this replica with Nsight Systems regardless of TOML role filters."
  echo "  --nsys-output-dir <path>              Directory to save Nsight Systems reports."
  echo "  --nsys-output-prefix <prefix>         Report file prefix. Default: nsys"
  echo "  --nsys-trace <domains>                Domains for nsys --trace. Default: cuda,nvtx,osrt,cudnn,cublas"
  echo "  --nsys-extra-arg <arg>                Additional argument appended to nsys profile. Can be repeated."
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

load_nsys_config() {
  if [ -z "$CONFIG" ]; then
    return
  fi

  while IFS= read -r line; do
    eval "$line"
  done < <(python - "$CONFIG" "$TYPE" <<'PY'
import os
import shlex
import sys

import toml

config_path = sys.argv[1]
role = sys.argv[2]

try:
    config = toml.load(config_path)
except Exception:
    sys.exit(0)

profiler = config.get("profiler", {})
train = config.get("train", {})

target_roles = profiler.get("nsys_target_roles", ["policy"])
if isinstance(target_roles, str):
    target_roles = [target_roles]

enable = bool(profiler.get("enable_nsys", False)) and role in target_roles

train_output_dir = train.get("output_dir", "./outputs")
default_output_dir = os.path.join(os.path.dirname(train_output_dir), "nsys_profile")
output_dir = profiler.get("nsys_output_dir") or default_output_dir

extra_args = profiler.get("nsys_extra_args", [])
if isinstance(extra_args, str):
    extra_args = [extra_args]

values = {
    "NSYS_CONFIG_ENABLE": "1" if enable else "0",
    "NSYS_CONFIG_OUTPUT_DIR": output_dir,
    "NSYS_CONFIG_OUTPUT_PREFIX": profiler.get("nsys_output_prefix", "nsys"),
    "NSYS_CONFIG_TRACE": profiler.get("nsys_trace", "cuda,nvtx,osrt,cudnn,cublas"),
    "NSYS_CONFIG_CAPTURE_RANGE": profiler.get("nsys_capture_range", "cudaProfilerApi"),
    "NSYS_CONFIG_CAPTURE_RANGE_END": profiler.get("nsys_capture_range_end", "stop"),
    "NSYS_CONFIG_EXTRA_ARGS": shlex.join([str(arg) for arg in extra_args]),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
  )

  if [ "$NSYS_EXPLICIT" != "1" ]; then
    NSYS="${NSYS_CONFIG_ENABLE:-0}"
  fi
  if [ -z "$NSYS_OUTPUT_DIR" ]; then
    NSYS_OUTPUT_DIR="${NSYS_CONFIG_OUTPUT_DIR:-}"
  fi
  if [ "$NSYS_OUTPUT_PREFIX" == "nsys" ]; then
    NSYS_OUTPUT_PREFIX="${NSYS_CONFIG_OUTPUT_PREFIX:-$NSYS_OUTPUT_PREFIX}"
  fi
  if [ "$NSYS_TRACE" == "cuda,nvtx,osrt,cudnn,cublas" ]; then
    NSYS_TRACE="${NSYS_CONFIG_TRACE:-$NSYS_TRACE}"
  fi
  if [ "$NSYS_CAPTURE_RANGE" == "cudaProfilerApi" ]; then
    NSYS_CAPTURE_RANGE="${NSYS_CONFIG_CAPTURE_RANGE:-$NSYS_CAPTURE_RANGE}"
  fi
  if [ "$NSYS_CAPTURE_RANGE_END" == "stop" ]; then
    NSYS_CAPTURE_RANGE_END="${NSYS_CONFIG_CAPTURE_RANGE_END:-$NSYS_CAPTURE_RANGE_END}"
  fi
  if [ -n "${NSYS_CONFIG_EXTRA_ARGS:-}" ]; then
    eval "NSYS_EXTRA_ARGS+=( ${NSYS_CONFIG_EXTRA_ARGS} )"
  fi
}

apply_nsys_wrapper() {
  if [ "$NSYS" != "1" ]; then
    return
  fi
  if ! command -v nsys >/dev/null 2>&1; then
    echo "Error: [profiler].enable_nsys is true or --nsys was passed, but 'nsys' is not available in PATH."
    exit 1
  fi

  if [ -z "$NSYS_OUTPUT_DIR" ]; then
    NSYS_OUTPUT_DIR="./outputs/nsys_profile"
  fi
  mkdir -p "$NSYS_OUTPUT_DIR"

  local output_pattern="${NSYS_OUTPUT_DIR}/${NSYS_OUTPUT_PREFIX}_${TYPE}_%h_%p"
  local nsys_cmd=(
    nsys profile
    --force-overwrite=true
    --capture-range="$NSYS_CAPTURE_RANGE"
    --trace="$NSYS_TRACE"
    --output="$output_pattern"
  )
  if [ "$NSYS_CAPTURE_RANGE" != "none" ]; then
    nsys_cmd+=(--capture-range-end="$NSYS_CAPTURE_RANGE_END")
  fi

  echo "[Cosmos-RL] ${TYPE^^} Nsight Systems profiling enabled. Reports: ${output_pattern}.nsys-rep"
  LAUNCH_CMD=("${nsys_cmd[@]}" "${NSYS_EXTRA_ARGS[@]}" "${LAUNCH_CMD[@]}")
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
  --nsys)
    NSYS="1"
    NSYS_EXPLICIT="1"
    shift
    ;;
  --nsys-output-dir)
    NSYS_OUTPUT_DIR="$2"
    shift 2
    ;;
  --nsys-output-prefix)
    NSYS_OUTPUT_PREFIX="$2"
    shift 2
    ;;
  --nsys-trace)
    NSYS_TRACE="$2"
    shift 2
    ;;
  --nsys-capture-range)
    NSYS_CAPTURE_RANGE="$2"
    shift 2
    ;;
  --nsys-capture-range-end)
    NSYS_CAPTURE_RANGE_END="$2"
    shift 2
    ;;
  --nsys-extra-arg)
    NSYS_EXTRA_ARGS+=("$2")
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

# NCCL related, to avoid potential unstable issues such as https://github.com/NVIDIA/nccl/issues/1234
# Only set if not set by user to avoid overwriting user specified envs. `NCCL_CUMEM_ENABLE` -> 0
if [ -z "$NCCL_CUMEM_ENABLE" ]; then
  set_env "NCCL_CUMEM_ENABLE" "0"
fi

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
elif [ "$TYPE" == "reference" ]; then
  DEFAULT_MODULE="cosmos_rl.reference.reference_entry"
  export COSMOS_ROLE="Reference"
  # Set a longer timeout for reference to avoid timeout when waiting for teacher requests
  # when reference is not used such as in validation mode.
  export COSMOS_GLOO_TIMEOUT="6000"
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
  # Disable TP_EP_INTERCHANGABLE_WITH_DP_FUSED for rollout to avoid potential unstable issues
  export TP_EP_INTERCHANGABLE_WITH_DP_FUSED=0
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

load_nsys_config
apply_nsys_wrapper

echo "Launching command: ${LAUNCH_CMD[@]}"

"${LAUNCH_CMD[@]}"
