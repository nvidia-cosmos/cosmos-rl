#!/usr/bin/env bash

PORT=""
CONFIG_FILE=""
LOG_FILE=""
SCRIPT="cosmos_rl.dispatcher.run_web_panel"

show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --port <port>       Specify the port number (if not set, automatically chosen in runtime)"
  echo "  --config <file>     Specify the configuration file"
  echo "  --log <file>        Specify the redis log file"
  echo "  --help              Show this help message and exit"
  echo "  <script>            Specify the script to run"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      SCRIPT="$1"
      echo "Using script: $SCRIPT"
      shift
      ;;
  esac
done

if [[ "$SCRIPT" != *.py ]]; then
  CMD="python -m $SCRIPT"
else
  CMD="python $SCRIPT"
fi

if [[ -n "$PORT" ]]; then
  CMD+=" --port $PORT"
fi

if [[ -n "$CONFIG_FILE" ]]; then
  CMD+=" --config $CONFIG_FILE"
fi

if [[ -n "$LOG_FILE" ]]; then
  CMD+=" --redis-logfile-path $LOG_FILE"
fi

echo "${CMD}"

export COSMOS_ROLE="Controller"
$CMD
