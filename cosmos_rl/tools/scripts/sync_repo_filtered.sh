#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sync_repo_filtered.sh --src <source_dir> --dst <destination_dir>

Description:
  Syncs repo files from source_dir to destination_dir while omitting:
    - reward_service/
    - any directory named wfm/
    - any Python file matching wfm_*.py

Examples:
  sync_repo_filtered.sh --src /home/liangf/workspace/cosmos-rl --dst /data/backup/cosmos-rl
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SRC_DIR=""
DEST_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      [[ $# -ge 2 ]] || { echo "Missing value for --src" >&2; usage; exit 1; }
      SRC_DIR="$2"
      shift 2
      ;;
    --dst)
      [[ $# -ge 2 ]] || { echo "Missing value for --dst" >&2; usage; exit 1; }
      DEST_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SRC_DIR" || -z "$DEST_DIR" ]]; then
  echo "Both --src and --dst are required." >&2
  usage
  exit 1
fi

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory does not exist: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

rsync -a \
  --exclude='.git/' \
  --exclude='reward_service/***' \
  --exclude='wfm/***' \
  --exclude='**/wfm/***' \
  --exclude='wfm_*.py' \
  --exclude='**/wfm_*.py' \
  "$SRC_DIR"/ "$DEST_DIR"/

echo "Sync complete: $SRC_DIR -> $DEST_DIR"
