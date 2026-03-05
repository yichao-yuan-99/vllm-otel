#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT_PROFILE_ID="4"
PORT_PROFILE_ID="${DEFAULT_PORT_PROFILE_ID}"
MAX_CONCURRENT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-profile-id)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --port-profile-id" >&2
        exit 1
      fi
      PORT_PROFILE_ID="$2"
      shift 2
      ;;
    --port-profile-id=*)
      PORT_PROFILE_ID="${1#*=}"
      shift
      ;;
    --max-concurrent)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --max-concurrent" >&2
        exit 1
      fi
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    --max-concurrent=*)
      MAX_CONCURRENT="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
usage: bash experiments/single-agent/run_all.sh [--port-profile-id <id>] [--max-concurrent <n>]

Run all four record dataset experiments sequentially.

Default:
  --port-profile-id 4
  --max-concurrent from config (default is 1 in these configs)
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${MAX_CONCURRENT}" ]]; then
  if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrent must be a positive integer" >&2
    exit 1
  fi
fi

DATASETS=(
  "terminal-bench@2.0"
  "livecodebench"
  "dabstep"
  "swebench-verified"
)

for dataset in "${DATASETS[@]}"; do
  CMD=(
    bash "${SCRIPT_DIR}/run_dataset.sh"
    --dataset "${dataset}"
    --port-profile-id "${PORT_PROFILE_ID}"
  )
  if [[ -n "${MAX_CONCURRENT}" ]]; then
    CMD+=(--max-concurrent "${MAX_CONCURRENT}")
  fi
  "${CMD[@]}"
done
