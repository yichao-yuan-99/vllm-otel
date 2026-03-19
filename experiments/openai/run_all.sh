#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT_PROFILE_ID="0"
AGENT="mini-swe-agent"
MODEL="openai/gpt-4.1-mini"
MAX_CONCURRENT="1"
RESULTS_ROOT="experiments/results/openai"
GATEWAY_HOST="127.0.0.1"
GATEWAY_URL=""
DRY_RUN="false"

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
    --agent)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --agent" >&2
        exit 1
      fi
      AGENT="$2"
      shift 2
      ;;
    --agent=*)
      AGENT="${1#*=}"
      shift
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --model" >&2
        exit 1
      fi
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
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
    --results-root)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --results-root" >&2
        exit 1
      fi
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --results-root=*)
      RESULTS_ROOT="${1#*=}"
      shift
      ;;
    --gateway-host)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --gateway-host" >&2
        exit 1
      fi
      GATEWAY_HOST="$2"
      shift 2
      ;;
    --gateway-host=*)
      GATEWAY_HOST="${1#*=}"
      shift
      ;;
    --gateway-url)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --gateway-url" >&2
        exit 1
      fi
      GATEWAY_URL="$2"
      shift 2
      ;;
    --gateway-url=*)
      GATEWAY_URL="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
usage: bash experiments/openai/run_all.sh [options]

Run all supported datasets through gateway-lite -> OpenAI using con-driver.

Options:
  --port-profile-id <id>      Profile used to resolve gateway_parse_port (default: 0)
  --agent <name>              Harbor agent name (default: mini-swe-agent)
  --model <name>              Harbor model string (default: openai/gpt-4.1-mini)
  --max-concurrent <n>        Max concurrent trials per dataset (default: 1)
  --results-root <path>       Root results directory (default: experiments/results/openai)
  --gateway-host <host>       Host for computed gateway URL (default: 127.0.0.1)
  --gateway-url <url>         Explicit gateway URL override
  --dry-run                   Pass --dry-run to each dataset run
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrent must be a positive integer" >&2
  exit 1
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
    --agent "${AGENT}"
    --model "${MODEL}"
    --max-concurrent "${MAX_CONCURRENT}"
    --results-root "${RESULTS_ROOT}"
    --gateway-host "${GATEWAY_HOST}"
  )
  if [[ -n "${GATEWAY_URL}" ]]; then
    CMD+=(--gateway-url "${GATEWAY_URL}")
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    CMD+=(--dry-run)
  fi
  "${CMD[@]}"
done
