#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_CONFIG="${SCRIPT_DIR}/configs/config.15.toml"
SOURCE_RESULTS_ROOT="${REPO_ROOT}/experiments/results/sweep-concurrency/15"

PORT_PROFILE_ID=""

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
    -h|--help)
      cat <<'EOF'
usage: bash experiments/sweep-concurrency/run_source.sh --port-profile-id <id>

Run the source con-driver workload at concurrency 15 and write it under:
experiments/results/sweep-concurrency/15/
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PORT_PROFILE_ID}" ]]; then
  echo "missing required --port-profile-id" >&2
  exit 1
fi

mkdir -p "${SOURCE_RESULTS_ROOT}"

echo "=== source run: concurrency 15 ==="
bash "${REPO_ROOT}/con-driver/run_con_driver.sh" \
  --config "${SOURCE_CONFIG}" \
  --port-profile-id "${PORT_PROFILE_ID}"
