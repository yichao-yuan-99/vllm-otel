#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_PORT_PROFILE_ID="4"

PORT_PROFILE_ID="${DEFAULT_PORT_PROFILE_ID}"
DATASET=""

list_datasets() {
  cat <<'EOF'
terminal-bench@2.0
livecodebench
dabstep
swebench-verified
EOF
}

config_path_for_dataset() {
  case "$1" in
    terminal-bench@2.0)
      printf '%s\n' "${SCRIPT_DIR}/configs/config.terminal-bench-2.0.toml"
      ;;
    livecodebench)
      printf '%s\n' "${SCRIPT_DIR}/configs/config.livecodebench.toml"
      ;;
    dabstep)
      printf '%s\n' "${SCRIPT_DIR}/configs/config.dabstep.toml"
      ;;
    swebench-verified)
      printf '%s\n' "${SCRIPT_DIR}/configs/config.swebench-verified.toml"
      ;;
    *)
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --dataset" >&2
        exit 1
      fi
      DATASET="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
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
    --list-datasets)
      list_datasets
      exit 0
      ;;
    -h|--help)
      cat <<'EOF'
usage: bash experiments/single-agent/run_dataset.sh --dataset <name> [--port-profile-id <id>]

Run one single-agent Harbor dataset experiment.

Datasets:
  terminal-bench@2.0
  livecodebench
  dabstep
  swebench-verified

Defaults:
  --port-profile-id 4
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET}" ]]; then
  echo "missing required --dataset" >&2
  exit 1
fi

if ! CONFIG_PATH="$(config_path_for_dataset "${DATASET}")"; then
  echo "unknown dataset: ${DATASET}" >&2
  echo "supported datasets:" >&2
  list_datasets >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/experiments/results/single-agent"

echo "=== single-agent dataset: ${DATASET} ==="
echo "config: ${CONFIG_PATH}"
echo "port_profile_id: ${PORT_PROFILE_ID}"

bash "${REPO_ROOT}/con-driver/run_con_driver.sh" \
  --config "${CONFIG_PATH}" \
  --port-profile-id "${PORT_PROFILE_ID}"
