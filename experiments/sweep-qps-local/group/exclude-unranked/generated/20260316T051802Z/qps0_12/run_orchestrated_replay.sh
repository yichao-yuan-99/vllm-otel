#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <assigned_vllm_port>" >&2
  exit 2
fi

ASSIGNED_VLLM_PORT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT_PROFILE_ID=0
PORT_PROFILE_ID_VALUE="${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[group-exclude-unranked-job] profile=${PORT_PROFILE_ID_VALUE} assigned_vllm_port=${ASSIGNED_VLLM_PORT}"

"${PYTHON_BIN}" -m replayer replay \
  --config "${SCRIPT_DIR}/replay.toml" \
  --port-profile-id "${PORT_PROFILE_ID_VALUE}"
