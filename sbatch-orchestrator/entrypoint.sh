#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR_PY="${SCRIPT_DIR}/orchestrator.py"
LOG_DIR="${SCRIPT_DIR}/logs"

JOB_LIST_FILE="${SBATCH_ORCHESTRATOR_JOB_LIST:-}"
PROFILE_LIST="${SBATCH_ORCHESTRATOR_PROFILE_IDS_CSV:-${GROUP_PROFILE_IDS_CSV:-}}"
PORT_LIST="${SBATCH_ORCHESTRATOR_PORTS_CSV:-${GROUP_VLLM_PORTS_CSV:-}}"
SUMMARY_PATH="${SBATCH_ORCHESTRATOR_SUMMARY_PATH:-}"

if [[ -z "${JOB_LIST_FILE}" ]]; then
  echo "SBATCH_ORCHESTRATOR_JOB_LIST is required for sbatch-orchestrator entrypoint." >&2
  exit 2
fi
if [[ -z "${PROFILE_LIST}" ]]; then
  echo "missing profile list: set SBATCH_ORCHESTRATOR_PROFILE_IDS_CSV or GROUP_PROFILE_IDS_CSV." >&2
  exit 2
fi
if [[ -z "${PORT_LIST}" ]]; then
  echo "missing port list: set SBATCH_ORCHESTRATOR_PORTS_CSV or GROUP_VLLM_PORTS_CSV." >&2
  exit 2
fi
if [[ ! -f "${ORCHESTRATOR_PY}" ]]; then
  echo "orchestrator script not found: ${ORCHESTRATOR_PY}" >&2
  exit 2
fi

if [[ -z "${SUMMARY_PATH}" ]]; then
  mkdir -p "${LOG_DIR}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SUMMARY_PATH="${LOG_DIR}/orchestrator-summary.${SLURM_JOB_ID}.json"
  else
    SUMMARY_PATH="${LOG_DIR}/orchestrator-summary.$(date -u +%Y%m%dT%H%M%SZ).json"
  fi
fi

echo "[sbatch-orchestrator] job_list=${JOB_LIST_FILE}" >&2
echo "[sbatch-orchestrator] profile_list=${PROFILE_LIST}" >&2
echo "[sbatch-orchestrator] port_list=${PORT_LIST}" >&2
echo "[sbatch-orchestrator] summary_path=${SUMMARY_PATH}" >&2

cmd=(
  python3
  "${ORCHESTRATOR_PY}"
  --job-list-file "${JOB_LIST_FILE}"
  --profile-list "${PROFILE_LIST}"
  --port-list "${PORT_LIST}"
  --summary-path "${SUMMARY_PATH}"
)

if [[ -n "${SBATCH_ORCHESTRATOR_POLL_INTERVAL_S:-}" ]]; then
  cmd+=(--poll-interval-s "${SBATCH_ORCHESTRATOR_POLL_INTERVAL_S}")
fi
if [[ "${SBATCH_ORCHESTRATOR_FAIL_FAST:-0}" == "1" ]]; then
  cmd+=(--fail-fast)
fi

exec "${cmd[@]}"
