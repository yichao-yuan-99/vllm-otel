#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_RESULTS_ROOT="${REPO_ROOT}/experiments/results/sweep-concurrency/15"
TARGETS=(15 30 60 90 120)
AGENT_TIMEOUT_S="3000"
AGENT_TIMEOUT_MODE="on"

PORT_PROFILE_ID=""
SOURCE_JOB_DIR=""

find_latest_run_dir() {
  local root="$1"
  local latest=""
  local latest_mtime=""
  local candidate=""

  shopt -s nullglob
  for candidate in "${root}"/*; do
    [[ -d "${candidate}" ]] || continue
    local candidate_name
    candidate_name="$(basename "${candidate}")"
    [[ "${candidate_name}" == job-* ]] || continue
    [[ "${candidate_name}" == *.replayed-* ]] && continue
    [[ -f "${candidate}/meta/config.toml" ]] || continue
    local mtime
    mtime="$(stat -c '%Y' "${candidate}")"
    if [[ -z "${latest}" || "${mtime}" -gt "${latest_mtime}" ]]; then
      latest="${candidate}"
      latest_mtime="${mtime}"
    fi
  done
  shopt -u nullglob

  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf '%s\n' "${latest}"
}

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
    --source-job-dir)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --source-job-dir" >&2
        exit 1
      fi
      SOURCE_JOB_DIR="$2"
      shift 2
      ;;
    --source-job-dir=*)
      SOURCE_JOB_DIR="${1#*=}"
      shift
      ;;
    --agent-timeout)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --agent-timeout" >&2
        exit 1
      fi
      AGENT_TIMEOUT_MODE="$2"
      shift 2
      ;;
    --agent-timeout=*)
      AGENT_TIMEOUT_MODE="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
usage: bash experiments/sweep-concurrency/run_replay.sh --port-profile-id <id> [--source-job-dir <dir>] [--agent-timeout on|off]

Compile replay-plan.json from a source 15-concurrency job and replay the same
workload at 30, 60, 90, and 120 concurrency.

If --source-job-dir is omitted, the latest directory under
experiments/results/sweep-concurrency/15/ is used.

This script always recompiles replay-plan.json before replay.
Use --agent-timeout on to compile swebench-verified replay plans with
--agent-timeout-s 3000, or --agent-timeout off to omit the agent timeout.
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

case "${AGENT_TIMEOUT_MODE}" in
  on|off)
    ;;
  *)
    echo "invalid --agent-timeout value: ${AGENT_TIMEOUT_MODE} (expected on or off)" >&2
    exit 1
    ;;
esac

if [[ -z "${SOURCE_JOB_DIR}" ]]; then
  SOURCE_JOB_DIR="$(find_latest_run_dir "${SOURCE_RESULTS_ROOT}")"
fi

SOURCE_JOB_DIR="$(cd -- "${SOURCE_JOB_DIR}" && pwd)"
if [[ ! -d "${SOURCE_JOB_DIR}" ]]; then
  echo "source job dir not found: ${SOURCE_JOB_DIR}" >&2
  exit 1
fi

PLAN_PATH="${SOURCE_JOB_DIR}/replay-plan.json"

echo "=== compile replay plan ==="
compile_cmd=(
  python3 -m replayer compile
  --job-dir "${SOURCE_JOB_DIR}"
  --port-profile-id "${PORT_PROFILE_ID}"
  --plan-out "${PLAN_PATH}"
)
if [[ "${AGENT_TIMEOUT_MODE}" == "on" ]]; then
  compile_cmd+=(--agent-timeout-s "${AGENT_TIMEOUT_S}")
fi
"${compile_cmd[@]}"

for target in "${TARGETS[@]}"; do
  target_root="${REPO_ROOT}/experiments/results/sweep-concurrency/${target}"
  mkdir -p "${target_root}"

  target_output_dir="${target_root}/$(basename "${SOURCE_JOB_DIR}").replayed-c${target}"
  override_json="$(printf '{"max_concurrent": %s, "seed": null, "pattern": {"name": "eager"}, "pattern_args": {}}' "${target}")"

  echo "=== replay target concurrency ${target} ==="
  python3 -m replayer replay \
    --plan "${PLAN_PATH}" \
    --port-profile-id "${PORT_PROFILE_ID}" \
    --output-dir "${target_output_dir}" \
    --launch-policy-override-json "${override_json}"
done
