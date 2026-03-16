#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  render-and-submit-start-group.sh --job-list path/to/jobs.txt [--half] [render-start-group args...] [-- sbatch args...]

Examples:
  bash sbatch-orchestrator/render-and-submit-start-group.sh \
    --job-list path/to/jobs.txt \
    -g bench_orch -L 0,1,2,3,4,5,6,7 -p mi3008x -m qwen3_coder_30b

  bash sbatch-orchestrator/render-and-submit-start-group.sh \
    --job-list path/to/jobs.txt --half \
    -g bench_orch -L 0,1,2,3,4,5,6,7 -p mi3008x -m qwen3_coder_30b

  bash sbatch-orchestrator/render-and-submit-start-group.sh \
    --job-list path/to/jobs.txt \
    -g bench_orch -L 0,1,2,3,4,5,6,7 -p mi3008x -m qwen3_coder_30b --lmcache 100 \
    -- --qos debug

Notes:
  - `--half` is handled by submit-start-group.sh and keeps only even grouped profile IDs.
  - Arguments before `--` are forwarded to render-start-group.sh.
  - Arguments after `--` are forwarded to sbatch via submit-start-group.sh.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_WRAPPER="${SCRIPT_DIR}/render-start-group.sh"
SUBMIT_WRAPPER="${SCRIPT_DIR}/submit-start-group.sh"
FIXED_SBATCH_PATH="${SCRIPT_DIR}/start-group.sbatch.sh"

if [[ ! -x "${RENDER_WRAPPER}" ]]; then
  echo "render wrapper missing or not executable: ${RENDER_WRAPPER}" >&2
  exit 2
fi
if [[ ! -x "${SUBMIT_WRAPPER}" ]]; then
  echo "submit wrapper missing or not executable: ${SUBMIT_WRAPPER}" >&2
  exit 2
fi

job_list_arg="${SBATCH_ORCHESTRATOR_JOB_LIST:-}"
half_mode=0
render_args=()
sbatch_args=()
parsing_submit_args=0

while [[ "$#" -gt 0 ]]; do
  if [[ "${parsing_submit_args}" -eq 1 ]]; then
    sbatch_args+=("$1")
    shift
    continue
  fi

  case "$1" in
    --job-list|-j)
      if [[ "$#" -lt 2 ]]; then
        echo "missing value for $1" >&2
        exit 2
      fi
      job_list_arg="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --half)
      half_mode=1
      shift
      ;;
    --)
      parsing_submit_args=1
      shift
      ;;
    *)
      render_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${job_list_arg}" ]]; then
  echo "job list is required (use --job-list or set SBATCH_ORCHESTRATOR_JOB_LIST)." >&2
  exit 2
fi

echo "[sbatch-orchestrator] rendering grouped sbatch..."
"${RENDER_WRAPPER}" "${render_args[@]}"

echo "[sbatch-orchestrator] submitting grouped sbatch..."
submit_args=(
  --job-list "${job_list_arg}"
  --sbatch-script "${FIXED_SBATCH_PATH}"
)
if [[ "${half_mode}" -eq 1 ]]; then
  submit_args+=(--half)
fi
if [[ "${#sbatch_args[@]}" -gt 0 ]]; then
  "${SUBMIT_WRAPPER}" "${submit_args[@]}" -- "${sbatch_args[@]}"
else
  "${SUBMIT_WRAPPER}" "${submit_args[@]}"
fi
