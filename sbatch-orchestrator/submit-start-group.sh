#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit-start-group.sh --job-list path/to/jobs.txt [--sbatch-script path/to/start-group.sbatch.sh] [--half] [-- sbatch args...]

Options:
  --job-list, -j       Job-list file path (relative or absolute).
  --sbatch-script, -s  Grouped sbatch script path (relative or absolute; default: ./sbatch-orchestrator/start-group.sbatch.sh).
  --half               Keep only even grouped profile IDs (and aligned port/gateway entries) in submitted sbatch copy.
  --help, -h           Show this help.

Notes:
  - Creates a timestamped run directory under ./sbatch-orchestrator/logs/<utc-timestamp>/
  - Copies and patches the sbatch script so Slurm/stdout/stderr and JOB_LOG_DIR point to that run directory.
  - Writes summary JSON to <run-dir>/orchestrator-summary.json.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SBATCH_SCRIPT="${SCRIPT_DIR}/start-group.sbatch.sh"
LOG_ROOT_DIR="${SCRIPT_DIR}/logs"

job_list_arg="${SBATCH_ORCHESTRATOR_JOB_LIST:-}"
sbatch_script_arg="${DEFAULT_SBATCH_SCRIPT}"
half_mode=0
sbatch_extra_args=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --job-list|-j)
      if [[ "$#" -lt 2 ]]; then
        echo "missing value for $1" >&2
        exit 2
      fi
      job_list_arg="$2"
      shift 2
      ;;
    --sbatch-script|-s)
      if [[ "$#" -lt 2 ]]; then
        echo "missing value for $1" >&2
        exit 2
      fi
      sbatch_script_arg="$2"
      shift 2
      ;;
    --half)
      half_mode=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      sbatch_extra_args=("$@")
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${job_list_arg}" ]]; then
  echo "job list is required (use --job-list or set SBATCH_ORCHESTRATOR_JOB_LIST)." >&2
  exit 2
fi

job_list_path="$(python3 - "$job_list_arg" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
sbatch_script_path="$(python3 - "$sbatch_script_arg" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

if [[ ! -f "${job_list_path}" ]]; then
  echo "job list file not found: ${job_list_path}" >&2
  exit 2
fi
if [[ ! -f "${sbatch_script_path}" ]]; then
  echo "sbatch script not found: ${sbatch_script_path}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${LOG_ROOT_DIR}/${timestamp}"
mkdir -p "${run_dir}"

submitted_sbatch_path="${run_dir}/start-group.sbatch.sh"
summary_path="${run_dir}/orchestrator-summary.json"
job_list_copy_path="${run_dir}/job-list.txt"
job_list_path_file="${run_dir}/job-list.path.txt"
submission_info_path="${run_dir}/submission-info.json"
sbatch_stdout_path="${run_dir}/sbatch.submit.out"
slurm_output_pattern="${run_dir}/slurm.%j.out"
slurm_error_pattern="${run_dir}/slurm.%j.err"

cp "${job_list_path}" "${job_list_copy_path}"
printf '%s\n' "${job_list_path}" > "${job_list_path_file}"
cp "${sbatch_script_path}" "${submitted_sbatch_path}"
chmod 750 "${submitted_sbatch_path}"

python3 - "${submitted_sbatch_path}" "${run_dir}" "${slurm_output_pattern}" "${slurm_error_pattern}" "${half_mode}" <<'PY'
import json
import re
import sys
from pathlib import Path

sbatch_path = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
slurm_output_pattern = sys.argv[3]
slurm_error_pattern = sys.argv[4]
half_mode = sys.argv[5] == "1"

content = sbatch_path.read_text(encoding="utf-8")
replacements = [
    (
        re.compile(r"^#SBATCH --output=.*$", flags=re.MULTILINE),
        f"#SBATCH --output={slurm_output_pattern}",
        "SBATCH --output",
    ),
    (
        re.compile(r"^#SBATCH --error=.*$", flags=re.MULTILINE),
        f"#SBATCH --error={slurm_error_pattern}",
        "SBATCH --error",
    ),
    (
        re.compile(r"^JOB_LOG_DIR=.*$", flags=re.MULTILINE),
        f"JOB_LOG_DIR={json.dumps(str(run_dir), ensure_ascii=True)}",
        "JOB_LOG_DIR",
    ),
]

updated = content
for pattern, replacement, label in replacements:
    updated, count = pattern.subn(replacement, updated, count=1)
    if count != 1:
        raise SystemExit(f"failed to patch {label} in copied sbatch: {sbatch_path}")

if half_mode:
    def parse_assignment(name: str, separator: str) -> tuple[list[str], str]:
        pattern = re.compile(rf"^{re.escape(name)}=(.*)$", flags=re.MULTILINE)
        match = pattern.search(updated)
        if match is None:
            raise SystemExit(f"failed to find {name} in copied sbatch: {sbatch_path}")
        rhs = match.group(1).strip()
        quote = ""
        if len(rhs) >= 2 and rhs[0] == rhs[-1] and rhs[0] in {"'", '"'}:
            quote = rhs[0]
            rhs = rhs[1:-1]
        values = [item.strip() for item in rhs.split(separator)]
        if not values or any(item == "" for item in values):
            raise SystemExit(f"invalid {name} list in copied sbatch: {sbatch_path}")
        return values, quote

    def update_assignment(
        content: str,
        name: str,
        values: list[str],
        separator: str,
        quote: str,
    ) -> str:
        joined = separator.join(values)
        if quote:
            joined = f"{quote}{joined}{quote}"
        pattern = re.compile(rf"^{re.escape(name)}=.*$", flags=re.MULTILINE)
        replacement = f"{name}={joined}"
        new_updated, count = pattern.subn(replacement, content, count=1)
        if count != 1:
            raise SystemExit(f"failed to update {name} in copied sbatch: {sbatch_path}")
        return new_updated

    profile_values, profile_quote = parse_assignment("GROUP_PROFILE_IDS_CSV", ",")
    selected_indices: list[int] = []
    for index, profile_value in enumerate(profile_values):
        try:
            profile_id = int(profile_value)
        except ValueError as exc:
            raise SystemExit(
                f"GROUP_PROFILE_IDS_CSV contains non-integer value {profile_value!r}"
            ) from exc
        if profile_id % 2 == 0:
            selected_indices.append(index)
    if not selected_indices:
        raise SystemExit("half mode produced empty profile selection")

    def select_even(values: list[str], name: str) -> list[str]:
        if len(values) != len(profile_values):
            raise SystemExit(
                f"{name} length {len(values)} does not match GROUP_PROFILE_IDS_CSV length {len(profile_values)}"
            )
        return [values[index] for index in selected_indices]

    csv_names = [
        "GROUP_PROFILE_IDS_CSV",
        "GROUP_VLLM_PORTS_CSV",
        "GROUP_GATEWAY_PORTS_CSV",
        "GROUP_GATEWAY_PARSE_PORTS_CSV",
        "GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV",
        "GROUP_JAEGER_UI_LOGIN_PORTS_CSV",
        "GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV",
    ]
    for name in csv_names:
        values, quote = parse_assignment(name, ",")
        updated = update_assignment(updated, name, select_even(values, name), ",", quote)

    visible_values, visible_quote = parse_assignment("GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON", ";")
    updated = update_assignment(
        updated,
        "GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON",
        select_even(visible_values, "GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON"),
        ";",
        visible_quote,
    )

    new_group_size = len(selected_indices)
    scalar_replacements = [
        (
            re.compile(r"^#SBATCH --ntasks=\d+\s*$", flags=re.MULTILINE),
            f"#SBATCH --ntasks={new_group_size}",
            "SBATCH --ntasks",
        ),
        (
            re.compile(r"^#SBATCH --ntasks-per-node=\d+\s*$", flags=re.MULTILINE),
            f"#SBATCH --ntasks-per-node={new_group_size}",
            "SBATCH --ntasks-per-node",
        ),
        (
            re.compile(r"^GROUP_SIZE=\d+\s*$", flags=re.MULTILINE),
            f"GROUP_SIZE={new_group_size}",
            "GROUP_SIZE",
        ),
    ]
    for pattern, replacement, label in scalar_replacements:
        updated, count = pattern.subn(replacement, updated, count=1)
        if count != 1:
            raise SystemExit(f"failed to patch {label} for half mode in copied sbatch: {sbatch_path}")

sbatch_path.write_text(updated, encoding="utf-8")
PY

python3 - "${submission_info_path}" "${job_list_path}" "${job_list_copy_path}" "${job_list_path_file}" "${sbatch_script_path}" "${submitted_sbatch_path}" "${summary_path}" "${run_dir}" "${timestamp}" "${half_mode}" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "timestamp": sys.argv[9],
    "run_dir": str(Path(sys.argv[8]).resolve()),
    "job_list_path": str(Path(sys.argv[2]).resolve()),
    "job_list_copy": str(Path(sys.argv[3]).resolve()),
    "job_list_path_file": str(Path(sys.argv[4]).resolve()),
    "source_sbatch_script": str(Path(sys.argv[5]).resolve()),
    "submitted_sbatch_script": str(Path(sys.argv[6]).resolve()),
    "summary_path": str(Path(sys.argv[7]).resolve()),
    "half_mode": sys.argv[10] == "1",
}

Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[sbatch-orchestrator] run_dir=${run_dir}" >&2
echo "[sbatch-orchestrator] submitted_sbatch=${submitted_sbatch_path}" >&2
echo "[sbatch-orchestrator] job_list_source=${job_list_path}" >&2
echo "[sbatch-orchestrator] job_list_submitted=${job_list_copy_path}" >&2
echo "[sbatch-orchestrator] summary_path=${summary_path}" >&2

set +e
sbatch_output="$(
  SBATCH_ORCHESTRATOR_JOB_LIST="${job_list_copy_path}" \
  SBATCH_ORCHESTRATOR_SUMMARY_PATH="${summary_path}" \
  sbatch "${sbatch_extra_args[@]}" "${submitted_sbatch_path}" 2>&1
)"
sbatch_exit_code=$?
set -e

printf '%s\n' "${sbatch_output}" | tee "${sbatch_stdout_path}"

job_id="$(printf '%s\n' "${sbatch_output}" | awk '/Submitted batch job/{print $4}' | tail -n1)"
if [[ -n "${job_id}" ]]; then
  printf '%s\n' "${job_id}" > "${run_dir}/slurm_job_id.txt"
fi

if [[ "${sbatch_exit_code}" -ne 0 ]]; then
  echo "sbatch submission failed; see ${sbatch_stdout_path}" >&2
  exit "${sbatch_exit_code}"
fi

if [[ -n "${job_id}" ]]; then
  echo "[sbatch-orchestrator] slurm_job_id=${job_id}" >&2
fi
echo "[sbatch-orchestrator] submission captured in ${run_dir}" >&2
