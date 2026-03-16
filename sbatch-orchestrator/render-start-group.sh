#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RENDER_SCRIPT="${REPO_ROOT}/servers/servers-amdhpc/render-sbatch.py"
DEFAULT_LOCAL_MODE_SCRIPT="${SCRIPT_DIR}/entrypoint.sh"
FIXED_SBATCH_PATH="${SCRIPT_DIR}/start-group.sbatch.sh"

if [[ ! -f "${RENDER_SCRIPT}" ]]; then
  echo "render-sbatch script not found: ${RENDER_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${DEFAULT_LOCAL_MODE_SCRIPT}" ]]; then
  echo "default local-mode entrypoint not found: ${DEFAULT_LOCAL_MODE_SCRIPT}" >&2
  exit 2
fi

has_local_mode=0
for arg in "$@"; do
  case "${arg}" in
    --local-mode|--local-mode=*)
      has_local_mode=1
      break
      ;;
  esac
done

cmd=(
  python3
  "${RENDER_SCRIPT}"
  start-group
  "$@"
)
if [[ "${has_local_mode}" -eq 0 ]]; then
  cmd+=(--local-mode "${DEFAULT_LOCAL_MODE_SCRIPT}")
fi

render_output="$("${cmd[@]}")"

source_sbatch_path="$(
  RENDER_OUTPUT="${render_output}" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["RENDER_OUTPUT"])
data = payload.get("data")
if not isinstance(data, dict):
    raise SystemExit("render output missing data object")
sbatch_script = data.get("sbatch_script")
if not isinstance(sbatch_script, str) or not sbatch_script.strip():
    raise SystemExit("render output missing data.sbatch_script")
print(sbatch_script)
PY
)"

if [[ ! -f "${source_sbatch_path}" ]]; then
  echo "rendered sbatch file not found: ${source_sbatch_path}" >&2
  exit 2
fi

cp "${source_sbatch_path}" "${FIXED_SBATCH_PATH}"
chmod 750 "${FIXED_SBATCH_PATH}"

RENDER_OUTPUT="${render_output}" python3 - "${FIXED_SBATCH_PATH}" "${source_sbatch_path}" <<'PY'
import json
import os
import sys

fixed = sys.argv[1]
source = sys.argv[2]
payload = json.loads(os.environ["RENDER_OUTPUT"])
data = payload.get("data")
if isinstance(data, dict):
    data["original_sbatch_script"] = source
    data["sbatch_script"] = fixed
print(json.dumps(payload, indent=2, sort_keys=True))
PY
