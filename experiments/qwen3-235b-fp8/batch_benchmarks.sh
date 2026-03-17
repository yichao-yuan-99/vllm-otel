#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# One hardcoded line per benchmark (mini-swe-agent then terminus-2).
# python3 "${SCRIPT_DIR}/start.py" --benchmark swebench-verified --agent mini-swe-agent "$@" && python3 "${SCRIPT_DIR}/start.py" --benchmark swebench-verified --agent terminus-2 "$@"
python3 "${SCRIPT_DIR}/start.py" --benchmark terminal-bench@2.0 --agent mini-swe-agent "$@" && python3 "${SCRIPT_DIR}/start.py" --benchmark terminal-bench@2.0 --agent terminus-2 "$@"
python3 "${SCRIPT_DIR}/start.py" --benchmark livecodebench --agent mini-swe-agent "$@" && python3 "${SCRIPT_DIR}/start.py" --benchmark livecodebench --agent terminus-2 "$@"
python3 "${SCRIPT_DIR}/start.py" --benchmark dabstep --agent mini-swe-agent "$@" && python3 "${SCRIPT_DIR}/start.py" --benchmark dabstep --agent terminus-2 "$@"
