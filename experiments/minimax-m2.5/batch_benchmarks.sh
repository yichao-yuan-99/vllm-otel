#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VALID_AGENTS=("mini-swe-agent" "terminus-2")
DEFAULT_AGENTS=("mini-swe-agent" "terminus-2")
# BENCHMARKS=("terminal-bench@2.0" "livecodebench" "dabstep")
BENCHMARKS=("swebench-verified")

usage() {
  cat <<'EOF'
Usage:
  bash experiments/minimax-m2.5/batch_benchmarks.sh [batch options] [start.py options]

Batch options:
  --agent <name>      Select an agent for all benchmarks (repeatable).
  --agent=<name>      Same as above.
  -h, --help          Show this help message.

Examples:
  # Default: run both mini-swe-agent and terminus-2
  bash experiments/minimax-m2.5/batch_benchmarks.sh --per-profile-conc 5

  # Run only one agent
  bash experiments/minimax-m2.5/batch_benchmarks.sh --agent mini-swe-agent --per-profile-conc 5

  # Run two selected agents (same as default)
  bash experiments/minimax-m2.5/batch_benchmarks.sh --agent mini-swe-agent --agent terminus-2 --per-profile-conc 5
EOF
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

is_valid_agent() {
  local agent="$1"
  local valid
  for valid in "${VALID_AGENTS[@]}"; do
    if [[ "$agent" == "$valid" ]]; then
      return 0
    fi
  done
  return 1
}

SELECTED_AGENTS=()
PASSTHROUGH_ARGS=()

while (($#)); do
  case "$1" in
    --agent)
      if (($# < 2)); then
        echo "missing value for --agent" >&2
        exit 1
      fi
      SELECTED_AGENTS+=("$2")
      shift 2
      ;;
    --agent=*)
      SELECTED_AGENTS+=("${1#*=}")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if ((${#SELECTED_AGENTS[@]} == 0)); then
  SELECTED_AGENTS=("${DEFAULT_AGENTS[@]}")
fi

UNIQUE_AGENTS=()
declare -A SEEN_AGENTS=()
for raw_agent in "${SELECTED_AGENTS[@]}"; do
  agent="$(trim_whitespace "$raw_agent")"
  if [[ -z "$agent" ]]; then
    continue
  fi
  if ! is_valid_agent "$agent"; then
    echo "invalid --agent '${agent}'. Valid agents: ${VALID_AGENTS[*]}" >&2
    exit 1
  fi
  if [[ -n "${SEEN_AGENTS[$agent]:-}" ]]; then
    continue
  fi
  UNIQUE_AGENTS+=("$agent")
  SEEN_AGENTS["$agent"]=1
done
SELECTED_AGENTS=("${UNIQUE_AGENTS[@]}")

if ((${#SELECTED_AGENTS[@]} == 0)); then
  echo "no agents selected" >&2
  exit 1
fi

for benchmark in "${BENCHMARKS[@]}"; do
  for agent in "${SELECTED_AGENTS[@]}"; do
    python3 "${SCRIPT_DIR}/start.py" \
      --benchmark "${benchmark}" \
      --agent "${agent}" \
      "${PASSTHROUGH_ARGS[@]}"
  done
done
