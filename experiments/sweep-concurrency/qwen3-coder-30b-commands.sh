#!/usr/bin/env bash
set -euo pipefail

SOURCE_RUN_DIRS=(
  # "results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z"
  # "results/qwen3-coder-30b/dabstep/terminus-2/dabstep-20260306T215045Z"
  "results/qwen3-coder-30b/livecodebench/mini-swe-agent/livecodebench-20260306T185912Z"
  # "results/qwen3-coder-30b/livecodebench/terminus-2/livecodebench-20260306T192339Z"
  # "results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z"
  # "results/qwen3-coder-30b/swebench-verified/terminus-2/swebench-verified-20260306T082357Z"
  # "results/qwen3-coder-30b/terminal-bench-2.0/mini-swe-agent/terminal-bench@2.0-20260306T163324Z"
  "results/qwen3-coder-30b/terminal-bench-2.0/terminus-2/terminal-bench@2.0-20260306T174037Z"
)

CONCURRENCY_LIST="20,30,40,50,70,100"
CONCURRENCY_VALUES=(20 30 40 50 70 100)
NUM_TASKS=350
PORT_PROFILE_ID_LIST="0,1,2,3,4"
COMPILE_PORT_PROFILE_ID="${PORT_PROFILE_ID_LIST%%,*}"
SWEEP_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMBINED_CONFIG_DIR="experiments/sweep-concurrency/generated/qwen3-coder-30b/big-batch-${SWEEP_TIMESTAMP}"
REPLAY_BATCH_ROOT="results/replay/${SWEEP_TIMESTAMP}"
ORCHESTRATOR_OUTPUT_DIR="${REPLAY_BATCH_ROOT}/orchestrator-replay-${SWEEP_TIMESTAMP}"

if [[ -z "$COMPILE_PORT_PROFILE_ID" ]]; then
  echo "PORT_PROFILE_ID_LIST must contain at least one profile id" >&2
  exit 1
fi

mkdir -p "$COMBINED_CONFIG_DIR"

sanitize_slug() {
  local source_run_dir="$1"
  local slug="${source_run_dir#results/}"
  slug="${slug//\//__}"
  slug="${slug//@/_at_}"
  slug="${slug//./_}"
  printf '%s\n' "$slug"
}

compile_pids=()
for source_run_dir in "${SOURCE_RUN_DIRS[@]}"; do
  python3 -m replayer compile \
    --job-dir "$source_run_dir" \
    --port-profile-id "$COMPILE_PORT_PROFILE_ID" &
  compile_pids+=("$!")
done
for pid in "${compile_pids[@]}"; do
  wait "$pid"
done

for source_run_dir in "${SOURCE_RUN_DIRS[@]}"; do
  python3 experiments/sweep-concurrency/generate_replay_configs.py \
    --source-run-dir "$source_run_dir" \
    --concurrency-list "$CONCURRENCY_LIST" \
    --num-tasks "$NUM_TASKS" \
    --output-config-dir "$COMBINED_CONFIG_DIR" \
    --replay-root-dir "$REPLAY_BATCH_ROOT"

  slug="$(sanitize_slug "$source_run_dir")"
  for concurrency in "${CONCURRENCY_VALUES[@]}"; do
    mv \
      "$COMBINED_CONFIG_DIR/replay.c${concurrency}.toml" \
      "$COMBINED_CONFIG_DIR/${slug}.replay.c${concurrency}.toml"
  done
  mv \
    "$COMBINED_CONFIG_DIR/manifest.json" \
    "$COMBINED_CONFIG_DIR/${slug}.manifest.json"
done

python -m orchestrator \
  --job-type replay \
  --jobs-dir "$COMBINED_CONFIG_DIR" \
  --output-dir "$ORCHESTRATOR_OUTPUT_DIR" \
  --port-profile-id-list "$PORT_PROFILE_ID_LIST"
