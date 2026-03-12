# Sweep Concurrency

Goal: generate replay jobs that sweep `max_concurrent` while replaying the same
source run plan.

This workflow has two steps:

1. Generate replay config files (`*.toml`) for each target concurrency.
2. Run those configs with `orchestrator` over a list of available profiles.

## Generator Script

Use:

```bash
python3 experiments/sweep-concurrency/generate_replay_configs.py \
  --source-run-dir results/qwen3-coder-30b/livecodebench/mini-swe-agent/livecodebench-20260306T185912Z \
  --concurrency-list 15,30,60,90,120 \
  --num-tasks 500 \
  --output-config-dir experiments/sweep-concurrency/generated/livecodebench-mswe
```

Required inputs:

- `--source-run-dir`: source run directory (must be under top-level `results/`)
- `--concurrency-list`: comma-separated target concurrencies
- `--num-tasks`: replay task count for all generated jobs
- `--output-config-dir`: where generated replay config files are written

Optional forwarding:

- `--plan-path` (default: `<source-run-dir>/replay-plan.json`)
- `--port-profile-id`
- `--vllm-log-interval-s`
- `--vllm-log-timeout-s`
- `--launch-policy-override-json`
- `--extra-replay-json` (generic JSON object merged into `[replay]`)

Replay always enables vLLM logging in the current code base. The generator only
forwards optional logging interval/timeout fields.

For each target concurrency `c`, the generator creates:

- config file: `<output-config-dir>/replay.c<c>.toml`
- replay output dir in config:
  `results/replay/<source-lineage-without-run-dir>/sweep-concurrency/<c>`

Example output mapping:

- source run:
  `results/qwen3-coder-30b/livecodebench/mini-swe-agent/livecodebench-20260306T185912Z`
- replay output root:
  `results/replay/qwen3-coder-30b/livecodebench/mini-swe-agent/sweep-concurrency/`

The generator also writes:

- `<output-config-dir>/manifest.json`

## Run With Orchestrator

```bash
python -m orchestrator \
  --job-type replay \
  --jobs-dir experiments/sweep-concurrency/generated/livecodebench-mswe \
  --port-profile-id-list 0,1,2,3,4
```

## Preset Batch Script (qwen3-coder-30b)

For the current qwen3-coder-30b sweep set, use:

```bash
bash experiments/sweep-concurrency/qwen3-coder-30b-commands.sh
```

This script runs three phases end-to-end:

1. Compiles all listed source runs in parallel (`python3 -m replayer compile`),
   using `COMPILE_PORT_PROFILE_ID` for compile-time tokenization.
2. Generates replay configs into one combined directory:
   `experiments/sweep-concurrency/generated/qwen3-coder-30b/big-batch-<utc-timestamp>`.
3. Runs one orchestrator invocation against that combined directory, with logs
   and summary under:
   `results/replay/<utc-timestamp>/orchestrator-replay-<utc-timestamp>/`.

Replay now always goes through gateway and writes `gateway-output/` artifacts
(matching con-driver style outputs).

Because `generate_replay_configs.py` emits fixed names (`replay.c*.toml`), the
script renames each generated config to include a source-run slug, for example:

- `<source-slug>.replay.c1.toml`
- `<source-slug>.replay.c2.toml`
- `<source-slug>.replay.c5.toml`
- `<source-slug>.replay.c10.toml`

Each source-run manifest is also preserved with source-specific naming:
`<source-slug>.manifest.json`.

To adjust workload sizing or profile routing, edit these variables in
`experiments/sweep-concurrency/qwen3-coder-30b-commands.sh`:

- `CONCURRENCY_LIST`
- `CONCURRENCY_VALUES`
- `NUM_TASKS`
- `PORT_PROFILE_ID_LIST`
- `COMPILE_PORT_PROFILE_ID` (defaults to first profile in `PORT_PROFILE_ID_LIST`)

## Batch Post-Process (qwen3-coder-30b)

After replay jobs finish, post-process all discovered sweep run directories in
parallel:

```bash
bash experiments/sweep-concurrency/qwen3-coder-30b-post-process.sh
```

Prerequisite:

- active `python3` environment includes `prometheus_client`

By default this scans:

- `results/replay/qwen3-coder-30b/**/sweep-concurrency/*/replay/summary.json`

and runs per run directory:

- `post-process/vllm-metrics/extract_run.py`
- `post-process/vllm-metrics/summarize_timeseries.py`
- `post-process/gateway/llm-requests/extract_run.py` (only if `gateway-output/` exists)

Options:

- `--root-dir <path>`: override replay root directory to scan
- `--max-procs <n>`: worker process count (default: `MAX_PROCS` env var, else `nproc`, else `4`)
- `--dry-run`: list discovered run directories without running post-processing
