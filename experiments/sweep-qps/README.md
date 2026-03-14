# Sweep QPS

Goal: generate replay jobs that sweep Poisson launch rate (requests per second)
for a single source run.

This workflow has two steps:

1. Generate replay config files (`*.toml`) for each target QPS.
2. Run those configs with `orchestrator` over available profiles.

## Generator Script

Use:

```bash
python3 experiments/sweep-qps/generate_replay_configs.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.05,0.1,0.2,0.4 \
  --time-constraint-s 1800 \
  --output-config-dir experiments/sweep-qps/generated/dabstep-mswe
```

Required inputs:

- `--source-run-dir`: source run directory (must be under top-level `results/`)
- `--poisson-seed`: seed for Poisson inter-arrival sampling
- `--randomize-seed`: seed for launch-order randomization
- `--qps-list`: comma-separated Poisson rates, requests per second
- `--time-constraint-s`: wall-clock replay limit in seconds
- `--output-config-dir`: directory where generated replay TOML files are written

Optional forwarding:

- `--plan-path` (default: `<source-run-dir>/replay-plan.json`)
- `--replay-root-dir` (default under `results/replay/<utc-timestamp>/.../sweep-qps`)
- `--port-profile-id`
- `--vllm-log-interval-s`
- `--vllm-log-timeout-s`
- `--launch-policy-override-json`
- `--extra-replay-json` (generic JSON object merged into `[replay]`)

Generated replay configs intentionally do not set `num_tasks`; they are
time-bounded by `time_constraint_s`.

For each target `qps`, the generator creates:

- config file: `<output-config-dir>/<utc-timestamp>/replay.qps<token>.toml`
- replay output dir in config:
  `results/replay/<utc-timestamp>/<source-lineage-without-run-dir>/sweep-qps/qps<token>`

Each generator invocation uses one UTC timestamp batch directory shared across
all generated QPS configs in that run.

The generator also writes:

- `<output-config-dir>/<utc-timestamp>/manifest.json`

## Run With Orchestrator

```bash
python -m orchestrator \
  --job-type replay \
  --jobs-dir experiments/sweep-qps/generated/dabstep-mswe/<utc-timestamp> \
  --output-dir results/replay/<utc-timestamp> \
  --port-profile-id-list 0,1,2,3,4
```

This writes orchestrator logs and `summary.json` under:
`results/replay/<utc-timestamp>/orchestrator-replay-<run-utc-timestamp>/`.
