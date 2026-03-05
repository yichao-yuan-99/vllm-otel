# Qwen3-Coder-30B Experiments

This folder contains `con-driver` config(s) for running Harbor datasets against a
live vLLM endpoint serving Qwen3-Coder-30B.

## Files

- `configs/config.swebench-verified.toml`

Current config values:

- `driver_backend = "harbor"`
- `pattern = "eager"`
- `pool = "swebench-verified"`
- `max_concurrent = 5`
- `n_task = 500`
- `sample_without_replacement = true`
- `agent = "mini-swe-agent"`
- `results_dir = "experiments/results/qwen3-coder-30b/swebench-verified"`

## Preconditions

1. Start vLLM + gateway for a port profile that is serving Qwen3-Coder-30B.
2. Ensure Harbor and `con-driver` are available in repo `.venv`.
3. Run commands from repo root.

## Run

```bash
bash con-driver/run_con_driver.sh \
  --config experiments/qwen3-coder-30b/configs/config.swebench-verified.toml \
  --port-profile-id 8
```

`port_profile_id` is intentionally passed via CLI (not set in this config), so you
can target whichever running profile has the intended model.

## Common Overrides

Override concurrency:

```bash
bash con-driver/run_con_driver.sh \
  --config experiments/qwen3-coder-30b/configs/config.swebench-verified.toml \
  --port-profile-id 8 \
  --max-concurrent 8
```

Dry run command construction:

```bash
bash con-driver/run_con_driver.sh \
  --config experiments/qwen3-coder-30b/configs/config.swebench-verified.toml \
  --port-profile-id 8 \
  --dry-run
```

Override results location:

```bash
bash con-driver/run_con_driver.sh \
  --config experiments/qwen3-coder-30b/configs/config.swebench-verified.toml \
  --port-profile-id 8 \
  --results-dir experiments/results/record/qwen3-coder-30b-swebench
```

Shard a 500-task dataset into two non-overlapping jobs:

```bash
# Shard A: tasks [0, 250)
bash con-driver/run_con_driver.sh \
  --config experiments/qwen3-coder-30b/configs/config.swebench-verified.toml \
  --port-profile-id 8 \
  --n-task 250 \
  --task-subset-start 0 \
  --task-subset-end 250

# Shard B: tasks [250, 500)
bash con-driver/run_con_driver.sh \
  --config experiments/qwen3-coder-30b/configs/config.swebench-verified.toml \
  --port-profile-id 8 \
  --n-task 250 \
  --task-subset-start 250 \
  --task-subset-end 500
```

## Output

By default, runs are written under:

- `experiments/results/qwen3-coder-30b/swebench-verified/`

Each invocation creates a timestamped run directory with metadata, trial logs, and
gateway artifacts.
