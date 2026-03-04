# Single-Agent

This experiment runs four Harbor datasets with `max_concurrent = 1`.

Datasets:

- `terminal-bench@2.0`
- `livecodebench`
- `dabstep`
- `swebench-verified`

The setup follows the same basic `con-driver` pattern as
[experiments/sweep-concurrency/README.md](/scratch/yichaoy2/work/vllm-otel/experiments/sweep-concurrency/README.md),
but there is no replay stage here. These are direct source runs only.

## Settings

Shared settings across all four configs:

- `driver_backend = "harbor"`
- `pattern = "eager"`
- `max_concurrent = 1`
- `sample_without_replacement = true`
- `agent = "terminus-2"`
- default `port_profile_id = 4` in the runner scripts

`n_task` is set to the full Harbor dataset size for each dataset:

- `terminal-bench@2.0`: `89`
- `livecodebench`: `100`
- `dabstep`: `450`
- `swebench-verified`: `500`

Those counts come from `harbor datasets list`.

## Files

- [config.terminal-bench-2.0.toml](/scratch/yichaoy2/work/vllm-otel/experiments/single-agent/configs/config.terminal-bench-2.0.toml)
- [config.livecodebench.toml](/scratch/yichaoy2/work/vllm-otel/experiments/single-agent/configs/config.livecodebench.toml)
- [config.dabstep.toml](/scratch/yichaoy2/work/vllm-otel/experiments/single-agent/configs/config.dabstep.toml)
- [config.swebench-verified.toml](/scratch/yichaoy2/work/vllm-otel/experiments/single-agent/configs/config.swebench-verified.toml)
- [run_dataset.sh](/scratch/yichaoy2/work/vllm-otel/experiments/single-agent/run_dataset.sh)
- [run_all.sh](/scratch/yichaoy2/work/vllm-otel/experiments/single-agent/run_all.sh)

## Preconditions

Before running:

1. Start the target vLLM server and gateway for port profile `4`, or pass a different `--port-profile-id`.
2. Make sure Harbor is installed and the repo `.venv` is usable.
3. Expect outputs under `experiments/results/single-agent/`.

## Run One Dataset

Default port profile:

```bash
bash experiments/single-agent/run_dataset.sh --dataset terminal-bench@2.0
```

Explicit port profile:

```bash
bash experiments/single-agent/run_dataset.sh \
  --dataset swebench-verified \
  --port-profile-id 4
```

List supported datasets:

```bash
bash experiments/single-agent/run_dataset.sh --list-datasets
```

## Run All Datasets

```bash
bash experiments/single-agent/run_all.sh
```

Or:

```bash
bash experiments/single-agent/run_all.sh --port-profile-id 4
```

## Output Layout

Each dataset writes under its own results root:

- `experiments/results/single-agent/terminal-bench-2.0/`
- `experiments/results/single-agent/livecodebench/`
- `experiments/results/single-agent/dabstep/`
- `experiments/results/single-agent/swebench-verified/`

Each `con-driver` invocation then creates a run directory under that root,
for example:

- `experiments/results/single-agent/terminal-bench-2.0/job-<timestamp>/`

Useful files inside each run:

- `meta/run_manifest.json`
- `meta/events.jsonl`
- `meta/results.json`
- `trials/`
- `gateway-output/`
- `vllm-log/`
