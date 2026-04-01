# Single MaxCon Local

This workflow sets up one local replay experiment in max-concurrency mode.

`generate_experiment.py` does all of the following in one step:

1. Looks up an already-compiled replay plan for the selected split mode (`full`, `exclude-unranked`, `top`, or `rest`).
2. Validates that the selected plan's `replay_target.model` matches `--target-model`.
3. Creates one generated bundle under `generated/<timestamp>/` with:
   - `replay.toml`
   - `run_replay.sh` (single script entrypoint; accepts port profile override)
   - `plan/<...>.json` (selected source plan copy)
   - `manifest.json`

Default replay output directory:

- `results/replay/single-maxcon/split/<split>/c<max-concurrent>/<timestamp>/`

## Generate One Experiment

```bash
python3 experiments/single-maxcon/local/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --randomize-seed 11 \
  --max-concurrent 8 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --split rest \
  --additional-suffix qwen3_fp8
```

or

```bash
python3 experiments/single-maxcon/local/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --randomize-seed 11 \
  --max-concurrent 2 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 12 \
  --split top \
  --additional-suffix qwen3_fp8
```

Supported split values:

- `full`
- `exclude-unranked` (alias typo `exclued-unranked` is accepted)
- `top`
- `rest`

For `top/rest`, lookup uses split two-group plans with metric
`token_usage` by default (override via `--split-two-group-metric`).

## Notes

- This script does not run `replayer compile`.
- Required plans must already exist under `--source-run-dir`.
- `--additional-suffix` is applied before `.json`, for example:
  - `replay-plan.token.rest.qwen3_fp8.json`
  - `replay-plan.qwen3_fp8.json`
- Replay keeps the compiled launch pattern and only overrides `max_concurrent`.

## Run The Generated Experiment

Use the emitted script from the generated batch directory:

```bash
bash experiments/single-maxcon/local/generated/<timestamp>/run_replay.sh
```

Override the port profile at runtime:

```bash
bash experiments/single-maxcon/local/generated/<timestamp>/run_replay.sh 3
```
