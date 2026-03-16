# Single QPS Local

This workflow sets up one local replay experiment in Poisson mode.

`generate_experiment.py` does all of the following in one step:

1. Compiles the replay plan for the selected split mode (`full`, `exclude-unranked`, `top`, or `rest`).
2. Reuses a compile cache under `generated/cache/` to avoid unnecessary recompilation.
3. Creates one generated bundle under `generated/<timestamp>/` with:
   - `replay.toml`
   - `run_replay.sh` (single script entrypoint; accepts port profile override)
   - `plan/<...>.json` (selected compiled plan copy)
   - `manifest.json`

Default replay output directory:

- `results/replay/single-qps/split/<split>/<qps>/<timestamp>/`

## Generate One Experiment

```bash
python3 experiments/single-qps/local/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps 0.05 \
  --time-constraint-s 1800 \
  --target-model qwen3_coder_30b \
  --port-profile 0 \
  --split full
```

Supported split values:

- `full`
- `exclude-unranked` (alias typo `exclued-unranked` is accepted)
- `top`
- `rest`

For `top/rest`, compile uses split two-group plans with metric
`token_usage` by default (override via `--split-two-group-metric`).

## Run The Generated Experiment

Use the emitted script from the generated batch directory:

```bash
bash experiments/single-qps/local/generated/<timestamp>/run_replay.sh
```

Override the port profile at runtime:

```bash
bash experiments/single-qps/local/generated/<timestamp>/run_replay.sh 3
```


