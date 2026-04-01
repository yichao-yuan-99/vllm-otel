# Single MaxCon Clean (Local)

This workflow sets up one local replay experiment in max-concurrency mode using
clean split plans only.

`generate_experiment.py` does all of the following in one step:

1. Looks up an already-compiled clean replay plan for the selected split mode.
2. Validates that the selected plan's `replay_target.model` matches `--target-model`.
3. Creates one generated bundle under `generated/<timestamp>/` with:
   - `replay.toml`
   - `run_replay.sh` (single script entrypoint; accepts port profile override)
   - `plan/<...>.json` (selected source plan copy)
   - `manifest.json`

This clean variant always looks up clean split plans:

- `replay-plan.clean.<metric>.top(.<suffix>).json`
- `replay-plan.clean.<metric>.rest(.<suffix>).json`
- `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

Default replay output root is changed from:

- `results/replay/single-maxcon/split/<split>/c<max-concurrent>/<timestamp>/`

to:

- `results/replay/single-maxcon-clean/<dataset>/<agent>/split/<split>/c<max-concurrent>/<timestamp>/`
  where `<dataset>/<agent>` is inferred from `--source-run-dir` by dropping
  the first (`<model>`) and last (`<run-dir>`) path segments.

## Generate One Experiment

```bash
python3 experiments/single-maxcon-clean/local/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --randomize-seed 11 \
  --max-concurrent 50 \
  --time-constraint-s 10800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --split rest \
  --additional-suffix qwen3_fp8
```

or

```bash
python3 experiments/single-maxcon-clean/local/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --randomize-seed 11 \
  --max-concurrent 2 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 12 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8
```

`--split` values are `top`, `rest`, or `exclude-unranked`.
`full` is accepted as a compatibility alias for `exclude-unranked`.

For plan lookup, `top/rest/exclude-unranked` all use
`--split-two-group-metric` (default: `token_usage`) because the clean plan
names are always metric-qualified.

## Notes

- This script does not run `replayer compile`.
- Required clean plans must already exist under `--source-run-dir`.
- `--additional-suffix` is applied before `.json`, for example:
  - `replay-plan.clean.token.rest.qwen3_fp8.json`
  - `replay-plan.clean.token.exclude-unranked.qwen3_fp8.json`
- Replay keeps the compiled launch pattern and only overrides `max_concurrent`.

## Run The Generated Experiment

Use the emitted script from the generated batch directory:

```bash
bash experiments/single-maxcon-clean/local/generated/<timestamp>/run_replay.sh
```

Override the port profile at runtime:

```bash
bash experiments/single-maxcon-clean/local/generated/<timestamp>/run_replay.sh 3
```
