# Sweep MaxCon Clean

This workflow is the clean-plan sweep counterpart to `single-maxcon-clean`.

Compared to `single-maxcon-clean`:

- input accepts `--max-concurrent-list` instead of one `--max-concurrent`
- generated runner executes all max-concurrency points sequentially
- plan lookup always uses clean split plans:
  - `replay-plan.clean.<metric>.top(.<suffix>).json`
  - `replay-plan.clean.<metric>.rest(.<suffix>).json`
  - `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

Default replay output layout:

- `results/replay/sweep-maxcon-clean/<dataset>/<agent>/split/<split>/c<max-concurrent>/<timestamp>/`
  where `<dataset>/<agent>` is inferred from `--source-run-dir` by dropping
  the first (`<model>`) and last (`<run-dir>`) path segments.

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-maxcon-clean/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --randomize-seed 11 \
  --max-concurrent-list 2,4,8,16 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 12 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8
```

`--concurrency-list` is accepted as an alias for `--max-concurrent-list`.

`--split` values are `top`, `rest`, or `exclude-unranked`.
`full` is accepted as a compatibility alias for `exclude-unranked`.

For plan lookup, `top/rest/exclude-unranked` all use
`--split-two-group-metric` (default: `token_usage`) because the clean plan
names are always metric-qualified.

## Run Generated Sweep

```bash
bash experiments/sweep-maxcon-clean/generated/<timestamp>/run_replay.sh
```

Override the port profile at runtime:

```bash
bash experiments/sweep-maxcon-clean/generated/<timestamp>/run_replay.sh 3
```

## Generated Files

For each concurrency slug `cN`:

- `generated/<timestamp>/cN/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<...>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Notes

- This script does not run `replayer compile`.
- Required clean plans must already exist under `--source-run-dir`.
- Replay keeps the compiled launch pattern and only overrides `max_concurrent`.
